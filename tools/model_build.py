#!/usr/bin/env python3
"""
tools/model_build.py — 单脚本自动构建 DeepStream 模型产物。

链路: best.pt →(ultralytics export nms)→ best.onnx →(trtexec fp16 动态batch)→ best_dyn_fp16.engine
      + labels.txt（类别裁剪）+ configs/pgie_config_<name>.txt（模板渲染）
      + models/common/libnvds_yolo_nms.so（全模型共享 parser, 首次自动编译）

产物布局（models/ 为 Git 忽略、运行时挂载目录）:
  models/<name>/best.onnx              # 中间产物, 供重建引擎
  models/<name>/best_dyn_fp16.engine   # 运行期引擎（环境绑定: GPU/TRT 必须与部署机一致）
  models/<name>/labels.txt             # 类别裁剪后的标签, 每行一类
  models/common/libnvds_yolo_nms.so    # 共享解析库, 只编译一次
  configs/pgie_config_<name>.txt       # 生成的 nvinfer 配置（Git 跟踪）

类别裁剪策略（基础 COCO 模型 >10 类 → 默认只出 person(class 0)；专用模型 <=10 类 → 全量）:
  由 num-detected-classes 落进 INI, 共享 parser 以 class_id < num-detected-classes 运行时裁剪。

注意:
  - 引擎与部署环境强绑定（TensorRT 版本 / GPU 架构 / 驱动）, 必须在目标机或
    同 TRT+GPU 的机器上构建, 不可跨机搬运。
  - 增量重跑幂等: 覆盖写入同名产物, 已有 engine 会重新构建。

用法:
  python tools/model_build.py --name person --pt models/person/yolo26n.pt            # uid 默认 1
  python tools/model_build.py --name helmet --pt models/helmet/best.pt --uid 3
  python tools/model_build.py --name helmet --pt models/helmet/best.pt --skip-engine # 只刷 labels/INI
   python tools/model_build.py --name vest_cls --pt <classify best.pt> --uid 6 --max-batch 32  # classify

classify 分支（二级分类器, 2 类 yes/no）:
  - ONNX 不烧入 NMS（末尾已带 Softmax, 输出即概率）, labels.txt 全量不裁剪
  - 动态 batch 默认 32（二级推理 batch = 每帧对象数）
  - 生成 configs/sgie_config_<name>.txt（二级分类器配置, process-mode=2 作用于 pgie 的 person 结果）
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEMPLATES = ROOT / "tools" / "templates"
MODELS = ROOT / "models"
CONFIGS = ROOT / "configs"

ONNX = "best.onnx"
ENGINE = "best_dyn_fp16.engine"
LABELS = "labels.txt"
PARSER_SO = "libnvds_yolo_nms.so"
PARSE_FUNC = "NvDsInferParseCustomYoloNMS"
INPUT_NAME = "images"  # ultralytics NMS 导出固定输入名

# 类别数 > 此值视为基础 COCO 轮廓模型（person/80 类），默认只输出 class 0 (person)
BASE_CLASS_THRESHOLD = 10
OP_SET = 12  # ONNX opset（NMS TopK/GatherElements 为标准算子，opset>=11 即可）
WORKSPACE_MB = 2048


def run(cmd: list[str], desc: str):
    print(f"\n### {desc}\n    $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def compile_shared_parser() -> Path:
    """共享 parser 若不存在则用 g++ 编译一次 → models/common/libnvds_yolo_nms.so。"""
    dst = MODELS / "common" / PARSER_SO
    if dst.exists():
        print(f"### 共享 parser 已存在: {dst.relative_to(ROOT)}（跳过编译）")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)

    ds = Path(os.environ.get("DEEPSTREAM_DIR", "/opt/nvidia/deepstream/deepstream"))
    if not (ds / "sources" / "includes").is_dir():
        sys.exit(f"未找到 DeepStream SDK 头文件: {ds / 'sources' / 'includes'}\n"
                 f"请设置 DEEPSTREAM_DIR 指向 DeepStream 安装目录")

    cuda = Path(os.environ.get("CUDA_DIR", "/usr/local/cuda"))
    if not cuda.is_dir():
        cands = sorted(Path("/usr/local").glob("cuda-*"))
        if not cands:
            sys.exit("未找到 CUDA 安装目录, 请设置 CUDA_DIR")
        cuda = cands[-1]

    run([
        "g++", "-o", str(dst),
        str(TEMPLATES / "libnvds_yolo_nms.cpp"),
        "-Wall", "-std=c++11", "-shared", "-fPIC",
        f"-I{ds / 'sources' / 'includes'}",
        f"-I{cuda / 'include'}",
        "-lnvinfer",
    ], "编译共享 parser")
    return dst


def model_imgsz(model) -> int | list[int]:
    """取模型的训练输入尺寸（int 或 [h,w]），缺省 640。"""
    imgsz = model.overrides.get("imgsz") or model.args.get("imgsz") or 640
    if isinstance(imgsz, (tuple, list)):
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def main():
    ap = argparse.ArgumentParser(description="DeepStream 模型一键构建 (pt→onnx→engine→labels→INI)")
    ap.add_argument("--name", required=True, help="模型逻辑名, 产物写入 models/<name>/ 与 configs/pgie_config_<name>.txt")
    ap.add_argument("--pt", required=True, help="ultralytics best.pt 路径")
    ap.add_argument("--max-batch", type=int, default=None,
                    help="引擎动态 batch 上限（>= 最大路数）, 默认: detector 12 / classify 32")
    ap.add_argument("--uid", type=int, default=1, help="gie-unique-id, 默认 1; 多检测器管线需各配不同 uid")
    ap.add_argument("--skip-engine", action="store_true",
                    help="跳过 ONNX 导出与 trtexec, 只刷新 labels.txt + INI（引擎已存在时用）")
    args = ap.parse_args()

    pt = Path(args.pt)
    if not pt.exists():
        sys.exit(f"--pt 不存在: {pt}")

    out_dir = MODELS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    compile_shared_parser()

    from ultralytics import YOLO
    model = YOLO(str(pt))
    names = list(model.names.values())
    n_classes = len(names)
    h, w = model_imgsz(model)
    is_classify = getattr(model, "task", None) == "classify"
    max_batch = args.max_batch or (32 if is_classify else 12)
    # 类别裁剪按任务分派: classify 全量; detector >10 类基础模型只留 class 0 (person)
    num_classes = n_classes if is_classify else (1 if n_classes > BASE_CLASS_THRESHOLD else n_classes)
    task_tag = " (task=classify)" if is_classify else ""
    print(f"\n### 模型 {args.name}{task_tag}: {n_classes} 类 -> 输出 {num_classes} 类 "
          f"(imgsz={h}x{w}, max-batch={max_batch}, uid={args.uid})")

    if not args.skip_engine:
        # --- 1) pt -> onnx (动态 batch), 按任务分派 ---
        if is_classify:
            # classify: 不烧入 NMS/opset, 仅 dynamic; ONNX 末尾已带 Softmax（概率）
            exported = Path(model.export(format="onnx", dynamic=True, imgsz=(h, w)))
        else:
            exported = Path(model.export(format="onnx", nms=True, dynamic=True,
                                         opset=OP_SET, imgsz=(h, w)))
        shutil.copyfile(exported, out_dir / ONNX)
        # ultralytics 把 onnx 写到 .pt 同目录（源文件 stem）。若与规范名不同，
        # 清掉旁路产物，保持 models/<name>/ 只留 best.onnx。
        if exported.resolve() != (out_dir / ONNX).resolve():
            exported.unlink(missing_ok=True)
        print(f"\n### ONNX 已生成: {out_dir / ONNX}")

        # --- 2) onnx -> engine (trtexec fp16, 动态 batch 1~max_batch) ---
        run([
            "/usr/bin/trtexec",
            f"--onnx={out_dir / ONNX}",
            "--fp16",
            f"--minShapes={INPUT_NAME}:1x3x{h}x{w}",
            f"--optShapes={INPUT_NAME}:{max_batch}x3x{h}x{w}",
            f"--maxShapes={INPUT_NAME}:{max_batch}x3x{h}x{w}",
            f"--memPoolSize=workspace:{WORKSPACE_MB}M",
            f"--saveEngine={out_dir / ENGINE}",
        ], f"构建 TensorRT 引擎 (动态 batch 1~{max_batch})")

    # --- 3) labels.txt: classify 全量, detector 只写输出类别 ---
    labels = names if is_classify else names[:num_classes]
    (out_dir / LABELS).write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"\n### labels.txt 已写入 {out_dir / LABELS}: {labels}")

    # --- 4) 渲染 INI 配置（按任务分派: classify=二级分类器 sgie / detector=pgie） ---
    if is_classify:
        tpl_path = TEMPLATES / "sgie_config.ini.tpl"
        ini_path = CONFIGS / f"sgie_config_{args.name}.txt"
    else:
        tpl_path = TEMPLATES / "pgie_config.ini.tpl"
        ini_path = CONFIGS / f"pgie_config_{args.name}.txt"
    tpl = tpl_path.read_text(encoding="utf-8")
    ini = (tpl.replace("{{name}}", args.name)
              .replace("{{num_classes}}", str(num_classes))
              .replace("{{uid}}", str(args.uid)))
    ini_path.write_text(ini, encoding="utf-8")

    print(f"\n### 完成: {args.name}{task_tag}")
    for rel in (out_dir / ONNX, out_dir / ENGINE, out_dir / LABELS, ini_path):
        if rel.exists():
            size_mb = rel.stat().st_size / 1e6
            print(f"    {rel.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    print(f"    共享 parser: {MODELS / 'common' / PARSER_SO}")
    print("\n注意: engine 与构建机 GPU/TRT 绑定, 换机或换 TRT 后需重跑本脚本。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
