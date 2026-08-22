#!/usr/bin/env python3
"""
tools/model_build.py — 从 config（单一事实来源）或 CLI 自动构建 DeepStream 模型产物。

两种用法:

1) 单模型（旧用法）:
   python tools/model_build.py --name vest_cls --pt <classify best.pt> --uid 6 --violation no_vest

2) 读 config（推荐，config 即事实来源）:
   python tools/model_build.py --config server/config.yaml
   → 遍历 model.gies，逐个: .pt → onnx → (classifier 且 violation 不在 index0 时交换
     ONNX 输出通道，使违规类落到 class0) → engine → generated/<name>/labels.txt + INI

产物布局:
  models/<name>/best.onnx            # 中间产物（供重建引擎）
  models/<name>/best_dyn_fp16.engine # 运行期引擎（环境绑定）
  generated/<name>/labels.txt        # 类别标签（violation 前置）
  generated/<name>/(pgie|sgie)_config.txt  # 生成的 nvinfer 配置（运行期读取）
  models/common/libnvds_yolo_nms.so  # 共享 parser（detector 用，编译一次）

class0 交换（关键，§8 的「二级分类器只挂 class0」quirk 的自动化）:
  classifier 模型若 violation 不在 index0，交换末端 linear 输出通道使 violation=class0，
  并同步重排 labels.txt。detector 全类可挂、无需交换。
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.model_spec import KIND_CLASSIFIER, KIND_DETECTOR, anchor_uid, parse_gies  # noqa: E402

TEMPLATES = ROOT / "tools" / "templates"
MODELS = ROOT / "models"
GENERATED = ROOT / "generated"

ONNX = "best.onnx"
ENGINE = "best_dyn_fp16.engine"
LABELS = "labels.txt"
PARSER_SO = "libnvds_yolo_nms.so"
PARSE_FUNC = "NvDsInferParseCustomYoloNMS"
INPUT_NAME = "images"  # ultralytics NMS 导出固定输入名

BASE_CLASS_THRESHOLD = 10
OP_SET = 12
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


def model_imgsz(model) -> tuple[int, int]:
    """取模型的训练输入尺寸（int 或 [h,w]），缺省 640。"""
    imgsz = model.overrides.get("imgsz") or model.args.get("imgsz") or 640
    if isinstance(imgsz, (tuple, list)):
        return int(imgsz[0]), int(imgsz[1])
    return int(imgsz), int(imgsz)


def _find_last_linear(module):
    """返回 module 里最后一个 nn.Linear（classify 头）。"""
    import torch.nn as nn
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    return last


def _swap_violation_to_class0(model, names: list[str], violation: str, name: str) -> list[str]:
    """把 classifier 的末端 linear 输出通道重排，使 violation 落到 class0，返回新类别顺序。"""
    import torch

    if violation not in names:
        raise ValueError(f"{name}: violation={violation!r} 不在模型类别 {names} 中")
    v = names.index(violation)
    if v == 0:
        print(f"### {name}: violation={violation!r} 已在 class0，无需交换")
        return names

    linear = _find_last_linear(model.model)
    if linear is None:
        raise RuntimeError(f"{name}: 未找到末端 nn.Linear 层，无法做 class0 交换")
    n = linear.weight.shape[0]
    perm = [v] + [i for i in range(n) if i != v]
    with torch.no_grad():
        linear.weight.copy_(linear.weight[perm])
        linear.bias.copy_(linear.bias[perm])
    new_names = [names[i] for i in perm]
    print(f"### {name}: violation={violation!r} 原 index={v} → 已换到 class0（新顺序 {new_names}）")
    return new_names


def build_one(name: str, source: str, uid: int, kind: str | None = None,
              violation: str | None = None, max_batch: int | None = None,
              operate_on_uid: int = 1, skip_engine: bool = False) -> None:
    """构建单个模型：pt→onnx→(class0 交换)→engine→generated/<name>/。"""
    from ultralytics import YOLO

    src = Path(source)
    if not src.exists():
        sys.exit(f"source 不存在: {src}")

    model = YOLO(str(src))
    names = list(model.names.values())
    n_classes = len(names)
    h, w = model_imgsz(model)
    is_classify = (kind == KIND_CLASSIFIER) if kind else (model.task == "classify")
    max_batch = max_batch or (32 if is_classify else 12)

    if violation is not None and is_classify:
        names = _swap_violation_to_class0(model, names, violation, name)
    elif violation is not None:
        if violation not in names:
            raise ValueError(f"{name}: violation={violation!r} 不在模型类别 {names} 中")
        # detector 全类可挂，无需交换，仅校验
    elif is_classify and violation is None:
        raise ValueError(f"{name}: classifier 必须声明 violation")

    # detector 类别裁剪：基础 COCO 模型(>10类)只出 class0(person)；专用模型全量
    num_classes = n_classes if is_classify else (1 if n_classes > BASE_CLASS_THRESHOLD else n_classes)
    task_tag = " (classifier)" if is_classify else ""
    print(f"\n### 模型 {name}{task_tag}: {n_classes} 类 -> labels {names} "
          f"(imgsz={h}x{w}, max-batch={max_batch}, uid={uid})")

    model_dir = MODELS / name
    model_dir.mkdir(parents=True, exist_ok=True)

    if not is_classify:
        compile_shared_parser()

    if not skip_engine:
        # --- 1) pt -> onnx (动态 batch) ---
        if is_classify:
            exported = Path(model.export(format="onnx", dynamic=True, imgsz=(h, w)))
        else:
            exported = Path(model.export(format="onnx", nms=True, dynamic=True,
                                         opset=OP_SET, imgsz=(h, w)))
        dst = model_dir / ONNX
        # ultralytics 把 onnx 写到 .pt 同目录（按源文件名 stem）；若与规范名相同则原地保留，
        # 否则拷贝到 models/<name>/best.onnx 并清理旁路产物。
        if exported.resolve() != dst.resolve():
            shutil.copyfile(exported, dst)
            exported.unlink(missing_ok=True)
        print(f"\n### ONNX 已生成: {dst}")

        # --- 2) onnx -> engine (trtexec fp16, 动态 batch) ---
        run([
            "/usr/bin/trtexec",
            f"--onnx={model_dir / ONNX}",
            "--fp16",
            f"--minShapes={INPUT_NAME}:1x3x{h}x{w}",
            f"--optShapes={INPUT_NAME}:{max_batch}x3x{h}x{w}",
            f"--maxShapes={INPUT_NAME}:{max_batch}x3x{h}x{w}",
            f"--memPoolSize=workspace:{WORKSPACE_MB}M",
            f"--saveEngine={model_dir / ENGINE}",
        ], f"构建 TensorRT 引擎 (动态 batch 1~{max_batch})")

    # --- 3) 产物落 generated/<name>/: labels.txt + INI ---
    # detector 按 num-detected-classes 裁剪标签；classifier 全量（violation 已前置）
    label_names = names if is_classify else names[:num_classes]
    out_dir = GENERATED / name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / LABELS).write_text("\n".join(label_names) + "\n", encoding="utf-8")
    print(f"\n### labels.txt -> {out_dir / LABELS}: {label_names}")

    tpl_name = "sgie_config.ini.tpl" if is_classify else "pgie_config.ini.tpl"
    tpl = (TEMPLATES / tpl_name).read_text(encoding="utf-8")
    ini = (tpl.replace("{{name}}", name)
              .replace("{{num_classes}}", str(num_classes))
              .replace("{{uid}}", str(uid))
              .replace("{{operate_on_uid}}", str(operate_on_uid)))
    ini_path = out_dir / ("sgie_config.txt" if is_classify else "pgie_config.txt")
    ini_path.write_text(ini, encoding="utf-8")

    print(f"\n### 完成: {name}{task_tag}")
    for rel in (model_dir / ONNX, model_dir / ENGINE, out_dir / LABELS, ini_path):
        if rel.exists():
            size_mb = rel.stat().st_size / 1e6
            print(f"    {rel.relative_to(ROOT)}  ({size_mb:.1f} MB)")


def build_from_config(config_path: str, skip_engine: bool = False) -> None:
    """读 config.yaml 的 model.gies，逐个构建。"""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    gies = parse_gies((cfg.get("model") or {}).get("gies"))
    anchor = anchor_uid(gies)
    for name, spec in gies.items():
        build_one(
            name=name,
            source=spec.source,
            uid=spec.uid,
            kind=spec.kind,
            violation=spec.violation,
            operate_on_uid=anchor,
            skip_engine=skip_engine,
        )


def main():
    ap = argparse.ArgumentParser(description="DeepStream 模型一键构建 (config 或 CLI)")
    ap.add_argument("--config", help="读 config.yaml 的 model.gies 构建全部模型")
    ap.add_argument("--name", help="[单模型] 逻辑名，产物写入 models/<name>/ 与 generated/<name>/")
    ap.add_argument("--pt", help="[单模型] ultralytics best.pt 路径")
    ap.add_argument("--max-batch", type=int, default=None,
                    help="引擎动态 batch 上限（默认: detector 12 / classify 32）")
    ap.add_argument("--uid", type=int, default=1, help="gie-unique-id（全局唯一）")
    ap.add_argument("--kind", choices=[KIND_DETECTOR, KIND_CLASSIFIER], default=None,
                    help="detector | classifier（缺省按 model.task 自动判定）")
    ap.add_argument("--violation", default=None, help="报警类标签（classifier 必填）")
    ap.add_argument("--skip-engine", action="store_true",
                    help="跳过 ONNX 导出与 trtexec，只刷新 labels.txt + INI")
    args = ap.parse_args()

    if args.config:
        build_from_config(args.config, skip_engine=args.skip_engine)
        return 0

    if not args.name or not args.pt:
        ap.error("需要 --config，或同时提供 --name 与 --pt")

    build_one(
        name=args.name,
        source=args.pt,
        uid=args.uid,
        kind=args.kind,
        violation=args.violation,
        max_batch=args.max_batch,
        skip_engine=args.skip_engine,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
