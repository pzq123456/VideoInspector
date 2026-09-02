#!/usr/bin/env python3
"""
tools/model_build.py — 从 config（单一事实来源）自动构建 DeepStream 模型产物。

用法:
    python tools/model_build.py --config deploy/config.yaml [--force]
    → 部署根 = config.yaml 所在目录（自包含部署单元），遍历 model.gies 逐个:
      .pt → onnx → (classifier 且 violation 不在 index0 时交换输出通道，
      使违规类落到 class0) → engine → generated/<name>/<版本键>/labels.txt + INI

产物布局（全部收敛到 <部署根>/generated/，可整目录删除重建；一版一目录）:
  generated/<name>/<版本键>/best.onnx              # 中间产物（供重建引擎）
  generated/<name>/<版本键>/best_dyn_fp16.engine   # 运行期引擎（环境绑定）
  generated/<name>/<版本键>/labels.txt             # 类别标签（violation 前置）
  generated/<name>/<版本键>/(pgie|sgie)_config.txt # 生成的 nvinfer 配置（运行期读取）
  generated/common/libnvds_yolo_nms.so             # 共享 parser（detector 用，编译一次）
  版本键 = source 相对 models/ 的父目录路径（server/model_spec.artifact_version
  统一推导）。换 config.source 即写新目录、旧版本原样留存 → 回滚零重建；
  约定版本目录不可变，换模型 = 新建版本文件夹（原地替换 .pt 属违规操作）。

增量构建（容器启动场景友好）:
  - best.onnx 比 .pt 新           → 跳过 ultralytics 导出
  - best_dyn_fp16.engine 比 onnx 新 → 跳过 trtexec
  - engine 另校验环境指纹 engine.meta.json（GPU 型号/算力 + TRT 版本），
    整目录拷贝到异构 GPU 服务器时自动判定过期并重建，避免加载失败
  - labels.txt / INI 始终重生成（廉价，保证与 config 一致）
  - --force 忽略以上检查全量重建

class0 交换（「二级分类器只挂 class0」quirk 的自动化）:
  classifier 模型若 violation 不在 index0，交换末端 linear 输出通道使 violation=class0，
  并同步重排 labels.txt。detector 全类可挂、无需交换。
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.model_spec import (  # noqa: E402
    KIND_CLASSIFIER,
    KIND_DETECTOR,
    anchor_uid,
    artifact_version,
    parse_gies,
)

TEMPLATES = ROOT / "tools" / "templates"
GENERATED_DIRNAME = "generated"

ONNX = "best.onnx"
ENGINE = "best_dyn_fp16.engine"
ENGINE_META = "engine.meta.json"
LABELS = "labels.txt"
PARSER_SO = "libnvds_yolo_nms.so"
INPUT_NAME = "images"  # ultralytics NMS 导出固定输入名

BASE_CLASS_THRESHOLD = 10
OP_SET = 12
WORKSPACE_MB = 2048


def run(cmd: list[str], desc: str):
    print(f"\n### {desc}\n    $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def compile_shared_parser(gen_root: Path) -> Path:
    """共享 parser 若不存在则用 g++ 编译一次 → <gen_root>/common/libnvds_yolo_nms.so。"""
    dst = gen_root / "common" / PARSER_SO
    if dst.exists():
        print(f"### 共享 parser 已存在: {dst}（跳过编译）")
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


def _perm_for_violation(names: list[str], violation: str, name: str) -> list[int] | None:
    """返回把 violation 挪到 class0 的索引重排；已在 index0 返回 None。fail fast。"""
    if violation not in names:
        raise ValueError(f"{name}: violation={violation!r} 不在模型类别 {names} 中")
    v = names.index(violation)
    if v == 0:
        print(f"### {name}: violation={violation!r} 已在 class0，无需交换")
        return None
    return [v] + [i for i in range(len(names)) if i != v]


def _apply_class0_swap(model, perm: list[int], name: str, new_names: list[str]) -> None:
    """导出前调用：把末端 nn.Linear 输出通道按 perm 重排（仅内存内修改）。"""
    linear = _find_last_linear(model.model)
    if linear is None:
        raise RuntimeError(f"{name}: 未找到末端 nn.Linear 层，无法做 class0 交换")
    import torch
    with torch.no_grad():
        linear.weight.copy_(linear.weight[perm])
        linear.bias.copy_(linear.bias[perm])
    print(f"### {name}: violation 已换到 class0（新顺序 {new_names}）")


def _env_fingerprint() -> dict[str, str | None]:
    """当前构建环境指纹：引擎与 GPU 型号/算力 + TensorRT 版本绑定，跨机不可复用。"""
    fp: dict[str, str | None] = {}
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(0)
        fp["gpu"] = f"{torch.cuda.get_device_name(0)}|cc{major}.{minor}"
    except Exception:
        fp["gpu"] = None
    try:
        import tensorrt
        fp["trt"] = tensorrt.__version__
    except Exception:
        fp["trt"] = None
    return fp


LEGACY_FILES = (ONNX, ENGINE, ENGINE_META, LABELS, "pgie_config.txt", "sgie_config.txt")


def _migrate_legacy_layout(legacy_dir: Path, out_dir: Path, name: str,
                           expected_labels: list[str]) -> None:
    """一次性迁移: 旧扁平布局 generated/<name>/{...} → generated/<name>/<版本键>/。

    rename（同文件系统）保留 mtime 且 engine.meta.json 随行，迁移后增量判定
    全命中 → 升级当天启动零重建、零人工步骤。新装机器无旧布局，零影响。

    校验: 旧 labels.txt 必须与当前 source 推导的标签一致——旧扁平目录里的产物
    理应是当前 source 对应版本（历史运维方式恒成立），不符即 fail loud（产物
    归属不明，静默迁移会复活「权重与标签错配」的老坑），提示人工处置。
    """
    legacy_onnx = legacy_dir / ONNX
    if not legacy_onnx.exists():
        return
    if (out_dir / ONNX).exists():
        print(f"### {name}: 旧扁平产物与版本目录并存（{out_dir} 已有产物）——"
              f"旧产物原样保留，请人工确认后删除 {legacy_dir} 下的散落文件")
        return

    legacy_labels = legacy_dir / LABELS
    if legacy_labels.exists():
        got = [ln for ln in legacy_labels.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if got != expected_labels:
            sys.exit(
                f"{name}: 旧扁平产物疑似不属于当前 source（labels 不一致:\n"
                f"    旧 {got}\n    新 {expected_labels}）\n"
                f"  拒绝自动迁移。请人工处置后重试:\n"
                f"    - 归档: mv {legacy_dir} <备份路径> && rm -f {legacy_dir}\n"
                f"    - 或确认作废后直接删除: rm -rf {legacy_dir}"
            )
    else:
        print(f"### {name}: 旧扁平产物缺少 labels.txt，无法校验版本归属，仅按 mtime 链继续")

    out_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for fname in LEGACY_FILES:
        src = legacy_dir / fname
        if src.exists():
            shutil.move(str(src), str(out_dir / fname))
            moved.append(fname)
    print(f"### {name}: 已迁移旧扁平布局 → {out_dir}（{', '.join(moved)}）"
          f"—— mtime/指纹保留，命中增量判定则零重建")


def build_one(name: str, source: str, uid: int, kind: str, violation: str | None,
              operate_on_uid: int, base: Path, force: bool = False,
              attach: str | None = None) -> None:
    """构建单个模型：pt→onnx→(class0 交换)→engine→<base>/generated/<name>/<版本键>/。

    source 相对路径相对部署根 base 解析；kind/violation/attach 已由 parse_gies 校验。
    attach 非 None 时为二级检测器（process-mode=2，挂在锚点检出框上）。
    """
    gen_root = Path(base) / GENERATED_DIRNAME
    src = Path(source)
    if not src.is_absolute():
        src = Path(base) / src
    if not src.exists():
        sys.exit(f"source 不存在: {src}")

    from ultralytics import YOLO
    model = YOLO(str(src))
    names = list(model.names.values())
    n_classes = len(names)
    h, w = model_imgsz(model)
    is_classify = (kind == KIND_CLASSIFIER)
    is_sgie_det = (kind == KIND_DETECTOR and attach is not None)
    max_batch = 32 if (is_classify or is_sgie_det) else 12

    # violation 校验 + 标签顺序（class0 前置；权重交换只在真正导出时做；
    # classifier 的 violation 由 parse_gies 保证必填）
    perm = None
    if is_classify:
        perm = _perm_for_violation(names, violation, name)
        if perm is not None:
            names = [names[i] for i in perm]
    elif violation is not None and violation not in names:
        raise ValueError(f"{name}: violation={violation!r} 不在模型类别 {names} 中")

    # detector 类别裁剪：基础 COCO 模型(>10类)只出 class0(person)；专用模型全量
    num_classes = n_classes if is_classify else (1 if n_classes > BASE_CLASS_THRESHOLD else n_classes)
    task_tag = (" (classifier)" if is_classify
                else " (sgie-detector)" if is_sgie_det else "")
    print(f"\n### 模型 {name}{task_tag}: {n_classes} 类 -> labels {names} "
          f"(imgsz={h}x{w}, max-batch={max_batch}, uid={uid})")

    # 产物目录：generated/<name>/<版本键>/（版本键由 source 推导，见 model_spec.artifact_version）
    # 命中旧扁平布局（升级自版本化之前）则先自动迁移，保证升级当天零重建。
    # 校验标签与下方落盘的 label_names 同一推导（classifier 已换序 / detector 裁剪）。
    ver_key = artifact_version(source)
    gen_name_dir = gen_root / name
    out_dir = gen_name_dir / ver_key
    _migrate_legacy_layout(gen_name_dir, out_dir, name,
                           expected_labels=names if is_classify else names[:num_classes])
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / ONNX
    engine_path = out_dir / ENGINE
    meta_path = out_dir / ENGINE_META
    cur_env = _env_fingerprint()

    def _env_match() -> bool:
        """引擎是否由当前环境构建——meta 缺失或指纹不符一律视为过期。"""
        if not meta_path.exists():
            return False
        try:
            return json.loads(meta_path.read_text(encoding="utf-8")) == cur_env
        except Exception:
            return False

    # --- 增量判定 ---
    pt_mtime = src.stat().st_mtime
    need_onnx = force or not onnx_path.exists() or onnx_path.stat().st_mtime < pt_mtime
    env_stale = engine_path.exists() and not _env_match()
    need_engine = (
        force or need_onnx or not engine_path.exists()
        or engine_path.stat().st_mtime < onnx_path.stat().st_mtime
        or env_stale
    )
    if env_stale:
        print(f"### {name}: 环境指纹变化（GPU/TRT 版本不符），引擎需重建")
    if not is_classify:
        compile_shared_parser(gen_root)

    if need_onnx:
        if perm is not None:
            _apply_class0_swap(model, perm, name, names)
        # --- 1) pt -> onnx (动态 batch) ---
        # 重定向 exporter 的输出锚点（ultralytics 按 model.pt_path 同名写 onnx），
        # 使 onnx 直接落到 generated/<name>/<版本键>/，不写源目录 —— models/ 在部署时为
        # :ro 挂载，任何旁路写入都会直接 PermissionError。
        model.model.pt_path = str(onnx_path.with_suffix(".pt"))
        if is_classify:
            exported = Path(model.export(format="onnx", dynamic=True, imgsz=(h, w)))
        else:
            exported = Path(model.export(format="onnx", nms=True, dynamic=True,
                                         opset=OP_SET, imgsz=(h, w)))
        # 正常情况重定向已直接命中 onnx_path；此块仅为 ultralytics 未来改动后的兜底。
        if exported.resolve() != onnx_path.resolve():
            shutil.copyfile(exported, onnx_path)
            exported.unlink(missing_ok=True)
        print(f"\n### ONNX 已生成: {onnx_path}")
    else:
        print(f"### {name}: {ONNX} 比 .pt 新，跳过导出")

    # --- 2) onnx -> engine (trtexec fp16, 动态 batch) ---
    if need_engine:
        run([
            "/usr/bin/trtexec",
            f"--onnx={onnx_path}",
            "--fp16",
            f"--minShapes={INPUT_NAME}:1x3x{h}x{w}",
            f"--optShapes={INPUT_NAME}:{max_batch}x3x{h}x{w}",
            f"--maxShapes={INPUT_NAME}:{max_batch}x3x{h}x{w}",
            f"--memPoolSize=workspace:{WORKSPACE_MB}M",
            f"--saveEngine={engine_path}",
        ], f"构建 TensorRT 引擎 (动态 batch 1~{max_batch})")
        meta_path.write_text(json.dumps(cur_env, ensure_ascii=False), encoding="utf-8")
    else:
        print(f"### {name}: {ENGINE} 已是最新且环境一致，跳过 trtexec")

    # --- 3) 产物落 generated/<name>/<版本键>/: labels.txt + INI（始终刷新，保证与 config 一致）---
    # detector 按 num-detected-classes 裁剪标签；classifier 全量（violation 已前置）
    label_names = names if is_classify else names[:num_classes]
    (out_dir / LABELS).write_text("\n".join(label_names) + "\n", encoding="utf-8")
    print(f"\n### labels.txt -> {out_dir / LABELS}: {label_names}")

    tpl_name = ("sgie_config.ini.tpl" if is_classify
                else "sgie_detector_config.ini.tpl" if is_sgie_det
                else "pgie_config.ini.tpl")
    tpl = (TEMPLATES / tpl_name).read_text(encoding="utf-8")
    ini = (tpl.replace("{{name}}", name)
              .replace("{{ver}}", ver_key)
              .replace("{{num_classes}}", str(num_classes))
              .replace("{{uid}}", str(uid))
              .replace("{{operate_on_uid}}", str(operate_on_uid)))
    ini_path = out_dir / ("sgie_config.txt" if (is_classify or is_sgie_det)
                          else "pgie_config.txt")
    ini_path.write_text(ini, encoding="utf-8")

    print(f"\n### 完成: {name}{task_tag}" +
          ("" if (need_onnx or need_engine) else "（产物已最新，未重建引擎）"))
    for rel in (onnx_path, engine_path, out_dir / LABELS, ini_path):
        if rel.exists():
            size_mb = rel.stat().st_size / 1e6
            print(f"    {rel.relative_to(base)}  ({size_mb:.1f} MB)")


def build_from_config(config_path: str, force: bool = False) -> None:
    """读 config.yaml 的 model.gies，逐个构建。部署根 = config.yaml 所在目录。"""
    import yaml

    cfg_path = Path(config_path).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    gies = parse_gies((cfg.get("model") or {}).get("gies"))
    anchor = anchor_uid(gies)
    base = cfg_path.parent
    print(f"### 部署根（config.yaml 所在目录）: {base}")
    for name, spec in gies.items():
        build_one(
            name=name,
            source=spec.source,
            uid=spec.uid,
            kind=spec.kind,
            violation=spec.violation,
            operate_on_uid=anchor,
            base=base,
            force=force,
            attach=spec.attach,
        )


def main():
    ap = argparse.ArgumentParser(description="DeepStream 模型一键构建（读 config.yaml 的 model.gies）")
    ap.add_argument("--config", required=True,
                    help="config.yaml 路径（部署根 = 其所在目录）")
    ap.add_argument("--force", action="store_true",
                    help="忽略增量检查，强制重新导出 onnx 并重建引擎")
    args = ap.parse_args()

    build_from_config(args.config, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
