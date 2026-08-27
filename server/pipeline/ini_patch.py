#!/usr/bin/env python3
"""nvinfer INI 配置的部署根锚定补丁。

nvinfer 解析 INI 内相对路径时基于进程 CWD，这里在启动时把模型路径显式锚定到
部署根（= config.yaml 所在目录：开发环境为 deploy/，镜像内为挂载的配置目录），
使同一份可移植 INI 两端通用，不依赖启动目录。
"""

import re
import tempfile
from pathlib import Path

_MODEL_PATH_KEYS = ("onnx-file", "model-engine-file", "labelfile-path", "custom-lib-path")
# 运行期必须存在、缺失即报错（onnx-file 仅用于重建引擎，不在此列）
_REQUIRED_PATH_KEYS = ("model-engine-file", "custom-lib-path", "labelfile-path")

_patched_dir: Path | None = None


def anchor_ini_config(src: Path, base: Path, classifier_threshold: float | None = None) -> str:
    """把 INI 内相对部署根的模型路径补全为绝对路径，返回 patched 文件路径。"""
    global _patched_dir
    if _patched_dir is None:
        _patched_dir = Path(tempfile.mkdtemp(prefix="safety-configs-"))

    out_lines, resolved = [], {}
    for raw in src.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_-]+)=(.*)$", raw.strip())
        if m and m.group(1) in _MODEL_PATH_KEYS:
            val = m.group(2).strip()
            if val and not Path(val).is_absolute():
                val = str(base / val)
                raw = f"{m.group(1)}={val}"
            resolved[m.group(1)] = val
        elif m and classifier_threshold is not None and m.group(1) == "classifier-threshold":
            raw = f"classifier-threshold={classifier_threshold}"
        out_lines.append(raw)

    missing = [
        f"{k}={v}" for k, v in resolved.items()
        if k in _REQUIRED_PATH_KEYS and v and not Path(v).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{src.name} 引用的模型产物缺失（请先运行 tools/model_build.py --config）:\n  "
            + "\n  ".join(missing)
        )

    out = _patched_dir / src.name
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return str(out)
