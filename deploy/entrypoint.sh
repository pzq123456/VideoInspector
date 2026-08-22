#!/usr/bin/env bash
# =============================================================================
# 容器入口: 增量构建模型产物 → 启动服务（"构建完成后自动启动"）。
#   - 产物已最新（onnx 比 .pt 新、engine 比 onnx 新、GPU/TRT 指纹一致）→ 秒级跳过
#   - 换模型 = 替换 deploy/models/ 下 .pt 后重启容器，自动只重建受影响部分
#   - 强制全量重建: 进容器执行 python3 tools/model_build.py --config "$SAFETY_CONFIG" --force
# =============================================================================
set -euo pipefail

CONFIG="${SAFETY_CONFIG:-/app/deploy/config.yaml}"

echo "[entrypoint] 增量构建模型产物 (config=${CONFIG}) ..."
python3 tools/model_build.py --config "${CONFIG}"

echo "[entrypoint] 启动服务..."
exec python3 -m server.main --config "${CONFIG}"
