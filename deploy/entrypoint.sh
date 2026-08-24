#!/usr/bin/env bash
# =============================================================================
# 容器入口: 环境诊断(不阻断) → CUDA 功能探测(唯一硬门禁) → 增量构建 → 启动服务
#
# 为什么不做 nvidia-smi / 设备节点 / gst 插件的硬校验:
#   NVIDIA Container Toolkit 的 CDI 模式按 spec 精确挂载设备节点，常不带
#   /dev/nvidia-caps 等管理通道 —— nvidia-smi(NVML) 因此报错，但 CUDA 运行时
#   本身正常。结构性检查误报会阻断一个本来能跑的服务，而真故障会在后续
#   trtexec / pipeline 启动时立刻 fail loud，故一律只做诊断。
# =============================================================================
set -euo pipefail

CONFIG="${SAFETY_CONFIG:-/app/deploy/config.yaml}"

echo "================================================="
echo "[entrypoint] 1. 环境诊断 (仅提示, 不阻断) ..."
echo "================================================="

if nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2> /dev/null; then
    echo "✓ NVML 可用"
else
    echo "⚠️ nvidia-smi 不可用 —— CDI 模式/WSL2 下属常见现象(CUDA 不受影响), 继续..."
fi

for dev in /dev/nvidiactl /dev/nvidia-uvm; do
    [ -c "$dev" ] || echo "⚠️ 设备节点未直接映射: $dev (CDI 精确挂载下属正常现象)"
done

if gst-inspect-1.0 nvv4l2decoder > /dev/null 2>&1 || gst-inspect-1.0 nvdec > /dev/null 2>&1; then
    echo "✓ GStreamer NVIDIA 硬件解码插件可用"
else
    echo "⚠️ 未找到 nvv4l2decoder/nvdec 解码插件 —— 若 pipeline 启动失败请优先排查此项"
fi

echo "================================================="
echo "[entrypoint] 2. CUDA 功能探测 (唯一硬门禁) ..."
echo "================================================="
python3 -c '
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() == False —— GPU 不可用，无法构建引擎或推理"
print(f"✓ CUDA 就绪: {torch.cuda.get_device_name(0)} (torch {torch.__version__})")
'

echo "================================================="
echo "[entrypoint] 3. 增量构建模型产物 (config=${CONFIG}) ..."
echo "================================================="
python3 tools/model_build.py --config "${CONFIG}"

echo "================================================="
echo "[entrypoint] 4. 启动服务..."
echo "================================================="
exec python3 -m server.main --config "${CONFIG}"
