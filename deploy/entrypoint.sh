#!/usr/bin/env bash
# =============================================================================
# 容器入口: 硬件设备校验 → 增量构建模型产物 → 启动服务
# =============================================================================
set -euo pipefail

CONFIG="${SAFETY_CONFIG:-/app/deploy/config.yaml}"

echo "================================================="
echo "[entrypoint] 1. 正在检查 GPU 及硬件解码设备挂载状态..."
echo "================================================="

# Check 1: 检查 CUDA GPU 基础状态 (已修正语法错误)
if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    echo "❌ ERROR: 未检测到 NVIDIA GPU 或 nvidia-smi 无法运行！"
    echo "提示: 请检查 Docker 宿主机驱动及 GPU 容器运行时配置。"
    exit 1
fi
echo "✓ NVIDIA GPU 访问正常"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  └─ /'

# Check 2: 检查核心 GPU 字符设备文件
REQUIRED_DEVS=("/dev/nvidiactl" "/dev/nvidia-uvm")
MISSING_DEVS=()

for dev in "${REQUIRED_DEVS[@]}"; do
    if [ ! -c "$dev" ]; then
        MISSING_DEVS+=("$dev")
    fi
done

if [ ${#MISSING_DEVS[@]} -ne 0 ]; then
    echo "⚠️ WARNING: 静态设备节点未直接映射: ${MISSING_DEVS[*]}"
    echo "  └─ 在部分 WSL2 环境下属正常现象，继续校验 GStreamer 解码能力..."
else
    echo "✓ GPU 核心设备节点挂载检查通过 (${REQUIRED_DEVS[*]})"
fi

# Check 3: 终极早停断言 — 验证 GStreamer 的 nvv4l2decoder / nvdec 硬件解码能力
echo "[entrypoint] 校验 GStreamer 硬件解码插件 (nvv4l2decoder) ..."
if gst-inspect-1.0 nvv4l2decoder > /dev/null 2>&1; then
    echo "✓ GStreamer nvv4l2decoder (NVMM 硬件解码器) 加载成功！"
elif gst-inspect-1.0 nvdec > /dev/null 2>&1; then
    echo "✓ GStreamer nvdec (NVDEC 硬件解码器) 加载成功！"
else
    echo "❌ ERROR: GStreamer 找不到任何 NVIDIA 硬件视频解码插件 (nvv4l2decoder 或 nvdec)！"
    echo "早停拦截：当前环境缺少视频硬件解码依赖，终止后续耗时的模型构建。"
    exit 1
fi

echo "================================================="
echo "[entrypoint] 2. 增量构建模型产物 (config=${CONFIG}) ..."
echo "================================================="
python3 tools/model_build.py --config "${CONFIG}"

echo "================================================="
echo "[entrypoint] 3. 启动服务..."
echo "================================================="
exec python3 -m server.main --config "${CONFIG}"