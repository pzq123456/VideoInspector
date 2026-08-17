# =============================================================================
# 安全帽/反光衣检测服务 — 生产镜像（DeepStream 9.0）
# =============================================================================
# 与 .devcontainer/devcontainer.json 同源镜像（nvcr.io/nvidia/deepstream:9.0-
# triton-multiarch），保证 pyservicemaker / GStreamer 插件（nvinfer / nvosd /
# nvrtspoutsinkbin / nvstreammux）与开发环境一致。
#
# 构建：
#   docker build -t safety-detector:latest .
# 启动（开发测试）：
#   cd deploy && docker compose up -d
# =============================================================================

FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    -r requirements.txt

# 模型产物（引擎/自定义 parser/标签）与可移植 INI 配置（相对 /app 根路径）
COPY configs/ ./configs/
COPY models/ ./models/
COPY server/ ./server/

CMD ["python3", "-m", "server.main"]
