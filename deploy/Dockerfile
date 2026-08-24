# =============================================================================
# 安全帽/反光衣检测服务 — 生产镜像（DeepStream 9.0）
# =============================================================================
# 与 .devcontainer 同源镜像（nvcr.io/nvidia/deepstream:9.0-triton-multiarch），
# 保证 pyservicemaker / GStreamer 插件（nvinfer/nvosd/nvrtspoutsinkbin）一致。
#
# 设计:
#   - 镜像只装代码与依赖；部署根 /app/deploy 的 config/models/generated/logs
#     由 docker-compose 挂载注入（见 deploy/docker-compose.yml）
#   - 引擎与 GPU/TRT 版本绑定，不能预烤进镜像: entrypoint 启动时增量构建
#     （pt→onnx→engine），产物持久化到宿主机 ./generated/，重启秒级跳过
#
# 构建: cd deploy && docker compose build
# 启动: cd deploy && docker compose up -d    # 首启自动构建引擎, 日志看 docker logs
# =============================================================================

FROM nvcr.io/nvidia/deepstream:9.0-triton-multiarch

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    SAFETY_CONFIG=/app/deploy/config.yaml

WORKDIR /app

# RTSP 输出服务依赖（开发容器曾手动安装的 4 个 apt 包，生产镜像需自带）
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-gi \
        gir1.2-gstreamer-1.0 \
        gir1.2-gst-plugins-base-1.0 \
        gir1.2-gst-rtsp-server-1.0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    -r requirements.txt

# ultralytics 会连带安装 GUI 版 opencv 覆盖 cv2 同路径，headless 强制收尾
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    --force-reinstall --no-deps opencv-python-headless==5.0.0.93

COPY tools/ ./tools/
COPY server/ ./server/

COPY deploy/entrypoint.sh /app/deploy/entrypoint.sh
RUN chmod +x /app/deploy/entrypoint.sh \
 && mkdir -p /app/deploy/models /app/deploy/generated /app/deploy/logs

ENTRYPOINT ["/app/deploy/entrypoint.sh"]
