# =============================================================================
# 抽烟检测服务 — 生产镜像
# =============================================================================
# 与 .devcontainer/Dockerfile 同源（pytorch/pytorch CUDA 系列），
# 生产用 runtime（无编译工具链），dev 用 devel。
#
# 构建：
#   docker build -t smoke-detector:latest .
#
# 启动（开发测试）：
#   cd deploy && docker compose up -d
# =============================================================================

FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# -- 系统依赖（OpenCV 运行时库） ------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -- Python 依赖 ---------------------------------------------------------------
# torch / torchvision 已包含在基础镜像中，这里只装额外依赖
RUN pip install --no-cache-dir --break-system-packages \
    loguru \
    opencv-python \
    ultralytics \
    pyyaml \
    fastapi \
    uvicorn \
    simplejpeg \
    --extra-index-url https://pypi.nvidia.com nvidia-dali-cuda120
# -- 应用代码 ------------------------------------------------------------------
COPY server/ ./server/

# config.yaml 在构建时放入默认值，运行时通过 volume 挂载覆盖
# （deploy/docker-compose.yml 中已配置）

CMD ["python", "-m", "server.main"]
