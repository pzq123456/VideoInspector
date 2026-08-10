"""
日志模块 — 基于 loguru
- 控制台彩色输出（开发友好）
- 文件 JSON 结构化日志（生产持久化）
- 自动轮转 + 压缩 + 过期清理
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    name: str = "smoke_detector",
    level: str = "INFO",
    log_file: str | None = None,
):
    """
    配置 loguru sinks。

    Args:
        name: 日志标识（写入 extra 字段）
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        log_file: 文件日志路径，None 表示只输出到控制台
    """
    logger.remove()  # 移除默认 sink

    # ── 控制台 sink：彩色文本，人类可读 ──
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<level>{message}</level>"
        ),
        level=level.upper(),
        colorize=True,
    )

    # ── 文件 sink：JSON 结构化，自动轮转 ──
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(log_path),
            format="{time} {level} {message}",
            serialize=True,          # JSON 格式
            rotation="10 MB",        # 每 10MB 轮转
            retention="30 days",     # 保留 30 天
            compression="gz",        # 旧文件压缩
            enqueue=True,            # 异步写入，不阻塞主线程
            level=level.upper(),
        )

    # 绑定服务名到所有日志
    return logger.bind(service=name)
