#!/usr/bin/env python3
"""子进程入口：初始化日志 → 构建管线 → 运行。

正常情况下永不返回（p.wait() 阻塞）。异常/看门狗触发时以非零码退出，
由 server/main.py 的监督循环决定是否重建。
"""

import os
from pathlib import Path

from server.pipeline.builder import PipelineBuilder
from server.utils.logger import setup_logger
from server.utils.stdio import forward_stdout_to_logger
from server.watchdog import StreamWatchdog


def serve(config: dict, deploy_root: Path) -> None:
    os.environ.setdefault("GST_DEBUG", "0")

    log_cfg = config.get("log") or {}
    log_file = os.environ.get("SAFETY_LOG_FILE") or log_cfg.get("file")
    if log_file and not Path(log_file).is_absolute():
        log_file = str(deploy_root / log_file)
    logger = setup_logger(
        name="safety_server",
        level=log_cfg.get("level", "INFO"),
        log_file=log_file,
    )
    watchdog = StreamWatchdog((config.get("source") or {}).get("reconnect"))
    forward_stdout_to_logger(logger, on_line=watchdog.feed)
    logger.info("startup safety_detection_server")

    builder = PipelineBuilder(config, deploy_root)
    runtime = builder.build(logger)
    logger.info("pipeline started cameras={}", len(builder.cameras))
    try:
        runtime["pipeline"].start().wait()
    finally:
        if runtime["rtsp_server"] is not None:
            runtime["rtsp_server"].stop()
        runtime["executor"].shutdown(wait=False, cancel_futures=True)
