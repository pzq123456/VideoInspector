#!/usr/bin/env python3
"""断流看门狗：解析 gst 输出判断管线是否已不可恢复。

背景（NVIDIA 论坛已知缺陷，DeepStream 7.x）:
  nvurisrcbin 内置 RTSP 重连在网络闪断后可能永久卡死：
    - 日志持续打印 "Resetting source -1, attempts: N"（或计数反复归 1）；
    - 该路 FPS 长期为 0 / 心跳缺失；
    - 真实错误被降级为 warning，应用层收不到任何 bus 消息。
  卡死后内置重连永远无法自愈，唯一可靠恢复手段是整条管线重建。

策略: 看门狗只在子进程内做"检测"，不做修复——检测到不可恢复即以
EXIT_PIPELINE_STUCK 退出，由主进程监督循环全量重建管线。
"""

import os
import re
import time

from server.config import EXIT_PIPELINE_STUCK, DEFAULT_RECONNECT

_RESET_RE = re.compile(r"Resetting source (-?\d+), attempts: (\d+)")
_FPS_RE = re.compile(r"\*\*FPS:")


class PipelineStuckError(RuntimeError):
    pass


class StreamWatchdog:
    """按行喂入 gst stdout 文本；卡死时抛 PipelineStuckError 或直接退出。"""

    def __init__(self, reconnect_cfg: dict | None = None):
        cfg = {**DEFAULT_RECONNECT, **(reconnect_cfg or {})}
        self._attempts_limit = int(cfg["attempts"]) * 3   # 计数器涨到 3×attempts 仍无进展 → 判卡死
        self._stall_seconds = float(cfg["stall_seconds"])
        self._last_healthy = time.monotonic()             # 最近一次有非零 FPS 的时刻
        self._saw_reset_since_healthy = False             # 距上次健康心跳间是否出现过重连
        self._max_attempt_seen = 0

    def feed(self, line: str) -> None:
        now = time.monotonic()

        m = _RESET_RE.search(line)
        if m:
            attempt = int(m.group(2))
            self._saw_reset_since_healthy = True
            if attempt > self._max_attempt_seen:
                self._max_attempt_seen = attempt
            # 单一来源计数器一路上涨却始终未见恢复 → 重连循环空转
            if self._attempts_limit > 0 and attempt >= self._attempts_limit and (
                now - self._last_healthy > self._stall_seconds
            ):
                self._trigger("reconnect 循环无进展", f"attempts={attempt}")
            return

        # FPS 心跳：出现即为活性信号（哪怕瞬时 0.00，也说明 mux 还在推帧）
        if _FPS_RE.search(line) and not line.endswith("0.00\t"):
            self._last_healthy = now
            self._saw_reset_since_healthy = False
            self._max_attempt_seen = 0
            return

        # 无任何活性信号超过 stall_seconds 且期间发生过断流重连 → 卡死
        if self._saw_reset_since_healthy and now - self._last_healthy > self._stall_seconds:
            self._trigger("长时间无健康帧心跳", f"{now - self._last_healthy:.0f}s")

    def _trigger(self, reason: str, detail: str) -> None:
        import threading

        threading.Thread(target=self._exit_process, args=(reason, detail),
                         daemon=True, name="watchdog-exit").start()

    @staticmethod
    def _exit_process(reason: str, detail: str) -> None:
        from loguru import logger as _log
        try:
            _log.error("看门狗判定管线不可恢复: {} ({})，退出码={} 等待监督循环重建",
                       reason, detail, EXIT_PIPELINE_STUCK)
        except Exception:
            print(f"[watchdog] pipeline stuck: {reason} ({detail})", flush=True)
        os._exit(EXIT_PIPELINE_STUCK)
