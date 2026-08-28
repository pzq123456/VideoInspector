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
# FPS 行格式: "**FPS:  24.60 (12.26)\t25.00 (12.95)" — 每源两列: 瞬时 (均值)
_FPS_VAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s+\(")


def _any_positive_fps(line: str) -> bool:
    """FPS 心跳行里是否存在任一源的瞬时帧率 > 0。"""
    return any(float(v) > 0 for v in _FPS_VAL_RE.findall(line))


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
        # FPS 心跳：只有存在瞬时 FPS > 0 的源才算活性信号。
        # 全 0 心跳（**FPS:  0.00 (x.xx) ...）不算健康——attempts 耗尽后 nvurisrcbin
        # 僵死，mux 仍按 batched-push-timeout 出空批次，FPS probe 会持续打 0.00，
        # 此前版本把这类心跳当活性导致看门狗永不触发（2026-08-27 事故根因）。
        elif _FPS_RE.search(line):
            if _any_positive_fps(line):
                self._last_healthy = now
                self._saw_reset_since_healthy = False
                self._max_attempt_seen = 0
            else:
                self._saw_reset_since_healthy = True

        # 卡死判定（每行统一出口）:
        # 出现过断流且超过 stall_seconds 没有任何健康帧 → 僵死。
        # attempts 耗尽后连 Resetting 日志都会消失，只剩全 0 心跳，只能靠此条件兜底。
        if self._saw_reset_since_healthy and now - self._last_healthy > self._stall_seconds:
            self._trigger("长时间无健康帧心跳", f"{now - self._last_healthy:.0f}s, "
                                             f"max_attempts={self._max_attempt_seen}")

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
