#!/usr/bin/env python3
"""断流看门狗：解析 gst 输出判断管线是否已不可恢复。

背景（NVIDIA 论坛已知缺陷，DeepStream 7.x~9.x）:
  nvurisrcbin 内置 RTSP 重连在网络闪断后可能永久卡死：
    - 日志持续打印 "Resetting source -1, attempts: N"（或计数反复归 1）；
    - 该路 FPS 长期为 0 / 心跳缺失；
    - 真实错误被降级为 warning，应用层收不到任何 bus 消息。
  卡死后内置重连永远无法自愈，唯一可靠恢复手段是整条管线重建。

策略: 本模块只负责"全局级"检测（所有源都无健康帧）；逐路检测由
server/pipeline/health.py 的 SourceHealthMonitor 负责。两者共用
exit_process() 以 EXIT_PIPELINE_STUCK 退出，由主进程监督循环重建。
"""

import re
import threading
import time

from server.config import DEFAULT_RECONNECT

_RESET_RE = re.compile(r"Resetting source (-?\d+), attempts: (\d+)")
_FPS_RE = re.compile(r"\*\*FPS:")
# FPS 行格式: "**FPS:  24.60 (12.26)\t25.00 (12.95)" — 每源两列: 瞬时 (均值)
_FPS_VAL_RE = re.compile(r"(\d+(?:\.\d+)?)\s+\(")


def _any_positive_fps(line: str) -> bool:
    """FPS 心跳行里是否存在任一源的瞬时帧率 > 0。"""
    return any(float(v) > 0 for v in _FPS_VAL_RE.findall(line))


def exit_process(trigger: str, reason: str, detail: str = "", extra: str = "") -> None:
    """子进程统一退出入口（唯一的 fatal 落点）：health.py 与本模块共用。

    约定：fatal 只记录已经做出的决定。调用方传入已算好的上下文（extra），
    本函数只补 process 身份（multiprocessing.current_process().name，即
    监督循环所起的 safety-pipeline-{attempt}），不重新计算任何状态。
    格式：fatal trigger=<health|watchdog> reason=<code> process=<name>
          [<extra>] detail="<人话>" exitcode=2
    """
    from loguru import logger as _log
    from server.config import EXIT_PIPELINE_STUCK

    try:
        import multiprocessing
        proc = multiprocessing.current_process().name
    except Exception:
        proc = "unknown"
    try:
        if extra:
            _log.error('fatal trigger={} reason={} process={} {} detail="{}" exitcode={} 等待监督循环重建',
                       trigger, reason, proc, extra, detail, EXIT_PIPELINE_STUCK)
        else:
            _log.error('fatal trigger={} reason={} process={} detail="{}" exitcode={} 等待监督循环重建',
                       trigger, reason, proc, detail, EXIT_PIPELINE_STUCK)
    except Exception:
        print(f"[fatal] trigger={trigger} reason={reason} process={proc} {extra} ({detail})", flush=True)

    import os
    os._exit(EXIT_PIPELINE_STUCK)


class StreamWatchdog:
    """按行喂入 gst stdout 文本；判定卡死后异步退出子进程。"""

    def __init__(self, reconnect_cfg: dict | None = None,
                 maintenance_window: str | None = None):
        cfg = {**DEFAULT_RECONNECT, **(reconnect_cfg or {})}
        self._stall_seconds = float(cfg["stall_seconds"])
        self._window = maintenance_window
        self._last_healthy = time.monotonic()             # 最近一次有非零 FPS 的时刻
        self._saw_reset_since_healthy = False             # 距上次健康心跳间是否出现过重连
        self._max_attempt_seen = 0
        self._triggered = False

    def feed(self, line: str) -> None:
        if self._triggered:
            return
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
            # 维护窗口内（网关每日定时重启）不升级，窗口结束后再判定
            from server.pipeline.health import in_maintenance_window
            if in_maintenance_window(self._window):
                return
            self._triggered = True
            detail = (f"{now - self._last_healthy:.0f}s 无健康帧, "
                      f"max_attempts={self._max_attempt_seen}")
            extra = f"stall={self._stall_seconds:.0f}s"
            threading.Thread(target=exit_process, args=("watchdog", "no-healthy-frame", detail, extra),
                             daemon=True, name="watchdog-exit").start()
