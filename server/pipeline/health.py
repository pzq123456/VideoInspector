#!/usr/bin/env python3
"""逐路源健康监控（Phase 1, cost-based）。

背景: nvurisrcbin 的内置重连消耗完 attempts 后进入"零信号僵尸态"（不重连、
不报错、不退出），且只要还有一路活着，FPS 心跳就显示健康——DS 对"部分死亡"
完全失明（2026-08-27/28 三轮泡测实证）。

本模块用 SafetyProbe 逐帧回调里的 source_id 做 ground truth:
  - mux 往下游推了帧 = 该路活着（帧计数 + 最近帧时刻，O(1) 开销）;
  - 巡检线程周期产出逐路健康日志 + stalled 集合;
  - 按代价模型决策，升级动作 = 子进程退出(exitcode=2)，由监督循环重建
    （Phase 2 换成 REST 单路重建）。

## 优化模型（带切换成本的在线区间覆盖）

目标: min( K·|R| + α·Σ UncoveredTime(i) )——用最少重建次数最大化
每路相机在线时段的 AI 覆盖。相机动态上下线（不可预知），重建代价固定 K。

在线策略 = 代价累积 + 防抖 + 维护窗口拟合:
  W(t) = Σ_i 未覆盖秒数（自该路停滞起累积；恢复即清零）
  触发重建:
    a) stalled ≥ 一半            → 立即（重大降级安全阀）
    b) W ≥ rebuild_cost_seconds  → 代价阈值触发（核心机制，K 的等价折算）
    且 now - last_rebuild ≥ min_rebuild_interval  # 防风暴硬地板
    且 非维护窗口（网关每日定时重启，窗口内只积累 W 不动手，窗口后一并处理）
    且 非预热期（子进程刚启动，源连接需要时间）
  多路同时断 → W 增长更快 → 响应自动加速；单路断 → 慢慢攒够 K 才动手。
  相机"事后回归"无需感知：其未覆盖时长持续累积 W，迟早触发一次重建将其接上。
"""

import threading
import time
from datetime import datetime, timedelta

from server.config import EXIT_PIPELINE_STUCK
from server.watchdog import exit_process


def in_maintenance_window(window: str | None, grace_minutes: int = 5) -> bool:
    """当前 UTC 时刻是否落在网关维护窗口（每日 HH:MM ± grace_minutes）内。

    注意时区: 服务器日志为 UTC，网关定时重启发生在北京时间 00:04 = UTC 16:04。
    配置里的 HH:MM 按 UTC 解释。
    """
    if not window:
        return False
    try:
        h, m = (int(x) for x in window.split(":"))
        target = datetime.utcnow().replace(hour=h, minute=m, second=0, microsecond=0)
    except (ValueError, AttributeError):
        return False
    delta = timedelta(minutes=grace_minutes)
    return target - delta <= datetime.utcnow() <= target + delta


class SourceHealthMonitor:
    """逐路帧计数 + 巡检决策。record() 在 GStreamer 流线程调用，必须 O(1) 无锁竞争。"""

    def __init__(self, cameras: list[dict], health_cfg: dict):
        from server.config import DEFAULT_HEALTH

        cfg = {**DEFAULT_HEALTH, **(health_cfg or {})}
        self._stall_seconds = float(cfg["source_stall_seconds"])
        self._check_interval = float(cfg["check_interval"])
        self._window = cfg.get("maintenance_window")
        self._min_uptime = float(cfg["min_uptime_seconds"])
        self._rebuild_cost = float(cfg["rebuild_cost_seconds"])
        self._min_rebuild_interval = float(cfg["min_rebuild_interval"])

        self._camera_ids = [c["id"] for c in cameras]
        self._last_frame = [None] * len(self._camera_ids)   # monotonic
        self._frames = [0] * len(self._camera_ids)
        self._ever_alive = [False] * len(self._camera_ids)  # 该路是否曾出过帧
        self._stall_start: list[float | None] = [None] * len(self._camera_ids)
        self._last_rebuild = time.monotonic()   # 子进程启动即视作一次"重建"
        self._lock = threading.Lock()

        self._started_at = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------------------------------------------------- 热路径（探针调用）
    def record(self, source_id: int) -> None:
        try:
            with self._lock:
                if 0 <= source_id < len(self._frames):
                    self._last_frame[source_id] = time.monotonic()
                    self._frames[source_id] += 1
                    self._ever_alive[source_id] = True
        except Exception:
            pass  # 健康监控绝不能影响推理热路径

    # ---------------------------------------------------------- 生命周期
    def start(self) -> None:
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="source-health")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ---------------------------------------------------------- 查询
    def snapshot(self) -> dict[int, float | None]:
        """{source_id: 距最近一帧的秒数}，从未出帧的源为 None。"""
        now = time.monotonic()
        with self._lock:
            return {
                i: (None if t is None else now - t)
                for i, t in enumerate(self._last_frame)
            }

    def stalled(self, stall_seconds: float | None = None) -> set[int]:
        limit = self._stall_seconds if stall_seconds is None else stall_seconds
        return {i for i, age in self.snapshot().items()
                if age is None or age > limit}

    # ---------------------------------------------------------- 巡检线程
    def _loop(self):
        from loguru import logger

        while not self._stop.wait(self._check_interval):
            try:
                self._check(logger)
            except Exception:
                logger.exception("source-health 巡检异常")

    def _check(self, logger) -> None:
        now = time.monotonic()
        ages = self.snapshot()
        stalled = {i for i, a in ages.items() if a is None or a > self._stall_seconds}

        for i, cam_id in enumerate(self._camera_ids):
            age = ages[i]
            status = "no-frame-yet" if age is None else (f"{age:.0f}s-ago")
            logger.info("health src={} camera={} frames={} last_frame={}",
                        i, cam_id, self._frames[i], status)

        # 维护未覆盖起点（stall_start）: 停滞即记起点，恢复即清零
        for i in range(len(self._camera_ids)):
            if i in stalled and self._stall_start[i] is None:
                self._stall_start[i] = now
            elif i not in stalled:
                self._stall_start[i] = None

        if not stalled:
            return
        if in_maintenance_window(self._window):
            # 窗口内继续累积 W，窗口结束后的巡检会统一结算（计划内重启顺延收编）
            logger.info("health: stalled={} 处于维护窗口({})，W 继续累积暂不升级",
                        sorted(stalled), self._window)
            return
        if now - self._started_at < self._min_uptime:
            logger.info("health: stalled={} 但子进程刚启动({:.0f}s < {:.0f}s)，暂不升级",
                        sorted(stalled), now - self._started_at, self._min_uptime)
            return

        n = len(self._camera_ids)
        ever = self._ever_alive
        zombie = sorted(i for i in stalled if ever[i])
        never = sorted(stalled - set(zombie))

        # --- 代价模型: W = Σ 未覆盖相机秒（自停滞起点累积） ---
        W = sum(now - s for s in self._stall_start if s is not None)
        since_rebuild = now - self._last_rebuild

        if len(stalled) * 2 >= n:
            self._escalate(logger, f"{len(stalled)}/{n} 路停滞（≥半数立即升级）", stalled)
        elif W >= self._rebuild_cost and since_rebuild >= self._min_rebuild_interval:
            self._escalate(logger,
                           f"累积未覆盖 W={W:.0f}s ≥ K={self._rebuild_cost:.0f}s"
                           f"（zombie={zombie} never={never}）", stalled)
        else:
            logger.warning(
                "health: stalled={} (zombie={} never={}) W={:.0f}/{:.0f}s "
                "距上次重建 {:.0f}s/{:.0f}s，观察中",
                sorted(stalled), zombie, never, W, self._rebuild_cost,
                since_rebuild, self._min_rebuild_interval)

    @staticmethod
    def _escalate(logger, reason: str, stalled: set[int]) -> None:
        logger.error("health: 判定需重建管线: {} (stalled={})，退出码={}",
                     reason, sorted(stalled), EXIT_PIPELINE_STUCK)
        exit_process("source-stalled", reason)

    @staticmethod
    def _escalate(logger, reason: str, stalled: set[int]) -> None:
        logger.error("health: 判定需重建管线: {} (stalled={})，退出码={}",
                     reason, sorted(stalled), EXIT_PIPELINE_STUCK)
        exit_process("source-stalled", reason)
