"""
告警管理器
- 显式状态机按「规则」独立（每摄像头每规则一套），规则之间互不竞争:
    每条规则独立 冷却窗口 / 连续违规帧计数 / 违规置信度门限
- 连续帧确认：连续 N 帧检测到才触发告警（防止单帧误报）
- 告警分发：获取证据帧快照（已由 nvdsosd 原生渲染）→ executor 线程 JPEG 编码 →
           daemon 线程 fire-and-forget 完成 base64/JSON/HTTP 推送
  调用线程仅做决策，不做图片。
"""

import base64
import threading
import time
from datetime import datetime, timezone
from enum import Enum

import cv2
import simplejpeg
from loguru import logger

from server.metadata import ObjectMeta
from server.alert.rules import RuleConfig
from server.alert.webhook import WebhookAlerter


class AlertState(Enum):
    """告警状态机的显式状态。"""
    IDLE = "idle"           # 无近期违规，待触发
    ARMING = "arming"       # 正在累计连续违规帧
    COOLDOWN = "cooldown"   # 告警后冷却窗口，期间不再判定


class _RuleState:
    """单条规则独立的状态机（冷却 / 连续帧确认）。

    与摄像头解耦：同摄像头多条规则各自演进，互不竞争。
    """

    def __init__(self, rule: RuleConfig):
        self.rule = rule
        self._state = AlertState.IDLE
        self._consecutive_hits = 0
        self._cooldown_until = 0.0  # monotonic 时间戳：冷却结束时刻

    def handle(self, alert_objects: list[ObjectMeta]) -> list[ObjectMeta] | None:
        """处理一帧的本规则命中结果。

        Args:
            alert_objects: 携带本规则违规属性的 ObjectMeta 列表（空=无违规）。

        Returns:
            达到连续帧阈值时返回触发对象列表并进入冷却，否则 None。
        """
        now = time.monotonic()

        # COOLDOWN：冷却期内不做任何判定/计数，过期后回到 IDLE 重新武装
        if self._state == AlertState.COOLDOWN:
            if now < self._cooldown_until:
                return None
            self._state = AlertState.IDLE
            self._consecutive_hits = 0

        if alert_objects:
            self._state = AlertState.ARMING
            self._consecutive_hits += 1
            if self._consecutive_hits >= self.rule.min_detection_count:
                self._state = AlertState.COOLDOWN
                self._cooldown_until = now + self.rule.cooldown_seconds
                self._consecutive_hits = 0
                return alert_objects
        else:
            # 无违规 → 衰减（每帧 -1，容忍推理偶发丢帧），减到 0 回 IDLE
            if self._state == AlertState.ARMING:
                self._consecutive_hits -= 1
                if self._consecutive_hits <= 0:
                    self._consecutive_hits = 0
                    self._state = AlertState.IDLE
        return None


class AlertManager:
    """管理单路摄像头的告警状态与分发（内部按规则拆分独立状态机）。

    检测线程仅做轻量决策（按规则过滤 + 状态迁移）。
    JPEG 编码在 executor 线程执行，base64/JSON/HTTP POST
    由独立 daemon 线程 fire-and-forget，不做任何等待。
    """

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        rules: list[RuleConfig],
        webhook: WebhookAlerter | None = None,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头显示名称
            rules: 本路激活的告警规则（每条独立状态机，alert_type=规则名）
            webhook: Webhook 推送实例，None 则不推送
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.webhook = webhook

        # 每条规则一个独立状态机（rule_name → _RuleState）
        self._rule_states: dict[str, _RuleState] = {
            rule.name: _RuleState(rule) for rule in rules
        }

    # ------------------------------------------------------------------
    # 主入口：每帧调用（检测线程，仅做决策，不做图片）
    # ------------------------------------------------------------------
    def handle(self, objects: list[ObjectMeta],
               snapshot=None, executor=None, frame_info=None) -> bool:
        """
        处理一帧的检测结果，按规则独立判定是否触发告警。

        本方法在检测线程中运行，仅执行轻量决策逻辑。
        所有渲染（已由 nvdsosd 完成）/编码/网络 I/O 均通过 executor 异步卸载。

        Args:
            objects: 探针翻译层产出的 ObjectMeta 列表
            snapshot: 证据帧（nvdsosd 已渲染的 BGR numpy 数组），None 表示暂无证据帧
            executor: 可选 ThreadPoolExecutor，用于异步执行 webhook 推送
            frame_info: 可选 (decision_fn, decision_ts, snapshot_fn, snapshot_ts)，
                仅用于触发时打 evidence 同步日志（decision 与 snapshot 的帧/PTS 差）

        Returns:
            True 表示本帧至少触发了一条规则
        """
        triggered = False
        for rule_name, state in self._rule_states.items():
            # 只挑携带本规则违规属性的对象
            alert_objects = [
                obj for obj in objects
                if any(attr.name == rule_name for attr in obj.attributes)
            ]
            if alert_objects:
                attr_names = [
                    ", ".join(a.name for a in obj.attributes)
                    for obj in alert_objects
                ]
                logger.debug("detect camera={} rule={} objs={}", self.camera_name,
                             rule_name, attr_names)
            fired = state.handle(alert_objects)
            if fired:
                self._trigger(rule_name, fired, snapshot, executor, frame_info)
                triggered = True
        return triggered

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _trigger(self, rule_name: str, alert_objects: list[ObjectMeta],
                 snapshot, executor=None, frame_info=None):
        """触发告警：轻量调度 → 卸载所有重操作到 executor。

        本方法在检测线程中运行，仅做日志记录和状态更新。
        JPEG 编码 / base64 / JSON / HTTP POST 全部在 _build_and_send 中执行。

        Args:
            rule_name: 触发的规则名（随 payload.alert_type 输出）
            alert_objects: 触发告警的 ObjectMeta 列表
            snapshot: 证据帧（nvdsosd 已渲染的 BGR numpy 数组），None 表示暂无证据帧
            executor: 可选 ThreadPoolExecutor
            frame_info: 可选 (decision_fn, decision_ts, snapshot_fn, snapshot_ts)
        """
        now = datetime.now(timezone.utc)
        iso_timestamp = now.isoformat()

        best_obj = max(alert_objects, key=lambda o: o.confidence)
        logger.info("🚨 alert camera={} rule={} class={} conf={:.2f}",
                    self.camera_name, rule_name, best_obj.class_name,
                    best_obj.confidence)

        # 证据帧同步观测：decision（判定帧）与 snapshot（缓存帧）的帧号/PTS 差，
        # 用于验证「告警已判定但证据帧无框」是否源于帧滞后（Phase 1 只观测不补框）。
        if frame_info is not None:
            self._log_evidence_sync(rule_name, frame_info)

        # JPEG 编码在 executor 线程，base64/JSON/HTTP 由 daemon 线程 fire-and-forget。
        # snapshot 为 None 时仍推送（frame_base64=null），保证无证据帧时不丢告警。
        if self.webhook and executor is not None:
            executor.submit(
                self._build_and_send,
                rule_name, snapshot, alert_objects, iso_timestamp,
            )

    def _log_evidence_sync(self, rule_name: str, frame_info):
        """打一条 evidence 同步日志（camera_id / rule / 帧号与 PTS 差）。

        仅观测，不改变判定与取证行为；delta 为负或元信息缺失时相应字段为 None。
        """
        decision_fn, decision_ts, snap_fn, snap_ts = frame_info
        delta_fn = (decision_fn - snap_fn) if (decision_fn >= 0 and snap_fn >= 0) else None
        delta_ms = (
            round((decision_ts - snap_ts) / 1e6, 1)
            if decision_ts and snap_ts else None
        )
        logger.info(
            "evidence camera={} rule={} decision_fn={} snapshot_fn={} "
            "delta_fn={} decision_ts={} snapshot_ts={} delta_ms={}",
            self.camera_id, rule_name, decision_fn, snap_fn,
            delta_fn, decision_ts, snap_ts, delta_ms,
        )

    # ------------------------------------------------------------------
    # 后台线程：JPEG 编码（executor）→ fire-and-forget 推送（daemon）
    # ------------------------------------------------------------------
    def _build_and_send(self, rule_name: str, snapshot,
                        alert_objects: list[ObjectMeta], iso_timestamp: str):
        """JPEG 编码在 executor 线程执行（C 扩展，释放 GIL）。

        snapshot 是**未经 OSD 的原始帧**（证据支路挂在 nvdsosd 之前）；违规 red 框 +
        标签由 _render_evidence 用 payload 里的 ObjectMeta.bbox 现画，因此证据图
        是否带框与 nvdsosd / 预览 / 帧缓存滞后彻底无关，保证告警取证必有框。
        base64 / JSON / HTTP POST 交由独立 daemon 线程 fire-and-forget，
        executor 线程立即返回，不等待网络响应。
        """
        try:
            # 2. 补画违规框 + JPEG 编码（C 扩展，释放 GIL）— executor 线程唯一重操作；
            #    snapshot 为 None 时跳过，payload.frame_base64=null
            buffer = None
            if snapshot is not None:
                frame = self._render_evidence(snapshot, rule_name, alert_objects)
                buffer = simplejpeg.encode_jpeg(frame, quality=85, colorspace='BGR')

            # 3. Fire-and-forget: base64 → JSON → HTTP 全部在独立 daemon 线程
            threading.Thread(
                target=self._send_payload,
                args=(rule_name, buffer, alert_objects, iso_timestamp),
                daemon=True,
                name=f"webhook-send-{self.camera_id}-{rule_name}",
            ).start()

        except Exception:
            logger.exception("Webhook 构建异常")

    @staticmethod
    def _render_evidence(snapshot, rule_name: str,
                         alert_objects: list[ObjectMeta]):
        """在原始证据帧上用 cv2 补画违规 person 红框 + 规则名标签。

        返回**新数组**（不原地修改 FrameCache 持有的帧）。坐标为管线系 (x1,y1,x2,y2)，
        与 snapshot 同分辨率，直接对齐；越界自动裁剪。
        """
        frame = snapshot.copy()
        h, w = frame.shape[:2]
        color = (0, 0, 255)  # BGR 红
        for obj in alert_objects:
            x1, y1, x2, y2 = obj.bbox
            x1 = max(0, min(int(x1), w - 1))
            y1 = max(0, min(int(y1), h - 1))
            x2 = max(0, min(int(x2), w - 1))
            y2 = max(0, min(int(y2), h - 1))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            (_, th), _ = cv2.getTextSize(rule_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            ty = y1 - 8 if y1 - 8 - th >= 0 else min(h - 1, y2 + th + 8)
            cv2.putText(frame, rule_name, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return frame

    def _send_payload(self, rule_name: str, buffer: bytes | None,
                      alert_objects: list[ObjectMeta], iso_timestamp: str):
        """Fire-and-forget: base64 → JSON → HTTP POST。独立 daemon 线程。

        GIL 密集操作（base64、json.dumps）和阻塞 I/O（HTTP）均在此线程完成，
        不阻塞 executor 线程或检测/预览管线。
        """
        try:
            # 1. Base64 编码（无证据帧时为 None）
            frame_base64 = None
            if buffer is not None:
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # 2. 构建告警 payload（alert_type = 触发的规则名）
            objects_payload = []
            for obj in alert_objects:
                entry = {
                    "class": obj.class_name,
                    "confidence": round(obj.confidence, 3),
                    "bbox": list(obj.bbox),
                }
                attrs = [a for a in obj.attributes if a.name == rule_name]
                if attrs:
                    entry["attributes"] = [
                        {
                            "class": attr.name,
                            "confidence": round(attr.confidence, 3),
                            "bbox": list(attr.bbox) if attr.bbox is not None else None,
                        }
                        for attr in attrs
                    ]
                objects_payload.append(entry)

            payload = {
                "alert_type": rule_name,
                "camera_id": self.camera_id,
                "camera_name": self.camera_name,
                "timestamp": iso_timestamp,
                "objects": objects_payload,
                "frame_base64": frame_base64,
            }

            # 3. HTTP POST（阻塞 I/O，释放 GIL）
            self.webhook.send(payload)

        except Exception:
            logger.exception("Webhook 推送异常")
