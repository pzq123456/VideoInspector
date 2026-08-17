"""
告警管理器
- 冷却期控制：同一摄像头两次告警的最小间隔
- 连续帧确认：连续 N 帧检测到才触发告警（防止单帧误报）
- 告警分发：获取证据帧快照 → executor 线程完成 JPEG 编码 →
           daemon 线程 fire-and-forget 完成 base64/JSON/HTTP 推送
  调用线程仅做决策，不做图片。
"""

import base64
import threading
import time
from datetime import datetime, timezone

import cv2
import simplejpeg
from loguru import logger

from server.metadata import ObjectMeta
from server.alert.webhook import WebhookAlerter


def draw_alert_boxes(frame, alert_objects: list[ObjectMeta]):
    """在证据帧上画触发告警的人物框 + 违规标签（executor 线程调用，仅告警时）。

    只画触发告警的对象（alert_objects 已是按 target_classes 过滤的违规实体），
    避免像 OSD 那样把 head/helmet/vest/person 一堆框都画上去。
    原地修改 frame 并返回（该数组是缓存换出的旧快照，不会污染后续帧）。
    """
    for obj in alert_objects:
        x1, y1, x2, y2 = obj.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 红色违规框
        labels = [a.name for a in obj.attributes]
        label = f"{obj.class_name} {' '.join(labels)}".strip()
        if label:
            cv2.putText(frame, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


class AlertManager:
    """管理单路摄像头的告警状态与分发。

    检测线程仅做轻量决策（冷却期 / 连续帧确认）。
    JPEG 编码在 executor 线程执行，base64/JSON/HTTP POST
    由独立 daemon 线程 fire-and-forget，不做任何等待。
    """

    def __init__(
        self,
        camera_id: str,
        camera_name: str,
        target_classes: list[str] | None = None,
        save_frame_overlay: bool = False,
        cooldown_seconds: float = 30.0,
        min_detection_count: int = 3,
        alert_type: str | None = None,
        webhook: WebhookAlerter | None = None,
    ):
        """
        Args:
            camera_id: 摄像头唯一 ID
            camera_name: 摄像头显示名称
            target_classes: 触发告警的目标类别列表，None 则匹配所有检测。
                            会同时匹配 ObjectMeta.class_name 和 attributes 中的 name。
            save_frame_overlay: 是否在证据帧上叠加摄像头名称/时间水印
            cooldown_seconds: 同一摄像头两次告警的最小间隔（秒）
            min_detection_count: 连续检测到目标的帧数阈值
            alert_type: 告警类型标识（如 "ppe"），随 payload 原样输出，None 则不输出该字段
            webhook: Webhook 推送实例，None 则不推送
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.target_classes = target_classes
        self.save_frame_overlay = save_frame_overlay
        self.cooldown_seconds = cooldown_seconds
        self.min_detection_count = min_detection_count
        self.alert_type = alert_type
        self.webhook = webhook

        # 内部状态
        self._consecutive_hits = 0
        self._last_alert_time: float = 0.0

    # ------------------------------------------------------------------
    # 主入口：每帧调用（检测线程，仅做决策，不做图片）
    # ------------------------------------------------------------------
    def handle(self, objects: list[ObjectMeta],
               snapshot=None, executor=None) -> bool:
        """
        处理一帧的检测结果，决定是否触发告警。

        本方法在检测线程中运行，仅执行轻量决策逻辑。
        所有渲染/编码/网络 I/O 均通过 executor 异步卸载。

        Args:
            objects: 探针翻译层产出的 ObjectMeta 列表
            snapshot: 证据帧（标注后 BGR numpy 数组），None 表示暂无证据帧
            executor: 可选 ThreadPoolExecutor，用于异步执行 webhook 推送

        Returns:
            True 表示触发了告警
        """
        try:
            # 按 target_classes 过滤：匹配主实体 class_name 或其属性的 name
            alert_objects = self._find_alert_objects(objects)

            if alert_objects:
                self._consecutive_hits += 1
                for obj in alert_objects:
                    attr_names = [a.name for a in obj.attributes]
                    logger.debug("detect camera={} class={} attrs={} hits={}/{}",
                                 self.camera_name, obj.class_name, attr_names,
                                 self._consecutive_hits, self.min_detection_count)

                if self._consecutive_hits >= self.min_detection_count:
                    if self._cooldown_passed():
                        self._trigger(snapshot, alert_objects, executor=executor)
                        self._consecutive_hits = 0
                        return True
                    else:
                        remain = self.cooldown_seconds - (time.time() - self._last_alert_time)
                        logger.debug("skip camera={} reason=cooldown remain={:.0f}s",
                                     self.camera_name, remain)
                        self._consecutive_hits = 0
            else:
                # 无检测 → 递减计数器（逐步递减，容忍偶尔丢帧）
                if self._consecutive_hits > 0:
                    self._consecutive_hits -= 1
                    logger.debug("hits camera={} {}→{}", self.camera_name,
                                 self._consecutive_hits + 1, self._consecutive_hits)

            return False
        except Exception:
            logger.exception("[{}] 告警处理异常，跳过本帧", self.camera_name)
            return False

    def _find_alert_objects(self, objects: list[ObjectMeta]) -> list[ObjectMeta]:
        """找出触发告警的实体。

        匹配规则（两级）：
        1. ObjectMeta.class_name 在 target_classes 中（单阶段：cigarette 直接命中）
        2. 任意 AttributeMeta.name 在 target_classes 中（两阶段：person 拥有 cigarette 属性）

        target_classes 为 None 时返回全部实体。
        """
        if self.target_classes is None:
            return objects
        return [
            obj for obj in objects
            if obj.class_name in self.target_classes
            or any(attr.name in self.target_classes for attr in obj.attributes)
        ]

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _cooldown_passed(self) -> bool:
        """检查冷却期是否已过。"""
        elapsed = time.time() - self._last_alert_time
        return elapsed >= self.cooldown_seconds

    def _trigger(self, snapshot, alert_objects: list[ObjectMeta], executor=None):
        """触发告警：轻量调度 → 卸载所有重操作到 executor。

        本方法在检测线程中运行，仅做日志记录和状态更新。
        JPEG 编码 / base64 / JSON / HTTP POST 全部在 _build_and_send 中执行。

        Args:
            snapshot: 证据帧（标注后 BGR numpy 数组），None 表示暂无证据帧
            alert_objects: 触发告警的 ObjectMeta 列表
            executor: 可选 ThreadPoolExecutor
        """
        now = datetime.now(timezone.utc)
        iso_timestamp = now.isoformat()

        # 在 reset 前捕获当前命中计数（executor 线程稍后读取）
        hits = self._consecutive_hits

        best_obj = max(alert_objects, key=lambda o: o.confidence)
        logger.info("🚨 alert camera={} class={} conf={:.2f} hits={}/{}",
                    self.camera_name, best_obj.class_name, best_obj.confidence,
                    hits, self.min_detection_count)

        self._last_alert_time = time.time()

        # JPEG 编码在 executor 线程，base64/JSON/HTTP 由 daemon 线程 fire-and-forget。
        # snapshot 为 None 时仍推送（frame_base64=null），保证无证据帧时不丢告警。
        if self.webhook and executor is not None:
            executor.submit(
                self._build_and_send,
                snapshot, alert_objects, iso_timestamp,
            )

    # ------------------------------------------------------------------
    # 后台线程：JPEG 编码（executor）→ fire-and-forget 推送（daemon）
    # ------------------------------------------------------------------
    def _build_and_send(self, snapshot, alert_objects: list[ObjectMeta],
                        iso_timestamp: str):
        """JPEG 编码在 executor 线程执行（C 扩展，释放 GIL）。

        base64 / JSON / HTTP POST 交由独立 daemon 线程 fire-and-forget，
        executor 线程立即返回，不等待网络响应。
        """
        try:
            # 1. 标注：画触发告警的人物框 + 违规标签（仅 snapshot 非空时）
            if snapshot is not None:
                snapshot = draw_alert_boxes(snapshot, alert_objects)

            # 2. 可选：摄像头/时间水印（仅 snapshot 非空时）
            if snapshot is not None and self.save_frame_overlay:
                overlay_text = [
                    f"Camera: {self.camera_name} ({self.camera_id})",
                    f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
                ]
                for i, text in enumerate(overlay_text):
                    cv2.putText(
                        snapshot, text,
                        (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
                    )

            # 3. JPEG 编码（C 扩展，释放 GIL）— executor 线程唯一重操作；
            #    snapshot 为 None 时跳过，payload.frame_base64=null
            buffer = None
            if snapshot is not None:
                buffer = simplejpeg.encode_jpeg(snapshot, quality=85, colorspace='BGR')

            # 4. Fire-and-forget: base64 → JSON → HTTP 全部在独立 daemon 线程
            threading.Thread(
                target=self._send_payload,
                args=(buffer, alert_objects, iso_timestamp),
                daemon=True,
                name=f"webhook-send-{self.camera_id}",
            ).start()

        except Exception:
            logger.exception("Webhook 构建异常")

    def _send_payload(self, buffer: bytes, alert_objects: list[ObjectMeta],
                      iso_timestamp: str):
        """Fire-and-forget: base64 → JSON → HTTP POST。独立 daemon 线程。

        GIL 密集操作（base64、json.dumps）和阻塞 I/O（HTTP）均在此线程完成，
        不阻塞 executor 线程或检测/预览管线。
        """
        try:
            # 1. Base64 编码（无证据帧时为 None）
            frame_base64 = None
            if buffer is not None:
                frame_base64 = base64.b64encode(buffer).decode("utf-8")

            # 2. 构建告警 payload
            objects_payload = []
            for obj in alert_objects:
                entry = {
                    "class": obj.class_name,
                    "confidence": round(obj.confidence, 3),
                    "bbox": list(obj.bbox),
                }
                if obj.attributes:
                    entry["attributes"] = [
                        {
                            "class": attr.name,
                            "confidence": round(attr.confidence, 3),
                            "bbox": list(attr.bbox) if attr.bbox is not None else None,
                        }
                        for attr in obj.attributes
                    ]
                objects_payload.append(entry)

            payload = {
                "alert_type": self.alert_type,
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
