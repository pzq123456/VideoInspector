"""
DeepStream 探针 → 告警状态机桥

把三模型整帧检测（person / helmet / vest）的 batch 元数据翻译成
server.metadata.ObjectMeta 列表，按 source_id 喂给对应的 AlertManager。

流程（复用 simple-demo3/two_stage_demo.py 已验证的空间关联逻辑）:
  - 三模型都是 process-mode=1 整帧检测器，结果按 gie-unique-id 挂在帧级:
      uid=1  person（yolo26n, 只出 person）
      uid=3  helmet（head/helmet）
      uid=4  vest（vest/no_vest）
  - 第一趟: 收集 helmet / vest 框的普通数值（class_id/conf/中心点），
    不持有元数据包装器（跨 pass 持有会段错误）。
  - 第二趟: 每个 person 做空间关联（检测框中心落在 person 框内），
    违规翻译为 AttributeMeta('no_helmet') / AttributeMeta('no_vest')。

约束:
  - 本探针在 GStreamer 流线程运行，只做轻量决策；
    JPEG / base64 / HTTP 由 AlertManager 内部 executor + daemon 线程处理。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from pyservicemaker import BatchMetadataOperator
from loguru import logger

from server.metadata import AttributeMeta, ObjectMeta

# 各整帧检测器的 gie-unique-id（探针按此区分对象来源，而非 class_id）
PERSON_UID = 1      # pgie (yolo26n)
HELMET_UID = 3      # helmet 整帧检测器
VEST_UID = 4        # vest 整帧检测器

# 类别 id 对应各模型 labels.txt 的行序
HELMET_CLASS_ID = 1   # helmet: 0=head(未戴), 1=helmet(已戴)
HEAD_CLASS_ID = 0
VEST_CLASS_ID = 0     # vest: 0=vest(已穿), 1=no_vest(未穿)
NO_VEST_CLASS_ID = 1

# 关联 person 时要求的最低置信度（低于则视为噪声，不参与状态判定）
HELMET_CONF_THRESHOLD = 0.5
VEST_CONF_THRESHOLD = 0.5


class SafetyProbe(BatchMetadataOperator):
    """逐帧: 空间关联 → ObjectMeta 列表 → 对应摄像头的 AlertManager.handle()。

    Args:
        alert_managers: {source_id(int): AlertManager}，每路摄像头一个状态机。
        executor: ThreadPoolExecutor，透传给 AlertManager 用于异步 webhook。
    """

    def __init__(self, alert_managers: dict, executor: ThreadPoolExecutor | None = None):
        super().__init__()
        self._managers = alert_managers
        self._executor = executor

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            # 第一趟: 收集 helmet/vest 检测框的普通数值，避免持有元数据包装器
            helmet_boxes = []  # [(class_id, conf, cx, cy)]
            vest_boxes = []    # [(class_id, conf, cx, cy)]
            for o in frame_meta.object_items:  # 一次性迭代器
                if o.unique_component_id == HELMET_UID:
                    helmet_boxes.append(self._box_vals(o))
                elif o.unique_component_id == VEST_UID:
                    vest_boxes.append(self._box_vals(o))

            # 第二趟: 每个 person → ObjectMeta（违规进 attributes）
            objects: list[ObjectMeta] = []
            for obj in frame_meta.object_items:
                if obj.unique_component_id != PERSON_UID:
                    continue

                h, h_conf = self._helmet_status(
                    self._matched_conf(obj, helmet_boxes, HELMET_CONF_THRESHOLD)
                )
                v, v_conf = self._vest_status(
                    self._matched_conf(obj, vest_boxes, VEST_CONF_THRESHOLD)
                )

                attrs = []
                if h == "no_helmet":
                    attrs.append(AttributeMeta("no_helmet", h_conf))
                if v == "no_vest":
                    attrs.append(AttributeMeta("no_vest", v_conf))

                objects.append(ObjectMeta(
                    class_name="person",
                    confidence=obj.confidence,
                    bbox=(
                        int(obj.rect_params.left),
                        int(obj.rect_params.top),
                        int(obj.rect_params.left + obj.rect_params.width),
                        int(obj.rect_params.top + obj.rect_params.height),
                    ),
                    attributes=tuple(attrs),
                ))

            manager = self._managers.get(frame_meta.source_id)
            if manager is not None:
                manager.handle(objects, snapshot=None, executor=self._executor)
            elif objects:
                logger.debug("source_id={} 无对应 AlertManager，跳过告警判定",
                             frame_meta.source_id)

    # ------------------------------------------------------------------
    # 静态工具：与 simple-demo3 的 SafetyMarker 同源
    # ------------------------------------------------------------------
    @staticmethod
    def _box_vals(o):
        """提取检测框的普通数值（class_id/confidence/中心点），不持有元数据包装器。"""
        return (
            o.class_id,
            o.confidence,
            o.rect_params.left + o.rect_params.width / 2,
            o.rect_params.top + o.rect_params.height / 2,
        )

    @staticmethod
    def _matched_conf(obj, boxes, conf_threshold):
        """中心点落在 obj 框内、且置信度达标的 box → {class_id: best_conf}。"""
        px1 = obj.rect_params.left
        py1 = obj.rect_params.top
        px2 = px1 + obj.rect_params.width
        py2 = py1 + obj.rect_params.height
        matched = {}
        for cls_id, conf, cx, cy in boxes:
            if conf < conf_threshold:
                continue
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                if conf > matched.get(cls_id, 0.0):
                    matched[cls_id] = conf
        return matched

    @classmethod
    def _helmet_status(cls, matched):
        """戴帽状态; helmet 优先（同时命中 head 与 helmet 时按已戴帽处理）。

        Returns:
            (status, conf): status ∈ {"helmet", "no_helmet", None}
        """
        if HELMET_CLASS_ID in matched:
            return "helmet", matched[HELMET_CLASS_ID]
        if HEAD_CLASS_ID in matched:
            return "no_helmet", matched[HEAD_CLASS_ID]
        return None, None

    @classmethod
    def _vest_status(cls, matched):
        """反光衣状态; no_vest 优先（安全告警倾向，宁可多报）。"""
        if NO_VEST_CLASS_ID in matched:
            return "no_vest", matched[NO_VEST_CLASS_ID]
        if VEST_CLASS_ID in matched:
            return "vest", matched[VEST_CLASS_ID]
        return None, None
