"""
DeepStream 探针 → 告警状态机桥

把两阶段检测（person/helmet 整帧 + harness/vest 二级分类器）的 batch
元数据翻译成 server.metadata.ObjectMeta 列表，按 source_id 喂给对应的
AlertManager；同时给底层对象上色（nvdsosd 原生渲染，证据帧与实时预览
共享同一渲染源）。

流程（空间关联逻辑源自 simple-demo3/two_stage_demo.py 已验证方案）:
  - 整帧检测（process-mode=1，结果按 gie-unique-id 挂在帧级）:
      uid=1  person（yolo26n, 只出 person）
      uid=3  helmet（head/helmet）
  - 二级分类器（作用于 person 整框，结果以 NvDsClassifierMeta 挂在 person 对象）:
      uid=5  harness（harness/no_harness）
      uid=6  vest（vest/no_vest）
  - 第一趟: 收集 helmet 框的普通数值（class_id/conf/中心点），
    不持有元数据包装器（跨 pass 持有会段错误），同时给这些框上色。
  - 第二趟: 每个 person 做 helmet 空间关联（检测框中心落在 person 框内），
    并读取 harness/vest 分类器结果，
    违规翻译为 AttributeMeta('no_helmet') / AttributeMeta('no_harness') /
    AttributeMeta('no_vest')，并按状态给 person 框上色
    （红=任一违规 / 绿=三达标 / 蓝=有维度未知），
    违规者叠加文本标签（nvdsosd 的 per-object text_params）。

渲染约定（证据帧 = 完整 OSD 渲染帧，与实时预览一致）:
  - 任一违规（no_helmet / no_harness / no_vest）的人 → 红色框 + 违规标签
  - 三达标（helmet 且 harness 且 vest）的人         → 绿色框
  - 任一维度未知                                     → 蓝色框
  - head/helmet 框                → 红=head(未戴), 绿=helmet(已戴)

约束:
  - 本探针在 GStreamer 流线程运行，只做轻量决策；
    JPEG / base64 / HTTP 由 AlertManager 内部 executor + daemon 线程处理。
  - 上色必须在同一 pass 内完成（不能把元数据包装器存到 list 跨 pass 复用）。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from pyservicemaker import BatchMetadataOperator, osd
from loguru import logger

from server.metadata import AttributeMeta, ObjectMeta

RED = osd.Color(1.0, 0.0, 0.0, 1.0)    # 违规（no_helmet / no_harness / no_vest）
GREEN = osd.Color(0.0, 1.0, 0.0, 1.0)  # 三达标（helmet 且 harness 且 vest）
BLUE = osd.Color(0.0, 0.0, 1.0, 1.0)   # 有维度未知

PERSON_UID = 1      # pgie (yolo26n)
HELMET_UID = 3                # helmet 整帧检测器
HARNESS_CLS_UID = 5   # harness 二级分类器（作用于 person）
VEST_CLS_UID = 6      # vest 二级分类器（作用于 person）

# helmet 框的类别 id 对应模型 labels.txt 的行序
HELMET_CLASS_ID = 1   # helmet: 0=head(未戴), 1=helmet(已戴)
HEAD_CLASS_ID = 0

HARNESS_OK_LABELS = {"harness"}
HARNESS_VIOLATION_LABELS = {"no_harness"}
VEST_OK_LABELS = {"vest"}
VEST_VIOLATION_LABELS = {"no_vest"}
CLASSIFIER_LABELS = HARNESS_OK_LABELS | HARNESS_VIOLATION_LABELS | VEST_OK_LABELS | VEST_VIOLATION_LABELS
# 分类器置信度解析失败时的回退值（= sgie INI 的 classifier-threshold）
CLASSIFIER_FALLBACK_CONF = 0.5


class SafetyProbe(BatchMetadataOperator):
    """逐帧: 空间关联 → ObjectMeta 列表 → 对应摄像头的 AlertManager.handle()。

    Args:
        alert_managers: {source_id(int): AlertManager}，每路摄像头一个状态机。
        executor: ThreadPoolExecutor，透传给 AlertManager 用于异步 webhook。
        frame_cache: 可选 FrameCache，触发告警时取该 source 最新已渲染帧作 snapshot。
        helmet_conf_threshold: 头盔框空间关联 person 的最低置信度（低于视为噪声）。
        person_conf_threshold: person 检测置信度门槛（低于视为噪声：不判定、不渲染、不告警）。
    """

    def __init__(self, alert_managers: dict,
                 executor: ThreadPoolExecutor | None = None,
                 frame_cache=None,
                 helmet_conf_threshold: float = 0.5,
                 person_conf_threshold: float = 0.4):
        super().__init__()
        self._managers = alert_managers
        self._executor = executor
        # 证据帧缓存（FrameCache）: 触发告警时取该 source 最新已渲染帧作 snapshot。
        # None 表示未启用证据帧采集，行为与之前一致（frame_base64=null）。
        self._frame_cache = frame_cache
        # 空间关联置信度门槛（来自 server/config.yaml alert.*，须 >= INI 的
        # pre-cluster-threshold，否则低置信度框已在模型侧被裁掉、探针看不到）
        self._helmet_conf_threshold = helmet_conf_threshold
        # person 检测置信度门槛（低于视为噪声：不参与关联/判定/渲染/告警）
        self._person_conf_threshold = person_conf_threshold

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            # 第一趟: 收集 helmet 检测框的普通数值，避免持有元数据包装器；
            # 同时给这些框上色（nvdsosd 原生渲染，与证据帧/预览共享）
            helmet_boxes = []  # [(class_id, conf, cx, cy)]
            for o in frame_meta.object_items:  # 一次性迭代器
                if o.unique_component_id == HELMET_UID:
                    o.rect_params.border_color = (
                        GREEN if o.class_id == HELMET_CLASS_ID else RED
                    )
                    helmet_boxes.append(self._box_vals(o))

            # 第二趟: 每个 person → ObjectMeta（违规进 attributes）+ 状态上色
            objects: list[ObjectMeta] = []
            for obj in frame_meta.object_items:
                if obj.unique_component_id != PERSON_UID:
                    continue
                if obj.confidence < self._person_conf_threshold:
                    continue

                h, h_conf = self._helmet_status(
                    self._matched_conf(obj, helmet_boxes, self._helmet_conf_threshold)
                )
                har_status, har_label, har_conf = self._classifier_status(
                    obj.classifier_items, HARNESS_CLS_UID,
                    HARNESS_OK_LABELS, HARNESS_VIOLATION_LABELS,
                )
                vest_status, vest_label, vest_conf = self._classifier_status(
                    obj.classifier_items, VEST_CLS_UID,
                    VEST_OK_LABELS, VEST_VIOLATION_LABELS,
                )

                attrs = []
                if h == "no_helmet":
                    attrs.append(AttributeMeta("no_helmet", h_conf))
                if har_status == "violation":
                    attrs.append(AttributeMeta(
                        har_label,
                        har_conf if har_conf is not None else CLASSIFIER_FALLBACK_CONF,
                    ))
                if vest_status == "violation":
                    attrs.append(AttributeMeta(
                        vest_label,
                        vest_conf if vest_conf is not None else CLASSIFIER_FALLBACK_CONF,
                    ))

                # 任一违规=红；三达标=绿；有维度未知=蓝
                violation = (
                    h == "no_helmet"
                    or har_status == "violation"
                    or vest_status == "violation"
                )
                obj.rect_params.border_width = 2
                if violation:
                    obj.rect_params.border_color = RED
                    self._set_osd_label(
                        obj, " ".join(a.name for a in attrs)
                    )
                elif h == "helmet" and har_status == "ok" and vest_status == "ok":
                    obj.rect_params.border_color = GREEN
                else:
                    obj.rect_params.border_color = BLUE

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
                # 快照只取缓存引用（不拷贝），仅在真正触发告警时才被读取/编码
                snapshot = (
                    self._frame_cache.latest(frame_meta.source_id)
                    if self._frame_cache else None
                )
                manager.handle(objects, snapshot=snapshot, executor=self._executor)
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
    def _set_osd_label(o, text: str):
        """在 person 框上方叠加违规文本标签（nvdsosd per-object text_params）。"""
        if not text:
            return
        o.text_params.display_text = text.encode("ascii")
        o.text_params.x_offset = int(o.rect_params.left)
        o.text_params.y_offset = max(0, int(o.rect_params.top) - 6)
        o.text_params.font_params.name = osd.FontFamily.Serif
        o.text_params.font_params.size = 12
        o.text_params.font_params.color = RED

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

    # 每个 uid 只 DEBUG 一次 get_n_label 原始格式（避免热路径逐人逐帧刷屏）
    _logged_label_formats: set[int] = set()

    @classmethod
    def _classifier_status(cls, classifier_items, uid, ok_labels, violation_labels):
        """读 person 对象的二级分类器结果 → (status, label, conf)。

        status: "ok" / "violation" / None（无结果或未命中）
        classifier_items 为 ObjectMetadata.classifier_items（可迭代），
        每个元素有 unique_component_id / n_labels / get_n_label(i)->str。
        """
        status = label = conf = None
        try:
            for clf in classifier_items:
                if getattr(clf, "unique_component_id", None) != uid:
                    continue
                for i in range(getattr(clf, "n_labels", 0)):
                    raw = clf.get_n_label(i)
                    parsed_label, parsed_conf = cls._parse_classifier_label(raw)
                    if uid not in cls._logged_label_formats:
                        cls._logged_label_formats.add(uid)
                        logger.debug("classifier uid={} raw={!r} parsed=({}, {})", uid, raw, parsed_label, parsed_conf)
                    if parsed_label in violation_labels:
                        return "violation", parsed_label, parsed_conf
                    if status is None and parsed_label in ok_labels:
                        status, label, conf = "ok", parsed_label, parsed_conf
        except Exception:
            logger.exception("读取分类器结果异常")
        return status, label, conf

    @staticmethod
    def _parse_classifier_label(raw: str):
        """防御性解析 get_n_label 字符串 → (label_name | None, confidence | None)。

        格式未知，可能为纯 label（"vest"）或带置信度（"vest:0.98" / "vest 0.98"）。
        用已知类名集合匹配 label token；数值 token 当作置信度。
        """
        if not raw:
            return None, None
        tokens = re.split(r"[: \t]+", raw.strip())
        label = None
        for t in tokens:
            if t in CLASSIFIER_LABELS:
                label = t
                break
        conf = None
        for t in tokens:
            try:
                conf = float(t)
            except ValueError:
                pass
        return label, conf
