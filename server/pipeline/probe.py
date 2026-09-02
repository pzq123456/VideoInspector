"""
DeepStream 探针 → 告警状态机桥（配置驱动）

把两阶段检测（person 锚点整帧检测 + detector 整帧 + classifier 二级分类）的 batch
元数据翻译成 server.metadata.ObjectMeta 列表，按 source_id 喂给对应的 AlertManager；
同时决定 OSD 渲染内容（nvdsosd 原生渲染，证据帧与实时预览共享同一渲染源）。

渲染策略「一刀切干净」：nvdsosd 默认会把每个对象按默认色画出来，因此**不画 = 显式隐藏**
（rect_params.border_width=0 + 清空 text_params.display_text——nvosd 对未设置 text 的
对象会自动显示 obj_label/分类器标签）。只保留违规 person 的红框 + 违规标签，其余一律
隐藏：detector 检测框（head/helmet/cigarette）、合规 person、低于置信度门槛的 person。
证据帧因此只含违规者标注，预览同。

nvosd 经验值：违规框 border_width 实测 2 不上屏、4 起正常渲染，取 4（1080p 下粗细
与辨识度均衡，GPU 光栅化对小宽度有下限）。

模型拓扑来自 config（model.gies + rules），探针不再硬编码任何 uid/规则名：
  - detector（如 helmet）：整帧检测器，空间关联（检测框中心落在 person 框内且
    label == violation）→ 违规
  - classifier（如 harness/vest）：读 person 对象上的 NvDsClassifierMeta，label == violation → 违规
  - 违规翻译为 AttributeMeta(rule_name)，rule_name 即 alert_type。

约束（与旧版一致）:
  - 本探针在 GStreamer 流线程运行，只做轻量决策；JPEG/base64/HTTP 由 AlertManager
    内部 executor + daemon 线程处理。
  - 隐藏/上色必须在同一 pass 内完成（object_items 是一次性迭代器，不能跨 pass 持有包装器）。
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from pyservicemaker import BatchMetadataOperator, osd
from loguru import logger

from server.metadata import AttributeMeta, ObjectMeta
from server.model_spec import KIND_CLASSIFIER, KIND_DETECTOR
from server.alert.rules import RuleConfig

RED = osd.Color(1.0, 0.0, 0.0, 1.0)    # 违规（唯一上屏色）

# 隐藏对象的两件事：边宽置 0（nvosd 不画框）+ 显式清空 display_text
# （nvosd 对未设置 text 的对象会自动显示 obj_label，如 helmet/person，必须显式压掉）
_HIDE_BOX = 0
_HIDE_TEXT = b""

# 分类器置信度解析失败时的回退值（payload 展示用，判定由 classifier-threshold 负责）
CLASSIFIER_FALLBACK_CONF = 0.5


class SafetyProbe(BatchMetadataOperator):
    """逐帧: 按激活规则计算违规 → ObjectMeta 列表 → 对应摄像头的 AlertManager.handle()。

    Args:
        alert_managers: {source_id(int): AlertManager}，每路摄像头一个状态机。
        executor: ThreadPoolExecutor，透传给 AlertManager 用于异步 webhook。
        frame_cache: 可选 FrameCache，触发告警时取该 source 最新已渲染帧作 snapshot。
        gies: {gie_name: GieSpec}，模型拓扑（来自 config）。
        rules: {rule_name: RuleConfig}，告警规则（gie 引用 gies）。
        person_uid: 锚点检测器（person）的 uid。
        person_conf_threshold: person 置信度门槛（低于视为噪声）。
        active_rules_by_source: {source_id: {rule_name}}，每路摄像头激活的规则。
    """

    def __init__(self, alert_managers: dict,
                 executor: ThreadPoolExecutor | None = None,
                 frame_cache=None,
                 gies: dict | None = None,
                 rules: dict[str, RuleConfig] | None = None,
                 person_uid: int = 1,
                 person_conf_threshold: float = 0.6,
                 active_rules_by_source: dict[int, set[str]] | None = None,
                 health=None):
        super().__init__()
        self._managers = alert_managers
        self._executor = executor
        self._frame_cache = frame_cache
        self._health = health   # SourceHealthMonitor（可选）: 逐路帧计数供看门狗判定
        self._gies = gies or {}
        self._rules = rules or {}
        self._person_uid = person_uid
        self._person_conf_threshold = person_conf_threshold
        self._active_rules_by_source = active_rules_by_source or {}
        self._all_rules = set(self._rules)

        # 预解析: rule → gie 规格（uid / kind / violation）
        # detector 与 classifier 分桶，热路径免查 kind。
        self._rule_binding: dict[str, tuple] = {}   # rule_name -> (uid, kind, violation)
        self._known_labels: set[str] = set()
        for rule_name, rule in self._rules.items():
            spec = self._gies[rule.gie]
            self._rule_binding[rule_name] = (spec.uid, spec.kind, spec.violation)
            if spec.violation:
                self._known_labels.add(spec.violation)

    def handle_metadata(self, batch_meta):
        health_record = self._health.record if self._health is not None else None
        for frame_meta in batch_meta.frame_items:
            source_id = frame_meta.source_id
            if health_record is not None:
                health_record(source_id)
            active = self._active_rules_by_source.get(source_id, self._all_rules)

            active_detector_uids: set[int] = set()
            for rule_name in active:
                uid, kind, _ = self._rule_binding[rule_name]
                if kind == KIND_DETECTOR:
                    active_detector_uids.add(uid)

            # 第一趟: 收集 detector 检测框的普通数值（label/conf/中心点），不持有包装器；
            # 同时隐藏这些框（违规呈现统一收敛到 person 红框 + 标签，不画 detector 框）。
            detector_boxes: dict[int, list] = {}
            for uid in active_detector_uids:
                boxes = []
                for o in frame_meta.object_items:  # 一次性迭代器
                    if o.unique_component_id == uid:
                        o.rect_params.border_width = _HIDE_BOX
                        o.text_params.display_text = _HIDE_TEXT
                        boxes.append((
                            getattr(o, "label", ""),
                            o.confidence,
                            o.rect_params.left + o.rect_params.width / 2,
                            o.rect_params.top + o.rect_params.height / 2,
                        ))
                detector_boxes[uid] = boxes

            # 第二趟: 每个 person → ObjectMeta（违规进 attributes）+ 渲染决策
            objects: list[ObjectMeta] = []
            for obj in frame_meta.object_items:
                if obj.unique_component_id != self._person_uid:
                    continue
                if obj.confidence < self._person_conf_threshold:
                    obj.rect_params.border_width = _HIDE_BOX  # 噪声框/字也须显式隐藏
                    obj.text_params.display_text = _HIDE_TEXT
                    continue

                attrs: list[AttributeMeta] = []
                for rule_name in active:
                    rule = self._rules[rule_name]
                    uid, kind, violation = self._rule_binding[rule_name]
                    if kind == KIND_DETECTOR:
                        conf = self._match_violation(
                            obj, detector_boxes.get(uid, []),
                            violation, rule.attribute_threshold,
                        )
                        if conf is not None:
                            attrs.append(AttributeMeta(rule_name, conf))
                    else:  # classifier
                        label = self._classifier_label(obj.classifier_items, uid)
                        if label == violation:
                            attrs.append(AttributeMeta(rule_name, CLASSIFIER_FALLBACK_CONF))

                # 有违规=红框+标签；无违规=隐藏（classifier 合规类不可观测）
                if attrs:
                    obj.rect_params.border_width = 4  # nvosd 怪癖: 2 不上屏，见模块 docstring
                    obj.rect_params.border_color = RED
                    self._set_osd_label(obj, " ".join(a.name for a in attrs))
                else:
                    obj.rect_params.border_width = _HIDE_BOX
                    obj.text_params.display_text = _HIDE_TEXT

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

            manager = self._managers.get(source_id)
            if manager is not None:
                snapshot = (
                    self._frame_cache.latest(source_id)
                    if self._frame_cache else None
                )
                manager.handle(objects, snapshot=snapshot, executor=self._executor)
            elif objects:
                logger.debug("source_id={} 无对应 AlertManager，跳过告警判定", source_id)

    # ------------------------------------------------------------------
    # 静态工具
    # ------------------------------------------------------------------
    @staticmethod
    def _match_violation(obj, boxes, violation_label, conf_threshold):
        """中心点落在 person 框内、label==violation、置信度达标的框 → 最高置信度（无则 None）。"""
        px1 = obj.rect_params.left
        py1 = obj.rect_params.top
        px2 = px1 + obj.rect_params.width
        py2 = py1 + obj.rect_params.height
        best = None
        for label, conf, cx, cy in boxes:
            if label != violation_label or conf < conf_threshold:
                continue
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                if best is None or conf > best:
                    best = conf
        return best

    @staticmethod
    def _set_osd_label(o, text: str):
        if not text:
            return
        o.text_params.display_text = text.encode("ascii")
        o.text_params.x_offset = int(o.rect_params.left)
        o.text_params.y_offset = max(0, int(o.rect_params.top) - 6)
        o.text_params.font_params.name = osd.FontFamily.Serif
        o.text_params.font_params.size = 12
        o.text_params.font_params.color = RED

    # 每个 uid 只 DEBUG 一次 get_n_label 原始格式（避免热路径逐人逐帧刷屏）
    _logged_label_formats: set[int] = set()

    @classmethod
    def _classifier_label(cls, classifier_items, uid: int) -> str | None:
        """读 person 对象的二级分类器结果 → 首个挂载的 label 名（class0，即违规类）。"""
        try:
            for clf in classifier_items:
                if getattr(clf, "unique_component_id", None) != uid:
                    continue
                for i in range(getattr(clf, "n_labels", 0)):
                    raw = clf.get_n_label(i)
                    label, _conf = cls._parse_classifier_label(raw)
                    if uid not in cls._logged_label_formats:
                        cls._logged_label_formats.add(uid)
                        logger.debug("classifier uid={} raw={!r} parsed={}", uid, raw, label)
                    return label
        except Exception:
            logger.exception("读取分类器结果异常")
        return None

    @staticmethod
    def _parse_classifier_label(raw: str):
        """防御性解析 get_n_label 字符串 → (label_name | None, confidence | None)。"""
        if not raw:
            return None, None
        tokens = re.split(r"[: \t]+", raw.strip())
        conf = None
        for t in tokens:
            try:
                conf = float(t)
            except ValueError:
                pass
        # label 取第一个非数值 token（get_n_label 只挂 class0，即违规类）
        for t in tokens:
            try:
                float(t)
            except ValueError:
                return t, conf
        return None, conf
