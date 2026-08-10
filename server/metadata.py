"""
检测元数据契约（DeepStream 风格：物理实体 + 附属属性分离）

这是 DeepStream 探针 → 告警状态机之间的数据桥梁：
- 探针（SafetyMarker 之类）把 batch 元数据转换成 ObjectMeta 列表
- AlertManager 只消费 ObjectMeta / AttributeMeta，不关心检测来自 YOLO 还是 nvinfer

frozen=True 保证线程安全：探针线程写 → 告警消费者线程读，shallow copy 即足够。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttributeMeta:
    """附属属性或细粒度检测结果。

    bbox=None 时为纯分类属性（如 "is_smoking"、"helmet" / "no_helmet" 状态）；
    bbox 非 None 时为带空间定位的细粒度检测（全帧坐标）。
    """
    name: str
    confidence: float
    bbox: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class ObjectMeta:
    """物理检测实体。

    class_name 与 bbox 严格一致——bbox 就是该物理实体的边界。
    附属的细粒度检测/状态标签存放在 attributes 元组中。

    frozen=True 保证线程安全：跨线程共享时 shallow copy 即足够。
    """
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    attributes: tuple[AttributeMeta, ...] = ()
