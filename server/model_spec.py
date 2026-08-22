"""
模型规格（model_spec）— config 中「报警模型」的声明式解析

config.yaml 的 model.gies 节是模型拓扑的单一事实来源：
- kind=detector   整帧检测器（process-mode=1）
- kind=classifier 二级分类器（process-mode=2，作用于 person 裁剪框）
- violation       报警类（即「no_xxx」对应的模型类别名；锚点检测器无此字段）

运行期（server/main.py、probe.py）只消费 GieSpec，不关心模型怎么编译；
编译期（tools/model_build.py --config）据此生成 generated/<name>/ 下的 INI + labels.txt。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

KIND_DETECTOR = "detector"
KIND_CLASSIFIER = "classifier"
KNOWN_KINDS = (KIND_DETECTOR, KIND_CLASSIFIER)


@dataclass(frozen=True)
class GieSpec:
    """单个推理模型（gie）的声明式规格。"""

    name: str
    kind: str                       # detector | classifier
    source: str                     # 模型来源 .pt/.onnx（编译入口，相对项目根）
    uid: int                        # gie-unique-id，全局唯一
    violation: str | None = None    # 报警类标签（classifier 必填；detector 报警模型必填）

    def config_path(self) -> str:
        """生成的 nvinfer INI 路径（generated/<name>/ 下）。"""
        prefix = "sgie_config" if self.kind == KIND_CLASSIFIER else "pgie_config"
        return f"generated/{self.name}/{prefix}.txt"


def parse_gies(raw: dict | None) -> dict[str, GieSpec]:
    """解析 model.gies 节为 {gie_name: GieSpec}，启动/编译时 fail fast。

    Raises:
        ValueError: 字段非法 / uid 冲突 / classifier 缺 violation。
    """
    gies: dict[str, GieSpec] = {}
    seen_uids: dict[int, str] = {}
    for name, cfg in (raw or {}).items():
        cfg = cfg or {}
        kind = cfg.get("kind")
        source = cfg.get("source")
        uid = cfg.get("uid")
        violation = cfg.get("violation")

        if kind not in KNOWN_KINDS:
            raise ValueError(f"model.gies.{name}.kind 必须是 {KNOWN_KINDS}，当前: {kind!r}")
        if not source or not isinstance(source, str):
            raise ValueError(f"model.gies.{name}.source 为必填项")
        if not isinstance(uid, int) or uid <= 0:
            raise ValueError(f"model.gies.{name}.uid 必须是正整数，当前: {uid!r}")
        if uid in seen_uids:
            raise ValueError(
                f"model.gies.{name}.uid={uid} 与 model.gies.{seen_uids[uid]} 冲突（gie-unique-id 必须全局唯一）"
            )
        if kind == KIND_CLASSIFIER and (not violation or not isinstance(violation, str)):
            raise ValueError(f"model.gies.{name} 是 classifier，必须声明 violation（报警类）")
        if violation is not None and not isinstance(violation, str):
            raise ValueError(f"model.gies.{name}.violation 必须是字符串，当前: {violation!r}")

        seen_uids[uid] = name
        gies[name] = GieSpec(
            name=name,
            kind=kind,
            source=source,
            uid=uid,
            violation=violation,
        )
    return gies


def anchor_uid(gies: dict[str, GieSpec]) -> int:
    """返回锚点检测器（无 violation 的 detector）的 uid，供 classifier 的 operate-on-gie-id 使用。"""
    for spec in gies.values():
        if spec.kind == KIND_DETECTOR and spec.violation is None:
            return spec.uid
    raise ValueError("model.gies 缺少锚点检测器（无 violation 的 detector），classifier 无作用对象")
