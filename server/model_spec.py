"""
模型规格（model_spec）— config 中「报警模型」的声明式解析

config.yaml 的 model.gies 节是模型拓扑的单一事实来源：
- kind=detector   整帧检测器（process-mode=1）
- kind=classifier 二级分类器（process-mode=2，作用于 person 裁剪框）
- violation       报警类（即「no_xxx」对应的模型类别名；锚点检测器无此字段）

运行期（server/main.py、probe.py）只消费 GieSpec，不关心模型怎么编译；
编译期（tools/model_build.py --config）据此生成 <部署根>/generated/<name>/ 下的 INI + labels.txt。
部署根 = config.yaml 所在目录；source 相对路径与产物路径均相对部署根解析。
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
    source: str                     # 模型来源 .pt/.onnx（编译入口，相对部署根 = config 所在目录）
    uid: int                        # gie-unique-id，全局唯一
    violation: str | None = None    # 报警类标签（classifier 必填；detector 报警模型必填）
    attach: str | None = None       # 二级 detector 附属的锚点 gie 名（如 person）；空 = 整帧

    @property
    def is_secondary(self) -> bool:
        """二级模型（附属在锚点检出框上推理）：classifier 恒真，detector 看 attach。"""
        return self.kind == KIND_CLASSIFIER or self.attach is not None

    def config_path(self) -> str:
        """生成的 nvinfer INI 路径（generated/<name>/ 下）。"""
        prefix = "sgie_config" if self.is_secondary else "pgie_config"
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
        attach = cfg.get("attach")
        if attach is not None:
            if kind != KIND_DETECTOR:
                raise ValueError(f"model.gies.{name}.attach 仅 detector 支持（classifier 天然二级），当前 kind={kind!r}")
            if not violation:
                raise ValueError(f"model.gies.{name} 声明了 attach（二级检测器），必须同时声明 violation（报警类）")
            if not isinstance(attach, str) or not attach:
                raise ValueError(f"model.gies.{name}.attach 必须是非空字符串（锚点 gie 名），当前: {attach!r}")

        seen_uids[uid] = name
        gies[name] = GieSpec(
            name=name,
            kind=kind,
            source=source,
            uid=uid,
            violation=violation,
            attach=attach,
        )

    # 第二遍校验 attach 引用（锚点可能声明在后面，须全量解析后再查）
    for spec in gies.values():
        if spec.attach is None:
            continue
        target = gies.get(spec.attach)
        if target is None:
            raise ValueError(f"model.gies.{spec.name}.attach={spec.attach!r} 引用了未定义的 gie")
        if target.kind != KIND_DETECTOR or target.violation is not None:
            raise ValueError(
                f"model.gies.{spec.name}.attach={spec.attach!r} 必须指向锚点检测器"
                f"（kind=detector 且无 violation），当前 kind={target.kind}, violation={target.violation!r}"
            )
    return gies


def anchor_uid(gies: dict[str, GieSpec]) -> int:
    """返回锚点检测器（无 violation 的 detector）的 uid，供 classifier 的 operate-on-gie-id 使用。"""
    for spec in gies.values():
        if spec.kind == KIND_DETECTOR and spec.violation is None:
            return spec.uid
    raise ValueError("model.gies 缺少锚点检测器（无 violation 的 detector），classifier 无作用对象")
