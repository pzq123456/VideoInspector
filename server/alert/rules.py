"""
告警规则配置模型（从 YAML rules 节解析）

每个告警规则（no_helmet / no_vest / no_harness）独立持有策略:
  - cooldown_seconds:    触发告警后的冷却时长（同一规则同一摄像头）
  - min_detection_count: 连续检测到违规的帧数阈值（防单帧误报）
  - attribute_threshold: 违规置信度门限；同时作为 helmet 空间关联的门限

规则之间相互独立，各自维护状态机，互不竞争。
"""

from __future__ import annotations

from dataclasses import dataclass

# 探针可计算的告警维度；rules 节只能引用这些名字
KNOWN_RULES = ("no_helmet", "no_vest", "no_harness")


@dataclass(frozen=True)
class RuleConfig:
    """单条告警规则的独立策略。"""

    name: str
    cooldown_seconds: float
    min_detection_count: int
    attribute_threshold: float


def parse_rules(raw: dict | None) -> dict[str, RuleConfig]:
    """把 YAML rules 节解析成 {rule_name: RuleConfig}。

    Raises:
        ValueError: 规则名未知或参数非法（启动时 fail fast）。
    """
    rules: dict[str, RuleConfig] = {}
    for name, cfg in (raw or {}).items():
        if name not in KNOWN_RULES:
            raise ValueError(
                f"rules.{name}: 未知规则名，仅支持 {list(KNOWN_RULES)}"
            )
        cfg = cfg or {}
        cooldown = cfg.get("cooldown_seconds")
        min_count = cfg.get("min_detection_count")
        threshold = cfg.get("attribute_threshold")
        if not isinstance(cooldown, (int, float)) or cooldown < 0:
            raise ValueError(f"rules.{name}.cooldown_seconds 必须是非负数")
        if not isinstance(min_count, int) or min_count < 1:
            raise ValueError(f"rules.{name}.min_detection_count 必须是 >=1 的整数")
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            raise ValueError(f"rules.{name}.attribute_threshold 必须是 0~1 的置信度")
        rules[name] = RuleConfig(
            name=name,
            cooldown_seconds=float(cooldown),
            min_detection_count=int(min_count),
            attribute_threshold=float(threshold),
        )
    return rules
