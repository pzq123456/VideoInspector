#!/usr/bin/env python3
"""配置加载与校验（deploy/config.yaml）。"""

import yaml
from pathlib import Path

from server.alert.rules import parse_rules
from server.model_spec import parse_gies


def load_config(config_path: str) -> dict:
    """加载并校验 YAML 配置文件。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    validate_config(config, path)
    return config


def validate_config(config: dict, path: Path):
    """校验: model.gies / rules / 摄像头(active_rules) / webhook / source.reconnect。"""
    errors = []

    model = config.get("model") or {}
    gies: dict = {}
    gies_raw = model.get("gies")
    if not gies_raw:
        errors.append("model.gies 为必填项（至少声明一个锚点检测器）")
    else:
        try:
            gies = parse_gies(gies_raw)
        except ValueError as exc:
            errors.append(str(exc))

    person_conf = model.get("person_conf_threshold")
    if person_conf is not None and not (
        isinstance(person_conf, (int, float)) and 0.0 <= person_conf <= 1.0
    ):
        errors.append(f"model.person_conf_threshold 必须是 0~1 的置信度阈值，当前: {person_conf!r}")

    rules_raw = config.get("rules")
    rule_names: set[str] = set()
    if not rules_raw:
        errors.append("缺少 'rules' 节（至少定义一个告警规则）")
    else:
        try:
            rules = parse_rules(rules_raw)
            rule_names = set(rules)
            for rule in rules.values():
                if rule.gie not in gies:
                    errors.append(f"rules.{rule.name}.gie 引用了未定义的模型: {rule.gie}")
        except ValueError as exc:
            errors.append(str(exc))

    cameras = [c for c in config.get("cameras") or [] if c.get("enabled", True)]
    if not cameras:
        errors.append("cameras 必须是非空列表（至少一个 enabled=true）")
    else:
        for cam in cameras:
            if cam.get("type", "rtsp") != "rtsp":
                errors.append(f"DeepStream 仅支持 RTSP 摄像头: {cam.get('id')} (type={cam.get('type')})")
            if not cam.get("rtsp_url"):
                errors.append(f"cameras[].rtsp_url 为必填项 (type=rtsp): {cam.get('id')}")
            active = cam.get("active_rules")
            if active is not None:
                if not isinstance(active, list) or not active:
                    errors.append(f"cameras[{cam.get('id')}].active_rules 必须是非空列表")
                else:
                    for rule in active:
                        if rule not in rule_names:
                            errors.append(f"cameras[{cam.get('id')}].active_rules 引用了未定义的规则: {rule}")

    webhook = config.get("webhook") or {}
    timeout = webhook.get("timeout")
    if webhook.get("url") and timeout is not None and not (
        isinstance(timeout, (int, float)) and timeout > 0
    ):
        errors.append(f"webhook.timeout 必须是正数，当前: {timeout!r}")

    errors.extend(_validate_source_reconnect((config.get("source") or {}).get("reconnect")))

    if errors:
        raise ValueError(f"配置校验失败 ({path}):\n  " + "\n  ".join(errors))


def _validate_int(v, name: str, minimum: int) -> str | None:
    if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
        return f"{name} 必须是不小于 {minimum} 的整数，当前: {v!r}"
    return None


def _validate_source_reconnect(rc_raw) -> list[str]:
    errors = []
    rc = rc_raw or {}
    checks = (
        ("attempts", 1),
        ("timeout", 1),
        ("interval", 1),
        ("latency", 0),
        ("stall_seconds", 60),
    )
    for key, minimum in checks:
        v = rc.get(key)
        if v is not None and (err := _validate_int(v, f"source.reconnect.{key}", minimum)):
            errors.append(err)
    return errors


def resolve_path(base: Path, p: str) -> str:
    """把配置里的相对路径解析为绝对路径（相对部署根 = config.yaml 所在目录）。"""
    return p if Path(p).is_absolute() else str(base / p)


# 管线子进程因「断流卡死被看门狗判定不可恢复」时的退出码。
# 主进程监督循环据此决定是否重建整条管线。
EXIT_PIPELINE_STUCK = 2

# 看门狗默认参数（可由 source.reconnect 覆盖）
DEFAULT_RECONNECT = {
    # nvurisrcbin 内置重连：消耗完 attempts 后该路源进入僵死状态，
    # 由看门狗检测并退出子进程、监督循环全量重建管线（见 server/watchdog.py）。
    "attempts": 5,        # 内置重连次数上限（有限值，不做无限空转）
    "timeout": 10,        # 多少秒无数据包即判为断流（5s 偏激进易误判）
    "interval": 10,       # 重连尝试间隔（秒）
    "latency": 500,       # jitterbuffer 缓冲（ms），过大延迟超时判定
    "stall_seconds": 180, # 连续多久没有任何健康帧心跳则认为管线卡死
}
