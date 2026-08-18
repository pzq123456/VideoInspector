"""
配置 Schema（Pydantic）：加载 + 校验 + 默认值

把 `server/config.yaml` 解析成强类型的 `AppConfig`，启动时 fail-fast 捕捉
非法配置（端口越界、置信度阈值越界、缺少必填字段、无启用摄像头等），
替代原先散落在 main.py 里的手写 `_validate_config`。

层级结构（对齐最新告警状态机与单端口 RTSP 输出拓扑）：
  model     — 三模型 nvinfer 配置文件路径
  cameras[] — 单路摄像头属性（id/name/url/enabled/targets 覆盖）
  alert     — 告警状态机参数（confirm_frames/clear_frames/cooldown/webhook）
  output    — RTSP 服务端全局配置（单端口/mount 前缀/shm 路径模板）
  log       — 日志
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class LogConfig(BaseModel):
    """日志配置。"""
    level: str = "INFO"
    file: str | None = None


class ModelConfig(BaseModel):
    """三模型 nvinfer 配置（INI 文本路径，相对项目根目录）。"""
    pgie_config: str
    helmet_config: str
    vest_config: str


class CameraConfig(BaseModel):
    """单路摄像头。

    targets 为可选：不填则回退到全局 `alert.targets`，填了则覆盖该路触发目标。
    """
    id: str
    name: str | None = None
    url: str
    enabled: bool = True
    targets: list[str] | None = None

    @field_validator("url")
    @classmethod
    def _rtsp_only(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith(("rtsp://", "rtsps://")):
            raise ValueError(f"仅支持 RTSP 摄像头，url 必须以 rtsp:// 或 rtsps:// 开头: {v!r}")
        return v


class WebhookConfig(BaseModel):
    """Webhook 推送配置。"""
    url: str | None = None
    timeout: float = 10.0
    retries: int = 2

    @field_validator("retries")
    @classmethod
    def _retries_ge_0(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"retries 必须 >= 0，当前 {v}")
        return v


class AlertConfig(BaseModel):
    """告警状态机参数（显式暴露防抖/恢复/冷却）。"""
    type: str = "ppe"
    targets: list[str] = Field(default_factory=lambda: ["no_helmet", "no_vest"])
    # 触发阈值：连续命中违规帧数才触发告警（防单帧误报）
    confirm_frames: int = 3
    # 恢复阈值：连续干净帧数才从 ARMING 回落到 IDLE（0 = 立即回落）
    clear_frames: int = 1
    # 告警冷却：同一摄像头两次告警的最小间隔（秒）
    cooldown_seconds: float = 10.0
    # 是否在证据帧上叠加摄像头名称/时间水印
    save_frame_overlay: bool = False
    # 空间关联置信度门槛（须 >= INI 的 pre-cluster-threshold=0.25）
    helmet_conf_threshold: float = 0.5
    vest_conf_threshold: float = 0.5
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)

    @field_validator("confirm_frames")
    @classmethod
    def _confirm_ge_1(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"confirm_frames 必须 >= 1，当前 {v}")
        return v

    @field_validator("clear_frames")
    @classmethod
    def _clear_ge_0(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"clear_frames 必须 >= 0，当前 {v}")
        return v

    @field_validator("cooldown_seconds")
    @classmethod
    def _cooldown_ge_0(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"cooldown_seconds 必须 >= 0，当前 {v}")
        return v

    @field_validator("helmet_conf_threshold", "vest_conf_threshold")
    @classmethod
    def _threshold_in_0_1(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"置信度阈值必须落在 [0, 1]，当前 {v}")
        return v


class OutputConfig(BaseModel):
    """RTSP 服务端全局配置（单端口 + 每路不同 mount path）。"""
    # 单一 RTSP 服务端口（所有摄像头共用）
    rtsp_port: int = 8554
    # 挂载点前缀，每路 = {mount_prefix}/{camera_id}
    mount_prefix: str = "/cam"
    codec: Literal["h264", "h265"] = "h264"
    bitrate: int = 4_000_000
    idrinterval: int = 30
    # shm socket 目录，每路 socket 路径 = {shm_dir}/vi_cam_{i}
    shm_dir: str = "/tmp"

    @field_validator("rtsp_port")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"rtsp_port 越界 [1, 65535]，当前 {v}")
        return v

    @field_validator("mount_prefix")
    @classmethod
    def _mount_starts_slash(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("/"):
            raise ValueError(f"mount_prefix 必须以 '/' 开头，当前 {v!r}")
        return v

    @field_validator("bitrate", "idrinterval")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"必须为正整数，当前 {v}")
        return v


class AppConfig(BaseModel):
    """服务根配置。"""
    model: ModelConfig
    cameras: list[CameraConfig]
    alert: AlertConfig = Field(default_factory=AlertConfig)
    output: OutputConfig | None = None
    log: LogConfig = Field(default_factory=LogConfig)

    @model_validator(mode="after")
    def _at_least_one_enabled(self) -> "AppConfig":
        if not any(c.enabled for c in self.cameras):
            raise ValueError("cameras 必须包含至少一个 enabled=true 的摄像头")
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError(f"cameras[].id 必须唯一，发现重复: {sorted(ids)}")
        return self

    @property
    def enabled_cameras(self) -> list[CameraConfig]:
        """返回启用中的摄像头（保持原始顺序）。"""
        return [c for c in self.cameras if c.enabled]

    def targets_for(self, cam: CameraConfig) -> list[str]:
        """某路摄像头的触发目标：优先该路覆盖，否则回退全局 alert.targets。"""
        return cam.targets if cam.targets is not None else self.alert.targets

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """从 YAML 文件加载并校验配置。"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"配置文件顶层必须是映射，当前: {type(data).__name__}")
        return cls.model_validate(data)
