#!/usr/bin/env python3
"""
安全帽 / 反光衣检测服务端入口（DeepStream 三模型整帧检测）

启动方式:
    python -m server.main                  # 使用默认配置 server/config.yaml
    python -m server.main --config my.yaml # 指定配置文件

工作流程:
    1. 加载并校验 YAML 配置（摄像头仅支持 RTSP，当前引擎 batch=1 → 1 路）
    2. 初始化日志
    3. 创建 Webhook 推送器 + 每路摄像头一个 AlertManager
    4. 构建 DeepStream pipeline:
           RTSP 源 → nvstreammux → nvinfer(pgie: person) → nvinfer(helmet)
                   → nvinfer(vest) → [nvosdbin → RTSP 输出 | fakesink]
    5. SafetyProbe 挂在 vest，把三模型检测元数据翻译成 ObjectMeta 喂给
       对应摄像头的 AlertManager（冷却 + 连续帧确认 + 异步 webhook 推送）
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process
from pathlib import Path

import yaml
from pyservicemaker import Pipeline, Probe

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中（支持 python -m server.main）
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from server.alert.manager import AlertManager
from server.alert.webhook import WebhookAlerter
from server.pipeline.probe import SafetyProbe
from server.utils.logger import setup_logger

# nvrtspoutsinkbin 的 codec 是枚举属性，pyservicemaker 只接受 int 值
CODEC_MAP = {"h264": 1, "h265": 2, "mpeg4": 3}


# ============================================================================
# 配置加载
# ============================================================================
def load_config(config_path: str) -> dict:
    """加载并校验 YAML 配置文件。"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _validate_config(config, path)
    return config


def _validate_config(config: dict, path: Path):
    """校验: model 三配置 / 摄像头（RTSP 且 <=1 路，batch=1 引擎）。"""
    errors = []

    model = config.get("model") or {}
    for key in ("pgie_config", "helmet_config", "vest_config"):
        if not model.get(key):
            errors.append(f"model.{key} 为必填项")

    cameras = [c for c in config.get("cameras") or [] if c.get("enabled", True)]
    if not cameras:
        errors.append("cameras 必须是非空列表（至少一个 enabled=true）")
    else:
        for cam in cameras:
            if cam.get("type", "rtsp") != "rtsp":
                errors.append(f"DeepStream 仅支持 RTSP 摄像头: {cam.get('id')} (type={cam.get('type')})")
            if not cam.get("rtsp_url"):
                errors.append(f"cameras[].rtsp_url 为必填项 (type=rtsp): {cam.get('id')}")
        if len(cameras) > 1:
            errors.append(
                f"当前三个引擎均为 batch=1，仅支持 1 路 RTSP；检测到 {len(cameras)} 路。"
                f"多路需用 batch=N 重建引擎（见 server/README.md）。"
            )

    if "alert" not in config:
        errors.append("缺少 'alert' 节")

    if errors:
        raise ValueError(f"配置校验失败 ({path}):\n  " + "\n  ".join(errors))


def _resolve(base: Path, p: str) -> str:
    """把配置里的相对路径解析为绝对路径（相对项目根目录）。"""
    return p if Path(p).is_absolute() else str(base / p)


# ============================================================================
# 子进程: 构建并运行 DeepStream pipeline
# ============================================================================
def _serve(config: dict):
    logger = setup_logger(
        name="safety_server",
        level=(config.get("log") or {}).get("level", "INFO"),
        log_file=(config.get("log") or {}).get("file"),
    )
    logger.info("startup safety_detection_server")

    model = config["model"]
    pgie_config = _resolve(_PROJECT_ROOT, model["pgie_config"])
    helmet_config = _resolve(_PROJECT_ROOT, model["helmet_config"])
    vest_config = _resolve(_PROJECT_ROOT, model["vest_config"])

    # Webhook 推送器（全局共享一个）
    webhook_cfg = config.get("alert", {}).get("webhook", {})
    webhook = None
    if webhook_cfg.get("url"):
        webhook = WebhookAlerter(
            url=webhook_cfg["url"],
            timeout=webhook_cfg.get("timeout", 10),
            retries=webhook_cfg.get("retries", 2),
        )
        logger.debug("webhook configured: {}", webhook_cfg["url"])
    else:
        logger.warning("未配置 Webhook URL，告警不会推送")

    # 每路摄像头一个 AlertManager（source_id 即 nvstreammux 的 pad 序）
    alert_cfg = config.get("alert", {})
    target_classes = alert_cfg.get("target_classes", ["no_helmet", "no_vest"])
    cameras = [c for c in config.get("cameras") or [] if c.get("enabled", True)]
    alert_managers = {}
    for i, cam in enumerate(cameras):
        alert_managers[i] = AlertManager(
            camera_id=cam["id"],
            camera_name=cam.get("name", cam["id"]),
            target_classes=target_classes,
            save_frame_overlay=alert_cfg.get("save_frame_overlay", False),
            cooldown_seconds=alert_cfg.get("cooldown_seconds", 30),
            min_detection_count=alert_cfg.get("min_detection_count", 3),
            webhook=webhook,
        )
        logger.info("startup camera={} src={} webhook={} target={}",
                    cam["id"], cam.get("rtsp_url"), bool(webhook), target_classes)

    # 异步 webhook executor（探针只做轻量决策，重活卸载到线程池）
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="webhook")

    num = len(cameras)
    out = config.get("output")

    p = Pipeline("safety-detector")
    p.add("nvstreammux", "mux", {
        "batch-size": num,
        "width": 1920,
        "height": 1080,
        "batched-push-timeout": 33000,
        "live-source": 1,  # 仅支持 RTSP
    })
    for i, cam in enumerate(cameras):
        p.add("nvurisrcbin", f"src{i}", {"uri": cam["rtsp_url"]})
        p.link((f"src{i}", "mux"), ("", "sink_%u"))  # CRITICAL: 必须用 "sink_%u"

    # 三个整帧检测器，顺序 pgie → helmet → vest（无次级 GIE，不会卡死）
    p.add("nvinfer", "pgie", {"config-file-path": pgie_config, "batch-size": num})
    p.add("nvinfer", "helmet", {"config-file-path": helmet_config, "batch-size": num})
    p.add("nvinfer", "vest", {"config-file-path": vest_config})

    if out:
        p.add("nvosdbin", "osd")
        p.add("nvrtspoutsinkbin", "rtspout", {
            "rtsp-port": out["rtsp_port"],
            "rtsp-mount-point": out["mount_point"],
            "codec": CODEC_MAP.get(out.get("codec", "h264"), 1),
            "bitrate": out.get("bitrate", 4000000),
            "idrinterval": out.get("idrinterval", 30),
            "sync": 0,
        })
        p.link("mux", "pgie", "helmet", "vest", "osd", "rtspout")
        print(f">>> RTSP 输出: rtsp://localhost:{out['rtsp_port']}{out['mount_point']}")
    else:
        p.add("fakesink", "sink", {"sync": False})
        p.link("mux", "pgie", "helmet", "vest", "sink")

    p.attach("vest", Probe("safety-probe", SafetyProbe(alert_managers, executor=executor)))
    p.attach("vest", "measure_fps_probe", name="fps-probe")

    logger.info("pipeline started cameras={} batch={}", num, num)
    try:
        p.start().wait()
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


# ============================================================================
# 主入口
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="安全帽/反光衣检测服务端 (DeepStream)")
    parser.add_argument(
        "--config", "-c",
        default=str(_PROJECT_ROOT / "server" / "config.yaml"),
        help="配置文件路径（默认: server/config.yaml）",
    )
    args = parser.parse_args()

    print(f"加载配置: {args.config}")
    config = load_config(args.config)  # 父进程先校验，fail fast

    # wait() 是阻塞调用，包一层子进程让 Ctrl+C 能立即生效
    proc = Process(target=_serve, args=(config,))
    try:
        proc.start()
        proc.join()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating...")
        proc.terminate()
        proc.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
