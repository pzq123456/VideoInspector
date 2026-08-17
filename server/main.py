#!/usr/bin/env python3
"""
安全帽 / 反光衣检测服务端入口（DeepStream 三模型整帧检测）

启动方式:
    python -m server.main                  # 使用默认配置 server/config.yaml
    python -m server.main --config my.yaml # 指定配置文件

工作流程:
    1. 加载并校验 YAML 配置（摄像头仅支持 RTSP，可多路；N 路需动态 batch 引擎）
    2. 初始化日志
    3. 创建 Webhook 推送器 + 每路摄像头一个 AlertManager（source_id → manager）
    4. 构建 DeepStream pipeline:
           RTSP 源×N → nvstreammux(batch=N) → nvinfer(pgie: person) → nvinfer(helmet)
                     → nvinfer(vest) → nvdsosd → tee → [nvstreamdemux → RTSP 输出×N | fakesink]
                                                   └── queue → nvvideoconvert → appsink(证据帧缓存, 按 source_id)
     5. SafetyProbe 挂在 vest，把三模型检测元数据翻译成 ObjectMeta 喂给
        对应摄像头的 AlertManager（冷却 + 连续帧确认 + 异步 webhook 推送），
        同时给对象上色（nvdsosd 原生渲染）；触发告警时带上缓存的最新已渲染帧，
        executor 线程 JPEG 编码 → payload。
"""

import argparse
import os
import re
import sys
import tempfile
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
from server.pipeline.frame_cache import FrameCache, add_evidence_capture
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

    if "alert" not in config:
        errors.append("缺少 'alert' 节")
    else:
        for key in ("helmet_conf_threshold", "vest_conf_threshold"):
            val = config["alert"].get(key)
            if val is not None and not (isinstance(val, (int, float)) and 0.0 <= val <= 1.0):
                errors.append(f"alert.{key} 必须是 0~1 的置信度阈值，当前: {val!r}")

    if errors:
        raise ValueError(f"配置校验失败 ({path}):\n  " + "\n  ".join(errors))


def _resolve(base: Path, p: str) -> str:
    """把配置里的相对路径解析为绝对路径（相对项目根目录）。"""
    return p if Path(p).is_absolute() else str(base / p)


# nvinfer 解析 INI 内相对路径时基于进程 CWD，这里在启动时把模型路径显式
# 锚定到项目根（_PROJECT_ROOT，开发容器=/workspaces/VideoInspector，镜像=/app），
# 使同一份可移植 INI 两端通用，不依赖启动目录。
_MODEL_PATH_KEYS = ("onnx-file", "model-engine-file", "labelfile-path", "custom-lib-path")
# 运行期必须存在、缺失即报错（onnx-file 仅用于重建引擎，不在此列）
_REQUIRED_PATH_KEYS = ("model-engine-file", "custom-lib-path", "labelfile-path")

_patched_dir: Path | None = None


def _anchor_ini_config(src: Path) -> str:
    """把 INI 内相对项目根的模型路径补全为绝对路径，返回 patched 文件路径。

    fail fast: 锚定后校验运行期必需的引擎/parser/标签文件存在，
    缺失时给出清晰报错（而非 nvinfer 的模糊告警）。
    """
    global _patched_dir
    if _patched_dir is None:
        _patched_dir = Path(tempfile.mkdtemp(prefix="safety-configs-"))

    out_lines, resolved = [], {}
    for raw in src.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_-]+)=(.*)$", raw.strip())
        if m and m.group(1) in _MODEL_PATH_KEYS:
            val = m.group(2).strip()
            if val and not Path(val).is_absolute():
                val = str(_PROJECT_ROOT / val)
                raw = f"{m.group(1)}={val}"
            resolved[m.group(1)] = val
        out_lines.append(raw)

    missing = [
        f"{k}={v}" for k, v in resolved.items()
        if k in _REQUIRED_PATH_KEYS and v and not Path(v).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{src.name} 引用的模型产物缺失（请确认根目录 models/ 已拷全）:\n  "
            + "\n  ".join(missing)
        )

    out = _patched_dir / src.name
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return str(out)


# ============================================================================
# 子进程: 构建并运行 DeepStream pipeline
# ============================================================================
def _serve(config: dict):
    log_cfg = config.get("log") or {}
    logger = setup_logger(
        name="safety_server",
        level=log_cfg.get("level", "INFO"),
        # 部署环境用 SAFETY_LOG_FILE 覆盖（容器内映射到宿主挂载卷）
        log_file=os.environ.get("SAFETY_LOG_FILE") or log_cfg.get("file"),
    )
    logger.info("startup safety_detection_server")

    model = config["model"]
    pgie_config = _anchor_ini_config(Path(_resolve(_PROJECT_ROOT, model["pgie_config"])))
    helmet_config = _anchor_ini_config(Path(_resolve(_PROJECT_ROOT, model["helmet_config"])))
    vest_config = _anchor_ini_config(Path(_resolve(_PROJECT_ROOT, model["vest_config"])))

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
    # 空间关联置信度门槛（须 >= INI 的 pre-cluster-threshold=0.25）
    helmet_conf_threshold = float(alert_cfg.get("helmet_conf_threshold", 0.5))
    vest_conf_threshold = float(alert_cfg.get("vest_conf_threshold", 0.5))
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
            alert_type=alert_cfg.get("alert_type", "ppe"),
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
    p.add("nvinfer", "vest", {"config-file-path": vest_config, "batch-size": num})

    # 证据帧采集：nvdsosd 后 tee 分流，缓存每路最新**已渲染**帧（分支在 frame_cache 模块建好）
    frame_cache = FrameCache()
    tee_name = add_evidence_capture(p, frame_cache, gpu_id=0)

    # 单路 nvdsosd 渲染器：始终位于 tee 之前，实时预览与证据帧共享同一渲染源，
    # 保证证据帧与操作者所见一致（违规红 / 合规绿 / 未知蓝由 SafetyProbe 上色）。
    p.add("nvdsosd", "osd", {
        "gpu-id": 0,
        "process-mode": 1,  # GPU 模式
        "display-bbox": 1,
        "display-text": 1,
    })

    if out:
        # 多路输出：batch 帧 → nvstreamdemux 拆成每路独立帧 → 每路一个 RTSP 输出。
        # 每个 nvrtspoutsinkbin 自带一个 RTSP server，必须独占端口：
        #   camera i → port = rtsp_port + i, mount = {mount_prefix}/{camera_id}
        p.add("nvstreamdemux", "demux")
        base_port = int(out.get("rtsp_port", 18003))
        mount_prefix = out.get("mount_prefix", "/cam")
        codec_val = CODEC_MAP.get(out.get("codec", "h264"), 1)
        bitrate = out.get("bitrate", 4000000)
        idrinterval = out.get("idrinterval", 30)
        for i, cam in enumerate(cameras):
            p.add("nvrtspoutsinkbin", f"rtspout{i}", {
                "rtsp-port": base_port + i,
                "rtsp-mount-point": f"{mount_prefix}/{cam['id']}",
                "codec": codec_val,
                "bitrate": bitrate,
                "idrinterval": idrinterval,
                "sync": 0,
            })
            # 请求 nvstreamdemux 的 src_%u request pad（请求顺序 = 源序号），连到该路输出
            p.link(("demux", f"rtspout{i}"), ("src_%u", ""))
            print(f">>> 相机 {cam['id']} RTSP 输出: "
                  f"rtsp://localhost:{base_port + i}{mount_prefix}/{cam['id']}")
        p.link("mux", "pgie", "helmet", "vest", "osd", tee_name)
        p.link(tee_name, "demux")
    else:
        p.add("fakesink", "sink", {"sync": False, "async": 0})
        p.link("mux", "pgie", "helmet", "vest", "osd", tee_name)
        p.link(tee_name, "sink")

    p.attach("vest", Probe("safety-probe",
                           SafetyProbe(alert_managers, executor=executor,
                                       frame_cache=frame_cache,
                                       helmet_conf_threshold=helmet_conf_threshold,
                                       vest_conf_threshold=vest_conf_threshold)))
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
