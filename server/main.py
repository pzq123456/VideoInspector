#!/usr/bin/env python3
"""
安全帽 / 反光衣检测服务端入口（DeepStream 两阶段：整帧检测 + 二级分类器）

启动方式:
    python -m server.main                  # 使用默认配置 server/config.yaml
    python -m server.main --config my.yaml # 指定配置文件

工作流程:
    1. 加载并校验 YAML 配置（摄像头仅支持 RTSP，可多路；N 路需动态 batch 引擎）
    2. 初始化日志
    3. 创建 Webhook 推送器 + 每路摄像头一个 AlertManager（source_id → manager）
    4. 构建 DeepStream pipeline（两阶段架构）:
           RTSP 源×N → nvstreammux(batch=N) → nvinfer(pgie: person, 整帧, uid=1)
                     → nvinfer(helmet, 整帧, uid=3)
                     → nvinfer(harness_cls, 二级分类器, process-mode=2, uid=5, 裁剪 person 整框)
                     → nvinfer(vest_cls,   二级分类器, process-mode=2, uid=6, 裁剪 person 整框)
                     → nvstreamdemux → 每路:
                           nvdsosd → tee → [ shmsink(→ 单端口 RTSP server) | appsink(证据帧缓存) ]
     5. SafetyProbe 挂在 vest_cls，把两阶段检测元数据翻译成 ObjectMeta 喂给
        对应摄像头的 AlertManager（冷却 + 连续帧确认 + 异步 webhook 推送），
        同时给对象上色（nvdsosd 原生渲染，demux 后每路独立，杜绝跨流污染）；
        触发告警时带上该路缓存的最新已渲染帧，executor 线程 JPEG 编码 → payload。
     6. 单端口 RTSP 输出（GstRtspServer）：所有摄像头共用 rtsp_port，
        通过 /cam/{camera_id} 区分路径。
"""

import argparse
import glob
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
from server.pipeline.rtsp_server import SinglePortRtspServer
from server.utils.logger import setup_logger


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
    for key in ("pgie_config", "helmet_config", "harness_cls_config", "vest_cls_config"):
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
        for key in ("person_conf_threshold", "helmet_conf_threshold", "harness_conf_threshold", "vest_conf_threshold"):
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


def _anchor_ini_config(src: Path, classifier_threshold: float | None = None) -> str:
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
        elif m and classifier_threshold is not None and m.group(1) == "classifier-threshold":
            raw = f"classifier-threshold={classifier_threshold}"
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


def _clean_stale_shm_sockets(socket_path: str, logger) -> None:
    """删除上次进程残留的 shm 控制 socket（base 与 .N 变体）。

    GStreamer shmsink 在 bind() 遇到 EADDRINUSE（上次进程异常退出后残留的
    socket 文件）时会**静默退避**成 socket-path.0/.1/...（见 gst-plugins-bad
    sys/shm/shmpipe.c sp_writer_create: `snprintf(..., "%s.%d", path, i)`），
    而 RTSP 侧 shmsrc 仍连接原始路径 —— 两端路径错位，预览报 503
    "Service Unavailable"。每次启动前清掉 base 及所有 .N 变体，
    保证 shmsink 总是绑定 shmsrc 期望的原始路径。
    """
    for p in glob.glob(socket_path) + glob.glob(f"{socket_path}.*"):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("清理残留 shm socket 失败: {} ({})", p, exc)


# ============================================================================
# 子进程: 构建并运行 DeepStream pipeline
# ============================================================================
def _serve(config: dict):
    # 默认收紧 GStreamer 日志，压掉 pyservicemaker/DeepStream 每次启动刷屏的
    # "Add Element ... / LINKING: ..." 与输入源 "Opening in BLOCKING MODE"
    # 等 INFO 噪音；nvinfer 模型加载日志走 g_print 不受影响。
    # 需要排查管道问题时显式设置 GST_DEBUG=4 即可覆盖。
    os.environ.setdefault("GST_DEBUG", "0")

    log_cfg = config.get("log") or {}
    logger = setup_logger(
        name="safety_server",
        level=log_cfg.get("level", "INFO"),
        # 部署环境用 SAFETY_LOG_FILE 覆盖（容器内映射到宿主挂载卷）
        log_file=os.environ.get("SAFETY_LOG_FILE") or log_cfg.get("file"),
    )
    logger.info("startup safety_detection_server")

    # PPE 告警置信度门槛（harness/vest 用于重写 sgie INI 的 classifier-threshold）
    alert_cfg = config.get("alert", {})
    person_conf_threshold = float(alert_cfg.get("person_conf_threshold", 0.4))
    harness_conf_threshold = float(alert_cfg.get("harness_conf_threshold", 0.5))
    vest_conf_threshold = float(alert_cfg.get("vest_conf_threshold", 0.5))

    model = config["model"]
    pgie_config = _anchor_ini_config(Path(_resolve(_PROJECT_ROOT, model["pgie_config"])))
    helmet_config = _anchor_ini_config(Path(_resolve(_PROJECT_ROOT, model["helmet_config"])))
    harness_cls_config = _anchor_ini_config(
        Path(_resolve(_PROJECT_ROOT, model["harness_cls_config"])),
        classifier_threshold=harness_conf_threshold,
    )
    vest_cls_config = _anchor_ini_config(
        Path(_resolve(_PROJECT_ROOT, model["vest_cls_config"])),
        classifier_threshold=vest_conf_threshold,
    )

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
    target_classes = alert_cfg.get("target_classes", ["no_helmet", "no_harness", "no_vest"])
    # 空间关联置信度门槛（须 >= INI 的 pre-cluster-threshold=0.25）
    helmet_conf_threshold = float(alert_cfg.get("helmet_conf_threshold", 0.5))
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

    # RTSP 输入传输协议（nvurisrcbin 的 select-rtp-protocol）:
    #   4=TCP(推荐), 1=UDP, 2=UDP-MCAST, 7=自动(UDP 优先, 失败回退 TCP)。
    # 摄像头/防火墙不通 UDP 时，默认的 7 会让每路启动都先等 5s UDP 超时才
    # 回退 TCP，并刷屏 "Could not receive any UDP packets ... Retrying using
    # a tcp connection" 告警。本项目摄像头实测必须走 TCP，直接锁定 4。
    rtsp_protocol = int((config.get("source") or {}).get("rtsp_protocol", 4))
    if rtsp_protocol not in (1, 2, 4, 7):
        logger.warning("未知 source.rtsp_protocol={}，回退默认 4 (TCP)", rtsp_protocol)
        rtsp_protocol = 4

    p = Pipeline("safety-detector")
    p.add("nvstreammux", "mux", {
        "batch-size": num,
        "width": 1920,
        "height": 1080,
        "batched-push-timeout": 33000,
        "live-source": 1,  # 仅支持 RTSP
    })
    for i, cam in enumerate(cameras):
        p.add("nvurisrcbin", f"src{i}", {
            "uri": cam["rtsp_url"],
            "select-rtp-protocol": rtsp_protocol,
        })
        p.link((f"src{i}", "mux"), ("", "sink_%u"))  # CRITICAL: 必须用 "sink_%u"

    # 第一阶段整帧检测 + 第二阶段二级分类器（裁剪 person 整框）:
    # 顺序 pgie → helmet → harness_cls → vest_cls
    p.add("nvinfer", "pgie", {"config-file-path": pgie_config, "batch-size": num})
    p.add("nvinfer", "helmet", {"config-file-path": helmet_config, "batch-size": num})
    p.add("nvinfer", "harness_cls", {"config-file-path": harness_cls_config, "batch-size": num})
    p.add("nvinfer", "vest_cls", {"config-file-path": vest_cls_config, "batch-size": num})

    # vest_cls 后立即拆流：之后每路独立渲染/采集/输出，彻底隔离跨流污染
    p.add("nvstreamdemux", "demux")

    # RTSP 输出参数（单端口 + 每路不同 mount path）
    codec = (out or {}).get("codec", "h264")
    bitrate = (out or {}).get("bitrate", 4000000)
    idrinterval = (out or {}).get("idrinterval", 30)
    mount_prefix = (out or {}).get("mount_prefix", "/cam")
    rtsp_port = int((out or {}).get("rtsp_port", 8554))
    enc_factory = "nvv4l2h265enc" if codec == "h265" else "nvv4l2h264enc"
    parser_factory = "h265parse" if codec == "h265" else "h264parse"

    frame_cache = FrameCache()
    rtsp_mounts: dict[str, str] = {}

    for i, cam in enumerate(cameras):
        # 每路独立 nvdsosd：SafetyProbe 上色（metadata 随 demux 保留到本路）
        p.add("nvdsosd", f"osd{i}", {
            "gpu-id": 0,
            "process-mode": 1,  # GPU 模式
            "display-bbox": 1,
            "display-text": 1,
        })
        # 证据帧 tee 分支：appsink 缓存该路已渲染帧（branch 在 frame_cache 模块建好）
        tee = add_evidence_capture(p, frame_cache, source_id=i, gpu_id=0, suffix=str(i))
        p.link((f"demux", f"osd{i}"), ("src_%u", ""))
        p.link(f"osd{i}", tee)

        if out:
            shm_socket = f"/tmp/vi_cam_{i}"
            # 关键修复: 清掉上次进程残留的 shm socket，否则 shmsink 会退避成
            # /tmp/vi_cam_0.N，与 RTSP 侧 shmsrc 的原始路径错位，预览 503。
            _clean_stale_shm_sockets(shm_socket, logger)
            p.add("nvvideoconvert", f"rtsp-conv{i}", {"gpu-id": 0, "compute-hw": 1})
            p.add("capsfilter", f"rtsp-caps{i}", {
                "caps": "video/x-raw(memory:NVMM), format=NV12",
            })
            p.add(enc_factory, f"enc{i}", {
                "bitrate": bitrate,
                "idrinterval": idrinterval,
                "insert-sps-pps": 1,
            })
            p.add(parser_factory, f"parse{i}")
            p.add("shmsink", f"shm{i}", {
                "socket-path": shm_socket,
                "wait-for-connection": False,
                "sync": False,
                "async": 0,
            })
            p.link(tee, f"rtsp-conv{i}", f"rtsp-caps{i}",
                   f"enc{i}", f"parse{i}", f"shm{i}")
            rtsp_mounts[f"{mount_prefix}/{cam['id']}"] = shm_socket

    p.link("mux", "pgie", "helmet", "harness_cls", "vest_cls", "demux")

    p.attach("vest_cls", Probe("safety-probe",
                                SafetyProbe(alert_managers, executor=executor,
                                            frame_cache=frame_cache,
                                            helmet_conf_threshold=helmet_conf_threshold,
                                            person_conf_threshold=person_conf_threshold)))
    p.attach("vest_cls", "measure_fps_probe", name="fps-probe")

    # 单端口 RTSP 输出服务：所有摄像头共用 rtsp_port，路径区分 /cam/{camera_id}
    rtsp_server = None
    if rtsp_mounts:
        rtsp_server = SinglePortRtspServer(rtsp_port, rtsp_mounts, codec=codec)
        rtsp_server.start()
        for mount in rtsp_mounts:
            logger.info("RTSP 输出: rtsp://localhost:{}{}", rtsp_port, mount)

    logger.info("pipeline started cameras={} batch={}", num, num)
    try:
        p.start().wait()
    finally:
        if rtsp_server is not None:
            rtsp_server.stop()
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
