#!/usr/bin/env python3
"""
安全帽 / 反光衣 / 安全带检测服务端入口（DeepStream 两阶段：整帧检测 + 二级分类器）

启动方式:
    python tools/model_build.py --config deploy/config.yaml   # ① 编译模型产物（generated/，增量）
    python -m server.main --config deploy/config.yaml          # ② 启动服务
    （容器内由 SAFETY_CONFIG 指定配置；部署根 = config.yaml 所在目录）

工作流程:
    1. 加载并校验 YAML 配置（model.gies 声明模型拓扑；rules 每条独立状态机）
    2. 初始化日志
    3. 解析告警规则（每条独立状态机）→ Webhook 推送器 + 每路摄像头一个 AlertManager
    4. 构建 DeepStream pipeline（由 model.gies 驱动，nvinfer 间补 queue，对齐官方 test5）:
           RTSP 源×N → nvstreammux(batch=N) → [detector/classifier 链] → nvstreamdemux
           → 每路: nvdsosd → tee → [ shmsink(→ RTSP) | appsink(证据帧) ]
    5. SafetyProbe 挂在最后一个 nvinfer，把两阶段检测元数据翻译成 ObjectMeta 喂给
       对应摄像头的 AlertManager（按 active_rules 只算激活维度；冷却 + 连续帧确认 +
       异步 webhook，alert_type=触发的规则名），同时给对象上色。
    6. 单端口 RTSP 输出（GstRtspServer），/cam/{camera_id} 区分路径。
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
from server.alert.rules import parse_rules
from server.alert.webhook import WebhookAlerter
from server.model_spec import KIND_CLASSIFIER, KIND_DETECTOR, anchor_uid, parse_gies
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
    """校验: model.gies / rules / 摄像头(active_rules) / webhook。"""
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

    if errors:
        raise ValueError(f"配置校验失败 ({path}):\n  " + "\n  ".join(errors))


def _resolve(base: Path, p: str) -> str:
    """把配置里的相对路径解析为绝对路径（相对部署根 = config.yaml 所在目录）。"""
    return p if Path(p).is_absolute() else str(base / p)


# nvinfer 解析 INI 内相对路径时基于进程 CWD，这里在启动时把模型路径显式锚定到
# 部署根（= config.yaml 所在目录：开发环境为 deploy/，镜像内为挂载的配置目录），
# 使同一份可移植 INI 两端通用，不依赖启动目录。
_MODEL_PATH_KEYS = ("onnx-file", "model-engine-file", "labelfile-path", "custom-lib-path")
# 运行期必须存在、缺失即报错（onnx-file 仅用于重建引擎，不在此列）
_REQUIRED_PATH_KEYS = ("model-engine-file", "custom-lib-path", "labelfile-path")

_patched_dir: Path | None = None


def _anchor_ini_config(src: Path, base: Path, classifier_threshold: float | None = None) -> str:
    """把 INI 内相对部署根的模型路径补全为绝对路径，返回 patched 文件路径。"""
    global _patched_dir
    if _patched_dir is None:
        _patched_dir = Path(tempfile.mkdtemp(prefix="safety-configs-"))

    out_lines, resolved = [], {}
    for raw in src.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_-]+)=(.*)$", raw.strip())
        if m and m.group(1) in _MODEL_PATH_KEYS:
            val = m.group(2).strip()
            if val and not Path(val).is_absolute():
                val = str(base / val)
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
            f"{src.name} 引用的模型产物缺失（请先运行 tools/model_build.py --config）:\n  "
            + "\n  ".join(missing)
        )

    out = _patched_dir / src.name
    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return str(out)


def _clean_stale_shm_sockets(socket_path: str, logger) -> None:
    """删除上次进程残留的 shm 控制 socket（base 与 .N 变体）。"""
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
def _serve(config: dict, deploy_root: Path):
    os.environ.setdefault("GST_DEBUG", "0")

    log_cfg = config.get("log") or {}
    log_file = os.environ.get("SAFETY_LOG_FILE") or log_cfg.get("file")
    if log_file and not Path(log_file).is_absolute():
        log_file = str(deploy_root / log_file)
    logger = setup_logger(
        name="safety_server",
        level=log_cfg.get("level", "INFO"),
        log_file=log_file,
    )
    logger.info("startup safety_detection_server")

    model = config["model"]
    gies = parse_gies(model.get("gies"))
    rules = parse_rules(config.get("rules"))
    person_uid = anchor_uid(gies)
    person_conf_threshold = float(model.get("person_conf_threshold", 0.6))

    # 每个 classifier gie 的 classifier-threshold 由引用它的规则 attribute_threshold 重写
    gie_thresholds = {rule.gie: rule.attribute_threshold for rule in rules.values()}

    # 推理链顺序：锚点 detector 最先 → 其余 detector → classifier（§3.5 约束）
    ordered = []
    ordered += [s for s in gies.values() if s.kind == KIND_DETECTOR and s.violation is None]
    ordered += [s for s in gies.values() if s.kind == KIND_DETECTOR and s.violation is not None]
    ordered += [s for s in gies.values() if s.kind == KIND_CLASSIFIER]

    webhook_cfg = config.get("webhook") or {}
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

    cameras = [c for c in config.get("cameras") or [] if c.get("enabled", True)]
    alert_managers = {}
    active_rules_by_source: dict[int, set[str]] = {}
    for i, cam in enumerate(cameras):
        active_names = cam.get("active_rules") or list(rules)
        active_rules_by_source[i] = set(active_names)
        alert_managers[i] = AlertManager(
            camera_id=cam["id"],
            camera_name=cam.get("name", cam["id"]),
            rules=[rules[name] for name in active_names],
            webhook=webhook,
        )
        logger.info("startup camera={} src={} webhook={} active_rules={}",
                    cam["id"], cam.get("rtsp_url"), bool(webhook), active_names)

    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="webhook")

    num = len(cameras)
    out = config.get("output")

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
        "live-source": 1,
    })
    for i, cam in enumerate(cameras):
        p.add("nvurisrcbin", f"src{i}", {
            "uri": cam["rtsp_url"],
            "select-rtp-protocol": rtsp_protocol,
        })
        p.link((f"src{i}", "mux"), ("", "sink_%u"))  # CRITICAL: 必须用 "sink_%u"

    # 推理链（对齐官方 test5：nvinfer 之间补 queue）
    prev = "mux"
    last_infer = None
    for spec in ordered:
        ini = _anchor_ini_config(
            Path(_resolve(deploy_root, spec.config_path())),
            base=deploy_root,
            classifier_threshold=(
                gie_thresholds.get(spec.name) if spec.kind == KIND_CLASSIFIER else None
            ),
        )
        p.add("nvinfer", spec.name, {"config-file-path": ini, "batch-size": num})
        p.link(prev, spec.name)
        last_infer = spec.name
        qname = f"q-{spec.name}"
        p.add("queue", qname, {"max-size-buffers": 4})
        p.link(spec.name, qname)
        prev = qname

    p.add("nvstreamdemux", "demux")
    p.link(prev, "demux")

    # RTSP 输出参数
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
        p.add("nvdsosd", f"osd{i}", {
            "gpu-id": 0,
            "process-mode": 1,
            "display-bbox": 1,
            "display-text": 1,
        })
        tee = add_evidence_capture(p, frame_cache, source_id=i, gpu_id=0, suffix=str(i))
        p.link((f"demux", f"osd{i}"), ("src_%u", ""))
        p.link(f"osd{i}", tee)

        if out:
            shm_socket = f"/tmp/vi_cam_{i}"
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

    p.attach(last_infer, Probe("safety-probe",
                               SafetyProbe(alert_managers, executor=executor,
                                           frame_cache=frame_cache,
                                           gies=gies, rules=rules,
                                           person_uid=person_uid,
                                           person_conf_threshold=person_conf_threshold,
                                           active_rules_by_source=active_rules_by_source)))
    p.attach(last_infer, "measure_fps_probe", name="fps-probe")

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
        default=os.environ.get("SAFETY_CONFIG")
                or str(_PROJECT_ROOT / "deploy" / "config.yaml"),
        help="配置文件路径（默认: $SAFETY_CONFIG 或 <项目根>/deploy/config.yaml）",
    )
    args = parser.parse_args()

    print(f"加载配置: {args.config}")
    config = load_config(args.config)
    deploy_root = Path(args.config).resolve().parent

    proc = Process(target=_serve, args=(config, deploy_root))
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
