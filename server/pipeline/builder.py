#!/usr/bin/env python3
"""DeepStream 管线构建器（源 → 推理链 → 分路输出）。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pyservicemaker import Pipeline, Probe

from server.alert.manager import AlertManager
from server.alert.rules import parse_rules
from server.alert.webhook import WebhookAlerter
from server.config import DEFAULT_RECONNECT
from server.model_spec import KIND_CLASSIFIER, KIND_DETECTOR, anchor_uid, parse_gies
from server.pipeline.frame_cache import FrameCache, add_evidence_capture
from server.pipeline.ini_patch import anchor_ini_config
from server.pipeline.probe import SafetyProbe


def _clean_stale_shm_sockets(socket_path: str, logger) -> None:
    """删除上次进程残留的 shm 控制 socket（base 与 .N 变体）。"""
    import glob
    import os

    for p in glob.glob(socket_path) + glob.glob(f"{socket_path}.*"):
        try:
            os.unlink(p)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.warning("清理残留 shm socket 失败: {} ({})", p, exc)


def _reconnect_settings(config: dict) -> dict:
    return {**DEFAULT_RECONNECT, **((config.get("source") or {}).get("reconnect") or {})}


def _supported_nvurisrcbin_props(logger) -> set[str] | None:
    """用 gst-inspect 探测本机 nvurisrcbin 支持的属性名（不同 DS 版本差异大）。

    返回 None 表示探测失败（届时不过滤，按原样下发，让 GStreamer 自行告警）。
    """
    import re
    import subprocess

    try:
        out = subprocess.run(
            ["gst-inspect-1.0", "nvurisrcbin"],
            capture_output=True, text=True, timeout=15,
        ).stdout
        return set(re.findall(r"^ {2}([a-z0-9-]+)\s*:", out, re.M))
    except Exception as exc:
        logger.warning("gst-inspect-1.0 nvurisrcbin 探测失败，跳过属性兼容性过滤: {}", exc)
        return None


class PipelineBuilder:
    """由 config 一次性构建整条 DeepStream 管线及其运行期组件。"""

    def __init__(self, config: dict, deploy_root: Path):
        self.config = config
        self.deploy_root = deploy_root

        model = config["model"]
        self.gies = parse_gies(model.get("gies"))
        self.rules = parse_rules(config.get("rules"))
        self.person_uid = anchor_uid(self.gies)
        self.person_conf_threshold = float(model.get("person_conf_threshold", 0.6))
        # 每个 classifier gie 的 classifier-threshold 由引用它的规则 attribute_threshold 重写
        self.gie_thresholds = {rule.gie: rule.attribute_threshold for rule in self.rules.values()}
        self.cameras = [c for c in config.get("cameras") or [] if c.get("enabled", True)]

    # ------------------------------------------------------------------ 构建
    def build(self, logger) -> dict:
        """构建并返回 {pipeline, last_infer, frame_cache, executor, rtsp_server}。"""
        cfg = self.config
        num = len(self.cameras)
        out = cfg.get("output")

        p = Pipeline("safety-detector")
        self._add_sources(p, logger)
        last_infer = self._add_inference_chain(p)
        frame_cache = FrameCache()
        rtsp_mounts = self._add_output_branches(p, out, frame_cache, logger)

        executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="webhook")
        p.attach(last_infer, Probe("safety-probe", SafetyProbe(
            self._build_alert_managers(cfg, executor, logger),
            executor=executor,
            frame_cache=frame_cache,
            gies=self.gies, rules=self.rules,
            person_uid=self.person_uid,
            person_conf_threshold=self.person_conf_threshold,
            active_rules_by_source=self._active_rules_by_source(),
        )))
        p.attach(last_infer, "measure_fps_probe", name="fps-probe")

        rtsp_server = None
        if rtsp_mounts:
            from server.pipeline.rtsp_server import SinglePortRtspServer

            codec = (out or {}).get("codec", "h264")
            rtsp_server = SinglePortRtspServer(
                int((out or {}).get("rtsp_port", 8554)), rtsp_mounts, codec=codec)
            rtsp_server.start()
            for mount in rtsp_mounts:
                logger.info("RTSP 输出: rtsp://localhost:{}{}",
                            (out or {}).get("rtsp_port", 8554), mount)

        return {
            "pipeline": p,
            "frame_cache": frame_cache,
            "executor": executor,
            "rtsp_server": rtsp_server,
        }

    # ------------------------------------------------------------------ 源
    def _add_sources(self, p: Pipeline, logger) -> None:
        rc = _reconnect_settings(self.config)
        src_cfg = self.config.get("source") or {}
        rtsp_protocol = int(src_cfg.get("rtsp_protocol", 4))
        if rtsp_protocol not in (1, 2, 4, 7):
            logger.warning("未知 source.rtsp_protocol={}，回退默认 4 (TCP)", rtsp_protocol)
            rtsp_protocol = 4

        # DS 版本属性差异探测：如 rtsp-reconnect-timeout 在 DS9.0 已被移除
        supported = _supported_nvurisrcbin_props(logger)

        def src_props(props: dict) -> dict:
            if supported is None:
                return props
            dropped = [k for k in props if k not in supported]
            for k in dropped:
                logger.warning("nvurisrcbin 不支持属性 {} (DS 版本差异)，已跳过", k)
            return {k: v for k, v in props.items() if k not in dropped}

        p.add("nvstreammux", "mux", {
            "batch-size": len(self.cameras),
            "width": 1920,
            "height": 1080,
            "batched-push-timeout": 33000,
            "live-source": 1,
        })
        for i, cam in enumerate(self.cameras):
            # 内置重连参数（有限次数；耗尽后由看门狗触发监督循环整体重建）。
            # init-rtsp-reconnect-interval 是 DS9.x 的已知回归坑: 发生错误后活跃
            # 重连间隔会切换到该值，默认 0 = 禁用重连监控，必须显式设置
            # （NVIDIA 论坛 Regression #377715 官方 workaround）。
            p.add("nvurisrcbin", f"src{i}", src_props({
                "uri": cam["rtsp_url"],
                "select-rtp-protocol": rtsp_protocol,
                "rtsp-reconnect-attempts": int(rc["attempts"]),
                "rtsp-reconnect-timeout": int(rc["timeout"]),
                "rtsp-reconnect-interval": int(rc["interval"]),
                "init-rtsp-reconnect-interval": int(max(1, rc["interval"] // 2)),
                "latency": int(rc["latency"]),
            }))
            p.link((f"src{i}", "mux"), ("", "sink_%u"))

    # ------------------------------------------------------------------ 推理链
    def _add_inference_chain(self, p: Pipeline) -> str:
        # 顺序：锚点 detector 最先 → 其余整帧 detector → 二级模型（§3.5 约束）
        ordered = [s for s in self.gies.values() if s.kind == KIND_DETECTOR and not s.is_secondary]
        ordered += [s for s in self.gies.values() if s.is_secondary]

        prev, last_infer = "mux", None
        for spec in ordered:
            ini = anchor_ini_config(
                Path(_resolve(self.deploy_root, spec.config_path())),
                base=self.deploy_root,
                classifier_threshold=(
                    self.gie_thresholds.get(spec.name) if spec.kind == KIND_CLASSIFIER else None
                ),
            )
            p.add("nvinfer", spec.name, {"config-file-path": ini, "batch-size": len(self.cameras)})
            p.link(prev, spec.name)
            last_infer = spec.name
            qname = f"q-{spec.name}"
            p.add("queue", qname, {"max-size-buffers": 4})
            p.link(spec.name, qname)
            prev = qname

        p.add("nvstreamdemux", "demux")
        p.link(prev, "demux")
        return last_infer

    # ------------------------------------------------------------------ 输出分支
    def _add_output_branches(self, p: Pipeline, out, frame_cache: FrameCache, logger) -> dict[str, str]:
        codec = (out or {}).get("codec", "h264")
        enc_factory = "nvv4l2h265enc" if codec == "h265" else "nvv4l2h264enc"
        parser_factory = "h265parse" if codec == "h265" else "h264parse"

        rtsp_mounts: dict[str, str] = {}
        for i in range(len(self.cameras)):
            p.add("nvdsosd", f"osd{i}", {
                "gpu-id": 0,
                "process-mode": 1,
                "display-bbox": 1,
                "display-text": 1,
            })
            tee = add_evidence_capture(p, frame_cache, source_id=i, gpu_id=0, suffix=str(i))
            p.link(("demux", f"osd{i}"), ("src_%u", ""))
            p.link(f"osd{i}", tee)

            if not out:
                continue
            shm_socket = f"/tmp/vi_cam_{i}"
            _clean_stale_shm_sockets(shm_socket, logger)
            p.add("nvvideoconvert", f"rtsp-conv{i}", {"gpu-id": 0, "compute-hw": 1})
            p.add("capsfilter", f"rtsp-caps{i}", {
                "caps": "video/x-raw(memory:NVMM), format=NV12",
            })
            p.add(enc_factory, f"enc{i}", {
                "bitrate": out.get("bitrate", 4000000),
                "idrinterval": out.get("idrinterval", 30),
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
            rtsp_mounts[f"{out.get('mount_prefix', '/cam')}/{self.cameras[i]['id']}"] = shm_socket
        return rtsp_mounts

    # ------------------------------------------------------------------ 告警
    def _active_rules_by_source(self) -> dict[int, set[str]]:
        return {
            i: set(cam.get("active_rules") or list(self.rules))
            for i, cam in enumerate(self.cameras)
        }

    def _build_alert_managers(self, config: dict, executor, logger) -> dict[int, AlertManager]:
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

        active_by_src = self._active_rules_by_source()
        managers: dict[int, AlertManager] = {}
        for i, cam in enumerate(self.cameras):
            names = sorted(active_by_src[i])
            managers[i] = AlertManager(
                camera_id=cam["id"],
                camera_name=cam.get("name", cam["id"]),
                rules=[self.rules[name] for name in names],
                webhook=webhook,
            )
            logger.info("startup camera={} src={} webhook={} active_rules={}",
                        cam["id"], cam.get("rtsp_url"), bool(webhook), names)
        return managers


def _resolve(base: Path, path: str) -> str:
    from server.config import resolve_path
    return resolve_path(base, path)
