#!/usr/bin/env python3
"""
two_stage_demo.py — 三模型整帧检测流水线 demo（pyservicemaker 版）

流水线:
    nvurisrcbin(源) → nvstreammux → nvinfer(pgie: yolo26n 人体检测)
        → nvinfer(helmet: 安全帽整帧检测) → nvinfer(vest: 反光衣整帧检测)
        → nvosdbin → 输出

阶段一 PGIE:    YOLO26n（COCO），自定义 parser 只输出 person(class_id=0)，uid=1。
阶段二 helmet:  head/helmet 整帧检测器（uid=3），对同一帧独立检测。
阶段三 vest:    vest/no_vest 整帧检测器（uid=4），对同一帧独立检测。

三个模型都是 process-mode=1（整帧）检测器，按顺序排即可（无次级 GIE，
规避「第二个整帧 nvinfer 放在次级 GIE 之后会卡死」的问题）。
探针做空间关联: 检测框中心落在 person 框内 → 归属该人。

渲染约定:
    no_vest 或 未戴帽(no_helmet) 的人 → 红色框
    vest 且 戴帽(helmet) 的人        → 绿色框
    任一项未关联上                   → 蓝色框
    head/helmet 框                  → 红=head(未戴), 绿=helmet(已戴)
    no_vest/vest 框                 → 红=no_vest, 绿=vest

每帧输出一条结构化 JSON（业务对接数据契约）:
    JSON: {"stream":..,"frame":..,"persons":[{bbox,conf,helmet,vest,violation}],"counts":{...}}
    同时追加写 output/structured.jsonl。

用法:
    ./run.sh                                  # RTSP in → RTSP out（读 configs/rtsp_in.yaml）
    python3 two_stage_demo.py --file <视频>    # 本地文件调试模式，存 output/frame_*.jpg
    python3 two_stage_demo.py <其他.yaml>      # 指定 RTSP 配置文件
"""

import argparse
import json
import os
import sys
from multiprocessing import Process

import yaml
from pyservicemaker import Pipeline, Probe, BatchMetadataOperator, osd

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML = os.path.join(DEMO_DIR, "configs", "rtsp_in.yaml")
OUTPUT_DIR = os.path.join(DEMO_DIR, "output")
MAX_SAVED_FRAMES = 30

# 各整帧检测器的 gie-unique-id（探针按此区分对象来源，而非 class_id）
PERSON_UID = 1      # pgie (yolo26n) 的 gie-unique-id
HELMET_UID = 3      # helmet 整帧检测器的 gie-unique-id
VEST_UID = 4        # vest 整帧检测器的 gie-unique-id

# 类别 id 对应各模型 labels.txt 的行序
HELMET_CLASS_ID = 1  # helmet: 0=head(未戴), 1=helmet(已戴)
HEAD_CLASS_ID = 0
VEST_CLASS_ID = 0    # vest: 0=vest(已穿), 1=no_vest(未穿)
NO_VEST_CLASS_ID = 1

# 检测框关联 person 时要求的最低置信度（低于则视为噪声，不参与状态判定）
HELMET_CONF_THRESHOLD = 0.5
VEST_CONF_THRESHOLD = 0.5

# nvrtspoutsinkbin 的 codec 是枚举属性，pyservicemaker 只接受 int 值
CODEC_MAP = {"h264": 1, "h265": 2, "mpeg4": 3}

RED = osd.Color(1.0, 0.0, 0.0, 1.0)   # 不达标（no_vest 或 no_helmet）
GREEN = osd.Color(0.0, 1.0, 0.0, 1.0)  # 达标（vest 且 helmet）
BLUE = osd.Color(0.0, 0.0, 1.0, 1.0)   # 有维度未知
WHITE = osd.Color(1.0, 1.0, 1.0, 1.0)
BLACK = osd.Color(0.0, 0.0, 0.0, 1.0)


class SafetyMarker(BatchMetadataOperator):
    """逐帧统计人员/反光衣/安全帽，上色，并输出结构化 JSON。

    三个模型的检测结果都挂在帧级（uid 区分来源）:
      - person 对象（uid=1）: yolo26n 检出的人
      - head/helmet 框（uid=3）: 安全帽整帧检测器
      - vest/no_vest 框（uid=4）: 反光衣整帧检测器
    状态 = 检测框中心点落在 person 框内（空间关联）;
    helmet 优先于 head；no_vest 优先于 vest（安全告警倾向，宁可多报）。
    """

    def __init__(self):
        super().__init__()
        self._json_out = None

    def _json_writer(self):
        """结构化 JSON 追加写 output/structured.jsonl（懒打开，保证 output/ 存在）。"""
        if self._json_out is None:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            self._json_out = open(
                os.path.join(OUTPUT_DIR, "structured.jsonl"), "a", encoding="utf-8"
            )
        return self._json_out

    def handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            persons = vest = no_vest = helmet = no_helmet = unknown = 0
            records = []

            # 第一趟: 收集 helmet 与 vest 检测框的原始数值（class_id/conf/中心点），
            # 并给这些框上色。不能把元数据对象存进 list 留到后面再用（迭代器跨
            # pass 后包装器失效会段错误），只保留普通 Python 数值。
            helmet_boxes = []  # [(class_id, conf, cx, cy)]
            vest_boxes = []    # [(class_id, conf, cx, cy)]
            for o in frame_meta.object_items:  # 一次性迭代器
                if o.unique_component_id == HELMET_UID:
                    o.rect_params.border_color = (
                        GREEN if o.class_id == HELMET_CLASS_ID else RED
                    )
                    helmet_boxes.append(self._box_vals(o))
                elif o.unique_component_id == VEST_UID:
                    o.rect_params.border_color = (
                        GREEN if o.class_id == VEST_CLASS_ID else RED
                    )
                    vest_boxes.append(self._box_vals(o))

            for obj in frame_meta.object_items:  # 重新取迭代器；object_items 不能用 len()
                if obj.unique_component_id != PERSON_UID:
                    continue
                persons += 1
                h = self._helmet_status(
                    self._matched_classes(obj, helmet_boxes, HELMET_CONF_THRESHOLD)
                )
                v = self._vest_status(
                    self._matched_classes(obj, vest_boxes, VEST_CONF_THRESHOLD)
                )

                if v == "vest":
                    vest += 1
                elif v == "no_vest":
                    no_vest += 1
                if h == "helmet":
                    helmet += 1
                elif h == "no_helmet":
                    no_helmet += 1
                if v is None or h is None:
                    unknown += 1

                # 任一不达标=红；双达标=绿；有维度未知=蓝
                violation = v == "no_vest" or h == "no_helmet"
                if violation:
                    obj.rect_params.border_color = RED
                elif v == "vest" and h == "helmet":
                    obj.rect_params.border_color = GREEN
                else:
                    obj.rect_params.border_color = BLUE

                records.append({
                    "bbox": [
                        round(obj.rect_params.left),
                        round(obj.rect_params.top),
                        round(obj.rect_params.left + obj.rect_params.width),
                        round(obj.rect_params.top + obj.rect_params.height),
                    ],
                    "conf": round(obj.confidence, 3),
                    "helmet": h if h is not None else "unknown",
                    "vest": v if v is not None else "unknown",
                    "violation": violation,
                })

            counts = {
                "persons": persons,
                "vest": vest,
                "no_vest": no_vest,
                "helmet": helmet,
                "no_helmet": no_helmet,
                "unknown": unknown,
            }
            rec = {
                "stream": frame_meta.source_id,
                "frame": frame_meta.frame_number,
                "time_ms": self._pts_ms(frame_meta),
                "persons": records,
                "counts": counts,
            }
            line = json.dumps(rec, ensure_ascii=False)
            print("JSON: " + line)
            self._json_writer().write(line + "\n")
            self._json_writer().flush()

            detail = (
                f"persons={persons} vest={vest} no_vest={no_vest} "
                f"helmet={helmet} no_helmet={no_helmet} unknown={unknown}"
            )
            print(
                f"src={frame_meta.source_id} Frame {frame_meta.frame_number}: "
                f"{detail}".rstrip()
            )

            display_meta = batch_meta.acquire_display_meta()
            text = osd.Text()
            text.display_text = detail.encode("ascii")
            text.x_offset = 10
            text.y_offset = 12
            text.font.name = osd.FontFamily.Serif
            text.font.size = 12
            text.font.color = WHITE
            text.set_bg_color = True
            text.bg_color = BLACK
            display_meta.add_text(text)
            frame_meta.append(display_meta)

    @staticmethod
    def _box_vals(o):
        """提取检测框的普通数值（class_id/confidence/中心点），避免持有元数据包装器。"""
        return (
            o.class_id,
            o.confidence,
            o.rect_params.left + o.rect_params.width / 2,
            o.rect_params.top + o.rect_params.height / 2,
        )

    @staticmethod
    def _pts_ms(frame_meta):
        """帧时间戳（ns → ms）；接口未暴露时返回 None。"""
        pts = getattr(frame_meta, "buffer_pts", None)
        if pts is None:
            pts = getattr(frame_meta, "ntp_timestamp", None)
        return round(pts / 1e6) if pts else None

    @staticmethod
    def _matched_classes(obj, boxes, conf_threshold):
        """返回中心点落在 obj 框内、且置信度达标的 box 类别集合（空间关联核心）。"""
        px1 = obj.rect_params.left
        py1 = obj.rect_params.top
        px2 = px1 + obj.rect_params.width
        py2 = py1 + obj.rect_params.height
        matched = set()
        for cls_id, conf, cx, cy in boxes:
            if conf < conf_threshold:
                continue
            if px1 <= cx <= px2 and py1 <= cy <= py2:
                matched.add(cls_id)
        return matched

    @classmethod
    def _helmet_status(cls, matched):
        """戴帽状态；helmet 优先（同时命中 head 与 helmet 时按已戴帽处理）。"""
        if HELMET_CLASS_ID in matched:
            return "helmet"
        if HEAD_CLASS_ID in matched:
            return "no_helmet"
        return None

    @classmethod
    def _vest_status(cls, matched):
        """反光衣状态；no_vest 优先（安全告警倾向，宁可多报）。"""
        if NO_VEST_CLASS_ID in matched:
            return "no_vest"
        if VEST_CLASS_ID in matched:
            return "vest"
        return None


def run_rtsp(cfg: dict) -> None:
    """RTSP 输入 → 三级推理 → RTSP 输出（nvrtspoutsinkbin 内嵌 RTSP 服务器）。"""
    sources = cfg["sources"]
    out = cfg["output"]
    num = len(sources)

    model = cfg.get("model", {})
    pgie_config = os.path.join(
        DEMO_DIR, "configs", model.get("pgie_config", "pgie_config_yolo26n.txt")
    )
    helmet_config = os.path.join(
        DEMO_DIR, "configs", model.get("helmet_config", "pgie_config_helmet.txt")
    )
    vest_config = os.path.join(
        DEMO_DIR, "configs", model.get("vest_config", "pgie_config_vest.txt")
    )

    if num > 1:
        print(
            f"警告: 当前 yolo26n/helmet/vest 引擎是 batch=1，仅支持 1 路输入；"
            f"检测到 {num} 路。多路需用 batch={num} 重新构建引擎。"
        )

    live = any(u["uri"].startswith(("rtsp://", "rtsps://")) for u in sources)

    p = Pipeline("three-stage-rtsp")
    p.add("nvstreammux", "mux", {
        "batch-size": num,
        "width": 1920,
        "height": 1080,
        "batched-push-timeout": 33000,
        "live-source": 1 if live else 0,
    })

    for i, src in enumerate(sources):
        p.add("nvurisrcbin", f"src{i}", {"uri": src["uri"]})
        p.link((f"src{i}", "mux"), ("", "sink_%u"))  # CRITICAL: 必须用 "sink_%u"

    p.add("nvinfer", "pgie", {"config-file-path": pgie_config, "batch-size": num})
    p.add("nvinfer", "helmet", {"config-file-path": helmet_config, "batch-size": num})
    p.add("nvinfer", "vest", {"config-file-path": vest_config})
    p.add("nvosdbin", "osd")
    p.add("nvrtspoutsinkbin", "rtspout", {
        "rtsp-port": out["rtsp_port"],
        "rtsp-mount-point": out["mount_point"],
        "codec": CODEC_MAP.get(out.get("codec", "h264"), 1),
        "bitrate": out.get("bitrate", 4000000),
        "idrinterval": out.get("idrinterval", 30),
        "sync": 0,
    })

    # 三个都是整帧检测器，顺序 pgie → helmet → vest（无次级 GIE，不会卡死）。
    p.link("mux", "pgie", "helmet", "vest", "osd", "rtspout")
    p.attach("vest", Probe("safety-marker", SafetyMarker()))
    p.attach("vest", "measure_fps_probe", name="fps-probe")

    print(f">>> RTSP 输出: rtsp://localhost:{out['rtsp_port']}{out['mount_point']}")
    print(f">>> 结构化 JSON → {os.path.join(OUTPUT_DIR, 'structured.jsonl')}")
    p.start().wait()


def run_file(filepath: str) -> None:
    """本地文件调试模式：三级推理后存 JPEG 帧，便于 headless 验证红/绿/蓝框。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    uri = (
        filepath
        if filepath.startswith(("rtsp://", "http://", "file://"))
        else f"file://{os.path.abspath(filepath)}"
    )
    pgie_config = os.path.join(DEMO_DIR, "configs", "pgie_config_yolo26n.txt")
    helmet_config = os.path.join(DEMO_DIR, "configs", "pgie_config_helmet.txt")
    vest_config = os.path.join(DEMO_DIR, "configs", "pgie_config_vest.txt")

    p = Pipeline("three-stage-file")
    p.add("nvurisrcbin", "src", {"uri": uri})
    p.add("nvstreammux", "mux", {
        "batch-size": 1,
        "width": 1920,
        "height": 1080,
        "batched-push-timeout": 33000,
    })
    p.add("nvinfer", "pgie", {"config-file-path": pgie_config})
    p.add("nvinfer", "helmet", {"config-file-path": helmet_config})
    p.add("nvinfer", "vest", {"config-file-path": vest_config})
    p.add("nvosdbin", "osd")
    p.add("nvvideoconvert", "convert")
    p.add("capsfilter", "caps", {"caps": "video/x-raw,format=RGB"})
    p.add("jpegenc", "enc")
    p.add("multifilesink", "sink", {
        "location": os.path.join(OUTPUT_DIR, "frame_%04d.jpg"),
        "max-files": MAX_SAVED_FRAMES,
    })
    p.link(("src", "mux"), ("", "sink_%u"))
    p.link("mux", "pgie", "helmet", "vest", "osd", "convert", "caps", "enc", "sink")
    p.attach("vest", Probe("safety-marker", SafetyMarker()))
    p.attach("vest", "measure_fps_probe", name="fps-probe")
    p.start().wait()


def main(argv):
    parser = argparse.ArgumentParser(
        description="三模型整帧检测流水线: 人体检测(yolo26n) → 安全帽(head/helmet) → 反光衣(vest/no_vest)"
    )
    parser.add_argument(
        "--file", metavar="VIDEO",
        help="本地文件调试模式（三级推理后存 output/frame_*.jpg），如 sample_720p.h264",
    )
    parser.add_argument(
        "yaml", nargs="?", default=DEFAULT_YAML,
        help="RTSP 模式配置文件（默认 configs/rtsp_in.yaml）",
    )
    args = parser.parse_args(argv[1:])

    if args.file:
        proc = Process(target=run_file, args=(args.file,))
    else:
        with open(args.yaml) as f:
            cfg = yaml.safe_load(f)
        proc = Process(target=run_rtsp, args=(cfg,))

    # wait() 是阻塞调用，包一层子进程让 Ctrl+C 能立即生效
    try:
        proc.start()
        proc.join()
    except KeyboardInterrupt:
        print("\nInterrupted. Terminating...")
        proc.terminate()
        proc.join()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
