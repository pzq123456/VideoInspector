#!/usr/bin/env python3
"""
tools/eval_cls_deepstream.py — 分类模型评测 Stage2: DeepStream 真实管线逐对象概率抓取。

用法:
    export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
    export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
    python tools/eval_cls_deepstream.py --sgie-config <sgie_cfg.txt> --out <out.json> [--frames 150]

要点:
  - sgie 配置须带 output-tensor-meta=1（覆盖 classifier_meta 输出, 二者互斥）,
    由此绕开「classifier_meta 只挂 class0 / 拿不到概率」的 DS quirk（notes §8）。
  - 抓取锚点 GIE（默认 uid=1 person pgie）每个对象的原始 [class0, class1] softmax。
  - 现役基线可直接用 deploy/generated/vest_cls/vest/<版本>/sgie_config.txt 追加 output-tensor-meta=1。
  - 相对路径锚定: 配置内相对路径按 <项目根>/deploy 解析（部署根 = config.yaml 所在目录）。
  - 对象按 (frame, bbox) 记录, 多次运行可用 eval_cls_report.py 精确配对对比。
"""
import argparse
import json
import os
import re
import tempfile
from multiprocessing import Process
from pathlib import Path

import numpy as np
import torch
from pyservicemaker import Pipeline, Probe, BatchMetadataOperator

ROOT = Path(__file__).resolve().parent.parent
KEY = ("onnx-file", "model-engine-file", "labelfile-path", "custom-lib-path")

ANCHOR_REL = "deploy"
CFG = OUT = FRAMES = None
GIE_ID = 1


def anchor(path: str) -> str:
    """把 nvinfer INI 中的相对模型路径锚定为绝对路径（基于 deploy/），写临时文件。"""
    out = []
    for raw in Path(path).read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_-]+)=(.*)$", raw.strip())
        if m and m.group(1) in KEY:
            v = m.group(2).strip()
            if v and not os.path.isabs(v):
                raw = f"{m.group(1)}={Path(ROOT, 'deploy', v)}"
        out.append(raw)
    f = tempfile.mktemp(suffix=".txt")
    Path(f).write_text("\n".join(out) + "\n")
    return f


class Cap(BatchMetadataOperator):
    """逐帧抓取锚点对象的原始分类概率（dlpack 读回 CUDA tensor）。"""

    def __init__(self, frames: int):
        super().__init__()
        self.frames = frames
        self.done = 0
        self.recs = []

    def handle_metadata(self, bm):
        for fm in bm.frame_items:
            if self.done >= self.frames:
                json.dump(self.recs, open(OUT, "w"))
                print(f"RESULT n_objects={len(self.recs)} -> {OUT}", flush=True)
                os._exit(0)  # GStreamer 线程内安全退出（结果已落盘）
            self.done += 1
            for o in fm.object_items:
                if o.unique_component_id != GIE_ID:
                    continue
                r = o.rect_params
                probs = None
                # 一次性迭代器: 边迭代边取值, 勿 list() 物化（notes §3.2）
                for it in o.user_meta_items(12):
                    try:
                        to = it.as_tensor_output()
                        for _name, t in to.get_layers().items():
                            try:
                                arr = np.from_dlpack(t)
                            except TypeError:
                                arr = torch.utils.dlpack.from_dlpack(t).cpu().numpy()
                            arr = np.asarray(arr).reshape(-1)
                            if arr.shape[0] == 2:  # 二元分类 softmax 输出
                                probs = arr.astype(float).tolist()
                    except Exception:
                        pass
                self.recs.append(dict(
                    frame=fm.frame_number,
                    bbox=[round(r.left), round(r.top), round(r.width), round(r.height)],
                    probs=probs))


def main() -> int:
    global CFG, OUT, FRAMES, GIE_ID
    ap = argparse.ArgumentParser(description="Stage2: DS 管线逐 person 概率抓取")
    ap.add_argument("--sgie-config", required=True, help="带 output-tensor-meta=1 的 sgie INI")
    ap.add_argument("--out", required=True, help="抓取结果 json 输出路径")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--video", default="tmp/Mobile Camera0676.mp4")
    ap.add_argument("--pgie-config", default="deploy/generated/person/person/pgie_config.txt")
    ap.add_argument("--gie-id", type=int, default=1,
                    help="锚点 GIE 的 gie-unique-id（对象过滤用）")
    ap.add_argument("--mux-width", type=int, default=1920)
    ap.add_argument("--mux-height", type=int, default=1080)
    args = ap.parse_args()

    CFG = args.sgie_config
    OUT = args.out
    FRAMES = args.frames
    GIE_ID = args.gie_id
    video = ROOT / args.video

    def run():
        sgie = anchor(CFG)
        pgie = anchor(str(ROOT / args.pgie_config))
        p = Pipeline("cls-eval")
        p.add("nvurisrcbin", "src", {"uri": "file://" + str(video)})
        p.add("nvstreammux", "mux", {"batch-size": 1,
                                     "width": args.mux_width,
                                     "height": args.mux_height,
                                     "batched-push-timeout": 33000})
        p.add("nvinfer", "pgie", {"config-file-path": pgie})
        p.add("nvinfer", "sgie", {"config-file-path": sgie})
        p.add("fakesink", "sink")
        p.link(("src", "mux"), ("", "sink_%u"))
        p.link("mux", "pgie", "sgie", "sink")
        p.attach("sgie", Probe("cap", Cap(FRAMES)))
        p.start().wait()

    # DS 管线跑在子进程: 探针在流线程里用 os._exit 退出, 进程包装保证可回收
    proc = Process(target=run)
    proc.start()
    proc.join()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
