#!/usr/bin/env python3
"""
tools/eval_cls_native.py — 分类模型评测 Stage1: 预处理鲁棒性扫描（离线快速筛查）。

用法:
    python tools/eval_cls_native.py --models <baseline.pt> <candidate.pt> [...]
        [--video "tmp/Mobile Camera0676.mp4"] [--out /tmp/opencode/cls_native_scan.json]

流程:
  部署版 person 检测器抽帧取 person 裁剪 → 每个裁剪做 4 种预处理变体
    centercrop = ultralytics 原生（最短边→imgsz + CenterCrop）
    stretch    = 拉伸填满（cv2 双线性）
    lb_black   = 黑边 letterbox（≈ DeepStream nvinfer 实际行为: 保宽高比+黑对称补边）
    lb_gray    = 灰 114 补边 letterbox（历史脆弱模式）
  → 各模型 torch forward + softmax → 汇总: 逐变体判定占比、跨变体翻转率（vs centercrop）、
    候选模型与基线的分歧。

解读:
  - 翻转率高 ⇒ 对 DeepStream nvinfer（GPU 缩放 + 黑边 letterbox）的输入分布脆弱。
  - 稳定性 ≠ 准确性: 本脚本只做筛查, 部署判定以 eval_cls_deepstream.py + GT 人工核验为准。
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent


def load_cls(pt: str):
    """加载 ultralytics classify .pt 的裸网络（float32 cuda eval）。"""
    m = torch.load(str(pt), map_location="cuda", weights_only=False)
    ck = m["model"] if isinstance(m, dict) else m
    net = ck.model.float().to("cuda").eval() if hasattr(ck, "model") else ck.float().to("cuda").eval()
    return net, ck.names


@torch.no_grad()
def forward_probs(net, x) -> np.ndarray:
    """x: [1,3,imgsz,imgsz] float 0-1 RGB → [n_classes] softmax 概率。"""
    out = net(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return torch.softmax(out, dim=1)[0].cpu().numpy()


def prep_variants(crop_bgr: np.ndarray, size: int) -> dict[str, torch.Tensor]:
    h, w = crop_bgr.shape[:2]
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    outs = {}
    # centercrop: 最短边缩放到 size, 取中间方窗
    r = size / min(h, w)
    nw, nh = max(size, int(round(w * r))), max(size, int(round(h * r)))
    rz = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0, x0 = (nh - size) // 2, (nw - size) // 2
    outs["centercrop"] = rz[y0:y0 + size, x0:x0 + size]
    # stretch: 拉伸填满
    outs["stretch"] = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    # letterbox 黑边 / 灰边（对称补边）
    for tag, color in [("lb_black", (0.0, 0.0, 0.0)), ("lb_gray", (114 / 255,) * 3)]:
        r2 = min(size / h, size / w)
        nw2, nh2 = int(round(w * r2)), int(round(h * r2))
        rz2 = cv2.resize(rgb, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((size, size, 3), color, dtype=np.float32)
        dx, dy = (size - nw2) // 2, (size - nh2) // 2
        canvas[dy:dy + nh2, dx:dx + nw2] = rz2
        outs[tag] = canvas
    return {k: torch.from_numpy(v.transpose(2, 0, 1)[None]).float().cuda() for k, v in outs.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--models", required=True, nargs="+",
                    help="best.pt 路径列表; 第一个为基线, 其余为候选")
    ap.add_argument("--video", default="tmp/Mobile Camera0676.mp4")
    ap.add_argument("--person-onnx", default="deploy/generated/person/person/best.onnx",
                    help="部署版 person 检测器（抽裁剪用）")
    ap.add_argument("--out", default="/tmp/opencode/cls_native_scan.json")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--stride", type=int, default=10, help="抽帧步长")
    ap.add_argument("--conf", type=float, default=0.4, help="person 检测置信度")
    ap.add_argument("--max-crops", type=int, default=400)
    ap.add_argument("--max-frames", type=int, default=3000)
    args = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(str(ROOT / args.person_onnx), task="detect")
    video = ROOT / args.video

    # ---- 抽取 person 裁剪 ----
    crops: list[tuple[int, tuple, float]] = []  # (frame_idx, (x1,y1,x2,y2), aspect)
    cap = cv2.VideoCapture(str(video))
    fi = 0
    while len(crops) < args.max_crops:
        ok, frame = cap.read()
        if not ok or fi > args.max_frames:
            break
        if fi % args.stride == 0:
            r = det.predict(frame, conf=args.conf, verbose=False, imgsz=960)[0]
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                if x2 - x1 < 30 or y2 - y1 < 60:
                    continue
                crops.append((fi, (x1, y1, x2, y2), (x2 - x1) / (y2 - y1)))
        fi += 1
    cap.release()
    print(f"video_frames_scanned={fi} person_crops={len(crops)}", flush=True)

    nets, names = [], []
    for p in args.models:
        net, n = load_cls(p)
        nets.append(net)
        names.append(n)
        print(f"model[{len(nets)-1}] {p} names={n}", flush=True)

    variants = ["centercrop", "stretch", "lb_black", "lb_gray"]
    recs = []
    for i, (fid, bb, aspect) in enumerate(crops):
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        crop = frame[bb[1]:bb[3], bb[0]:bb[2]]
        tensors = prep_variants(crop, args.imgsz)
        rec = dict(i=i, frame=fid, bbox=bb, aspect=round(aspect, 3))
        for mi in range(len(nets)):
            rec[f"m{mi}"] = {v: forward_probs(nets[mi], tensors[v]).tolist() for v in variants}
        recs.append(rec)
        if (i + 1) % 50 == 0:
            print(f"progress {i+1}/{len(crops)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(args.out, "w"))
    print(f"saved -> {args.out}", flush=True)

    # ---- 汇总 ----
    def pred(p):
        return int(np.argmax(p))

    print("\n=== 各模型逐预处理判定 vest 占比 (%) ===", flush=True)
    for mi in range(len(nets)):
        row = {v: round(100 * sum(pred(r[f"m{mi}"][v]) for r in recs) / len(recs), 1)
               for v in variants}
        print(f"m{mi}({Path(args.models[mi]).parent.parent.name}): {row}", flush=True)

    print("\n=== 跨预处理翻转率（vs centercrop, 越低越鲁棒）===", flush=True)
    for mi in range(len(nets)):
        base = [pred(r[f"m{mi}"]["centercrop"]) for r in recs]
        for v in ["stretch", "lb_black", "lb_gray"]:
            flips = [r["i"] for j, r in enumerate(recs) if pred(r[f"m{mi}"][v]) != base[j]]
            print(f"m{mi}: flip_cc_vs_{v}: {len(flips)}/{len(recs)} ({100*len(flips)/len(recs):.1f}%) idx={flips[:15]}", flush=True)

    print("\n=== 候选 vs 基线 判定分歧 ===", flush=True)
    for mi in range(1, len(nets)):
        for v in variants:
            dis = [r["i"] for j, r in enumerate(recs)
                   if pred(r[f"m{mi}"][v]) != pred(r["m0"][v])]
            print(f"m{mi} vs m0 [{v}]: {len(dis)}/{len(recs)} idx={dis[:15]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
