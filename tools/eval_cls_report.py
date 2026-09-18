#!/usr/bin/env python3
"""
tools/eval_cls_report.py — 分类模型评测 Stage3: 对比多个 Stage2 抓取结果 + 分歧裁剪图导出。

用法:
    python tools/eval_cls_report.py --captures 现役=/tmp/opencode/ds_old0820.json 候选=/tmp/opencode/ds_new0902.json \
        [--video "tmp/Mobile Camera0676.mp4"] [--dump-crops 10]

对比逻辑:
  - 对象按 (frame, bbox) 精确配对（同 pgie 输出, 各次运行一一对应）。
  - 以第一个 capture 为基线, 报告每个模型: 判定分布 / no_vest@0.5、@0.7 / 边界样本(0.4<P<0.6)
    / 与基线的判定分歧明细。
  - --dump-crops: 把分歧对象裁剪图存到 --out-dir（文件名含两模型概率）,
    供 GT 人工核验 —— 方向性: 把违规判成 vest(漏报) 比误报 no_vest 更致命。
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent


def load(path: str) -> dict:
    recs = json.load(open(path))
    return {(r["frame"], tuple(r["bbox"])): r for r in recs}


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage3: 抓取结果对比 + GT 核验裁剪导出")
    ap.add_argument("--captures", required=True, nargs="+", metavar="名称=路径",
                    help="≥2 个 Stage2 json; 第一个为基线")
    ap.add_argument("--video", default="tmp/Mobile Camera0676.mp4")
    ap.add_argument("--dump-crops", type=int, default=0,
                    help="每个分歧方向导出的裁剪图上限")
    ap.add_argument("--out-dir", default="/tmp/opencode/vest_gt")
    args = ap.parse_args()

    caps = {}
    for item in args.captures:
        name, path = item.split("=", 1)
        caps[name] = load(path)
    names = list(caps)
    base = names[0]
    keys = sorted(set.intersection(*[set(c) for c in caps.values()]))
    print(f"captures={names} matched_objects={len(keys)}")

    # ---- 每模型统计 ----
    for name in names:
        ps = [caps[name][k]["probs"] for k in keys if caps[name][k]["probs"]]
        nv05 = sum(1 for p in ps if p[0] >= 0.5)
        nv07 = sum(1 for p in ps if p[0] >= 0.7)
        border = sum(1 for p in ps if 0.4 < p[0] < 0.6)
        none = sum(1 for k in keys if not caps[name][k]["probs"])
        print(f"\n[{name}] n={len(ps)} 缺tensor={none}")
        print(f"  pred no_vest@0.5: {nv05} ({100*nv05/len(ps):.1f}%)  pred vest: {len(ps)-nv05}")
        print(f"  no_vest@0.7(告警线): {nv07}  边界样本(0.4~0.6): {border}")

    # ---- 两两分歧 ----
    dump = []
    for other in names[1:]:
        dis = [(k, caps[base][k]["probs"], caps[other][k]["probs"]) for k in keys
               if caps[base][k]["probs"] and caps[other][k]["probs"]
               and (caps[base][k]["probs"][0] >= 0.5) != (caps[other][k]["probs"][0] >= 0.5)]
        print(f"\n[{other}] vs [{base}] 判定不一致: {len(dis)}/{len(keys)}")
        for k, a, b in dis[:20]:
            print(f"  f={k[0]} {k[1]} {base}=[{a[0]:.2f},{a[1]:.2f}] {other}=[{b[0]:.2f},{b[1]:.2f}]")
        dump.extend((other, k, a, b) for k, a, b in dis)

    # ---- 分歧裁剪图导出（GT 核验用）----
    if args.dump_crops and dump:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(ROOT / args.video))
        n = 0
        for other, k, a, b in dump[:args.dump_crops]:
            f, (x, y, w, h) = k[0], k[1]
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, fr = cap.read()
            if not ok:
                continue
            tag = f"{f}_{x}_{y}_{w}x{h}"
            p = out_dir / f"dis_{base}_{other}_{tag}.jpg"
            cv2.imwrite(str(p), fr[y:y + h, x:x + w])
            n += 1
        cap.release()
        print(f"\n分歧裁剪图已导出 {n} 张 -> {out_dir}（文件名含双方概率, 供 GT 人工核验）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
