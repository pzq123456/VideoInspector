# Vest 分类模型评测流程（固定化 SOP）

> 目的：训练指标（val top1）不反映 DeepStream 部署分布，不能作为上线依据。
> 本流程 = 「DS 真实管线 A/B + GT 人工核验」的固定验收门禁。
> 实测案例与结论见文末表格（2026-08~09 系列 vest 模型）。

---

## 流程总览

```
Stage1 预处理鲁棒性扫描   eval_cls_native.py     （离线, 快速筛查, 不能定论）
Stage2 DS 真实管线抓取    eval_cls_deepstream.py （逐 person 原始 softmax, 部署判定）
Stage3 对比 + GT 核验     eval_cls_report.py     （分歧裁剪图 → 人工看图仲裁）
```

## 准备：候选模型引擎构建（与 tools/model_build.py 同参数）

```bash
# 1) 导出 ONNX（与 model_build.py 完全一致: dynamic + imgsz 320）
python3 - <<'EOF'
from ultralytics import YOLO
from pathlib import Path
import shutil
out = Path("/tmp/opencode/<候选>/best.onnx"); out.parent.mkdir(parents=True, exist_ok=True)
m = YOLO("deploy/models/vest/<候选>/weights/best.pt", task="classify")
m.model.pt_path = str(out.with_suffix(".pt"))
p = Path(m.export(format="onnx", dynamic=True, imgsz=(320, 320)))
if p.resolve() != out.resolve(): shutil.copyfile(p, out)
EOF
# 2) trtexec fp16 引擎（classifier max_batch=32）
/usr/bin/trtexec --onnx=/tmp/opencode/<候选>/best.onnx --fp16 \
  --minShapes=images:1x3x320x320 --optShapes=images:32x3x320x320 \
  --maxShapes=images:32x3x320x320 --memPoolSize=workspace:2048M \
  --saveEngine=/tmp/opencode/<候选>/best_dyn_fp16.engine
# 3) labels + sgie 配置: 以现役版本目录的 sgie_config.txt 为模板,
#    改 onnx/engine/labels 三行路径, 并把 maintain-aspect-ratio=1 替换为 output-tensor-meta=1
cp deploy/generated/vest_cls/vest/<基线>/labels.txt /tmp/opencode/<候选>/
sed 's|^maintain-aspect-ratio=1|output-tensor-meta=1|' \
  deploy/generated/vest_cls/vest/<候选版本>/sgie_config.txt \
  | sed 's|generated/vest_cls/vest/<候选版本>|/tmp/opencode/<候选>|g' > /tmp/opencode/sgie_<候选>_tensor.txt
```

## Stage1 — 预处理鲁棒性扫描

```bash
python3 tools/eval_cls_native.py \
  --models deploy/models/vest/<基线>/weights/best.pt \
           deploy/models/vest/<候选>/weights/best.pt \
  [--video "tmp/Mobile Camera0676.mp4"] [--max-crops 400] [--out /tmp/opencode/cls_native_scan.json]
```

- 部署版 person 检测器抽帧取裁剪 → 每裁剪 4 种预处理：`centercrop`（ultralytics 原生）、
  `stretch`（拉伸）、`lb_black`（黑边 letterbox ≈ DS nvinfer 实际行为）、`lb_gray`（历史脆弱模式）
- **解读**：跨变体翻转率高 ⇒ 对 nvinfer（GPU 缩放+黑边）输入分布脆弱
- **警示**：稳定性 ≠ 准确性。0827 曾以 6~7% 翻转率掩盖了系统性 vest 偏移（见下表）。
  本阶段只筛查，定论靠 Stage2+3。

## Stage2 — DS 真实管线逐对象概率抓取

```bash
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH

# 基线（现役引擎）:
sed 's|^maintain-aspect-ratio=1|output-tensor-meta=1|' \
  deploy/generated/vest_cls/vest/<基线版本>/sgie_config.txt > /tmp/opencode/sgie_base_tensor.txt
python3 tools/eval_cls_deepstream.py --sgie-config /tmp/opencode/sgie_base_tensor.txt \
  --out /tmp/opencode/ds_base.json --frames 150
# 候选:
python3 tools/eval_cls_deepstream.py --sgie-config /tmp/opencode/sgie_<候选>_tensor.txt \
  --out /tmp/opencode/ds_cand.json --frames 150
```

- 原理：`output-tensor-meta=1` + 探针读 `user_meta_items(12)` → `as_tensor_output()` →
  dlpack 回读每个 person 的 `[class0, class1]` softmax。
  绕开两个 DS quirk：`classifier_meta` 只挂 class0、`get_n_label` 拿不到概率（见
  `deepstream-metadata-exploration.md` §8/§9）。
- 输出 json：`{frame, bbox, probs}`，对象按 `(frame, bbox)` 跨运行精确配对（同 pgie 输出）。

## Stage3 — 对比与 GT 仲裁

```bash
python3 tools/eval_cls_report.py \
  --captures 基线=/tmp/opencode/ds_base.json 候选A=/tmp/opencode/ds_a.json 候选B=/tmp/opencode/ds_b.json \
  --dump-crops 20
```

- 统计：no_vest@0.5（违规判定）、no_vest@0.7（对齐 `attribute_threshold`）、
  边界样本（0.4~0.6）、与基线分歧明细
- `--dump-crops` 导出分歧裁剪图（文件名含双方概率）→ **用 Read 工具看图判定 GT**
  （本测试视频 GT：车内灰衣人员=违规 no_vest；荧光黄反光衣工人=合规 vest）
- **方向性原则**：把违规判成 vest（漏报）比把合规判成 no_vest（误报）更致命。
  漏报是本分类器存在的意义所在。
- 告警链路注意：no_vest 检出须 `probs[0] >= attribute_threshold`（config 中现为 0.8）
  且满足 `min_detection_count`；`output-tensor-meta` 运行的概率即真实告警依据。

---

## 历次结论（tmp/Mobile Camera0676.mp4 前 150 帧, 310 对象, GT 核验）

| 模型 | 违规召回 | 合规误报 | 跨预处理翻转 | 结论 |
|---|---|---|---|---|
| 0820（曾现役） | 98%（149/152） | 0 | 28~36% | 基线, 但预处理脆弱 |
| 0827 | 28% | 0 | 6~7% | 弃用：黑边⇒vest 系统性漏报（GT 全对却判 vest 0.9+） |
| 0902 | 98% | 2（边界 0.54/0.69） | 9~12% | 当时推荐 |
| 1104 | 94% | 0 | 8.5~16.5% | 不如 0902：新增 6 个高置信漏报 |
| 0904（现役） | 待评测 | 冒烟显示 no_vest@0.7=0（置信度不足告警线） | 待评测 | **需完整跑本流程** |

## 经验教训（必读）

1. **val top1 会骗人**：0827 val top1 0.995「最好」，DS 实测召回仅 28%。val split 不覆盖
   「DS letterbox × 难例人群」分布。
2. **预处理同源是这类分类器的命门**：ultralytics 训练用 Resize+CenterCrop，nvinfer 用
   黑边 letterbox + GPU 缩放。评测必须覆盖后者（Stage1 lb_black / Stage2 真管线）。
3. **错误方向分级**：违规判 vest（漏报）> 合规判 no_vest（误报）> 边界摇摆。
   候选比基线"错误总数"少但漏报多的，仍然不合格（1104 案例）。
4. **GT 必须人工看图核验**，分歧样本逐个确认；GT 假设错一步, 结论全错。
5. 单视频验收不充分, 换数据集/换参数训练后建议追加 1~2 个其他场景视频同流程复测。
