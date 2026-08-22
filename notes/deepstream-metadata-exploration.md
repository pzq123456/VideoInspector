# DeepStream 元数据探索笔记

> 探索日记：摸清 DeepStream 里「检测到的目标」真实数据结构是什么样，以及跑通
> Python (pyservicemaker) 探针时的各种坑。配套工具：`tools/explore_metadata.py`。
>
> 日期：2026-08-20 · 环境：`nvcr.io/nvidia/deepstream:9.0-triton-multiarch`

---

## 1. 环境事实（先说清楚，避免下次重新摸）

| 项 | 值 |
|----|----|
| DeepStream SDK | 9.0.0（`/opt/nvidia/deepstream/deepstream`） |
| 容器镜像 | `nvcr.io/nvidia/deepstream:9.0-triton-multiarch` |
| GPU | RTX 5060（Ampere 之后的消费级，显存 8GB） |
| Python | 3.12.3（`/usr/bin/python3`） |
| Python 绑定 | **pyservicemaker 已装**（`/usr/local/lib/python3.12/dist-packages/pyservicemaker`） |
| pyds | **未安装**（DeepStream 9.0 起 Python bindings 已废弃，不再发 wheel；官方建议用 pyservicemaker） |
| 模型 | person(yolo26n) / helmet / harness_cls / vest_cls，引擎已预构建在 `models/` |

### 关键环境变量
```bash
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH
```
不设 `GST_PLUGIN_PATH`，Python 里 `Gst.Registry.get().find_plugin()` 找不到
nvinfer/nvstreammux 等（缓存 registry 问题）。但注意：**`ElementFactory.find()` 能拿到**
（会触发重扫），所以能建元素；`find_plugin` 反而可能返回 missing，别被它误导。

---

## 2. DeepStream 检测目标真实数据结构

### 2.1 层级关系（C 侧，`nvdsmeta.h` 权威定义）
```
GstBuffer ──> NvDsBatchMeta
                └─ frame_meta_list (NvDsFrameMeta)     每批一个/源一个
                     └─ obj_meta_list (NvDsObjectMeta)  每个检测目标一个
                          └─ classifier_meta_list (NvDsClassifierMeta)  二级分类结果
                               └─ label_info_list (NvDsLabelInfo)
```

### 2.2 pyservicemaker 视角暴露的真实字段（探针 `handle_metadata(batch_meta)` 里）

**FrameMetadata**（=`NvDsFrameMeta`，`batch_meta.frame_items` 迭代）：
`pad_index, batch_id, frame_number, buffer_pts, ntp_timestamp, source_id,
source_width, source_height, pipeline_width, pipeline_height,
object_items, user_meta_items, tensor_items, segmentation_items,
nvdsanalytics_frame_items`

> 注意：**没有 `num_obj_meta`**，用 `object_items` 迭代（一次性迭代器）。
> 也没有 `source_frame_width/height`，而是 `source_width/height`。

**ObjectMetadata**（=`NvDsObjectMeta`，`frame_meta.object_items` 迭代）：
`append, class_id, label, confidence, tracker_confidence, object_id,
unique_component_id, rect_params, text_params, mask_params,
classifier_items, user_meta_items, tensor_items, obj_reid_items,
nvdsanalytics_obj_items`

> 字段名是 `label`（不是 C 侧的 `obj_label`）。

**rect_params**（=`NvOSD_RectParams`）：
`left, top, width, height, border_width, rotation_angle, has_bg_color,
border_color(r,g,b,a), bg_color(r,g,b,a)`

### 2.3 实测一帧真实输出（person 整帧检测，uid=1）
```
object_meta:
  .unique_component_id = 1        # gie-unique-id，多模型用这个区分来源，勿用 class_id
  .class_id = 0                   # 对应 labels.txt 行序（person=0）
  .label = 'person'
  .confidence = 0.8188
  .tracker_confidence = 0.0
  .object_id = 18446744073709551615   # == UNTRACKED_OBJECT_ID（未跟踪）
  .rect_params: left=228 top=160.9 width=226.5 height=562.1
```
`object_id` 未跟踪时是 `18446744073709551615`（= `UNTRACKED_OBJECT_ID`，uint64 全 1），
不是 -1，判断时要留意。

### 2.4 二级分类器元数据（关键结论）
实测（`--full` 跑 person→helmet→harness_cls→vest_cls）：
```
object_meta(class_id=0, label='person').classifier_items:
  classifier_meta:
    .unique_component_id = 5      # harness_cls 的 gie-unique-id
    .n_labels = 1
    .get_n_label(0) = 'harness'   # ⚠️ 只有 label 字符串，没有置信度！
```
**结论：`get_n_label(i)` 只返回 label 名，不含概率。** 这印证了
`server/README.md:94` 的担忧 —— 二级分类置信度拿不到，只能用
`classifier-threshold` 回退（`CLASSIFIER_FALLBACK_CONF=0.5`）。
若需真实概率，必须在 nvinfer 侧扩展自定义 classifier parser 输出概率。

> 观察：本例每 person 只挂到 harness(uid=5) 一个 classifier_meta；
> vest(uid=6) 未出现（可能该样本被阈值裁掉）。多分类器时按
> `unique_component_id` 区分每个分类器结果。

---

## 3. 运行时踩坑（重点，防止重蹈覆辙）

### 3.1 本地文件输入用 `nvurisrcbin`，别用 filesrc+qtdemux
- 官方 sample（`deepstream_sr_test.py` 等）对文件 URI 都用 `nvurisrcbin`，好用。
- 我用 `nvurisrcbin` + `file://` 读 MP4 正常（数百帧）。
- `filesrc → qtdemux → h264parse → nvv4l2decoder` 那条链：qtdemux 是动态 pad，
  `p.link(("demux","parse"),("","sink"))` 接不上会**挂死/超时**。别走这条路。
- 多路 `p.link((f"src{i}","mux"),("","sink_%u"))` —— **必须用 `"sink_%u"`**（项目代码注释也强调）。

### 3.2 `object_items` 是一次性迭代器 —— 不能 list() 复用
- **`list(fm.object_items)` 物化后再访问 wrapper 会静默崩溃 / 段错误**。
- 必须**边迭代边取值**：`for o in fm.object_items: 用 o.xxx`。
- 同一帧要两趟就**重新取迭代器**（`probe.py` 就是这么做的）。
- `len()` 也不可用，想计数就迭代累加。

### 3.3 全属性 `dir()`+`getattr` 遍历会硬崩溃
- 对 wrapper 逐个 `getattr` 遍历所有 `dir()` 字段，某些字段（如
  `user_meta_items`、分类器 `classifier_type`）会触发
  **`pybind11::error_already_set` → `std::terminate` → 进程被 kill**，
  `except Exception` **捕不到**（是 C++ 异常）。
- 安全做法：只读**已核验**的字段（见第 2 节列表）；列字段名用 `dir()` 即可，别取值。

### 3.4 探针里 print 要 flush，别用 `os._exit(0)` 前不 flush
- 探针跑在 GStreamer 流线程，`os._exit(0)` 会**丢弃未 flush 的 stdout 缓冲**，
  结果看起来「没打印」。
- 每个 print 带 `flush=True`，或用文件重定向 + flush（`tools/explore_metadata.py`
  的 `_p()` 就是干这个的）。

### 3.5 多模型整帧检测 + 次级 GIE 的顺序坑（项目已有经验）
- 多个 `process-mode=1` 整帧检测器**按顺序排**即可（pgie→helmet→vest），
  不要把一个整帧 nvinfer 放在次级 GIE 之后（会卡死）。
- 二级分类器用 `process-mode=2` + `operate-on-gie-id=1`，自动裁剪 person 整框。

### 3.6 INI 里相对路径基于 config 文件所在目录解析
- `models/person/best.onnx` 写在 `configs/pgie_config_person.txt` 里会被解析成
  `configs/models/person/...` → 找不到。
- 必须像 `server/main.py:_anchor_ini_config` 那样启动时把模型路径**锚定为绝对路径**
  （`tools/explore_metadata.py:_anchor()` 也实现了同样的逻辑）。

---

## 4. 探索工具

`tools/explore_metadata.py`（本会话新增，已验证可用）：

```bash
# 环境（每次跑前）
export GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins
export LD_LIBRARY_PATH=/opt/nvidia/deepstream/deepstream/lib:$LD_LIBRARY_PATH

# ① 看字段结构（dir，安全）
python3 tools/explore_metadata.py --file 'tmp/Mobile Camera0676.mp4' --dump-schema

# ② 打印前 N 帧每个目标真实值（person 检测）
python3 tools/explore_metadata.py --file 'tmp/Mobile Camera0676.mp4' --dump-frame 5

# ③ 叠加 helmet+harness_cls+vest_cls，转储二级分类器元数据
python3 tools/explore_metadata.py --file 'tmp/Mobile Camera0676.mp4' --full --dump-frame 5
```

---

## 5. 参考
- C 侧权威结构体：`/opt/nvidia/deepstream/deepstream/sources/includes/nvdsmeta.h`
  （`NvDsFrameMeta` L301、`NvDsObjectMeta` L360、`NvDsClassifierMeta` L418）
- pyservicemaker 官方 sample：`/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/python/{flow_api,pipeline_api}`
- 本项目已复用的 demo：`tmp/simple-demo3/two_stage_demo.py`（空间关联已验证方案）
- 本项目生产代码：`server/pipeline/probe.py`（探针）、`server/main.py`（管线构建）

---

## 6. 排查：DeepStream 报 no_vest 而本地 ultralights 报 vest（真实 bug 定位）

### 6.1 现象
同一人物、同一 `models/vest_cls/best.onnx`：
- **本地 ultralights 原生**（best.pt / onnx via YOLO）→ **vest**（conf ≈ 1.0）
- **DeepStream**（vest sgie）→ **no_vest**

### 6.2 真实答案（看图确认）
用 viewer 识别裁剪图 `tmp/opencode/person_big_context.jpg`：
> 人物**穿着荧光黄/黄绿色高可视反光背心**（带反光条、印字），并戴黄色安全帽。

→ **ground truth = vest**。所以 **ultralights 对、DeepStream 错**（用户判断正确）。

### 6.3 排除了"标签/元数据结构错位"
- 模型真实类别顺序（读训练 `best.pt` 的 `names`）与 `labels.txt` **完全一致**：
  - vest: `{0:'no_vest', 1:'vest'}` ↔ labels.txt `no_vest/vest`
  - harness: `{0:'harness', 1:'no_harness'}` ↔ labels.txt `harness/no_harness`
- 所以 **不是 class_id→label 映射错配**（不是"名不副实"的标签文件问题）。

### 6.4 根因：模型过于脆弱 + DeepStream 预处理不同
对同一 person 裁剪（226x562 竖条）用 onnxruntime 扫描预处理变体：

| 预处理 | vest 判定 |
|---|---|
| 拉伸到 320x320（任意插值） | **vest**（1.0）|
| 黑色补边 letterbox（=nvinfer 源码里的 padding 色）| **vest**（0.94）|
| 灰色补边 letterbox | **no_vest**（0.78）|

- ultralights 训练/推理用**拉伸** → vest。
- nvinfer（`gstnvinfer.cpp:1372`）用 **NvBufSurfTransform GPU 硬件缩放器 + 黑色补边**，
  其输出分布与该脆弱模型不匹配 → no_vest。
- **关键：改 `maintain-aspect-ratio`(0/1/删行)、`symmetric-padding` 都无法翻转**
  （实测 80 帧仍 68~70/80 报 no_vest）。说明靠 nvinfer 配置无法对齐 ultralights 的输入分布。
- harness 模型**对预处理鲁棒**（拉伸/补边都 1.0），所以只有 vest 出问题。

### 6.5 结论 / 解决方案建议
vest 分类器对输入缩放/补边**极其敏感**（同样的人，置信度从 1.0 的 vest 翻到 0.5~0.78 的
no_vest），这是一个**模型健壮性问题**，不是代码/配置 bug。

- **正解（推荐）**：重训/微调 vest 分类器，加入能模拟部署预处理的增强
  （黑色 letterbox 补边、随机缩放/插值、宽高比抖动），使其对
  DeepStream nvinfer 的实际输入分布鲁棒。当前模型上线即脆。
- 备选：训练/推理统一走同一套预处理管线（含 nvinfer 同款 GPU 缩放），
  但这需要能精确复现 `NvBufSurfTransform` 的数值，工程上脆弱。
- 阈值调整无济于事：no_vest 置信度 0.5~0.78 常在阈值之上。

### 6.6 复现脚本
- 原生对比：`/tmp/opencode/native_cls.py`（ultralights 预处理 vs 手写预处理）
- DeepStream 统计：`/tmp/opencode/run_cmp.py <vest_cfg> <harness_cfg> <frames>`
  （正常配置 vest 报 no_vest 70/80 帧）
- 参考预处理扫描：见本会话内联 python（letterbox 黑白/拉伸/插值 对比）

> 附加观察：vest(uid=6) 分类器 meta 只有在正常配置下才挂到 person；
> 给 vest 配置加 `output-tensor-meta=1` 后 classifier_meta 反而不出现（tensor 输出覆盖）。

### 6.7 配置实验（试图用最小改动修好）—— 结论：纯配置无法可靠修复

尝试在 vest sgie 配置上改 `maintain-aspect-ratio` / `scaling-filter` / `symmetric-padding`：

| 配置 | 80帧 vest 判定 |
|---|---|
| 原配置 (mar=1, sf=默认6=GPU-Nearest, sym=1) | no_vest 70/80 |
| mar=0 | no_vest 68/80 |
| mar=0 + scaling-filter=1(双线性) | no_vest 24/60（约减半，但**未根治**）|
| mar=0 + scaling-filter=0(最近邻) | no_vest 48/60 |

- `scaling-filter` 取值（`nvbufsurftransform.h:85`）：0=Nearest, 1=Bilinear, 2~5=Algo*, **6=Default(GPU-Nearest)**。
- nvinfer 默认用 **GPU 最近邻**缩放 + GPU 硬件缩放器（`NvBufSurfTransformInter_Default`），
  与 ultralights 的 cv2 双线性不一致 → 是翻转因素之一。
- 但 `mar=0 + 双线性` 只把误报减半、不能根治：GPU 缩放器的数值仍与 cv2 有差异，
  而这个 vest 模型太脆，微小差异就翻车。

### 6.8 附加发现：vest classifier_meta 挂载不稳定
- 前 ~20 帧 vest(uid=6) 分类器 meta **完全不挂到 person**（`get_n_label` 无结果）；
  只有更后面的帧才开始挂（且挂上多为 no_vest）。
- harness(uid=5) 则稳定挂载。两个二级分类器串联 (harness→vest) 与否不影响该现象。
- 含义：除"判错"外，vest 结果还存在**挂载延迟/丢失**问题，值得单独排查。

### 6.9 最终判断
- **纯配置改动无法可靠修复**（已实测多种组合）。
- 根因是 **vest 模型对输入预处理过于敏感（脆弱/过拟合）** + nvinfer GPU 缩放与训练预处理不一致。
- **可靠的最小改动方向**（二选一）：
  1. 让模型鲁棒：重训/微调 vest 分类器，增强里加入 nvinfer 同款预处理
     （GPU 缩放、黑色 letterbox/拉伸、宽高比抖动）—— 治本，需重训。
  2. 管线层面用 `nvdspreprocess` / `input-tensor-meta` 喂与 ultralights 完全一致的
     预处理裁剪 —— 改动中等，需自写预处理。
- 可先试的廉价改法：`maintain-aspect-ratio=0` + `scaling-filter=1`（减半误报，但不根治）。

### 6.10 配置旋钮穷尽 + 仓库结论
- ultralights classify **实际喂拉伸填满 320x320**（内容占满全图、无补边）→ vest。
  所以 nvinfer 方向应是 `maintain-aspect-ratio=0`（拉伸）。
- 但 nvinfer 用 **GPU 硬件缩放器**（`NvBufSurfTransform`），其数值与 cv2 拉伸有差异，
  且 **`scaling-compute-hw` 在 x86 只支持 Default/GPU**（CPU 不可用，VIC 仅 Jetson），
  无法强制 CPU 缩放。
- 配置旋钮穷尽：`maintain-aspect-ratio`、`scaling-filter`、`symmetric-padding`、
  `scaling-compute-hw` 均无法让 nvinfer 精确复现 ultralights 的 cv2 预处理。
  `mar=0 + scaling-filter=1`(双线性) 只能把误报减半，不能根治。

- **DeepStream-Yolo 仓库的价值**（tmp/DeepStream-Yolo-master）：
  * 印证核心原则：`maintain-aspect-ratio` / `net-scale-factor`(+offsets) /
    `model-color-format` **必须按训练预处理设置**（不同模型 0/1 不同）；
  * 印证 person(yolo26) 检测器配置 `maintain-aspect-ratio=1` 与仓库推荐一致；
  * 但该仓库是**检测模型解析器**（bbox parser），**不覆盖分类器**，
    未提供分类器预处理对齐的现成方案。

---

## 7. 新模型评估 + 二阶段二元分类器元数据挂载问题（2026-08-21）

> 评估对象：`models/original/vest/yolo26mcls_ppeVest_20260820_0923`（新训练 vest 分类器，
> 目标：修复 §6 的「预处理脆弱」）。方法：同一 person 裁剪框，对比 native(python YOLO)
> 与 DeepStream 引擎输出，并单独量化二级分类器元数据挂载率。配套脚本见 `/tmp/opencode/`
> （`mount_rate.py` / `diag2.py` / `native_vs_ds.py` / `full_native.py`）。

### 7.1 重要更正：ultralights classify 的真实推理预处理
§6.10 里写「ultralights classify 实际喂拉伸填满 320x320」是**错的**。
实测 `yolo26m-cls` 的 `model.transforms` 是：
```
Compose(
  Resize(size=320, bilinear, antialias=True)   # 最短边缩放到 320
  CenterCrop((320,320))                          # 裁中间方窗
  ToTensor()                                     # /255
  Normalize(mean=[0,0,0], std=[1,1,1])           # 恒等
)
```
→ 即 **Resize(最短边→320) + CenterCrop(320) + /255**，既不是 letterbox 也不是纯拉伸。
而 nvinfer 二级分类器是 **黑边 letterbox + /255**（`maintain-aspect-ratio=1`,
`symmetric-padding=1`, `net-scale-factor=1/255`）。**两者不同源**。

### 7.2 模型层结论：重训修复了「letterbox 脆弱」，但 native vs DS 仍不一致
- **旧模型**(`vest_cls`)：letterbox 家族里脆弱，灰边就翻（F1 gray→no_vest 0.873）。
- **新模型**：letterbox 家族（黑/灰/拉伸）之间已稳定一致（F1 gray/black/stretch 全 vest）。
  → **「灰补边脆弱」已修好，模型层目标基本达成。**
- 但新模型在 **native(centercrop) vs DS(黑letterbox)** 之间仍频繁不一致，
  40 帧里大个反光衣目标约 **10 帧两者判定相反**（F17/F18/F24/F26/F27/F28/F31/F32/F33…）。
  - 例如 F18: native→vest(0.806) vs DS黑→no_vest(0.556)；F28: native→no_vest(0.972) vs DS黑→vest(0.992)。
  - 有趣：GT=vest，DS 黑边多数反而判对（native centercrop 对超高裁剪框只取中间 320 常误判）。
  - 影响：按「DS 接近 python YOLO 表现」标准，新模型**仍不完全达标**（二者预处理不同源）。

### 7.3 管线层（主阻塞）：二级分类器元数据挂载严重不稳定 —— 与模型无关
对 `tmp/Mobile Camera0676.mp4` 跑 150 帧，person 目标共 311 个，统计挂上
`NvDsClassifierMeta` 的比例：

| 二级分类器 | 挂载率 |
|---|---|
| harness (uid=5) | 236/311 ≈ **76%** |
| vest (uid=6) 新模型 | 165/311 ≈ **53%** |
| vest (uid=6) 旧模型 | 197/311 ≈ 63% |

排除实验（均已实测）：
- **非阈值**：`classifier-threshold` 降到 0.05，F1-17 仍 `n_clf=0`（完全没有 classifier_meta）。
- **非 helmet 干扰**：去掉 helmet 整帧检测器，结果相同（165/311）。
- **非模型置信度**：F1 黑边 vest 0.997 却不挂载，F18 黑边 no_vest 0.556 反而挂载。
- **确定性可复现**：多次运行一致。
- 挂载偏态：**高瘦大框（aspect≈0.4~0.6 的整人身框）最常丢**，小框（head/torso）较稳，
  但并非纯几何规则（同 aspect 也时挂时不挂），更像 nvinfer process-mode=2 的
  批次/帧对齐竞态。

### 7.4 生产影响
`server/pipeline/probe.py:_classifier_status`（读 `classifier_items`/`get_n_label`）
与诊断读法一致 → 生产环境约**一半 person 的 vest 维度被判「未知」**（蓝框），
no_vest 违规会被静默漏报一半。harness 也有 24% 漏。对安全检测是致命缺陷，
**必须先于模型问题修复**。

### 7.5 后续深挖方向（二阶段二元分类器挂载 bug）
- 定位 nvinfer `process-mode=2` 下 `NvDsClassifierMeta` 的 attach 逻辑/批次对齐；
  怀疑与 `batched-push-timeout` 或 nvinfer 二级批次内部帧错位有关。
- 验证是否 `NvDsClassifierMeta` 挂到了**错误的 person/帧**（帧偏移），
  而非真正缺失（可解释「哪个分类器配置影响挂载率」）。
- 检查 harness(uid=5) 挂载 76% vs vest(uid=6) 53% 的差异来源（是否与输出分布/批处理有关）。
- 若确认为 DeepStream 9.0 + pyservicemaker 的问题，考虑绕道：
  自写 classifier parser 直接把概率写进 label / 或改用 `input-tensor-meta` 自喂预处理裁剪。

---

## 8. 二级分类器挂载 bug 真根因：只挂 class 0（2026-08-21）

> 本文档 §7.3/§7.5 的「批次/帧对齐竞态」假设**是错的**。经抓取 nvinfer 原始输出
> tensor 逐对象比对后，根因是**类别索引**问题。

### 8.1 决定性证据：抓取 DeepStream 引擎的原始 softmax（output-tensor-meta=1 + torch dlpack 读）
pyservicemaker 的 `get_n_label` 拿不到概率，但用 `output-tensor-meta=1` 让 nvinfer 挂
`NvDsInferTensorMeta`，再 `torch.utils.dlpack.from_dlpack(tensor).cpu().numpy()` 可读回
每个 person 的原始 `[class0, class1]` 概率（注意：`np.from_dlpack` 对 CUDA tensor 报
TypeError，必须用 torch；且 output-tensor-meta 会覆盖/抑制 classifier_meta，二者分开测）。

**harness `[harness=0, no_harness=1]` 实测：**

| 目标 | raw softmax | 预测 | 挂载? |
|---|---|---|---|
| F1 大个 | [0.992, 0.008] | harness(0) | ✅ clf=harness |
| F33 小框 | [0.0098, 0.990] | no_harness(1), conf 0.99 | ❌ clf=- |
| F34 小框 | [0.0001, 1.000] | no_harness(1), conf 1.0 | ❌ clf=- |

**vest `[no_vest=0, vest=1]` 实测：** 大个穿反光衣目标多数帧 raw=[0.02,0.98]（class1 vest,
conf 0.95~0.99）却 `clf=-` 不挂；F18-21 判成 no_vest(0) 的那几帧才挂上 no_vest。

### 8.2 结论
**DeepStream 二级分类器只挂载「class 0（labels.txt 第一个类）」的预测；class 1 的预测被
静默丢弃——无论置信度多高（1.0 也不挂）。** 这是类别索引相关，不是批次/阈值/几何/模型置信度。

- vest `[no_vest=0, vest=1]`：no_vest(0) 能挂能告警；vest(1) 永不挂 → 穿反光衣者永远「未知」。
- harness `[harness=0, no_harness=1]`：**no_harness(1)=违规，永不挂 → no_harness 告警完全失效**。

### 8.3 这解释了 §7 所有怪象
- 挂载率 ≈「预测为 class 0 的对象占比」（vest 53%、harness 76% → **模型相关**，因为不同模型
  在场景里判 class0 的比例不同）。
- 阈值 0.05/0.5 无差别（class1 对象根本没进 attach 流程）；0.99 才变（只影响 class0 低置信）。
- 与 helmet/几何无关（纯类别索引）。

### 8.4 C 侧代码核查（仍存疑的点）
DeepStream 源码 `nvdsinfer_context_impl_output_parsing.cpp` 的 `parseAttributesFromSoftmaxLayers`
（L608-666）对两个 class 都正常产出（argmax），`attach_metadata_classifier`
（`gstnvinfer_meta_utils.cpp:222`）也无类别过滤。→ C 侧理论上 class1 也应挂上。
**因此「只挂 class0」要么是 C 侧某处未定位行为，要么是 pyservicemaker 绑定只暴露 class0**
（绑定是编译好的 `_pydeepstream.so`，无源码）。→ 决定修法：改模型类别顺序 vs 换读取层。

### 8.5 修法方向
1. **把「要检测的违规类」做成 class 0**（最省事、与现有读取方式兼容）：
   - vest 已是（no_vest=0）✓；harness 需让模型把 no_harness 输出为 class0（重训交换类别
     或导出后交换输出索引）。
2. **确认是否 pyservicemaker 绑定只暴露 class0**：若 C 侧其实有 class1 meta，改到 C 侧
   （pyds / 自定义 parser 输出概率 / 换读取 API）即可不重训。
3. 探针 `_classifier_status` 本身读法正确（`classifier_items` 迭代 + `get_n_label`），
   class1 是 C 侧/绑定就没生成，不是探针读漏 —— 需官方 demo/文档佐证正确读法。

### 8.6 决定性确认：nvdsosd 直读 C 元数据 + 方案实测验证
- **OSD 直读 C 验证**（排除「pyservicemaker 绑定隐藏」）：跑 `pgie→harness→nvdsosd→appsink`
  抓渲染帧（`/tmp/opencode/osd_frames/`，用本地视觉模型 `inspect_image.py` 读）。
  - class-0(harness) 大框 → 渲染文字 `person harness`；class-1(no_harness) 小框 → 只有 `person`，
    **没有分类标签**。→ class1 在 C 元数据里就是**不存在**，不是绑定隐藏。
- **方案实测验证：交换输出通道让违规类=class0**：
  - harness ONNX 末端为 `...→Flatten→Gemm(linear [2,1280])→Softmax→output0[B,2]`。
  - 交换 `model.10.linear.weight` 的两行 + `model.10.linear.bias` 两元素（`init[::-1]`），
    另存 ONNX → trtexec 重建 fp16 引擎 → 新 `models/harness_viol/`（labels.txt=`no_harness\nharness`）。
  - 实测挂载：`mounted=75 labels={'no_harness':75}`（原模型只有 `harness` 236，no_harness 0）。
    → **no_harness(违规) 现在能挂载能告警了。**
- 结论：**修法 = 让每个二级二元分类器的「违规类」处于 class 0**（交换输出通道即可，无需重训）。
  - vest：no_vest 已 class0，**无需改**。
  - harness：交换后 no_harness=class0，**可生效**（`models/harness_viol/`）。
  - 代价：合规类变 class1 不挂 → 穿 harness/vest 者显示「未知/蓝框」，无法确认「合规/绿框」。
    对「只告警违规」场景足够；若要两态需 tracker+async 路线。

---

## 9. DeepStream 二级分类器底层设计要点（最值得记住的）

> 这些是排查「二级分类器为什么只有 class0」时沉淀的最重要的底层机制，记下来避免下次重摸。

### 9.1 二级推理（process-mode=2）完整数据流
```
GstBuffer → gst_nvinfer_process_objects(gstnvinfer.cpp:1915)
  逐 frame_meta / 逐 obj_meta(person, uid=1)：
    should_infer_object() 决定是否推理（untracked→无 history→恒 TRUE, class0/无max尺寸不过滤）
    get_converted_buffer(gstnvinfer.cpp:1357)  按 NvOSD_RectParams 设 src_rect/dst_rect，
        黑边 letterbox(maintain-aspect-ratio=1,symmetric-padding=1, GPU NvBufSurfTransform)
    入 batch->frames（每对象一个 GstNvInferFrame, 含 obj_meta 指针）
  满 max_batch_size 或帧末 → convert_batch_and_push_to_input_thread(gstnvinfer.cpp:1633)
    NvBufSurfTransformAsync 批量变换 tmp_surf(numFilled) → mem->surf(输入张量)
  输入线程 queueInputBatch → TensorRT 推理 → 输出线程 dequeueOutputBatch
  输出线程(gstnvinfer.cpp:2704) for each frame：
    attach_metadata_classifier(gstnvinfer_meta_utils.cpp:222)
      若 attributes.size()==0 或 label 空 → 直接 return（不挂 NvDsClassifierMeta）
      否则 acquire classifier_meta + 逐 attr 加 NvDsLabelInfo(result_class_id/result_prob/result_label)
```

### 9.2 关键：class0-only 是「C 侧就没生成 NvDsClassifierMeta」，不是读取问题
- parser `parseAttributesFromSoftmaxLayers`（output_parsing.cpp:608）对 2 类 softmax 取 argmax，
  两 class 都正常产出 1 个 attr；attach 也无类别过滤 —— **源码看两 class 都该挂**。
- 但实测（pyservicemaker n_clf=0 / raw tensor / nvdsosd 渲染三方一致）：**class1 不挂**。
- 结论：要么某处 C 行为未定位，要么是该 DeepStream 构建的 classifier attach 对 2 类只落 class0。
  → **务实绕开：让违规类=class0**（§8.6 已验证），不跟底层纠缠。

### 9.3 同步 vs 异步挂载（classifier-async-mode）
- 同步（默认，本 pipeline）：输出线程里 `attach_metadata_classifier(frame, new_info)` 直接挂到
  `frame.obj_meta`。untracked 对象也能挂（本场景走这条）。
- 异步（官方样例 `classifier-async-mode=1`）：输入线程用 `obj_history->cached_info` 挂载
  （gstnvinfer.cpp:2031-2037），**要求 object_id 有效（必须开 tracker）**，
  untracked 直接 skip（gstnvinfer.cpp:1961）。异步挂载路径可能没有 class0-only 限制，
  是「想同时拿合规+违规两态」的候选改法，但需引入 nvtracker、未验证。

### 9.4 官方 classifier 配置要点（对照样例 deepstream-app/config_infer_secondary_vehicletypes.txt）
- 用 `is-classifier=1` / `classifier-type=xxx`（`network-type=1` 是等效新名，均合法）。
- **不设 `num-detected-classes`**（那是 detector 专属；实测删掉无影响）。
- 官方是多类(6类)分类器 + `classifier-async-mode=1` + tracker（test5 resnet_tracker_sgie）。
- 我们的 2 类 softmax 场景里 class0-only 是实测 quirk，官方多类样例逻辑上应全类可用。

### 9.5 探针/probe 定位
- probe `_classifier_status` 读法正确：`classifier_items` 一次性迭代 + `unique_component_id` 过滤
  + `get_n_label(i)`。class1 是源头就没生成，**不是探针读漏**。
- pyservicemaker 的 ClassifierMetadata 只暴露 `unique_component_id/n_labels/get_n_label(i)`，
  **拿不到 result_prob/result_class_id**（无置信度）→ 告警置信度只能回退 classifier-threshold。
- 想要真实概率/两态，需 C 侧：pyds 直读 NvDsLabelInfo，或自定义 classifier parser 输出概率。
