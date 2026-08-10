# simple-demo3 — 三模型整帧检测流水线（人 + 安全帽 + 反光衣）

在一帧上同时跑三个整帧检测模型，输出逐人「戴帽 + 穿反光衣」状态，并每帧产生
结构化 JSON 供业务对接。

## 流水线

```
RTSP 源 → nvstreammux → nvinfer(pgie: yolo26n 人体检测, uid=1)
                        → nvinfer(helmet: head/helmet 安全帽检测, uid=3)
                        → nvinfer(vest: vest/no_vest 反光衣检测, uid=4)
                        → nvosdbin → RTSP 输出
```

三个模型都是 **整帧检测器**（process-mode=1），对同一帧各自独立推理，结果都挂在
帧级，探针用 **框中心落在 person 框内** 做空间关联，判定每个人的戴帽 / 反光衣状态。

> 模型输出格式统一：`[1,3,640,640] → output0 [1,300,6]`（end2end，NMS 已烧入模型），
> 解析器均为自定义 bbox parser，`cluster-mode=4`（不二次 NMS）。

## 模型

| 阶段 | 模型 | 输出类 | gie-unique-id |
|---|---|---|---|
| person | `models/person/yolo26n`（COCO，只留 person） | `person` | 1 |
| helmet | `models/helmet/best` | `head`(0) / `helmet`(1) | 3 |
| vest | `models/vest/vest` | `vest`(0) / `no_vest`(1) | 4 |

引擎均为 FP16、batch=1，预构建于 `models/<model>/`。原始 ONNX 只在引擎重建时需要
（helmet 在 `simple-demo2/models/helmet/`，vest 在仓库根 `models/vest.onnx`）。

## 配置

- `configs/rtsp_in.yaml` — 输入源 / 输出 / 三个模型配置（切换模型只改这里）
- `configs/pgie_config_yolo26n.txt`、`pgie_config_helmet.txt`、`pgie_config_vest.txt`

## 运行

```bash
./run.sh                                  # RTSP in → RTSP out（读 configs/rtsp_in.yaml）
python3 two_stage_demo.py --file <视频>    # 本地文件调试，存 output/frame_*.jpg
```

输出：
- `output/run.log` — 运行日志（逐帧统计 + 帧率）
- `output/structured.jsonl` — 结构化 JSON（业务数据契约，见下）
- RTSP 流 — 默认 `rtsp://localhost:18003/vest`

## 渲染约定

- 红框：`no_vest` 或 `no_helmet`（不达标）
- 绿框：`vest` 且 `helmet`（双达标）
- 蓝框：有维度未关联上
- helmet 框本身：红=`head`（未戴）、绿=`helmet`；vest 框：红=`no_vest`、绿=`vest`

## 结构化数据契约（每帧一条 JSON 行 → structured.jsonl）

```json
{
  "stream": 0,
  "frame": 1234,
  "time_ms": 1723456789123,
  "persons": [
    {
      "bbox": [840, 210, 1020, 760],
      "conf": 0.87,
      "helmet": "helmet",
      "vest": "no_vest",
      "violation": true
    }
  ],
  "counts": {
    "persons": 3,
    "vest": 1,
    "no_vest": 2,
    "helmet": 2,
    "no_helmet": 1,
    "unknown": 0
  }
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `stream` | int | 输入流序号（多路时区分） |
| `frame` | int | 帧号 |
| `time_ms` | int\|null | 帧时间戳（ms）；接口未暴露时为 `null` |
| `persons[].bbox` | [int×4] | 原图像素坐标 `[left, top, right, bottom]` |
| `persons[].conf` | float | person 检测置信度 |
| `persons[].helmet` | enum | `helmet` / `no_helmet` / `unknown` |
| `persons[].vest` | enum | `vest` / `no_vest` / `unknown` |
| `persons[].violation` | bool | 任一不达标（`no_vest` 或 `no_helmet`）→ `true` |
| `counts` | object | 帧级各状态人数统计 |

判定规则：
- 空间关联：helmet / vest 检测框**中心点**落在 person 框内，且置信度 ≥ 0.5，即归属该人
- `helmet` 优先于 `head`（同时命中按已戴帽处理）
- `no_vest` 优先于 `vest`（同时命中按未穿处理，安全告警倾向宁可多报）
- 均未命中 → `unknown`

后续对接 Kafka 等消息中间件时，可直接把该 JSON 作为消息体（`nvmsgconv` 换 schema
或自写 producer 均可），探针里的 JSON 组装是唯一需要改动的位置。
