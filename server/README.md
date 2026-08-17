# 安全帽 / 反光衣检测服务端（DeepStream）

基于 DeepStream 的三模型整帧检测实时后台服务：对每路 RTSP 摄像头同时跑
**人体 / 安全帽 / 反光衣** 三个检测模型，逐人判定「戴帽 + 穿反光衣」状态，
检测到违规（`no_helmet` / `no_vest`）后通过 Webhook 推送告警（含 bbox 与
违规属性）。

推理全部由 DeepStream（`nvinfer` + `nvstreammux` batch + 探针空间关联）完成，
替代旧版 ultralytics 抽烟检测管线。

## 目录结构

```
server/
├── main.py                 # 入口：加载配置 → 构建 DeepStream pipeline → 运行
├── config.yaml             # 配置文件（修改此处即可，无需改代码）
├── metadata.py             # 数据契约：ObjectMeta / AttributeMeta（探针 → 状态机桥）
├── pipeline/
│   ├── probe.py            # SafetyProbe：三模型检测 → 空间关联 → 违规属性 + nvdsosd 上色 → AlertManager
│   ├── frame_cache.py      # 证据帧采集：每路 tee → appsink 缓存最新已渲染帧（FrameCache/Retriever）
│   └── rtsp_server.py      # 单端口 RTSP 输出（GstRtspServer，/cam/{id} 多挂载点）
├── alert/
│   ├── manager.py          # 告警管理（冷却 + 连续帧确认 + 异步 webhook 推送）
│   └── webhook.py          # Webhook HTTP POST 推送
└── utils/
    └── logger.py           # loguru 日志配置
```

## 快速开始

```bash
# 1. 启动 Webhook 接收器（模拟消费端，保存证据帧 + 记录数据）
python local/webhook_receiver.py

# 2. 启动检测服务
python -m server.main

# 指定配置文件
python -m server.main --config my_config.yaml
```

## 推理流水线

```
RTSP 源×N → nvstreammux(batch=N) → nvinfer(pgie: yolo26n 人体检测, uid=1)
                                   → nvinfer(helmet: head/helmet 安全帽检测, uid=3)
                                   → nvinfer(vest: vest/no_vest 反光衣检测, uid=4)
                                   → nvstreamdemux → 每路:
                                        nvdsosd → tee → [ shmsink(→ 单端口 RTSP server) | appsink(证据帧缓存) ]
                                            ↑
                                  SafetyProbe（挂在 vest 后，上色 + 桥接）
```

三个模型都是**整帧检测器**（process-mode=1），对同一帧各自独立推理。探针做
**空间关联**（helmet/vest 检测框**中心点**落在 person 框内 → 归属该人），
把违规翻译成 `AttributeMeta('no_helmet')` / `AttributeMeta('no_vest')`
附加到对应 person 的 `ObjectMeta`，按 `source_id` 喂给对应摄像头的
`AlertManager`。模型 / 引擎 / 自定义 parser 全部复用根目录 `models/` 产物。

## 配置文件参考 (config.yaml)

### model — 模型参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `pgie_config` | string | 人体检测（yolo26n）nvinfer 配置文件路径 |
| `helmet_config` | string | 安全帽整帧检测 nvinfer 配置文件路径 |
| `vest_config` | string | 反光衣整帧检测 nvinfer 配置文件路径 |

路径相对项目根目录解析；INI 配置内部引用的模型路径同样相对根目录
（`models/` / `configs/`），启动时由 `server/main.py` 统一锚定为绝对路径，
开发容器与生产镜像（`/app`）通用。

### cameras — 摄像头列表

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识，用于告警 payload |
| `name` | string | 显示名称 |
| `type` | string | 仅支持 `rtsp` |
| `rtsp_url` | string | RTSP 流地址 |
| `enabled` | bool | 是否启用 |

> **多路支持**：引擎为动态 batch（1~12），mux/nvinfer 的 `batch-size` 自动取
> `len(cameras)`。每路摄像头独立 AlertManager + 独立 OSD 渲染（demux 后每路独立，
> 杜绝跨流污染）+ 独立证据帧缓存。

### alert — 告警参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_classes` | list | `['no_helmet','no_vest']` | 触发告警的违规属性名（匹配 `ObjectMeta.attributes`） |
| `cooldown_seconds` | float | `30` | 同一摄像头两次告警的最小间隔（秒） |
| `min_detection_count` | int | `3` | 连续检测到违规的帧数阈值（防止单帧误报） |
| `save_frame_overlay` | bool | `false` | 是否在证据帧上叠加摄像头名称/时间水印（暂无证据帧，保留字段） |
| `helmet_conf_threshold` | float | `0.5` | 头盔框空间关联 person 的最低置信度（须 ≥ INI 的 `pre-cluster-threshold=0.25`） |
| `vest_conf_threshold` | float | `0.5` | 反光衣框空间关联 person 的最低置信度（须 ≥ INI 的 `pre-cluster-threshold=0.25`） |

#### alert.webhook — Webhook 推送

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | string | `null` | Webhook 接收地址。为 null 时不推送 |
| `timeout` | float | `10` | 单次请求超时（秒） |
| `retries` | int | `2` | 失败后重试次数（不含首次） |

### output — RTSP 输出（可选，替代原 MJPEG 预览）

单端口 RTSP 服务：所有摄像头共用 `rtsp_port`，通过挂载点路径区分。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rtsp_port` | int | `8554` | 单一 RTSP 服务端口（所有摄像头共用） |
| `mount_prefix` | string | `/cam` | 挂载点前缀，每路 = `/cam/<camera_id>` |
| `codec` | string | `h264` | `h264` / `h265` |
| `bitrate` | int | `4000000` | 编码码率 (bps) |
| `idrinterval` | int | `30` | 关键帧间隔（帧） |

示例（6 路）：`rtsp://localhost:8554/cam/1363`、`rtsp://localhost:8554/cam/1384` … 依此类推。
单端口由 `GstRtspServer`（PyGObject）承载，每路通过 shm 桥接（`shmsink` → `shmsrc`）。

注释掉整个 `output` 节即关闭 RTSP 输出（检测/告警仍照常）。

### log — 日志参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `file` | string | `"logs/server.log"` | 日志文件路径（JSON 格式，自动轮转 10MB） |

## 检测流程

```
读取帧 → [person 检测] + [helmet 检测] + [vest 检测]（同一帧并行推理）
                                              ↓
                    探针空间关联 → 逐人 helmet/vest 状态 → nvdsosd 上色
                                              ↓
                            有违规（no_helmet 或 no_vest）？
                                              ↓ 是（进入 ARMING）
                                     连续帧计数器 +1
                                              ↓
                                 达到 min_detection_count？
                                              ↓ 是
                                 🚨 触发告警（进入 COOLDOWN）
                                 ├── 从 FrameCache 取该路最新已渲染帧（executor 线程 JPEG 编码）
                                 ├── 构建 payload（bbox + 违规属性 + frame_base64）
                                 └── Webhook POST（带 JPEG 证据帧）
```

## Webhook Payload 格式

```json
{
  "camera_id": "1363",
  "camera_name": "Mobile Camera 1363",
  "timestamp": "2026-08-10T09:30:00+00:00",
  "objects": [
    {
      "class": "person",
      "confidence": 0.87,
      "bbox": [840, 210, 1020, 760],
      "attributes": [
        { "class": "no_helmet", "confidence": 0.91, "bbox": null },
        { "class": "no_vest",   "confidence": 0.88, "bbox": null }
      ]
    }
  ],
  "frame_base64": null
}
```

- `frame_base64` 为告警时的 **JPEG 证据帧**（nvdsosd 已原生渲染检测框 + 违规标签，
  由 nvvideoconvert 采集后 executor 线程 `simplejpeg` 编码）。
  无证据帧时仍为 `null`（缓存尚未就绪，告警不丢）。

## 关键设计

- **复用 `AlertManager`**：冷却 / 连续帧确认 / 异步 webhook 决策逻辑重构为显式状态机
  （`IDLE`/`ARMING`/`COOLDOWN`），仅扩展为允许 `snapshot=None` 时仍推送
  （`frame_base64=null`），避免无证据帧时丢告警。
- **探针不阻塞流线程**：探针只做轻量决策 + nvdsosd 上色，JPEG / base64 / HTTP 由
  executor 线程池 + daemon 线程 fire-and-forget。
- **每路摄像头独立状态机**：`dict[source_id → AlertManager]`，`source_id`
  即 `nvstreammux` 的 pad 序号。
- **证据帧 = 实时预览帧**：每路在 `nvstreamdemux` 后独立 `nvdsosd` 渲染，
  实时预览与证据帧共享同一渲染源，保证证据帧与操作者所见一致
  （违规红 / 合规绿 / 未知蓝），且彻底隔离跨流污染。
- **单端口 RTSP 输出**：所有摄像头共用一个 `GstRtspServer` 端口，
  `/cam/{camera_id}` 路径区分，客户端配置统一。

## 稳定性说明

- Ctrl+C → 子进程终止，干净退出。
- Webhook 失败自动重试（`retries` 次），异常不阻塞检测管线。
- RTSP 断线由 `nvurisrcbin` 内部处理。
- 单端口 RTSP 输出依赖 PyGObject（`python3-gi`）+ `gir1.2-gst-rtsp-server-1.0`
  （容器已装）；未启用 `output` 节时无此依赖。

## 已知限制 / 后续项

- **动态 batch 上限 12**：8GB GPU 下三个模型（yolo26n + 2×25M 参检测器）同时以
  batch≤12 构建引擎；如需更多路需重出更大 maxShapes 引擎并评估显存，或降低分辨率。
- **证据帧有约 1 帧滞后**：告警快照取的是缓存中的最新帧（探针在 vest、appsink
  在下游，存在流水线时序差）。由于每帧都已由 nvdsosd 原生渲染（框与像素自洽），
  滞后帧仍是合法证据；如需严格帧同步可改为按 frame_number/pts 关联。
- 违规着色由探针（SafetyProbe）在 nvdsosd 上游上色，实时预览与证据帧同步生效。
