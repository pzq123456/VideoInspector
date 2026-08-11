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
│   ├── probe.py            # SafetyProbe：三模型检测 → 空间关联 → 违规属性 → AlertManager
│   └── frame_cache.py      # 证据帧采集：tee → appsink 缓存最新原始帧（FrameCache/Retriever）
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
RTSP 源 → nvstreammux → nvinfer(pgie: yolo26n 人体检测, uid=1)
                       → nvinfer(helmet: head/helmet 安全帽检测, uid=3)
                       → nvinfer(vest: vest/no_vest 反光衣检测, uid=4)
                       → tee → [nvosdbin → RTSP 输出 | fakesink]
                          └── queue → nvvideoconvert → appsink（证据帧缓存）
                              ↑
                    SafetyProbe（挂在 vest 后）
```

三个模型都是**整帧检测器**（process-mode=1），对同一帧各自独立推理。探针做
**空间关联**（helmet/vest 检测框**中心点**落在 person 框内 → 归属该人），
把违规翻译成 `AttributeMeta('no_helmet')` / `AttributeMeta('no_vest')`
附加到对应 person 的 `ObjectMeta`，按 `source_id` 喂给对应摄像头的
`AlertManager`。模型 / 引擎 / 自定义 parser 全部复用 `simple-demo3/` 产物。

## 配置文件参考 (config.yaml)

### model — 模型参数

| 字段 | 类型 | 说明 |
|------|------|------|
| `pgie_config` | string | 人体检测（yolo26n）nvinfer 配置文件路径 |
| `helmet_config` | string | 安全帽整帧检测 nvinfer 配置文件路径 |
| `vest_config` | string | 反光衣整帧检测 nvinfer 配置文件路径 |

路径相对项目根目录解析；配置文件内引用 engine/parser 的绝对路径，
默认指向 `simple-demo3/models/` 已构建产物。

### cameras — 摄像头列表

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 唯一标识，用于告警 payload |
| `name` | string | 显示名称 |
| `type` | string | 仅支持 `rtsp` |
| `rtsp_url` | string | RTSP 流地址 |
| `enabled` | bool | 是否启用 |

> **多路限制**：当前三个引擎均为 batch=1，仅支持 **1 路** RTSP 输入。
> 多路需用 `batch=N` 重建引擎（`trtexec --minShapes ... --optShapes ...`），
> 然后把 mux/nvinfer 的 batch-size 设成 N。

### alert — 告警参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_classes` | list | `['no_helmet','no_vest']` | 触发告警的违规属性名（匹配 `ObjectMeta.attributes`） |
| `cooldown_seconds` | float | `30` | 同一摄像头两次告警的最小间隔（秒） |
| `min_detection_count` | int | `3` | 连续检测到违规的帧数阈值（防止单帧误报） |
| `save_frame_overlay` | bool | `false` | 是否在证据帧上叠加摄像头名称/时间水印（暂无证据帧，保留字段） |

#### alert.webhook — Webhook 推送

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | string | `null` | Webhook 接收地址。为 null 时不推送 |
| `timeout` | float | `10` | 单次请求超时（秒） |
| `retries` | int | `2` | 失败后重试次数（不含首次） |

### output — RTSP 输出（可选，替代原 MJPEG 预览）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rtsp_port` | int | `18003` | 输出流监听端口 |
| `mount_point` | string | `/vest` | 播放地址 `rtsp://localhost:18003/vest` |
| `codec` | string | `h264` | `h264` / `h265` |
| `bitrate` | int | `4000000` | 编码码率 (bps) |
| `idrinterval` | int | `30` | 关键帧间隔（帧） |

注释掉整个 `output` 节即关闭 RTSP 输出（改用 fakesink 丢弃，检测/告警不受影响）。

### log — 日志参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `file` | string | `"logs/server.log"` | 日志文件路径（JSON 格式，自动轮转 10MB） |

## 检测流程

```
读取帧 → [person 检测] + [helmet 检测] + [vest 检测]（同一帧并行推理）
                                              ↓
                    探针空间关联 → 逐人 helmet/vest 状态
                                              ↓
                            有违规（no_helmet 或 no_vest）？
                                              ↓ 是
                                     连续帧计数器 +1
                                              ↓
                                 达到 min_detection_count？
                                              ↓ 是
                                 冷却期已过？
                                              ↓ 是
                                 🚨 触发告警
                                 ├── 从 FrameCache 取该路最新原始帧（executor 线程画人物框 + 违规标签）
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

- `frame_base64` 为告警时的 **JPEG 证据帧**（BGR → RGB 后由 nvvideoconvert 采集，
  在 executor 线程画触发告警的人物框 + 违规标签后 `simplejpeg` 编码）。
  无证据帧时仍为 `null`（缓存尚未就绪，告警不丢）。

## 关键设计

- **复用 `AlertManager`**：冷却 / 连续帧确认 / 异步 webhook 决策逻辑保持不变，
  仅扩展为允许 `snapshot=None` 时仍推送（`frame_base64=null`），避免无证据帧时丢告警。
- **探针不阻塞流线程**：探针只做轻量决策，JPEG / base64 / HTTP 由
  executor 线程池 + daemon 线程 fire-and-forget。
- **每路摄像头独立状态机**：`dict[source_id → AlertManager]`，`source_id`
  即 `nvstreammux` 的 pad 序号。

## 稳定性说明

- Ctrl+C → 子进程终止，干净退出。
- Webhook 失败自动重试（`retries` 次），异常不阻塞检测管线。
- RTSP 断线由 `nvurisrcbin` 内部处理。

## 已知限制 / 后续项

- **仅 1 路 RTSP**：引擎 batch=1，多路需重建引擎。证据帧分支当前是单 appsink
  （缓存键=source_id）；多路需 `tee` → `nvstreamdemux` → `appsink×N` 再各自缓存。
- **证据帧有约 1 帧滞后**：告警快照取的是缓存中的最新帧（探针在 vest、appsink
  在下游，存在流水线时序差），作为违规瞬间的证据可接受；如需严格帧同步可改为在
  appsink 侧用 `buffer.batch_meta` 直接判定。
- **RTSP 预览不带违规着色框**：探针专注告警桥接，未给对象上色（demo 才有）。
