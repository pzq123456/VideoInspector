# 抽烟检测服务端

基于 YOLO 的两阶段抽烟行为实时检测后台服务。先检测人体（COCO 预训练模型），再在人体区域内判定是否抽烟，大幅降低误报。支持单阶段回退模式、多路 RTSP / 本地摄像头接入，检测到目标后标注帧并通过 Webhook 推送告警（含 base64 证据图片）。

## 目录结构

```
smoke/
├── server/
│   ├── main.py                 # 入口
│   ├── config.yaml             # 配置文件（修改此处即可，无需改代码）
│   ├── core/
│   │   ├── streamer.py         # 视频流读取器（RTSP + 本地摄像头）
│   │   ├── detector.py         # YOLO 模型封装（支持单阶段 / 两阶段切换）
│   │   └── camera_worker.py    # 单路摄像头 Worker 线程
│   ├── alert/
│   │   ├── webhook.py          # Webhook HTTP POST 推送
│   │   └── manager.py          # 告警管理（冷却 + 连续帧确认 + 帧标注编码）
│   ├── utils/
│   │   └── logger.py           # loguru 日志配置
│   └── README.md
├── local/
│   └── webhook_receiver.py     # Webhook 接收器（模拟第三方消费端）
└── pyproject.toml
```

## 快速开始

```bash
# 安装依赖（uv）
uv sync

# 终端 1：启动 Webhook 接收器（模拟消费端，保存证据帧 + 记录数据）
python local/webhook_receiver.py

# 终端 2：启动检测服务
python -m server.main

# 指定配置文件
python -m server.main --config my_config.yaml
```

## 配置文件参考 (config.yaml)

### model — 模型参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | **必填** | 抽烟模型权重路径，相对于项目根目录（第二阶段） |
| `person_model` | object | 可选 | **人体检测模型配置（第一阶段）**。不配置则回退为单阶段全帧检测模式 |
| `person_model.path` | string | 必填 | 人体检测模型路径（COCO 预训练，class 0=person），如 `yolo26n.pt` |
| `person_model.conf` | float | `0.4` | 人体检测置信度阈值 |
| `conf` | float/dict | `0.35` | 抽烟检测置信度阈值。float 时全局统一；dict 时逐类设定（如 `{smoking: 0.35}`）。YOLO 内部使用最低值保证召回率，后置逐类提纯 |
| `device` | int/string/null | `0` | GPU 设备 ID；`0`=第一块GPU；`null`或`"cpu"`=CPU推理 |
| `target_classes` | list | `["smoking"]` | 触发告警的目标类别名，必须与抽烟模型类别名一致。两阶段模式下仅需 `['smoking']` |

**检测模式说明：**

- **两阶段（推荐）**：配置 `person_model` 后启用。先检测人体 → 裁剪人体 ROI → 在 ROI 内检测抽烟。仅人体附近的烟蒂才会被识别，显著降低误报。
- **单阶段（回退）**：不配置 `person_model`。直接在全帧上检测抽烟目标。适合已有低误报模型的场景，或需要检测无人场景中烟雾的场景。

**完整配置示例：**

```yaml
model:
  path: "runs/detect/yolo26s_smoking_20260625_0033/weights/best.pt"
  person_model:                         # 可选：注释掉或删除此块即回退单阶段
    path: "yolo26n.pt"
    conf: 0.4
  conf:
    smoking: 0.35
  device: 0
  target_classes: ['smoking']
```

### cameras — 摄像头列表

每路摄像头是一个数组元素：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | string | **必填** | 唯一标识，用于告警 payload |
| `name` | string | **必填** | 显示名称，用于日志和告警 |
| `type` | string | `"rtsp"` | 摄像头类型：`rtsp`（RTSP网络流）或 `local`（本地USB/内置摄像头） |
| `enabled` | bool | `true` | 是否启用该摄像头 |

**type=rtsp 时额外字段：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `rtsp_url` | string | **必填** | RTSP 流地址 |

**type=local 时额外字段：**

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `device_id` | int | `0` | OpenCV 摄像头设备 ID，`0`=默认摄像头 |

**示例：**

```yaml
cameras:
  # RTSP 网络摄像头
  - id: "gate"
    name: "工厂大门"
    type: "rtsp"
    rtsp_url: "rtsp://192.168.1.100:554/stream"
    enabled: true

  # 本地 USB 摄像头
  - id: "local_test"
    name: "本地测试"
    type: "local"
    device_id: 0
    enabled: true
```

### alert — 告警参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cooldown_seconds` | float | `30` | 同一摄像头两次告警的最小间隔（秒）。防止同一根烟产生几十条告警 |
| `min_detection_count` | int | `3` | 连续检测到目标的帧数阈值。只有连续 N 帧都检测到才触发告警，防止单帧误报 |
| `save_frame_overlay` | bool | `false` | 是否在证据帧上叠加摄像头名称/时间水印。摄像头流已自带则无需开启 |

#### alert.webhook — Webhook 推送

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | string | `null` | Webhook 接收地址。为 null 时不推送 |
| `timeout` | float | `10` | 单次请求超时（秒） |
| `retries` | int | `2` | 失败后重试次数（不含首次） |

### log — 日志参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | 日志级别：`DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `file` | string | `"logs/server.log"` | 日志文件路径（JSON 格式，自动轮转 10MB，保留 30 天，旧文件 gz 压缩） |

日志由 **loguru** 管理，支持：
- 控制台彩色输出（开发友好）
- 文件 JSON 结构化记录（方便检索分析）
- 自动轮转 + 压缩 + 过期清理

## 检测流程

```
读取帧 → [人体检测（COCO）] → [裁剪人体 ROI] → [抽烟检测（ROI 内）]
                                                    ↓
                                              检测到 smoking？
                                                    ↓ 是
                                              连续帧计数器 +1
                                                    ↓
                                              达到 min_detection_count？
                                                    ↓ 是
                                              冷却期已过？
                                                    ↓ 是
                                              🚨 触发告警
                                              ├── 标注帧（bbox + 可选水印）
                                              ├── JPEG 编码 → base64
                                              └── 构建 payload → Webhook POST
                                                                   ↓
                                                         接收端消费
                                                         ├── 解码 base64 → 写入磁盘
                                                         └── 结构化记录推送数据
```

> 注：未配置 `person_model` 时，跳过人体检测和 ROI 裁剪步骤，直接在全帧上执行抽烟检测。

**关键设计：**
- **连续帧确认**：只有连续 `min_detection_count` 帧（默认 3 帧）都检测到才触发，杜绝单帧噪点误报
- **冷却期**：同一摄像头 `cooldown_seconds` 秒内（默认 30 秒）只告警一次，避免告警风暴
- **计数器递减**：没有检测到目标时，连续计数器逐步递减（而非直接清零），容忍偶尔丢帧
- **base64 证据帧**：标注后的 JPEG 直接编码进 payload，接收端无需访问检测端文件系统
- **检测端不写磁盘**：帧保存由消费端负责，检测端只推送

## Webhook Payload 格式

```json
{
  "camera_id": "gate",
  "camera_name": "测试",
  "timestamp": "2026-06-27T09:30:00+00:00",
  "detections": [
    {
      "class": "cigarette",
      "confidence": 0.89,
      "bbox": [320, 240, 400, 380]
    }
  ],
  "frame_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

## 测试工具

### Webhook 接收器

`local/webhook_receiver.py` 模拟第三方消费端：

```bash
# 终端1：启动接收器（默认 0.0.0.0:9999，帧保存到 alerts/）
python local/webhook_receiver.py

# 终端2：启动检测服务（config.yaml 中 webhook.url 设为 http://localhost:9999/api/smoke-alert）
python -m server.main
```

参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` / `-p` | `9999` | 监听端口 |
| `--save-dir` | `alerts` | 证据帧保存目录 |

接收端文件输出结构：

```
alerts/
├── payload.jsonl              # 结构化 payload 日志（JSON 格式，50MB 轮转，保留 30 天）
└── gate/
    ├── 20260627_093000.jpg    # 解码后的证据帧
    └── 20260627_093045.jpg
```

### 本地摄像头测试

修改 `config.yaml`，启用本地摄像头配置段即可在不连接 RTSP 流的情况下测试。

## 日志级别说明

| 级别 | 内容 | 频率 |
|------|------|------|
| INFO | 启动/停止、告警触发、Webhook 推送、每 60 秒运行摘要 | 低频 |
| DEBUG | 每 100 帧的详细统计（FPS、推理延迟）、检测命中 | 高频 |
| WARNING | RTSP 断线重连、帧读取失败 | 按需 |
| ERROR | 检测异常、Webhook 推送失败、Worker 意外退出 | 按需 |

生产环境推荐使用 `INFO` 级别，调试时使用 `DEBUG`。

## 稳定性设计

### 多层异常防护

每个线程都有顶层 try/except 安全网，确保单个异常不会杀死线程：

```
streamer._update_loop()     ← 顶层 try/except，异常后 1s 恢复
camera_worker._run()        ← 顶层 try/except，异常后跳过当前帧继续
alert_manager.handle()      ← 顶层 try/except，异常后返回 False
webhook.send()              ← 内部重试 + 异常捕获
```

### RTSP 断线重连

RTSP 流断开后自动重连，使用指数退避策略（初始 2 秒，每次失败翻倍，最大 60 秒），连接成功后重置延迟。

### Worker 健康监控

主线程每 30 秒检查所有 Worker 线程是否存活，发现死亡立即记录 ERROR 日志。

### 优雅退出

Ctrl+C → 主线程捕获 KeyboardInterrupt → 依次 stop 所有 Worker → 释放资源 → 退出。不使用 signal 模块（Windows 下与 GPU 线程交互存在已知可靠性问题）。

### Snap Tip：多摄像头启动时的竞态误报

Ultralytics 的首次 `predict()` 会触发 lazy warmup + fuse，这个过程**非线程安全**。多路摄像头同时启动时，共享模型的 Worker 线程可能撞出 `'Conv' object has no attribute 'bn'` 错误。

不影响业务——仅启动瞬间丢几帧，之后完全正常。根治只需在 `SmokeDetector.__init__()` 末尾加 dummy 预热：

```python
import numpy as np
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
self._model.predict(dummy, verbose=False, device=self._device)
if self._person_model is not None:
    self._person_model.predict(dummy, verbose=False, device=self._device)
```
