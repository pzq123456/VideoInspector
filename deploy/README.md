# 抽烟检测服务 — Docker 部署

## 快速开始

改变端口映射，默认内部的 preview 运行在 8080，宿主机映射到 18080，避免与其他服务冲突。
参见:

```yaml
    ports:
      - "18080:8080"            # 视频流预览端口（需与 config.yaml preview.port 一致）
    restart: unless-stopped
```

### 1. 放入模型

```bash
cp runs/detect/<训练目录>/weights/best.pt deploy/models/
```

> 不拷贝也行——`config.yaml` 中 `model.path` 直接写 `/runs/detect/.../best.pt` 即可引用训练产出，两种路径 `docker-compose.yml` 都已挂载。

### 2. 编辑配置

```bash
vim deploy/config.yaml
```

必改项：

| 配置项 | 说明 |
|--------|------|
| `model.path` | 容器内路径：拷贝到 models → `/models/best.pt`；直接引用 → `/runs/detect/.../best.pt` |
| `model.conf` | 置信度阈值。`0.35` 全局统一；逐类设定写 `face: 0.5, smoking: 0.25`（YOLO 内部用最低值保召回，后置逐类提纯） |
| `model.target_classes` | 告警目标类别。新模型为 `['face', 'smoking']`，多类时自动要求同时存在 |
| `cameras` | 摄像头列表，`type: rtsp` 填 `rtsp_url`，`type: local` 可选 `device_id` |
| `alert.webhook.url` | 告警推送地址。测试用 `http://host.docker.internal:9999/api/smoke-alert` |

完整参数参见 [配置参数速查](#配置参数速查)。

### 3. 启动

```bash
cd deploy
docker compose up -d
```

### 4. 查看日志

```bash
docker compose logs -f          # 容器日志
tail -f logs/server.log         # 持久化日志
```

---

## 运维命令

| 操作 | 命令 | 说明 |
|------|------|------|
| 改配置 | `vim config.yaml` → `docker compose restart` | 配置挂载进容器，**不需重建** |
| 换模型 | 替换 `models/` 文件 → 改 `model.path` → `docker compose restart` | **不需重建** |
| 改代码 | `docker compose build --no-cache && docker compose up -d` | Python 源码打进镜像，**必须重建** |
| 查看状态 | `docker compose ps` | |
| 停止 | `docker compose down` | |

> **判断是否重建**：改了 `server/` 下的 `.py` 文件 → 必须 rebuild。只改了 `deploy/config.yaml`、模型文件 → restart 即可。

---

## 本地测试（Windows）

Docker Desktop 在 WSL2 中无法直接访问宿主机 USB 摄像头。用 `local/camera_stream.py` 推 MJPEG 流桥接：

```bash
# 终端 1：摄像头推流（宿主机）
PYTHONIOENCODING=utf-8 uv run python local/camera_stream.py

# 终端 2：Webhook 接收器（宿主机）
PYTHONIOENCODING=utf-8 uv run python local/webhook_receiver.py

# 终端 3：Docker 检测服务
cd deploy && docker compose up -d && docker compose logs -f

# 测试完毕
docker compose down
# Ctrl+C 停止推流和接收器
```

`deploy/config.yaml` 中摄像头指向宿主机：
```yaml
cameras:
  - id: "test"
    name: "本地测试"
    type: "rtsp"
    rtsp_url: "http://host.docker.internal:8080/stream"
```

---

## 配置参数速查

### model — 模型参数

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | **必填** | 模型权重在容器内的路径 |
| `conf` | float/dict | `0.35` | float 全局统一；dict 逐类设定（`{face: 0.5, smoking: 0.25}`）。dict 时 YOLO 用最低值保召回，后置逐类提纯 |
| `device` | int/string | `0` | GPU ID；`"cpu"` 用 CPU |
| `target_classes` | list | `["cigarette"]` | 告警目标类别。多类时自动 AND（需同时存在），单类时 OR；也可显式设 `alert.require_all_targets` 覆盖 |

### cameras — 摄像头列表

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `id` | string | **必填** | 唯一标识 |
| `name` | string | **必填** | 显示名称 |
| `type` | string | `"rtsp"` | `rtsp`（网络流/MJPEG）或 `local`（USB 摄像头） |
| `rtsp_url` | string | type=rtsp 必填 | 流地址 |
| `device_id` | int | `0` | type=local 时的 OpenCV 设备 ID |
| `enabled` | bool | `true` | 是否启用 |

### alert — 告警控制

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `require_all_targets` | bool | 自动 | 不填时自动推断（多类 AND，单类 OR）；显式指定则按指定值 |
| `cooldown_seconds` | float | `10` | 同摄像头两次告警最小间隔 |
| `min_detection_count` | int | `1` | 连续命中帧数阈值（防止单帧误报） |
| `save_frame_overlay` | bool | `false` | 是否在证据帧叠加摄像头/时间水印 |

### alert.webhook — 推送目标

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `url` | string | `null` | Webhook 地址；`null` 不推送 |
| `timeout` | float | `10` | 请求超时（秒） |
| `retries` | int | `2` | 失败重试次数 |

### log — 日志

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `level` | string | `"INFO"` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `file` | string | `"/logs/server.log"` | 容器内日志路径（JSON，10MB 轮转，保留 30 天） |

### preview — 视频流预览（可选）

取消注释即启用；注释掉（默认）即关闭。无额外 `enabled` 开关。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `host` | string | `"0.0.0.0"` | 绑定地址 |
| `port` | int | `8080` | HTTP 端口（需与 docker-compose.yml 端口映射一致） |
| `jpeg_quality` | int | `30` | JPEG 编码质量 1-100 |

启用后访问 `http://<host>:<port>/stream/<camera_id>` 查看实时 MJPEG 视频流。

---

## 告警决策流程

```
帧输入
  │
  ▼
YOLO 全局初筛 (conf=min(逐类阈值))       ← 第1层：保召回
  │
  ▼
逐类后置过滤 (face≥0.5, smoking≥0.25)    ← 第2层：提精度
  │
  ▼
target_classes 同时存在? (face+smoking)   ← 第3层：类别组合
  │
  ▼
连续 min_detection_count 帧确认           ← 第4层：时序确认
  │
  ▼
距离上次告警 > cooldown_seconds?          ← 第5层：间隔控制
  │
  ▼
🚨 推送 Webhook
```

- 未命中时计数器**递减**（非清零），容忍偶尔丢帧
- 触发后计数器归零，更新冷却时间戳
- 冷却期内命中：计数器归零但不触发，避免冷却结束后立即再次告警

---

## Webhook 对接

### 推送格式

HTTP POST，Content-Type `application/json; charset=utf-8`。超时 10s，失败重试 2 次。

```json
{
  "camera_id": "gate",
  "camera_name": "工厂大门",
  "timestamp": "2026-06-27T09:30:00.123456+00:00",
  "detections": [
    {"class": "face",    "confidence": 0.92, "bbox": [200, 150, 360, 420]},
    {"class": "smoking", "confidence": 0.87, "bbox": [310, 240, 380, 310]}
  ],
  "frame_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `camera_id` | string | 对应配置 `cameras[].id` |
| `camera_name` | string | 对应配置 `cameras[].name` |
| `timestamp` | string | ISO 8601 UTC |
| `detections[].class` | string | 类别名（`face` / `smoking`） |
| `detections[].confidence` | float | 置信度 0–1 |
| `detections[].bbox` | [int×4] | `[x1, y1, x2, y2]` 像素坐标 |
| `frame_base64` | string | 标注后的证据帧 JPEG base64 |

接收端示例：`local/webhook_receiver.py`。

---

## 挂载关系

| 宿主机 | 容器内 | 读写 | 用途 |
|--------|--------|:----:|------|
| `deploy/config.yaml` | `/app/server/config.yaml` | 只读 | 配置文件 |
| `deploy/models/` | `/models/` | 只读 | 模型权重 |
| `../runs/` | `/runs/` | 只读 | 训练产出（免拷贝直接引用） |
| `deploy/logs/` | `/logs/` | 读写 | 服务日志持久化 |
