# 告警状态机设计（DeepStream 版）

> 目标：复用 `server/alert/manager.py` 现有状态机，**逻辑零改动**。
> 检测侧从 ultralytics 切到 DeepStream（person / helmet / vest 三模型整帧推理），
> 数据契约通过 `server/metadata.py` 的 `ObjectMeta` / `AttributeMeta` 桥接。

## 已拍板的决策

| 决策点 | 结论 | 对状态机的影响 |
|--------|------|----------------|
| 触发条件 | **任一违规即触发**：`target_classes: ['no_helmet', 'no_vest']`，OR 匹配 | 零改动（现有 `_find_alert_objects` OR 语义直接命中） |
| 告警粒度 | **摄像头级**：每摄像头冷却期内最多一条告警，不区分具体的人 | 零改动（每摄像头一个 `AlertManager` 实例） |
| 状态机 | `AlertManager` + `WebhookAlerter` 原样保留 | — |
| 预览 | **删除 MJPEG**（`server/preview/` 已挪 tmp），用 DeepStream 内置 RTSP 预览 | 无 |
| debug_render | **删除**（历史遗留，已从 `AlertManager`/config 清理）；attributes **始终**进 payload | 已删 `debug_render` 参数与门控 |
| 证据帧 | **只画人物框 + 违规标签**（如 `no_vest`），不渲染全部检测 | snapshot 语义改为「证据帧快照」，渲染在 executor 线程做 |

## 数据流

```
DeepStream 流水线（GStreamer 流线程）
  nvurisrcbin×N → nvstreammux(batch=N)
    → nvinfer(pgie: person, uid=1)
    → nvinfer(helmet, uid=3)
    → nvinfer(vest, uid=4)
    → [nvtracker]（可选，摄像头级暂不加）
    → [探针]（逐人状态 → ObjectMeta → 路由到 AlertManager，纯决策）
    → tee ──→ nvosd → nvrtspoutsinkbin（RTSP 预览：渲染全部检测框）
         └──→ nvstreamdemux → appsink×N（每路缓存最新**原始帧** = 证据帧源）

证据帧（告警触发时，executor 线程）
  读该路最新原始帧 → 用违规人框 + 违规标签（no_vest / no_helmet）自定义渲染
  → simplejpeg 编码 → 进现有 payload → webhook fire-and-forget（不阻塞流线程）
```

## 接口契约

### 1. 探针 → 翻译层（新增代码，唯一核心改动点）

每路 source 每帧产出 `list[ObjectMeta]`：

```python
ObjectMeta(
    class_name='person',
    confidence=person_conf,
    bbox=(x1, y1, x2, y2),
    attributes=(
        AttributeMeta('no_helmet', conf, bbox=None),   # 仅违规时出现
        AttributeMeta('no_vest',   conf, bbox=None),   # 仅违规时出现
        # 达标人员可带正向属性（helmet/vest），不参与告警匹配
    ),
)
```

**翻译规则**（对 simple-demo3 `SafetyMarker` 的逐人状态）：

| person 状态 | 翻译结果 | 是否触发告警 |
|-------------|----------|:-----------:|
| `no_helmet` | `AttributeMeta('no_helmet', conf)` | ✅（命中 target_classes） |
| `no_vest` | `AttributeMeta('no_vest', conf)` | ✅ |
| `helmet` / `vest`（达标） | 正向属性（可选保留） | ❌ |
| `unknown`（未关联到任何框） | 不产出违规属性 | ❌（宁漏不误报，后续可加配置） |

- 同时违规时同时带 `no_helmet` + `no_vest` 两个属性。
- 违反属性 confidence = 关联上的 head/no_vest 框的置信度。

### 2. 路由

`alert_managers: dict[str, AlertManager]`，key = `frame_meta.source_id` 对应的 camera id。
探针内**只做翻译 + 调 `handle()`**（纯决策，轻量），不做任何网络 I/O。

### 3. 证据帧（唯一结构性新增）

- 旧设计：预览线程缓存最新渲染帧，告警时 `snapshot()` 懒读。
- 新设计：nvosd **前** `tee → nvstreamdemux → appsink×N`，每路 appsink 缓存**最新原始帧**（drop-oldest）。
- 告警触发时，executor 线程在原始帧上**自定义渲染**：只画人物框，框上标违规类别（`no_vest` / `no_helmet`），
  不渲染 helmet/vest/head 等全部检测 → `simplejpeg` 编码 → 进现有 payload。
- 渲染函数是新增小工具（旧 `utils/draw.py::_draw_alert` 的简化版，原文件在 `tmp/` 可参考）。
- 与旧设计「最新帧胜出」语义一致；`manager.py` **决策逻辑不变**，`handle()` 的 `snapshot` 语义改为「证据帧快照」。

### 4. Webhook payload

保持 `AlertManager` 现有格式：`camera_id / camera_name / timestamp / objects / frame_base64`。
`objects[].class='person'`，attributes（`no_helmet` / `no_vest` 违规明细）**始终进 payload**
（已删 `debug_render` 门控）。

## 配置（新服务 config.yaml）

```yaml
model:
  pgie_config: pgie_config_yolo26n.txt
  helmet_config: pgie_config_helmet.txt
  vest_config: pgie_config_vest.txt

cameras:            # → nvurisrcbin 的 sources
  - id: "1384"
    name: "Mobile Camera 1384"
    rtsp_url: "rtsp://..."
    enabled: true

alert:
  target_classes: ['no_helmet', 'no_vest']
  cooldown_seconds: 10
  min_detection_count: 3
  webhook: { url, timeout, retries }

# 预览走 DeepStream 内置 RTSP out（nvrtspoutsinkbin），不再有 preview 节
```

## 后续任务

- [x] 删 MJPEG 预览（`server/preview/` → tmp）
- [x] 删 `debug_render`（AlertManager + 两份 config）
- [x] attributes 始终进 payload
- [ ] 服务入口 + 探针翻译层（task #3）
- [ ] 证据帧捕获：原始帧缓存 + 人物框/违规标签渲染（task #4）
- [ ] deploy 切 DeepStream 容器、config 对齐
- [ ] 人级别告警（nvtracker + track_id）——v2，本期不做
