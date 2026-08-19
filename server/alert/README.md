# 告警状态机 与 Webhook 推送格式

本文档对应 `server/alert/manager.py`（状态机）+ `server/alert/webhook.py`（推送）。

每路摄像头一个独立的 `AlertManager` 实例（`main.py` 中 `dict[source_id → AlertManager]`），
内部按**激活规则**持有若干独立状态机（`_RuleState`，每规则一套
`AlertState`：`IDLE` / `ARMING` / `COOLDOWN`）。**规则之间互不竞争**——
`no_helmet` 触发冷却不会锁住 `no_vest`，各自独立累计连续违规帧。

## 1. 状态机

`handle(objects, snapshot, executor)` 每帧调用一次，内部遍历各规则状态机，
每套逻辑如下：

```
        ┌─ COOLDOWN 且未过冷却？── 是 ──→ 直接返回（不判定、不计数）
        │            │ 否（冷却结束）
        │            ▼
        │        回到 IDLE（hits 清零，重新武装）
        │
        ▼
  按属性过滤：attributes.name == 本规则名（no_helmet / no_vest / no_harness）
        │
  有违规实体？──── 否 ──→ ARMING 中 hits -= 1（缓慢衰减，容忍偶尔丢帧）
        │                       减到 0 → 回到 IDLE
        │ 是
        ▼
  进入 ARMING，hits += 1
        │
  hits >= 本规则 min_detection_count？── 否 ──→ 继续累计 ──→ 无告警
        │ 是
        ▼
  🚨 触发告警（alert_type = 本规则名）
    ├─ 记录日志（代表对象 = 置信度最高者）
    ├─ executor.submit(_build_and_send)   # 检测线程立即返回，只做决策
    └─ 进入 COOLDOWN（hits 清零，记录 cooldown_until = monotonic() + 本规则 cooldown_seconds）
```

要点：

- **按规则独立三态**：每条规则各自的 `IDLE → ARMING → COOLDOWN`，冷却时长、
  连续帧阈值、违规置信度门限均独立配置，互不竞争。
- **冷却期置顶判定**：冷却期内直接返回，不再逐帧计数/打印误导性 `hits` 日志；
  冷却结束后回到 `IDLE`，需重新连续命中 `min_detection_count` 帧才会再触发。
- **连续帧确认**：hits 是"连续多少帧**出现至少一个**本规则违规实体"的计数（按规则，
  不按人）。只有 `hits >= min_detection_count` 才触发，防止单帧误报。
- **衰减**：无违规帧时 hits 每帧 `-1`（不立即清零），容忍推理偶发丢帧。
- **单调时钟**：冷却用 `time.monotonic()`，不受系统 NTP 校时跳变影响。
- **目标匹配**：按 `AttributeMeta.name == 规则名` 过滤（探针只把违规翻译成
  `no_helmet`/`no_harness`/`no_vest` 属性挂在 person 上）。
- **以检测到人为前提**：违规实体本身已是「person + attributes」结构，且探针已在源头按
  `model.person_conf_threshold` 过滤掉低置信度 person，因此状态机只会在确实存在人的前提下累计
  （帧内无有效 person 时根本不会有违规实体进入 `handle`）。

## 2. 关键参数（来自 config.yaml 的 `rules` 节，每条规则独立）

| 字段 | 默认值 | 作用 |
|------|--------|------|
| `rules.<rule>.cooldown_seconds` | — | 该规则触发告警后冷却时长（秒），独立于其他规则 |
| `rules.<rule>.min_detection_count` | — | 该规则连续命中帧数阈值（防单帧误报） |
| `rules.<rule>.attribute_threshold` | — | 违规置信度门限；harness/vest 重写进 sgie INI 的 `classifier-threshold`，helmet 同时作为空间关联门限 |
| `webhook.url` | `null` | 推送地址；为 `null` 时不推送 |
| `webhook.timeout` | `10` | 单次 HTTP 请求超时（秒） |
| `webhook.retries` | `2` | 失败重试次数（总尝试 = retries + 1） |
| `model.person_conf_threshold` | `0.6` | person 检测置信度门槛，低于此的 person 被探针整体跳过（不判定、不渲染、不告警）；是「一切报警必须基于检测到人」的强制前提，用于压误报 |
| `cameras[].active_rules` | 全部规则 | 本路跟踪的规则名；只对激活维度计算/判定/渲染 |

## 3. 异步分发链路（不阻塞检测线程）

```
检测线程（handle / _trigger）  →  只做决策 + 记录状态
        │  executor.submit(_build_and_send, rule_name, snapshot, objects, ts)
        ▼
executor 线程池（_build_and_send）
  - snapshot 已由 nvdsosd 原生渲染（含检测框 + 违规标签），无需再画框
  - simplejpeg 编码 JPEG（C 扩展，释放 GIL）
        │  启动 daemon 线程 fire-and-forget
        ▼
daemon 线程（_send_payload）
  - base64 编码 → 组装 payload → webhook.send()（HTTP POST）
```

- `snapshot` 来自 `FrameCache`（`pipeline/frame_cache.py`），是告警瞬间该路摄像头最新
  **已渲染帧**（nvdsosd 原生画框，与实时预览一致）；
  缓存尚未就绪时为 `None`，此时仍照常推送，`frame_base64=null`，告警不丢。
- Webhook 失败自动重试，异常只记日志，不影响管线。

## 4. Webhook 推送结构

HTTP `POST`，`Content-Type: application/json; charset=utf-8`，
`json.dumps(payload, ensure_ascii=False, default=str)`。

```json
{
  "alert_type": "no_vest",
  "camera_id": "1363",
  "camera_name": "Mobile Camera 1363",
  "timestamp": "2026-08-11T02:15:33.123456+00:00",
  "objects": [
    {
      "class": "person",
      "confidence": 0.87,
      "bbox": [840, 210, 1020, 760],
      "attributes": [
        { "class": "no_vest", "confidence": 0.5, "bbox": null }
      ]
    }
  ],
  "frame_base64": "<base64 JPEG 证据帧，或 null>"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `alert_type` | string | 触发的规则名（`no_helmet` / `no_vest` / `no_harness`），由该规则独立状态机触发 |
| `camera_id` | string | 摄像头 ID（payload 里的路由键） |
| `camera_name` | string | 摄像头显示名 |
| `timestamp` | string | 触发告警时刻，UTC ISO-8601（`datetime.now(timezone.utc).isoformat()`） |
| `objects` | array | 触发告警的违规实体列表（只含携带本规则违规属性的对象，按探针输出顺序，不排序） |
| `objects[].class` | string | 主实体类别（当前恒为 `person`） |
| `objects[].confidence` | float | 主实体置信度（保留 3 位小数） |
| `objects[].bbox` | array[4] | 全帧坐标 `[x1, y1, x2, y2]` |
| `objects[].attributes` | array\|省略 | 附属违规属性（只含本规则命中的那条）；无属性时整段省略 |
| `objects[].attributes[].class` | string | 属性名（与 `alert_type` 一致） |
| `objects[].attributes[].confidence` | float | 属性置信度（保留 3 位小数） |
| `objects[].attributes[].bbox` | array\|null | 当前为纯分类属性，恒为 `null` |
| `frame_base64` | string\|null | JPEG 证据帧 base64；无证据帧时为 `null` |

字段值在 `manager.py`（`_send_payload`）与 `webhook.py:28-48`（`WebhookAlerter.send`）中组装。

> `alert_type` 直接取**触发的规则名**（`manager.py` 由 `_RuleState` 触发时注入），
> 与 `config.yaml` 的 `rules` 节一一对应；每路摄像头只对 `active_rules` 中声明的
> 规则判定，因此 payload 的类型与平台约定天然一致。

### 4.1 告警类型约定（`alert_type`）

平台按告警规则分别提供回调接口。`alert_type` 是 payload 内的**权威类型标识**，
URL 只做分发、不承载语义——这样即使换 URL、或未来单一消费端要按类型路由，
类型信息都不丢。

| `alert_type` | 含义 |
|------|------|
| `no_helmet` | 未戴安全帽 |
| `no_vest` | 未穿反光衣 |
| `no_harness` | 未系安全带 |

取值用 snake_case，与现有字段风格一致。每起告警只属于一条规则
（`alert_type` = 规则名，`objects[].attributes[].class` 与之一致）。
后续加手套/口罩等规则，在 `rules` 节新增一条即可，`alert_type` 自动取新规则名。

## 5. 本地联调

`local/webhook_receiver.py` 是模拟消费端：接收告警 → 剥离 `frame_base64` 存为 JPEG 证据帧 →
把其余字段追加写 `alerts/payload.jsonl`。默认监听 `0.0.0.0:9999`，
与 `config.yaml` 里 `webhook.url: "http://localhost:9999/api/ppe-alert"` 对应。
