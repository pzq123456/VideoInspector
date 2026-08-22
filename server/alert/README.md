# 告警状态机 与 Webhook 推送

对应 `server/alert/manager.py`（状态机）+ `server/alert/webhook.py`（推送）。

每路摄像头一个 `AlertManager`，内部按激活规则持有独立状态机 `_RuleState`（三态
`IDLE → ARMING → COOLDOWN`）。规则之间互不竞争：`no_helmet` 冷却不会锁住 `no_vest`。

## 状态机

`handle(objects, snapshot, executor)` 每帧调用一次，逐规则判定：

- 冷却期内直接返回；冷却结束回 `IDLE`。
- 有本规则违规实体（`AttributeMeta.name == 规则名`）→ `hits += 1`；无 → `hits -= 1`（容忍偶发丢帧）。
- `hits >= min_detection_count` → 触发告警（`alert_type = 规则名`），进入冷却。
- 触发时 `executor.submit(_build_and_send)`，检测线程只做决策；JPEG/base64/HTTP 由
  executor + daemon 线程异步完成，不阻塞管线。

## Payload

```json
{
  "alert_type": "no_vest",
  "camera_id": "1363",
  "camera_name": "Mobile Camera 1363",
  "timestamp": "2026-08-11T02:15:33.12+00:00",
  "objects": [{ "class": "person", "confidence": 0.87, "bbox": [840,210,1020,760],
                "attributes": [{ "class": "no_vest", "confidence": 0.5, "bbox": null }] }],
  "frame_base64": "<JPEG 证据帧 base64，或 null>"
}
```

- `alert_type` = 触发的规则名（权威类型标识），`objects[].attributes[].class` 与其一致。
- `objects` 只含携带本规则违规属性的 person；`frame_base64` 来自 `FrameCache` 的最新已渲染帧，
  未就绪时为 `null`（告警不丢）。
- classifier 属性置信度固定回退 `0.5`（pyservicemaker 读不到分类概率）。

## 联调

`local/webhook_receiver.py` 是模拟消费端：剥离 `frame_base64` 存为 JPEG，其余追加写
`alerts/payload.jsonl`，默认监听 `0.0.0.0:9999`。
