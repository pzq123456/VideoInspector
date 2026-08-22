# 安全帽 / 反光衣 / 安全带检测服务端（DeepStream）

实时后台服务：对每路 RTSP 摄像头先整帧跑人体/安全帽检测，再对每个 person 裁剪框跑
安全带/反光衣二级分类，检出违规（`no_helmet` / `no_harness` / `no_vest`）后经 Webhook
推送告警（含 bbox + 证据帧）。每个报警模型一套独立状态机，一路视频可叠加多个报警。

## 快速开始

```bash
python tools/model_build.py --config server/config.yaml   # 编译模型产物到 generated/
python -m server.main --config server/config.yaml          # 启动服务
python local/webhook_receiver.py                           # （可选）模拟消费端
```

## 目录

```
server/
├── main.py            # 入口：读 config → 构建 pipeline → 运行
├── config.yaml        # 唯一事实来源：模型拓扑 + 规则 + 摄像头
├── model_spec.py      # model.gies 的声明式解析/校验
├── metadata.py        # 数据契约 ObjectMeta / AttributeMeta
├── pipeline/          # probe(元数据→告警) / frame_cache(证据帧) / rtsp_server(输出)
├── alert/             # manager(状态机) / rules(规则配置) / webhook(推送)
└── utils/logger.py
tools/model_build.py   # pt→onnx→(class0 交换)→engine→generated/
```

## 配置要点

`config.yaml` 是唯一事实来源，改动后「先 build 再 run」。

```yaml
model:
  person_conf_threshold: 0.6
  gies:                                  # 模型拓扑；只写报警类 violation，不写全量 labels
    person:      { kind: detector,   source: models/person/yolo26n.pt, uid: 1 }
    helmet:      { kind: detector,   source: models/helmet/best.pt,    uid: 3, violation: head }
    harness_cls: { kind: classifier, source: models/original/harness/.../best.pt, uid: 5, violation: no_harness }
    vest_cls:    { kind: classifier, source: models/original/vest/.../best.pt,   uid: 6, violation: no_vest }

rules:                                  # 每条独立状态机；gie 引用上面的模型
  no_helmet:  { gie: helmet,      cooldown_seconds: 10, min_detection_count: 3, attribute_threshold: 0.5 }
  no_vest:    { gie: vest_cls,    cooldown_seconds: 15, min_detection_count: 5, attribute_threshold: 0.5 }
  no_harness: { gie: harness_cls, cooldown_seconds: 5,  min_detection_count: 2, attribute_threshold: 0.5 }

cameras:                                # active_rules 声明本路叠加哪些报警
  - { id: "1363", rtsp_url: "rtsp://...", active_rules: [no_helmet, no_vest, no_harness] }
```

- `gie` 类型：`detector`（整帧检测，`violation` 为报警类，探针做空间关联）/
  `classifier`（二级分类，作用于 person 裁剪框）。
- `violation` 只写报警类。构建时若它不在 class0，工具自动交换输出通道使其落到 class0
  （DeepStream 二级分类器只挂 class0，详见 `notes/deepstream-metadata-exploration.md` §8）。
- `attribute_threshold`：detector = 空间关联门限；classifier = 重写进 INI 的 `classifier-threshold`
  （误报旋钮，每个模型独立；如把 vest 调严就写 `no_vest.attribute_threshold: 0.8`）。

## 推理流水线

```
RTSP×N → nvstreammux(batch=N) → nvinfer(person) → queue → nvinfer(helmet)
                               → queue → nvinfer(harness_cls) → queue → nvinfer(vest_cls)
                               → nvstreamdemux → 每路: nvdsosd → tee → [ shmsink(→RTSP) | appsink(证据帧) ]
```

探针挂在最后一个 nvinfer，把元数据翻译成 `ObjectMeta` 喂给对应摄像头的 `AlertManager`
（按规则独立状态机 `IDLE→ARMING→COOLDOWN`），并给对象上色（违规红 / 其余蓝）。
classifier 合规类不可观测，故无绿框。

## Webhook payload

```json
{
  "alert_type": "no_vest",
  "camera_id": "1363",
  "timestamp": "2026-08-11T02:15:33.12+00:00",
  "objects": [{ "class": "person", "confidence": 0.87, "bbox": [840,210,1020,760],
                "attributes": [{ "class": "no_vest", "confidence": 0.5, "bbox": null }] }],
  "frame_base64": null
}
```

`alert_type` = 规则名（区分报警）；`frame_base64` = nvdsosd 已渲染的 JPEG 证据帧。
classifier 属性置信度固定回退 0.5（pyservicemaker 拿不到分类概率）。

## 说明

- 告警以「检测到人」为前提：探针跳过 person 置信度 < `person_conf_threshold` 的实体。
- 探针不阻塞流线程：JPEG/HTTP 由 executor + daemon 线程异步处理。
- 生成物（`generated/`、`models/*/engine`、`models/*/onnx`）皆可由 config 重建，不入库。
