# 规范化告警流水线设计（v4 · 极简）

> 状态：设计稿 v4（待评审）
> 日期：2026-08-21
> 关联：`notes/deepstream-metadata-exploration.md` §8–9（二级分类器 class0-only quirk）
> 变更：v4 只声明 `violation`（报警类），不手写全量 labels；产物 `generated/<name>/`；
> 管线迁到 test5 声明式 YAML（nodes/edges）。

---

## 0. 一句话结论

告警状态机与 `config.yaml` 结构**不动**。唯一要解决的是 §8 的「二级分类器只挂 class0」。
做法：config 每个报警模型只声明 `violation`（如 `head`/`no_harness`/`no_vest`）→
构建工具读 `.pt` 自动定位该类别索引、把违规类换到 class0 → 产物落 `generated/<name>/` →
管线用 test5 声明式 YAML（nodes/edges）搭建。

---

## 1. 范围（明确不做）

- **不做**：绿框/合规态、多类分类器、每规则独立 webhook 路径、async+tracker 两态方案。
- **只做**：`no_xxx` 违规类能告警（class0 化）+ config 声明 `violation` + 产物 `generated/`。

---

## 2. config 改动（最小）

`rules` / `cameras`(active_rules) / `webhook` / `source` / `output` / `log` **全部不变**。
`model` 节按模型声明「来源 + kind + uid + violation（报警类）」：

```yaml
model:
  person_conf_threshold: 0.6
  gies:
    person:                                  # 锚点检测器（不告警，无 violation）
      kind: detector
      source: models/person/yolo26n.pt
      uid: 1

    helmet:
      kind: detector
      source: models/helmet/best.pt
      violation: head                        # 报警类：head = 未戴帽
      uid: 3

    harness_cls:
      kind: classifier
      source: models/harness_cls/best.pt
      violation: no_harness                  # 报警类：构建期自动换到 class0
      uid: 5

    vest_cls:
      kind: classifier
      source: models/vest_cls/best.pt
      violation: no_vest
      uid: 6
```

**只写报警类**：不手写全量 `labels`。构建工具读 `source`（`.pt`）的 `names` 得到完整类别
与索引，定位 `violation` 的位置，自动处理 class0 化。`violation` 必须 ∈ `.pt` 的 `names`（fail fast）。

---

## 3. 构建（`tools/model_build.py --config`）

```
读 config.model.gies → 逐个:
  source(.pt) → 读 names → 定位 violation 索引 v
    ├─ kind=classifier 且 v != 0: 交换 ONNX 末端 linear 输出通道（§8.6 已验证）→ v 落 class0
    └─ kind=detector: 不换类（检测器全类都挂，无 class0 问题）
  → onnx → engine（trtexec fp16 动态 batch）
  → generated/<name>/labels.txt   （violation 前置）
  → generated/<name>/(pgie|sgie)_config.txt
```

- **产物目录**：`generated/<name>/` 放 INI + labels.txt；engine/onnx 留 `models/<name>/`。
- **class0 交换只对 classifier**（detector 全类可挂，`violation` 仅告诉探针哪个类是报警类）。

---

## 4. 管线（迁到 test5 声明式 YAML）

参考：`/opt/nvidia/deepstream/deepstream/service-maker/sources/apps/python/pipeline_api/deepstream_test5_app/`。
test5 用 `Pipeline(name, config_file=yaml)`，nodes + edges 声明式描述，且 **nvinfer 之间插 `queue`**、
`unique-id` 作为元素属性显式声明。

- 管线 config（nodes/edges）由 `main.py` 从 `config.yaml` 生成或直接维护一份 YAML，
  推理链 = `pgie → queue → helmet → queue → harness_cls → queue → vest_cls`。
- 先**尝试跑通**，跑通即迁移（不盲改）。

---

## 5. 阈值语义（回答「阈值有意义吗」）

| 模型类型 | 阈值 | 作用 | 置信度 |
|---|---|---|---|
| classifier | `classifier-threshold` | class0（违规类）置信度门限，低于不挂载→不告警 | 不可读，payload 回退 0.5 |
| detector | `attribute_threshold` | 空间关联门限（helmet/head 框置信度 ≥ 阈值才算落在人身上） | 可读，真值 |

- class0 化后，classifier 阈值 =「模型对 `no_xxx` 置信度要多高才报警」，是其唯一误报旋钮。
- 若要「出现即告警」，classifier 阈值设低/固定即可。

---

## 6. 迁移（每步独立验证）

| 步骤 | 内容 | 风险 |
|------|------|------|
| S0 | 快修：config 指到 `models/harness_viol/`（no_harness 已 class0），恢复 no_harness 告警 | 低，纯配置 |
| S1 | `model_build.py --config`：读 gies → 编译 + classifier 换 class0 + `generated/<name>/` 落盘 + violation 校验 | 中，ONNX 图探测 |
| S2 | 管线迁 test5 声明式 YAML（nodes/edges + queue + unique-id 元素属性），先跑通 | 中 |
| S3 | 回归：三规则各自告警 + 多路叠加 + 证据帧 + RTSP 输出 | — |

> S0 独立、修复当前静默失效 bug；S1–S3 不动状态机/探针读法（§9.5 已证探针读法正确）。
