# ============================================================================
# 自动生成: tools/model_build.py --name {{name}} (task=classify, 二级分类器)
# 模型: models/{{name}}/best.onnx（ultralytics classify 导出, Softmax 已烧入）
#       引擎 models/{{name}}/best_dyn_fp16.engine（trtexec --fp16 预构建, 动态 batch）
# 输出: output0 [B,C] = 各类别概率（Softmax 后, 和=1）
# 模式: process-mode=2 二级推理 — 自动裁剪 operate-on-gie-id 检出的 person 整框,
#       letterbox 到网络输入尺寸后分类; 结果以 NvDsClassifierMeta 挂在 person 对象。
# 路径相对项目根目录, server/main.py 启动时锚定为绝对路径。
# ============================================================================

[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=models/{{name}}/best.onnx
model-engine-file=models/{{name}}/best_dyn_fp16.engine
labelfile-path=generated/{{name}}/labels.txt
network-mode=2
num-detected-classes={{num_classes}}
network-type=1
process-mode=2
operate-on-gie-id={{operate_on_uid}}
operate-on-class-ids=0
classifier-threshold=0.5
gie-unique-id={{uid}}
interval=0
maintain-aspect-ratio=1
symmetric-padding=1
