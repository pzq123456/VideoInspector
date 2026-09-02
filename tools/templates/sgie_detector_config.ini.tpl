# ============================================================================
# 自动生成: tools/model_build.py --config ({{name}}, 二级检测器 sgie-detector)
# 模型: generated/{{name}}/{{ver}}/best.onnx（ultralytics NMS 导出, 动态 batch 1~N）
#       引擎 generated/{{name}}/{{ver}}/best_dyn_fp16.engine（trtexec --fp16 预构建）
#       解析 generated/common/libnvds_yolo_nms.so（全模型共享, 编译一次）
# 输出: output0 [B,300,6] = [x1,y1,x2,y2,conf,cls]（post-NMS, 源图坐标系）
# 模式: process-mode=2 二级检测 — 自动裁剪 operate-on-gie-id 检出的 person 整框,
#       letterbox 到网络输入尺寸后检测; 结果以 NvDsObjectMeta 挂在批次
#       （坐标已映射回源图, 探针按空间关联归属到 person）。
# 类别裁剪: 由 num-detected-classes 决定, class_id < num-detected-classes 才输出
# 路径相对部署根目录（config.yaml 所在目录）, server/main.py 启动时锚定为绝对路径。
# ============================================================================

[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=generated/{{name}}/{{ver}}/best.onnx
model-engine-file=generated/{{name}}/{{ver}}/best_dyn_fp16.engine
labelfile-path=generated/{{name}}/{{ver}}/labels.txt
network-mode=2
num-detected-classes={{num_classes}}
process-mode=2
operate-on-gie-id={{operate_on_uid}}
operate-on-class-ids=0
gie-unique-id={{uid}}
interval=0
network-type=0
cluster-mode=4
maintain-aspect-ratio=1
symmetric-padding=1
custom-lib-path=generated/common/libnvds_yolo_nms.so
parse-bbox-func-name=NvDsInferParseCustomYoloNMS

[class-attrs-all]
pre-cluster-threshold=0.25
