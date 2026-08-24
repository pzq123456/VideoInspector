# ============================================================================
# 自动生成: tools/model_build.py --config ({{name}}, detector)
# 模型: generated/{{name}}/best.onnx（ultralytics NMS 导出, 动态 batch 1~N）
#       引擎 generated/{{name}}/best_dyn_fp16.engine（trtexec --fp16 预构建）
#       解析 generated/common/libnvds_yolo_nms.so（全模型共享, 编译一次）
# 输出: output0 [B,300,6] = [x1,y1,x2,y2,conf,cls]（post-NMS, 网络输入空间）
# 类别裁剪: 由 num-detected-classes 决定, class_id < num-detected-classes 才输出
#   - 基础 COCO 模型(>10类) → 1, 只出 person;  专用模型(<=10类) → 全量
# 路径相对部署根目录（config.yaml 所在目录）, server/main.py 启动时锚定为绝对路径。
# ============================================================================

[property]
gpu-id=0
net-scale-factor=0.00392156862745098
model-color-format=0
onnx-file=generated/{{name}}/best.onnx
model-engine-file=generated/{{name}}/best_dyn_fp16.engine
labelfile-path=generated/{{name}}/labels.txt
network-mode=2
num-detected-classes={{num_classes}}
process-mode=1
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
