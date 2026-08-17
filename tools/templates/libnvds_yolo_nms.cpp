/*
 * 全模型共享 bbox 解析器 —— 任意 ultralytics `export(nms=True)` 导出的 ONNX。
 *
 * 适用模型: 输出固定 output0 [B, 300, 6]，每行 = (x1, y1, x2, y2, confidence, class_id)，
 *   坐标为网络输入空间（letterboxed 640x640），nvinfer 按
 *   maintain-aspect-ratio / symmetric-padding 自动映射回原帧。
 *   行数用 inferDims.numElements / 6 计算，对 [B,300,6] 或 [300,6] 都成立，
 *   不依赖 nvinfer 是否折叠 batch 维。
 *
 * 类别裁剪策略（运行时由 num-detected-classes 决定，无需重新编译）:
 *   仅输出 class_id < num-detected-classes 的框。因此:
 *     - 基础 COCO 模型（person, 80 类）配 num-detected-classes=1 → 只出 person
 *     - 专用模型（helmet/vest, 2 类）  配 num-detected-classes=类别数 → 全量输出
 *   再按 [class-attrs-all] pre-cluster-threshold 过滤低置信度垃圾框。
 *
 * 由于 NMS 已烧入模型，nvinfer 配置用 cluster-mode=4（不二次 NMS），
 * 此解析器仅做: 类别裁剪 + 置信度过滤 + 像素坐标裁剪 + 填 NvDsInferObjectDetectionInfo。
 */

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <vector>

extern "C" bool
NvDsInferParseCustomYoloNMS(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
  if (outputLayersInfo.empty()) {
    return false;
  }

  const NvDsInferLayerInfo &output = outputLayersInfo[0];
  const float *buffer = static_cast<const float *>(output.buffer);
  if (buffer == nullptr) {
    return false;
  }

  const unsigned int numDetections = output.inferDims.numElements / 6;
  const float netW = static_cast<float>(networkInfo.width);
  const float netH = static_cast<float>(networkInfo.height);
  const unsigned int numClasses = detectionParams.numClassesConfigured;

  // [class-attrs-all] 缺失时兜底阈值（现有配置始终提供，此处仅防越界）
  const float defaultThr = 0.25f;

  for (unsigned int i = 0; i < numDetections; ++i) {
    const float *p = buffer + i * 6;
    const float x1 = p[0], y1 = p[1], x2 = p[2], y2 = p[3];
    const float conf = p[4];
    const int cls = static_cast<int>(p[5]);

    // 类别裁剪: class_id 必须在配置的类别范围内（num-detected-classes）
    if (cls < 0 || static_cast<unsigned int>(cls) >= numClasses) {
      continue;
    }
    const float thr = detectionParams.perClassPreclusterThreshold.empty()
                          ? defaultThr
                          : detectionParams.perClassPreclusterThreshold[cls];
    if (conf < thr) {
      continue;
    }

    // 零初始化结构体（保证 rotation_angle=0，否则 OBB 元数据未定义会画出斜框）
    NvDsInferObjectDetectionInfo obj{};
    obj.classId = static_cast<unsigned int>(cls);
    obj.detectionConfidence = conf;
    obj.left = std::max(0.0f, std::min(x1, netW));
    obj.top = std::max(0.0f, std::min(y1, netH));
    obj.width = std::max(0.0f, std::min(x2, netW)) - obj.left;
    obj.height = std::max(0.0f, std::min(y2, netH)) - obj.top;

    if (obj.width >= 1.0f && obj.height >= 1.0f) {
      objectList.push_back(obj);
    }
  }
  return true;
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseCustomYoloNMS);
