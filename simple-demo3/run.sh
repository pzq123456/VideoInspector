#!/usr/bin/env bash
# 运行三模型整帧检测流水线 demo: 人体检测(yolo26n) → 安全帽(head/helmet) → 反光衣(vest/no_vest)
#
# 用法:
#   ./run.sh                                  # RTSP 输入 → RTSP 输出（读 configs/rtsp_in.yaml）
#   ./run.sh <其他.yaml>                        # 指定 RTSP 配置文件
#   python3 two_stage_demo.py --file <视频>    # 本地文件调试模式，存 output/frame_*.jpg
#
# 输出:
#   output/run.log     完整运行日志（逐帧 persons/vest/no_vest/helmet/no_helmet + 帧率）
#   RTSP 流             默认 rtsp://localhost:18003/vest（本地转发端口后 VLC 观看）
set -e
cd "$(dirname "$0")"
mkdir -p output
echo ">>> 运行日志 → $(pwd)/output/run.log"
python3 two_stage_demo.py "$@" 2>&1 | tee output/run.log
