#!/usr/bin/env python3
"""
安全帽 / 反光衣 / 安全带检测服务端入口（DeepStream 两阶段：整帧检测 + 二级分类器）

启动方式:
    python tools/model_build.py --config deploy/config.yaml   # ① 编译模型产物（generated/，增量）
    python -m server.main --config deploy/config.yaml          # ② 启动服务
    （容器内由 SAFETY_CONFIG 指定配置；部署根 = config.yaml 所在目录）

架构:
    本模块 = 主进程监督循环（supervisor）
        └── server/service.py:serve  = 子进程：日志 + 看门狗 + DeepStream 管线
                └── server/pipeline/builder.py:PipelineBuilder = 管线构建

    RTSP 源×N → nvstreammux(batch=N) → [detector/classifier 链] → nvstreamdemux
    → 每路: nvdsosd → tee → [ shmsink(→ RTSP) | appsink(证据帧) ]

断流自愈（Plan A + B）:
    A. nvurisrcbin 内置重连参数收敛为有限次数（source.reconnect.attempts），
       timeout/interval 提至 10s（5s 在 DS7.x 上易触发重连风暴后永久卡死，
       见 NVIDIA 论坛 nvurisrcbin reconnect 相关议题）。
    B. 子进程内 StreamWatchdog 监听 gst stdout：
       重连循环无进展 / 长时间无健康帧心跳 → 以 EXIT_PIPELINE_STUCK 退出；
       主进程监督循环退避重启、全量重建管线。无限重试(-1)已被证明会死锁，
       因此不再使用。
"""

import argparse
import os
import sys
import time
from multiprocessing import Process
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from server.config import EXIT_PIPELINE_STUCK, load_config

# 兼容旧引用（tools/explore_metadata.py 等）从 server.main 导入的入口
from server.pipeline.ini_patch import anchor_ini_config as _anchor_ini_config  # noqa: F401


def _run_supervised(config: dict, deploy_root: Path) -> int:
    """监督循环：启动子进程；异常退出按退避策略重建。Ctrl-C 干净终止。"""
    attempt = 0
    while True:
        attempt += 1
        proc = Process(target=_child_target, args=(config, deploy_root),
                       name=f"safety-pipeline-{attempt}")
        proc.start()
        try:
            proc.join()
        except KeyboardInterrupt:
            print("\nInterrupted. Terminating...")
            proc.terminate()
            proc.join()
            return 0

        exitcode = proc.exitcode
        if exitcode is None or exitcode == 0:
            return exitcode or 0

        wait = min(2 * attempt, 30)
        print(f"pipeline exited code={exitcode} (attempt {attempt}), "
              f"restarting in {wait}s", flush=True)
        time.sleep(wait)


def _child_target(config: dict, deploy_root: Path) -> None:
    from server.service import serve
    serve(config, deploy_root)


def main():
    parser = argparse.ArgumentParser(description="安全帽/反光衣检测服务端 (DeepStream)")
    parser.add_argument(
        "--config", "-c",
        default=os.environ.get("SAFETY_CONFIG")
                or str(_PROJECT_ROOT / "deploy" / "config.yaml"),
        help="配置文件路径（默认: $SAFETY_CONFIG 或 <项目根>/deploy/config.yaml）",
    )
    args = parser.parse_args()

    print(f"加载配置: {args.config}")
    config = load_config(args.config)
    deploy_root = Path(args.config).resolve().parent
    return _run_supervised(config, deploy_root)


if __name__ == "__main__":
    sys.exit(main())
