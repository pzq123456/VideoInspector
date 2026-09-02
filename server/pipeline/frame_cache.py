"""
证据帧采集：每路摄像头独立的 tee 分支 → appsink → 缓存最新已渲染帧

SafetyProbe（BatchMetadataOperator）只拿得到 batch 元数据、拿不到像素，
因此告警证据帧必须另走一条帧分支。新拓扑在 nvstreamdemux **拆流后**，
每路独立挂载一条：

    demux.src_i → nvdsosd_i → tee_i → [ rtsp 支路 | 证据支路 ]
                                          └── queue → nvvideoconvert → capsfilter(NVMM RGB) → appsink

nvdsosd_i 已在该路帧上**原生渲染**违规红框/违规标签（SafetyProbe 决定渲染内容，
非违规对象显式隐藏），appsink 采集到的即是与实时预览一致的 OSD 渲染帧（且已按
source 隔离，不会跨流污染）。appsink 的 FrameCaptureRetriever.consume() 把最新一帧转成
BGR numpy 缓存进 FrameCache（{source_id: frame}）。告警触发时探针从缓存取
该 source 的帧作为 snapshot 交给 AlertManager，由 executor 线程 JPEG 编码。

线程模型（与 alert/manager.py 的 fire-and-forget 分工一致）：
  - appsink 流线程  ：consume() 写缓存。每帧换入**新数组**、绝不原地改旧数组，
                      这样其他线程持有的旧快照引用不会受影响。
  - GStreamer 流线程：SafetyProbe 读 latest(source_id)。
  - executor 线程   ：_build_and_send 读快照并编码（只读，不再画框）。
FrameCache 用一把锁保护 dict 的 get/set，引用替换是原子的，无需深拷贝。

约束:
  - 每路一个 appsink：source_id 在构造时固定，consume() 不再遍历 batch。
  - consume() 只做「取最新帧 + 缓存」，立即 return 1 放行，不背压主链；
    GPU→CPU 拷贝约 6MB/帧（1080p RGB），单路可接受，若压力大可改缓存 GPU 张量。
  - 证据帧有约 1 帧滞后（探针在 vest、appsink 在下游）：由于每帧都已被
    nvdsosd 原生渲染（框与像素自洽），滞后帧仍是合法证据，不影响可用性。
"""

from __future__ import annotations

import threading

import cv2
import numpy as np
from loguru import logger
from pyservicemaker import BufferRetriever, Receiver

try:  # cupy 用于 GPU 张量 → numpy；本容器已装（torch 未装）
    import cupy
except Exception:  # pragma: no cover
    cupy = None


class FrameCache:
    """线程安全的最新帧缓存：{source_id: BGR numpy 数组}。"""

    def __init__(self):
        self._frames: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()

    def set(self, source_id: int, frame: np.ndarray):
        with self._lock:
            self._frames[source_id] = frame

    def latest(self, source_id: int) -> np.ndarray | None:
        with self._lock:
            return self._frames.get(source_id)


class FrameCaptureRetriever(BufferRetriever):
    """appsink 接收器：把最新一帧（已渲染、单路）转成 BGR numpy 缓存，立即放行。

    Args:
        cache: 共享 FrameCache，探针/AlertManager 从其中取 snapshot。
        source_id: 本 appsink 对应的摄像头序号（即 mux/demux 的 pad 序）。
    """

    def __init__(self, cache: FrameCache, source_id: int):
        super().__init__()
        self._cache = cache
        self._source_id = source_id

    def consume(self, buffer):
        try:
            # 单路（demux 后）缓冲：直接取 batch_id=0，source_id 由构造时固定
            frame = self._to_bgr(buffer, 0)
            if frame is not None:
                self._cache.set(self._source_id, frame)
        except Exception:
            logger.exception("证据帧缓存失败，跳过本帧")
        return 1  # 成功放行，不阻塞下游

    @staticmethod
    def _to_bgr(buffer, batch_id: int) -> np.ndarray | None:
        """buffer.extract(batch_id) → BGR numpy 数组。

        NVMM RGB 张量（GPU）用 cupy.from_dlpack；异常回退 numpy.from_dlpack
        （CPU 内存张量）。Tensor 具备 __dlpack__ / __dlpack_device__，已确认。
        """
        # 必须先 clone：extract 返回的是 NVMM 表面上的视图，dlpack 会接管内存
        # 所有权；不克隆会导致 cupy 与 GStreamer 池各自释放同一块内存（双释放）。
        tensor = buffer.extract(batch_id).clone()
        try:
            if cupy is not None:
                arr = cupy.from_dlpack(tensor).get()   # GPU RGB → CPU numpy
            else:
                arr = np.from_dlpack(tensor).copy()    # CPU 内存回退
        except Exception:
            arr = np.from_dlpack(tensor).copy()        # CPU 内存回退

        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.shape[2] == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr


def add_evidence_capture(pipeline, cache: FrameCache, source_id: int,
                         gpu_id: int = 0, suffix: str = "") -> str:
    """为单路摄像头插入证据帧 tee 分支并挂接收器。

    Args:
        pipeline: pyservicemaker Pipeline 实例。
        cache: 共享 FrameCache。
        source_id: 本路摄像头序号（缓存键）。
        gpu_id: nvvideoconvert 使用的 GPU。
        suffix: 元素名后缀（多路时用于去重）。

    Returns:
        tee 元素名；主链把该路 nvdsosd 的 src 连到它，再分两路：
        tee → (rtsp 支路 | 证据捕获分支，本函数已连好后者)。
    """
    tee = f"ev-tee-{suffix}" if suffix else "ev-tee"
    q = f"ev-queue-{suffix}" if suffix else "ev-queue"
    conv = f"ev-convert-{suffix}" if suffix else "ev-convert"
    caps = f"ev-caps-{suffix}" if suffix else "ev-caps"
    sink = f"ev-sink-{suffix}" if suffix else "ev-sink"
    rec = f"ev-rec-{suffix}" if suffix else "ev-rec"

    pipeline.add("tee", tee)
    # 少量缓冲 + downstream leaky：捕获分支处理不过来时丢老帧，绝不背压主链
    pipeline.add("queue", q, {"max-size-buffers": 4, "leaky": 2})
    pipeline.add("nvvideoconvert", conv, {"gpu-id": gpu_id, "compute-hw": 1})
    pipeline.add("capsfilter", caps, {"caps": "video/x-raw(memory:NVMM), format=RGB"})
    pipeline.add("appsink", sink, {
        "emit-signals": True,
        "sync": False,
        "async": 0,  # tee 分流时所有 sink 需 async=0，避免状态机卡 PAUSED
    })
    pipeline.attach(sink, Receiver(rec, FrameCaptureRetriever(cache, source_id)),
                    tips="new-sample")
    pipeline.link(tee, q, conv, caps, sink)
    return tee
