"""
单端口 RTSP 输出服务（GstRtspServer + GstRTSPMountPoints）

每路摄像头一个挂载点 /cam/{camera_id}，从共享内存 socket 读取主管线侧
shmsink 写入的 H264 字节流，重新打包成 RTP 对外提供。这样 N 路输出共用
同一个 RTSP 端口（默认 8554），客户端只需用不同路径区分摄像头，无需
为每路配置独立端口。

线程模型：
  - 本服务在独立线程跑 GMainLoop（server 在该线程内 attach，使用其默认
    主上下文）。
  - 主管线（pyservicemaker，另一个进程/线程）通过 /tmp/vi_cam_{i} 的
    shm socket 单向传递已编码字节流，二者解耦。
  - 客户端未拉流时 shmsink 设置 wait-for-connection=false 不阻塞主管线。

依赖：PyGObject（python3-gi）+ gir1.2-gst-rtsp-server-1.0（容器已装）。
"""

from __future__ import annotations

import threading

import gi

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GstRtspServer, GLib  # noqa: E402


def _build_launch(shm_socket_path: str, codec: str) -> str:
    """构造单个挂载点的媒体工厂 launch：shm 读流 → 打包 RTP。"""
    if codec == "h265":
        caps = "video/x-h265,stream-format=byte-stream,alignment=au"
        return (
            f"( shmsrc socket-path={shm_socket_path} is-live=true do-timestamp=true "
            f"! {caps} ! h265parse ! rtph265pay name=pay0 pt=96 )"
        )
    caps = "video/x-h264,stream-format=byte-stream,alignment=au"
    return (
        f"( shmsrc socket-path={shm_socket_path} is-live=true do-timestamp=true "
        f"! {caps} ! h264parse ! rtph264pay name=pay0 pt=96 )"
    )


class SinglePortRtspServer:
    """在单一端口上挂载多路编码流的 RTSP 服务器。

    Args:
        rtsp_port: 监听端口（默认 8554）。
        mounts: {mount_path: shm_socket_path}，例如
                {"/cam/1363": "/tmp/vi_cam_0"}。mount_path 必须以 "/" 开头。
        codec: 编码格式 "h264" / "h265"，与主管线侧编码器保持一致。
    """

    def __init__(self, rtsp_port: int, mounts: dict[str, str], codec: str = "h264"):
        self._port = int(rtsp_port)
        self._mounts = dict(mounts)
        self._codec = codec
        self._server = None
        self._loop = None
        self._thread: threading.Thread | None = None

    def start(self):
        """启动 RTSP 服务线程（阻塞直到 loop 退出）。"""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="rtsp-server",
        )
        self._thread.start()

    def _run(self):
        Gst.init(None)
        server = GstRtspServer.RTSPServer()
        server.set_service(str(self._port))

        mount_points = server.get_mount_points()
        for path, socket_path in self._mounts.items():
            factory = GstRtspServer.RTSPMediaFactory()
            factory.set_launch(_build_launch(socket_path, self._codec))
            factory.set_shared(True)
            mount_points.add_factory(path, factory)

        server.attach(None)
        self._server = server
        loop = GLib.MainLoop()
        self._loop = loop
        loop.run()

    def stop(self):
        """尽力退出主循环（进程退出时 daemon 线程也会随之结束）。"""
        loop = self._loop
        if loop is not None:
            GLib.idle_add(loop.quit)
