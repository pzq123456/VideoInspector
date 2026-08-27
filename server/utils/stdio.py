#!/usr/bin/env python3
"""C++ 层 stdout 转发到 loguru，并把每行喂给回调（供看门狗解析）。"""

import os
import threading


def forward_stdout_to_logger(logger, on_line=None) -> None:
    """把 C++ 层写进 fd1 的输出（measure_fps_probe 的 **FPS 心跳等）转发到 loguru。

    measure_fps_probe 用 std::cout 直打 stdout，绕过 Python 日志系统，导致
    server.log 里没有任何运行期心跳。这里在进程内把 fd1 换成管道，由后台
    线程逐行读回并经 loguru 输出（console + 文件双落）。

    on_line: 可选回调 (line_text: str)，每行非空文本都会调用（看门狗用）。
    """
    r, w = os.pipe()
    os.dup2(w, 1)
    os.close(w)

    def _pump():
        buf = b""
        while True:
            chunk = os.read(r, 4096)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for line in lines:
                text = line.decode(errors="replace").strip()
                if not text:
                    continue
                logger.info("gst| {}", text)
                if on_line is not None:
                    try:
                        on_line(text)
                    except Exception:
                        pass

    threading.Thread(target=_pump, daemon=True, name="stdout-forward").start()
