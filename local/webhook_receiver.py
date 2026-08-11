#!/usr/bin/env python3
"""
Webhook 接收器 — 模拟第三方消费端
- 接收检测服务端推送的告警（含 base64 证据帧）
- 解码并保存证据帧到本地磁盘
- 运维日志 → loguru；推送数据 → 手动写 JSONL

启动方式:
    python local/webhook_receiver.py
    python local/webhook_receiver.py --port 8888
    python local/webhook_receiver.py --save-dir evidence
"""

import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from loguru import logger


# ── 模块级状态 ──────────────────────────────────────────────────────────
_receive_count = 0
_save_dir: Path | None = None
_payload_log: Path | None = None


# ── Handler ──────────────────────────────────────────────────────────────
class WebhookHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        logger.debug("HTTP {} {}", args[0], self.client_address[0])

    def do_POST(self):
        global _receive_count

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            logger.warning("收到非法 JSON，来源: {}", self.client_address[0])
            self._respond(400, {"error": "Invalid JSON"})
            return

        _receive_count += 1
        count = _receive_count

        # 立即响应，避免磁盘 I/O / 日志输出阻塞发送端
        self._respond(200, {"status": "ok", "received": count})

        # ── 以下操作在 HTTP 响应之后执行，不影响发送端 ──

        # 1. 解码并保存证据帧（base64 主动剥离，避免 JSONL 膨胀）
        frame_saved_to = None
        frame_b64 = payload.pop("frame_base64", None)
        if frame_b64 and _save_dir:
            try:
                camera_id = payload.get("camera_id", "unknown")
                ts = payload.get("timestamp", datetime.now(timezone.utc).isoformat())
                ts_str = ts[:19].replace(":", "").replace("T", "_")
                frame_dir = _save_dir / camera_id
                frame_dir.mkdir(parents=True, exist_ok=True)
                frame_path = frame_dir / f"{ts_str}.jpg"
                frame_path.write_bytes(base64.b64decode(frame_b64))
                frame_saved_to = str(frame_path)
                logger.info("帧已保存: {}", frame_path)
            except Exception:
                logger.exception("解码/保存帧失败")

        # 2. 推送数据追加到 JSONL（不含 frame_base64，避免日志膨胀）
        if _payload_log:
            record = {
                "received_at": datetime.now(timezone.utc).isoformat(),
                **payload,
                "frame_saved_to": frame_saved_to,
            }
            with open(_payload_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        # 3. 控制台摘要（通过 logger，无阻塞 flush）
        self._log_summary(count, payload, frame_saved_to)

    def do_GET(self):
        self._respond(200, {"service": "Webhook Receiver", "received_count": _receive_count})

    def _respond(self, status, data):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _log_summary(self, count, payload, frame_path):
        """通过 logger 输出告警摘要，无阻塞 flush。"""
        objects = payload.get("objects", [])  # 修复: 键名为 "objects" 而非 "detections"
        logger.info(
            "Alert #{} | Camera: {} ({}) | Objects: {} | Frame: {}",
            count,
            payload.get("camera_name", "?"),
            payload.get("camera_id", "?"),
            len(objects),
            frame_path or "N/A",
        )
        for i, d in enumerate(objects[:10]):
            bbox = d.get("bbox", [])
            logger.debug(
                "  [{}/{}] {} conf={:.2f} bbox={}",
                i + 1, len(objects), d.get("class", "?"),
                d.get("confidence", 0), bbox,
            )
        if len(objects) > 10:
            logger.debug("  ... and {} more", len(objects) - 10)


# ── 入口 ──────────────────────────────────────────────────────────────────
def main():
    global _save_dir, _payload_log

    parser = argparse.ArgumentParser(description="Webhook 接收器")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=9999)
    parser.add_argument("--save-dir", default="alerts")
    args = parser.parse_args()

    _save_dir = Path(args.save_dir)
    _save_dir.mkdir(parents=True, exist_ok=True)
    _payload_log = _save_dir / "payload.jsonl"

    # loguru：控制台
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    server = HTTPServer((args.host, args.port), WebhookHandler)
    logger.info("Receiver 启动 — {}:{}, 目录 {}", args.host, args.port, args.save_dir)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()
        logger.info("Receiver 已停止 — 共 {} 条", _receive_count)


if __name__ == "__main__":
    main()
