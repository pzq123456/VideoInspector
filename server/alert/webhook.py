"""
Webhook 告警推送
- HTTP POST JSON 到配置的 URL
- 支持超时和自动重试
"""

import json
import urllib.request
import urllib.error

from loguru import logger


class WebhookAlerter:
    """通过 HTTP POST 推送告警的简单客户端。"""

    def __init__(self, url: str, timeout: float = 10, retries: int = 2):
        """
        Args:
            url: Webhook 接收地址
            timeout: 单次请求超时（秒）
            retries: 失败后重试次数
        """
        self.url = url
        self.timeout = timeout
        self.retries = retries

    def send(self, payload: dict) -> bool:
        """
        发送告警。

        Args:
            payload: 告警数据字典，会被序列化为 JSON

        Returns:
            True 表示发送成功
        """
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")

        for attempt in range(self.retries + 1):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=data,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    status = resp.status
                    if 200 <= status < 300:
                        logger.debug("Webhook 发送成功 (status={}): {}", status, self.url)
                        return True
                    else:
                        logger.warning(
                            "Webhook 返回非 2xx (attempt={}/{}, status={})",
                            attempt + 1, self.retries + 1, status,
                        )
            except urllib.error.HTTPError as e:
                logger.warning(
                    "Webhook HTTP 错误 (attempt={}/{}, status={}): {}",
                    attempt + 1, self.retries + 1, e.code, e.reason,
                )
            except urllib.error.URLError as e:
                logger.warning(
                    "Webhook 连接错误 (attempt={}/{}): {}",
                    attempt + 1, self.retries + 1, e.reason,
                )
            except Exception as e:
                logger.error("Webhook 未知错误 (attempt={}/{}): {}", attempt + 1, self.retries + 1, e)

        logger.error("Webhook 发送失败（已达最大重试次数）: {}", self.url)
        return False
