"""通过独立 OpenAI 兼容配置读取 OCR 文字，不修改其他模型的连接配置。"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
import io
import logging
import os
import time
from urllib.parse import urlsplit

from openai import APIConnectionError, APIStatusError, OpenAI

from app.utils.bailian_client import BailianError
from app.utils.ocr_errors import OCRContentRejected, is_content_rejection, safe_http_error

LOGGER = logging.getLogger(__name__)
DEFAULT_BASE_URL = "https://aicp.ccnu.edu.cn/v1"
DEFAULT_MODEL = "qwen/qwen-vl-ocr-2025-11-20"
TEXT_PROMPT = ("Please output only the text content from the image "
               "without any additional descriptions or formatting.")
MAX_TOKENS = 8192


class AICPOCRTruncated(BailianError):
    """响应被截断，由 PDF 解析层分区域恢复。"""


@dataclass(frozen=True)
class OCRSettings:
    """保存一次文档 OCR 的独立连接配置，避免与生成模型凭据混用。

    Attributes:
        base_url: 已校验的 HTTPS /v1 基础地址。
        model: OCR 模型名称。
        api_key: 独立密钥，repr 隐藏该字段。
    """
    base_url: str
    model: str
    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls):
        """读取并校验独立 OCR 环境配置。

        Returns:
            本次文档可固定使用的不可变设置。

        Raises:
            BailianError: 地址不符合 HTTPS /v1 约定，或模型、独立密钥缺失。
        """
        base_url = os.getenv("FOREIGN_OCR_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path != "/v1"):
            raise BailianError("FOREIGN_OCR_BASE_URL 必须为 HTTPS /v1 基础地址")
        model = os.getenv("FOREIGN_OCR_MODEL", DEFAULT_MODEL).strip()
        key = os.getenv("FOREIGN_OCR_API_KEY", "").strip()
        if not model:
            raise BailianError("请配置 FOREIGN_OCR_MODEL")
        if not key:
            raise BailianError("请配置独立的 FOREIGN_OCR_API_KEY")
        return cls(base_url, model, key)


class AICPOCRClient:
    """每个文档独立持有客户端，由同进程的 HTTP 线程共享，结束时关闭。"""

    def __init__(self, settings, timeout):
        """创建本次文档复用的客户端，并关闭 SDK 自动重试。

        Args:
            settings: 本文档固定使用的独立 OCR 配置。
            timeout: SDK 默认请求超时，具体识别可覆盖。

        settings 保留连接配置，client 持有文档结束后需要关闭的 HTTP 资源。
        """
        self.settings = settings
        # 统一在此做有限重试，关闭 SDK 自动重试，避免额外叠加。
        self.client = OpenAI(base_url=settings.base_url, api_key=settings.api_key,
                             timeout=timeout, max_retries=0)

    def __enter__(self):
        """进入文档上下文，返回可共享 HTTP 连接池的当前客户端。"""
        return self

    def __exit__(self, *_):
        """退出文档上下文时关闭客户端，不抑制处理异常。"""
        self.client.close()

    def recognize(self, image, timeout):
        """识别单幅图像，对可恢复请求错误最多尝试三次。

        Args:
            image: 待编码的 Pillow 页面或分区图像。
            timeout: 单次 HTTP 超时秒数，不是所有重试的总时限。

        Returns:
            去除首尾空白的文本，空白页可返回空字符串。

        Raises:
            OCRContentRejected: HTTP 400 明确报告内容审核拒绝，由 PDF 层跳页。
            AICPOCRTruncated: 输出达到上限，交由上层分区域重试。
            BailianError: 重试耗尽、不可重试状态或响应无效，错误内容已脱敏。
        """
        # 1. 将本次页面或分区编码为 PNG；请求正文始终使用既有 OCR 提示词
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        # 2. 只对连接错误和暂时性状态有限退避；不可重试错误脱敏后交给索引任务处理
        for attempt in range(3):
            try:
                result = self.client.chat.completions.create(
                    model=self.settings.model,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": TEXT_PROMPT}]}],
                    max_tokens=MAX_TOKENS, stream=False, timeout=timeout)
                break
            except APIConnectionError:
                if attempt == 2:
                    raise BailianError("AICP OCR 网络请求失败") from None
                time.sleep(2 ** attempt)
            except APIStatusError as error:
                detail = safe_http_error(error.response, secrets=(self.settings.api_key,))
                LOGGER.warning('AICP OCR request failed: model=%s max_tokens=%s image_pixels=%sx%s '
                               'attempt=%s/3 %s', self.settings.model, MAX_TOKENS,
                               image.width, image.height, attempt + 1, detail)
                if is_content_rejection(error.response):
                    raise OCRContentRejected(detail) from None
                retryable = error.status_code in (408, 409, 429) or error.status_code >= 500
                if not retryable or attempt == 2:
                    raise BailianError(detail) from None
                delay = 2 ** attempt
                try:
                    delay = max(delay, min(30, float(error.response.headers.get("Retry-After", "0"))))
                except ValueError:
                    pass
                time.sleep(delay)
            except Exception:
                # SDK 的远程错误可能含请求或凭据，不把原异常写入任务错误字段。
                raise BailianError("AICP OCR 响应格式错误") from None
        # 3. 验证完整结束及文字类型，截断单独抛出给分区恢复，不能当作正常识别发布
        choices = getattr(result, "choices", None)
        if not isinstance(choices, list) or not choices:
            raise BailianError("AICP OCR 响应缺少识别结果")
        choice = choices[0]
        reason = getattr(choice, "finish_reason", None)
        if reason == "length":
            raise AICPOCRTruncated("AICP OCR 输出截断，需要分区域识别")
        if reason != "stop":
            raise BailianError("AICP OCR 未正常完成识别")
        message = getattr(choice, "message", None)
        text = getattr(message, "content", None)
        if not isinstance(text, str) or getattr(message, "refusal", None):
            raise BailianError("AICP OCR 未返回可用文字")
        usage = getattr(result, "usage", None)
        LOGGER.info("AICP OCR model=%s request_id=%s input_tokens=%s output_tokens=%s",
                    self.settings.model, getattr(result, "id", None),
                    getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None))
        return text.strip()
