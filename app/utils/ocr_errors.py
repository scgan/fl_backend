"""脱敏 OCR 错误，识别内容审核拒绝，并收集单次索引构建的跳页警告。"""
from contextlib import contextmanager
from contextvars import ContextVar
import re

from app.utils.bailian_client import BailianError


class OCRContentRejected(BailianError):
    """上游明确拒绝当前图像内容，允许 PDF 层跳过整页 OCR。"""


# 页面结果由调用线程收集；按构建隔离警告，避免并发文献或后续重试串用信息。
_ocr_warnings = ContextVar("ocr_warnings", default=None)


def is_content_rejection(response):
    """仅将 HTTP 400 的明确内容审核错误码判为可跳页，不依赖错误消息文本。"""
    if response.status_code != 400:
        return False
    try:
        body = response.json()
    except (TypeError, ValueError):
        return False
    if not isinstance(body, dict):
        return False
    error = body.get("error")
    details = error if isinstance(error, dict) else body
    return (details.get("code") or details.get("type")) in (
        "data_inspection_failed", "DataInspectionFailed")


@contextmanager
def collect_ocr_warnings():
    """收集当前构建的跳页摘要，退出后恢复调用方上下文。

    Yields:
        可供成功发布流程写入 error_message 的警告列表。
    """
    warnings = []
    token = _ocr_warnings.set(warnings)
    try:
        yield warnings
    finally:
        _ocr_warnings.reset(token)


def record_ocr_warning(message):
    """将已脱敏的单页警告加入当前构建；独立解析时由调用方日志保留。"""
    warnings = _ocr_warnings.get()
    if warnings is not None:
        warnings.append(message)


def safe_http_error(response, *, secrets=()):
    """从远程错误中提取有限诊断字段，剔除凭据、图像及请求输入。

    Args:
        response: 可读取状态码、头和 JSON 的上游 HTTP 响应。
        secrets: 本次请求中必须额外替换的敏感文本序列。

    Returns:
        单行且受长度约束的摘要，保留可排障的错误类型、参数和请求 ID。
    """
    def clean(value, limit):
        """仅接受标量诊断值，脱敏后压平空白并限制长度，其他结构不直接输出。"""
        if not isinstance(value, (str, int, float)):
            return ''
        text = str(value)
        for secret in secrets:
            if secret:
                text = text.replace(secret, '<redacted>')
        text = re.sub(r'data:image/[^\s"<>]+', '<image omitted>', text, flags=re.I)
        text = re.sub(r'Bearer\s+[^\s,;"<>]+', 'Bearer <redacted>', text, flags=re.I)
        text = re.sub(r'sk-[A-Za-z0-9_-]{8,}', '<redacted>', text)
        text = re.sub(r'[A-Za-z0-9+/=_-]{100,}', '<long value omitted>', text)
        return ' '.join(text.split())[:limit]

    summary = f'AICP OCR HTTP {response.status_code}'
    try:
        body = response.json()
    except (TypeError, ValueError):
        body = {}
    body = body if isinstance(body, dict) else {}
    error = body.get('error')
    details = error if isinstance(error, dict) else body
    fields = []
    for name in ('type', 'code', 'param'):
        value = clean(details.get(name), 100)
        if value:
            fields.append(f'{name}={value}')
    request_id = next((response.headers.get(name) for name in
                       ('x-request-id', 'x-requestid', 'x-oneapi-request-id', 'request-id')
                       if response.headers.get(name)), None)
    request_id = clean(request_id or body.get('request_id') or details.get('request_id'), 100)
    if request_id:
        fields.append(f'request_id={request_id}')
    if fields:
        summary += ' [' + ', '.join(fields) + ']'
    message = details.get('message')
    if not isinstance(message, str):
        message = error if isinstance(error, str) else body.get('detail')
    if isinstance(message, list):
        # FastAPI 校验错误只保留原因和位置，不包含 input/ctx 或原请求对象。
        message = '; '.join(
            str(item.get('msg', '')) + ' loc=' + str(item.get('loc', ''))
            for item in message[:3] if isinstance(item, dict))
    message = clean(message, 700)
    if message:
        summary += ': ' + message
    return summary
