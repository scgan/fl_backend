"""显式传递凭据，校验完整响应，避免 SDK 全局状态影响并发调用。"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
import time

import requests

LOGGER = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"


class BailianError(RuntimeError):
    """错误不包含凭据、请求正文或未经筛选的远程错误消息。"""


def api_key():
    """读取百炼凭据，优先使用 DASHSCOPE_API_KEY，兼容 QWEN_API_KEY。

    Returns:
        去除首尾空白的凭据文本。

    Raises:
        BailianError: 凭据未配置。
    """
    value = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY", "")).strip()
    if not value:
        raise BailianError("请配置 DASHSCOPE_API_KEY 或 QWEN_API_KEY")
    return value


def api_base():
    # 原生 API 与 research 的 OpenAI 兼容接口使用不同路径。
    """读取百炼原生 API 地址，不使用 OpenAI 兼容路径。

    Returns:
        不含末尾斜杠的 HTTPS /api/v1 地址。

    Raises:
        BailianError: 地址不符合原生接口约定。
    """
    value = os.getenv("FOREIGN_BAILIAN_API_BASE", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
    if not value.startswith("https://") or not value.endswith("/api/v1"):
        raise BailianError("FOREIGN_BAILIAN_API_BASE 必须为 HTTPS 原生 /api/v1 地址")
    return value


@dataclass(frozen=True)
class EmbeddingConfig:
    """描述索引端与查询端必须一致的向量配置。

    Attributes:
        model: 当前适配器支持的模型。
        dimension: 须与 Qdrant 集合一致的向量维度。
        batch_size: 每次请求的文本数量上限。
    """
    model: str
    dimension: int
    batch_size: int


def embedding_config():
    """校验模型、维度和每批数量，防止写入不兼容的集合。

    Returns:
        当前不可变向量配置。

    Raises:
        BailianError: 数值无法解析、模型不支持，或维度和批量大小超出约束。
    """
    try:
        config = EmbeddingConfig(
            os.getenv("FOREIGN_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            int(os.getenv("FOREIGN_EMBEDDING_DIMENSION", "1024")),
            int(os.getenv("FOREIGN_EMBEDDING_BATCH_SIZE", "20")),
        )
    except ValueError:
        raise BailianError("向量维度和批量大小必须为整数") from None
    if config.model != DEFAULT_EMBEDDING_MODEL:
        raise BailianError("当前适配器仅支持 qwen3.7-text-embedding-flash")
    if config.dimension not in (256, 512, 768, 1024) or not 1 <= config.batch_size <= 20:
        raise BailianError("向量维度必须为 256/512/768/1024，批量大小为 1..20")
    return config


def post_json(path, payload, *, key=None, timeout=60):
    """请求百炼原生 API，最多三次尝试并屏蔽原始远程错误内容。

    Args:
        path: 相对原生基础地址的接口路径。
        payload: 包含 model 的请求字典。
        key: 可选显式凭据，未提供时读取环境。
        timeout: 单次读取超时秒数，连接超时固定为 10 秒。

    Returns:
        HTTP 和业务状态都成功的响应字典。

    Raises:
        BailianError: 重试耗尽、HTTP 失败、非 JSON 或业务状态失败。
    """
    headers = {"Authorization": "Bearer " + (key or api_key()), "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            response = requests.post(api_base() + path, headers=headers,
                                     json=payload, timeout=(10, timeout))
        except requests.RequestException as error:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise BailianError(f"百炼网络请求失败 ({type(error).__name__})") from None
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
        if response.status_code != 200:
            raise BailianError(f"百炼 HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise BailianError("百炼返回非 JSON 响应") from None
        if not isinstance(body, dict) or body.get("code"):
            raise BailianError("百炼返回业务错误，请通过 request_id 检查请求")
        LOGGER.info("Bailian model=%s request_id=%s usage=%s", payload["model"],
                    body.get("request_id"), body.get("usage", {}))
        return body


def embed_texts(texts, *, text_type="document", key=None):
    """分批生成向量，恢复输入顺序并验证每项完整性。

    Args:
        texts: 非空白文本组成的列表，空列表返回空列表。
        text_type: 入库使用 document，召回使用 query。
        key: 可选显式百炼凭据。

    Returns:
        与输入一一对应的有限非零向量列表。

    Raises:
        ValueError: 文本类型或输入结构无效。
        BailianError: 配置、调用或向量数量、索引、维度和数值无效。
    """
    config = embedding_config()
    if text_type not in ("document", "query"):
        raise ValueError("text_type 必须为 document 或 query")
    if not isinstance(texts, list) or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("向量输入必须为非空文本列表")
    vectors = []
    for start in range(0, len(texts), config.batch_size):
        batch = texts[start:start + config.batch_size]
        body = post_json("/services/embeddings/text-embedding/text-embedding", {
            "model": config.model, "input": {"texts": batch},
            "parameters": {"dimension": config.dimension, "text_type": text_type},
        }, key=key)
        output = body.get("output")
        items = output.get("embeddings") if isinstance(output, dict) else None
        if not isinstance(items, list) or len(items) != len(batch):
            raise BailianError("百炼向量数量与输入不一致")
        ordered = {}
        for item in items:
            if not isinstance(item, dict):
                raise BailianError("百炼向量响应格式错误")
            index = item.get("index", item.get("text_index"))
            vector = item.get("embedding")
            if type(index) is not int or not 0 <= index < len(batch) or index in ordered:
                raise BailianError("百炼向量输入索引缺失、重复或越界")
            if (not isinstance(vector, list) or len(vector) != config.dimension
                    or any(type(v) not in (float, int) or not math.isfinite(v) for v in vector)
                    or not any(vector)):
                raise BailianError("百炼向量维度或数值无效")
            ordered[index] = vector
        vectors.extend(ordered[i] for i in range(len(batch)))
    return vectors
