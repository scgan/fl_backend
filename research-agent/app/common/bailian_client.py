"""显式传递凭据，校验完整响应，避免 SDK 全局状态影响并发调用。"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass

import requests

LOGGER = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding-flash"


class BailianError(RuntimeError):
    """错误不包含凭据、请求正文或未经筛选的远程错误消息。"""


def api_key():
    """按优先级读取百炼原生 API 凭据，不维护跨请求的 SDK 全局状态。

    Returns:
        去除两端空白的 DASHSCOPE_API_KEY，缺省时回退 QWEN_API_KEY。

    Raises:
        BailianError: 按优先级选出的凭据为空白，或两个环境变量均未配置。
    """
    value = (os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY", "")).strip()
    if not value:
        raise BailianError("请配置 DASHSCOPE_API_KEY 或 QWEN_API_KEY")
    return value


def api_base():
    """读取并校验百炼原生 API 地址，与 OpenAI 兼容接口路径隔离。

    Returns:
        去掉末尾斜杠、以 /api/v1 结尾的 HTTPS 基址。

    Raises:
        BailianError: 地址协议或路径不符合当前原生 Embedding 适配器约束。
    """
    # 原生 API 与 research 的 OpenAI 兼容接口使用不同路径。
    value = os.getenv("FOREIGN_BAILIAN_API_BASE", "https://dashscope.aliyuncs.com/api/v1").rstrip("/")
    if not value.startswith("https://") or not value.endswith("/api/v1"):
        raise BailianError("FOREIGN_BAILIAN_API_BASE 必须为 HTTPS 原生 /api/v1 地址")
    return value


@dataclass(frozen=True)
class EmbeddingConfig:
    """一次调用使用的只读向量模型配置，查询与文档索引必须保持一致。"""

    # 当前适配器支持的百炼 Embedding 模型，与 Qdrant payload 的模型标识一致。
    model: str
    # 向量长度，既约束服务响应，也用于检索时过滤不兼容的索引点。
    dimension: int
    # 单次原生 API 请求的文本数量上限，范围为 1～20。
    batch_size: int


def embedding_config():
    """解析环境配置并限制模型、维度与批量大小的兼容组合。

    Returns:
        当前调用的 EmbeddingConfig；默认使用 1024 维、每批二十条文本。

    Raises:
        BailianError: 数值配置无法解析，或模型、维度、批量大小超出支持范围。
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
    """向百炼原生接口提交 JSON，并对暂时性错误做至多三次有界尝试。

    Args:
        path: 拼接到原生 API 基址后的服务路径。
        payload: 包含模型标识的请求数据；不写入日志或错误信息。
        key: 显式调用凭据，未提供时按环境配置读取。
        timeout: 每次读取超时秒数；连接超时固定为十秒。

    Returns:
        已确认 HTTP 成功、JSON 对象格式有效且未报告业务错误的响应体。

    Raises:
        BailianError: 凭据或基址无效，重试耗尽，HTTP 不成功，或响应格式/业务状态异常。
    """
    # 1. 每次请求显式构造认证头；日志只保留模型、request_id 和用量元数据。
    headers = {"Authorization": "Bearer " + (key or api_key()), "Content-Type": "application/json"}
    for attempt in range(3):
        # 2. 网络错误、限流和服务端异常可重试；其他 HTTP 或业务错误立即失败。
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
        # 3. HTTP 成功仍需检查 JSON 格式及业务码，不能将错误对象当成向量结果。
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
    """分批编码文本并恢复原始输入顺序，拒绝缺失或不兼容的向量结果。

    Args:
        texts: 非空白字符串组成的列表；空列表不发出远程请求。
        text_type: document 用于索引，query 用于检索，传给服务端区分编码用途。
        key: 当前调用显式提供的凭据；缺省时由请求层读取环境配置。

    Returns:
        与输入长度及顺序一致的向量列表，每条向量均满足配置维度和有效数值约束。

    Raises:
        ValueError: 文本容器、文本内容或编码用途不合法。
        BailianError: 配置、远程请求或返回向量的数量、索引、维度及数值校验失败。
    """
    # 1. 先验证调用输入，再按原生 API 批量上限拆分，避免非法批次触发远程计费请求。
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
        # 2. 服务返回顺序不作为契约；使用显式 index/text_index 重建每批输入顺序。
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
        # 3. 当前批次全部通过验证才追加；任一异常直接抛出，不把部分向量用于建索引。
        vectors.extend(ordered[i] for i in range(len(batch)))
    return vectors
