"""读取 Foreign 共库模式所需的数据库、认证、模型和远程向量服务配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import quote_plus


@dataclass(frozen=True)
class ForeignSettings:
    """保存同一 Foreign MySQL、JWT、Qdrant 和模型网关的连接契约。"""

    database_url: str
    jwt_secret: str
    qdrant_url: str
    qdrant_collection: str
    embedding_api_key: str
    model_name: str
    model_base_url: str
    public_api_base: str
    min_vector_score: float
    # Foreign 独立词法索引和可解释排序参数。
    opensearch_url: str = "http://localhost:19200"
    opensearch_index: str = "foreign_chunks_v1"
    rrf_k: int = 60
    min_evidence_score: float = 0.18
    min_lexical_coverage: float = 0.5
    literature_analysis_concurrency: int = 4
    # 向量服务每次只读请求的超时秒数，避免沿用 HTTP 客户端约五秒的默认值。
    qdrant_timeout_seconds: int = 30


def get_foreign_settings() -> ForeignSettings:
    """从 Foreign 现有环境变量生成共库配置，缺失关键配置时安全失败。

    Returns:
        已校验的共库连接配置。

    Raises:
        RuntimeError: 缺少数据库、JWT、Qdrant 或 Embedding 所需配置。
    """
    database_url = os.getenv("FOREIGN_DATABASE_URL")
    if not database_url:
        required = ("FOREIGN_USERNAME", "FOREIGN_PASSWORD", "FOREIGN_HOST", "FOREIGN_DBNAME")
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise RuntimeError("Foreign 数据库配置缺失：" + ", ".join(missing))
        user = quote_plus(os.environ["FOREIGN_USERNAME"])
        password = quote_plus(os.environ["FOREIGN_PASSWORD"])
        host = os.environ["FOREIGN_HOST"]
        port = os.getenv("FOREIGN_PORT", "3306")
        name = quote_plus(os.environ["FOREIGN_DBNAME"])
        database_url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{name}?charset=utf8mb4"
    jwt_secret = os.getenv("FOREIGN_SECRET_KEY", "")
    embedding_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("QWEN_API_KEY", "")
    qdrant_url = os.getenv("FOREIGN_QDRANT_URL")
    if not qdrant_url:
        qdrant_host = os.getenv("QDRANT_HOST", "")
        qdrant_port = os.getenv("QDRANT_PORT", "6333")
        qdrant_url = f"http://{qdrant_host}:{qdrant_port}" if qdrant_host else ""
    missing = [
        label
        for label, value in (
            ("FOREIGN_SECRET_KEY", jwt_secret),
            ("DASHSCOPE_API_KEY/QWEN_API_KEY", embedding_key),
            ("FOREIGN_QDRANT_URL/QDRANT_HOST", qdrant_url),
        )
        if not value
    ]
    if missing:
        raise RuntimeError("Foreign Agent 连接配置缺失：" + ", ".join(missing))
    return ForeignSettings(
        database_url=database_url,
        jwt_secret=jwt_secret,
        qdrant_url=qdrant_url,
        qdrant_collection=os.getenv("FOREIGN_QDRANT_COLLECTION", "corpus_chunks"),
        embedding_api_key=embedding_key,
        # Foreign 普通 QA 可用的 qwen-long 不支持本 Agent 所需的 JSON Schema
        # 结构化节点；研究 Agent 采用自身经过验证的默认模型。
        model_name=os.getenv("MODEL_NAME") or os.getenv("AGENT_MODEL_NAME", "qwen3.8-flash"),
        model_base_url=os.getenv("OPENAI_BASE_URL")
        or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        public_api_base=os.getenv("FOREIGN_PUBLIC_API_BASE", "http://localhost:9503").rstrip("/"),
        min_vector_score=float(os.getenv("FOREIGN_MIN_VECTOR_SCORE", "0.35")),
        opensearch_url=os.getenv("FOREIGN_OPENSEARCH_URL", "http://localhost:19200"),
        opensearch_index=os.getenv("FOREIGN_OPENSEARCH_INDEX", "foreign_chunks_v1"),
        rrf_k=max(1, int(os.getenv("FOREIGN_RRF_K", "60"))),
        min_evidence_score=float(os.getenv("FOREIGN_MIN_EVIDENCE_SCORE", "0.18")),
        min_lexical_coverage=float(os.getenv("FOREIGN_MIN_LEXICAL_COVERAGE", "0.5")),
        literature_analysis_concurrency=max(
            1, min(20, int(os.getenv("FOREIGN_LITERATURE_ANALYSIS_CONCURRENCY", "4")))
        ),
        qdrant_timeout_seconds=max(
            1, min(120, int(os.getenv("FOREIGN_QDRANT_TIMEOUT_SECONDS", "30")))
        ),
    )
