"""建立 Foreign 共库短事务会话、知识库授权，并兼容旧版范围指纹。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.foreign_integration.config import get_foreign_settings
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_knowledge_base,
    foreign_knowledge_base_document,
)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """创建 Foreign MySQL 引擎并在连接复用前探测断线。

    Returns:
        共库 SQLAlchemy 引擎。
    """
    return create_engine(get_foreign_settings().database_url, pool_pre_ping=True)


def session_factory() -> sessionmaker[Session]:
    """返回独立短事务会话工厂。

    Returns:
        绑定 Foreign MySQL 的会话工厂。
    """
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_session() -> Iterator[Session]:
    """为一次 HTTP 请求提供会话，后台图使用自己的会话。

    Yields:
        请求专属数据库会话。
    """
    with session_factory()() as session:
        yield session


def authorize_knowledge_base(session: Session, user_id: int, base_id: int) -> None:
    """按 Foreign 私有知识库规则复核所有权及未删除状态。

    Args:
        session: 当前短事务会话。
        user_id: 已认证的 Foreign 用户 ID。
        base_id: 知识库 ID。

    Raises:
        PermissionError: 知识库不存在、被删除或不属于当前用户。
    """
    row = session.execute(
        select(foreign_knowledge_base.c.id).where(
            foreign_knowledge_base.c.id == base_id,
            foreign_knowledge_base.c.owner_id == user_id,
            foreign_knowledge_base.c.is_deleted == 0,
        )
    ).first()
    if row is None:
        raise PermissionError("知识库不存在或无权访问")


def eligible_corpus_ids(session: Session, base_id: int) -> list[int]:
    """从 Foreign 主表读取当前已审核、已向量化的知识库文献。

    Args:
        session: 当前短事务会话。
        base_id: 已通过所有权检查的知识库 ID。

    Returns:
        升序且去重的可检索文献 ID。
    """
    rows = session.execute(
        select(foreign_corpus.c.id)
        .join(
            foreign_knowledge_base_document,
            foreign_knowledge_base_document.c.corpus_id == foreign_corpus.c.id,
        )
        .where(
            foreign_knowledge_base_document.c.knowledge_base_id == base_id,
            foreign_knowledge_base_document.c.is_deleted == 0,
            foreign_corpus.c.is_deleted == 0,
            foreign_corpus.c.status == 2,
                    foreign_corpus.c.owner_id.is_(None),
            foreign_corpus.c.is_vectorized == 1,
        )
        .order_by(foreign_corpus.c.id)
    ).scalars()
    return list(dict.fromkeys(int(value) for value in rows))


def scope_version(corpus_ids: list[int]) -> int:
    """兼容旧版文献集合指纹；新运行不调用它冻结或复核整库范围。

    Args:
        corpus_ids: 已排序的 Foreign 文献 ID。

    Returns:
        适合 MySQL BIGINT 的正整数指纹；它不是知识库自增版本号。
    """
    digest = hashlib.sha256(",".join(map(str, corpus_ids)).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
