"""集中解析研究来源、授权和文献范围；知识库与根主题数据库共用一条研究执行链。"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.foreign_integration.database import authorize_knowledge_base
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_corpus_database,
    foreign_database,
    foreign_knowledge_base,
    foreign_knowledge_base_document,
    foreign_user,
)


def root_databases(session: Session) -> list[dict]:
    """按 ID 返回有效根主题数据库，不把子分类暴露成独立研究入口。"""
    return [
        dict(row)
        for row in session.execute(
            select(foreign_database.c.id, foreign_database.c.database_name)
            .where(
                foreign_database.c.parent_id == -1,
                foreign_database.c.is_deleted == 0,
            )
            .order_by(foreign_database.c.id)
        ).mappings()
    ]


@dataclass(frozen=True)
class ResearchScope:
    """固定一种研究来源，负责所有执行阶段共用的授权及范围解析。"""

    # 来源类型决定业务授权方式，禁止把数据库 ID 伪装成真实知识库 ID。
    kind: Literal["KNOWLEDGE_BASE", "DATABASE", "GLOBAL"]
    # 真实业务资源整数主键，不是模型生成值。
    resource_id: int

    @classmethod
    def from_ids(
        cls,
        knowledge_base_id: int | None,
        database_id: int | None = None,
    ) -> ResearchScope:
        """要求且只允许一种正整数资源，避免二义性或无范围检索。

        Raises:
            ValueError: 来源缺失、两种来源同时指定或 ID 不是严格正整数。
        """
        if (knowledge_base_id is None) == (database_id is None):
            raise ValueError("必须且只能选择一个知识库或根数据库")
        resource_id = database_id if database_id is not None else knowledge_base_id
        if type(resource_id) is not int or resource_id <= 0:
            raise ValueError("研究来源 ID 必须为正整数")
        return cls("DATABASE" if database_id is not None else "KNOWLEDGE_BASE", resource_id)

    @classmethod
    def from_record(cls, record) -> ResearchScope:
        """从会话或运行的真实绑定恢复来源，不根据入口文案猜测资源。

        Raises:
            ValueError: 绑定无效，或会话入口与真实资源类型不一致。
        """
        source = record.get("entry_source")
        if (
            source in {"RESEARCH", "DATABASE", "AI", "WRITING", "LITERATURE"}
            and not record["knowledge_base_id"]
            and not record.get("database_id")
        ):
            return cls("GLOBAL", 0)
        scope = cls.from_ids(record["knowledge_base_id"], record.get("database_id"))
        if scope.kind == "DATABASE" and source not in {None, "RESEARCH", "DATABASE", "AI"}:
            raise ValueError("研究入口与资源绑定不一致")
        return scope

    @classmethod
    def for_run(cls, run) -> ResearchScope:
        """从真实资源绑定恢复运行范围，AI 写作使用请求中明确选择的知识库。"""
        writing = (run.get("request_payload") or {}).get("writing")
        if run.get("entry_source") == "AI" and writing:
            return cls.from_ids(writing["knowledge_base_id"])
        return cls.from_record(run)

    @property
    def graph_id(self) -> uuid.UUID:
        """为既有图协议提供有命名空间的范围令牌，两种来源同号也不相等。"""
        return uuid.UUID(int=self.resource_id | ((1 << 127) if self.kind == "DATABASE" else 0))

    def authorize(self, session: Session, user_id: int) -> None:
        """复核私有知识库所有权或公共根数据库状态，账号角色由统一认证层校验。

        Raises:
            PermissionError: 私有库越权，或数据库不存在、已删除、不是根节点。
        """
        if self.kind == "KNOWLEDGE_BASE":
            authorize_knowledge_base(session, user_id, self.resource_id)
        elif self.kind == "GLOBAL":
            if not session.scalar(
                select(foreign_user.c.id).where(
                    foreign_user.c.id == user_id,
                    foreign_user.c.status == 1,
                    foreign_user.c.is_deleted == 0,
                )
            ):
                raise PermissionError("账号已停用")
        elif (
            session.scalar(
                select(foreign_database.c.id).where(
                    foreign_database.c.id == self.resource_id,
                    foreign_database.c.parent_id == -1,
                    foreign_database.c.is_deleted == 0,
                )
            )
            is None
        ):
            raise PermissionError("根数据库不存在或已不可用")

    def corpus_ids(
        self, session: Session, user_id: int, *, include_unindexed: bool = False
    ) -> list[int]:
        """按需读取当前成员 ID；创建运行不调用此方法或复制成员清单。

        Args:
            session: 调用方管理的短事务同步会话。
            user_id: 统一认证后的用户，私有知识库按所有权授权。
            include_unindexed: 题录搜索包含未建索引的文献，正文检索仍检查索引状态。

        Returns:
            根数据库包含全部有效后代；删除的中间节点会切断其子树，不跨库兜底。

        Raises:
            PermissionError: 来源在创建任务后被删除、撤权或移为子节点。
        """
        return list(session.scalars(
            self.corpus_query(session, user_id, include_unindexed=include_unindexed)
            .order_by(foreign_corpus.c.id)
        ))

    def corpus_query(
        self, session: Session, user_id: int, *, include_unindexed: bool = False
    ) -> Select:
        """生成当前授权范围的 SQL 子查询，不把整库成员加载到应用内存。

        Args:
            session: 当前线程管理的同步会话，用于资源授权及分类树读取。
            user_id: 已认证用户，私有知识库和个人材料按所有权限定。
            include_unindexed: 题录查询可包含未建索引文献。

        Returns:
            去重的文献 ID 查询，可作为题录或候选校验的关联条件。

        Raises:
            PermissionError: 来源资源已不可访问。
        """
        # 1. 只复核资源权限；成员新增、移出或索引变化不使正在执行的任务失效。
        self.authorize(session, user_id)
        c = foreign_corpus.c
        statement = select(c.id).where(
            c.is_deleted == 0,
            True if include_unindexed else c.is_vectorized == 1,
        )
        if self.kind == "GLOBAL":
            from app.foreign_integration.literature import readable_corpus_clause

            return statement.where(readable_corpus_clause(user_id))
        statement = statement.where(c.status == 2, c.owner_id.is_(None))
        if self.kind == "KNOWLEDGE_BASE":
            members = select(foreign_knowledge_base_document.c.corpus_id).where(
                foreign_knowledge_base_document.c.knowledge_base_id == self.resource_id,
                foreign_knowledge_base_document.c.is_deleted == 0,
            )
            return statement.where(c.id.in_(members))
        # 2. 一次读取有效分类，迭代展开并记录访问集合，避免逐层查询和异常树死循环。
        children = defaultdict(list)
        for node_id, parent_id in session.execute(
            select(
                foreign_database.c.id,
                foreign_database.c.parent_id,
            ).where(foreign_database.c.is_deleted == 0)
        ):
            children[parent_id].append(node_id)
        selected, pending = set(), [self.resource_id]
        while pending:
            node_id = pending.pop()
            if node_id not in selected:
                selected.add(node_id)
                pending.extend(children[node_id])
        # 3. 使用成员子查询限制当前分类树，重复分类归属不放大题录或候选结果。
        return statement.where(
            c.id.in_(select(foreign_corpus_database.c.corpus_id).where(
                foreign_corpus_database.c.database_id.in_(selected),
            ))
        )


def accessible_scopes(session: Session, user_id: int, records) -> set[ResearchScope]:
    """批量标记当前页历史的来源可用性，保持删除资源后的历史只读体验。"""
    scopes = {
        ResearchScope.from_record(row)
        for row in records
        if row["knowledge_base_id"] is not None or row.get("database_id") is not None
    }
    knowledge_ids = {scope.resource_id for scope in scopes if scope.kind == "KNOWLEDGE_BASE"}
    database_ids = {scope.resource_id for scope in scopes if scope.kind == "DATABASE"}
    result = {ResearchScope("GLOBAL", 0)}
    if knowledge_ids:
        result.update(
            ResearchScope("KNOWLEDGE_BASE", key)
            for key in session.scalars(
                select(foreign_knowledge_base.c.id).where(
                    foreign_knowledge_base.c.id.in_(knowledge_ids),
                    foreign_knowledge_base.c.owner_id == user_id,
                    foreign_knowledge_base.c.is_deleted == 0,
                )
            )
        )
    if database_ids:
        result.update(
            ResearchScope("DATABASE", key)
            for key in session.scalars(
                select(foreign_database.c.id).where(
                    foreign_database.c.id.in_(database_ids),
                    foreign_database.c.parent_id == -1,
                    foreign_database.c.is_deleted == 0,
                )
            )
        )
    return result
