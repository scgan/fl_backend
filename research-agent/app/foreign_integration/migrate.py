"""校验 Foreign 主表，幂等补齐研究表结构，并登记研究、阅读和写作入口的模型与提示词版本。"""

from __future__ import annotations

import uuid

from sqlalchemy import and_, case, insert, inspect, select, text, update

from app.agents.catalog import (
    LITERATURE_SYSTEM_PROMPT,
    QIMING_SYSTEM_PROMPT,
    QUILL_SYSTEM_PROMPT,
    RONGTONG_SYSTEM_PROMPT,
    SUZHI_SYSTEM_PROMPT,
    ZHIJIAN_SYSTEM_PROMPT,
)
from app.foreign_integration.config import get_foreign_settings
from app.foreign_integration.database import get_engine
from app.foreign_integration.schema import (
    RESEARCH_TABLES,
    foreign_corpus,
    foreign_knowledge_base,
    foreign_user,
    metadata,
    research_agent,
    research_conversation,
    research_model_config,
    research_prompt,
    research_run,
)


def stable_id(kind: str, value: str) -> str:
    """为可重复执行的种子配置生成稳定 UUID。

    Args:
        kind: 种子类型。
        value: 稳定业务编码。

    Returns:
        UUID 十六进制字符串。
    """
    return uuid.uuid5(uuid.NAMESPACE_URL, f"foreign-research:{kind}:{value}").hex


def migrate() -> None:
    """检查 Foreign 主表并升级 research_* 表，不触碰旧 EduResearch 表。

    Raises:
        RuntimeError: Foreign 主表缺失、连接不可用或旧 Run 与会话知识库不一致。
    """
    # 1. 检查既有 Foreign 业务主表，再幂等创建研究表和补齐兼容字段
    settings = get_foreign_settings()
    engine = get_engine()
    with engine.connect() as connection:
        for table in (foreign_user, foreign_corpus, foreign_knowledge_base):
            if not engine.dialect.has_table(connection, table.name):
                raise RuntimeError(f"Foreign 主表缺失：{table.name}")
    metadata.create_all(engine, tables=RESEARCH_TABLES, checkfirst=True)
    _ensure_evidence_relevance_score(engine)
    _ensure_selection_scope(engine)
    _ensure_conversation_sources(engine)
    _remove_run_knowledge_base(engine)
    if engine.dialect.name == "mysql":
        _ensure_utf8mb4(engine)
    # 2. 用稳定 ID 固定本次种子模型及各入口的提示词槽位，重复运行不重复插入
    provider = "OPENAI_COMPATIBLE" if settings.model_name else "EXTRACTIVE"
    model_id = stable_id("model", f"{provider}:{settings.model_name}")
    definitions = (
        ("QIMING", "启明·智能问答", QIMING_SYSTEM_PROMPT, 0),
        ("ZHIJIAN", "智鉴·比对分析", ZHIJIAN_SYSTEM_PROMPT, 1),
        ("SUZHI", "溯知·文献综述", SUZHI_SYSTEM_PROMPT, 2),
        ("RONGTONG", "融通·国际传播", RONGTONG_SYSTEM_PROMPT, 1),
        ("LITERATURE", "文献阅读与问答", LITERATURE_SYSTEM_PROMPT, 2),
        ("QUILL", "Quill·论文写作", QUILL_SYSTEM_PROMPT, 1),
    )
    # 3. 在同一配置事务内新增缺失种子、更新默认模型指针并归档旧提示词
    # 已有运行冻结的模型及提示词外键仍保留，不能因升级破坏历史可恢复性。
    with engine.begin() as connection:
        if (
            connection.execute(
                select(research_model_config.c.id).where(research_model_config.c.id == model_id)
            ).first()
            is None
        ):
            connection.execute(
                insert(research_model_config).values(
                    id=model_id,
                    provider=provider,
                    model_name=settings.model_name,
                    api_base=settings.model_base_url,
                    secret_ref="QWEN_API_KEY",
                    enabled=True,
                    is_deleted=False,
                )
            )
        for code, name, prompt, version in definitions:
            agent_id = stable_id("agent", code)
            existing_agent = connection.execute(
                select(research_agent.c.id, research_agent.c.model_config_id).where(
                    research_agent.c.id == agent_id
                )
            ).first()
            if existing_agent is None:
                connection.execute(
                    insert(research_agent).values(
                        id=agent_id,
                        code=code,
                        name=name,
                        description=f"{name}的证据优先研究工作流",
                        model_config_id=model_id,
                        enabled=True,
                        is_deleted=False,
                    )
                )
            elif existing_agent.model_config_id != model_id:
                # 环境指定的 Agent 模型变化时只更新稳定种子 Agent 的默认指针；
                # 历史 Run 保留创建时冻结的旧模型配置。
                connection.execute(
                    update(research_agent)
                    .where(research_agent.c.id == agent_id)
                    .values(model_config_id=model_id)
                )
            prompt_id = stable_id("prompt", f"{code}:{version}")
            if code == "LITERATURE":
                # 升级内置 v1，历史 Run 的 Prompt 外键继续保留；自定义配置不覆盖。
                connection.execute(
                    update(research_prompt)
                    .where(research_prompt.c.id == stable_id("prompt", "LITERATURE:1"))
                    .values(status="ARCHIVED")
                )
            if code == "QIMING":
                # 唯一活动配置固定为槽位 0；旧记录只归档，不破坏历史 Run 的外键。
                connection.execute(
                    update(research_prompt)
                    .where(
                        research_prompt.c.agent_id == agent_id, research_prompt.c.id != prompt_id
                    )
                    .values(status="ARCHIVED")
                )
                connection.execute(
                    update(research_prompt)
                    .where(research_prompt.c.id == prompt_id)
                    .values(system_prompt=prompt, status="ACTIVE", is_deleted=False)
                )
            if (
                connection.execute(
                    select(research_prompt.c.id).where(research_prompt.c.id == prompt_id)
                ).first()
                is None
            ):
                connection.execute(
                    insert(research_prompt).values(
                        id=prompt_id,
                        agent_id=agent_id,
                        version=version,
                        system_prompt=prompt,
                        status="ACTIVE",
                        is_deleted=False,
                    )
                )


def _ensure_selection_scope(engine) -> None:
    """允许全局搜索阅读会话不绑定知识库；Run 不再保存知识库字段。"""
    if engine.dialect.name != "mysql":
        return
    with engine.begin() as connection:
        column = next(
            item for item in inspect(connection).get_columns("research_conversation")
            if item["name"] == "knowledge_base_id"
        )
        if not column["nullable"]:
            connection.exec_driver_sql(
                "ALTER TABLE research_conversation MODIFY COLUMN knowledge_base_id BIGINT NULL"
            )


def _remove_run_knowledge_base(engine) -> None:
    """校验历史绑定后幂等删除 Run 的重复知识库字段及其 MySQL 外键。

    Args:
        engine: 待升级研究表所在的数据库引擎；升级时必须先停止旧版 Agent。

    Raises:
        RuntimeError: 任一历史 Run 缺少会话或与会话知识库绑定不一致。
    """
    with engine.begin() as connection:
        inspector = inspect(connection)
        if not any(item["name"] == "knowledge_base_id"
                   for item in inspector.get_columns("research_run")):
            return
        # 1. 包含软删除历史一起核对，NULL 与非 NULL 也视为不同绑定；异常时不删列。
        mismatch = connection.execute(text(
            "SELECT r.id FROM research_run r "
            "LEFT JOIN research_conversation c ON c.id = r.conversation_id "
            "WHERE c.id IS NULL OR r.knowledge_base_id <> c.knowledge_base_id "
            "OR (r.knowledge_base_id IS NULL AND c.knowledge_base_id IS NOT NULL) "
            "OR (r.knowledge_base_id IS NOT NULL AND c.knowledge_base_id IS NULL) LIMIT 1"
        )).first()
        if mismatch is not None:
            raise RuntimeError(f"Run {mismatch[0]} 与会话知识库绑定不一致，停止删除旧字段")
        # 2. MySQL 的外键与列使用同一条 ALTER 删除；外键名从实际表结构读取并转义。
        drops = []
        if engine.dialect.name == "mysql":
            quote = engine.dialect.identifier_preparer.quote
            drops = [
                f"DROP FOREIGN KEY {quote(item['name'])}"
                for item in inspector.get_foreign_keys("research_run")
                if "knowledge_base_id" in item["constrained_columns"]
            ]
        drops.append("DROP COLUMN knowledge_base_id")
        connection.exec_driver_sql("ALTER TABLE research_run " + ", ".join(drops))


def _ensure_conversation_sources(engine) -> None:
    """补齐旧会话归属，并将数据库研究幂等合并到 RESEARCH，保留真实资源绑定。

    旧阅读会话不绑定知识库或数据库；包含 Quill 运行的旧会话归为写作，其余归为研究。
    只有启明运行的旧写作会话没有入口证据，无法自动区别于研究会话。
    database_id 绑定根主题数据库；升级只补字段，不创建或修改业务数据库及文献关联。
    """
    with engine.begin() as connection:
        # 1. 幂等补齐会话字段；旧记录暂留空入口，以便按历史证据回填。
        columns = {item["name"]: item for item in inspect(connection).get_columns(
            "research_conversation"
        )}
        if "database_id" not in columns:
            connection.exec_driver_sql(
                "ALTER TABLE research_conversation ADD COLUMN database_id INTEGER NULL"
            )
        if "entry_source" not in columns:
            # 先保持 NULL，避免默认 RESEARCH 掩盖需要回填的旧写作/阅读记录。
            connection.exec_driver_sql(
                "ALTER TABLE research_conversation ADD COLUMN entry_source VARCHAR(16) NULL"
            )
        # 2. 仅回填未分类历史；已有数据库绑定的会话不能因知识库为空被归为阅读。
        quill_run = select(research_run.c.id).join(
            research_agent, research_agent.c.id == research_run.c.agent_id,
        ).where(
            research_run.c.conversation_id == research_conversation.c.id,
            research_agent.c.code.in_(("QUILL", "DIAOLONG")),
        ).exists()
        connection.execute(
            update(research_conversation)
            .where(research_conversation.c.entry_source.is_(None))
            .values(entry_source=case(
                (and_(
                    research_conversation.c.knowledge_base_id.is_(None),
                    research_conversation.c.database_id.is_(None),
                ), "LITERATURE"),
                (quill_run, "WRITING"),
                else_="RESEARCH",
            ))
        )
        # 3. 合并两种研究入口，只改来源；会话 ID、范围、时间与消息均保持原值。
        # 包含已软删除记录，避免恢复旧记录后重新产生隔离来源；重复执行不会再更新。
        connection.execute(
            update(research_conversation)
            .where(research_conversation.c.entry_source == "DATABASE")
            .values(entry_source="RESEARCH", updated_at=research_conversation.c.updated_at)
        )
        # 4. 回填后收紧 MySQL 入口约束并补齐列表索引，保留其他业务入口的分类。
        if engine.dialect.name == "mysql" and (
            "entry_source" not in columns or columns["entry_source"]["nullable"]
        ):
            connection.exec_driver_sql(
                "ALTER TABLE research_conversation MODIFY COLUMN "
                "entry_source VARCHAR(16) NOT NULL DEFAULT 'RESEARCH'"
            )
        index = next(item for item in research_conversation.indexes
                     if item.name == "ix_research_conversation_source_updated")
        index.create(connection, checkfirst=True)


def _ensure_evidence_relevance_score(engine) -> None:
    """兼容已有证据表，仅增加可空评分列，不重算或回填历史证据。"""
    with engine.begin() as connection:
        columns = inspect(connection).get_columns("research_evidence")
        if "relevance_score" not in {column["name"] for column in columns}:
            connection.exec_driver_sql(
                "ALTER TABLE research_evidence ADD COLUMN relevance_score FLOAT NULL"
            )


def _ensure_utf8mb4(engine) -> None:
    """仅修复空的 research_* 新表，拒绝自动转换已有研究数据。

    Args:
        engine: 当前 Foreign MySQL 引擎。

    Raises:
        RuntimeError: 旧字符集表已经含有用户数据，需单独制定迁移。
    """
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT TABLE_NAME, TABLE_COLLATION FROM information_schema.TABLES "
                "WHERE TABLE_SCHEMA=DATABASE() AND TABLE_NAME LIKE :pattern"
            ),
            {"pattern": "research_%"},
        ).all()
        wrong = {
            name
            for name, collation in rows
            if name in {table.name for table in RESEARCH_TABLES}
            and not str(collation).startswith("utf8mb4_")
        }
        if not wrong:
            return
        for name in sorted(wrong):
            count = connection.exec_driver_sql(f"SELECT COUNT(*) FROM `{name}`").scalar_one()
            if count:
                raise RuntimeError(f"{name} 已含数据，拒绝自动转换字符集")
        connection.exec_driver_sql("SET FOREIGN_KEY_CHECKS=0")
        try:
            for name in sorted(wrong):
                connection.exec_driver_sql(
                    f"ALTER TABLE `{name}` CONVERT TO CHARACTER SET utf8mb4 "
                    "COLLATE utf8mb4_unicode_ci"
                )
        finally:
            connection.exec_driver_sql("SET FOREIGN_KEY_CHECKS=1")


if __name__ == "__main__":
    migrate()
