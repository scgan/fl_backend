"""声明 Foreign 主表映射和仅供研究 Agent 使用的软删除表。"""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.mysql import LONGTEXT

metadata = MetaData()
LARGE_TEXT = Text().with_variant(LONGTEXT(), "mysql")

# Foreign 主表只用于外键和受控读取；迁移时绝不创建或修改这些表。
foreign_user = Table(
    "user",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("username", String(31)),
    Column("role", String(31)),
    Column("status", Integer),
    Column("is_deleted", Boolean),
)
foreign_corpus = Table(
    "corpus",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("title_cn", String(1000)),
    Column("title_en", String(1000)),
    Column("author", String(255)),
    Column("source", String(510)),
    Column("original_source", String(510)),
    Column("keyword_cn", Text),
    Column("keyword_en", Text),
    # 主库已有摘要字段的只读映射，不创建新列或修改主库结构。
    Column("abstract_cn", Text),
    Column("abstract_en", Text),
    Column("language", String(31)),
    Column("publish_time", Integer),
    Column("attachment", Text),
    Column("status", Integer),
    Column("owner_id", BigInteger, comment="个人上传所有者，公共文献为空；由 Backend 迁移"),
    Column("is_deleted", Boolean),
    Column("is_vectorized", Boolean),
)
foreign_knowledge_base = Table(
    "knowledge_base",
    metadata,
    Column("id", BigInteger, primary_key=True),
    Column("owner_id", BigInteger),
    Column("is_deleted", Boolean),
)
foreign_corpus_index_job = Table(
    "corpus_index_job",
    metadata,
    Column("corpus_id", BigInteger, primary_key=True),
    Column("status", String(16)),
    Column("build_id", String(32)),
    Column("attachment_digest", String(64)),
    Column("lexical_index", String(128)),
    Column("chunk_count", Integer),
    Column("error_message", Text),
    Column("generation", Integer),
    Column("vector_status", String(16)),
    Column("lexical_status", String(16)),
    Column("is_deleted", Boolean),
)
foreign_knowledge_base_document = Table(
    "knowledge_base_document",
    metadata,
    Column("knowledge_base_id", BigInteger, primary_key=True),
    Column("corpus_id", BigInteger, primary_key=True),
    Column("is_deleted", Boolean),
)

# 主题数据库业务表仅读取，不纳入 RESEARCH_TABLES 或迁移创建范围。
foreign_database = Table(
    "database", metadata,
    Column("id", Integer, primary_key=True),
    Column("database_name", String(127)),
    Column("parent_id", Integer),
    Column("is_deleted", Boolean),
)
foreign_corpus_database = Table(
    "corpus_database", metadata,
    Column("id", Integer, primary_key=True),
    Column("corpus_id", BigInteger),
    Column("database_id", Integer),
)

research_model_config = Table(
    "research_model_config",
    metadata,
    Column("id", String(32), primary_key=True, comment="模型配置 UUID 十六进制"),
    Column("provider", String(64), nullable=False, comment="模型网关类型"),
    Column("model_name", String(255), nullable=False, comment="供应商模型名"),
    Column("api_base", String(512), comment="兼容 API 根地址"),
    Column("secret_ref", String(128), comment="环境密钥引用名，不存明文"),
    Column("enabled", Boolean, nullable=False, server_default="1", comment="是否允许新运行使用"),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
    UniqueConstraint("provider", "model_name", name="uq_research_model_provider_name"),
)
research_agent = Table(
    "research_agent",
    metadata,
    Column("id", String(32), primary_key=True, comment="Agent 标识"),
    Column("code", String(32), nullable=False, unique=True, comment="稳定业务编码"),
    Column("name", String(128), nullable=False, comment="展示名称"),
    Column("description", Text, nullable=False, comment="职责边界"),
    Column(
        "model_config_id",
        String(32),
        ForeignKey("research_model_config.id"),
        comment="默认模型配置",
    ),
    Column("enabled", Boolean, nullable=False, server_default="1", comment="新运行开关"),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
)
research_prompt = Table(
    "research_agent_prompt",
    metadata,
    Column("id", String(32), primary_key=True, comment="Prompt 版本标识"),
    Column("agent_id", String(32), ForeignKey("research_agent.id"), nullable=False),
    Column("version", Integer, nullable=False, comment="Agent 内不可覆盖的版本号"),
    Column("system_prompt", Text, nullable=False, comment="系统提示词"),
    Column("status", String(16), nullable=False, comment="DRAFT/ACTIVE/RETIRED"),
    Column("created_by", BigInteger, ForeignKey("user.id"), comment="Foreign 用户 ID"),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
    UniqueConstraint("agent_id", "version", name="uq_research_prompt_version"),
)
research_conversation = Table(
    "research_conversation",
    metadata,
    Column("id", String(32), primary_key=True, comment="会话标识"),
    Column(
        "user_id", BigInteger, ForeignKey("user.id"), nullable=False, comment="Foreign 会话所有者"
    ),
    Column(
        "knowledge_base_id", BigInteger, ForeignKey("knowledge_base.id"), nullable=True,
        comment="会话固定绑定的知识库；文献阅读不绑定，Run 通过会话读取",
    ),
    Column("database_id", Integer, nullable=True, comment="数据库研究固定绑定的根主题数据库 ID"),
    Column(
        "entry_source", String(16), nullable=False, server_default="RESEARCH",
        comment="会话入口：AI/LITERATURE/WRITING/RESEARCH；研究资源类型由绑定字段决定",
    ),
    Column("title", String(255), nullable=False),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("updated_at", DateTime, nullable=False, server_default=func.now()),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
    Index("ix_research_conversation_owner_updated", "user_id", "is_deleted", "updated_at"),
    Index(
        "ix_research_conversation_source_updated",
        "user_id", "entry_source", "is_deleted", "updated_at",
    ),
)
research_run = Table(
    "research_run",
    metadata,
    Column("id", String(32), primary_key=True, comment="运行标识"),
    Column(
        "request_id", String(128), nullable=False, unique=True, comment="全局幂等键，删除后也不复用"
    ),
    Column("user_id", BigInteger, ForeignKey("user.id"), nullable=False),
    Column("conversation_id", String(32), ForeignKey("research_conversation.id"), nullable=False),
    Column("agent_id", String(32), ForeignKey("research_agent.id"), nullable=False),
    Column("executor_agent_code", String(32), comment="实际执行的专业 Agent"),
    Column("prompt_id", String(32), ForeignKey("research_agent_prompt.id"), nullable=False),
    Column("model_config_id", String(32), ForeignKey("research_model_config.id"), nullable=False),
    Column(
        "scope_version",
        BigInteger,
        nullable=False,
        comment="旧运行保留范围哈希；新运行固定为1兼容图协议，范围按资源查询",
    ),
    Column(
        "status", String(16), nullable=False, comment="PENDING/RUNNING/SUCCESS/FAILED/CANCELLED"
    ),
    Column("query", Text, nullable=False, comment="原始任务"),
    Column("mode", String(16), nullable=False, comment="FAST/DEEP"),
    Column("output_language", String(16), nullable=False, comment="输出语言"),
    Column("intent", String(48), comment="启明路由意图"),
    Column("evidence_level", String(16), comment="证据充分度"),
    Column("final_answer", LARGE_TEXT, comment="通过引用校验的最终正文快照"),
    Column("result_payload", JSON, comment="专业结果及版本化结构快照"),
    Column("request_payload", JSON, comment="已校验的专业请求参数"),
    Column("error_code", String(128), comment="脱敏错误码"),
    Column("error_message", Text, comment="脱敏错误说明"),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
    Index("ix_research_run_user_created", "user_id", "is_deleted", "created_at"),
    Index("ix_research_run_status_created", "status", "created_at"),
)
# 仅保留旧表定义供历史数据库兼容及运维识别；业务不再读取、写入或创建此表。
research_run_scope = Table(
    "research_run_corpus",
    metadata,
    Column("run_id", String(32), ForeignKey("research_run.id"), primary_key=True),
    Column("corpus_id", BigInteger, ForeignKey("corpus.id"), primary_key=True),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
)
research_message = Table(
    "research_message",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("conversation_id", String(32), ForeignKey("research_conversation.id"), nullable=False),
    Column("run_id", String(32), ForeignKey("research_run.id"), nullable=False),
    Column("role", String(16), nullable=False, comment="USER/ASSISTANT"),
    Column("content", LARGE_TEXT, nullable=False, comment="问题或已验证回答"),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
    UniqueConstraint("run_id", "role", name="uq_research_message_run_role"),
    Index("ix_research_message_conversation", "conversation_id", "is_deleted", "created_at"),
)
research_step = Table(
    "research_step",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("run_id", String(32), ForeignKey("research_run.id"), nullable=False),
    Column("sequence", Integer, nullable=False, comment="本轮从 1 开始的节点顺序"),
    Column("node_name", String(64), nullable=False),
    Column("status", String(16), nullable=False),
    Column("input_summary", JSON),
    Column("output_summary", JSON),
    Column("retry_count", Integer, nullable=False, server_default="0"),
    Column("error_message", Text),
    Column("started_at", DateTime, nullable=False, server_default=func.now()),
    Column("finished_at", DateTime),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
    UniqueConstraint("run_id", "sequence", name="uq_research_step_sequence"),
)
research_evidence = Table(
    "research_evidence",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("run_id", String(32), ForeignKey("research_run.id"), nullable=False),
    Column("corpus_id", BigInteger, ForeignKey("corpus.id"), nullable=False),
    Column("collection_name", String(128), nullable=False),
    Column("point_id", String(128), nullable=False, comment="Qdrant 集合内唯一 point ID"),
    Column("chunk_key", String(512), comment="Foreign payload 的文件名切块键"),
    Column("file_path", String(1024), comment="当次检索的附件路径"),
    Column("page_start", Integer, comment="真实页码；来源缺失时为空"),
    Column("page_end", Integer, comment="真实页码；来源缺失时为空"),
    Column("content_snapshot", LARGE_TEXT, nullable=False, comment="当次实际使用的原文"),
    Column("content_sha256", String(64), nullable=False, comment="原文指纹"),
    Column("retrieval_score", Float, nullable=False, comment="混合检索 RRF 排名融合分"),
    Column("relevance_score", Float, nullable=True, comment="综合重排相关性分；旧记录为空"),
    Column("evidence_rank", Integer, nullable=False),
    Column("assignment", String(16), comment="智鉴 A/B/SHARED 归属"),
    Column("assignment_payload", JSON, comment="专业归属审计摘要"),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
    UniqueConstraint("run_id", "collection_name", "point_id", name="uq_research_evidence_point"),
)
research_citation = Table(
    "research_citation",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("message_id", String(32), ForeignKey("research_message.id"), nullable=False),
    Column("evidence_id", String(32), ForeignKey("research_evidence.id"), nullable=False),
    Column("citation_index", Integer, nullable=False, comment="回答中的 1-based 引用号"),
    Column("quote_text", Text, nullable=False, comment="未经模型改写的短摘录"),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
    UniqueConstraint("message_id", "citation_index", name="uq_research_citation_index"),
)
research_writing_artifact = Table(
    "research_writing_artifact",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("conversation_id", String(32), ForeignKey("research_conversation.id"), nullable=False),
    Column("run_id", String(32), ForeignKey("research_run.id"), nullable=False),
    Column("stage", String(16), nullable=False, comment="产物所属Quill阶段 RQ/PLAN"),
    Column("version", Integer, nullable=False, comment="会话内同阶段递增版本号"),
    Column("parent_artifact_id", String(32), comment="引用的前序产物（阶段衔接链）"),
    Column("section_ref", String(64), comment="章节定位，如大纲 section_id"),
    Column("content", JSON, nullable=False, comment="阶段结构化产物快照"),
    Column("evidence_ids", JSON, comment="产物引用的运行证据主键列表"),
    Column("citation_ids", JSON, comment="产物实际使用的引用编号列表"),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Column("is_deleted", Boolean, nullable=False, server_default="0", comment="软删除标记"),
)

RESEARCH_TABLES = (
    research_model_config,
    research_agent,
    research_prompt,
    research_conversation,
    research_run,
    research_message,
    research_step,
    research_evidence,
    research_citation,
    research_writing_artifact,
)

# Foreign 测试库的默认排序规则可能仍是 latin1；新表必须独立声明 UTF-8，
# 否则中文 Agent 名称、Prompt 与证据正文无法持久化。
for research_table in RESEARCH_TABLES:
    research_table.dialect_options["mysql"]["charset"] = "utf8mb4"
    research_table.dialect_options["mysql"]["collate"] = "utf8mb4_unicode_ci"
