"""编排 Foreign 共库中的 Agent 运行、步骤、证据、会话和伪删除。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import func, insert, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.ai_assistant_graph import AssistantGraph
from app.agents.ai_assistant_state import AssistantState
from app.agents.gateway import answer_sink, get_model_gateway, reasoning_sink
from app.agents.literature_graph import MAX_HISTORY_BYTES, LiteratureGraph
from app.agents.literature_language import READING_CHINESE_OUTPUT_CONTRACT
from app.agents.literature_state import LiteratureState
from app.agents.qiming_graph import QimingGraph
from app.agents.qiming_state import QimingState
from app.agents.quill_draft_graph import QuillDraftGraph
from app.agents.quill_draft_state import QuillDraftState
from app.agents.quill_graph import QuillGraph
from app.agents.quill_plan_graph import QuillPlanGraph
from app.agents.quill_plan_state import QuillPlanState
from app.agents.quill_polish_graph import QuillPolishGraph
from app.agents.quill_polish_state import QuillPolishState
from app.agents.quill_revise_graph import QuillReviseGraph
from app.agents.quill_revise_state import QuillReviseState
from app.agents.quill_rq_graph import QuillRQGraph
from app.agents.quill_rq_state import QuillRQState
from app.agents.quill_state import QuillState
from app.agents.rongtong_graph import RongtongGraph
from app.agents.rongtong_state import RongtongState
from app.agents.suzhi_graph import SuzhiGraph
from app.agents.suzhi_state import SuzhiState
from app.agents.zhijian_graph import ZhijianGraph
from app.agents.zhijian_state import ZhijianState
from app.common.config import get_settings
from app.foreign_integration.ai_reading import (
    freeze_reading_scope,
    previous_reading_analysis,
    reading_followup_scope,
    reading_sources,
)
from app.foreign_integration.ai_writing import (
    existing_writing_action,
    writing_artifact,
    writing_history_summary,
    writing_scope,
)
from app.foreign_integration.auth import ForeignActor
from app.foreign_integration.config import get_foreign_settings
from app.foreign_integration.database import session_factory
from app.foreign_integration.literature import (
    LiteratureMaterialTool,
    selection_rows,
    selection_version,
)
from app.foreign_integration.research_scope import ResearchScope
from app.foreign_integration.retrieval import ForeignRetrievalTool
from app.foreign_integration.schema import (
    foreign_user,
    research_agent,
    research_citation,
    research_conversation,
    research_evidence,
    research_message,
    research_model_config,
    research_prompt,
    research_run,
    research_step,
    research_writing_artifact,
)

AGENT_CODES = frozenset({"QIMING", "ZHIJIAN", "SUZHI", "RONGTONG", "LITERATURE", "QUILL"})
TERMINAL_STATUSES = frozenset({"SUCCESS", "FAILED", "CANCELLED"})

# 会话追问识别：命中强续做词，或命中弱指代词且句子较短，或句子短到无法独立成立
# （如「继续」「改成800字」）时才做查询改写；自包含的长问题不改写，避免语义漂移。
STRONG_FOLLOW_UP_PATTERN = re.compile(
    r"(继续|接着|然后|往下|再继续|接着做|继续做|继续上|接着上|刚刚|刚才|上一个|上一步|"
    r"上述|前面|之前|该任务|再来|再写|再补)"
)
WEAK_FOLLOW_UP_PATTERN = re.compile(r"(这个|那个|这条|它|改成|改为|换成|扩写|缩写|扩到|缩到)")
FOLLOW_UP_MAX_SHORT_LENGTH = 12
# 弱指代词只在很短的句子里才算追问（如「把它改成800字」）；
# 超过这个长度通常已经是自包含问题，不改写以免语义漂移。
FOLLOW_UP_WEAK_MAX_LENGTH = 24
# 超过该长度的问题本身已足够具体（也包含前端已补全过的追问），不再改写，
# 避免多一次模型调用与二次改写带来的语义漂移。
FOLLOW_UP_MAX_QUERY_LENGTH = 80


def _step_output_summary(output: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """统计实际节点产物；分侧检索按去重 Chunk 计数，无新增证据时保留已有数量。

    Args:
        output: 当前节点的增量输出，分侧检索将结果保存在嵌套查询列表中。
        state: 节点执行前的图状态，供未修改 Evidence 的节点沿用已有数量。

    Returns:
        与原步骤格式兼容的摘要；检索节点另含查询数和原始候选命中总数。
    """
    summary = {"evidence_count": len(output.get("evidence", state.get("evidence", []))),
               "fields": sorted(output)}
    branches = [result for key in ("object_a_retrieval", "object_b_retrieval")
                for result in output.get(key, [])]
    if branches:
        summary.update(
            evidence_count=len({
                item.chunk_id for result in branches for item in result["evidence"]
            }),
            query_count=len(branches),
            candidate_count=sum(result["candidate_count"] for result in branches),
        )
    if "merged_candidates" in output:
        summary["evidence_count"] = len(output["merged_candidates"])
        summary["candidate_count"] = output["retrieved_candidates"]
    return summary


class ResolvedFollowUpQuery(BaseModel):
    """指代补全的模型输出：把追问改写为可独立检索的问题。"""

    query: str = Field(min_length=2, max_length=4000)


def _looks_like_follow_up(query: str) -> bool:
    """判断一句话是否需要在检索前结合历史补全。

    Args:
        query: 本轮用户原始提问。

    Returns:
        命中强续做词、短句中的指代词、或整体过短（无法独立成立）时为 True。
    """
    text = (query or "").strip()
    if not text:
        return False
    if len(text) > FOLLOW_UP_MAX_QUERY_LENGTH:
        # 已经足够具体（含前端补全过的追问），无需再改写
        return False
    if STRONG_FOLLOW_UP_PATTERN.search(text):
        return True
    if len(text) <= FOLLOW_UP_MAX_SHORT_LENGTH:
        return True
    return bool(WEAK_FOLLOW_UP_PATTERN.search(text)) and len(text) <= FOLLOW_UP_WEAK_MAX_LENGTH


def _selection_bound(run: dict) -> bool:
    """判断是否使用支持全库题录发现及明确选文阅读的材料工具。"""
    return (
        run["executor_agent_code"] == "LITERATURE"
        or run["entry_source"] == "AI"
        or (
            run["entry_source"] in {"RESEARCH", "DATABASE"}
            and not run["knowledge_base_id"] and not run["database_id"]
        )
        or (
            run["entry_source"] == "WRITING"
            and not run["knowledge_base_id"]
            and not run["database_id"]
        )
    )


def _new_id() -> str:
    """生成不依赖数据库自增序列的研究记录主键。

    Returns:
        UUID 十六进制文本。
    """
    return uuid.uuid4().hex


def _now() -> datetime:
    """生成适合 Foreign MySQL DATETIME 的 UTC 时刻。

    Returns:
        无时区标记的 UTC 时间。
    """
    return datetime.now(UTC).replace(tzinfo=None)


def _prompt_and_model(session: Session, code: str) -> tuple[dict, dict, dict]:
    """取得尚未伪删除的 Agent、活动 Prompt 与模型配置。

    Args:
        session: 共库事务会话。
        code: 四个业务 Agent 之一。

    Returns:
        Agent、Prompt、模型配置的行映射。

    Raises:
        LookupError: Agent 或运行配置不可用。
    """
    agent = (
        session.execute(
            select(research_agent).where(
                research_agent.c.code == code,
                research_agent.c.enabled == 1,
                research_agent.c.is_deleted == 0,
            )
        )
        .mappings()
        .first()
    )
    if not agent:
        raise LookupError(f"{code} 未启用")
    prompt = (
        session.execute(
            select(research_prompt)
            .where(
                research_prompt.c.agent_id == agent["id"],
                research_prompt.c.status == "ACTIVE",
                research_prompt.c.is_deleted == 0,
            )
            .order_by(research_prompt.c.version.desc())
            .limit(1)
        )
        .mappings()
        .first()
    )
    model = (
        session.execute(
            select(research_model_config).where(
                research_model_config.c.id == agent["model_config_id"],
                research_model_config.c.enabled == 1,
                research_model_config.c.is_deleted == 0,
            )
        )
        .mappings()
        .first()
    )
    if not prompt or not model:
        raise LookupError(f"{code} 缺少可用 Prompt 或模型")
    return dict(agent), dict(prompt), dict(model)


def normalize_entry_source(source: str | None) -> str | None:
    """将旧数据库入口归一到研究入口，资源类型继续由真实绑定决定。"""
    return "RESEARCH" if source == "DATABASE" else source


def resolve_entry_source(code: str, entry_source: str | None = None) -> str:
    """固定会话入口；写作中的启明问答仍属于 WRITING，专业委派不改变来源。"""
    allowed = {
        "LITERATURE": {"LITERATURE"},
        "QUILL": {"WRITING"},
        "QIMING": {"RESEARCH", "WRITING", "DATABASE", "AI"},
        "ZHIJIAN": {"RESEARCH", "DATABASE"},
        "SUZHI": {"RESEARCH", "DATABASE"},
        "RONGTONG": {"RESEARCH", "DATABASE"},
    }
    source = normalize_entry_source(entry_source) or {
        "LITERATURE": "LITERATURE",
        "QUILL": "WRITING",
    }.get(code, "RESEARCH")
    if source not in allowed.get(code, set()):
        raise ValueError("会话入口与 Agent 不匹配")
    return source


def _load_run_with_conversation(session: Session, run_id: str) -> dict:
    """读取运行及其会话来源绑定，供执行与历史接口共用。

    Args:
        session: 调用方管理的同步数据库会话。
        run_id: 运行标识。

    Returns:
        运行字段和会话中的知识库、数据库及入口；Run 表不重复存储来源绑定。

    Raises:
        LookupError: 运行或所属会话不存在、已删除或所有者不一致。
    """
    row = (
        session.execute(
            select(
                research_run,
                research_conversation.c.knowledge_base_id,
                research_conversation.c.database_id,
                research_conversation.c.entry_source,
            )
            .join(
                research_conversation, research_conversation.c.id == research_run.c.conversation_id
            )
            .where(
                research_run.c.id == run_id,
                research_run.c.is_deleted == 0,
                research_conversation.c.is_deleted == 0,
                research_conversation.c.user_id == research_run.c.user_id,
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        raise LookupError("运行或所属会话不存在")
    return dict(row)


def create_run(
    session: Session,
    *,
    actor: ForeignActor,
    code: str,
    request_id: str,
    knowledge_base_id: int | None,
    query: str,
    conversation_id: str | None,
    mode: str,
    options: dict[str, object],
    entry_source: str | None = None,
    database_id: int | None = None,
) -> dict[str, object]:
    """原子创建运行和用户消息，范围只记录资源绑定或用户明确选择的文献。

    Args:
        session: 共库请求会话。
        actor: 已认证 Foreign 用户。
        code: 业务 Agent 编码。
        request_id: 客户端幂等键。
        knowledge_base_id: 会话绑定的 Foreign 知识库 ID，仅在会话表持久化。
        query: 原始研究问题。
        conversation_id: 可选的同库会话标识。
        mode: FAST 或 DEEP。
        options: 专业运行的已验证参数。
        entry_source: 页面入口；为空时兼容按 Agent 推导，后续不能切换。
        database_id: 数据库研究固定绑定的根主题数据库，与知识库互斥。

    Returns:
        包含是否新建、运行 ID、会话 ID 和兼容范围字段的接受响应。

    Raises:
        ValueError: 幂等键重用或参数不合法。
        PermissionError: 会话或知识库越权。
    """
    code = code.upper()
    entry_source = resolve_entry_source(code, entry_source)
    if options.get("reading") and (
        entry_source != "AI" or code != "QIMING" or not conversation_id
        or options.get("writing") or options.get("reading_context") or options.get("corpus_ids")
    ):
        raise ValueError("文献追问需要原AI会话，不能混用写作或其他材料范围")
    selection_bound = (
        code == "LITERATURE"
        or entry_source == "AI"
        or (entry_source == "RESEARCH" and knowledge_base_id is None and database_id is None)
        or (entry_source == "WRITING" and knowledge_base_id is None and database_id is None)
    )
    scope = None if selection_bound else ResearchScope.from_ids(knowledge_base_id, database_id)
    if database_id is not None and entry_source not in {"RESEARCH", "AI"}:
        raise ValueError("当前入口不支持数据库范围")
    query = query.strip()
    if code not in AGENT_CODES or mode not in {"FAST", "DEEP"} or not 2 <= len(query) <= 4000:
        raise ValueError("Agent、运行模式或任务长度无效")
    if not 8 <= len(request_id) <= 128:
        raise ValueError("request_id 长度必须为 8–128")
    if code == "LITERATURE" and not conversation_id and options.get("task") == "ANALYZE":
        rows = selection_rows(session, actor.id, options["corpus_ids"])
        _, initial_prompt, initial_model = _prompt_and_model(session, code)
        signature = json.dumps(
            [
                actor.id,
                options,
                query,
                selection_version(rows),
                initial_prompt["id"],
                initial_model["id"],
                "literature-with-titled-innovations-v6",
                READING_CHINESE_OUTPUT_CONTRACT,
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        initial_key = "literature:" + hashlib.sha256(signature.encode()).hexdigest()
        deleted = session.scalar(
            select(research_run.c.is_deleted).where(research_run.c.request_id == initial_key)
        )
        # 已删除的幂等键不再复用；用户显式删除后可重新建立阅读会话。
        if not deleted:
            request_id = initial_key
    existing = (
        session.execute(select(research_run).where(research_run.c.request_id == request_id))
        .mappings()
        .first()
    )
    if existing:
        existing_conversation = (
            session.execute(
                select(research_conversation).where(
                    research_conversation.c.id == existing["conversation_id"],
                )
            )
            .mappings()
            .first()
        )
        if (
            existing["is_deleted"]
            or not existing_conversation
            or existing_conversation["is_deleted"]
            or existing_conversation["entry_source"] != entry_source
            or (conversation_id is not None and conversation_id != existing["conversation_id"])
            or existing["user_id"] != actor.id
            or existing_conversation["user_id"] != actor.id
            or existing_conversation["knowledge_base_id"] != knowledge_base_id
            or existing_conversation["database_id"] != database_id
            or existing["query"] != query
            or existing["request_payload"] != options
            or existing["mode"] != mode
            or existing["agent_id"] != _prompt_and_model(session, code)[0]["id"]
        ):
            raise ValueError("request_id 已被不同运行使用")
        if selection_bound:
            if options.get("corpus_ids"):
                selection_rows(session, actor.id, options["corpus_ids"])
            else:
                ResearchScope("GLOBAL", 0).authorize(session, actor.id)
            if options.get("writing"):
                writing_scope(session, actor.id, options["writing"])
            elif database_id is not None:
                ResearchScope.from_ids(None, database_id).authorize(session, actor.id)
        else:
            scope.authorize(session, actor.id)
        return {
            "created": False,
            "run_id": existing["id"],
            "conversation_id": existing["conversation_id"],
            "scope_version": existing["scope_version"],
            "status": existing["status"],
        }

    # 1. 只校验资源身份或明确选文；全库不扫描成员，知识库和数据库不复制关联清单。
    # 用户与资源授权仍在提交和实际检索时执行，范围变化不作为任务失败条件。
    corpus_ids = list(options.get("corpus_ids") or [])
    selected_rows = {}
    if selection_bound:
        if knowledge_base_id is not None or (
            database_id is not None and entry_source not in {"AI", "RESEARCH"}
        ):
            raise ValueError("文献阅读仅使用所选文献范围，不绑定知识库")
        if entry_source == "AI" and options.get("writing"):
            if corpus_ids or database_id is not None:
                raise ValueError("知识库写作不能混用其他文献范围")
            writing_scope(session, actor.id, options["writing"])
        elif entry_source == "AI" and database_id is not None:
            if corpus_ids:
                raise ValueError("数据库范围不能与选文范围同时指定")
            ResearchScope.from_ids(None, database_id).authorize(session, actor.id)
        else:
            ResearchScope("GLOBAL", 0).authorize(session, actor.id)
        if corpus_ids:
            selected_rows = selection_rows(session, actor.id, corpus_ids)
    else:
        scope.authorize(session, actor.id)
    agent, prompt, model = _prompt_and_model(session, code)
    if entry_source == "AI" and options.get("writing"):
        # 写作执行使用 Quill 的冻结配置；入口 Agent 与会话来源仍为 AI/QIMING。
        _, prompt, model = _prompt_and_model(session, "QUILL")
        if model["provider"] != "OPENAI_COMPATIBLE":
            raise LookupError("写作任务需要结构化外部模型")
    if code != "QIMING" and model["provider"] != "OPENAI_COMPATIBLE":
        raise LookupError("专业 Agent 需要结构化外部模型")
    if conversation_id:
        conversation_query = select(research_conversation).where(
            research_conversation.c.id == conversation_id,
            research_conversation.c.is_deleted == 0,
        )
        if selection_bound:
            conversation_query = conversation_query.with_for_update()
        conversation = session.execute(conversation_query).mappings().first()
        if not conversation or conversation["user_id"] != actor.id:
            raise PermissionError("会话不存在或不属于当前用户")
        if conversation["entry_source"] != entry_source:
            raise ValueError("已有会话不能切换入口，请在当前页面新建会话")
        if (conversation["knowledge_base_id"], conversation["database_id"]) != (
            knowledge_base_id,
            database_id,
        ):
            raise ValueError("已有会话不能切换知识库或数据库")
        if entry_source == "AI" and options.get("writing"):
            repeated = existing_writing_action(
                session, conversation_id, actor.id, options["writing"], query
            )
            if repeated:
                return {
                    "created": False,
                    "run_id": repeated["id"],
                    "conversation_id": conversation_id,
                    "scope_version": repeated["scope_version"],
                    "status": repeated["status"],
                }
        if options.get("reading"):
            reading_followup_scope(
                session, actor.id, conversation_id, options["reading"]["source_run_id"]
            )
        if selection_bound:
            pending = session.scalar(
                select(research_run.c.id)
                .where(
                    research_run.c.conversation_id == conversation_id,
                    research_run.c.is_deleted == 0,
                    research_run.c.status.in_(["PENDING", "RUNNING"]),
                )
                .with_for_update()
            )
            if pending:
                raise ValueError("当前会话仍有任务正在执行，请等待完成")
        if selection_bound and entry_source == "WRITING":
            previous = session.execute(
                select(research_run.c.request_payload)
                .where(
                    research_run.c.conversation_id == conversation_id,
                    research_run.c.is_deleted == 0,
                )
                .order_by(research_run.c.created_at.desc(), research_run.c.id.desc())
            ).first()
            if previous and previous[0].get("corpus_ids", []) != options.get("corpus_ids", []):
                raise ValueError("写作会话不能切换选文范围，请新建会话")
        if code == "LITERATURE":
            prior = (
                session.execute(
                    select(research_run)
                    .where(
                        research_run.c.conversation_id == conversation_id,
                        research_run.c.is_deleted == 0,
                    )
                    .order_by(research_run.c.created_at.desc())
                )
                .mappings()
                .first()
            )
            if (
                not prior
                or prior["executor_agent_code"] != "LITERATURE"
                or prior["request_payload"]["corpus_ids"] != corpus_ids
            ):
                raise ValueError("阅读会话不能切换文献范围，请新建会话")
            if prior["status"] in {"PENDING", "RUNNING"}:
                raise ValueError("当前会话仍有分析正在执行，请等待完成")
            if options["task"] == "ASK" and not session.scalar(
                select(research_run.c.id).where(
                    research_run.c.conversation_id == conversation_id,
                    research_run.c.is_deleted == 0,
                    research_run.c.status == "SUCCESS",
                    research_run.c.result_payload["payload"]["task"].as_string() == "ANALYZE",
                )
            ):
                raise ValueError("请先完成文献分析再继续提问")
    else:
        if entry_source == "AI" and (options.get("writing") or {}).get("source_run_id"):
            raise ValueError("卡片操作需要已有 AI 会话")
        if code == "LITERATURE" and options["task"] == "ASK":
            raise ValueError("追问需要已有阅读会话")
        conversation_id = _new_id()
        session.execute(
            insert(research_conversation).values(
                id=conversation_id,
                user_id=actor.id,
                knowledge_base_id=knowledge_base_id,
                database_id=database_id,
                entry_source=entry_source,
                title=(
                    " / ".join(
                        row["title_cn"] or row["title_en"] or f"文献 {key}"
                        for key, row in selected_rows.items()
                    )[:255]
                    if code == "LITERATURE"
                    else query[:255]
                ),
                is_deleted=False,
            )
        )

    # 2. Quill 阶段衔接引用的前序产物必须属于当前会话，越权或不匹配立即失败；
    # DRAFT 还要求 section_ref 命中大纲中的章节。
    if entry_source == "AI" and options.get("writing"):
        writing_artifact(session, conversation_id, actor.id, options["writing"])
    if code == "QUILL" and options.get("artifact_id"):
        artifact = _load_writing_artifact(
            session,
            str(options["artifact_id"]),
            conversation_id,
            actor.id,
            expect_stage="PLAN" if options.get("stage") == "DRAFT" else "RQ",
        )
        if options.get("stage") == "DRAFT":
            section_ref = str(options.get("section_ref") or "")
            sections = (artifact.get("content") or {}).get("sections") or []
            if not any(section.get("section_id") == section_ref for section in sections):
                raise ValueError("起草的章节编号不在所引用的大纲中")

    # 3. 会话保存资源绑定，明确选文已在 request_payload 中；不再逐篇写任务范围表。
    # scope_version 保留图协议要求的正整数占位，新运行为 1，不表示整库内容版本。
    # MySQL DATETIME 的秒级创建时间可能相同；阅读 Run ID 的时间前缀保持历史顺序。
    run_id = f"{time.time_ns():016x}{uuid.uuid4().hex[:16]}" if selection_bound else _new_id()
    version = 1
    output_language = (
        "ZH" if code == "LITERATURE" else str(options.get("target_language") or "AUTO")
    )
    session.execute(
        insert(research_run).values(
            id=run_id,
            request_id=request_id,
            user_id=actor.id,
            conversation_id=conversation_id,
            agent_id=agent["id"],
            executor_agent_code=code,
            prompt_id=prompt["id"],
            model_config_id=model["id"],
            scope_version=version,
            status="PENDING",
            query=query,
            mode=mode,
            output_language=output_language,
            request_payload=options,
            is_deleted=False,
        )
    )
    session.execute(
        insert(research_message).values(
            id=_new_id(),
            conversation_id=conversation_id,
            run_id=run_id,
            role="USER",
            content=query,
            is_deleted=False,
        )
    )
    session.execute(
        update(research_conversation)
        .where(research_conversation.c.id == conversation_id)
        .values(updated_at=_now())
    )
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise ValueError("并发提交了相同 request_id，请重试读取既有运行") from None
    return {
        "created": True,
        "run_id": run_id,
        "conversation_id": conversation_id,
        "scope_version": version,
        "status": "PENDING",
    }


def _load_writing_artifact(
    session: Session,
    artifact_id: str,
    conversation_id: str,
    user_id: int,
    expect_stage: str = "RQ",
) -> dict[str, Any]:
    """读取Quill前序阶段产物，并校验会话归属与阶段类型。

    Args:
        session: 共库事务会话。
        artifact_id: 前序产物主键。
        conversation_id: 当前运行的会话主键。
        user_id: 当前用户主键，仅用于审计注释；归属由会话所有权保证。
        expect_stage: 允许被引用的产物阶段（PLAN 引用 RQ，DRAFT 引用 PLAN）。

    Returns:
        产物行映射。

    Raises:
        LookupError: 产物不存在或已删除。
        ValueError: 产物不属于当前会话或阶段类型不能被引用。
    """
    artifact = (
        session.execute(
            select(research_writing_artifact).where(
                research_writing_artifact.c.id == artifact_id,
                research_writing_artifact.c.is_deleted == 0,
            )
        )
        .mappings()
        .first()
    )
    if not artifact:
        raise LookupError("引用的前序阶段产物不存在")
    if artifact["conversation_id"] != conversation_id:
        raise ValueError("前序阶段产物不属于当前会话，请在同一会话内继续")
    if artifact["stage"] != expect_stage:
        raise ValueError(f"该阶段只能引用 {expect_stage} 阶段的产物")
    return dict(artifact)


def _persist_writing_artifact(
    session: Session,
    *,
    run_id: str,
    conversation_id: str,
    payload: dict[str, Any],
    used_citation_indices: list[int],
) -> str:
    """把Quill RQ/PLAN 结果快照持久化为阶段产物，供后续阶段衔接。

    Args:
        session: 发布事务会话。
        run_id: 产出该结果的运行主键。
        conversation_id: 运行所属会话主键。
        payload: 阶段结构化结果（QuillRQResult / QuillPlanResult 快照）。
        used_citation_indices: 结果实际使用的引用编号。

    Returns:
        新建产物主键。
    """
    stage = str(payload["stage"])
    max_version = session.scalar(
        select(func.max(research_writing_artifact.c.version)).where(
            research_writing_artifact.c.conversation_id == conversation_id,
            research_writing_artifact.c.stage == stage,
            research_writing_artifact.c.is_deleted == 0,
        )
    )
    artifact_id = _new_id()
    payload["version"] = int(max_version or 0) + 1
    session.execute(
        insert(research_writing_artifact).values(
            id=artifact_id,
            conversation_id=conversation_id,
            run_id=run_id,
            stage=stage,
            version=int(max_version or 0) + 1,
            section_ref=str(payload.get("section_ref") or "") or None,
            content=payload,
            citation_ids=used_citation_indices,
            is_deleted=False,
        )
    )
    return artifact_id


# ---------------- 运行中节点的实时思维链/回答正文缓冲 ----------------
# 网关在思考流式调用中把 reasoning_content 增量实时追加进 _LIVE_REASONING[run_id]，
# 回答正文 content 增量追加进 _LIVE_ANSWER[run_id]，/agent-runs/{id}/reasoning
# 轮询端点读取；运行结束（发布或失败）时清理。
_LIVE_REASONING: dict[str, list[str]] = {}
_LIVE_ANSWER: dict[str, list[str]] = {}
_LIVE_STEP: dict[str, str] = {}
_LIVE_LOCK = threading.Lock()


def _discard_live(run_id: str) -> None:
    """运行结束后清空该运行的实时思维链/回答正文缓冲。"""
    with _LIVE_LOCK:
        _LIVE_REASONING.pop(run_id, None)
        _LIVE_ANSWER.pop(run_id, None)
        _LIVE_STEP.pop(run_id, None)


class _LiveReasoningSink(list):
    """思维链捕获桶：聚合进步骤摘要的同时，镜像到运行级实时缓冲供轮询端点读取。"""

    def __init__(self, mirror: list[str]) -> None:
        """建立步骤内捕获列表，并绑定运行级实时文本列表。

        Args:
            mirror: 已登记在运行级缓冲中的列表引用，不复制，供轮询同步读取。
        """
        super().__init__()
        # 当前节点的局部内容供步骤落库；该引用同步维护运行轮询接口看到的内容。
        self._mirror = mirror

    def append(self, item: str) -> None:
        """追加节点文本，并把非空增量镜像到运行级缓冲。

        Args:
            item: 网关推送的一段思考或回答正文；空串只保留在局部列表中。
        """
        super().append(item)
        if item:
            self._mirror.append(item)

    def clear(self) -> None:
        """同时清空局部列表及其绑定的运行级镜像，避免重试继续展示已撤销文本。"""
        super().clear()
        self._mirror.clear()

    def extend(self, items) -> None:
        """逐项复用追加规则，保证批量文本也同步进入实时镜像。

        Args:
            items: 网关提供的可迭代文本增量；保持输入顺序。
        """
        for item in items:
            self.append(item)


class RunExecutor:
    """在独立短事务中执行研究、阅读与写作图，并原子发布可审计结果。"""

    def __init__(self, run_id: str) -> None:
        """保存目标运行及并行图节点共享的步骤序号锁。

        Args:
            run_id: 已提交的研究运行标识。
        """
        # 所有领取、步骤、来源及发布写入都限定在这一持久化 Run。
        self.run_id = run_id
        # 并行图节点只在分配步骤序号时串行，外部检索和模型调用不持有此锁。
        self.step_lock = asyncio.Lock()
        # 执行时按任务类型绑定检索 Tool；其真实来源缓存供最终发布复核。
        self.retrieval: ForeignRetrievalTool | None = None
        # 执行开始时加载的模型配置，阅读子图只调整输出预算，不重新选取模型版本。
        self.model_config: dict | None = None

    async def execute(self) -> None:
        """原子领取运行，按资源引用和本轮模型配置执行图，并发布快照或记录失败。

        仅 PENDING Run 能进入执行；重复调度者领取失败后直接返回。图内异常由
        _fail 写入终态，成功结果经 _publish 核对本轮证据后以独立事务发布；两种结局
        都释放本运行的实时文本缓冲。领取事务自身的数据库错误仍向调用方传播。
        """
        # 1. 用条件更新原子领取并立即提交；不跨模型等待持有事务或允许重复执行。
        with session_factory()() as session:
            claimed = session.execute(
                update(research_run)
                .where(
                    research_run.c.id == self.run_id,
                    research_run.c.status == "PENDING",
                    research_run.c.is_deleted == 0,
                )
                .values(status="RUNNING", started_at=_now())
            )
            session.commit()
            if claimed.rowcount != 1:
                return
        try:
            # 2. 检查资源身份，恢复明确选文及本轮固定的 Prompt、模型。
            run, prompt, model, fixed_scope = self._load_context()
            self.model_config = model
            reading = run["executor_agent_code"] == "LITERATURE"
            retrieval_type = (
                LiteratureMaterialTool if _selection_bound(run) else ForeignRetrievalTool
            )
            resource = ResearchScope.for_run(run)
            # 3. 工具按资源实时查询范围；只有明确选文传入 ID，不加载整库题录。
            self.retrieval = retrieval_type(
                actor_id=int(run["user_id"]),
                knowledge_base_id=resource.resource_id if resource.kind == "KNOWLEDGE_BASE" else 0,
                fixed_scope=fixed_scope,
                fixed_version=int(run["scope_version"]),
                settings=get_foreign_settings(),
                **({"database_id": resource.resource_id} if resource.kind == "DATABASE" else {}),
            )
            # 同一网关也供启明委派的溯知等子图使用，不能只延长直接入口的等待时间。
            model_timeout = get_settings().research_model_timeout_seconds
            gateway = get_model_gateway(
                model["provider"],
                model_name=model["model_name"],
                api_base=model["api_base"],
                timeout_seconds=max(180, model_timeout) if reading else model_timeout,
                max_tokens=16000 if reading else 6000,
                temperature=0.0,
            )
            # 4. 图只生成状态和来源；发布层统一复核并原子写入消息、引用、结构化结果和成功终态。
            final = await self._run_graph(run, prompt, gateway)
            self._publish(run, final)
        except Exception as exc:
            # 5. 图执行或发布失败转成失败终态，已记录的步骤保留供排查，不发布半条回答。
            self._fail(exc)
        finally:
            # 运行结束（成功或失败）清空实时思维链缓冲，避免内存累积
            _discard_live(self.run_id)

    def _load_context(self) -> tuple[dict, dict, dict, list[int] | None]:
        """恢复资源引用或明确选文及本轮配置，不读取历史任务范围关联表。

        Returns:
            Run 上下文、Prompt、模型及明确选文；选文为 None 时按资源查询当前范围。

        Raises:
            PermissionError: 知识库访问被撤销。
            LookupError: 运行、所属会话或配置关联记录已不可用。
        """
        # 1. 通过会话取得唯一来源绑定，不在 Run 重复存储；重读权限拒绝撤权上下文。
        with session_factory()() as session:
            run = _load_run_with_conversation(session, self.run_id)
            ResearchScope.for_run(run).authorize(session, int(run["user_id"]))
            # 2. 按运行冻结 ID 加载提示词和模型，不因活动配置变更而替换本轮输入
            prompt = (
                session.execute(
                    select(research_prompt).where(
                        research_prompt.c.id == run["prompt_id"],
                        research_prompt.c.is_deleted == 0,
                    )
                )
                .mappings()
                .first()
            )
            model = (
                session.execute(
                    select(research_model_config).where(
                        research_model_config.c.id == run["model_config_id"],
                        research_model_config.c.is_deleted == 0,
                    )
                )
                .mappings()
                .first()
            )
            if not prompt or not model:
                raise LookupError("运行固定的 Prompt 或模型已不可用")
            # 3. 明确选文的顺序已保存在请求内；旧运行也按原资源或请求恢复，不依赖旧表。
            fixed_scope = list((run["request_payload"] or {}).get("corpus_ids") or []) or None
            return dict(run), dict(prompt), dict(model), fixed_scope

    async def _run_graph(self, run: dict, prompt: dict, gateway: Any) -> dict[str, Any]:
        """按冻结的执行入口和写作阶段构造对应图状态，再执行各图自身的校验流程。

        Args:
            run: 含固定入口、资源引用、模型及请求参数的运行行。
            prompt: 当前运行固定的提示词版本，不在执行过程中重选。
            gateway: 当前运行固定的模型网关。

        Returns:
            对应业务图的最终状态，交由统一发布层再次核验来源；文献阅读图不执行语义 Reviewer。

        Raises:
            ValueError: 写作所引用的大纲中没有目标章节。
            AssertionError: 执行器尚未建立受控检索工具。
        """
        assert self.retrieval is not None
        # 1. 按冻结的执行入口路由，独立阅读先恢复其历史和原分析，不依赖知识库占位 ID
        code = run["executor_agent_code"]
        if run["entry_source"] == "AI":
            history, _ = self._reading_history(run)
            def load_reading_sources():
                """独立线程内恢复文献指代来源，不跨模型等待持有数据库会话。"""
                with session_factory()() as session:
                    try:
                        return reading_sources(
                            session, run, lexical_index=get_foreign_settings().opensearch_index
                        )
                    except (ValueError, PermissionError) as exc:
                        # 过期页面提示只阻止基于该材料的阅读，不应阻止用户重新搜索或另问主题。
                        hint = run["request_payload"].get("reading_context") or {}
                        return [{
                            "source_run_id": hint.get("source_run_id", run["id"]),
                            "kind": "CANDIDATES", "query": "", "documents": [],
                            "error": str(exc),
                        }]

            sources = [] if run["request_payload"].get("reading") else (
                await asyncio.to_thread(load_reading_sources)
            )
            # AI 分流搜索、研究、阅读与写作；完整启明继续识别普通问答、比较与溯源。
            # 专业图使用同一检索工具及步骤回调，全部结果仍由原 AI Run 发布。
            delegated = self._delegated_graphs(gateway)
            qiming = QimingGraph(
                self.retrieval,
                gateway,
                prompt["system_prompt"],
                self._run_step,
                zhijian_graph=delegated.get("ZHIJIAN"),
                suzhi_graph=delegated.get("SUZHI"),
            )
            graph = AssistantGraph(
                self.retrieval,
                gateway,
                prompt["system_prompt"],
                self._run_step,
                qiming_graph=qiming,
                writing_runner=lambda state: self._run_ai_writing(run, prompt, gateway, state),
                reading_runner=lambda state: self._run_ai_reading(run, prompt, gateway, state),
            )
            return await graph.run(
                AssistantState(
                    request_id=run["request_id"],
                    user_id=str(run["user_id"]),
                    conversation_id=run["conversation_id"],
                    query=run["query"],
                    knowledge_base_id=str(uuid.UUID(int=self.retrieval.knowledge_base_id)),
                    knowledge_base_version=int(run["scope_version"]),
                    history=history,
                    reading_sources=sources,
                    reading_options=run["request_payload"].get("reading", {}),
                    mode=run["mode"],
                    output_language=run["output_language"],
                    thinking=bool(run["request_payload"].get("thinking")),
                    writing_options=run["request_payload"].get("writing", {}),
                    scope_type="KNOWLEDGE_BASE"
                    if run["request_payload"].get("writing")
                    else "DATABASE"
                    if run["database_id"] is not None
                    else "SELECTED"
                    if run["request_payload"].get("corpus_ids")
                    else "GLOBAL",
                    errors=[],
                )
            )
        if code == "LITERATURE":
            history, previous = self._reading_history(run)
            graph = LiteratureGraph(
                self.retrieval,
                gateway,
                prompt["system_prompt"],
                self._run_step,
                analysis_concurrency=get_foreign_settings().literature_analysis_concurrency,
            )
            return await graph.run(
                LiteratureState(
                    request_id=run["request_id"],
                    user_id=str(run["user_id"]),
                    conversation_id=run["conversation_id"],
                    query=run["query"],
                    task=run["request_payload"]["task"],
                    focus_corpus_id=run["request_payload"].get("focus_corpus_id"),
                    history=history,
                    previous_analysis=previous,
                    mode=run["mode"],
                    errors=[],
                )
            )
        # 2. 为研究建立统一身份、来源范围令牌和模式；专业委派仍复用本次运行上下文
        base_state = dict(
            request_id=run["request_id"],
            user_id=str(run["user_id"]),
            conversation_id=run["conversation_id"],
            query=run["query"],
            # 图协议中的旧字段承载范围令牌，资源类型与真实 ID 只由 Scope 解析。
            knowledge_base_id=str(uuid.UUID(int=self.retrieval.knowledge_base_id)),
            knowledge_base_version=int(run["scope_version"]),
            output_language=run["output_language"],
            mode=run["mode"],
            agent_route=[],
            retry_count=0,
            retrieval_filters={},
            errors=[],
        )
        step_runner = self._run_step
        if code == "QIMING":
            # 深度思考随运行冻结：启明的 GenerateAnswer 统一读取该值。
            base_state["thinking"] = bool(dict(run["request_payload"] or {}).get("thinking"))
            # 追问补全：检索与生成都以本轮 query 为唯一来源，所以「继续刚刚的任务」
            # 「把它扩写成 800 字」这类指代必须先结合会话历史改写成语义完整的问题，
            # 否则检索拿不到证据会直接走确定性拒答。任何失败都回落到原始 query。
            history, _ = self._reading_history(run)
            if history:
                base_state["history"] = history
            resolved = await self._resolve_follow_up(run, gateway, history)
            if resolved:
                base_state["query"] = resolved
            delegated = self._delegated_graphs(gateway)
            graph = QimingGraph(
                self.retrieval,
                gateway,
                prompt["system_prompt"],
                step_runner,
                zhijian_graph=delegated.get("ZHIJIAN"),
                suzhi_graph=delegated.get("SUZHI"),
            )
            return await graph.run(QimingState(**base_state))
        if code == "ZHIJIAN":
            graph = ZhijianGraph(self.retrieval, gateway, prompt["system_prompt"], step_runner)
            return await graph.run(ZhijianState(**base_state))
        options = dict(run["request_payload"] or {})
        if code == "SUZHI":
            graph = SuzhiGraph(self.retrieval, gateway, prompt["system_prompt"], step_runner)
            return await graph.run(
                SuzhiState(
                    **base_state,
                    research_topic=run["query"],
                    requested_concept=options.get("concept"),
                    year_from=options.get("year_from"),
                    year_to=options.get("year_to"),
                    review_length=str(options.get("review_length", "MEDIUM")),
                    citation_style=str(options.get("citation_style", "GBT7714")),
                )
            )
        # 3. 写作按阶段恢复所需前序产物，并在独立会话校验归属与章节引用，避免跨会话串用
        if code == "QUILL":
            # 主题用于检索；用户确认的写作规格独立传入综合节点，避免混成研究对象。
            base_state["writing_requirements"] = str(options.get("writing_requirements") or "")
            # 深度思考随运行冻结，图内所有模型调用统一读取该值。
            base_state["thinking"] = bool(options.get("thinking"))
            stage = str(options.get("stage", "THEORY"))
            if stage == "POLISH":
                graph = QuillPolishGraph(gateway, prompt["system_prompt"], step_runner)
                return await graph.run(
                    QuillPolishState(
                        **base_state,
                        selected_text=run["query"],
                        action=str(options["action"]),
                        context_before=str(options.get("context_before") or ""),
                        context_after=str(options.get("context_after") or ""),
                    )
                )
            if stage == "RQ":
                graph = QuillRQGraph(self.retrieval, gateway, prompt["system_prompt"], step_runner)
                return await graph.run(
                    QuillRQState(
                        **base_state,
                        research_topic=run["query"],
                        stage=stage,
                        year_from=options.get("year_from"),
                        year_to=options.get("year_to"),
                        citation_style=str(options.get("citation_style", "GBT7714")),
                    )
                )
            if stage == "PLAN":
                artifact_id = options.get("artifact_id")
                rq_artifact = None
                if artifact_id:
                    with session_factory()() as session:
                        artifact = _load_writing_artifact(
                            session,
                            str(artifact_id),
                            run["conversation_id"],
                            int(run["user_id"]),
                        )
                    rq_artifact = dict(artifact["content"] or {})
                graph = QuillPlanGraph(
                    self.retrieval, gateway, prompt["system_prompt"], step_runner
                )
                return await graph.run(
                    QuillPlanState(
                        **base_state,
                        research_topic=run["query"],
                        stage=stage,
                        year_from=options.get("year_from"),
                        year_to=options.get("year_to"),
                        citation_style=str(options.get("citation_style", "GBT7714")),
                        rq_artifact=rq_artifact,
                        based_on_artifact_id=str(artifact_id) if artifact_id else None,
                    )
                )
            if stage == "DRAFT":
                artifact_id = options.get("artifact_id")
                section_ref = str(options.get("section_ref") or "")
                plan_artifact = None
                section = None
                if artifact_id:
                    with session_factory()() as session:
                        artifact = _load_writing_artifact(
                            session,
                            str(artifact_id),
                            run["conversation_id"],
                            int(run["user_id"]),
                            expect_stage="PLAN",
                        )
                    plan_artifact = dict(artifact["content"] or {})
                    section = next(
                        (
                            item
                            for item in plan_artifact.get("sections") or []
                            if item.get("section_id") == section_ref
                        ),
                        None,
                    )
                    if section is None:
                        raise ValueError("起草的章节编号不在所引用的大纲中")
                graph = QuillDraftGraph(
                    self.retrieval, gateway, prompt["system_prompt"], step_runner
                )
                return await graph.run(
                    QuillDraftState(
                        **base_state,
                        research_topic=run["query"],
                        stage=stage,
                        year_from=options.get("year_from"),
                        year_to=options.get("year_to"),
                        citation_style=str(options.get("citation_style", "GBT7714")),
                        plan_artifact=plan_artifact,
                        based_on_artifact_id=str(artifact_id) if artifact_id else None,
                        section_ref=section_ref,
                        section=section,
                    )
                )
            if stage == "REVISE":
                graph = QuillReviseGraph(
                    self.retrieval, gateway, prompt["system_prompt"], step_runner
                )
                return await graph.run(
                    QuillReviseState(
                        **base_state,
                        research_topic=run["query"][:200],
                        stage=stage,
                        year_from=options.get("year_from"),
                        year_to=options.get("year_to"),
                        citation_style=str(options.get("citation_style", "GBT7714")),
                        revise_instruction=str(options.get("revise_instruction") or ""),
                    )
                )
            graph = QuillGraph(self.retrieval, gateway, prompt["system_prompt"], step_runner)
            return await graph.run(
                QuillState(
                    **base_state,
                    research_topic=run["query"],
                    stage=stage,
                    year_from=options.get("year_from"),
                    year_to=options.get("year_to"),
                    citation_style=str(options.get("citation_style", "GBT7714")),
                    literature_limit=int(options.get("literature_limit", 10)),
                )
            )
        # 4. 其余既有融通入口保留目标语言，图结果交由统一发布事务处理
        graph = RongtongGraph(self.retrieval, gateway, prompt["system_prompt"], step_runner)
        return await graph.run(
            RongtongState(
                **base_state,
                policy_or_concept=run["query"],
                target_language=str(options["target_language"]),
            )
        )

    async def _run_ai_reading(
        self, run: dict, prompt: dict, gateway: Any, state: AssistantState
    ) -> dict:
        """在原AI运行内解析文献子集并调用阅读图，来源及结果随当前会话保存。

        子集来自父 Run 当前授权的资源或明确选文；材料工具只读取这些篇目，
        发布使用取得的真实证据快照，不重新比较父资源成员或材料版本。

        Args:
            run: 已加载并授权的当前AI运行。
            prompt: 本轮冻结的提示词配置。
            gateway: 本轮固定的模型网关。
            state: 包含真实来源及结构化指代计划的AI图状态。

        Returns:
            阅读结果、已校验引用和可恢复的范围快照，或待澄清说明。
        """
        assert self.retrieval is not None

        async def resolve():
            """在线程内校验来源并冻结阅读范围，返回可供步骤审计的快照。"""
            def freeze():
                """短事务读取题录和同范围分析，不在模型调用期间占用连接。"""
                with session_factory()() as session:
                    scope = freeze_reading_scope(
                        session, run, state, self.retrieval.fixed_scope
                    )
                    previous = previous_reading_analysis(session, run, scope) if (
                        scope["status"] == "READY"
                    ) else None
                    return {"reading_scope": scope, "previous_analysis": previous}
            return await asyncio.to_thread(freeze)

        # 1. 在正文读取前单独提交范围快照；无来源或有歧义时只返回澄清。
        resolved = await self._run_step("ResolveReadingScope", state, resolve)
        scope = resolved["reading_scope"]
        if scope["status"] != "READY":
            return {
                "reading_answer": scope["message"], "reading_result": scope,
                "evidence_level": "INSUFFICIENT", "agent_route": ["AI_ASSISTANT"],
                "output_language": "ZH",
            }
        tool = LiteratureMaterialTool(
            actor_id=int(run["user_id"]), knowledge_base_id=0,
            fixed_scope=scope["corpus_ids"], fixed_version=scope["scope_version"],
            settings=get_foreign_settings(),
        )

        async def reading_step(name, child_state, operation):
            """复用当前Run的步骤事务，在保存证据前同步子工具核验过的来源。"""
            async def with_sources():
                """子节点产生证据时立即交给父发布层验证，不等整个阅读图结束。"""
                output = await operation()
                self.retrieval.sources.update(tool.sources)
                return output
            return await self._run_step(name, child_state, with_sources)

        # 2. ASK可直接读取原文回答；AI入口不要求先执行整套分析，独立阅读页规则保持原样。
        if self.model_config is not None:
            gateway = get_model_gateway(
                self.model_config["provider"], model_name=self.model_config["model_name"],
                api_base=self.model_config["api_base"],
                timeout_seconds=max(180, get_settings().research_model_timeout_seconds),
                max_tokens=16000, temperature=0.0,
            )
        graph = LiteratureGraph(
            tool, gateway, prompt["system_prompt"], reading_step,
            analysis_concurrency=get_foreign_settings().literature_analysis_concurrency,
        )
        result = await graph.run(LiteratureState(
            request_id=run["request_id"], user_id=str(run["user_id"]),
            conversation_id=run["conversation_id"], query=state["query"],
            task=scope["task"], history=state.get("history", []),
            previous_analysis=resolved["previous_analysis"], mode=run["mode"], errors=[],
        ))
        # 3. 保留阅读结构和材料来源；综合分析允许无逐句引用，追问沿用原图引用校验。
        return {
            "reading_answer": result["final_answer"],
            "output_language": "ZH",
            "reading_result": {**result["literature_result"], **scope, "status": "SUCCESS"},
            "evidence": result.get("evidence", []),
            "citations": result.get("citations", []),
            "validated_citations": result.get("validated_citations", []),
            "evidence_level": result.get("evidence_level", "INSUFFICIENT"),
            "agent_route": ["AI_ASSISTANT", "LITERATURE"],
        }

    async def _run_ai_writing(
        self, run: dict, prompt: dict, gateway: Any, state: AssistantState
    ) -> dict:
        """在原 AI Run 内复用 Quill 子图，产物和引用仍由统一发布事务保存。"""
        options = dict(state["writing_options"])

        def load_source():
            """在短会话中再次验证前序产物，不跨模型等待持有数据库连接。"""
            with session_factory()() as session:
                return writing_artifact(
                    session, run["conversation_id"], int(run["user_id"]), options
                )

        # 1. 起草与修改只读取明确且同库的前序版本，原稿绝不由分类模型重建。
        source = await asyncio.to_thread(load_source)
        query = state["query"]
        quill_options = {
            key: options[key] for key in ("stage", "artifact_id", "section_ref") if options.get(key)
        }
        quill_options["thinking"] = bool(run["request_payload"].get("thinking"))
        quill_options["writing_requirements"] = options.get("writing_requirements", "")
        if options["stage"] == "REVISE":
            query = (
                options.get("source_text")
                or (source or {}).get("revised_text")
                or (source or {}).get("draft_text", "")
            )
            quill_options.pop("artifact_id", None)
            quill_options["revise_instruction"] = (
                options.get("revise_instruction") or state["query"][:500]
            )
        # 2. 仅替换子图输入，不创建 WRITING 会话；检索工具、Prompt、模型、步骤仍绑定当前 Run。
        result = await self._run_graph(
            {
                **run,
                "entry_source": "WRITING",
                "executor_agent_code": "QUILL",
                "query": query,
                "request_payload": quill_options,
            },
            prompt,
            gateway,
        )
        payload = next(
            result[key]
            for key in ("rq_result", "plan_result", "draft_result", "revise_result", "quill_result")
            if result.get(key) is not None
        )
        payload = (
            payload.model_dump(mode="json") if hasattr(payload, "model_dump") else dict(payload)
        )
        payload["knowledge_base_id"] = options["knowledge_base_id"]
        payload["writing_requirements"] = options.get("writing_requirements", "")
        payload["research_topic"] = state["query"]
        for key in ("source_run_id", "source_action"):
            if options.get(key):
                payload[key] = options[key]
        if options.get("artifact_id"):
            payload["based_on_artifact_id"] = options["artifact_id"]
        # 3. 只投影子图已校验的输出，不在汇合节点再改写引用正文。
        return {
            "writing_answer": result["final_answer"],
            "writing_result": payload,
            **{
                key: result.get(key, [])
                for key in ("evidence", "citations", "validated_citations", "used_citation_indices")
            },
            "evidence_level": result.get("evidence_level"),
            "agent_route": ["AI_ASSISTANT", "QUILL"],
        }

    def _reading_history(self, run: dict) -> tuple[list[dict], dict | None]:
        """在字节预算内组装最近成功对话，并单独恢复最新一次分析摘要。

        Args:
            run: 当前运行的会话、用户及 ID，用于隔离归属并排除本次尚未完成的结果。

        Returns:
            按时间正序排列的完整问答对，以及可选的最新 ANALYZE 概览和逐篇首段摘要。
            历史只用于指代解析和背景理解，不能代替本轮重新读取的原文 Evidence。
        """
        # 1. 倒序读取当前用户同一会话的成功轮次；失败和本次运行都不进入历史上下文。
        with session_factory()() as session:
            rows = (
                session.execute(
                    select(research_run)
                    .where(
                        research_run.c.conversation_id == run["conversation_id"],
                        research_run.c.user_id == run["user_id"],
                        research_run.c.status == "SUCCESS",
                        research_run.c.is_deleted == 0,
                        research_run.c.id != run["id"],
                    )
                    .order_by(research_run.c.created_at.desc(), research_run.c.id.desc())
                )
                .mappings()
                .all()
            )
        # 2. 从新到旧保留完整问答对；历史满额后仍继续查找较早的最新分析，避免丢失阅读背景。
        history, size, previous = [], 0, None
        history_full = False
        for row in rows:
            payload = (row["result_payload"] or {}).get("payload", {})
            if payload.get("task") == "ANALYZE" and previous is None:
                # 只保留概览和逐篇摘要，完整模板始终可从原 Run 恢复。
                previous = {
                    "overview": payload.get("overview"),
                    "documents": [
                        {
                            "corpus_id": item["corpus_id"],
                            "title": item["title"][:200],
                            "summary": (item.get("analysis") or {}).get("summary", [])[:1],
                        }
                        for item in payload.get("documents", [])
                    ],
                }
            # 3. 分析轮只使用概览，超长追问只保留前几段；不把整篇六分区模板反复塞入模型输入。
            assistant_content = row["final_answer"]
            if row["result_payload"] and row["result_payload"].get("result_type") == "AI_ASSISTANT":
                assistant_content = json.dumps(
                    {
                        "answer": (row["final_answer"] or "")[:3000],
                        "writing": writing_history_summary(payload.get("writing_result")),
                        "documents": [
                            {
                                "rank": index + 1,
                                "corpus_id": item["corpus_id"],
                                "title": item["title"],
                            }
                            for index, item in enumerate(payload.get("documents", []))
                        ],
                    },
                    ensure_ascii=False,
                )
            elif payload.get("task") == "ANALYZE":
                assistant_content = json.dumps(payload.get("overview"), ensure_ascii=False)
            elif len((assistant_content or "").encode()) > MAX_HISTORY_BYTES:
                assistant_content = "\n\n".join(
                    item["text"] for item in payload.get("answer", {}).get("answer", [])[:3]
                )
            pair = [
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": assistant_content},
            ]
            pair_size = len(json.dumps(pair, ensure_ascii=False).encode())
            # 每对一起纳入或舍弃，再从头插入，既限制预算也恢复用户可理解的时间顺序。
            if not history_full and size + pair_size <= MAX_HISTORY_BYTES:
                history[0:0] = pair
                size += pair_size
            else:
                history_full = True
        return history, previous

    def _latest_unfinished_query(self, run: dict) -> str:
        """取同会话最近一次失败/取消运行的原始提问。

        这些运行不会进入成功历史，但用户说「继续刚刚的任务」时，所指往往正是
        被停止的那一轮，因此单独取出来作为改写上下文。

        Args:
            run: 当前运行行（用于限定会话、用户并排除自身）。

        Returns:
            最近一次未完成运行的提问；没有则返回空串。
        """
        with session_factory()() as session:
            row = session.execute(
                select(research_run.c.query)
                .where(
                    research_run.c.conversation_id == run["conversation_id"],
                    research_run.c.user_id == run["user_id"],
                    research_run.c.status.in_(("FAILED", "CANCELLED")),
                    research_run.c.is_deleted == 0,
                    research_run.c.id != run["id"],
                )
                .order_by(research_run.c.created_at.desc(), research_run.c.id.desc())
                .limit(1)
            ).first()
        return (row[0] or "").strip() if row and row[0] else ""

    async def _resolve_follow_up(
        self,
        run: dict,
        gateway: Any,
        history: list[dict[str, str]],
    ) -> str:
        """把指代性追问改写成本轮自包含的问题。

        检索与生成都以本轮 query 为唯一来源，因此「继续刚刚的任务」「把它扩写成
        800 字」这类追问若不补全，检索必然拿不到证据（会走确定性拒答）。这里用
        当前运行的同一模型把追问改写为独立问题。

        Args:
            run: 当前运行行，提供本轮原始 query。
            gateway: 当前运行固定的模型网关。
            history: 同会话此前的问答对（按时间正序）。

        Returns:
            改写后的自包含问题；无需改写或改写失败时返回空串，由调用方回落到原始 query。
        """
        query = (run["query"] or "").strip()
        if not query:
            return ""
        # 只有出现强续做词、短句里的指代词，或问题短到无法独立成立时才改写
        if not _looks_like_follow_up(query):
            return ""
        # 只取最近两轮（最多 4 条消息），助手回答截断，避免长文反复进入模型输入
        lines = [
            f"{'用户' if item.get('role') == 'user' else '助手'}："
            f"{(item.get('content') or '')[:800]}"
            for item in history[-4:]
            if (item.get("content") or "").strip()
        ]
        # 最近一次被停止/失败的运行不进入成功历史，但它的提问往往正是「继续」的对象
        unfinished = self._latest_unfinished_query(run)
        if unfinished and unfinished != query:
            lines.append(f"用户（上一轮，未完成）：{unfinished}")
        transcript = "\n".join(lines)
        if not transcript:
            return ""
        instruction = (
            "下面是同一会话的对话记录，以及用户最新的一句话。\n"
            "请把最新这句话改写成不依赖上下文也能独立理解的中文问题：\n"
            "1) 补全「这个/它/上面/继续」等指代，明确指向对话中的具体对象；\n"
            "2) 保留用户的最新诉求（继续某阶段、扩写、缩短、换角度等）；\n"
            "3) 不要编造对话中未出现的事实，也不要回答问题本身；\n"
            "4) 只输出改写后的问题，20~150 字，不要任何前缀或解释。\n\n"
            f"对话记录：\n{transcript}\n\n最新一句：{query}"
        )
        try:
            resolved = await gateway.structured(
                instruction=instruction,
                schema=ResolvedFollowUpQuery,
                system_prompt="你是检索查询改写助手，只输出改写后的问题。",
                thinking=False,
            )
        except Exception:
            # 抽取式 Provider 不支持结构化生成、或模型超时：安全回落到原始问题
            return ""
        text = (resolved.query or "").strip()
        if len(text) < 4 or text == query:
            return ""
        return text[:3900]

    def _delegated_graphs(self, gateway: Any) -> dict[str, Any]:
        """按冻结版本构造启明允许自动委派的专业图。

        Args:
            gateway: 当前运行固定的模型网关。

        Returns:
            以专业编码为键的图实例。
        """
        assert self.retrieval is not None
        graphs: dict[str, Any] = {}
        with session_factory()() as session:
            for code, graph_type in (
                ("ZHIJIAN", ZhijianGraph),
                ("SUZHI", SuzhiGraph),
            ):
                try:
                    _, prompt, _ = _prompt_and_model(session, code)
                except LookupError:
                    continue
                graphs[code] = graph_type(
                    self.retrieval, gateway, prompt["system_prompt"], self._run_step
                )
        return graphs

    async def _run_step(
        self,
        node_name: str,
        state: dict[str, Any],
        operation: Any,
    ) -> dict[str, Any]:
        """持久化节点开始、结果和真实证据，失败时留下脱敏步骤。

        Args:
            node_name: LangGraph 稳定节点名。
            state: 当前图状态，仅提取计数作为摘要。
            operation: 真实节点异步业务动作。

        Returns:
            节点产生的状态增量。

        Raises:
            Exception: 原节点失败；运行入口负责写失败终态。
        """
        # 并行 A/B 检索节点可能同时开始；只在分配 sequence 时加锁，模型与
        # Qdrant 调用仍可并发，唯一约束不会因为步骤竞争而误报业务失败。
        # 1. 仅在分配步骤序号时串行，并先提交 RUNNING，外部操作不占用该锁
        async with self.step_lock:
            with session_factory()() as session:
                sequence = (
                    int(
                        session.scalar(
                            select(func.coalesce(func.max(research_step.c.sequence), 0)).where(
                                research_step.c.run_id == self.run_id
                            )
                        )
                        or 0
                    )
                    + 1
                )
                step_id = _new_id()
                session.execute(
                    insert(research_step).values(
                        id=step_id,
                        run_id=self.run_id,
                        sequence=sequence,
                        node_name=node_name,
                        status="RUNNING",
                        input_summary={"evidence_count": len(state.get("evidence", []))},
                        is_deleted=False,
                    )
                )
                session.commit()
        try:
            # 捕获本节点产生的思维链：thinking 流式调用按增量实时进入运行级缓冲
            # （_LIVE_REASONING，供轮询端点在节点执行期间读取），聚合文本随步骤落库；
            # 非思考模式桶为空，输出摘要保持原样。
            # 2. 为本协程绑定实时增量缓冲，执行后无论成功或异常都恢复 ContextVar
            with _LIVE_LOCK:
                live_buffer = _LIVE_REASONING.setdefault(self.run_id, [])
                live_answer = _LIVE_ANSWER.setdefault(self.run_id, [])
                _LIVE_STEP[self.run_id] = node_name
            sink = _LiveReasoningSink(live_buffer)
            reasoning_token = reasoning_sink.set(sink)
            answer_token = answer_sink.set(_LiveReasoningSink(live_answer))
            try:
                output = await operation()
            finally:
                reasoning_sink.reset(reasoning_token)
                answer_sink.reset(answer_token)
            # 3. 只从实际节点输出提取摘要和证据，与步骤 SUCCESS 在独立事务一起提交
            output_summary = _step_output_summary(output, state)
            if node_name == "RecommendLiterature":
                output_summary["web_recommendation_details"] = output.get(
                    "web_recommendation_details",
                )
            if "reading_scope" in output:
                output_summary["reading_scope"] = output["reading_scope"]
            thinking_text = "\n\n".join(part.strip() for part in sink if part and part.strip())
            if thinking_text:
                # 思维链只做过程展示，截断防止 output_summary 无界膨胀
                output_summary["thinking"] = thinking_text[:6000]
            with session_factory()() as session:
                self._save_evidence(session, output)
                session.execute(
                    update(research_step)
                    .where(research_step.c.id == step_id)
                    .values(
                        status="SUCCESS",
                        finished_at=_now(),
                        output_summary=output_summary,
                    )
                )
                session.commit()
            return output
        except (Exception, asyncio.CancelledError) as exc:
            # 4. 异常或取消在新的短事务记录终态，再抛原异常供运行级收尾
            with session_factory()() as session:
                session.execute(
                    update(research_step)
                    .where(research_step.c.id == step_id)
                    .values(
                        status="CANCELLED" if isinstance(exc, asyncio.CancelledError) else "FAILED",
                        finished_at=_now(),
                        # 网关异常消息均为脱敏文案，记录类型+消息便于远程诊断
                        error_message=(f"{type(exc).__name__}: {exc}")[:500] or type(exc).__name__,
                    )
                )
                session.commit()
            raise

    def _save_evidence(self, session: Session, output: dict[str, Any]) -> None:
        """只把受控检索工具实际返回的片段写为运行证据。

        Args:
            session: 步骤完成事务会话。
            output: 当前节点输出，可能包含 Evidence 与智鉴归属。

        Raises:
            ValueError: 图引用了不属于本次检索结果的片段。
        """
        assert self.retrieval is not None
        assignments = {
            str(item.get("chunk_id")): item
            for item in output.get("evidence_assignments", [])
            if isinstance(item, dict)
        }
        for item in output.get("evidence", []):
            source = self.retrieval.sources.get(item.chunk_id)
            if source is None or source.corpus_id != item.document_id.int:
                raise ValueError("图产生了未由受控检索返回的证据")
            existing = session.execute(
                select(research_evidence.c.id).where(
                    research_evidence.c.run_id == self.run_id,
                    research_evidence.c.collection_name == get_foreign_settings().qdrant_collection,
                    research_evidence.c.point_id == source.point_id,
                )
            ).first()
            if existing:
                continue
            assignment = assignments.get(str(item.chunk_id), {})
            session.execute(
                insert(research_evidence).values(
                    id=_new_id(),
                    run_id=self.run_id,
                    corpus_id=source.corpus_id,
                    collection_name=get_foreign_settings().qdrant_collection,
                    point_id=source.point_id,
                    chunk_key=source.chunk_key,
                    file_path=source.file_path,
                    page_start=source.page_start,
                    page_end=source.page_end,
                    content_snapshot=source.content,
                    content_sha256=hashlib.sha256(source.content.encode()).hexdigest(),
                    retrieval_score=source.score,
                    relevance_score=item.relevance_score,
                    evidence_rank=item.citation_index,
                    assignment=assignment.get("assignment"),
                    assignment_payload=json.loads(json.dumps(assignment, default=str)),
                    is_deleted=False,
                )
            )

    def _publish(self, run: dict, final: dict[str, Any]) -> None:
        """基于检索时取得的证据快照，原子写入回答、引用及成功终态。

        Args:
            run: 已固定运行。
            final: 原图审核完成的最终状态。

        Raises:
            ValueError: 缺少最终结果、引用不属于本次检索或证据快照不一致。
        """
        assert self.retrieval is not None
        # 1. 检查最终正文及各入口的引用要求，不把中间草稿当成可发布回答
        answer = final.get("final_answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Agent 图未发布最终回答")
        citations = list(final.get("validated_citations", []))
        if (
            run["executor_agent_code"] != "LITERATURE"
            and final.get("evidence_level") != "INSUFFICIENT"
            # 搜索与未执行的任务确认只发布说明；实际知识库写作仍要求有效引用。
            and not (
                run["entry_source"] == "AI"
                and (
                    final.get("intent") == "SEARCH"
                    or (
                        final.get("intent") == "READING"
                        and (final.get("reading_result") or {}).get("status") == "SUCCESS"
                        and final["reading_result"].get("task") == "ANALYZE"
                    )
                    or (
                        (final.get("writing_result") or {}).get("status") == "NEEDS_INPUT"
                        and not (run.get("request_payload") or {}).get("writing")
                    )
                )
            )
            and not citations
        ):
            raise ValueError("研究结论没有通过 Citation 校验")
        # 2. 本轮沿用已经授权检索得到的快照；库成员、题录、附件后续变化不阻止发布。
        # 不扫描整库或重读文献主表，历史引用保留当时取得的正文、页码和摘录。
        with session_factory()() as session:
            message_id = _new_id()
            # 3. 引用必须来自本轮受控检索及已持久化证据，防止模型拼造来源或摘录。
            session.execute(
                insert(research_message).values(
                    id=message_id,
                    conversation_id=run["conversation_id"],
                    run_id=self.run_id,
                    role="ASSISTANT",
                    content=answer,
                    is_deleted=False,
                )
            )
            for citation in citations:
                source = self.retrieval.sources.get(citation.chunk_id)
                if not source or source.corpus_id != citation.document_id.int:
                    raise ValueError("回答引用不属于本次受控检索")
                normalized_source = " ".join(source.content.split())
                if " ".join(citation.quote.split()) not in normalized_source:
                    raise ValueError("回答引用摘录与来源原文不符")
                evidence = (
                    session.execute(
                        select(research_evidence).where(
                            research_evidence.c.run_id == self.run_id,
                            research_evidence.c.point_id == source.point_id,
                            research_evidence.c.is_deleted == 0,
                        )
                    )
                    .mappings()
                    .first()
                )
                if not evidence:
                    raise ValueError("回答引用的运行证据尚未持久化")
                if (
                    evidence["corpus_id"] != source.corpus_id
                    or evidence["file_path"] != source.file_path
                    or evidence["content_sha256"]
                    != hashlib.sha256(source.content.encode()).hexdigest()
                ):
                    raise ValueError("回答引用的证据快照与检索来源不一致")
                session.execute(
                    insert(research_citation).values(
                        id=_new_id(),
                        message_id=message_id,
                        evidence_id=evidence["id"],
                        citation_index=citation.citation_index,
                        quote_text=citation.quote,
                        is_deleted=False,
                    )
                )
            # 唯一回答消息、全部 Citation 和 SUCCESS 在同一事务可见。
            # 4. 按业务类型封装结构化结果，写作产物与本次回答共用当前事务
            result = (
                {
                    "result_type": "LITERATURE_READING",
                    "schema_version": 1,
                    "payload": final["literature_result"],
                }
                if "literature_result" in final
                else None
            )
            for key, result_type, schema_version in (
                ("assistant_result", "AI_ASSISTANT", 1),
                ("zhijian_result", "ZHIJIAN_COMPARISON", 3),
                ("suzhi_result", "SUZHI_LITERATURE_REVIEW", 4),
                ("rongtong_result", "RONGTONG_POLICY_COMMUNICATION", 3),
                ("quill_result", "QUILL_WRITING_ASSISTANCE", 1),
                ("rq_result", "QUILL_WRITING_ASSISTANCE", 1),
                ("plan_result", "QUILL_WRITING_ASSISTANCE", 1),
                ("draft_result", "QUILL_WRITING_ASSISTANCE", 1),
                ("revise_result", "QUILL_WRITING_ASSISTANCE", 1),
                ("polish_result", "QUILL_WRITING_ASSISTANCE", 1),
            ):
                value = final.get(key)
                if value is not None:
                    result = {
                        "result_type": result_type,
                        "schema_version": schema_version,
                        "payload": value
                        if isinstance(value, dict)
                        else value.model_dump(mode="json"),
                    }
                    break
            # Quill RQ/PLAN/DRAFT/REVISE 结果同步持久化为阶段产物，供会话内后续
            # 阶段引用；产物主键回填进结果快照，前端据此串联完整写作流程。
            if result and result["result_type"] == "AI_ASSISTANT":
                writing = result["payload"].get("writing_result") or {}
                if (
                    writing.get("stage") in {"RQ", "PLAN", "DRAFT", "REVISE"}
                    and writing.get("status") == "SUCCESS"
                    and writing.get("evidence_level") != "INSUFFICIENT"
                ):
                    writing["artifact_id"] = _persist_writing_artifact(
                        session,
                        run_id=self.run_id,
                        conversation_id=run["conversation_id"],
                        payload=writing,
                        used_citation_indices=sorted(final.get("used_citation_indices", [])),
                    )
            if (
                run["executor_agent_code"] == "QUILL"
                and result is not None
                and result["payload"].get("stage") in {"RQ", "PLAN", "DRAFT", "REVISE"}
            ):
                artifact_id = _persist_writing_artifact(
                    session,
                    run_id=self.run_id,
                    conversation_id=run["conversation_id"],
                    payload=result["payload"],
                    used_citation_indices=sorted(final.get("used_citation_indices", [])),
                )
                result["payload"]["artifact_id"] = artifact_id
            route = list(final.get("agent_route") or [run["executor_agent_code"]])
            executor = next(
                (code for code in reversed(route) if code in AGENT_CODES),
                run["executor_agent_code"],
            )
            # 5. 条件推进仍在运行的记录为 SUCCESS，消息、引用、结果与会话更新时间原子提交
            # 条件更新未命中时抛错，整个事务回滚，防止运行已取消却仍发布一条回答。
            published = session.execute(
                update(research_run)
                .where(
                    research_run.c.id == self.run_id,
                    research_run.c.status == "RUNNING",
                    research_run.c.is_deleted == 0,
                )
                .values(
                    status="SUCCESS",
                    finished_at=_now(),
                    final_answer=answer,
                    result_payload=result,
                    executor_agent_code=executor,
                    intent=final.get("intent"),
                    evidence_level=final.get("evidence_level"),
                    output_language=final.get("output_language", run["output_language"]),
                )
            )
            if published.rowcount != 1:
                raise ValueError("运行状态已改变，取消发布回答")
            session.execute(
                update(research_conversation)
                .where(research_conversation.c.id == run["conversation_id"])
                .values(updated_at=_now())
            )
            session.commit()

    def _fail(self, exc: Exception) -> None:
        """把失败安全落库，不发布图的中间草稿或供应商响应。

        Args:
            exc: 图、检索或持久化异常。
        """
        with session_factory()() as session:
            session.execute(
                update(research_run)
                .where(
                    research_run.c.id == self.run_id,
                    research_run.c.status == "RUNNING",
                )
                .values(
                    status="FAILED",
                    finished_at=_now(),
                    error_code=type(exc).__name__[:128],
                    error_message="智能体运行失败；请查看步骤状态或重新发起任务",
                )
            )
            session.commit()


def cancel_run(session: Session, run_id: str, actor: ForeignActor) -> dict[str, object]:
    """把尚未结束的运行标记为 CANCELLED（幂等）。

    只更新 PENDING/RUNNING 的记录，已进入终态的运行原样返回当前状态。
    运行协程后续的「发布回答」（WHERE status='RUNNING'）与失败落库都是条件更新，
    因此取消后不会把半成品写成正式回答，也不会被 FAILED 覆盖。

    Args:
        session: 共库会话。
        run_id: 运行 ID。
        actor: 当前 Foreign 用户。

    Returns:
        {"run_id": …, "status": …}，status 为取消后的真实状态。

    Raises:
        LookupError: 运行不存在或已删除。
        PermissionError: 非运行所有者。
    """
    row = (
        session.execute(
            select(research_run.c.user_id, research_run.c.status).where(
                research_run.c.id == run_id,
                research_run.c.is_deleted == 0,
            )
        )
        .mappings()
        .first()
    )
    if not row:
        raise LookupError("运行不存在")
    if row["user_id"] != actor.id:
        raise PermissionError("无权操作该运行")
    if row["status"] in TERMINAL_STATUSES:
        return {"run_id": run_id, "status": row["status"]}
    session.execute(
        update(research_run)
        .where(
            research_run.c.id == run_id,
            research_run.c.status.in_(("PENDING", "RUNNING")),
        )
        .values(status="CANCELLED", finished_at=_now(), error_message="用户已停止生成")
    )
    session.commit()
    session.expire_all()
    current = session.scalar(select(research_run.c.status).where(research_run.c.id == run_id))
    return {"run_id": run_id, "status": current or "CANCELLED"}


def read_live_reasoning(session: Session, run_id: str, actor: ForeignActor) -> dict[str, object]:
    """读取运行中节点的实时思维链（内存缓冲，仅运行所有者可读）。

    Args:
        session: 共库会话，仅用于运行归属校验。
        run_id: 运行 ID。
        actor: 当前 Foreign 用户。

    Returns:
        {"node": 当前执行节点名, "thinking": 已产出的思维链文本（尾部 6000 字）}。

    Raises:
        LookupError: 运行不存在或已伪删除。
        PermissionError: 当前用户无权读取。
    """
    owner_id = session.scalar(
        select(research_run.c.user_id).where(
            research_run.c.id == run_id,
            research_run.c.is_deleted == 0,
        )
    )
    if owner_id is None:
        raise LookupError("运行不存在")
    if owner_id != actor.id:
        raise PermissionError("无权读取该运行")
    with _LIVE_LOCK:
        buffer = _LIVE_REASONING.get(run_id) or []
        answer_buffer = _LIVE_ANSWER.get(run_id) or []
        node = _LIVE_STEP.get(run_id)
    return {
        "node": node,
        "thinking": "".join(buffer)[-6000:],
        "answer": "".join(answer_buffer)[-6000:],
    }


def read_run(session: Session, run_id: str, actor: ForeignActor) -> dict[str, object]:
    """校验所有者后读取历史运行与步骤，知识库 ID 从所属会话派生。

    历史读取不关联知识库主表，因此知识库删除或撤权不会阻止所有者查看历史。

    Args:
        session: 请求事务会话。
        run_id: 运行标识。
        actor: 当前 Foreign 用户。

    Returns:
        可对外展示的运行投影。

    Raises:
        LookupError: 运行或所属会话不存在、已伪删除或所有者不一致。
        PermissionError: 当前用户无权读取。
    """
    # 1. 加载运行和会话绑定，保留接口中的派生知识库字段并校验历史所有者。
    run = _load_run_with_conversation(session, run_id)
    if run["user_id"] != actor.id:
        raise PermissionError("无权读取该运行")
    active_user = session.execute(
        select(foreign_user.c.id).where(
            foreign_user.c.id == actor.id,
            foreign_user.c.status == 1,
            foreign_user.c.is_deleted == 0,
        )
    ).first()
    if active_user is None:
        raise PermissionError("Foreign 用户已停用")
    # 2. 读取本次运行的入口 Agent 和步骤，不用会话最后一次路由覆盖历史。
    entry_agent = session.scalar(
        select(research_agent.c.code).where(research_agent.c.id == run["agent_id"])
    )
    steps = (
        session.execute(
            select(research_step)
            .where(
                research_step.c.run_id == run_id,
                research_step.c.is_deleted == 0,
            )
            .order_by(research_step.c.sequence)
        )
        .mappings()
        .all()
    )
    return {
        "id": run["id"],
        "request_id": run["request_id"],
        "conversation_id": run["conversation_id"],
        "knowledge_base_id": run["knowledge_base_id"],
        "database_id": run["database_id"],
        "entry_source": run["entry_source"],
        "scope_version": run["scope_version"],
        "status": run["status"],
        "query": run["query"],
        "mode": run["mode"],
        "output_language": run["output_language"],
        "entry_agent": entry_agent,
        "executor_agent": run["executor_agent_code"],
        "intent": run["intent"],
        "evidence_level": run["evidence_level"],
        "final_answer": run["final_answer"],
        "result_payload": run["result_payload"],
        "request_payload": run["request_payload"],
        "error_code": run["error_code"],
        "error_message": run["error_message"],
        "steps": [dict(step) for step in steps],
        "created_at": run["created_at"],
        "finished_at": run["finished_at"],
    }


def _authorize_history_delete(session: Session, owner_id: int, actor: ForeignActor) -> None:
    """允许有效用户清理自己的历史，不要求原知识库仍可访问。

    Args:
        session: 请求事务。
        owner_id: 历史记录所有者。
        actor: 当前用户。

    Raises:
        PermissionError: 用户已停用或记录属于他人。
    """
    if owner_id != actor.id:
        raise PermissionError("无权删除该历史记录")
    active = session.scalar(
        select(foreign_user.c.id).where(
            foreign_user.c.id == actor.id,
            foreign_user.c.status == 1,
            foreign_user.c.is_deleted == 0,
        )
    )
    if active is None:
        raise PermissionError("Foreign 用户已停用")


def soft_delete_run(
    session: Session,
    run_id: str,
    actor: ForeignActor,
    *,
    commit: bool = True,
) -> None:
    """伪删除已结束运行及其消息、步骤、证据和引用。

    Args:
        session: 共库请求事务。
        run_id: 待删除运行。
        actor: 当前 Foreign 用户。
        commit: 是否由当前函数提交；删除整个会话时由上层统一提交。

    Raises:
        LookupError: 运行不存在或已删除。
        PermissionError: 用户无效或不是运行所有者。
        ValueError: 运行仍在执行。
    """
    # 删除不读取结果/证据，也不复用要求知识库权限的 read_run。
    run = (
        session.execute(
            select(research_run)
            .where(
                research_run.c.id == run_id,
                research_run.c.is_deleted == 0,
            )
            .with_for_update()
        )
        .mappings()
        .first()
    )
    if run is None:
        raise LookupError("运行不存在")
    _authorize_history_delete(session, run["user_id"], actor)
    if run["status"] not in TERMINAL_STATUSES:
        raise ValueError("运行中任务不能删除")
    message_ids = list(
        session.execute(
            select(research_message.c.id).where(
                research_message.c.run_id == run_id,
            )
        ).scalars()
    )
    if message_ids:
        session.execute(
            update(research_citation)
            .where(research_citation.c.message_id.in_(message_ids))
            .values(is_deleted=True)
        )
    for table in (research_message, research_step, research_evidence):
        session.execute(update(table).where(table.c.run_id == run_id).values(is_deleted=True))
    session.execute(update(research_run).where(research_run.c.id == run_id).values(is_deleted=True))
    if commit:
        session.commit()


def soft_delete_conversation(
    session: Session,
    conversation_id: str,
    actor: ForeignActor,
) -> None:
    """伪删除会话及全部结束运行，活动运行阻止删除。

    Args:
        session: 共库请求事务。
        conversation_id: 用户会话标识。
        actor: 已认证 Foreign 用户。

    Raises:
        LookupError: 会话不存在。
        PermissionError: 会话越权。
        ValueError: 会话包含活动运行。
    """
    conversation = (
        session.execute(
            select(research_conversation).where(
                research_conversation.c.id == conversation_id,
                research_conversation.c.is_deleted == 0,
            )
        )
        .mappings()
        .first()
    )
    if not conversation:
        raise LookupError("会话不存在")
    _authorize_history_delete(session, conversation["user_id"], actor)
    runs = session.execute(
        select(research_run.c.id, research_run.c.status).where(
            research_run.c.conversation_id == conversation_id,
            research_run.c.is_deleted == 0,
        )
    ).all()
    if any(status not in TERMINAL_STATUSES for _, status in runs):
        raise ValueError("会话包含运行中任务")
    for run_id, _ in runs:
        soft_delete_run(session, run_id, actor, commit=False)
    session.execute(
        update(research_conversation)
        .where(research_conversation.c.id == conversation_id)
        .values(is_deleted=True)
    )
    session.commit()


def recover_stale_runs(session: Session) -> int:
    """将服务重启前滞留的旧后台运行标记为失败，避免永久转圈。

    Args:
        session: 启动阶段独立事务会话。

    Returns:
        本次恢复的运行条数。
    """
    cutoff = _now() - timedelta(minutes=20)
    result = session.execute(
        update(research_run)
        .where(
            research_run.c.status == "RUNNING",
            research_run.c.started_at < cutoff,
            research_run.c.is_deleted == 0,
        )
        .values(
            status="FAILED",
            finished_at=_now(),
            error_code="AGENT_PROCESS_INTERRUPTED",
            error_message="服务中断，运行未完成，请重新发起任务",
        )
    )
    session.commit()
    return int(result.rowcount or 0)
