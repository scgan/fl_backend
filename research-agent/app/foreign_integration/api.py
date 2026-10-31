"""只暴露 Foreign 共库 Agent、会话、证据和运行状态接口。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from io import BytesIO
from typing import Annotated, Any, Literal
from urllib.parse import urlencode

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import case, func, select, update
from sqlalchemy.orm import Session

from app.agents.ai_assistant_schemas import (
    AssistantReadingContext,
    AssistantReadingOptions,
    AssistantWritingOptions,
)
from app.common.responses import envelope
from app.foreign_integration.auth import ForeignActor, current_actor
from app.foreign_integration.config import get_foreign_settings
from app.foreign_integration.database import (
    get_session,
    session_factory,
)
from app.foreign_integration.docx_export import build_document_from_blocks, build_quill_document
from app.foreign_integration.research_scope import (
    ResearchScope,
    accessible_scopes,
    root_databases,
)
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_user,
    research_agent,
    research_citation,
    research_conversation,
    research_evidence,
    research_message,
    research_run,
    research_writing_artifact,
)
from app.foreign_integration.service import (
    RunExecutor,
    cancel_run,
    create_run,
    normalize_entry_source,
    read_live_reasoning,
    read_run,
    resolve_entry_source,
    soft_delete_conversation,
    soft_delete_run,
)

# 运行中的协程登记表：run_id -> asyncio.Task，供「停止生成」在当前进程内立即取消。
# 跨进程/重启后表为空时，取消仍通过数据库终态生效（发布回答是条件更新）。
_RUN_TASKS: dict[str, asyncio.Task[None]] = {}

router = APIRouter()
EntrySource = Literal["LITERATURE", "WRITING", "RESEARCH", "DATABASE", "AI"]


class RunCreate(BaseModel):
    """Foreign 前端未来提交给独立 Agent 服务的统一运行参数。"""

    request_id: str = Field(min_length=8, max_length=128)
    knowledge_base_id: int | None = Field(default=None, gt=0)
    database_id: int | None = Field(default=None, gt=0)
    conversation_id: str | None = Field(default=None, min_length=32, max_length=32)
    entry_source: EntrySource | None = None
    query: str = Field(min_length=2, max_length=4000)
    mode: str = Field(default="FAST", pattern="^(FAST|DEEP)$")
    thinking: bool | None = Field(default=None, description="是否启用模型思考模式")
    options: dict[str, object] = Field(default_factory=dict)


def _validate_options(code: str, options: dict[str, object]) -> dict[str, object]:
    """在保存运行前限制四个 Agent 各自可接受的业务参数。

    Args:
        code: Agent 稳定编码。
        options: API 收到的额外参数。

    Returns:
        可落库的白名单参数。

    Raises:
        HTTPException: 目标语言或研究范围不合法。
    """
    if code == "LITERATURE":
        if set(options) - {"corpus_ids", "task", "template_version", "focus_corpus_id"}:
            raise HTTPException(422, "文献阅读包含未知参数")
        ids = options.get("corpus_ids")
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 20
            or any(type(value) is not int or not 0 < value < (1 << 63) for value in ids)
            or len(set(ids)) != len(ids)
        ):
            raise HTTPException(422, "请提供 1~20 篇不重复的有效文献 ID")
        task = options.get("task", "ANALYZE")
        version = options.get("template_version", 1)
        if (
            not isinstance(task, str)
            or task not in {"ANALYZE", "ASK"}
            or type(version) is not int
            or version != 1
        ):
            raise HTTPException(422, "文献阅读任务或模板版本无效")
        focus = options.get("focus_corpus_id")
        if focus is not None and (type(focus) is not int or focus not in ids):
            raise HTTPException(422, "当前预览文献必须属于所选范围")
        result = {"corpus_ids": ids, "task": task, "template_version": 1}
        if task == "ASK" and focus is not None:
            result["focus_corpus_id"] = focus
        return result
    if code == "RONGTONG":
        language = options.get("target_language")
        if language not in {"ZH", "EN", "BILINGUAL"}:
            raise HTTPException(422, "融通需要 target_language=ZH/EN/BILINGUAL")
        return {"target_language": language}
    if code == "SUZHI":
        allowed = {"concept", "year_from", "year_to", "review_length", "citation_style"}
        if set(options) - allowed:
            raise HTTPException(422, "溯知参数包含未知字段")
        start, end = options.get("year_from"), options.get("year_to")
        if any(
            value is not None and (type(value) is not int or not 1000 <= value <= 2100)
            for value in (start, end)
        ) or (start and end and start > end):
            raise HTTPException(422, "溯知年份范围无效")
        if options.get("review_length", "MEDIUM") not in {"SHORT", "MEDIUM", "LONG"}:
            raise HTTPException(422, "综述长度无效")
        if options.get("citation_style", "GBT7714") not in {"GBT7714", "APA", "MLA"}:
            raise HTTPException(422, "引用格式无效")
        concept = options.get("concept")
        if concept is not None and (not isinstance(concept, str) or len(concept) > 300):
            raise HTTPException(422, "研究概念无效")
        return dict(options)
    if code == "QUILL":
        return _validate_quill_options(options)
    if options:
        raise HTTPException(422, "该 Agent 不接受额外参数")
    return {}


def _validate_quill_options(options: dict[str, object]) -> dict[str, object]:
    """限制Quill运行的阶段白名单参数并输出规范化结果。

    Args:
        options: API 收到的Quill额外参数。

    Returns:
        已验证的规范化阶段参数。

    Raises:
        HTTPException: 阶段、年份、引用格式、推荐上限或润改参数不合法。
    """
    stage = options.get("stage", "THEORY")
    if stage == "POLISH":
        return _validate_quill_polish_options(options)
    return _validate_quill_research_options(stage, options)


def _validate_quill_research_options(
    stage: object, options: dict[str, object]
) -> dict[str, object]:
    """限制Quill RQ/PLAN/THEORY 运行的白名单参数并输出规范化结果。

    Args:
        stage: 目标阶段编码。
        options: API 收到的Quill额外参数。

    Returns:
        只包含阶段、年份范围、引用格式及阶段专属参数的已验证结果。

    Raises:
        HTTPException: 阶段、年份、引用格式、推荐上限或产物引用不合法。
    """
    allowed = {"stage", "citation_style", "year_from", "year_to"}
    if stage == "THEORY":
        allowed.add("literature_limit")
    if stage == "PLAN":
        allowed.add("artifact_id")
    if stage == "DRAFT":
        allowed.update({"artifact_id", "section_ref"})
    if stage == "REVISE":
        allowed.add("revise_instruction")
    if set(options) - allowed:
        raise HTTPException(422, "Quill参数包含未知字段")
    # 设计方案冻结的阶段编码；六个阶段已全部开放。
    if stage not in {"RQ", "PLAN", "THEORY", "DRAFT", "REVISE"}:
        raise HTTPException(422, "当前版本支持 stage=RQ/PLAN/THEORY/DRAFT/REVISE/POLISH")
    start, end = options.get("year_from"), options.get("year_to")
    if any(
        value is not None and (type(value) is not int or not 1000 <= value <= 2100)
        for value in (start, end)
    ) or (start and end and start > end):
        raise HTTPException(422, "Quill年份范围无效")
    if options.get("citation_style", "GBT7714") not in {"GBT7714", "APA", "MLA"}:
        raise HTTPException(422, "引用格式无效")
    validated: dict[str, object] = {
        "stage": stage,
        "citation_style": options.get("citation_style", "GBT7714"),
    }
    if start is not None:
        validated["year_from"] = start
    if end is not None:
        validated["year_to"] = end
    if stage == "THEORY":
        limit = options.get("literature_limit", 10)
        if type(limit) is not int or not 5 <= limit <= 20:
            raise HTTPException(422, "文献推荐数量无效")
        validated["literature_limit"] = limit
    if stage in {"PLAN", "DRAFT"}:
        artifact_id = options.get("artifact_id")
        if artifact_id is not None and (
            not isinstance(artifact_id, str)
            or len(artifact_id) != 32
            or any(character not in "0123456789abcdef" for character in artifact_id)
        ):
            raise HTTPException(422, "Quill产物引用无效")
        if artifact_id is not None:
            validated["artifact_id"] = artifact_id
    if stage == "DRAFT":
        # 逐节起草必须指定目标章节，且引用写作方案产物。
        section_ref = options.get("section_ref")
        if not isinstance(section_ref, str) or not 1 <= len(section_ref.strip()) <= 16:
            raise HTTPException(422, "逐节起草需要 section_ref 指定章节编号")
        if validated.get("artifact_id") is None:
            raise HTTPException(422, "逐节起草必须引用写作方案（PLAN）产物")
        validated["section_ref"] = section_ref.strip()
    if stage == "REVISE":
        instruction = options.get("revise_instruction")
        if not isinstance(instruction, str) or not 2 <= len(instruction.strip()) <= 500:
            raise HTTPException(422, "学术修改需要 2-500 字的修改要求")
        validated["revise_instruction"] = instruction.strip()[:500]
    return validated


def _validate_quill_polish_options(options: dict[str, object]) -> dict[str, object]:
    """限制Quill POLISH 划词润改的白名单参数并输出规范化结果。

    Args:
        options: API 收到的Quill额外参数；query 字段承载选区原文。

    Returns:
        只包含 stage、action 和上下文的已验证参数。

    Raises:
        HTTPException: 操作编码未知或上下文超长。
    """
    allowed = {"stage", "action", "context_before", "context_after"}
    if set(options) - allowed:
        raise HTTPException(422, "Quill参数包含未知字段")
    action = options.get("action")
    if action not in {
        "CONTINUE",
        "EXPAND",
        "SHORTEN",
        "REWRITE",
        "POLISH",
        "ZH2EN",
        "EN2ZH",
    }:
        raise HTTPException(422, "Quill润改操作编码无效")
    for field in ("context_before", "context_after"):
        value = options.get(field)
        if value is not None and (not isinstance(value, str) or len(value) > 3000):
            raise HTTPException(422, "Quill润改上下文超长")
    validated: dict[str, object] = {"stage": "POLISH", "action": action}
    for field in ("context_before", "context_after"):
        value = options.get(field)
        if isinstance(value, str) and value.strip():
            validated[field] = value[:3000]
    return validated


@router.post("/agents/{code}/runs", status_code=202)
def start_run(
    code: str,
    payload: RunCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """验证 Foreign 用户和知识库后创建一次幂等 Agent 运行。

    Args:
        code: QIMING/ZHIJIAN/SUZHI/RONGTONG。
        payload: 已解析的运行输入。
        request: 请求追踪上下文。
        background_tasks: 响应后图执行调度器。
        actor: Foreign 用户。
        session: 共库事务会话。

    Returns:
        运行与会话标识、状态和 SSE 地址。
    """
    agent_code = code.upper()
    if agent_code not in {"QIMING", "ZHIJIAN", "SUZHI", "RONGTONG", "LITERATURE", "QUILL"}:
        raise HTTPException(404, "Agent 不存在")
    # 旧客户端的 DATABASE 声明仅作兼容别名，新建记录统一保存 RESEARCH。
    try:
        payload.entry_source = resolve_entry_source(agent_code, payload.entry_source)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    selection_entry = (
        payload.entry_source == "AI"
        or (payload.entry_source == "RESEARCH" and payload.knowledge_base_id is None
            and payload.database_id is None)
        or (
            payload.entry_source == "WRITING"
            and payload.knowledge_base_id is None
            and payload.database_id is None
        )
    )
    raw_options = dict(payload.options)
    reading = raw_options.pop("reading", None) if payload.entry_source == "AI" else None
    if reading is not None:
        try:
            reading = AssistantReadingOptions.model_validate(reading).model_dump()
        except ValidationError as exc:
            raise HTTPException(422, "文献追问操作参数无效") from exc
        if "writing" in raw_options or "reading_context" in raw_options:
            raise HTTPException(422, "文献追问不能混用写作或其他文献来源")
    reading_context = raw_options.pop("reading_context", None) if (
        payload.entry_source == "AI"
    ) else None
    if reading_context is not None:
        try:
            reading_context = AssistantReadingContext.model_validate(reading_context).model_dump()
        except ValidationError as exc:
            raise HTTPException(422, "文献来源或篇目选择参数无效") from exc
        if "writing" in raw_options:
            raise HTTPException(422, "阅读上下文不能混入知识库写作操作")
    selected_ids = raw_options.pop("corpus_ids", []) if selection_entry else None
    if selection_entry:
        if (
            not isinstance(selected_ids, list)
            or len(selected_ids) > 20
            or any(type(value) is not int or not 0 < value < (1 << 63) for value in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)
        ):
            raise HTTPException(422, "请选择最多 20 篇不重复的有效文献")
        if payload.knowledge_base_id is not None or (
            payload.database_id is not None and payload.entry_source != "AI"
        ):
            raise HTTPException(422, "全库或选文入口不能同时绑定知识库")
        if payload.entry_source == "AI" and agent_code != "QIMING":
            raise HTTPException(422, "AI 入口必须使用启明执行器")
    if payload.entry_source == "AI" and "writing" in raw_options:
        if set(raw_options) != {"writing"} or selected_ids or payload.database_id is not None:
            raise HTTPException(422, "知识库写作不能混用数据库或选文范围")
        try:
            writing = AssistantWritingOptions.model_validate(raw_options["writing"])
        except ValidationError as exc:
            raise HTTPException(422, "写作阶段或知识库参数无效") from exc
        options = {"writing": writing.model_dump(exclude_none=True)}
    else:
        options = _validate_options(agent_code, raw_options)
    if selection_entry:
        options["corpus_ids"] = selected_ids
    if reading_context is not None:
        options["reading_context"] = reading_context
    if reading is not None:
        options["reading"] = reading
    if agent_code in {"QUILL", "QIMING"}:
        # 深度思考属于运行级模型参数，随运行落库保证幂等与审计一致。
        options["thinking"] = bool(payload.thinking)
    if (
        agent_code != "LITERATURE"
        and not selection_entry
        and ((payload.knowledge_base_id is None) == (payload.database_id is None))
    ):
        raise HTTPException(422, "必须且只能选择一个知识库或根数据库")
    try:
        result = create_run(
            session,
            actor=actor,
            code=agent_code,
            request_id=payload.request_id,
            knowledge_base_id=payload.knowledge_base_id,
            query=payload.query,
            conversation_id=payload.conversation_id,
            mode=payload.mode,
            options=options,
            entry_source=payload.entry_source,
            database_id=payload.database_id,
        )
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (ValueError, LookupError) as exc:
        raise HTTPException(409, str(exc)) from exc
    if result["created"]:
        background_tasks.add_task(_execute_run, str(result["run_id"]))
    result["stream_url"] = f"/api/v1/agent-runs/{result['run_id']}/stream"
    return envelope(request=request, data=result, message="accepted")


@router.get("/research-databases")
def get_research_databases(
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """向已认证数据收集者提供有效根数据库选项，不读取私有知识库。"""
    return envelope(request=request, data={"items": root_databases(session)})


async def _execute_run(run_id: str) -> None:
    """用独立会话执行响应后 Agent 图。

    Args:
        run_id: 已持久化的运行 ID。
    """
    # 登记当前协程，供 /agent-runs/{run_id}/cancel 在当前进程内及时取消（停止生成）
    task = asyncio.current_task()
    if task is not None:
        _RUN_TASKS[run_id] = task
    try:
        await RunExecutor(run_id).execute()
    finally:
        _RUN_TASKS.pop(run_id, None)


def _database_history_conditions(source: str, database_id: int | None) -> tuple:
    """构造固定数据库历史条件；仅数据库入口接受此筛选，必须在分页前应用。

    Args:
        source: 当前请求的实际入口来源。
        database_id: 可选的根数据库 ID；省略时兼容原列表行为。

    Returns:
        数据库绑定条件或空元组。

    Raises:
        HTTPException: 非数据库入口错误地声明数据库筛选。
    """
    if database_id is None:
        return ()
    if normalize_entry_source(source) != "RESEARCH":
        raise HTTPException(422, "database_id 仅用于研究入口")
    return (research_conversation.c.database_id == database_id,)


@router.get("/agent-runs")
def get_run_history(
    request: Request,
    executor_agent_code: str = Query(pattern="^(QIMING|ZHIJIAN|SUZHI|RONGTONG|LITERATURE|QUILL)$"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    entry_source: EntrySource | None = Query(default=None),
    database_id: int | None = Query(default=None, gt=0),
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """按实际执行模块分页读取本人历史，并标记当前知识库访问权限。

    Args:
        request: HTTP 请求上下文。
        executor_agent_code: 实际执行模块编码，包含启明委派的运行。
        page: 从 1 开始的页码。
        page_size: 每页记录数。
        entry_source: 会话入口；省略时按模块使用兼容默认值，委派不改变归属。
        database_id: 固定智能体的根数据库绑定；筛选在统计总数与分页之前执行。
        actor: 当前 Foreign 用户。
        session: 共库会话。

    Returns:
        轻量历史列表与准确总数；知识库 ID 从会话派生，撤权记录不返回回答摘要。
    """
    # 未传来源的旧调用按模块使用默认入口；写作的专业委派必须显式查 WRITING。
    # 1. 同时限定所有者、执行模块、入口来源和软删除状态，避免混入其他侧栏历史
    source = normalize_entry_source(entry_source) or resolve_entry_source(executor_agent_code)
    conditions = (
        research_run.c.user_id == actor.id,
        research_run.c.is_deleted == 0,
        research_run.c.executor_agent_code == executor_agent_code,
        research_conversation.c.is_deleted == 0,
        research_conversation.c.entry_source == source,
        *_database_history_conditions(source, database_id),
    )
    total = session.scalar(
        select(func.count())
        .select_from(research_run)
        .join(
            research_conversation,
            research_conversation.c.id == research_run.c.conversation_id,
        )
        .where(*conditions)
    )
    rows = (
        session.execute(
            select(
                research_run.c.id,
                research_run.c.conversation_id,
                research_run.c.query,
                research_run.c.status,
                research_conversation.c.knowledge_base_id,
                research_conversation.c.database_id,
                research_conversation.c.entry_source,
                research_run.c.executor_agent_code,
                research_run.c.created_at,
                func.substr(research_run.c.final_answer, 1, 240).label("summary"),
                research_agent.c.code.label("entry_agent_code"),
                research_conversation.c.title.label("conversation_title"),
            )
            .join(research_agent, research_agent.c.id == research_run.c.agent_id)
            .join(
                research_conversation, research_conversation.c.id == research_run.c.conversation_id
            )
            .where(*conditions)
            .order_by(research_run.c.created_at.desc(), research_run.c.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .mappings()
        .all()
    )
    # 2. 只对当前页知识库批量复核权限，失去原库权限的历史保留列表但隐藏摘要
    accessible = accessible_scopes(session, actor.id, rows)
    items = []
    for row in rows:
        allowed = (
            row["entry_source"] == "AI"
            or executor_agent_code == "LITERATURE"
            or (
                row["entry_source"] == "WRITING"
                and not row["knowledge_base_id"]
                and not row["database_id"]
            )
            or ResearchScope.from_record(row) in accessible
        )
        items.append(
            {
                **dict(row),
                "knowledge_base_id": str(row["knowledge_base_id"])
                if row["knowledge_base_id"]
                else None,
                "summary": (row["summary"] or "") if allowed else "",
                "knowledge_base_accessible": allowed,
                "database_accessible": allowed if row["database_id"] is not None else None,
                "knowledge_base_name": None,
                "knowledge_base_version": 0,
                "result_type": None,
                "schema_version": None,
            }
        )
    return envelope(
        request=request,
        data={
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    )


@router.get("/agent-runs/{run_id}")
def get_run(
    run_id: str,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
    entry_source: Annotated[EntrySource | None, Query()] = None,
    database_id: Annotated[int | None, Query(gt=0)] = None,
) -> dict:
    """按 Foreign 用户和知识库权限读取运行详情。

    Args:
        run_id: 运行 ID。
        request: 请求追踪上下文。
        actor: 当前 Foreign 用户。
        session: 共库会话。
        entry_source: 可选入口声明，避免页面恢复其他入口的运行。
        database_id: 可选固定数据库绑定；跨库运行统一返回不存在。

    Returns:
        运行及步骤投影。
    """
    data = _authorized_run(session, run_id, actor)
    if entry_source is not None and data["entry_source"] != normalize_entry_source(entry_source):
        raise HTTPException(404, "运行不属于当前入口")
    if database_id is not None and (
        data["entry_source"] != "RESEARCH" or data["database_id"] != database_id
    ):
        raise HTTPException(404, "运行不属于当前智能体")
    return envelope(request=request, data=data)


@router.post("/agent-runs/{run_id}/cancel")
def cancel_agent_run(
    run_id: str,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """停止一个尚未结束的运行（前端「停止生成」按钮）。

    先把运行标记为 CANCELLED（条件更新，幂等），再尝试取消当前进程内的协程，
    使模型调用尽快中止；已进入终态的运行原样返回当前状态。

    Args:
        run_id: 运行 ID。
        request: 请求追踪上下文。
        actor: 当前 Foreign 用户。
        session: 共库会话。

    Returns:
        {"run_id": …, "status": …}。

    Raises:
        HTTPException: 运行不存在（404）或非本人运行（403）。
    """
    try:
        result = cancel_run(session, run_id, actor)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    task = _RUN_TASKS.get(run_id)
    if task is not None and not task.done():
        task.cancel()
    return envelope(request=request, data=result, message="cancelled")


@router.get("/agent-runs/{run_id}/documents")
def get_assistant_documents(
    run_id: str,
    request: Request,
    page: Annotated[int, Query(ge=1, le=50)] = 1,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """按已有 AI 运行的搜索条件和当前资源范围翻页，不调用模型或新建运行。

    Args:
        run_id: 当前用户已成功完成的 AI 查文献或写作交接运行。
        request: 请求追踪上下文。
        page: 一开始的题录页码，每页固定十篇，最多五十页。
        actor: 已认证的当前用户。
        session: FastAPI 管理的同步数据库会话。

    Returns:
        固定检索快照中当前仍可访问的题录及该快照的可浏览总数。

    Raises:
        HTTPException: 运行不是可分页的 AI 检索，或来源资源已无访问权限。
    """
    from app.agents.ai_assistant_schemas import AssistantSearchPlan
    from app.foreign_integration.assistant_search import search_documents

    # 1. 按运行所有者授权，条件只从服务保存的结果读取，不接受客户端重写查询。
    run = _authorized_run(session, run_id, actor)
    payload = (run["result_payload"] or {}).get("payload", {})
    if run["entry_source"] != "AI" or run["status"] != "SUCCESS" or not payload.get("search_plan"):
        raise HTTPException(409, "本轮结果不支持题录分页，请重新查找文献")
    # 2. 新任务首次检索已保存 ID 快照，翻页只读取十篇详情；旧任务懒加载一次快照。
    # 空快照同样有效，避免无结果请求反复执行全库检索。
    from app.foreign_integration.assistant_search import search_document_index

    plan = AssistantSearchPlan.model_validate(payload["search_plan"])
    corpus_ids = (run["request_payload"] or {}).get("corpus_ids") or None
    cached = payload.get("document_index")
    try:
        if isinstance(cached, dict) and isinstance(cached.get("ids"), list):
            result = search_documents(
                session,
                actor.id,
                corpus_ids,
                plan,
                page=page,
                lexical_index=get_foreign_settings().opensearch_index,
                scope=ResearchScope.for_run(run),
                document_ids=list(cached["ids"]),
                total=int(cached.get("total") or 0),
            )
        else:
            index = search_document_index(
                session,
                actor.id,
                corpus_ids,
                plan,
                scope=ResearchScope.for_run(run),
            )
            # 2.1 旧成功任务补存快照，包含空结果；原有回答和其他载荷字段完整保留。
            stored = dict(run["result_payload"] or {})
            stored_payload = dict(stored.get("payload") or {})
            stored_payload["document_index"] = index
            stored["payload"] = stored_payload
            session.execute(
                update(research_run)
                .where(research_run.c.id == run_id)
                .values(result_payload=stored)
            )
            session.commit()
            result = search_documents(
                session,
                actor.id,
                corpus_ids,
                plan,
                page=page,
                lexical_index=get_foreign_settings().opensearch_index,
                scope=ResearchScope.for_run(run),
                document_ids=index["ids"],
                total=index["total"],
            )
    except PermissionError as exc:
        raise HTTPException(409, str(exc)) from exc
    return envelope(request=request, data=result)


@router.get("/agent-runs/{run_id}/reasoning")
def get_run_reasoning(
    run_id: str,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """轮询读取运行中节点的实时思维链。

    思考流式调用产生的 reasoning_content 增量实时写入内存缓冲；节点完成后
    并入 research_step 落库并随运行结束清理。仅供运行所有者读取。
    """
    try:
        data = read_live_reasoning(session, run_id, actor)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    return envelope(request=request, data=data, message="ok")


def _authorized_run(session: Session, run_id: str, actor: ForeignActor) -> dict:
    """把领域层的不存在和越权映射为稳定 HTTP 状态。

    Args:
        session: 请求事务会话。
        run_id: 运行 ID。
        actor: 当前 Foreign 用户。

    Returns:
        可对外展示的运行信息。
    """
    try:
        return read_run(session, run_id, actor)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc


def _flag_value(value: object) -> bool:
    """把主库 BIT(1) 布尔列的取值安全转为 Python 布尔。

    Args:
        value: BIT(1) 经驱动返回的 b"\\x00"/b"\\x01" 字节，或常规 0/1/布尔。

    Returns:
        规范化后的布尔值；注意 b"\\x00" 是非空字节，直接 bool() 会误判为真。
    """
    if isinstance(value, bytes | bytearray):
        return bool(int.from_bytes(value, "big"))
    return bool(value)


@router.get("/agent-runs/{run_id}/evidence")
def get_evidence(
    run_id: str,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """读取已经持久化的原文证据并标记当前来源可用状态。

    Args:
        run_id: 运行 ID。
        request: 请求上下文。
        actor: 当前 Foreign 用户。
        session: 共库会话。

    Returns:
        Evidence Panel 所需的原文、定位和可用状态。
    """
    _authorized_run(session, run_id, actor)
    rows = (
        session.execute(
            select(
                research_evidence,
                foreign_corpus.c.status,
                foreign_corpus.c.owner_id,
                foreign_corpus.c.attachment,
                foreign_corpus.c.title_cn,
                foreign_corpus.c.title_en,
                foreign_corpus.c.author,
                foreign_corpus.c.source,
                foreign_corpus.c.publish_time,
                # BIT(1) 列经 Boolean 类型读取会因 bool(b"\x00") 误判为已删除；
                # 在 SQL 层用 IS TRUE 转为 0/1 整数，杜绝字节进 Python。
                case(
                    (foreign_corpus.c.is_deleted.is_(True), 1),
                    else_=0,
                ).label("source_deleted"),
            )
            .join(foreign_corpus, foreign_corpus.c.id == research_evidence.c.corpus_id)
            .where(research_evidence.c.run_id == run_id, research_evidence.c.is_deleted == 0)
            .order_by(research_evidence.c.evidence_rank)
        )
        .mappings()
        .all()
    )
    items = []
    for row in rows:
        data = dict(row)
        try:
            attachments = json.loads(row["attachment"] or "[]")
        except (TypeError, ValueError):
            attachments = []
        current_paths = (
            {item.get("file_path") for item in attachments if isinstance(item, dict)}
            if isinstance(attachments, list)
            else set()
        )
        readable = (row["status"] == 2 and row["owner_id"] is None) or (
            row["status"] == 4 and row["owner_id"] == actor.id
        )
        data["source_unavailable"] = _flag_value(row["source_deleted"]) or (
            not readable or row["file_path"] not in current_paths
        )
        data.pop("source_deleted", None)
        data.pop("status", None)
        data.pop("attachment", None)
        data.pop("owner_id", None)
        if not data["source_unavailable"] and data["file_path"]:
            endpoint = (
                f"/literature/documents/{data['corpus_id']}/file"
                if row["status"] == 4
                else "/public/file/corpus"
            )
            data["file_url"] = (
                f"{get_foreign_settings().public_api_base}{endpoint}?"
                + urlencode({"corpus_id": data["corpus_id"], "attachment_path": data["file_path"]})
                + (f"#page={data['page_start']}" if data["page_start"] else "")
            )
        else:
            data["file_url"] = None
        items.append(data)
    return envelope(request=request, data=items)


@router.delete("/agent-runs/{run_id}")
def delete_run(
    run_id: str,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """伪删除当前用户的一次已结束运行。

    Args:
        run_id: 运行 ID。
        request: 请求上下文。
        actor: Foreign 用户。
        session: 共库事务。

    Returns:
        已伪删除的运行标识。
    """
    try:
        soft_delete_run(session, run_id, actor)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return envelope(request=request, data={"run_id": run_id, "is_deleted": True})


@router.get("/conversations")
def list_conversations(
    request: Request,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    entry_source: EntrySource = Query(default="RESEARCH"),
    database_id: int | None = Query(default=None, gt=0),
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """分页列出 Foreign 用户自己的未伪删除研究会话。

    Args:
        request: 请求追踪上下文。
        page: 从 1 开始的页号。
        page_size: 每页数量。
        entry_source: 只返回此入口的会话，先过滤再分页。
        database_id: 固定智能体的根数据库绑定，与入口同时筛选。
        actor: Foreign 用户。
        session: 共库会话。

    Returns:
        会话摘要与分页信息。
    """
    entry_source = normalize_entry_source(entry_source)
    base = select(research_conversation).where(
        research_conversation.c.user_id == actor.id,
        research_conversation.c.is_deleted == 0,
        research_conversation.c.entry_source == entry_source,
        *_database_history_conditions(entry_source, database_id),
    )
    total = session.scalar(select(func.count()).select_from(base.subquery()))
    rows = (
        session.execute(
            base.order_by(
                research_conversation.c.updated_at.desc(), research_conversation.c.id.desc()
            )
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        .mappings()
        .all()
    )
    return envelope(
        request=request,
        data={
            "items": [dict(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
        },
    )


@router.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
    entry_source: Annotated[EntrySource | None, Query()] = None,
    database_id: Annotated[int | None, Query(gt=0)] = None,
) -> dict:
    """读取本人的历史会话及其问题、回答和真实引用。

    Args:
        conversation_id: 研究会话 ID。
        request: 请求上下文。
        actor: Foreign 用户。
        session: 共库会话。
        entry_source: 页面声明的入口；指定后拒绝恢复其他入口的会话。
        database_id: 可选固定数据库绑定；跨库会话统一返回不存在。

    Returns:
        会话与按时间排序的消息。
    """
    # 1. 先复核会话所有者与入口来源，未知或跨入口会话统一返回不存在
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
    if not conversation or conversation["user_id"] != actor.id:
        raise HTTPException(404, "会话不存在")
    if (
        entry_source is not None
        and conversation["entry_source"] != normalize_entry_source(entry_source)
    ):
        raise HTTPException(404, "会话不属于当前入口")
    if database_id is not None and (
        conversation["entry_source"] != "RESEARCH" or conversation["database_id"] != database_id
    ):
        raise HTTPException(404, "会话不属于当前智能体")
    # 2. 按入口恢复消息顺序；阅读消息按运行和用户/助手角色排序，保持问答配对
    message_order = [research_message.c.created_at, research_message.c.id]
    if conversation["entry_source"] in {"LITERATURE", "AI"}:
        message_order = [
            research_run.c.created_at,
            research_run.c.id,
            case((research_message.c.role == "USER", 0), else_=1),
        ]
    messages = (
        session.execute(
            select(
                research_message,
                research_agent.c.code.label("entry_agent_code"),
                research_run.c.executor_agent_code,
                research_run.c.evidence_level,
                research_run.c.result_payload["result_type"].as_string().label("result_type"),
                research_run.c.result_payload["schema_version"]
                .as_integer()
                .label("schema_version"),
            )
            .join(research_run, research_run.c.id == research_message.c.run_id)
            .join(research_agent, research_agent.c.id == research_run.c.agent_id)
            .where(
                research_message.c.conversation_id == conversation_id,
                research_message.c.is_deleted == 0,
                research_run.c.is_deleted == 0,
            )
            .order_by(*message_order)
        )
        .mappings()
        .all()
    )
    # 3. 按消息加载已发布引用并保留阅读原文快照，避免用最新来源覆盖历史引用
    items = []
    for message in messages:
        citations = (
            session.execute(
                select(
                    research_citation.c.citation_index,
                    research_citation.c.quote_text,
                    research_evidence.c.corpus_id,
                    research_evidence.c.point_id,
                    research_evidence.c.file_path,
                    research_evidence.c.page_start,
                    research_evidence.c.page_end,
                    *(
                        [research_evidence.c.content_snapshot.label("content")]
                        if message["executor_agent_code"] == "LITERATURE"
                        else []
                    ),
                )
                .join(research_evidence, research_evidence.c.id == research_citation.c.evidence_id)
                .where(
                    research_citation.c.message_id == message["id"],
                    research_citation.c.is_deleted == 0,
                    research_evidence.c.is_deleted == 0,
                )
                .order_by(research_citation.c.citation_index)
            )
            .mappings()
            .all()
        )
        # 模块元数据来自消息自己的 Run，混合专业会话不能使用最后一次运行的路由。
        items.append({**dict(message), "citations": [dict(row) for row in citations]})
    data = {**dict(conversation), "messages": items}
    if conversation["entry_source"] == "AI":
        data["assistant_runs"] = [
            dict(row)
            for row in session.execute(
                select(
                    research_run.c.id,
                    research_run.c.status,
                    research_run.c.query,
                    research_run.c.final_answer,
                    research_run.c.result_payload,
                )
                .where(
                    research_run.c.conversation_id == conversation_id,
                    research_run.c.is_deleted == 0,
                )
                .order_by(research_run.c.created_at, research_run.c.id)
            ).mappings()
        ]
    if conversation["entry_source"] == "LITERATURE":
        data["reading_runs"] = [
            dict(row)
            for row in session.execute(
                select(
                    research_run.c.id,
                    research_run.c.status,
                    research_run.c.created_at,
                    research_run.c.request_payload,
                )
                .where(
                    research_run.c.conversation_id == conversation_id,
                    research_run.c.is_deleted == 0,
                    research_run.c.executor_agent_code == "LITERATURE",
                )
                .order_by(research_run.c.created_at, research_run.c.id)
            ).mappings()
        ]
    return envelope(request=request, data=data)


@router.get("/literature/conversations")
def reading_conversations(
    request: Request,
    page: int = Query(default=1, ge=1),
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """阅读历史与知识库研究会话隔离；只返回本人轻量会话摘要。"""
    conditions = (
        research_conversation.c.user_id == actor.id,
        research_conversation.c.is_deleted == 0,
        research_conversation.c.entry_source == "LITERATURE",
    )
    rows = (
        session.execute(
            select(research_conversation)
            .where(*conditions)
            .order_by(research_conversation.c.updated_at.desc())
            .offset((page - 1) * 20)
            .limit(20)
        )
        .mappings()
        .all()
    )
    total = session.scalar(
        select(func.count()).select_from(research_conversation).where(*conditions)
    )
    return envelope(
        request=request,
        data={"items": [dict(row) for row in rows], "total": total, "page": page, "page_size": 20},
    )


@router.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: str,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """伪删除当前用户会话及其全部已结束运行。

    Args:
        conversation_id: 会话 ID。
        request: 请求上下文。
        actor: Foreign 用户。
        session: 共库事务。

    Returns:
        已伪删除会话标识。
    """
    try:
        soft_delete_conversation(session, conversation_id, actor)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return envelope(request=request, data={"conversation_id": conversation_id, "is_deleted": True})


@router.get("/agent-runs/{run_id}/stream")
async def stream_run(
    run_id: str,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """根据 MySQL 已提交的步骤事实生成可重连 SSE 进度。

    Args:
        run_id: 运行 ID。
        last_event_id: 客户端已确认的稳定事件 ID。
        actor: Foreign 用户。
        session: 首次授权会话。

    Returns:
        不暴露未审核草稿的事件流。
    """
    _authorized_run(session, run_id, actor)
    # 长连接随后自行轮询短事务；首次授权连接不应占住 MySQL 连接池。
    session.close()
    return StreamingResponse(
        _events(run_id, actor, last_event_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _events(
    run_id: str,
    actor: ForeignActor,
    last_event_id: str | None,
) -> AsyncIterator[str]:
    """轮询短数据库事务，重连重放当前步骤快照与终态。

    Args:
        run_id: 运行 ID。
        actor: 已认证 Foreign 用户。
        last_event_id: 客户端重连游标；快照会更新，故不以游标存在作为发送前提。

    Yields:
        已审核的步骤状态与终态事件。
    """
    # 1. 为本次连接建立已发送集合；重连重新投影快照，不依赖旧连接内存游标
    sent: set[str] = set()
    # 步骤状态会变化，旧游标可能已不在快照中；重连重放当前快照，客户端按序号幂等合并。
    while True:
        # 2. 每轮短会话复核运行所有权和账号状态，等待期间不持有数据库事务
        with session_factory()() as session:
            projection = _authorized_run(session, run_id, actor)
            user = session.execute(
                select(foreign_user.c.id).where(
                    foreign_user.c.id == actor.id,
                    foreign_user.c.status == 1,
                    foreign_user.c.is_deleted == 0,
                )
            ).first()
            if user is None:
                return
        # 3. 为步骤状态和终态生成稳定事件 ID，连接内去重，终态发送后结束轮询
        events = [
            (
                f"{run_id}:step:{step['sequence']}:{step['status']}",
                "step",
                {
                    "sequence": step["sequence"],
                    "name": step["node_name"],
                    "status": step["status"],
                },
            )
            for step in projection["steps"]
        ]
        if projection["status"] in {"SUCCESS", "FAILED", "CANCELLED"}:
            events.append(
                (
                    f"{run_id}:terminal",
                    "complete",
                    {
                        "status": projection["status"],
                        "answer": projection["final_answer"]
                        if projection["status"] == "SUCCESS"
                        else None,
                        "error_code": projection["error_code"],
                    },
                )
            )
        for event_id, event_type, payload in events:
            if event_id in sent:
                continue
            sent.add(event_id)
            encoded = json.dumps(payload, ensure_ascii=False)
            yield f"id: {event_id}\nevent: {event_type}\ndata: {encoded}\n\n"
        if projection["status"] in {"SUCCESS", "FAILED", "CANCELLED"}:
            return
        await asyncio.sleep(1)


class QuillExportRequest(BaseModel):
    """Quill 导出请求：整篇会话、单个产物、或编辑器当前内容三选一。"""

    conversation_id: str | None = Field(default=None, min_length=32, max_length=32)
    artifact_id: str | None = Field(default=None, min_length=32, max_length=32)
    # 左侧编辑器当前内容（前端把 contenteditable 拍平成块列表），
    # 用于「导出我写的这一稿」——不依赖是否已生成过产物
    blocks: list[dict[str, Any]] | None = None
    title: str | None = Field(default=None, max_length=200)


def _owned_conversation_row(session: Session, conversation_id: str, actor: ForeignActor) -> dict:
    """校验会话存在且属于当前用户。

    Args:
        session: 共库事务会话。
        conversation_id: 会话主键。
        actor: 当前 Foreign 用户。

    Returns:
        会话行映射。

    Raises:
        HTTPException: 会话不存在或不属于当前用户，统一 404 防探测。
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
    if not conversation or conversation["user_id"] != actor.id:
        raise HTTPException(404, "会话不存在")
    return dict(conversation)


@router.get("/writing-artifacts/{conversation_id}")
def list_quill_artifacts(
    conversation_id: str,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> dict:
    """列出会话内的 Quill 阶段产物，按阶段与版本排序。

    Args:
        conversation_id: 会话主键。
        request: 请求追踪上下文。
        actor: 当前 Foreign 用户。
        session: 共库事务会话。

    Returns:
        产物列表：阶段、版本、章节定位、内容快照与创建时间。
    """
    _owned_conversation_row(session, conversation_id, actor)
    rows = (
        session.execute(
            select(research_writing_artifact)
            .where(
                research_writing_artifact.c.conversation_id == conversation_id,
                research_writing_artifact.c.is_deleted == 0,
            )
            .order_by(research_writing_artifact.c.stage, research_writing_artifact.c.version)
        )
        .mappings()
        .all()
    )
    items = [
        {
            "id": row["id"],
            "stage": row["stage"],
            "version": row["version"],
            "section_ref": row["section_ref"],
            "created_at": row["created_at"],
            "content": row["content"],
        }
        for row in rows
    ]
    return envelope(request=request, data={"items": items, "total": len(items)})


@router.post("/quill/export-docx")
def export_quill_docx(
    payload: QuillExportRequest,
    request: Request,
    actor: ForeignActor = Depends(current_actor),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """把 Quill 产物导出为 Word 文档（整篇会话或单个产物）。

    Args:
        payload: 会话导出或单产物导出请求。
        request: 请求追踪上下文。
        actor: 当前 Foreign 用户。
        session: 共库事务会话。

    Returns:
        .docx 文件流。

    Raises:
        HTTPException: 参数缺失、会话越权或没有可导出产物。
    """
    if not payload.blocks and not payload.conversation_id and not payload.artifact_id:
        raise HTTPException(422, "需要 conversation_id、artifact_id 或编辑器内容")
    if payload.blocks:
        # 编辑器内容导出：不依赖产物与知识库，纯文本块转 Word
        try:
            editor_document = build_document_from_blocks(payload.blocks, payload.title or "")
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        editor_buffer = BytesIO()
        editor_document.save(editor_buffer)
        editor_buffer.seek(0)
        return StreamingResponse(
            editor_buffer,
            media_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
            headers={"Content-Disposition": 'attachment; filename="quill_editor_draft.docx"'},
        )
    if payload.artifact_id:
        artifact = (
            session.execute(
                select(research_writing_artifact).where(
                    research_writing_artifact.c.id == payload.artifact_id,
                    research_writing_artifact.c.is_deleted == 0,
                )
            )
            .mappings()
            .first()
        )
        if not artifact:
            raise HTTPException(404, "产物不存在")
        _owned_conversation_row(session, artifact["conversation_id"], actor)
        artifacts = [dict(artifact)]
        filename = f"quill_{artifact['stage'].lower()}_{payload.artifact_id[:8]}.docx"
    else:
        _owned_conversation_row(session, payload.conversation_id, actor)
        rows = (
            session.execute(
                select(research_writing_artifact)
                .where(
                    research_writing_artifact.c.conversation_id == payload.conversation_id,
                    research_writing_artifact.c.is_deleted == 0,
                )
                .order_by(research_writing_artifact.c.stage, research_writing_artifact.c.version)
            )
            .mappings()
            .all()
        )
        artifacts = [dict(row) for row in rows]
        if not artifacts:
            raise HTTPException(404, "该会话暂无可导出的产物")
        filename = f"quill_paper_{payload.conversation_id[:8]}.docx"
    document = build_quill_document(artifacts)
    buffer = BytesIO()
    document.save(buffer)
    buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers=headers,
    )
