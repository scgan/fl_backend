"""校验 AI 会话内的知识库写作范围和前序产物，防止跨库、跨会话复用稿件。"""

from sqlalchemy import select

from app.agents.ai_assistant_schemas import AssistantWritingOptions
from app.foreign_integration.research_scope import ResearchScope
from app.foreign_integration.schema import research_run, research_writing_artifact


def writing_history_summary(payload: dict | None) -> dict | None:
    """仅保留阶段、主题、选题和章节供指代解析，不重复发送整篇稿件及来源快照。"""
    if not payload:
        return None
    return {
        "stage": payload.get("stage"),
        "research_topic": payload.get("research_topic"),
        "section_ref": payload.get("section_ref"),
        "research_questions": [
            item.get("question") for item in payload.get("research_questions", [])
        ],
        "sections": [
            {"section_id": item.get("section_id"), "title": item.get("title")}
            for item in payload.get("sections", [])
        ],
    }


def writing_scope(session, actor_id: int, options: dict) -> ResearchScope:
    """授权写作知识库并检查是否有可检索材料，只保存资源引用而不枚举成员。

    Args:
        session: 调用方持有的短同步会话。
        actor_id: 当前已认证用户。
        options: 包含知识库 ID 和阶段的已校验写作参数。

    Returns:
        已授权的知识库引用，后续检索按该资源查询当前成员。

    Raises:
        ValueError: 写作参数无效，或所选知识库没有已建索引的材料。
        PermissionError: 知识库不属于当前用户或已不可用。
    """
    writing = AssistantWritingOptions.model_validate(options)
    scope = ResearchScope.from_ids(writing.knowledge_base_id)
    if session.scalar(scope.corpus_query(session, actor_id).limit(1)) is None:
        raise ValueError("该知识库暂无已审核并建立索引的文献，请补充材料或选择其他知识库")
    return scope


def existing_writing_action(
    session, conversation_id: str, actor_id: int, options: dict, query: str
) -> dict | None:
    """校验来源卡并复用已接受的相同操作；调用方必须先锁定所属会话。

    成功和执行中的操作只允许一次；失败或取消后可以创建新尝试。旧版没有来源字段的
    阶段操作仅在前序产物、阶段和具体选题/章节完全对应时兼容识别。

    Raises:
        ValueError: 来源卡、操作类型、选题或前序产物不对应。
    """
    source_id = options.get("source_run_id")
    if not source_id:
        return None
    # 1. 来源必须是当前用户同一 AI 会话内的成功卡片，不能使用其他会话的 Run。
    source = (
        session.execute(
            select(research_run)
            .where(
                research_run.c.id == source_id,
                research_run.c.conversation_id == conversation_id,
                research_run.c.user_id == actor_id,
                research_run.c.is_deleted == 0,
                research_run.c.status == "SUCCESS",
            )
            .with_for_update()
        )
        .mappings()
        .first()
    )
    result = ((source or {}).get("result_payload") or {}).get("payload", {}).get(
        "writing_result"
    ) or {}
    action = options["source_action"]
    valid = action == "confirm" and result.get("status") == "NEEDS_INPUT"
    if (
        valid
        and result.get("task_selection_mode") == "FIXED"
        and options["stage"] != result.get("stage")
        and not options.get("task_corrected")
    ):
        raise ValueError("写作任务已明确，请通过更正任务主动确认变更")
    if result.get("artifact_id") and options.get("artifact_id") == result["artifact_id"]:
        if (
            action.startswith("plan:")
            and result.get("stage") == "RQ"
            and options["stage"] == "PLAN"
        ):
            index = action.removeprefix("plan:")
            valid = index.isdigit() and int(index) < len(result.get("research_questions", []))
            if valid:
                valid = query == result["research_questions"][int(index)]["question"]
        elif (
            action.startswith("draft:")
            and result.get("stage") == "PLAN"
            and options["stage"] == "DRAFT"
        ):
            valid = action == f"draft:{options.get('section_ref')}" and any(
                item["section_id"] == options.get("section_ref")
                for item in result.get("sections", [])
            )
        elif action == "revise" and result.get("stage") in {"DRAFT", "REVISE"}:
            valid = options["stage"] == "REVISE"
    if not source or not valid:
        raise ValueError("写作操作与来源卡片不匹配，请刷新当前会话")
    # 2. 会话行锁串行化同一卡片的请求；运行结果按操作标识复用，不因新 request_id 重复生成。
    candidates = session.execute(
        select(research_run)
        .where(
            research_run.c.conversation_id == conversation_id,
            research_run.c.user_id == actor_id,
            research_run.c.is_deleted == 0,
            research_run.c.status.in_(["PENDING", "RUNNING", "SUCCESS"]),
        )
        .order_by(research_run.c.created_at.desc(), research_run.c.id.desc())
        .with_for_update()
    ).mappings()
    for row in candidates:
        writing = (row["request_payload"] or {}).get("writing") or {}
        if writing.get("source_run_id"):
            if writing["source_run_id"] == source_id and writing.get("source_action") == action:
                return dict(row)
            continue
        # 3. 旧卡片只按精确目标匹配，不把其他选题或章节的生成误判成重复。
        if writing.get("stage") != options["stage"]:
            continue
        # 数据库时间精度可能只有秒，UUID 大小不代表先后；同秒通过真实前序产物判定。
        if row["id"] == source_id or row["created_at"] < source["created_at"]:
            continue
        if writing.get("knowledge_base_id") != options["knowledge_base_id"]:
            continue
        if action == "confirm":
            match = (
                not writing.get("artifact_id")
                and row["query"] == query
                and query in {result.get("research_topic"), source["query"]}
            )
        else:
            match = writing.get("artifact_id") == result.get("artifact_id")
            if action.startswith("draft:"):
                match = match and writing.get("section_ref") == options.get("section_ref")
            elif action.startswith("plan:"):
                match = match and row["query"] == query
        if match:
            return dict(row)
    return None


def writing_artifact(session, conversation_id: str, actor_id: int, options: dict) -> dict | None:
    """恢复同用户、同 AI 会话、同知识库的指定阶段产物。

    Raises:
        ValueError: 缺少输入、章节不存在，或产物阶段、知识库、归属不一致。
    """
    stage = options["stage"]
    artifact_id = options.get("artifact_id")
    if not artifact_id:
        if stage == "DRAFT":
            raise ValueError("请先生成提纲，并选择要起草的章节")
        if stage == "REVISE" and not options.get("source_text", "").strip():
            raise ValueError("请提供待修改原文，或选择已有章节稿件")
        return None
    # 同时验证来源 Run，避免客户端借用同会话内另一个知识库的产物。
    row = (
        session.execute(
            select(
                research_writing_artifact,
                research_run.c.request_payload,
                research_run.c.user_id,
                research_run.c.status,
            )
            .join(research_run, research_run.c.id == research_writing_artifact.c.run_id)
            .where(
                research_writing_artifact.c.id == artifact_id,
                research_writing_artifact.c.conversation_id == conversation_id,
                research_writing_artifact.c.is_deleted == 0,
                research_run.c.is_deleted == 0,
            )
        )
        .mappings()
        .first()
    )
    expected = {"PLAN": {"RQ"}, "DRAFT": {"PLAN"}, "REVISE": {"DRAFT", "REVISE"}}
    if (
        not row
        or row["user_id"] != actor_id
        or row["status"] != "SUCCESS"
        or row["stage"] not in expected.get(stage, set())
        or (row["request_payload"].get("writing") or {}).get("knowledge_base_id")
        != options["knowledge_base_id"]
    ):
        raise ValueError("前序产物必须属于当前会话、同一知识库及正确写作阶段")
    content = dict(row["content"])
    if stage == "DRAFT" and not any(
        section["section_id"] == options.get("section_ref")
        for section in content.get("sections", [])
    ):
        raise ValueError("请选择当前提纲中的有效章节")
    return content
