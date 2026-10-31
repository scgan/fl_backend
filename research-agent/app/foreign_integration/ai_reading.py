"""从同一AI会话恢复有序文献集合，解析阅读指代并冻结受控材料子集。

文献来源独立于聊天截断预算；题录、引用和阅读集合保持各自顺序。
页面提示只定位显示页与勾选项，材料按来源资源的当前权限与搜索条件校验。
"""

from __future__ import annotations

from sqlalchemy import select

from app.agents.ai_assistant_schemas import AssistantSearchPlan
from app.foreign_integration.assistant_search import search_documents
from app.foreign_integration.literature import selection_rows, selection_version
from app.foreign_integration.research_scope import ResearchScope
from app.foreign_integration.schema import research_conversation, research_run


def _documents(items: list[dict]) -> list[dict]:
    """将真实题录投影成统一展示字段，并按首次出现的顺序去重。"""
    result, seen = [], set()
    for item in items:
        key = item.get("corpus_id")
        if type(key) is not int or key <= 0:
            continue
        if key in seen:
            if item.get("citation_index"):
                next(doc for doc in result if doc["corpus_id"] == key)["citation_indices"].append(
                    item["citation_index"]
                )
            continue
        seen.add(key)
        result.append({
            "corpus_id": key,
            "title": item.get("title") or item.get("document_title") or f"文献 {key}",
            "author": item.get("author") or "",
            "year": item.get("year"),
            "source": item.get("source") or "",
            "snippet": item.get("snippet") or "",
            "citation_indices": [item["citation_index"]] if item.get("citation_index") else [],
        })
    return result


def reading_target_documents(reading: dict) -> list[dict]:
    """按原顺序投影本轮ASK实际目标，兼容旧快照仍保存整个阅读集合的情况。

    Raises:
        ValueError: 本轮目标不属于原阅读集合，不能用其他篇目替代。
    """
    documents = _documents(reading.get("source_documents", []))
    targets = reading.get("target_corpus_ids")
    if reading.get("task") != "ASK" or not targets:
        return documents
    allowed = {item["corpus_id"] for item in documents}
    if not set(targets) <= allowed:
        raise ValueError("阅读目标不属于原材料集合，请重新选择")
    return [item for item in documents if item["corpus_id"] in set(targets)]


def reading_sources(session, run: dict, *, lexical_index: str) -> list[dict]:
    """读取同用户同会话的最近成功来源，并校验页面提供的分页和勾选提示。

    Args:
        session: 当前线程管理的短数据库会话，不跨异步等待复用。
        run: 当前AI运行，包含已持久化的页面提示。
        lexical_index: 与题录分页接口相同的索引就绪判定配置。

    Returns:
        新到旧的真实来源，每组包含原Run、类型及有序题录；无材料轮次不占组数。

    Raises:
        ValueError: 页面提示指向其他会话、失效搜索或不属于来源的勾选项。
    """
    # 1. 不依赖history的消息数/字节数，避免长回答挤掉“上述文献”的真实ID。
    rows = session.execute(
        select(research_run).where(
            research_run.c.conversation_id == run["conversation_id"],
            research_run.c.user_id == run["user_id"],
            research_run.c.status == "SUCCESS",
            research_run.c.is_deleted == 0,
            research_run.c.id != run["id"],
        ).order_by(research_run.c.created_at.desc(), research_run.c.id.desc())
    ).mappings().all()
    hint = run["request_payload"].get("reading_context") or {}
    if hint and hint["source_run_id"] not in {row["id"] for row in rows}:
        raise ValueError("文献来源不属于当前会话，请重新选择")
    sources = []
    for row in rows:
        if len({item["source_run_id"] for item in sources}) >= 10 and (
            row["id"] != hint.get("source_run_id")
        ):
            continue
        payload = (row["result_payload"] or {}).get("payload", {})
        if (row["result_payload"] or {}).get("result_type") != "AI_ASSISTANT":
            continue
        reading = payload.get("reading_result") or {}
        groups = []
        if reading.get("status") == "SUCCESS":
            groups.append(("READING", reading_target_documents(reading)))
        if payload.get("intent") == "SEARCH":
            groups.append(("CANDIDATES", _documents(payload.get("documents", []))))
        elif payload.get("intent") == "QA":
            # 候选Evidence可能未被正文采用；“刚才引用”只使用通过校验的实际引用。
            groups.append(("CITATIONS", _documents(payload.get("citations", []))))
            candidates = _documents(payload.get("documents", []))
            if candidates:
                groups.append(("CANDIDATES", candidates))
        # 2. 当前页优先沿用检索 ID 快照，并在资源当前权限内复核；旧结果兼容原条件查询。
        if hint.get("source_run_id") == row["id"]:
            selected = hint.get("selected_ids", [])
            if payload.get("search_plan") and (hint.get("page", 1) != 1 or selected):
                ordered = row["request_payload"].get("corpus_ids") or None
                resource = ResearchScope.for_run({**run, **row})
                plan = AssistantSearchPlan.model_validate(payload["search_plan"])
                index = payload.get("document_index") or {}
                page = search_documents(
                    session, run["user_id"], ordered, plan,
                    page=hint.get("page", 1), lexical_index=lexical_index, scope=resource,
                    document_ids=index.get("ids"), total=index.get("total"),
                )
                groups = [("CANDIDATES", _documents(page["documents"]))]
                if selected:
                    if ordered is not None and not set(selected) <= set(ordered):
                        raise ValueError("勾选文献超出原搜索范围")
                    matches = []
                    for number in range(1, (len(selected) + 9) // 10 + 1):
                        matches += search_documents(
                            session, run["user_id"], selected, plan,
                            page=number, lexical_index=lexical_index, scope=resource,
                        )["documents"]
                    by_id = {item["corpus_id"]: item for item in matches}
                    if set(by_id) != set(selected):
                        raise ValueError("勾选文献不符合原搜索条件")
                    groups.append(("SELECTED", _documents([by_id[key] for key in selected])))
            elif selected:
                by_id = {item["corpus_id"]: item for _, docs in groups for item in docs}
                if not set(selected) <= set(by_id):
                    raise ValueError("勾选文献不属于原结果")
                groups.append(("SELECTED", [by_id[key] for key in selected]))
        for kind, documents in groups:
            if documents:
                sources.append({
                    "source_run_id": row["id"], "kind": kind,
                    "query": row["query"][:200], "documents": documents,
                    "total": payload.get("total") if kind == "CANDIDATES" else len(documents),
                    # 页面连续编号与模型使用的页内位置分开，避免第十一篇被当成越界位置。
                    "page": hint.get("page", 1) if (
                        kind == "CANDIDATES" and payload.get("search_plan")
                        and hint.get("source_run_id") == row["id"]
                    ) else 1,
                })
        # 保留最近十个有文献的轮次；仅聊天不缩短可指代的来源窗口。
        source_ids = {item["source_run_id"] for item in sources}
        if len(source_ids) >= 10 and (not hint or hint["source_run_id"] in source_ids):
            break
    if hint:
        # 页面选中的旧卡片可以明确成为当前来源，不因有更新的搜索结果而被悄悄替换。
        sources.sort(key=lambda item: item["source_run_id"] != hint["source_run_id"])
    return sources


def reading_followup_scope(
    session, actor_id: int, conversation_id: str, source_run_id: str
) -> dict:
    """核验阅读卡片归属，按原材料顺序恢复 ASK 并取得当前可读材料版本。

    Args:
        session: 当前线程拥有的短数据库会话。
        actor_id: 当前已认证用户。
        conversation_id: 当前AI会话，不能跨会话继承阅读卡片。
        source_run_id: 用户点击的原阅读分析或追问卡片。

    Returns:
        仅来自该阅读卡片的固定材料快照，任务始终为ASK。

    Raises:
        ValueError: 卡片不属于当前用户/会话、未成功完成，或不是有效的阅读结果。
        PermissionError: 原材料已删除或无读取权限。
    """
    # 1. 核验来源与会话归属；即使客户端伪造卡片ID，也不能读取其他会话材料。
    row = session.execute(select(research_run).join(
        research_conversation, research_run.c.conversation_id == research_conversation.c.id,
    ).where(
        research_run.c.id == source_run_id,
        research_run.c.conversation_id == conversation_id,
        research_run.c.user_id == actor_id,
        research_run.c.status == "SUCCESS", research_run.c.is_deleted == 0,
        research_conversation.c.user_id == actor_id,
        research_conversation.c.entry_source == "AI", research_conversation.c.is_deleted == 0,
    )).mappings().first()
    payload = (row["result_payload"] or {}) if row else {}
    reading = (payload.get("payload") or {}).get("reading_result") or {}
    if payload.get("result_type") != "AI_ASSISTANT" or reading.get("status") != "SUCCESS":
        raise ValueError("文献追问需要当前会话中已完成的阅读卡片")
    documents = _documents(reading.get("source_documents", []))
    ids = [item["corpus_id"] for item in documents]
    if not 1 <= len(ids) <= 20 or ids != reading.get("corpus_ids"):
        raise ValueError("阅读卡片缺少有效材料，请重新分析后继续提问")
    # 2. 继续追问只沿用卡片实际回答的篇目；当前版本用于匹配可复用分析，不阻止新一轮读取。
    documents = reading_target_documents(reading)
    ids = [item["corpus_id"] for item in documents]
    version = selection_version(selection_rows(session, actor_id, ids))
    return {
        "status": "READY", "task": "ASK", "source_run_id": source_run_id,
        "source_kind": "READING", "source_documents": documents,
        "corpus_ids": ids, "scope_version": version,
    }


def freeze_reading_scope(session, run: dict, state: dict, allowed: set[int] | None) -> dict:
    """从服务提供的来源解析序号并冻结子集，拒绝伪造来源、越界序号和超量选择。

    Args:
        session: 当前线程拥有的短数据库会话。
        run: 当前运行的身份信息。
        state: 文献来源及模型提出的来源类型、序号、任务。
        allowed: 父运行明确选文；None 时只按绑定资源及当前文献权限限制。

    Returns:
        含状态、来源、有序篇目及版本的快照，歧义或无材料时返回可展示的澄清。

    Raises:
        PermissionError: 目标超出当前Run的授权范围，或材料已不可读。
    """
    # 1. 显式推荐追问直接恢复原卡片，完全不采用模型对任务或材料来源的判断。
    options = run["request_payload"].get("reading") or {}
    if options:
        scope = reading_followup_scope(
            session, int(run["user_id"]), run["conversation_id"], options["source_run_id"]
        )
        _authorize_reading_ids(session, run, scope["corpus_ids"], allowed)
        return scope
    # 2. 自然语言指代不确定时结束本轮并询问，不能退回全库检索冒充阅读指定文献。
    def clarify(message):
        """生成可持久化的缺少输入结果，不启动正文读取。"""
        return {"status": "NEEDS_INPUT", "message": message, "source_documents": []}

    if state.get("needs_clarification"):
        return clarify(state.get("clarification_question") or "请明确要分析哪一组文献。")
    sources = state.get("reading_sources", [])
    source_id, kind = state.get("reading_source_run_id"), state.get("reading_source_kind")
    reference = state.get("reading_reference", "AUTO")
    positions = state.get("reading_positions", [])
    if reference == "SPECIFIC" and not positions:
        return clarify("没有定位到指定文献，请确认篇名或提供当前列表中的序号。")
    # 语义由模型解释，范围优先级由服务落实；泛指不能把勾选项扩大成整个搜索列表。
    # 只在同一来源内切换集合，显式指定旧来源、篇名、序号或整组时仍尊重用户限定。
    active_id = source_id or (sources[0]["source_run_id"] if sources else None)
    selected = next((item for item in sources if (
        item["source_run_id"] == active_id and item["kind"] == "SELECTED"
    )), None)
    if reference == "SELECTED" or (kind == "SELECTED" and reference == "AUTO"):
        if not selected:
            return clarify("当前没有勾选文献，请先选择要分析的篇目。")
        source_id, kind = active_id, "SELECTED"
    elif reference == "AUTO" and not positions and not state.get("reading_all_results"):
        if selected:
            source_id, kind = active_id, "SELECTED"
    elif reference == "LIST" and kind == "SELECTED":
        kind = None
    matching = [item for item in sources if (
        (not source_id or item["source_run_id"] == source_id)
        and (not kind or item["kind"] == kind)
        and (kind == "SELECTED" or item["kind"] != "SELECTED")
    )]
    if not matching:
        return clarify("没有找到对应的前文文献，请先搜索文献，或明确要分析的篇目。")
    source = matching[0]
    if source.get("error"):
        return clarify(source["error"])
    documents = source["documents"]
    if state.get("reading_all_results") and (source.get("total") or 0) > len(documents):
        return clarify(
            f"检索共{source['total']}篇，当前显示{len(documents)}篇；"
            "一次最多分析20篇，请指定篇目或勾选要分析的文献。"
        )
    if positions:
        if any(type(pos) is not int or not 1 <= pos <= len(documents) for pos in positions):
            return clarify(f"这组共有 {len(documents)} 篇文献，请提供有效的篇目序号。")
        if len(set(positions)) != len(positions):
            return clarify("篇目序号有重复，请重新指定要分析的文献。")
        documents = [documents[pos - 1] for pos in positions]
    if len(documents) > 20:
        return clarify("一次最多分析20篇文献，请指定篇目序号或勾选要分析的文献。")
    ids = [item["corpus_id"] for item in documents]
    if not ids:
        raise PermissionError("分析文献超出当前运行可读范围，请重新搜索")
    _authorize_reading_ids(session, run, ids, allowed)
    # 3. 只记录实际阅读子集的顺序和版本，供历史展示及选择可复用分析。
    version = selection_version(selection_rows(session, int(run["user_id"]), ids))
    return {
        "status": "READY", "source_run_id": source["source_run_id"],
        "source_kind": source["kind"], "corpus_ids": ids, "scope_version": version,
        "source_documents": documents, "task": state.get("reading_task", "ANALYZE"),
    }


def _authorize_reading_ids(session, run: dict, ids: list[int], allowed: set[int] | None) -> None:
    """只查询实际阅读篇目的当前资源归属，不枚举父运行的整库成员。

    Args:
        session: 当前线程管理的只读同步会话。
        run: 含用户、资源绑定及请求参数的父运行。
        ids: 本轮实际需要阅读的有序文献 ID。
        allowed: 父运行明确选文；None 表示没有额外选文限制。

    Raises:
        PermissionError: 任意篇目不属于明确选文或当前资源可读范围。
    """
    if allowed is not None and not set(ids) <= allowed:
        raise PermissionError("分析文献超出当前运行可读范围，请重新搜索")
    from app.foreign_integration.schema import foreign_corpus

    query = ResearchScope.for_run(run).corpus_query(
        session, int(run["user_id"]), include_unindexed=True,
    ).where(foreign_corpus.c.id.in_(ids))
    if set(session.scalars(query)) != set(ids):
        raise PermissionError("分析文献超出当前运行可读范围，请重新搜索")


def previous_reading_analysis(session, run: dict, scope: dict) -> dict | None:
    """恢复当前阅读范围的分析概览，显式点击沿来源链，自然语言使用最近同版本分析。

    Args:
        session: 当前线程的短数据库会话。
        run: 当前运行，显式追问参数用于定位用户点击的阅读卡片。
        scope: 已核验的篇目顺序及材料版本。

    Returns:
        同范围同版本的分析概览；没有匹配分析或来源链失效时为空。
    """
    requested = (run["request_payload"].get("reading") or {}).get("source_run_id")
    if requested:
        # 1. 显式点击旧卡片时沿父阅读链恢复分析，不被同材料后来的其他分析覆盖。
        seen = set()
        while requested and requested not in seen:
            seen.add(requested)
            payload = session.scalar(select(research_run.c.result_payload).where(
                research_run.c.id == requested,
                research_run.c.conversation_id == run["conversation_id"],
                research_run.c.user_id == run["user_id"],
                research_run.c.status == "SUCCESS", research_run.c.is_deleted == 0,
            ))
            reading = ((payload or {}).get("payload") or {}).get("reading_result") or {}
            if reading.get("corpus_ids") != scope["corpus_ids"] or (
                reading.get("scope_version") != scope["scope_version"]
            ):
                return None
            if reading.get("task") == "ANALYZE":
                return {"overview": reading.get("overview")}
            requested = reading.get("source_run_id") if (
                reading.get("source_kind") == "READING"
            ) else None
        return None
    # 2. 自然语言没有指定卡片时，沿用最近一次同材料同版本的分析。
    rows = session.execute(select(research_run.c.result_payload).where(
        research_run.c.conversation_id == run["conversation_id"],
        research_run.c.user_id == run["user_id"],
        research_run.c.status == "SUCCESS", research_run.c.is_deleted == 0,
    ).order_by(research_run.c.created_at.desc(), research_run.c.id.desc()))
    for payload in rows.scalars():
        reading = ((payload or {}).get("payload") or {}).get("reading_result") or {}
        if (reading.get("task") == "ANALYZE" and reading.get("status") == "SUCCESS"
            and reading.get("corpus_ids") == scope["corpus_ids"]
            and reading.get("scope_version") == scope["scope_version"]):
            return {"overview": reading.get("overview")}
    return None
