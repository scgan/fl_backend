"""把 AI 提取的固定条件编译为参数化题录 SQL，提供计数、稳定分页和正文就绪标记。"""

from __future__ import annotations

import hashlib

from sqlalchemy import and_, case, func, or_, select

from app.agents.ai_assistant_schemas import AssistantSearchPlan
from app.foreign_integration.literature import readable_corpus_clause
from app.foreign_integration.research_scope import ResearchScope
from app.foreign_integration.schema import foreign_corpus, foreign_corpus_index_job


def _build_query(session, actor_id, corpus_ids, plan: AssistantSearchPlan, scope):
    """构造与检索一致的过滤条件和权重，供分页与 id 快照复用。"""
    c = foreign_corpus.c
    fields = [c.keyword_cn, c.keyword_en]
    if not plan.keyword_only:
        fields += [c.title_cn, c.title_en, c.abstract_cn, c.abstract_en]
    scope = scope or ResearchScope("GLOBAL", 0)
    conditions = [
        c.id.in_(scope.corpus_query(session, actor_id, include_unindexed=True)),
        c.is_deleted == 0,
        readable_corpus_clause(actor_id),
    ]
    if corpus_ids is not None:
        conditions.append(c.id.in_(corpus_ids))
    weights = []
    for group in plan.concepts:
        group_matches = []
        for word in group:
            for column in fields:
                match = column.icontains(word, autoescape=True)
                group_matches.append(match)
                weight = (
                    3 if column.name.startswith("title") else 2 if "keyword" in column.name else 1
                )
                weights.append(case((match, weight), else_=0))
        conditions.append(or_(*group_matches))
    if plan.author:
        conditions.append(c.author.icontains(plan.author.strip(), autoescape=True))
    if plan.language:
        aliases = (
            ["中文", "汉语", "chinese", "zh-cn", "zh"]
            if plan.language == "ZH"
            else ["英文", "英语", "english", "en-us", "en"]
        )
        conditions.append(func.lower(c.language).in_(aliases))
    if plan.year_from:
        conditions.append(c.publish_time >= plan.year_from)
    if plan.year_to:
        conditions.append(c.publish_time <= plan.year_to)
    return and_(*conditions), weights


def search_document_index(
    session,
    actor_id,
    corpus_ids,
    plan: AssistantSearchPlan,
    *,
    limit=500,
    scope: ResearchScope | None = None,
) -> dict:
    """按检索条件返回有序题录 id 快照与可浏览总数，供后续翻页复用。

    Args:
        session: 只读查询用的同步会话。
        actor_id: 已校验归属的用户。
        corpus_ids: 明确选文集合；None 表示不附加选文限制。
        plan: 已校验的检索条件。
        limit: 快照最多保留的 id 数（与对外 500 篇上限一致）。
        scope: 资源范围。

    Returns:
        `{"ids": [...], "total": 可浏览总数}`，ids 顺序与检索排序一致。
    """
    where, weights = _build_query(session, actor_id, corpus_ids, plan, scope)
    c = foreign_corpus.c
    ordered = select(c.id).where(where)
    ordered = ordered.order_by(sum(weights).desc() if weights else c.id.desc(), c.id.desc())
    ids = [row[0] for row in session.execute(ordered.limit(limit))]
    # 对外总数本就封顶五百，直接使用快照长度，避免为同一条件额外全表计数。
    return {"ids": ids, "total": min(len(ids), 500)}


def search_documents(
    session,
    actor_id,
    corpus_ids,
    plan: AssistantSearchPlan,
    *,
    page=1,
    page_size=10,
    lexical_index="foreign_chunks_v1",
    scope: ResearchScope | None = None,
    document_ids: list[int] | None = None,
    total: int | None = None,
) -> dict:
    """按当前资源关联和明确选文查询题录，不加载整库成员或执行模型提供的 SQL。

    Args:
        session: 调用方持有的短同步会话，仅执行只读查询。
        actor_id: 已通过账号与运行归属校验的用户。
        corpus_ids: 明确选文集合；None 表示不添加选文限制，空列表仍表示空集合。
        plan: 已验证的主题与元数据筛选条件。
        page: 从一开始的页码。
        page_size: 每页固定十篇，最多读取前五十页。
        lexical_index: 当前配置的正文词法索引名称。
        scope: 知识库、根数据库或全库来源，按当前关联查询并复核资源权限。
        document_ids: 首次检索保存的有序 ID 快照；None 表示兼容旧的直接检索。
        total: 快照的可浏览总数，最大五百篇。

    Returns:
        截为五百篇的可浏览总数及当前页题录；上限由后端统一控制。

    Raises:
        ValueError: 分页参数越界。
    """
    if not 1 <= page <= 50 or page_size != 10:
        raise ValueError("文献分页参数不合适")
    c = foreign_corpus.c
    scope = scope or ResearchScope("GLOBAL", 0)
    # 1. 快照仅固定排序，不授予永久权限；当前页仍复核资源成员和材料可见性。
    # 不再次匹配或排序主题词，详情更新不会触发全库扫描；移出资源的材料不会泄露。
    if document_ids is not None:
        visible = scope.corpus_query(session, actor_id, include_unindexed=True)
        chunk = list(document_ids[(page - 1) * page_size : page * page_size])
        rows = []
        if chunk:
            j = foreign_corpus_index_job.c
            fetched = session.execute(
                select(
                    foreign_corpus,
                    j.status.label("index_status"),
                    j.vector_status,
                    j.lexical_status,
                    j.lexical_index,
                    j.attachment_digest,
                )
                .outerjoin(foreign_corpus_index_job, (j.corpus_id == c.id) & (j.is_deleted == 0))
                .where(
                    c.id.in_(chunk), c.id.in_(visible), readable_corpus_clause(actor_id),
                    True if corpus_ids is None else c.id.in_(corpus_ids),
                )
            ).mappings()
            by_id = {row["id"]: row for row in fetched}
            rows = [by_id[cid] for cid in chunk if cid in by_id]
        documents = []
        for row in rows:
            ready = (
                row["index_status"] == row["vector_status"] == row["lexical_status"] == "SUCCESS"
                and row["lexical_index"] == lexical_index
                and bool(row["is_vectorized"])
                and row["attachment_digest"]
                == hashlib.sha256((row["attachment"] or "").encode()).hexdigest()
            )
            documents.append(
                {
                    "corpus_id": row["id"],
                    "title": row["title_cn"] or row["title_en"] or f"文献 {row['id']}",
                    "author": row["author"] or "",
                    "year": row["publish_time"],
                    "source": row["source"] or "",
                    "snippet": (row["abstract_cn"] or row["abstract_en"] or "")[:350],
                    "readable": ready,
                }
            )
        return {
            "documents": documents,
            "total": min(int(total or 0), 500),
            "page": page,
            "page_size": page_size,
            "search_plan": plan.model_dump(mode="json"),
        }

    # 2. 兼容无快照的直接查询；词项绑定并转义，不加载全量题录到应用内存。
    where, weights = _build_query(session, actor_id, corpus_ids, plan, scope)
    total = session.scalar(select(func.count()).select_from(foreign_corpus).where(where))
    j = foreign_corpus_index_job.c
    rows = session.execute(
        select(
            foreign_corpus,
            j.status.label("index_status"),
            j.vector_status,
            j.lexical_status,
            j.lexical_index,
            j.attachment_digest,
        )
        .outerjoin(foreign_corpus_index_job, (j.corpus_id == c.id) & (j.is_deleted == 0))
        .where(where)
        .order_by(sum(weights).desc() if weights else c.id.desc(), c.id.desc())
        .limit(page_size)
        .offset((page - 1) * page_size)
    ).mappings()
    # 3. 只投影真实数据库题录；摘要不是回答证据，正文状态仅作选文提示。
    documents = []
    for row in rows:
        ready = (
            row["index_status"] == row["vector_status"] == row["lexical_status"] == "SUCCESS"
            and row["lexical_index"] == lexical_index
            and bool(row["is_vectorized"])
            and row["attachment_digest"]
            == hashlib.sha256((row["attachment"] or "").encode()).hexdigest()
        )
        documents.append(
            {
                "corpus_id": row["id"],
                "title": row["title_cn"] or row["title_en"] or f"文献 {row['id']}",
                "author": row["author"] or "",
                "year": row["publish_time"],
                "source": row["source"] or "",
                "snippet": (row["abstract_cn"] or row["abstract_en"] or "")[:350],
                "readable": ready,
            }
        )
    return {
        "documents": documents,
        "total": min(total, 500),
        "page": page,
        "page_size": page_size,
        "search_plan": plan.model_dump(mode="json"),
    }
