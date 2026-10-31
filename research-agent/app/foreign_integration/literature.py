"""按资源查询题录并授权明确选文，读取和保存实际采用的索引材料。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from contextlib import closing
from types import SimpleNamespace
from urllib.parse import urlencode

from qdrant_client import QdrantClient, models
from sqlalchemy import and_, or_, select

from app.foreign_integration.database import session_factory
from app.foreign_integration.retrieval import ForeignRetrievalTool, _source_from_hit
from app.foreign_integration.schema import foreign_corpus, foreign_corpus_index_job, foreign_user
from app.retrieval.schemas import CitationRead, EvidenceRead


def readable_corpus_clause(actor_id: int):
    """返回公共审核文献或本人个人材料的读取条件，删除状态由调用方另行限定。"""
    return or_(
        and_(foreign_corpus.c.status == 2, foreign_corpus.c.owner_id.is_(None)),
        and_(foreign_corpus.c.status == 4, foreign_corpus.c.owner_id == actor_id),
    )


def selection_rows(session, actor_id: int, corpus_ids: list[int]) -> dict[int, dict]:
    """读取固定范围的公共文献或本人个人材料，保留用户提交的顺序。

    Args:
        session: 调用方管理的只读数据库会话，本函数不提交事务。
        actor_id: 当前用户 ID，必须对应未停用、未删除账号。
        corpus_ids: 用户明确选择的文献 ID 列表，不自动扩展检索范围。

    Returns:
        按选择顺序插入的文献映射，同时附带当前索引构建及附件摘要状态。

    Raises:
        PermissionError: 账号失效，或任意所选文献不存在、删除、未审核或属于他人。
    """
    # 1. 公开题录仍受账号有效性约束；避免持有旧令牌的停用账号继续读取材料。
    if not session.scalar(
        select(foreign_user.c.id).where(
            foreign_user.c.id == actor_id,
            foreign_user.c.status == 1,
            foreign_user.c.is_deleted == 0,
        )
    ):
        raise PermissionError("账号已停用")
    # 2. 题录与索引状态一并读取；未建索引的文献也保留，后续以未就绪原因反馈用户。
    state = foreign_corpus_index_job.c
    rows = (
        session.execute(
            select(
                foreign_corpus,
                state.build_id.label("index_build_id"),
                state.attachment_digest.label("index_attachment_digest"),
                state.chunk_count.label("index_chunk_count"),
                state.status.label("index_status"),
                state.vector_status,
                state.lexical_status,
                state.lexical_index,
            )
            .outerjoin(
                foreign_corpus_index_job,
                (state.corpus_id == foreign_corpus.c.id) & (state.is_deleted == 0),
            )
            .where(
                foreign_corpus.c.id.in_(corpus_ids),
                readable_corpus_clause(actor_id),
                foreign_corpus.c.is_deleted == 0,
            )
        )
        .mappings()
        .all()
    )
    # 3. 整个选择集合必须可读，不能静默丢失其中一篇后改变“第一篇/第二篇”的语义。
    result = {int(row["id"]): dict(row) for row in rows}
    if set(result) != set(corpus_ids):
        raise PermissionError("部分所选文献不存在或已不可阅读，请返回搜索重新选择")
    return {key: result[key] for key in corpus_ids}


def selection_version(rows: dict[int, dict]) -> int:
    """为实际阅读材料生成稳定指纹，用于初次分析去重和匹配可复用的历史分析。

    Args:
        rows: 按用户选择顺序组织的题录映射；顺序也参与指纹计算。

    Returns:
        截为非负 63 位整数的 SHA-256 摘要，不用于冻结全库或资源成员。
        公开文献保持旧版指纹兼容；个人材料额外冻结身份。该值不是递增版本号。
    """
    data = []
    for key, row in rows.items():
        item = (
            key,
            row["attachment"],
            row["index_build_id"],
            row["index_attachment_digest"],
            row["title_cn"],
            row["title_en"],
        )
        # 旧公共会话的 scope_version 继续有效；只有个人来源需要追加所有权快照。
        if row["status"] == 4:
            item += ("PERSONAL", row["owner_id"])
        data.append(item)
    digest = hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class LiteratureMaterialTool(ForeignRetrievalTool):
    """复用真实 Qdrant 来源契约，完整读取所选文献而不执行主题 top-k。"""

    async def discover(self, plan):
        """在线程内按当前资源关联和搜索条件查询首批题录。

        Args:
            plan: 模型提取并通过 Schema 验证的元数据检索条件。

        Returns:
            首批十篇题录、有序 ID 快照、最多五百篇的总数及固定检索条件。

        Raises:
            PermissionError: 当前账号或资源已无访问权限。
        """
        return await asyncio.to_thread(self._discover_sync, plan)

    def _discover_sync(self, plan):
        """首次检索生成有序 ID 快照，只读取首批题录详情，后续翻页不重做主题检索。"""
        from app.foreign_integration.assistant_search import search_document_index, search_documents

        with session_factory()() as session:
            index = search_document_index(
                session, self.actor_id, self.document_order, plan, scope=self.scope,
            )
            result = search_documents(
                session, self.actor_id, self.document_order, plan,
                lexical_index=self.settings.opensearch_index, scope=self.scope,
                document_ids=index["ids"], total=index["total"],
            )
            return {**result, "document_index": index}

    async def recommend(self, readings):
        """在已选择材料的当前权限内，异步调用库内文献推荐器。

        Args:
            readings: 图中逐篇分析的结果，作为推荐查询的题录和摘要输入。

        Returns:
            推荐名单与检索完成状态；候选不会写入本工具的回答来源映射。

        Raises:
            PermissionError: 推荐开始时账号或所选材料不可读。
        """
        from app.foreign_integration.recommendations import LiteratureRecommender

        self._selection()
        # 推荐器内使用同步数据库及检索客户端，移到线程避免阻塞汇总分支的模型等待。
        result = await asyncio.to_thread(
            LiteratureRecommender(self.actor_id, self.settings).recommend, readings
        )
        return result

    async def filter_external_recommendations(self, papers, readings):
        """在推荐前授权实际选文，并排除与整个文献库重复的联网题录。"""
        from app.foreign_integration.recommendations import LiteratureRecommender

        self._selection()
        result = await asyncio.to_thread(
            LiteratureRecommender(self.actor_id, self.settings).filter_external, papers, readings,
        )
        return result

    def _selection(self) -> dict:
        """读取实际选文的当前题录及索引状态，不比较旧任务范围或材料指纹。

        Returns:
            按用户选择顺序排列的最新题录与索引状态。

        Raises:
            PermissionError: 账号或文献不可读，或材料不属于当前资源。
            ValueError: 完整阅读没有明确选文。
        """
        if not self.document_order:
            raise ValueError("完整阅读需要明确选择文献")
        with session_factory()() as session:
            members = self.scope.corpus_query(
                session, self.actor_id, include_unindexed=True,
            ).where(foreign_corpus.c.id.in_(self.document_order))
            if set(session.scalars(members)) != set(self.document_order):
                raise PermissionError("所选文献不属于当前可读范围")
            return selection_rows(session, self.actor_id, self.document_order)

    async def load_materials(self) -> tuple[list[dict], list[EvidenceRead], list[CitationRead]]:
        """在线程中完整加载所选材料，避免同步数据库和 Qdrant 分页阻塞图执行。

        Returns:
            按选择顺序排列的题录、全部已核验原文 Evidence 和对应 Citation。
        """
        return await asyncio.to_thread(self._load_materials_sync)

    def _load_materials_sync(self):
        """逐篇完整回取固定构建的片段，并建立本轮可信原文来源映射。

        Returns:
            documents 保留每篇文献的就绪状态与原因；evidence 和 citations 只包含
            全部片段均通过版本、哈希及原文定位检查的文献，编号跨文献连续递增。

        Raises:
            PermissionError: 开始读取时账号、所选文献或资源归属已不可用。
            ValueError: 单篇片段超过十万条，拒绝静默截断后宣称已阅读全文。
        """
        # 1. 读取实际选文；按选择顺序建立题录，未就绪文献向图报告明确原因。
        rows = self._selection()
        documents, evidence, citations = [], [], []
        with closing(QdrantClient(url=self.settings.qdrant_url, timeout=60)) as client:
            for corpus_id, row in rows.items():
                document = {
                    "corpus_id": corpus_id,
                    "title": row["title_cn"] or row["title_en"] or f"文献 {corpus_id}",
                    "author": row["author"] or "",
                    "year": row["publish_time"],
                    "keyword_cn": row["keyword_cn"],
                    "keyword_en": row["keyword_en"],
                    "status": "UNAVAILABLE",
                    "reason": "文献解析尚未就绪",
                    "attachment_paths": [],
                    "chunk_count": 0,
                }
                documents.append(document)
                digest = hashlib.sha256((row["attachment"] or "").encode()).hexdigest()
                if (
                    not row["is_vectorized"]
                    or row["index_status"] != "SUCCESS"
                    or row["vector_status"] != "SUCCESS"
                    or not row["index_build_id"]
                    or row["index_attachment_digest"] != digest
                ):
                    continue
                # 2. 按文献和构建 ID 分页取全量 payload；这里不做主题 top-k，也不下载原始 PDF。
                points, offset = [], None
                while True:
                    batch, offset = client.scroll(
                        collection_name=self.settings.qdrant_collection,
                        scroll_filter=models.Filter(
                            must=[
                                models.FieldCondition(
                                    key="corpus_id", match=models.MatchValue(value=corpus_id)
                                ),
                                models.FieldCondition(
                                    key="build_id",
                                    match=models.MatchValue(value=row["index_build_id"]),
                                ),
                            ]
                        ),
                        limit=256,
                        offset=offset,
                        with_payload=True,
                        with_vectors=False,
                    )
                    points.extend(batch)
                    # 不静默截断大材料；需要分批的正文由 graph 处理。
                    if len(points) > 100_000:
                        raise ValueError("单篇文献解析片段过多，请拆分材料后重新选择")
                    if offset is None:
                        break
                # 3. 对照索引登记数量，并按附件与片段序号恢复阅读顺序，避免缺页材料被当作完整输入。
                if len(points) != row["index_chunk_count"]:
                    document["reason"] = "解析片段不完整，请重新构建文献索引"
                    continue
                points.sort(
                    key=lambda point: (
                        str((point.payload or {}).get("file_path", "")),
                        int((point.payload or {}).get("chunk_index", 0)),
                    )
                )
                # 4. 每个片段都必须通过哈希与原文定位核验；一片失败会撤下整篇的分析资格。
                verified = []
                for point in points:
                    payload = point.payload or {}
                    content = payload.get("text")
                    if (
                        not isinstance(content, str)
                        or payload.get("attachment_digest") != digest
                        or payload.get("content_hash")
                        != hashlib.sha256(content.encode()).hexdigest()
                    ):
                        break
                    source = _source_from_hit(
                        SimpleNamespace(id=point.id, payload=payload, score=1.0), row
                    )
                    if source is None:
                        break
                    verified.append((point, source))
                if not verified or len(verified) != len(points):
                    document["reason"] = "解析材料缺少可核验的原文或页码"
                    continue
                # 5. 完整核验后才标记 READY，并生成共享编号的 Evidence/Citation 与发布来源映射。
                document.update(
                    status="READY",
                    reason="",
                    chunk_count=len(verified),
                    attachment_paths=sorted({source.file_path for _, source in verified}),
                )
                for point, source in verified:
                    chunk_id = uuid.UUID(int=int(point.id))
                    self.sources[chunk_id] = source
                    index = len(evidence) + 1
                    evidence.append(
                        EvidenceRead(
                            citation_index=index,
                            document_id=uuid.UUID(int=corpus_id),
                            chunk_id=chunk_id,
                            document_title=document["title"],
                            authors=[{"name": document["author"]}] if document["author"] else [],
                            publication_year=row["publish_time"],
                            journal=row["source"],
                            language_code=row["language"] or "",
                            page_start=source.page_start,
                            page_end=source.page_end,
                            section=None,
                            content=source.content,
                            relevance_score=1.0,
                            retrieval_score=1.0,
                            vector_rank=None,
                            lexical_rank=None,
                        )
                    )
                    citations.append(
                        CitationRead(
                            citation_index=index,
                            document_id=uuid.UUID(int=corpus_id),
                            chunk_id=chunk_id,
                            document_title=document["title"],
                            page_start=source.page_start,
                            page_end=source.page_end,
                            section=None,
                            quote=" ".join(source.content.split())[:360],
                            file_url=(
                                f"{self.settings.public_api_base}/literature/documents/{corpus_id}/file?"
                                + urlencode(
                                    {"corpus_id": corpus_id, "attachment_path": source.file_path}
                                )
                                + f"#page={source.page_start}"
                            ),
                        )
                    )
        # 6. 返回本次已经核验的材料快照，之后的资源成员或附件变化不使本次读取失败。
        return documents, evidence, citations

    def __init__(self, **kwargs):
        """保留选择顺序，并初始化继承的鉴权检索及真实来源缓存。

        Args:
            **kwargs: 传给 ForeignRetrievalTool 的账号、范围、版本和配置参数；
                fixed_scope 为有序明确选文，None 表示按资源进行题录发现或检索。
        """
        # 父类范围可用于白名单判断；单独保留列表避免后续集合操作打乱指代顺序。
        self.document_order = (
            list(kwargs["fixed_scope"]) if kwargs["fixed_scope"] is not None else None
        )
        super().__init__(**kwargs)
