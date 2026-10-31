"""编排 Foreign 向量与 BM25 混合召回，核验共享快照并生成真实引用。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlencode

from httpx import TimeoutException
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException
from sqlalchemy import select

from app.common.bailian_client import embed_texts, embedding_config
from app.foreign_integration.config import ForeignSettings
from app.foreign_integration.database import session_factory
from app.foreign_integration.lexical import lexical_search
from app.foreign_integration.research_scope import ResearchScope
from app.foreign_integration.schema import foreign_corpus, foreign_corpus_index_job
from app.retrieval.languages import normalize_source_language
from app.retrieval.query_processing import BilingualQueryProcessor
from app.retrieval.ranking import RankedHit, reciprocal_rank_fusion, rerank_evidence, term_coverage
from app.retrieval.schemas import (
    CitationRead,
    EvidenceRead,
    RetrievalFilters,
    RetrievalResponse,
)

LOGGER = logging.getLogger(__name__)


class RetrievalDependencyError(RuntimeError):
    """仅携带受控阶段和错误类型，允许步骤记录诊断原因而不暴露供应商响应。"""


def _is_qdrant_timeout(error: Exception) -> bool:
    """识别 Qdrant 包装的 HTTP 超时，不把配置或服务端业务错误当成可重试错误。"""
    if isinstance(error, ResponseHandlingException):
        error = error.source
    return isinstance(error, TimeoutException | TimeoutError)


@dataclass(frozen=True)
class SourcePoint:
    """固定一次真实 Qdrant 命中的可回查定位与原文。"""

    corpus_id: int
    point_id: str
    chunk_key: str
    file_path: str
    content: str
    page_start: int | None
    page_end: int | None
    score: float


class ForeignRetrievalTool:
    """绑定 Foreign 用户和运行范围，向四个现有 LangGraph 提供受控证据。"""

    def __init__(
        self,
        *,
        actor_id: int,
        knowledge_base_id: int,
        fixed_scope: list[int] | None,
        fixed_version: int,
        settings: ForeignSettings,
        database_id: int | None = None,
    ) -> None:
        """保存运行时授权边界，外部客户端按实际查询延迟建立。

        Args:
            actor_id: 已认证的 Foreign 用户 ID。
            knowledge_base_id: 运行固定的私有知识库 ID。
            fixed_scope: 用户明确选择的文献 ID；None 表示按当前资源查询。
            fixed_version: 兼容图协议的历史字段，不再用于校验整库版本。
            settings: Foreign Qdrant、Embedding 和文件入口配置。
            database_id: 可选的根数据库 ID，与知识库来源互斥。
        """
        self.actor_id = actor_id
        # 真实来源集中解析；图仍调用同一个检索协议，两种来源不复制召回逻辑。
        # 资源引用实时解析成员，明确选文才保留小范围 ID；全库不展开整库清单。
        self.scope = (
            ResearchScope.from_ids(knowledge_base_id or None, database_id)
            if knowledge_base_id or database_id else ResearchScope("GLOBAL", 0)
        )
        self.knowledge_base_id = self.scope.graph_id.int
        self.fixed_scope = set(fixed_scope) if fixed_scope is not None else None
        self.fixed_version = fixed_version
        self.settings = settings
        # 本工具实际召回并核验的切块来源映射，发布层据此拒绝模型伪造的引用。
        self.sources: dict[uuid.UUID, SourcePoint] = {}

    async def search(
        self,
        *,
        knowledge_base_id: uuid.UUID,
        query: str,
        mode: str,
        filters: RetrievalFilters | None = None,
        evidence_limit: int | None = None,
    ) -> RetrievalResponse:
        """在线复核 Foreign 范围，以共享向量空间和 BM25 混合检索真实片段。

        Args:
            knowledge_base_id: 图内 UUID 包装的 Foreign 知识库整数 ID。
            query: 当前研究问题。
            mode: FAST 或 DEEP。
            filters: 只会缩小当前知识库范围的结构化筛选。
            evidence_limit: 专业 Agent 请求的最大证据数量。

        Returns:
            满足现有图契约的 Evidence First 检索结果。

        Raises:
            PermissionError: 资源无权访问或图内资源与运行绑定不一致。
        """
        if knowledge_base_id.int != self.knowledge_base_id:
            raise PermissionError("Agent 检索知识库与运行上下文不符")
        return await asyncio.to_thread(
            self._search_sync, query, mode, filters or RetrievalFilters(), evidence_limit
        )

    def _search_sync(
        self,
        query: str,
        mode: str,
        filters: RetrievalFilters,
        evidence_limit: int | None,
    ) -> RetrievalResponse:
        """在独立线程和短数据库会话中完成白名单、混合召回与来源回查。

        Args:
            query: 用户问题或图生成的受限查询。
            mode: 证据数量模式。
            filters: 文献元数据收窄条件。
            evidence_limit: 可选上限。

        Returns:
            真实来源的检索结果；未达分数或页码要求时明确证据不足。

        Raises:
            RetrievalDependencyError: 某个检索依赖失败，附带脱敏阶段说明。
            RuntimeError: 其他召回异常，底层内容不进入用户提示。
        """
        processed = BilingualQueryProcessor().process(query)
        # 1. 全库普通查询直接召回；有资源或明确筛选时按当前关联收窄候选。
        # 全库只校验账号，不读取全量题录及构建列表，命中后仅回查有限候选的权限和索引。
        unbounded = (
            self.scope.kind == "GLOBAL" and self.fixed_scope is None
            and filters == RetrievalFilters()
        )
        if unbounded:
            with session_factory()() as session:
                self.scope.authorize(session, self.actor_id)
            corpus_rows = None
        else:
            corpus_rows = self._load_scope(filters)
            if not corpus_rows:
                return self._empty(query, processed)

        # 2. 两路召回后从 Qdrant 按 ID 回取候选，词法独有命中也能生成引用。
        try:
            vector_hits, lexical_hits, points = self._recall(processed.bilingual, corpus_rows, mode)
        except RetrievalDependencyError:
            raise
        except Exception as error:
            LOGGER.warning("Retrieval failed stage=recall error_type=%s", type(error).__name__)
            raise RuntimeError("检索索引暂不可用，请稍后重试") from error

        # 3. 只回查真实命中文献；变化或失效的候选被剔除，不因其他成员变化终止问答。
        candidate_ids = set()
        for point in points:
            try:
                candidate_ids.add(uuid.UUID(int=int((point.payload or {})["corpus_id"])))
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
        if filters.document_ids:
            candidate_ids.intersection_update(filters.document_ids)
        if not candidate_ids:
            return self._empty(query, processed)
        candidate_filters = filters.model_copy(update={"document_ids": list(candidate_ids)})
        corpus_rows = self._load_scope(candidate_filters)
        vector_hits, lexical_hits, sources = _verified_candidates(
            vector_hits, lexical_hits, points, corpus_rows
        )
        fused = reciprocal_rank_fusion(vector_hits, lexical_hits, k=self.settings.rrf_k)
        ranked = rerank_evidence(
            processed.bilingual, fused, {key: value.content for key, value in sources.items()}
        )
        limit = min(evidence_limit or (10 if mode == "DEEP" else 5), 20)
        evidence: list[EvidenceRead] = []
        citations: list[CitationRead] = []
        # 4. 重排分与绝对支持条件共同决定证据，BM25 独有命中不要求向量阈值。
        for hit, score in ranked:
            source = sources[hit.chunk_id]
            semantic_support = (
                hit.vector_score is not None and hit.vector_score >= self.settings.min_vector_score
            )
            lexical_support = (
                hit.lexical_rank is not None
                and term_coverage(processed.bilingual, source.content)
                >= self.settings.min_lexical_coverage
            )
            if score < self.settings.min_evidence_score or not (
                semantic_support or lexical_support
            ):
                continue
            source = replace(source, score=hit.rrf_score)
            corpus_id = source.corpus_id
            row = corpus_rows[corpus_id]
            point_uuid = hit.chunk_id
            citation_index = len(evidence) + 1
            self.sources[point_uuid] = source
            title = str(row["title_cn"] or row["title_en"] or f"文献 {corpus_id}")
            author = str(row["author"] or "").strip()
            authors = [{"name": name.strip()} for name in author.split(",") if name.strip()]
            year = _year(row["publish_time"])
            language = _language(str(row["language"] or ""))
            quote = " ".join(source.content.split())[:360]
            file_url = (
                f"{self.settings.public_api_base}/public/file/corpus?"
                + urlencode({"corpus_id": corpus_id, "attachment_path": source.file_path})
                + f"#page={source.page_start}"
            )
            evidence.append(
                EvidenceRead(
                    citation_index=citation_index,
                    document_id=uuid.UUID(int=corpus_id),
                    chunk_id=point_uuid,
                    document_title=title,
                    authors=authors,
                    publication_year=year,
                    journal=row["source"],
                    language_code=language,
                    page_start=source.page_start,
                    page_end=source.page_end,
                    section=None,
                    content=source.content,
                    relevance_score=score,
                    retrieval_score=hit.rrf_score,
                    vector_rank=hit.vector_rank,
                    lexical_rank=hit.lexical_rank,
                )
            )
            citations.append(
                CitationRead(
                    citation_index=citation_index,
                    document_id=uuid.UUID(int=corpus_id),
                    chunk_id=point_uuid,
                    document_title=title,
                    page_start=source.page_start,
                    page_end=source.page_end,
                    section=None,
                    quote=quote,
                    file_url=file_url,
                )
            )
            if len(evidence) >= limit:
                break
        return RetrievalResponse(
            knowledge_base_id=uuid.UUID(int=self.knowledge_base_id),
            knowledge_base_version=self.fixed_version,
            original_query=query,
            detected_language=processed.language,
            rewritten_query=processed.rewritten,
            expanded_queries={"zh": processed.chinese, "en": processed.english},
            candidate_count=len(fused),
            evidence=evidence,
            citations=citations,
            insufficient_evidence=not evidence,
            message=None if evidence else "当前知识库没有可核验的相关证据",
        )

    def _load_scope(self, filters: RetrievalFilters) -> dict:
        """查询当前授权文献及已发布快照，不把旧向量状态当成词法就绪。

        Args:
            filters: 只用于缩小当前知识库范围的筛选条件。

        Returns:
            以文献 ID 为键的题录和当前构建版本。
        """
        with session_factory()() as session:
            state = foreign_corpus_index_job.c
            statement = (
                select(
                    foreign_corpus,
                    state.build_id.label("index_build_id"),
                    state.attachment_digest.label("index_attachment_digest"),
                )
                .join(foreign_corpus_index_job, state.corpus_id == foreign_corpus.c.id)
                .where(
                    foreign_corpus.c.id.in_(self.scope.corpus_query(session, self.actor_id)),
                    state.status == "SUCCESS",
                    state.vector_status == "SUCCESS",
                    state.lexical_status == "SUCCESS",
                    state.is_deleted == 0,
                    state.lexical_index == self.settings.opensearch_index,
                )
            )
            if self.fixed_scope is not None:
                statement = statement.where(foreign_corpus.c.id.in_(self.fixed_scope))
            if filters.document_ids:
                statement = statement.where(
                    foreign_corpus.c.id.in_(item.int for item in filters.document_ids)
                )
            if filters.year_from:
                statement = statement.where(foreign_corpus.c.publish_time >= filters.year_from)
            if filters.year_to:
                statement = statement.where(foreign_corpus.c.publish_time <= filters.year_to)
            return {
                int(row["id"]): row
                for row in session.execute(statement).mappings()
                if _metadata_matches(row, filters)
                and hashlib.sha256((row["attachment"] or "").encode()).hexdigest()
                == row["index_attachment_digest"]
            }

    def _recall(self, query: str, corpus_rows: dict | None, mode: str) -> tuple:
        """在当前查询范围内召回，并批量读取供数据库授权校验的候选正文。

        Args:
            query: 双语查询。
            corpus_rows: 当前资源及筛选内的文献版本；None 表示全库，无成员清单过滤。
            mode: FAST 或 DEEP，控制每路候选数量。

        Returns:
            向量命中、词法命中及候选 point 列表。

        Raises:
            RetrievalDependencyError: 编码、向量查询、词法查询或结果回读失败。
        """
        # 1. 构建查询向量，编码服务错误与索引服务错误分开记录。
        try:
            config = embedding_config()
            vector = embed_texts([query], text_type="query", key=self.settings.embedding_api_key)[0]
        except Exception as error:
            LOGGER.warning("Retrieval failed stage=embedding error_type=%s", type(error).__name__)
            raise RetrievalDependencyError(
                f"查询向量编码失败（{type(error).__name__}），请检查向量编码服务"
            ) from error
        # 2. 沿用当前模型及资源白名单过滤，不因超时恢复放宽来源校验。
        limit = 50 if mode == "DEEP" else 30
        builds = (
            [row["index_build_id"] for row in corpus_rows.values()]
            if corpus_rows is not None else None
        )
        must = [
            models.FieldCondition(
                key="embedding_model", match=models.MatchValue(value=config.model),
            ),
            models.FieldCondition(
                key="embedding_dimension", match=models.MatchValue(value=config.dimension),
            ),
        ]
        if corpus_rows is not None:
            must.extend([
                models.FieldCondition(
                    key="corpus_id", match=models.MatchAny(any=list(corpus_rows)),
                ),
                models.FieldCondition(key="build_id", match=models.MatchAny(any=builds)),
            ])
        client = QdrantClient(
            url=self.settings.qdrant_url, timeout=self.settings.qdrant_timeout_seconds,
        )
        try:
            # 3. 顺序完成向量和词法召回；仅幂等向量读取允许超时重试一次。
            vector_hits = self._qdrant_read(partial(
                client.query_points,
                collection_name=self.settings.qdrant_collection,
                query=vector,
                query_filter=models.Filter(must=must),
                limit=limit,
                with_payload=True,
            ), "向量检索").points
            try:
                lexical_hits = lexical_search(
                    self.settings, query,
                    list(corpus_rows) if corpus_rows is not None else None, builds, limit,
                )
            except Exception as error:
                LOGGER.warning("Retrieval failed stage=lexical error_type=%s", type(error).__name__)
                raise RetrievalDependencyError(
                    f"词法检索失败（{type(error).__name__}），请检查词法索引服务"
                ) from error
            # 4. 从向量服务回读两路候选，交给后续数据库权限与快照校验。
            ids = {int(hit.id) for hit in vector_hits}
            ids.update(int(hit["_id"]) for hit in lexical_hits)
            points = (
                self._qdrant_read(partial(
                    client.retrieve,
                    collection_name=self.settings.qdrant_collection,
                    ids=sorted(ids),
                    with_payload=True,
                    with_vectors=False,
                ), "向量结果回读")
                if ids
                else []
            )
            return vector_hits, lexical_hits, points
        finally:
            client.close()

    def _qdrant_read(self, operation: Callable[[], Any], stage: str) -> Any:
        """对幂等只读请求的 HTTP 超时重试一次，耗尽后记录脱敏阶段错误。

        Args:
            operation: 当前向量查询或结果回读，不允许传入索引写操作。
            stage: 由调用方固定的中文阶段名称，不包含查询或供应商正文。

        Returns:
            首次或重试成功的向量服务响应。

        Raises:
            RetrievalDependencyError: 请求超时重试耗尽，或发生不可重试错误。
        """
        for attempt in range(2):
            try:
                return operation()
            except Exception as error:
                timeout = _is_qdrant_timeout(error)
                LOGGER.warning(
                    "Retrieval failed stage=%s attempt=%s timeout=%s error_type=%s",
                    stage, attempt + 1, timeout, type(error).__name__,
                )
                if timeout and attempt == 0:
                    continue
                reason = "超时（已重试一次）" if timeout else f"失败（{type(error).__name__}）"
                raise RetrievalDependencyError(f"{stage}{reason}，请稍后重试") from error

    def _empty(self, query: str, processed) -> RetrievalResponse:
        """为授权范围内的空候选构造明确的证据不足响应。

        Args:
            query: 原始检索查询。
            processed: 已完成确定性查询处理的结果。

        Returns:
            不含虚构来源的空证据响应。
        """
        return RetrievalResponse(
            knowledge_base_id=uuid.UUID(int=self.knowledge_base_id),
            knowledge_base_version=self.fixed_version,
            original_query=query,
            detected_language=processed.language,
            rewritten_query=processed.rewritten,
            expanded_queries={"zh": processed.chinese, "en": processed.english},
            candidate_count=0,
            evidence=[],
            citations=[],
            insufficient_evidence=True,
            message="当前知识库没有符合条件且双路索引已就绪的文献，请完成索引构建",
        )


def _verified_candidates(vector_hits, lexical_hits, points, corpus_rows):
    """剔除跨文献、跨版本、正文哈希不符或无法定位的候选后保留两路独立分值。

    Args:
        vector_hits: 向量查询结果。
        lexical_hits: OpenSearch 查询结果。
        points: 从 Qdrant 回取的真实片段。
        corpus_rows: 查询完成后重新核验的数据库白名单。

    Returns:
        两路 RankedHit 列表和真实 SourcePoint 映射。
    """
    # 1. 以回库白名单核对真实向量点的文献、模型、构建、附件和正文哈希
    valid = {}
    sources = {}
    for point in points:
        payload = point.payload or {}
        try:
            row = corpus_rows[int(payload.get("corpus_id"))]
            point_uuid = uuid.UUID(int=int(point.id))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        content = payload.get("text")
        if (
            not isinstance(content, str)
            or payload.get("embedding_model") != embedding_config().model
            or payload.get("embedding_dimension") != embedding_config().dimension
            or payload.get("build_id") != row["index_build_id"]
            or payload.get("attachment_digest") != row["index_attachment_digest"]
            or payload.get("content_hash") != hashlib.sha256(content.encode()).hexdigest()
        ):
            continue
        source = _source_from_hit(SimpleNamespace(id=point.id, payload=payload, score=0.0), row)
        if source is not None:
            valid[str(point.id)] = payload
            sources[point_uuid] = source
    # 2. 对向量与词法候选分别核对同一点的快照字段，去重后保留分路原始分值
    branches = []
    for is_lexical, hits in ((False, vector_hits), (True, lexical_hits)):
        ranked = []
        seen = set()
        for hit in hits:
            identifier = str(hit["_id"] if is_lexical else hit.id)
            payload = hit["_source"] if is_lexical else hit.payload or {}
            current = valid.get(identifier)
            if not current or identifier in seen:
                continue
            if any(
                str(payload.get(key)) != str(current.get(key))
                for key in (
                    "corpus_id",
                    "build_id",
                    "attachment_digest",
                    "content_hash",
                )
            ):
                continue
            seen.add(identifier)
            ranked.append(
                RankedHit(
                    uuid.UUID(int=int(identifier)),
                    float(hit["_score"] if is_lexical else hit.score),
                )
            )
        branches.append(ranked)
    return branches[0], branches[1], sources


def _source_from_hit(hit, corpus_row) -> SourcePoint | None:
    """核对 Qdrant payload 确实属于 Foreign 文献的当前附件。

    Args:
        hit: Qdrant 返回的 point。
        corpus_row: 已通过数据库白名单的文献行。

    Returns:
        通过附件与页码核验的来源，或空值。
    """
    payload = hit.payload or {}
    file_path = payload.get("file_path")
    content = payload.get("text")
    if not isinstance(file_path, str) or not isinstance(content, str) or not content.strip():
        return None
    try:
        attachments = json.loads(corpus_row["attachment"] or "[]")
    except (TypeError, ValueError):
        return None
    if not isinstance(attachments, list) or file_path not in {
        item.get("file_path") for item in attachments if isinstance(item, dict)
    }:
        return None
    blocks = payload.get("bbox_blocks")
    if blocks:
        # Foreign 旧分块器在 flush 后未重设 page_start；逐块 page_idx 才是
        # MinerU 的真实 0-based 页码，不能沿用索引中的汇总起始页。
        if not isinstance(blocks, list) or any(
            not isinstance(block, dict)
            or type(block.get("page_idx")) is not int
            or block["page_idx"] < 0
            for block in blocks
        ):
            return None
        pages = [block["page_idx"] for block in blocks]
        start, end = min(pages) + 1, max(pages) + 1
    else:
        # 非 MinerU 路径已经以 1-based page_num 保存；缺失页码不能猜测。
        start, end = payload.get("page_start"), payload.get("page_end")
        if type(start) is not int or type(end) is not int or start < 1 or end < start:
            return None
    if start < 1:
        return None
    return SourcePoint(
        corpus_id=int(corpus_row["id"]),
        point_id=str(hit.id),
        chunk_key=str(payload.get("chunk_id") or ""),
        file_path=file_path,
        content=content,
        page_start=start,
        page_end=end,
        score=float(hit.score),
    )


def _metadata_matches(row, filters: RetrievalFilters) -> bool:
    """把额外条件限制在 Foreign 文献元数据内，不扩大授权集合。

    Args:
        row: Foreign 已授权文献行。
        filters: Agent 图提供的结构化过滤器。

    Returns:
        文献满足语言、作者与来源条件时为真。
    """
    if filters.language_codes and _language(str(row["language"] or "")) not in {
        normalize_source_language(value) for value in filters.language_codes
    }:
        return False
    if filters.authors and not any(
        value.casefold() in str(row["author"] or "").casefold() for value in filters.authors
    ):
        return False
    return not filters.journals or any(
        value.casefold() in str(row["source"] or "").casefold() for value in filters.journals
    )


def _language(value: str) -> str:
    """把 Foreign 文献语言规范到图使用的简短代码。

    Args:
        value: Foreign 原始语言字段。

    Returns:
        统一语言代码，例如 `zh`、`en`、`ja`；未知语言保留规范化后的名称。
    """
    return normalize_source_language(value)


def _year(value: object) -> int | None:
    """安全解析 Foreign YEAR 列，缺失时保持为空。

    Args:
        value: 数据库驱动返回的年份。

    Returns:
        合法四位年份或空值。
    """
    try:
        year = int(value)
    except (TypeError, ValueError):
        return None
    return year if 1000 <= year <= 2100 else None
