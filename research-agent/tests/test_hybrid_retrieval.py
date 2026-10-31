"""验证混合召回的独有命中、去重、版本隔离和查询后权限复核。"""

import hashlib
import json
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, insert
from sqlalchemy.orm import Session

from app.foreign_integration.config import ForeignSettings
from app.foreign_integration.database import scope_version
from app.foreign_integration.lexical import lexical_search
from app.foreign_integration.retrieval import (
    ForeignRetrievalTool,
    _metadata_matches,
    _verified_candidates,
)
from app.foreign_integration.schema import (
    foreign_corpus,
    foreign_corpus_index_job,
    foreign_knowledge_base,
    foreign_knowledge_base_document,
    metadata,
)
from app.retrieval.ranking import RankedHit, reciprocal_rank_fusion
from app.retrieval.schemas import RetrievalFilters


def example():
    """构造同一已发布快照的题录、point 和 BM25 命中。"""
    attachment = json.dumps([{"file_path": "paper.pdf"}])
    digest = hashlib.sha256(attachment.encode()).hexdigest()
    row = dict(
        id=11,
        attachment=attachment,
        title_cn="",
        title_en="Learning evidence",
        author="Author",
        publish_time=2024,
        source="Journal",
        language="en",
        index_build_id="build-1",
        index_attachment_digest=digest,
    )
    payload = dict(
        corpus_id=11,
        file_path="paper.pdf",
        text="learning evidence",
        page_start=1,
        page_end=2,
        build_id="build-1",
        attachment_digest=digest,
        content_hash=hashlib.sha256(b"learning evidence").hexdigest(),
        embedding_model="qwen3.7-text-embedding-flash",
        embedding_dimension=1024,
    )
    point = SimpleNamespace(id=11000000, payload=payload, score=0.8)
    lexical = {"_id": "11000000", "_score": 12.0, "_source": dict(payload)}
    return {11: row}, point, lexical


def tool():
    """创建不会自行联网的测试检索器。"""
    settings = ForeignSettings(
        "sqlite://",
        "secret",
        "http://qdrant",
        "corpus_chunks",
        "key",
        "model",
        "http://model",
        "/api",
        0.35,
    )
    return ForeignRetrievalTool(
        actor_id=7,
        knowledge_base_id=9,
        fixed_scope=[11],
        fixed_version=scope_version([11]),
        settings=settings,
    )


def test_bm25_only_produces_real_citation_and_persistable_source():
    """词法独有结果无需向量分，引用仍指向真实 Qdrant point 和页码。"""
    rows, point, lexical = example()
    retrieval = tool()
    with (
        patch.object(retrieval, "_load_scope", return_value=rows),
        patch.object(retrieval, "_recall", return_value=([], [lexical], [point])),
    ):
        result = retrieval._search_sync("learning evidence", "FAST", RetrievalFilters(), None)
    assert len(result.evidence) == 1
    assert result.evidence[0].vector_rank is None
    assert result.evidence[0].lexical_rank == 1
    assert result.evidence[0].retrieval_score == pytest.approx(1 / 61)
    assert result.citations[0].page_start == 1
    assert result.citations[0].file_url.endswith("#page=1")
    assert retrieval.sources[result.evidence[0].chunk_id].point_id == "11000000"


def test_two_branches_merge_and_low_vector_can_have_lexical_support():
    """两路同片段只返回一次，低向量分不会覆盖有效的词法支持。"""
    rows, point, lexical = example()
    point.score = 0.1
    retrieval = tool()
    with (
        patch.object(retrieval, "_load_scope", return_value=rows),
        patch.object(retrieval, "_recall", return_value=([point], [lexical], [point])),
    ):
        result = retrieval._search_sync("learning evidence", "DEEP", RetrievalFilters(), None)
    assert result.candidate_count == 1
    assert result.evidence[0].retrieval_score == pytest.approx(2 / 61)
    assert result.evidence[0].vector_rank == result.evidence[0].lexical_rank == 1


def test_rrf_uses_rank_not_bm25_magnitude():
    """BM25 原始分再大也不会直接支配融合分。"""
    one, two = uuid.UUID(int=1), uuid.UUID(int=2)
    hits = reciprocal_rank_fusion([RankedHit(one, 0.9)], [RankedHit(two, 1000000)], k=60)
    assert hits[0].rrf_score == hits[1].rrf_score


def test_lexical_query_is_scoped_and_partial_results_are_errors():
    """词法请求同时下推权限和版本，超时或分片失败不能返回不完整证据。"""
    with patch("app.foreign_integration.lexical.OpenSearch") as constructor:
        client = constructor.return_value
        client.search.return_value = {"hits": {"hits": []}}
        assert lexical_search(tool().settings, "teacher feedback", [11], ["build-1"], 30) == []
        body = client.search.call_args.kwargs["body"]
        assert body["query"]["bool"]["filter"] == [
            {"terms": {"corpus_id": [11]}},
            {"terms": {"build_id": ["build-1"]}},
        ]
        client.search.return_value = {"timed_out": True}
        with pytest.raises(RuntimeError):
            lexical_search(tool().settings, "teacher feedback", [11], ["build-1"], 30)
        client.search.return_value = {"_shards": {"failed": 1}}
        with pytest.raises(RuntimeError):
            lexical_search(tool().settings, "teacher feedback", [11], ["build-1"], 30)


@pytest.mark.parametrize(
    "field,value",
    [
        ("build_id", "old"),
        ("attachment_digest", "old"),
        ("content_hash", "wrong"),
        ("corpus_id", 999),
        ("file_path", "other.pdf"),
    ],
)
def test_invalid_canonical_source_never_becomes_evidence(field, value):
    """外部索引不能绕过版本、正文哈希、文献归属和原附件校验。"""
    rows, point, lexical = example()
    point.payload[field] = value
    vector, words, sources = _verified_candidates([point], [lexical], [point], rows)
    assert not vector and not words and not sources


def test_old_embedding_is_rejected_even_when_lexical_branch_matches():
    """验证词法命中也不能绕过旧向量模型过滤。"""
    rows, point, lexical = example()
    point.payload["embedding_model"] = "text-embedding-v4"
    assert _verified_candidates([point], [lexical], [point], rows) == ([], [], {})


def test_recall_uses_shared_query_embedding_and_current_model_filter():
    """验证召回复用 query 角色的向量接口，并过滤为当前模型和维度。"""
    rows, _, _ = example()
    retrieval = tool()
    with patch("app.foreign_integration.retrieval.embed_texts",
               return_value=[[0.1] * 1024]) as embed, \
         patch("app.foreign_integration.retrieval.QdrantClient") as constructor, \
         patch("app.foreign_integration.retrieval.lexical_search", return_value=[]):
        constructor.return_value.query_points.return_value.points = []
        retrieval._recall("learning evidence", rows, "FAST")
    embed.assert_called_once_with(["learning evidence"], text_type="query", key="key")
    must = constructor.return_value.query_points.call_args.kwargs["query_filter"].must
    values = {condition.key: condition.match for condition in must}
    assert values["embedding_model"].value == "qwen3.7-text-embedding-flash"
    assert values["embedding_dimension"].value == 1024


def test_old_lexical_hit_does_not_add_score_to_new_vector_point():
    """同 ID 的历史词法文档不得为新向量片段贡献 RRF 分。"""
    rows, point, lexical = example()
    lexical["_source"]["build_id"] = "old"
    vector, words, _ = _verified_candidates([point], [lexical], [point], rows)
    assert len(vector) == 1 and not words


def test_rank_one_without_absolute_support_is_rejected():
    """排名第一不等于证据充分，低相似度和无词项覆盖仍须拒绝。"""
    rows, point, lexical = example()
    point.score = 0.1
    retrieval = tool()
    with (
        patch.object(retrieval, "_load_scope", return_value=rows),
        patch.object(retrieval, "_recall", return_value=([point], [lexical], [point])),
    ):
        result = retrieval._search_sync("unrelated topic", "FAST", RetrievalFilters(), None)
    assert result.insufficient_evidence and not result.citations


def test_invalidated_candidates_are_skipped_but_backend_failure_is_an_error():
    """候选变为不可用时排除该证据；基础设施故障仍显式失败。"""
    rows, point, lexical = example()
    retrieval = tool()
    with (
        patch.object(retrieval, "_load_scope", side_effect=[rows, {}]),
        patch.object(retrieval, "_recall", return_value=([point], [lexical], [point])),
    ):
        result = retrieval._search_sync("learning", "FAST", RetrievalFilters(), None)
        assert result.insufficient_evidence and not result.citations
    with (
        patch.object(retrieval, "_load_scope", return_value=rows),
        patch.object(retrieval, "_recall", side_effect=RuntimeError("private-address")),
        pytest.raises(RuntimeError, match="检索索引暂不可用") as error,
    ):
        retrieval._search_sync("learning", "FAST", RetrievalFilters(), None)
    assert "private-address" not in str(error.value)


def test_database_requires_both_success_and_current_attachment():
    """只有任务表两路成功且附件匹配才进入白名单；不新增第二张状态表。"""
    rows, _, _ = example()
    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    with Session(engine) as session:
        session.execute(
            insert(foreign_corpus).values(
                id=11,
                title_en="Learning",
                attachment=rows[11]["attachment"],
                status=2,
                is_deleted=False,
                is_vectorized=True,
            )
        )
        session.execute(insert(foreign_knowledge_base).values(id=9, owner_id=7, is_deleted=False))
        session.execute(
            insert(foreign_knowledge_base_document).values(
                knowledge_base_id=9,
                corpus_id=11,
                is_deleted=False,
            )
        )
        session.execute(
            insert(foreign_corpus_index_job).values(
                corpus_id=11,
                status="SUCCESS",
                vector_status="SUCCESS",
                lexical_status="PENDING",
                attachment_digest=rows[11]["index_attachment_digest"],
                build_id="build-1",
                lexical_index="foreign_chunks_v1",
                is_deleted=False,
            )
        )
        session.commit()
    retrieval = tool()
    with patch(
        "app.foreign_integration.retrieval.session_factory", return_value=lambda: Session(engine)
    ):
        assert not retrieval._load_scope(RetrievalFilters())
        with engine.begin() as connection:
            connection.execute(foreign_corpus_index_job.update().values(lexical_status="SUCCESS"))
        assert list(retrieval._load_scope(RetrievalFilters())) == [11]
        with engine.begin() as connection:
            connection.execute(
                foreign_corpus.update().values(attachment='[{"file_path":"new.pdf"}]')
            )
        assert not retrieval._load_scope(RetrievalFilters())
    engine.dispose()


@pytest.mark.parametrize(
    ("language", "allowed", "matches"),
    [
        ("日语", ["JA"], True),
        ("jp", ["ja-JP"], True),
        ("朝鲜语", ["ko"], True),
        ("法语", ["EN", "FR"], True),
        ("ru-RU", ["俄语"], True),
        ("西班牙语", ["es"], True),
        ("de-DE", ["德语"], True),
        ("it-IT", ["it"], True),
        ("日语", ["en", "fr"], False),
        ("中文", ["ja"], False),
        ("", ["ja"], False),
        ("日语", [], True),
    ],
)
def test_language_filter_matches_real_metadata_aliases(language, allowed, matches):
    """文献名称及历史地区代码能匹配所选语种，不相关语言和未知语种不能混入。"""
    row = {"language": language, "author": "Author", "source": "Journal"}
    assert _metadata_matches(row, RetrievalFilters(language_codes=allowed)) is matches


def test_foreign_language_evidence_keeps_normalized_source_and_original_quote():
    """外语检索结果发布真实页码和原文引用，语言代码使用统一归一结果。"""
    rows, point, _ = example()
    rows[11]["language"] = "日语"
    point.payload["text"] = "日本の教育政策は教員研修と学習支援を重視する。"
    point.payload["content_hash"] = hashlib.sha256(
        point.payload["text"].encode(),
    ).hexdigest()
    retrieval = tool()
    with (
        patch.object(retrieval, "_load_scope", return_value=rows),
        patch.object(retrieval, "_recall", return_value=([point], [], [point])),
    ):
        result = retrieval._search_sync(
            "learning evidence", "FAST", RetrievalFilters(language_codes=["ja"]), None,
        )
    assert result.evidence[0].language_code == "ja"
    assert result.citations[0].quote == point.payload["text"]
    assert result.citations[0].page_start == 1


def test_missing_requested_language_does_not_fall_back_to_other_sources():
    """指定语种无就绪文献时返回证据不足，不取消过滤或召回其他语种。"""
    retrieval = tool()
    with (
        patch.object(retrieval, "_load_scope", return_value={}),
        patch.object(retrieval, "_recall") as recall,
    ):
        result = retrieval._search_sync(
            "learning evidence", "FAST", RetrievalFilters(language_codes=["ja"]), None,
        )
    assert result.insufficient_evidence and not result.evidence and not result.citations
    recall.assert_not_called()
