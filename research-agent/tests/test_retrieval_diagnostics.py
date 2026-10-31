"""验证研究检索的超时恢复、分阶段脱敏错误及分侧检索步骤的真实证据统计。"""

import uuid
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from httpx import ReadTimeout
from qdrant_client.http.exceptions import ResponseHandlingException

from app.common.bailian_client import BailianError
from app.foreign_integration.config import ForeignSettings, get_foreign_settings
from app.foreign_integration.retrieval import ForeignRetrievalTool, RetrievalDependencyError
from app.foreign_integration.service import _step_output_summary


@pytest.fixture
def retrieval():
    """构造不自行访问网络和数据库的检索对象。"""
    settings = ForeignSettings(
        database_url="sqlite://", jwt_secret="secret", qdrant_url="http://qdrant",
        qdrant_collection="chunks", embedding_api_key="test-key", model_name="model",
        model_base_url="http://model", public_api_base="/api", min_vector_score=0.35,
    )
    return ForeignRetrievalTool(actor_id=7, knowledge_base_id=9, fixed_scope=[11],
                                fixed_version=1, settings=settings)


@pytest.mark.parametrize("operation_name", ["query_points", "retrieve"])
def test_vector_reads_retry_timeout_once_and_use_explicit_timeout(retrieval, operation_name):
    """查询和回读遇到一次超时后能恢复，不将短暂超时直接变成任务失败。"""
    with (
        patch("app.foreign_integration.retrieval.embed_texts", return_value=[[0.1] * 1024]),
        patch("app.foreign_integration.retrieval.QdrantClient") as constructor,
        patch("app.foreign_integration.retrieval.lexical_search", return_value=[]),
    ):
        client = constructor.return_value
        response = SimpleNamespace(points=[SimpleNamespace(id=123)])
        client.query_points.return_value = response
        client.retrieve.return_value = []
        target = getattr(client, operation_name)
        success = response if operation_name == "query_points" else []
        target.side_effect = [ResponseHandlingException(ReadTimeout("private-error")), success]
        vector, lexical, points = retrieval._recall("learning", None, "FAST")
    assert len(vector) == 1 and not lexical and not points
    assert target.call_count == 2
    constructor.assert_called_once_with(url="http://qdrant", timeout=30)
    client.close.assert_called_once()


def test_timeout_exhaustion_keeps_stage_and_hides_transport_details(retrieval, caplog):
    """连续超时只尝试两次，步骤文案和日志都不包含底层请求细节。"""
    operation = Mock(side_effect=ResponseHandlingException(ReadTimeout("private-error")))
    with pytest.raises(RetrievalDependencyError, match="向量检索超时（已重试一次）") as error:
        retrieval._qdrant_read(operation, "向量检索")
    assert operation.call_count == 2
    assert "private-error" not in str(error.value)
    assert "private-error" not in caplog.text


def test_non_timeout_failure_is_not_retried(retrieval):
    """无效请求等非超时错误不通过重复请求增加生产负担。"""
    operation = Mock(side_effect=ValueError("private-error"))
    with pytest.raises(RetrievalDependencyError, match="向量结果回读失败（ValueError）"):
        retrieval._qdrant_read(operation, "向量结果回读")
    operation.assert_called_once()


@pytest.mark.parametrize("stage", ["embedding", "lexical"])
def test_dependency_error_identifies_failing_service_without_vendor_body(retrieval, stage):
    """向量编码与词法失败使用不同阶段提示，且不透出供应商错误正文。"""
    with (
        patch(
            "app.foreign_integration.retrieval.embed_texts", return_value=[[0.1] * 1024],
        ) as embed,
        patch("app.foreign_integration.retrieval.QdrantClient") as constructor,
        patch("app.foreign_integration.retrieval.lexical_search", return_value=[]) as lexical,
    ):
        constructor.return_value.query_points.return_value.points = []
        if stage == "embedding":
            embed.side_effect = BailianError("private-error")
        else:
            lexical.side_effect = RuntimeError("private-error")
        with pytest.raises(RetrievalDependencyError) as error:
            retrieval._recall("learning", None, "FAST")
    assert ("查询向量编码失败" if stage == "embedding" else "词法检索失败") in str(error.value)
    assert "private-error" not in str(error.value)
    if stage == "lexical":
        constructor.return_value.close.assert_called_once()


def test_branch_step_counts_nested_deduplicated_evidence_and_candidates():
    """同一 Chunk 在多维度重复命中只统计一次，不误报为零。"""
    shared = SimpleNamespace(chunk_id=uuid.uuid4())
    other = SimpleNamespace(chunk_id=uuid.uuid4())
    summary = _step_output_summary({"object_b_retrieval": [
        {"evidence": [shared], "candidate_count": 10},
        {"evidence": [shared, other], "candidate_count": 12},
    ]}, {"evidence": []})
    assert summary["evidence_count"] == 2
    assert summary["query_count"] == 2
    assert summary["candidate_count"] == 22


def test_unchanged_evidence_survives_step_summary_but_explicit_empty_resets_count():
    """没有新增 Evidence 的分析节点仍显示已有数量，显式清空时正确显示零。"""
    state = {"evidence": [object(), object()]}
    assert _step_output_summary({"evidence_balance": {}}, state)["evidence_count"] == 2
    assert _step_output_summary({"evidence": []}, state)["evidence_count"] == 0
    merged = _step_output_summary({"merged_candidates": [object()], "retrieved_candidates": 12}, {})
    assert merged["evidence_count"] == 1 and merged["candidate_count"] == 12


def test_vector_timeout_is_configurable_with_bounded_default(monkeypatch):
    """研究向量请求有独立配置，避免依赖 SDK 隐式短超时或无限等待。"""
    for key, value in {
        "FOREIGN_DATABASE_URL": "sqlite://", "FOREIGN_SECRET_KEY": "test",
        "DASHSCOPE_API_KEY": "test", "FOREIGN_QDRANT_URL": "http://qdrant",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("FOREIGN_QDRANT_TIMEOUT_SECONDS", raising=False)
    assert get_foreign_settings().qdrant_timeout_seconds == 30
    monkeypatch.setenv("FOREIGN_QDRANT_TIMEOUT_SECONDS", "60")
    assert get_foreign_settings().qdrant_timeout_seconds == 60
    monkeypatch.setenv("FOREIGN_QDRANT_TIMEOUT_SECONDS", "300")
    assert get_foreign_settings().qdrant_timeout_seconds == 120
