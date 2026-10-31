"""验证 OpenAI-compatible Model Gateway 的最终内容与空响应重试约束。"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, SecretStr, ValidationError

from app.agents.gateway import (
    OpenAICompatibleModelGateway,
    StructuredSchemaValidationError,
    _has_valid_citations,
    _normalize_citation_markers,
    _schema_validation_issues,
)
from app.agents.zhijian_schemas import EvidenceAssignmentBatch, ParsedCompareRequest
from app.retrieval.schemas import EvidenceRead


def test_openai_compatible_gateway_retries_empty_content(monkeypatch) -> None:
    """确认网关不发布 reasoning_content，并对空最终回答只重试一次。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """第一次返回仅推理内容，第二次返回带 Citation 的最终答案。

        Args:
            request: HTTPX 捕获的兼容 API 请求。

        Returns:
            当前调用次数对应的合成响应。
        """
        nonlocal request_count
        request_count += 1
        payload = json.loads(request.content)
        assert payload["messages"][0] == {
            "role": "system",
            "content": "测试固定 Prompt",
        }
        if request_count == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "", "reasoning_content": "不得发布的推理"}}
                    ]
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "证据支持该结论。[1]"}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=200,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )
    evidence = EvidenceRead(
        citation_index=1,
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_id="00000000-0000-0000-0000-000000000002",
        document_title="测试文献",
        authors=[],
        publication_year=2024,
        journal=None,
        language_code="zh",
        page_start=1,
        page_end=1,
        section="结果",
        content="证据支持该结论。",
        relevance_score=0.9,
        retrieval_score=0.8,
        vector_rank=1,
        lexical_rank=1,
    )

    async def execute() -> str:
        """执行一次测试网关调用。

        Returns:
            网关发布的最终回答。
        """
        return await gateway.generate_answer(
            query="结论是什么？",
            evidence=[evidence],
            output_language="ZH",
            detected_language="zh",
            system_prompt="测试固定 Prompt",
        )

    answer = asyncio.run(execute())
    assert answer == "证据支持该结论。[1]"
    assert "不得发布的推理" not in answer
    assert request_count == 2


def test_citation_marker_normalization_and_validation() -> None:
    """确认中文复合引用可规范化，而越界引用不能通过网关预检。"""
    normalized = _normalize_citation_markers("结论一【1，2】；结论二［3］。")
    assert normalized == "结论一[1][2]；结论二[3]。"
    assert _has_valid_citations(normalized, 3)
    assert not _has_valid_citations("无引用回答", 3)
    assert not _has_valid_citations("越界回答[4]", 3)


@pytest.mark.parametrize("model_name", ["qwen3.8-flash", "deepseek-test-model", "test-model"])
@pytest.mark.parametrize("default_thinking", [False, True])
def test_streamed_answer_respects_per_call_thinking(
    monkeypatch, model_name: str, default_thinking: bool,
) -> None:
    """验证真实 SDK 请求遵循单次开关，开启后关闭不残留状态且保留全局默认值。

    Args:
        monkeypatch: 替换环境配置，避免读取真实密钥。
        model_name: 千问、DeepSeek 或不接收厂商思考参数的兼容模型。
        default_thinking: 调用未显式指定开关时使用的全局默认值。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(
            openai_api_key=SecretStr("test-only-key"),
            model_thinking_enabled=default_thinking,
        ),
    )
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """捕获最终请求体并返回合法的 SSE 正文，全程不访问外网。"""
        requests.append(json.loads(request.content))
        chunk = {
            "id": "test-stream", "object": "chat.completion.chunk", "created": 0,
            "model": model_name,
            "choices": [{"index": 0, "delta": {"content": "结论。[1]"},
                         "finish_reason": "stop"}],
        }
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
        )

    evidence = EvidenceRead(
        citation_index=1,
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_id="00000000-0000-0000-0000-000000000002",
        document_title="测试文献", authors=[], publication_year=2024, journal=None,
        language_code="zh", page_start=1, page_end=1, section=None,
        content="证据支持结论。", relevance_score=0.9, retrieval_score=0.8,
        vector_rank=1, lexical_rank=1,
    )
    gateway = OpenAICompatibleModelGateway(
        model_name=model_name, api_base="https://model.example.test",
        timeout_seconds=10, max_tokens=200, temperature=0,
    )

    async def execute() -> None:
        """复用同一网关先开启、再关闭、最后使用默认值，检验参数隔离。"""
        async with AsyncOpenAI(
            api_key="test-only-key", base_url=gateway.api_base,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            max_retries=0,
        ) as client:
            monkeypatch.setattr(gateway, "_openai_async_client", lambda: client)
            for thinking in (True, False, None):
                answer = await gateway.generate_answer(
                    query="结论是什么？", evidence=[evidence], output_language="ZH",
                    detected_language="zh", system_prompt="测试固定 Prompt",
                    thinking=thinking,
                )
                assert answer == "结论。[1]"

    asyncio.run(execute())
    assert len(requests) == 3
    for payload, enabled in zip(requests, (True, False, default_thinking), strict=True):
        assert payload["stream"] is True
        if model_name.startswith("qwen"):
            assert payload["enable_thinking"] is enabled
            assert "thinking" not in payload
        elif model_name.startswith("deepseek"):
            assert payload["thinking"] == {"type": "enabled" if enabled else "disabled"}
            assert "enable_thinking" not in payload
        else:
            assert "enable_thinking" not in payload
            assert "thinking" not in payload


def test_generate_answer_can_disable_thinking_per_call(monkeypatch) -> None:
    """确认普通回答可仅为指定调用关闭 DeepSeek 思考模式。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """验证请求包含关闭思考参数并返回合法引用正文。

        Args:
            request: HTTPX 捕获的兼容 API 请求。

        Returns:
            包含一个真实测试引用的合成响应。
        """
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "disabled"}
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "结论。[1]"}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="deepseek-test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=200,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )
    evidence = EvidenceRead(
        citation_index=1,
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_id="00000000-0000-0000-0000-000000000002",
        document_title="测试文献",
        authors=[],
        publication_year=2024,
        journal=None,
        language_code="zh",
        page_start=1,
        page_end=1,
        section="结果",
        content="证据支持结论。",
        relevance_score=0.9,
        retrieval_score=0.8,
        vector_rank=1,
        lexical_rank=1,
    )

    async def execute() -> str:
        """执行显式关闭思考模式的普通回答请求。

        Returns:
            通过引用校验的测试正文。
        """
        return await gateway.generate_answer(
            query="结论是什么？",
            evidence=[evidence],
            output_language="ZH",
            detected_language="zh",
            system_prompt="测试固定 Prompt",
            thinking=False,
        )

    assert asyncio.run(execute()) == "结论。[1]"


def test_qwen_generate_answer_uses_enable_thinking(monkeypatch) -> None:
    """确认千问兼容接口使用其要求的顶层思考模式参数。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """验证千问请求参数并返回合法引用正文。

        Args:
            request: HTTPX 捕获的兼容 API 请求。

        Returns:
            包含一个真实测试引用的合成响应。
        """
        payload = json.loads(request.content)
        assert payload["enable_thinking"] is False
        assert "thinking" not in payload
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": "结论。[1]"}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="qwen3.8-flash",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=200,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )
    evidence = EvidenceRead(
        citation_index=1,
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_id="00000000-0000-0000-0000-000000000002",
        document_title="测试文献",
        authors=[],
        publication_year=2024,
        journal=None,
        language_code="zh",
        page_start=1,
        page_end=1,
        section="结果",
        content="证据支持结论。",
        relevance_score=0.9,
        retrieval_score=0.8,
        vector_rank=1,
        lexical_rank=1,
    )

    async def execute() -> str:
        """执行显式关闭千问思考模式的普通回答请求。

        Returns:
            通过引用校验的测试正文。
        """
        return await gateway.generate_answer(
            query="结论是什么？",
            evidence=[evidence],
            output_language="ZH",
            detected_language="zh",
            system_prompt="测试固定 Prompt",
            thinking=False,
        )

    assert asyncio.run(execute()) == "结论。[1]"


def test_qwen_structured_output_supports_thinking_override(monkeypatch) -> None:
    """确认千问结构化调用默认关闭思考且允许单次开启。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(
            openai_api_key=SecretStr("test-only-key"),
            model_thinking_enabled=False,
        ),
    )
    expected_modes = iter((False, True))

    def handler(request: httpx.Request) -> httpx.Response:
        """验证结构化请求关闭思考并返回合法比较请求。

        Args:
            request: HTTPX 捕获的兼容 API 请求。

        Returns:
            满足智鉴解析 Schema 的合成响应。
        """
        payload = json.loads(request.content)
        assert payload["enable_thinking"] is next(expected_modes)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert "max_tokens" not in payload
        response_schema = payload["response_format"]["json_schema"]["schema"]
        assert response_schema["additionalProperties"] is False
        assert set(response_schema["required"]) == set(response_schema["properties"])
        content = json.dumps(
            {
                "topic": "人工智能教育政策",
                "object_a": "中国",
                "object_b": "美国",
                "year_from": None,
                "year_to": None,
                "language_filter": "ALL",
                "source_language_requirement": None,
                "output_language": "ZH",
                "requested_dimensions": [],
            },
            ensure_ascii=False,
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="qwen3.8-flash",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=500,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )

    async def execute() -> tuple[ParsedCompareRequest, ParsedCompareRequest]:
        """依次执行默认关闭和显式开启思考的结构化请求。

        Returns:
            两次均经 Pydantic 校验的比较请求。
        """
        default_result = await gateway.structured(
            instruction="解析中国和美国的人工智能教育政策比较。",
            schema=ParsedCompareRequest,
            system_prompt="固定智鉴 Prompt",
        )
        override_result = await gateway.structured(
            instruction="解析中国和美国的人工智能教育政策比较。",
            schema=ParsedCompareRequest,
            system_prompt="固定智鉴 Prompt",
            thinking=True,
        )
        return default_result, override_result

    default_result, override_result = asyncio.run(execute())
    assert default_result.object_a == "中国"
    assert override_result.object_b == "美国"


def test_openai_compatible_gateway_retries_truncated_answer(monkeypatch) -> None:
    """确认 ``finish_reason=length`` 的带引用正文也不会被发布。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    responses = iter(
        (
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "未完成的句子[1]"},
                    }
                ]
            },
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "完整结论。[1]"},
                    }
                ]
            },
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        """依次返回截断响应和正常结束响应。

        Args:
            request: HTTPX 捕获的兼容 API 请求。

        Returns:
            当前迭代位置对应的合成响应。
        """
        return httpx.Response(200, json=next(responses), request=request)

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=200,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )
    evidence = EvidenceRead(
        citation_index=1,
        document_id="00000000-0000-0000-0000-000000000001",
        chunk_id="00000000-0000-0000-0000-000000000002",
        document_title="测试文献",
        authors=[],
        publication_year=2024,
        journal=None,
        language_code="zh",
        page_start=1,
        page_end=1,
        section="结果",
        content="证据支持完整结论。",
        relevance_score=0.9,
        retrieval_score=0.8,
        vector_rank=1,
        lexical_rank=1,
    )

    async def execute() -> str:
        """执行包含截断重试的一次网关调用。

        Returns:
            仅允许正常结束的最终回答。
        """
        return await gateway.generate_answer(
            query="结论是什么？",
            evidence=[evidence],
            output_language="ZH",
            detected_language="zh",
            system_prompt="测试固定 Prompt",
        )

    assert asyncio.run(execute()) == "完整结论。[1]"


def test_openai_compatible_gateway_validates_structured_output(monkeypatch) -> None:
    """确认结构化调用经 JSON 模式返回并在格式错误后执行一次纠正。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """先返回非法结构，再返回满足智鉴解析 Schema 的 JSON。

        Args:
            request: HTTPX 捕获的兼容 API 请求。

        Returns:
            与当前请求次数对应的模型响应。
        """
        nonlocal request_count
        request_count += 1
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        if request_count == 1:
            content = '{"topic":'
        else:
            assert "上一次结构化结果无效" in payload["messages"][-1]["content"]
            content = json.dumps(
                {
                    "topic": "人工智能教育政策",
                    "object_a": "中国",
                    "object_b": "日本",
                    "year_from": 2020,
                    "year_to": 2025,
                    "language_filter": "ALL",
                    "source_language_requirement": None,
                    "output_language": "ZH",
                    "requested_dimensions": [],
                },
                ensure_ascii=False,
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=500,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )

    async def execute() -> ParsedCompareRequest:
        """执行一次智鉴自然语言结构化解析。

        Returns:
            经 Pydantic 最终校验的比较请求。
        """
        return await gateway.structured(
            instruction="解析中国和日本的人工智能教育政策比较。",
            schema=ParsedCompareRequest,
            system_prompt="固定智鉴 Prompt",
        )

    parsed = asyncio.run(execute())
    assert parsed.object_a == "中国"
    assert parsed.object_b == "日本"
    assert request_count == 2


def test_structured_gateway_keeps_safe_field_diagnostics_after_two_failures(monkeypatch) -> None:
    """两轮结构错误均保留字段诊断，但不泄露模型输出中的值。

    Args:
        monkeypatch: 隔离测试密钥配置。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """先返回年份越界，再返回包含敏感输入值的非法枚举。

        Args:
            request: 当前结构化模型请求。

        Returns:
            用于触发字段级 Pydantic 错误的兼容响应。
        """
        requests.append(json.loads(request.content))
        content = (
            {"year_from": 999, "output_language": "ZH", "topic": "SENSITIVE_MARKER"}
            if len(requests) == 1
            else {"year_from": 2024, "output_language": "SENSITIVE_MARKER"}
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=500,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )

    async def execute() -> None:
        """执行预期连续两次失败的结构化解析。

        Raises:
            StructuredSchemaValidationError: 保存两轮安全诊断。
        """
        await gateway.structured(
            instruction="解析比较请求",
            schema=ParsedCompareRequest,
            system_prompt="固定智鉴 Prompt",
        )

    with pytest.raises(StructuredSchemaValidationError) as raised:
        asyncio.run(execute())
    diagnostics = raised.value.diagnostics
    assert diagnostics["schema"] == "ParsedCompareRequest"
    assert diagnostics["attempts"][0]["issues"][0]["path"] == "$.year_from"
    assert diagnostics["attempts"][1]["issues"][0]["path"] == "$.output_language"
    assert diagnostics["attempts"][1]["issues"][0]["type"] == "literal_error"
    assert "SENSITIVE_MARKER" not in str(raised.value)
    assert "SENSITIVE_MARKER" not in json.dumps(diagnostics)
    assert "SENSITIVE_MARKER" not in requests[1]["messages"][-1]["content"]


def test_assignment_schema_retry_explains_cross_field_rule_without_echoing_values(
    monkeypatch,
) -> None:
    """归属结构化纠错指明 supports 与 relevance 关系且不复述模型原文。

    Args:
        monkeypatch: 隔离模型密钥配置。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    requests: list[dict[str, object]] = []
    chunk_id = "218a8ee4-8785-4569-90ab-832840070c01"

    def handler(request: httpx.Request) -> httpx.Response:
        """先返回相关但无支持项的非法归属，再按规则纠正。

        Args:
            request: 捕获的兼容 API 请求。

        Returns:
            当前尝试对应的 JSON 响应。
        """
        requests.append(json.loads(request.content))
        content = {
            "decisions": [{
                "chunk_id": chunk_id,
                "supports": [],
                "relevance": "LOW" if len(requests) == 1 else "IRRELEVANT",
                "assignment_reason": "SENSITIVE_MARKER",
            }]
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(content)}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=500,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )

    async def execute() -> EvidenceAssignmentBatch:
        """执行单批归属并获取本地校验后的结果。

        Returns:
            满足 supports 与 relevance 联动约束的决策。
        """
        return await gateway.structured(
            instruction="判断候选归属",
            schema=EvidenceAssignmentBatch,
            system_prompt="固定智鉴 Prompt",
        )

    result = asyncio.run(execute())
    assert result.decisions[0].relevance == "IRRELEVANT"
    assert len(requests) == 2
    correction = requests[1]["messages"][-1]["content"]
    assert "assignment_relevant_without_supports" in correction
    assert "supports 为空时 relevance 必须为 IRRELEVANT" in correction
    assert "SENSITIVE_MARKER" not in correction


def test_schema_diagnostics_hide_dynamic_model_keys() -> None:
    """动态字典键可能含用户文本，审计路径只保留 Schema 声明字段。"""

    class DynamicResult(BaseModel):
        """用动态映射模拟模型自行生成的字段键。"""

        # 映射键由模型生成，用于验证审计路径不会保存动态键的实际文本。
        payload: dict[str, int]

    with pytest.raises(ValidationError) as raised:
        DynamicResult.model_validate({"payload": {"SENSITIVE_MARKER": "invalid"}})
    error_count, issues = _schema_validation_issues(raised.value, DynamicResult)
    assert error_count == 1
    assert issues[0]["path"] == "$.payload.*"
    assert "SENSITIVE_MARKER" not in json.dumps(issues)


def test_structured_gateway_reports_timeout_separately(monkeypatch) -> None:
    """确认结构化模型读取超时不会被误报为普通连接失败。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.agents.gateway.asyncio.sleep", sleep)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """模拟兼容模型超过配置读取窗口。

        Args:
            request: HTTPX 捕获的兼容 API 请求。

        Raises:
            httpx.ReadTimeout: 模拟供应商未在窗口内返回正文。
        """
        nonlocal request_count
        request_count += 1
        raise httpx.ReadTimeout("test timeout", request=request)

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=500,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )

    async def execute() -> ParsedCompareRequest:
        """发起预期超时的结构化请求。

        Returns:
            正常情况下返回解析结果，本测试路径不会产生返回值。
        """
        return await gateway.structured(
            instruction="解析中国和美国的人工智能教育政策比较。",
            schema=ParsedCompareRequest,
            system_prompt="固定智鉴 Prompt",
        )

    with pytest.raises(RuntimeError, match="模型服务请求超时"):
        asyncio.run(execute())
    assert request_count == 3
    assert sleep.await_count == 2


def test_structured_gateway_retries_transient_transport_error(monkeypatch) -> None:
    """结构化调用应在瞬时协议错误后退避重试并复用同一业务请求。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.agents.gateway.asyncio.sleep", sleep)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """前两次断开协议连接，第三次返回合法结构化结果。

        Args:
            request: HTTPX 捕获的兼容 API 请求。

        Returns:
            第三次调用的成功响应。

        Raises:
            httpx.RemoteProtocolError: 前两次模拟供应商中断连接。
        """
        nonlocal request_count
        request_count += 1
        if request_count < 3:
            raise httpx.RemoteProtocolError("server disconnected", request=request)
        content = json.dumps(
            {
                "topic": "人工智能教育政策",
                "object_a": "中国",
                "object_b": "美国",
                "year_from": None,
                "year_to": None,
                "language_filter": "ALL",
                "source_language_requirement": None,
                "output_language": "ZH",
                "requested_dimensions": [],
            },
            ensure_ascii=False,
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=500,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )

    async def execute() -> ParsedCompareRequest:
        """执行预期在第三次传输成功的结构化请求。"""
        return await gateway.structured(
            instruction="解析中国和美国的人工智能教育政策比较。",
            schema=ParsedCompareRequest,
            system_prompt="固定智鉴 Prompt",
        )

    parsed = asyncio.run(execute())
    assert parsed.object_b == "美国"
    assert request_count == 3
    assert sleep.await_count == 2


def test_structured_gateway_retries_429_and_honors_retry_after(monkeypatch) -> None:
    """HTTP 429 应遵循 Retry-After 后重试，成功响应只发布一次。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.agents.gateway.asyncio.sleep", sleep)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """先返回限流，再返回合法结构化结果。"""
        nonlocal request_count
        request_count += 1
        if request_count == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
                request=request,
            )
        content = json.dumps(
            {
                "topic": "人工智能教育政策",
                "object_a": "中国",
                "object_b": "美国",
                "year_from": None,
                "year_to": None,
                "language_filter": "ALL",
                "source_language_requirement": None,
                "output_language": "ZH",
                "requested_dimensions": [],
            },
            ensure_ascii=False,
        )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
            request=request,
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=500,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )

    async def execute() -> ParsedCompareRequest:
        """执行一次包含供应商限流退避的结构化请求。"""
        return await gateway.structured(
            instruction="解析中国和美国的人工智能教育政策比较。",
            schema=ParsedCompareRequest,
            system_prompt="固定智鉴 Prompt",
        )

    assert asyncio.run(execute()).object_a == "中国"
    assert request_count == 2
    sleep.assert_awaited_once_with(0.0)


def test_structured_gateway_does_not_retry_deterministic_http_error(monkeypatch) -> None:
    """HTTP 400 等确定性请求错误应立即失败，避免重复消耗模型额度。

    Args:
        monkeypatch: Pytest 属性替换夹具。
    """
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(openai_api_key=SecretStr("test-only-key")),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.agents.gateway.asyncio.sleep", sleep)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """持续返回无需重试的请求参数错误。"""
        nonlocal request_count
        request_count += 1
        return httpx.Response(400, request=request)

    gateway = OpenAICompatibleModelGateway(
        model_name="test-model",
        api_base="https://model.example.test",
        timeout_seconds=10,
        max_tokens=500,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )

    async def execute() -> ParsedCompareRequest:
        """发起预期立即失败的结构化请求。"""
        return await gateway.structured(
            instruction="解析中国和美国的人工智能教育政策比较。",
            schema=ParsedCompareRequest,
            system_prompt="固定智鉴 Prompt",
        )

    with pytest.raises(RuntimeError, match="模型服务返回 HTTP 400"):
        asyncio.run(execute())
    assert request_count == 1
    sleep.assert_not_awaited()
