"""联网推荐必须有搜索来源，并排除全库及名单内重复，不要求 DOI。"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import test_literature_recommendations as fixtures
from pydantic import SecretStr
from sqlalchemy import update

from app.agents.gateway import OpenAICompatibleModelGateway
from app.agents.literature_graph import LiteratureGraph
from app.agents.web_literature import (
    extract_doi,
    public_url,
    same_title,
    search_candidates,
)
from app.foreign_integration.schema import foreign_corpus, foreign_user


def response_for(papers, urls):
    """构造供应商真实工具来源与独立助手文本，二者不能相互替代。"""
    return {
        "status": "completed",
        "output": [
            {
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "sources": [{"url": url} for url in urls],
                },
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "检索完成\n```json\n" + json.dumps({"papers": papers}) + "\n```",
                    }
                ],
            },
        ],
    }


def paper(title="Generative AI in teaching", doi="10.1234/example"):
    """一篇来自联网搜索的候选。"""
    return {
        "title": title,
        "doi": doi,
        "corpus_id": None,
        "origin": "WEB",
        "external_url": "https://doi.org/" + doi,
        "attachment_paths": [],
    }


def test_gateway_uses_frozen_model_and_real_search_sources(monkeypatch):
    """工具只对联网推荐启用；未出现在搜索来源中的模型链接被丢弃。"""
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(
            openai_api_key=SecretStr("test-key"),
        ),
    )
    url = "https://publisher.example/article/123"

    def handler(request):
        """验证模型、搜索次数和有限主题输入，返回有来源及无来源两种候选。"""
        payload = json.loads(request.content)
        assert request.url.path == "/v1/responses"
        assert payload["model"] == "frozen-model"
        assert payload["tools"] == [{"type": "web_search"}]
        assert payload["max_tool_calls"] == 3
        assert request.extensions["timeout"]["read"] == 120
        assert request.extensions["timeout"]["connect"] == 10
        assert "最多推荐5篇" in payload["input"]
        assert payload["stream"] is True
        assert "secret-body" not in payload["input"]
        return httpx.Response(
            200,
            json=response_for(
                [
                    {"title": "Real scholarly paper", "url": url},
                    {"title": "Invented paper", "url": "https://invented.example/paper"},
                ],
                [url],
            ),
        )

    gateway = OpenAICompatibleModelGateway(
        model_name="frozen-model",
        api_base="https://model.example/v1",
        timeout_seconds=60,
        max_tokens=6000,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(gateway.search_literature(topics=["AI education"]))
    assert len(result) == 1
    assert result[0]["title"] == "Real scholarly paper"
    assert result[0]["external_url"] == url
    assert result[0]["doi"] == ""
    assert result[0]["author"] == "" and result[0]["year"] is None


def test_streaming_search_is_bounded_and_accepts_completed_sources(monkeypatch):
    """流式检索完成前等待，完整响应才能返回，挂起时总预算仍能取消请求。"""
    monkeypatch.setattr(
        "app.agents.gateway.get_settings",
        lambda: SimpleNamespace(
            openai_api_key=SecretStr("test-key"),
        ),
    )
    url = "https://publisher.example/article/stream"

    class Stream(httpx.AsyncByteStream):
        """模拟先创建后完成的 SSE 流，区分完整响应和超时取消。"""
        async def __aiter__(self):
            """先发创建事件，延迟后发送包含真实来源的完成事件。

            Yields:
                编码为字节的 SSE 帧，供实际 HTTP 流解析器消费。
            """
            yield b'data: {"type":"response.created"}\n\n'
            await asyncio.sleep(0.03)
            event = {
                "type": "response.completed",
                "response": response_for(
                    [{"title": "A completed streamed paper", "url": url}],
                    [url],
                ),
            }
            yield ("data: " + json.dumps(event) + "\n\n").encode()

    def handler(request):
        """将离线事件流包装成 text/event-stream 响应，不进行真实网络请求。"""
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Stream())

    gateway = OpenAICompatibleModelGateway(
        model_name="test",
        api_base="https://model.example/v1",
        timeout_seconds=60,
        max_tokens=6000,
        temperature=0,
        transport=httpx.MockTransport(handler),
    )
    monkeypatch.setattr("app.agents.gateway.WEB_SEARCH_TIMEOUT_SECONDS", 0.2)
    assert len(asyncio.run(gateway.search_literature(topics=["fixed public topic"]))) == 1
    monkeypatch.setattr("app.agents.gateway.WEB_SEARCH_TIMEOUT_SECONDS", 0.005)
    with pytest.raises(TimeoutError):
        asyncio.run(gateway.search_literature(topics=["fixed public topic"]))


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "http://127.0.0.1/paper",
        "http://192.168.1.2/paper",
        "http://localhost/paper",
        "https://user:password@publisher.example/paper",
    ],
)
def test_unsafe_external_links_are_not_accepted(url):
    """库外链接不可成为脚本或本机/私网地址。"""
    assert public_url(url) == ""


def test_search_without_actual_sources_or_completed_output_is_rejected():
    """非空论文缺少真实来源或响应未完成时拒绝发布；完整空名单直接结束。"""
    assert search_candidates(response_for([], [])) == []
    data = response_for([{"title": "Unverified paper", "url": "https://example.org"}], [])
    with pytest.raises(ValueError, match="来源"):
        search_candidates(data)
    data["status"] = "incomplete"
    with pytest.raises(ValueError, match="完整"):
        search_candidates(data)
    assert same_title("Ｇｅｎｅｒａｔｉｖｅ AI: in teaching!", "Generative AI in teaching")
    assert extract_doi("https://doi.org/10.1000/test(foo)") == "10.1000/test(foo)"


def test_search_keeps_five_papers_without_doi_or_metadata_and_uses_source_urls():
    """真实搜索来源即可进入候选，不再要求登记机构收录；最多五篇。"""
    urls = [f"https://publisher.example/paper/{index}" for index in range(12)]
    papers = [{"title": f"中文教育研究论文{index}", "url": url} for index, url in enumerate(urls)]
    papers[0].update(author="甲，乙", year="2025", source="教育研究")
    papers[1].update(author=None, year=True, source=["invalid"])
    result = search_candidates(response_for(papers, urls))
    assert len(result) == 5
    assert result[0]["author"] == "甲，乙" and result[0]["year"] == 2025
    assert result[1]["author"] == "" and result[1]["source"] == ""
    assert result[1]["year"] is None
    assert all(
        item["doi"] == "" and item["external_url"] == urls[i] for i, item in enumerate(result)
    )


def test_no_doi_candidates_are_distinct_and_capped_after_library_dedupe():
    """空 DOI 不可互相判重，题名和 URL 重复仍排除，过滤后至多五篇。"""
    fixture = fixtures.RecommendationTests()
    fixture.setUp()
    try:
        candidates = [
            {
                **paper(f"独立中文教育课题{index}", ""),
                "external_url": f"https://publisher.example/paper/{index}",
            }
            for index in range(12)
        ]
        # 与首篇同 URL、与第二篇同题名，均不能增加推荐数量。
        duplicates = [
            {**candidates[0], "title": "同一链接的另一个标题"},
            {**candidates[1], "external_url": "https://publisher.example/duplicate"},
        ]
        found = fixture.tool.filter_external(candidates[:2] + duplicates + candidates[2:], [])
        assert len(found) == 5
        assert [item["title"] for item in found] == [item["title"] for item in candidates[:5]]
    finally:
        fixture.tearDown()


def test_external_dedupe_checks_all_titles_languages_and_unindexed_pending_records():
    """排除并未参与当前推荐的库内文献，包含英文别名、未索引/未审核题录及 DOI。"""
    fixture = fixtures.RecommendationTests()
    fixture.setUp()
    try:
        fixture.session.execute(
            update(foreign_corpus)
            .where(foreign_corpus.c.id == 15)
            .values(
                title_en="Generative AI in teaching",
                is_vectorized=False,
            )
        )
        fixture.session.execute(
            update(foreign_corpus)
            .where(foreign_corpus.c.id == 14)
            .values(
                original_source="https://doi.org/10.1234/already-in-library",
            )
        )
        fixture.session.commit()
        found = fixture.tool.filter_external(
            [
                paper("Generative AI in Teaching!"),
                paper("Another registered title", "10.1234/already-in-library"),
                paper("An actually new paper", "10.1234/new"),
                paper("An actually new paper!", "10.1234/duplicate"),
            ],
            fixture.readings,
        )
        assert [item["doi"] for item in found] == ["10.1234/new"]
        fixture.session.execute(update(foreign_user).values(status=0))
        fixture.session.commit()
        with pytest.raises(PermissionError):
            fixture.tool.filter_external([paper()], fixture.readings)
    finally:
        fixture.tearDown()


@pytest.mark.parametrize("library_fails,web_fails", [(False, False), (True, False), (False, True)])
def test_recommend_node_merges_independent_branches_and_keeps_evidence_unchanged(
    library_fails, web_fails
):
    """默认启用联网，两路并行执行，任一路失败仍保留另一条。"""

    async def check():
        """用事件屏障验证库内和联网推荐并行，单路失败仍保留另一分路且不更改原证据。"""
        library_entered, web_entered = asyncio.Event(), asyncio.Event()

        class Materials:
            """提供与联网搜索同步启动的库内推荐替身。"""
            async def recommend(self, readings):
                """两路互等开始事件，验证确实并行。"""
                library_entered.set()
                await web_entered.wait()
                if library_fails:
                    raise RuntimeError("library")
                return {
                    "recommended_documents": [{"corpus_id": 99, "title": "Library paper"}],
                    "recommendation_status": "SUCCESS",
                }

            async def filter_external_recommendations(self, papers, readings):
                """独立测试已覆盖实际全库过滤，这里保留联网候选。"""
                return papers

        class Gateway:
            """提供带真实来源候选的联网替身，可独立注入搜索失败。"""
            async def search_literature(self, *, topics):
                """模拟搜索失败或有真实来源的候选。"""
                web_entered.set()
                await library_entered.wait()
                if web_fails:
                    raise RuntimeError("web")
                return [paper("New web paper")]

        steps = []

        async def step(name, state, operation):
            """只记录现有推荐节点，避免新增图节点。"""
            steps.append(name)
            return await operation()

        state = {"readings": [{"title": "Selected paper"}], "materials": ["original"]}
        result = await asyncio.wait_for(
            LiteratureGraph(Materials(), Gateway(), "fixed", step)._recommend(state),
            3,
        )
        assert steps == ["RecommendLiterature"]
        origins = {item["origin"] for item in result["recommended_documents"]}
        assert origins == (
            {"WEB"} if library_fails else {"LIBRARY"} if web_fails else {"WEB", "LIBRARY"}
        )
        assert result["recommendation_status"] == (
            "PARTIAL" if library_fails or web_fails else "SUCCESS"
        )
        assert state["materials"] == ["original"] and "evidence" not in result

    asyncio.run(check())


@pytest.mark.parametrize(
    "scenario", ["TIMEOUT", "EMPTY_SEARCH", "ALL_FILTERED", "SUCCESS", "FILTER_FAILED"]
)
def test_recommend_diagnostics_distinguish_timeout_empty_search_and_dedupe(scenario):
    """默认联网只搜索一批；空结果不去重、全过滤不补搜，失败诊断留在后端。"""

    async def check():
        """执行不同搜索与去重场景，确认诊断原因和阶段最终随阅读结果发布。"""
        class Gateway:
            """按场景模拟联网超时、空结果或正常候选。"""
            async def search_literature(self, *, topics):
                """注入超时或固定联网候选，检查错误详情不会泄露私有提示内容。"""
                if scenario == "TIMEOUT":
                    raise TimeoutError("private prompt must not appear")
                return [] if scenario == "EMPTY_SEARCH" else [paper("External research")]

        class Materials:
            """固定库内推荐，并可单独模拟联网候选去重失败或全过滤。"""
            async def recommend(self, readings):
                """始终返回库内候选，确保联网失败时仍有独立推荐结果。"""
                return {
                    "recommended_documents": [{"corpus_id": 99, "title": "Local paper"}],
                    "recommendation_status": "SUCCESS",
                }

            async def filter_external_recommendations(self, papers, readings):
                """按场景返回原候选、全过滤或抛出私有错误以验证脱敏。"""
                if scenario == "FILTER_FAILED":
                    raise PermissionError("private actor details must not appear")
                return [] if scenario == "ALL_FILTERED" else papers

        async def step(name, state, operation):
            """直接执行测试节点，不额外改变诊断耗时和结果。"""
            return await operation()

        materials, gateway = Materials(), Gateway()
        gateway.search_literature = AsyncMock(wraps=gateway.search_literature)
        materials.filter_external_recommendations = AsyncMock(
            wraps=materials.filter_external_recommendations,
        )
        graph = LiteratureGraph(materials, gateway, "fixed", step)
        state = {"readings": [{"title": "Input"}]}
        result = await graph._recommend(state)
        gateway.search_literature.assert_awaited_once()
        if scenario in {"TIMEOUT", "EMPTY_SEARCH"}:
            materials.filter_external_recommendations.assert_not_awaited()
        else:
            materials.filter_external_recommendations.assert_awaited_once_with(
                [paper("External research")], state["readings"],
            )
        details = result["web_recommendation_details"]
        assert details["reason"] == scenario
        assert details["elapsed_ms"] >= 0
        assert "private" not in json.dumps(details)
        if scenario in {"TIMEOUT", "FILTER_FAILED"}:
            assert result["web_recommendation_status"] == "FAILED"
            assert result["recommendation_status"] == "PARTIAL"
            assert details["returned_count"] is None
            assert details["stage"] == ("SEARCH" if scenario == "TIMEOUT" else "DEDUPE")
            assert details["candidate_count"] == (None if scenario == "TIMEOUT" else 1)
        else:
            assert result["web_recommendation_status"] == "SUCCESS"
            assert details["candidate_count"] == (0 if scenario == "EMPTY_SEARCH" else 1)
            assert details["returned_count"] == (1 if scenario == "SUCCESS" else 0)
            assert details["filtered_count"] == (1 if scenario == "ALL_FILTERED" else 0)
        final = await graph._finalize(
            {
                "task": "ANALYZE",
                "materials": [],
                "literature_result": {},
                **result,
            }
        )
        assert final["literature_result"]["web_recommendation_details"] == details

    asyncio.run(check())


@pytest.mark.parametrize("library_fails", [False, True])
def test_reading_recommendations_can_explicitly_pause_network_search(library_fails, monkeypatch):
    """显式暂停联网后不调用搜索及全库去重；库内推荐照常执行并独立反映失败。"""
    monkeypatch.setattr("app.agents.literature_graph.ENABLE_WEB_RECOMMENDATIONS", False)
    local = {"recommended_documents": [{"corpus_id": 99, "title": "库内文献"}],
             "recommendation_status": "SUCCESS"}
    materials = SimpleNamespace(
        recommend=AsyncMock(return_value=local, side_effect=(
            RuntimeError("库内暂不可用") if library_fails else None
        )),
        filter_external_recommendations=AsyncMock(side_effect=AssertionError("不应全库去重")),
    )
    gateway = SimpleNamespace(
        search_literature=AsyncMock(side_effect=AssertionError("不应联网搜索")),
    )

    async def step(_name, _state, operation):
        """执行真实推荐节点，避免测试依赖数据库或模型服务。"""
        return await operation()

    state = {"readings": [{"title": "所选文献"}], "materials": ["原文"]}
    result = asyncio.run(LiteratureGraph(materials, gateway, "fixed", step)._recommend(state))
    materials.recommend.assert_awaited_once_with(state["readings"])
    gateway.search_literature.assert_not_awaited()
    materials.filter_external_recommendations.assert_not_awaited()
    assert result["web_recommendation_status"] == "SKIPPED"
    assert result["web_recommendation_details"]["reason"] == "DISABLED"
    assert result["recommendation_status"] == ("FAILED" if library_fails else "SUCCESS")
    assert len(result["recommended_documents"]) == (0 if library_fails else 1)
    assert "evidence" not in result and state["materials"] == ["原文"]
