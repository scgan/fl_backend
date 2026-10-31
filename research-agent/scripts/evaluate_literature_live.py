"""真实 LLM 烟测，默认仅发送脚本内编写的虚构材料，无数据库连接或写入。"""

import asyncio
import json
import os
import sys
import time
import uuid

from app.agents.catalog import LITERATURE_SYSTEM_PROMPT
from app.agents.gateway import get_model_gateway
from app.agents.literature_graph import LiteratureGraph
from app.foreign_integration.config import get_foreign_settings
from app.retrieval.schemas import CitationRead, EvidenceRead

FIXTURES = [
    (
        101,
        "教师协作与备课质量（虚构测试材料）",
        [
            "这是为软件验证编写的虚构研究材料，不能作为真实研究引用。"
            "研究关注教师协作备课与教学准备质量之间的关系。研究团队对6所学校的72名教师"
            "进行了8周跟踪观察，使用教师日志和备课评价量表。未随机分配教师，也没有对照组。",
            "在这项虚构研究中，参与者报告协作有助于共享教学资源，备课评价得分从平均68分"
            "提高到74分。作者强调这是观察性关联，不能据此声称协作导致提升。样本来自同一城市，"
            "不能直接推广到所有地区。研究贡献是描述协作过程，局限是缺少对照和长期追踪。",
        ],
    ),
    (
        102,
        "课后服务中的教师体验（虚构测试材料）",
        [
            "这是为软件验证编写的虚构研究材料，不能作为真实研究引用。研究关注课后服务"
            "对教师日常工作体验的影响。研究者访谈了3所学校的24名教师，使用半结构化访谈"
            "与主题分析，没有实施实验，也没有测量学生成绩变化。",
            "在这项虚构研究中，受访教师描述了时间安排压力、同事协作和资源支持三个主题。"
            "作者认为支持条件影响教师对课后服务的感受，不能把访谈描述解释为因果效应。"
            "作者建议改进排班并提供教学资源，局限是访谈样本较小且依赖教师自述。",
        ],
    ),
]


class FixtureMaterials:
    """只提供脚本自编虚构材料的阅读工具，避免真实模型烟测访问业务数据。

    Attributes:
        selected: 本轮选中的固定材料 ID，不查询数据库。
    """
    def __init__(self, selected):
        """固定本轮虚构材料范围，后续加载不扩大选中集合。"""
        self.selected = selected

    async def recommend(self, readings):
        """关闭本次烟测的外部推荐分支，返回成功且无候选的结果。"""
        return {"recommended_documents": [], "recommendation_status": "SUCCESS"}

    async def load_materials(self):
        """将选中的固定段落映射为文献、证据和引用。

        Returns:
            文献、证据及引用列表的三元组，引用序号按段落连续生成。
        """
        documents, evidence, citations = [], [], []
        for corpus_id, title, paragraphs in FIXTURES:
            if corpus_id not in self.selected:
                continue
            documents.append(
                {
                    "corpus_id": corpus_id,
                    "title": title,
                    "author": "软件测试材料",
                    "year": None,
                    "status": "READY",
                    "reason": "",
                    "chunk_count": len(paragraphs),
                    "attachment_paths": [f"fixture-{corpus_id}.pdf"],
                }
            )
            for page, content in enumerate(paragraphs, 1):
                index = len(evidence) + 1
                item = EvidenceRead(
                    citation_index=index,
                    document_id=uuid.UUID(int=corpus_id),
                    chunk_id=uuid.UUID(int=corpus_id * 100 + page),
                    document_title=title,
                    authors=[],
                    publication_year=None,
                    journal=None,
                    language_code="zh",
                    page_start=page,
                    page_end=page,
                    section=None,
                    content=content,
                    relevance_score=1,
                    retrieval_score=1,
                    vector_rank=None,
                    lexical_rank=None,
                )
                evidence.append(item)
                citations.append(
                    CitationRead(
                        citation_index=index,
                        document_id=item.document_id,
                        chunk_id=item.chunk_id,
                        document_title=title,
                        page_start=page,
                        page_end=page,
                        section=None,
                        quote=content,
                        file_url=f"/fixtures/{corpus_id}.pdf#page={page}",
                    )
                )
        return documents, evidence, citations


async def main():
    """用真实模型验证虚构材料的单篇、多篇及追问阅读，将结果写入参数指定的报告文件。

    不连接数据库，但会发送模型请求并产生模型用量；断言用于检验篇幅、创新方向和引用。

    Raises:
        AssertionError: 阅读结构、最低篇幅或追问的文献范围不符合预期。
    """
    # 1. 固定本次真实模型网关，只使用脚本内的虚构材料替身，不建立业务库会话
    settings = get_foreign_settings()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or os.getenv("QWEN_API_KEY", "")
    gateway = get_model_gateway(
        "OPENAI_COMPATIBLE",
        model_name=settings.model_name,
        api_base=settings.model_base_url,
        timeout_seconds=180,
        max_tokens=16000,
        temperature=0.0,
    )
    report = {
        "model": settings.model_name,
        "materials": "SCRIPT_AUTHORED_FICTIONAL_FIXTURES",
        "checks": [],
        "database_connections": 0,
        "database_writes": 0,
    }

    async def tracked(name, state, operation):
        """执行图阶段并打印完成标记，不持久化远程研究运行。"""
        result = await operation()
        print(json.dumps({"stage": name, "status": "PASS"}), flush=True)
        return result

    async def run(scope, task="ANALYZE", query="", history=None, previous=None, focus=None):
        """将选中文献、任务、历史和预览焦点传给同一阅读图，复用虚构材料与真实网关。"""
        return await LiteratureGraph(
            FixtureMaterials(scope), gateway, LITERATURE_SYSTEM_PROMPT, tracked
        ).run(
            {
                "task": task,
                "query": query,
                "history": history or [],
                "previous_analysis": previous,
                "focus_corpus_id": focus,
            }
        )

    # 2. 依次验证单篇、多篇、连续追问和预览指代，记录真实模型结果及耗时
    started = time.monotonic()
    single = await run([101])
    assert single["literature_result"]["documents"][0]["analysis"]
    single_overview = single["literature_result"]["overview"]["overview"]
    single_overview_chars = sum(len(item["text"]) for item in single_overview)
    print(
        json.dumps(
            {
                "scope": "single",
                "overview_paragraphs": len(single_overview),
                "overview_characters": single_overview_chars,
            }
        ),
        flush=True,
    )
    assert len(single_overview) >= 3 and single_overview_chars >= 300, (
        f"单篇概览未达到最低篇幅：{len(single_overview)} 段 / {single_overview_chars} 字"
    )
    assert 3 <= len(single["literature_result"]["overview"]["potential_innovations"]) <= 5
    report["checks"].append(
        {
            "name": "single_document",
            "citations": len(single["validated_citations"]),
            "overview_paragraphs": len(single_overview),
            "overview_characters": single_overview_chars,
            "potential_innovations": len(
                single["literature_result"]["overview"]["potential_innovations"]
            ),
            "result": single["literature_result"],
        }
    )
    multi = await run([101, 102])
    assert all(item["analysis"] for item in multi["literature_result"]["documents"])
    multi_overview = multi["literature_result"]["overview"]["overview"]
    multi_overview_chars = sum(len(item["text"]) for item in multi_overview)
    print(
        json.dumps(
            {
                "scope": "multiple",
                "overview_paragraphs": len(multi_overview),
                "overview_characters": multi_overview_chars,
            }
        ),
        flush=True,
    )
    assert len(multi_overview) >= 5 and multi_overview_chars >= 500, (
        f"多篇概览未达到最低篇幅：{len(multi_overview)} 段 / {multi_overview_chars} 字"
    )
    assert 3 <= len(multi["literature_result"]["overview"]["potential_innovations"]) <= 5
    report["checks"].append(
        {
            "name": "multiple_documents",
            "citations": len(multi["validated_citations"]),
            "overview_paragraphs": len(multi_overview),
            "overview_characters": multi_overview_chars,
            "potential_innovations": len(
                multi["literature_result"]["overview"]["potential_innovations"]
            ),
            "result": multi["literature_result"],
        }
    )
    background = {
        "overview": multi["literature_result"]["overview"],
        "documents": [
            {
                "corpus_id": item["corpus_id"],
                "title": item["title"],
                "summary": item["analysis"]["summary"],
            }
            for item in multi["literature_result"]["documents"]
        ],
    }
    first = await run(
        [101, 102], task="ASK", query="请介绍第一篇文献的核心观点。", previous=background
    )
    assert first["literature_result"]["target_corpus_ids"] == [101]
    second = await run(
        [101, 102],
        task="ASK",
        query="第二篇呢？",
        previous=background,
        history=[
            {"role": "user", "content": "请介绍第一篇文献的核心观点。"},
            {"role": "assistant", "content": first["final_answer"]},
        ],
    )
    assert second["literature_result"]["target_corpus_ids"] == [102]
    assert second["validated_citations"]
    assert all(item.document_id.int == 102 for item in second["validated_citations"])
    report["checks"].extend(
        [
            {"name": "first_document_question", "answer": first["final_answer"]},
            {
                "name": "second_document_followup",
                "answer": second["final_answer"],
                "standalone_question": second["literature_result"]["standalone_question"],
            },
        ]
    )
    focused = await run(
        [101, 102], task="ASK", query="这篇主要研究什么？", previous=background, focus=102
    )
    assert focused["literature_result"]["target_corpus_ids"] == [102]
    report["checks"].append(
        {"name": "preview_document_reference", "answer": focused["final_answer"]}
    )
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    # 3. 将通过断言的本轮结果写成报告文件，不将虚构材料发布到业务数据库
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": len(report["checks"]),
                "database_writes": 0,
                "elapsed_seconds": report["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
