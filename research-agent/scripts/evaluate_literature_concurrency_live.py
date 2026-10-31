"""用真实双减文献和 LLM 验证限量并发；远程数据库只读，步骤写入报告目录 SQLite。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session

from app.agents.gateway import get_model_gateway
from app.agents.literature_graph import LiteratureGraph
from app.foreign_integration.config import get_foreign_settings
from app.foreign_integration.literature import (
    LiteratureMaterialTool,
    selection_rows,
    selection_version,
)
from app.foreign_integration.schema import metadata, research_step
from app.foreign_integration.service import RunExecutor, _prompt_and_model


async def main():
    """只读发现真实候选，再用真实模型验证限量并发，步骤仅保存到报告目录 SQLite。

    --discover-only 只读查询并写选文报告；正常模式会读取原文索引、调用模型，
    断言并发峰值、输出顺序和步骤终态，并在失败时仍尽力写报告。

    Raises:
        AssertionError: 就绪候选不足十篇，或并发、顺序、步骤及阅读内容断言失败。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--discover-only", action="store_true", help="只读列出候选文献和模型目的地，不调用 LLM"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = get_foreign_settings()
    os.environ["OPENAI_API_KEY"] = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("QWEN_API_KEY")
        or os.getenv("DASHSCOPE_API_KEY", "")
    )
    # 1. 用数据库会话级只读配置发现候选及当前活动模型，发现模式不调用模型
    remote = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        connect_args={"init_command": "SET SESSION TRANSACTION READ ONLY", "connect_timeout": 15},
    )
    with Session(remote) as session:
        runtime = session.execute(text("SELECT DATABASE() AS db")).mappings().one()
        actors = (
            session.execute(
                text("SELECT id,role FROM user WHERE status=1 AND is_deleted=0 ORDER BY id")
            )
            .mappings()
            .all()
        )
        actor_id = next(
            row["id"]
            for row in actors
            if 2 in {int(value) for value in json.loads(row["role"] or "[]")}
        )
        candidates = (
            session.execute(
                text(
                    "SELECT c.id,c.title_cn AS title,j.chunk_count FROM corpus c "
                    "JOIN corpus_index_job j ON j.corpus_id=c.id "
                    "WHERE c.is_deleted=0 AND c.status=2 AND c.is_vectorized=1 "
                    "AND j.is_deleted=0 AND j.status='SUCCESS' AND j.vector_status='SUCCESS' "
                    "AND j.lexical_status='SUCCESS' AND j.lexical_index=:lexical "
                    "AND c.title_cn LIKE :topic ORDER BY c.id LIMIT 40"
                ),
                {"topic": "%双减%", "lexical": settings.opensearch_index},
            )
            .mappings()
            .all()
        )
        _, prompt, model = _prompt_and_model(session, "LITERATURE")
    destination = urlsplit(model["api_base"] or "")
    if args.discover_only:
        discovery = {
            "database": runtime["db"],
            "remote_database_read_only": True,
            "llm_calls": 0,
            "remote_database_writes": 0,
            "model": model["model_name"],
            "destination": f"{destination.scheme}://{destination.hostname}{destination.path}",
            "candidates": [dict(row) for row in candidates],
        }
        (args.output / "selection.json").write_text(
            json.dumps(discovery, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(discovery, ensure_ascii=False), flush=True)
        remote.dispose()
        return
    print(
        json.dumps(
            {
                "stage": "discovery",
                "database": runtime["db"],
                "candidate_count": len(candidates),
                "model": model["model_name"],
            }
        ),
        flush=True,
    )
    # 使用正式材料工具完成账号、审批、附件版本、片段哈希和页码核验。
    scope = []
    # 2. 逐篇核验原文可用性，冻结十篇真实文献范围及附件版本
    with patch(
        "app.foreign_integration.literature.session_factory", return_value=lambda: Session(remote)
    ):
        for candidate in candidates:
            key = candidate["id"]
            with Session(remote) as session:
                rows = selection_rows(session, actor_id, [key])
            probe = LiteratureMaterialTool(
                actor_id=actor_id,
                knowledge_base_id=0,
                fixed_scope=[key],
                fixed_version=selection_version(rows),
                settings=settings,
            )
            documents, _, _ = await probe.load_materials()
            print(
                json.dumps(
                    {
                        "stage": "candidate",
                        "corpus_id": key,
                        "title": documents[0]["title"],
                        "status": documents[0]["status"],
                        "chunks": documents[0]["chunk_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if documents[0]["status"] == "READY":
                scope.append(key)
            if len(scope) == 10:
                break
        assert len(scope) == 10, f"仅找到 {len(scope)} 篇可核验的双减文献"
        with Session(remote) as session:
            rows = selection_rows(session, actor_id, scope)
        materials = LiteratureMaterialTool(
            actor_id=actor_id,
            knowledge_base_id=0,
            fixed_scope=scope,
            fixed_version=selection_version(rows),
            settings=settings,
        )
        # 3. 将执行器的步骤会话切换到本地 SQLite，远程数据库继续保持只读
        local = create_engine(f"sqlite:///{args.output / 'steps.sqlite'}")
        metadata.create_all(local)
        executor = RunExecutor(os.urandom(16).hex())
        executor.retrieval = materials
        gateway = get_model_gateway(
            model["provider"],
            model_name=model["model_name"],
            api_base=model["api_base"],
            timeout_seconds=180,
            max_tokens=16000,
            temperature=0.0,
        )
        events, active, peak = [], set(), 0
        started = time.monotonic()

        async def tracked(name, state, operation):
            """追踪阶段耗时和文献分析并发峰值，通过执行器把步骤写入本地 SQLite。

            Args:
                name: 当前图节点名，AnalyzeDocument 前缀计入文献并发。
                state: 当前图状态。
                operation: 本阶段异步业务操作。

            Returns:
                原阶段结果，异常与取消向上传播，finally 始终更新本地计时。
            """
            nonlocal peak
            document = name.startswith("AnalyzeDocument:")
            if document:
                active.add(name)
                peak = max(peak, len(active))
            entry = {"node": name, "started_seconds": round(time.monotonic() - started, 3)}
            events.append(entry)
            print(
                json.dumps({"stage": name, "status": "RUNNING", "active": len(active)}), flush=True
            )
            try:
                result = await executor._run_step(name, state, operation)
                entry["status"] = "SUCCESS"
                return result
            except BaseException as exc:
                entry["status"] = type(exc).__name__
                raise
            finally:
                entry["finished_seconds"] = round(time.monotonic() - started, 3)
                if document:
                    active.remove(name)
                print(json.dumps(entry), flush=True)

        # 4. 运行真实并发分析并断言峰值、顺序及终态，成功或失败都在 finally 写本地报告
        report = {
            "database": runtime["db"],
            "remote_database_read_only": True,
            "remote_database_writes": 0,
            "step_storage": "local SQLite via RunExecutor._run_step",
            "model": model["model_name"],
            "corpus_ids": scope,
            "analysis_concurrency": settings.literature_analysis_concurrency,
            "events": events,
        }
        try:
            with patch(
                "app.foreign_integration.service.session_factory",
                return_value=lambda: Session(local),
            ):
                result = await LiteratureGraph(
                    materials,
                    gateway,
                    prompt["system_prompt"],
                    tracked,
                    analysis_concurrency=settings.literature_analysis_concurrency,
                ).run({"task": "ANALYZE"})
            doc_events = [item for item in events if item["node"].startswith("AnalyzeDocument:")]
            synthesis = next(item for item in events if item["node"] == "SynthesizeReading")
            with Session(local) as session:
                steps = [
                    dict(row)
                    for row in session.execute(
                        select(research_step)
                        .where(research_step.c.run_id == executor.run_id)
                        .order_by(research_step.c.sequence)
                    ).mappings()
                ]
            # 先保存完整输出与计时，内容校验失败也能据实检查结果。
            report.update(
                peak_concurrent_documents=peak,
                steps=steps,
                result=result["literature_result"],
                document_duration_sum_seconds=round(
                    sum(item["finished_seconds"] - item["started_seconds"] for item in doc_events),
                    3,
                ),
                analysis_wall_seconds=round(
                    max(item["finished_seconds"] for item in doc_events)
                    - min(item["started_seconds"] for item in doc_events),
                    3,
                ),
            )
            (args.output / "analysis.md").write_text(result["final_answer"], encoding="utf-8")
            documents = result["literature_result"]["documents"]
            assert [item["corpus_id"] for item in documents] == scope
            assert all(item["status"] == "READY" and item["analysis"] for item in documents)
            assert all(len(item["analysis"]["sections"]) == 6 for item in documents)
            assert peak == settings.literature_analysis_concurrency
            assert synthesis["started_seconds"] >= max(
                item["finished_seconds"] for item in doc_events
            )
            assert len(steps) == 14 and all(item["status"] == "SUCCESS" for item in steps)
            overview = result["literature_result"]["overview"]
            assert len(overview["overview"]) >= 5, (
                f"综合概览仅 {len(overview['overview'])} 段，要求至少 5 段"
            )
            assert overview["common_themes"] and overview["differences"]
            assert 3 <= len(overview["potential_innovations"]) <= 5
            report["status"] = "PASS"
        except BaseException as exc:
            report.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            report["elapsed_seconds"] = round(time.monotonic() - started, 3)
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in report.items()
                        if key not in {"events", "steps", "result"}
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            local.dispose()
    remote.dispose()


if __name__ == "__main__":
    asyncio.run(main())
