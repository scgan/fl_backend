"""只运行共库 Agent 能力的 FastAPI 入口，不注册旧文档和账号路由。"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect, text

from app.common.responses import AppError, app_error_handler, validation_error_handler
from app.foreign_integration.api import router
from app.foreign_integration.database import get_engine, session_factory
from app.foreign_integration.schema import RESEARCH_TABLES
from app.foreign_integration.service import recover_stale_runs


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时回收已过期运行，避免容器重启留下永久运行状态。

    Args:
        _app: 当前 Agent 应用。

    Yields:
        已完成启动检查的应用生命周期。
    """
    with session_factory()() as session:
        recover_stale_runs(session)
    yield


app = FastAPI(title="EduResearch Agent for Foreign", lifespan=lifespan)
app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(RequestValidationError, validation_error_handler)

# 仅在显式配置来源时开放跨域（供独立编辑器演示页调用）；未配置时保持同源策略。
_cors_origins = [
    origin.strip()
    for origin in os.getenv("FOREIGN_AGENT_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        max_age=600,
    )


@app.middleware("http")
async def trace_request(request: Request, call_next):
    """在所有共库 Agent 响应中保留稳定请求追踪头。

    Args:
        request: 当前 HTTP 请求。
        call_next: 下游处理器。

    Returns:
        携带请求 ID 的响应。
    """
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(router, prefix="/api/v1")


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    """返回不依赖外部服务的容器存活状态。

    Returns:
        存活状态。
    """
    return {"status": "ok", "service": "eduresearch-foreign-agent"}


@app.get("/api/v1/health/ready")
def ready() -> dict[str, object]:
    """确认 Foreign 共库和全部研究表已经可访问。

    Returns:
        当前数据库与研究表就绪状态。

    Raises:
        AppError: 数据库或新研究表尚不可用。
    """
    try:
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            connection.execute(
                text(
                    "SELECT vector_status, lexical_status, build_id, lexical_index, chunk_count "
                    "FROM corpus_index_job LIMIT 0"
                )
            )
            existing = set(inspect(connection).get_table_names())
        missing = [table.name for table in RESEARCH_TABLES if table.name not in existing]
        if missing:
            raise RuntimeError("研究表缺失")
    except Exception as exc:
        raise AppError("SERVICE_NOT_READY", "Foreign 共库尚未就绪", 503) from exc
    return {"status": "ready", "database": "foreign", "research_tables": len(RESEARCH_TABLES)}
