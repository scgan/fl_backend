"""验证清理后共库入口完整，退休接口和独立基础设施不会重新进入运行链路。"""

import ast
from pathlib import Path

from fastapi.testclient import TestClient

from app.foreign_integration.database import get_session
from app.foreign_integration.main import app


def test_shared_api_preserves_health_and_rejects_anonymous_runs():
    """只调用无数据库入口，确保健康和认证边界不因清理改变。"""
    # 不进入 lifespan，避免测试触发真实数据库的过期运行回收。
    app.dependency_overrides[get_session] = lambda: None
    try:
        client = TestClient(app)
        assert client.get("/api/v1/health").json()["status"] == "ok"
        assert client.get("/api/v1/agent-runs").status_code == 401
        for route in ("/api/v1/auth/login", "/api/v1/documents", "/api/v1/users"):
            assert client.get(route).status_code == 404
    finally:
        app.dependency_overrides.pop(get_session, None)


def test_runtime_has_no_retired_infrastructure_imports():
    """扫描所有保留源码，防止延迟导入或类型导入残留旧基础设施。"""
    root = Path(__file__).resolve().parents[1]
    retired = {"alembic", "celery", "minio", "fastembed", "redis", "fitz"}
    for path in (root / "app").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            modules = (
                [node.module]
                if isinstance(node, ast.ImportFrom) and node.module
                else [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for module in modules:
                assert module.split(".")[0] not in retired, (path, module)
                if module.startswith("app."):
                    target = root.joinpath(*module.split("."))
                    exists = target.with_suffix(".py").exists() or (target / "__init__.py").exists()
                    assert exists, (path, module)
