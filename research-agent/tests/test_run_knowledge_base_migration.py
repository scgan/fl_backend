"""验证旧 Run 知识库删列的数据保护、可重复执行与 MySQL 外键处理，不连接业务数据库。"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.dialects import mysql

from app.foreign_integration.migrate import _remove_run_knowledge_base


def legacy_engine(run_base, conversation_base, *, orphan=False, deleted=False):
    """建立删列前的最小旧表，并允许构造绑定冲突、孤立运行和软删除历史。"""
    engine = create_engine("sqlite://")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE research_conversation (id TEXT PRIMARY KEY, knowledge_base_id BIGINT)"
        )
        connection.exec_driver_sql(
            "CREATE TABLE research_run (id TEXT PRIMARY KEY, conversation_id TEXT, "
            "knowledge_base_id BIGINT, is_deleted BOOLEAN)"
        )
        if not orphan:
            connection.execute(text("INSERT INTO research_conversation VALUES ('c', :base)"),
                               {"base": conversation_base})
        connection.execute(text("INSERT INTO research_run VALUES ('r', 'c', :base, :deleted)"),
                           {"base": run_base, "deleted": deleted})
    return engine


@pytest.mark.parametrize("base", [None, 9])
def test_drop_preserves_history_and_is_repeatable(base):
    """研究和阅读历史均保留运行及会话绑定，删列后可再次升级。"""
    engine = legacy_engine(base, base)
    try:
        _remove_run_knowledge_base(engine)
        _remove_run_knowledge_base(engine)
        with engine.connect() as connection:
            assert connection.exec_driver_sql("SELECT * FROM research_run").one() == ("r", "c", 0)
            assert connection.exec_driver_sql(
                "SELECT knowledge_base_id FROM research_conversation"
            ).scalar_one_or_none() == base
    finally:
        engine.dispose()


@pytest.mark.parametrize("run_base,conversation_base,orphan,deleted", [
    (None, 9, False, False), (9, None, False, False), (9, 10, False, False),
    (9, 9, True, False), (9, 10, False, True),
])
def test_mismatch_or_orphan_blocks_drop(run_base, conversation_base, orphan, deleted):
    """NULL 差异、不同 ID、孤立运行及软删除冲突都必须保留旧列等待核对。"""
    engine = legacy_engine(run_base, conversation_base, orphan=orphan, deleted=deleted)
    try:
        with pytest.raises(RuntimeError, match="停止删除旧字段"):
            _remove_run_knowledge_base(engine)
        assert "knowledge_base_id" in {item["name"] for item in inspect(engine).get_columns(
            "research_run"
        )}
        with engine.connect() as connection:
            assert connection.exec_driver_sql(
                "SELECT knowledge_base_id FROM research_run"
            ).scalar_one_or_none() == run_base
    finally:
        engine.dispose()


def test_mysql_drops_only_knowledge_base_foreign_key_with_column():
    """MySQL 按实际名称转义并同时删除目标外键和列，保留会话外键。"""
    engine = MagicMock()
    engine.dialect = mysql.dialect()
    connection = engine.begin.return_value.__enter__.return_value
    connection.execute.return_value.first.return_value = None
    inspector = MagicMock()
    inspector.get_columns.return_value = [{"name": "knowledge_base_id"}]
    inspector.get_foreign_keys.return_value = [
        {"name": "legacy`kb", "constrained_columns": ["knowledge_base_id"]},
        {"name": "conversation_fk", "constrained_columns": ["conversation_id"]},
    ]
    with patch("app.foreign_integration.migrate.inspect", return_value=inspector):
        _remove_run_knowledge_base(engine)
    connection.exec_driver_sql.assert_called_once_with(
        "ALTER TABLE research_run DROP FOREIGN KEY `legacy``kb`, DROP COLUMN knowledge_base_id"
    )
