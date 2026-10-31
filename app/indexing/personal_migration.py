"""幂等升级个人上传所有者列；保持旧文献为空，不修改其审核状态或归属。"""

from sqlalchemy import inspect, text


def migrate_personal_owner(engine):
    """为既有 corpus 增加可空 owner_id 和查询索引，由部署迁移调用。"""
    inspector = inspect(engine)
    if not inspector.has_table('corpus'):
        raise RuntimeError('corpus 主表不存在，请先创建业务数据库')
    columns = {column['name'] for column in inspector.get_columns('corpus')}
    with engine.begin() as connection:
        if 'owner_id' not in columns:
            connection.execute(text('ALTER TABLE corpus ADD COLUMN owner_id BIGINT NULL'))
        indexes = {index['name'] for index in inspect(connection).get_indexes('corpus')}
        if 'ix_corpus_owner_id' not in indexes:
            connection.execute(text('CREATE INDEX ix_corpus_owner_id ON corpus (owner_id)'))
