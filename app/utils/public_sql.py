"""把旧自然语言 SQL 的 corpus 读取限定为公开审核文献，避免新个人材料进入聚合或检索。"""

import re


def public_corpus_sql(sql):
    """为受支持的单条 SELECT 的 corpus 表引用注入公共子查询，保留已有别名。

    Args:
        sql: 旧 NL2SQL 生成的 SELECT；仅支持当前不带库名前缀的业务表引用。

    Returns:
        仅能通过 corpus 引用读取公共已审核记录的 SQL。

    Raises:
        ValueError: 非 SELECT、多语句、注释或带库名前缀的 corpus 引用。
    """
    statement = sql.strip().removesuffix(';').strip()
    if (not re.match(r'^SELECT\b', statement, re.I) or ';' in statement
            or re.search(r'--|/\*|#', statement)
            or re.search(r'(?:\bFROM|\bJOIN|\bTABLE|,)\s*\(\s*`?corpus\b|\bTABLE\s+`?corpus\b', statement, re.I)
            or re.search(r'\b(?:FROM|JOIN)\s*["\[]', statement, re.I)
            or re.search(r'[`.\w]+\s*\.\s*`?corpus\b', statement, re.I)):
        raise ValueError('仅支持单条、不带库名前缀的公开文献 SELECT 查询')
    keywords = {'where', 'join', 'left', 'right', 'inner', 'outer', 'cross', 'on',
                'group', 'order', 'limit', 'having', 'union', 'for', 'offset', 'use', 'force', 'ignore'}
    public_table = '(SELECT * FROM corpus WHERE status = 2 AND owner_id IS NULL AND is_deleted = 0)'

    def replace_table(match):
        """已有别名保留原文，无别名时使用 corpus，保持限定列名仍能解析。"""
        tail = statement[match.end():]
        alias = re.match(r'\s*(?:AS\b\s*)?(`?[A-Za-z_]\w*`?)', tail, re.I)
        suffix = '' if alias and alias.group(1).strip('`').lower() not in keywords else ' AS corpus'
        return f'{match.group(1)} {public_table}{suffix}'

    return re.sub(r'(\bFROM\b|\bJOIN\b|,)\s*(?:`corpus`|corpus\b)',
                  replace_table, statement, flags=re.I)
