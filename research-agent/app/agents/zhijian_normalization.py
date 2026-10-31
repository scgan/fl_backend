"""提供智鉴比较对象的确定性规范化与歧义判断规则。

规则层只负责把已解析的对象映射为稳定内部表示，不执行检索或专业分析。国家表
覆盖一期演示与 Gold Dataset 常见对象，未知对象会按可扩展 OTHER 类型保守处理。
"""

from __future__ import annotations

import re

from app.agents.zhijian_schemas import CompareObject

# 每项包含稳定标识、展示名和跨语言检索别名；别名先做 casefold 后匹配。
COUNTRIES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "中国": ("CN", "中国", ("中国", "中华人民共和国", "China", "Chinese")),
    "美国": (
        "US",
        "美国",
        ("美国", "美", "United States", "US", "USA", "American"),
    ),
    "日本": ("JP", "日本", ("日本", "Japan", "Japanese")),
    "韩国": ("KR", "韩国", ("韩国", "South Korea", "Korea", "Korean")),
    "英国": ("GB", "英国", ("英国", "United Kingdom", "UK", "British")),
    "德国": ("DE", "德国", ("德国", "Germany", "German")),
    "法国": ("FR", "法国", ("法国", "France", "French")),
    "新加坡": ("SG", "新加坡", ("新加坡", "Singapore", "Singaporean")),
    "澳大利亚": ("AU", "澳大利亚", ("澳大利亚", "澳洲", "Australia", "Australian")),
    "加拿大": ("CA", "加拿大", ("加拿大", "Canada", "Canadian")),
}

# 这些简称会显著改变实体含义，验证节点必须要求用户澄清。
AMBIGUOUS_OBJECTS: dict[str, str] = {
    "刚果": "“刚果”可能指刚果共和国或刚果民主共和国，请明确要比较的国家。",
    "韩国和朝鲜": "请明确需要比较韩国、朝鲜中的哪一个对象。",
}


def normalize_compare_object(raw: str) -> CompareObject:
    """把模型解析出的对象转换为稳定内部表示。

    Args:
        raw: 用户原文中的对象名称。

    Returns:
        国家使用 ISO 标识，其他对象使用规范文本标识的比较对象。
    """
    value = " ".join(raw.split()).strip("，,。.!！？")
    folded = value.casefold()
    for _, (identifier, canonical, aliases) in COUNTRIES.items():
        if folded in {alias.casefold() for alias in aliases}:
            return CompareObject(
                raw=value,
                canonical_name=canonical,
                type="COUNTRY",
                identifier=identifier,
                aliases=list(dict.fromkeys((canonical, *aliases))),
            )
    # 非国家对象保留用户语义，只消除空白和大小写差异；类型由有限规则粗分，
    # 后续扩充词典不会改变当前 Run 已持久化的规范结果。
    object_type = "OTHER"
    if any(token in value for token in ("理论", "主义")):
        object_type = "THEORY"
    elif any(token in value for token in ("政策", "纲要", "计划")):
        object_type = "POLICY"
    elif any(token in value for token in ("方法", "研究法")):
        object_type = "METHOD"
    elif any(token in value for token in ("教育模式", "教学模式", "学习模式")):
        object_type = "EDUCATION_MODEL"
    identifier = re.sub(r"\s+", "-", folded)
    return CompareObject(
        raw=value,
        canonical_name=value,
        type=object_type,
        identifier=identifier,
        aliases=[value],
    )


def ambiguity_question(raw: str) -> str | None:
    """判断对象是否命中会改变任务含义的已知歧义。

    Args:
        raw: 待判断对象原文。

    Returns:
        需要追问时的自然语言问题，否则为空。
    """
    normalized = " ".join(raw.split())
    for key, question in AMBIGUOUS_OBJECTS.items():
        if key in normalized:
            return question
    return None
