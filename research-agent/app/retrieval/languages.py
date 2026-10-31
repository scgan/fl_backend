"""统一来源语言名称、历史代码与地区代码，供比较请求和文献过滤共同使用。"""

# 别名用于兼容现有题录与上传表单，不构成可检索语言的白名单。
_LANGUAGE_ALIASES = {
    "中文": "zh", "汉语": "zh", "chinese": "zh", "zho": "zh", "chi": "zh",
    "英文": "en", "英语": "en", "english": "en", "eng": "en",
    "日文": "ja", "日语": "ja", "japanese": "ja", "jp": "ja", "jpn": "ja",
    "韩文": "ko", "韩语": "ko", "朝语": "ko", "朝鲜语": "ko",
    "korean": "ko", "kr": "ko", "kor": "ko",
    "法文": "fr", "法语": "fr", "french": "fr", "fra": "fr", "fre": "fr",
    "德文": "de", "德语": "de", "german": "de", "deu": "de", "ger": "de",
    "俄文": "ru", "俄语": "ru", "russian": "ru", "rus": "ru",
    "西语": "es", "西班牙文": "es", "西班牙语": "es", "spanish": "es", "spa": "es",
    "葡语": "pt", "葡萄牙语": "pt", "portuguese": "pt", "por": "pt",
    "阿语": "ar", "阿拉伯语": "ar", "arabic": "ar", "ara": "ar",
}


def normalize_source_language(value: str) -> str:
    """将来源语言归一到小写代码，未知名称保留以匹配同名题录。

    Args:
        value: 题录语言或用户明确要求的语言名称、代码及地区变体。

    Returns:
        统一语言代码；地区变体使用主语言代码，空输入返回空串。
    """
    lowered = value.strip().casefold().replace("_", "-")
    primary = lowered.split("-", 1)[0]
    return _LANGUAGE_ALIASES.get(primary, primary)
