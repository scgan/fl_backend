"""稳定的 Qwen3 分块预算估算；百炼实际计费用量以 API usage 为准。"""

from functools import lru_cache
import os
from pathlib import Path

from app.utils.bailian_client import DEFAULT_EMBEDDING_MODEL

EMBEDDING_MODEL = DEFAULT_EMBEDDING_MODEL
TOKENIZER_NAME = "Qwen/Qwen3-Embedding-0.6B"
# 此词表仅用于维持现有检索粒度，不能宣称等于新模型的 tokenizer。
# 1 是分块预留量；新模型的实际输入含任务包装，不与本地估算强行相等。
INPUT_OVERHEAD = 1


@lru_cache(maxsize=1)
def get_tokenizer():
    """读取随镜像发布的词表；禁止缺失时退回字符数或其他模型词表。"""
    from tokenizers import Tokenizer

    path = os.getenv("FOREIGN_EMBEDDING_TOKENIZER_PATH") or (
        Path(__file__).parent / "tokenizers/qwen3-embedding/tokenizer.json"
    )
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer


def count_tokens(text):
    """返回分块估算值，包含预留量，不用于 API 计费或模型输入硬上限。"""
    return len(get_tokenizer().encode(text, add_special_tokens=False).ids) + INPUT_OVERHEAD
