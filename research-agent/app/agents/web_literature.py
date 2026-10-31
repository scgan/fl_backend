"""从真实联网搜索来源构建推荐；不访问题录核验服务，不生成回答 Evidence。"""

from __future__ import annotations

import html
import ipaddress
import json
import re
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import unquote, urlsplit, urlunsplit

# 单次只获取至多五篇候选；全库去重后不补搜、不凑数。
MAX_WEB_RECOMMENDATIONS = 5
# 搜索最多等待两分钟，去重和步骤落库分别保留独立收尾时间。
WEB_SEARCH_TIMEOUT_SECONDS = 120
WEB_RECOMMEND_TIMEOUT_SECONDS = 135
RECOMMEND_NODE_TIMEOUT_SECONDS = 140


def normalized_title(value: str) -> str:
    """统一题名的 Unicode、大小写、空白和标点，保留各语言字母及数字。"""
    value = html.unescape(re.sub(r"<[^>]*>", "", str(value or "")))
    return "".join(c for c in unicodedata.normalize("NFKC", value).casefold() if c.isalnum())


def same_title(left: str, right: str) -> bool:
    """识别同一题名的精确或轻微排版差异，避免用主题词重合代替去重。"""
    a, b = normalized_title(left), normalized_title(right)
    return bool(
        a
        and b
        and (a == b or (min(len(a), len(b)) >= 12 and SequenceMatcher(None, a, b).ratio() >= 0.94))
    )


def public_url(value: str) -> str:
    """仅保留可供浏览器打开的公开 HTTP(S) URL；不请求这些模型提供的网址。"""
    if not isinstance(value, str) or len(value) > 2048 or any(ord(c) < 32 for c in value):
        return ""
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname or ""
        if (
            parsed.scheme not in {"http", "https"}
            or not host
            or parsed.username
            or parsed.password
            or parsed.port not in {None, 80, 443}
            or "." not in host
            or host.endswith((".local", ".localhost", ".internal"))
        ):
            return ""
        try:
            if not ipaddress.ip_address(host).is_global:
                return ""
        except ValueError:
            pass
        return urlunsplit((parsed.scheme, parsed.netloc.casefold(), parsed.path, parsed.query, ""))
    except ValueError:
        return ""


def extract_doi(value: str) -> str:
    """从既有来源文本或论文 URL 提取 DOI，供题录查询及全库去重。"""
    match = re.search(r"10\.\d{4,9}/[^\s<>\"?#]+", unquote(str(value or "")), re.IGNORECASE)
    doi = match.group().rstrip(".,;。；").casefold() if match else ""
    while doi.endswith(")") and doi.count(")") > doi.count("("):
        doi = doi[:-1]
    return doi


def _text_field(paper: dict, key: str, limit: int) -> str:
    """将搜索题录字段限制为纯文本，缺失或格式错误时留空。"""
    value = paper.get(key)
    return (
        html.unescape(re.sub(r"<[^>]*>", "", value)).strip()[:limit]
        if isinstance(value, str) else ""
    )


def search_candidates(data: dict) -> list[dict]:
    """只接收真实 web_search_call 来源中的链接，拒绝模型凭记忆补写的 URL。"""
    if data.get("status") != "completed" or data.get("error"):
        raise ValueError("联网搜索未完整完成")
    outputs = data.get("output") or []
    urls = {
        public_url(source.get("url", ""))
        for item in outputs
        if item.get("type") == "web_search_call" and item.get("status") == "completed"
        for source in (item.get("action") or {}).get("sources", [])
    } - {""}
    # Responses 可能先输出检索说明；只解析最后一条含 papers 对象的助手消息。
    for item in reversed(outputs):
        if item.get("type") != "message" or item.get("role") != "assistant":
            continue
        text = "".join(
            part.get("text", "")
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        match = re.search(r'\{\s*"papers"\s*:', text)
        if match is None:
            continue
        result, _ = json.JSONDecoder().raw_decode(text[match.start() :])
        papers = result.get("papers")
        if not isinstance(papers, list):
            raise ValueError("联网文献名单格式无效")
        # 明确的空名单直接结束；非空候选仍必须有真实搜索来源，不能用模型记忆补齐。
        if not papers:
            return []
        if not urls:
            raise ValueError("模型未返回可核验的联网搜索来源")
        candidates, seen = [], set()
        for paper in papers[:30]:
            if not isinstance(paper, dict):
                continue
            url = public_url(paper.get("url", ""))
            title = paper.get("title", "")
            if (
                url not in urls
                or url in seen
                or not isinstance(title, str)
                or not 4 <= len(title.strip()) <= 1000
            ):
                continue
            seen.add(url)

            # 题录字段来自联网检索输出，缺失则留空；不要求 DOI 或登记机构收录。
            year = paper.get("year")
            if isinstance(year, str) and re.fullmatch(r"\d{4}", year):
                year = int(year)
            candidates.append(
                {
                    "corpus_id": None,
                    "origin": "WEB",
                    "title": _text_field(paper, "title", 1000),
                    "author": _text_field(paper, "author", 1000),
                    "year": year if type(year) is int and 1000 <= year <= 2100 else None,
                    "source": _text_field(paper, "source", 500),
                    "doi": extract_doi(url),
                    "external_url": url,
                    "search_url": url,
                    "attachment_paths": [],
                }
            )
            if len(candidates) >= MAX_WEB_RECOMMENDATIONS:
                break
        return candidates
    raise ValueError("联网搜索未返回文献名单")
