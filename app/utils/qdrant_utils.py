"""
Foreign 索引兼容入口和向量适配器：共享分块写入 Qdrant/OpenSearch，保留普通向量搜索。

整体流程（向量化索引）:
    corpus_id + attachments
        ↓
    读取原文与坐标 (_read_index_blocks)
        ↓
    文本分块 (_blocks_to_chunks) —— 保留 chunk 在原文中的字符位置
        ↓
    批量向量化 (_get_embedding) —— qwen3.7-text-embedding-flash，每批最多 20 条
        ↓
    构建共享 PointStruct，批量写入 Qdrant 和 OpenSearch，携带同一构建版本和正文哈希
        ↓
    外层索引服务/Worker 复核任务后发布 SUCCESS 和 Corpus.is_vectorized = True

检索流程（向量搜索）:
    用户问题文本
        ↓
    向量化 (_get_embedding) —— 得到 1024 维向量
        ↓
    Qdrant 余弦相似度检索 —— 按 corpus_id 可选过滤，返回 top_k 个最相关 chunk
        ↓
    提取每个 chunk 的 text / score / 原文位置等信息返回

数据结构设计:
    - 一个文献 (corpus_id) 的一个附件文件，会被切分成 N 个 chunk。
    - 每个 chunk 在 Qdrant 中是一个独立的 point。
    - point.id = corpus_id * 1_000_000 + attachment_index * 10_000 + chunk_index
    - point.payload 包含：文献级元数据（标题/作者/分类等）、文件级信息（路径/扩展名）、
      chunk 级定位（字符区间、前后邻 chunk ID、高亮文本）。
    - 相邻 chunk 之间通过 prev_chunk_id / next_chunk_id 形成链式结构，便于上下文扩展。
"""

from __future__ import annotations

import os
import hashlib
import logging
import json
import math
import tempfile
from io import BytesIO
from urllib.parse import unquote, urlsplit

import pdfplumber
import qdrant_client
import requests
from docx import Document
from qdrant_client import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.utils.bailian_client import embed_texts, embedding_config
from app.utils.pdf_layout import plumber_lines, words_to_lines

_qdrant_client = None

COLLECTION_NAME = os.getenv("FOREIGN_QDRANT_COLLECTION", "corpus_chunks")
VECTOR_SIZE = embedding_config().dimension
BATCH_SIZE = embedding_config().batch_size
PAYLOAD_VERSION = "v4"  # 两路共享 build_id、附件指纹与正文哈希


def _attachment_path(file_path: str) -> str:
    """仅解析完整 HTTP(S) URL；普通文件路径中的 ?、#、; 和 % 均为文件名。"""
    if file_path.lower().startswith(("http://", "https://")):
        return unquote(urlsplit(file_path).path).lstrip('/')
    return file_path


def _normalize_file_path(file_path: str, base_dir: str = None) -> str:
    """
    将文件路径规范化为相对路径形式，存入 Qdrant。

    - 绝对路径：去掉 base_dir（或常见 /home/ccnu/Foreign/FileDev/ 前缀）得到相对路径
    - 完整 URL：去掉 scheme + host 取 path 部分
    - 已是相对路径：直接返回
    - 存为 uploaded_files/corpus/xxx.pdf 形式

    Args:
        file_path: 原始路径（绝对/URL/相对）
        base_dir: 基准目录，None 时自动探测常见路径

    Returns:
        相对路径字符串
    """
    if not file_path:
        return file_path

    # 1. 完整 URL → 取 path
    file_path = _attachment_path(file_path)

    # 2. 绝对路径 → 去掉 base_dir
    if os.path.isabs(file_path):
        # 优先用传入的 base_dir
        if base_dir and file_path.startswith(base_dir):
            rel = file_path[len(base_dir):].lstrip(os.sep)
            if rel:
                return rel
        # 自动探测常见前缀
        for prefix in [
            "/home/ccnu/Foreign/FileDev/",
            "/home/ccnu/FileDev/",
            "/app/",
        ]:
            if file_path.startswith(prefix):
                rel = file_path[len(prefix):]
                if rel:
                    return rel
        # 无法规范化则直接 strip os.sep
        return os.path.basename(file_path)

    return file_path


def _get_embedding(texts: list[str], text_type="document") -> list[list[float]]:
    """
    入库与查询共用百炼向量模型，显式区分 document/query。

    Args:
        texts: 文本列表（内部按配置自动分批调用 API）

    Returns:
        list[list[float]]: 每个文本对应一个 1024 维 float 向量，顺序与输入一致
    """
    return embed_texts(texts, text_type=text_type)


def get_qdrant_client(host=None, port=None):
    """
    延迟初始化 Qdrant 客户端（首次调用时创建客户端并确保 collection 存在）。
    后续调用直接返回缓存的客户端实例。

    连接策略：
        1. 优先使用显式传入的 host/port
        2. 否则使用环境变量 QDRANT_HOST/QDRANT_PORT
        3. 若主机名不可解析或不可连通，自动回退到 localhost / 127.0.0.1
    这样可以兼容 Docker 网络内（host=qdrant）和宿主机直连两种运行方式。
    """
    global _qdrant_client
    if _qdrant_client is None:
        port = int(port or os.getenv("QDRANT_PORT", 6333))
        primary_host = host or os.getenv("QDRANT_HOST", "localhost")

        candidate_hosts = []
        for candidate in (primary_host, "localhost", "127.0.0.1"):
            if candidate and candidate not in candidate_hosts:
                candidate_hosts.append(candidate)

        last_error = None
        for candidate_host in candidate_hosts:
            try:
                client = qdrant_client.QdrantClient(host=candidate_host, port=port)
                client.get_collections()
                _ensure_collection(client)
                _qdrant_client = client
                break
            except Exception as exc:
                last_error = exc
                continue

        if _qdrant_client is None:
            raise RuntimeError(
                f"无法连接 Qdrant，已尝试 hosts={candidate_hosts}, port={port}，最后错误: {last_error}"
            )
    return _qdrant_client


def _ensure_collection(client=None):
    """
    若 collection 不存在则创建。
    使用余弦相似度 (COSINE) 作为向量距离度量，适合语义相似性检索。
    HNSW 索引参数 m=16 / ef_construct=100 平衡检索精度与构建速度。
    """
    client = client if client is not None else _qdrant_client
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
            hnsw_config=models.HnswConfigDiff(
                m=16,
                ef_construct=100,
            ),
        )
    else:
        vectors = client.get_collection(COLLECTION_NAME).config.params.vectors
        if (not isinstance(vectors, models.VectorParams) or vectors.size != VECTOR_SIZE
                or vectors.distance != models.Distance.COSINE):
            raise RuntimeError("Qdrant 集合维度或距离与当前 embedding 配置不一致")


def _read_file_text(file_path: str) -> str | None:
    """从文件服务读取并解析原文，供旧正文与引用片段入口使用。

    Args:
        file_path: 文件引用或 HTTP(S) 地址，先提取文件服务相对路径。

    Returns:
        PDF 成功返回含 text、pages 的字典；TXT、DOCX 成功返回文本。
        获取或解析失败、无正文及不支持的格式返回 None。
    """
    from io import BytesIO
    from app.utils.utils import get_file_content

    # 提取相对路径，兼容：
    #   - 完整 URL: http://host/uploads/uploaded_files/corpus/xxx.pdf  → uploaded_files/corpus/xxx.pdf
    #   - 相对路径: uploaded_files/corpus/xxx.pdf  → uploaded_files/corpus/xxx.pdf
    #   - 纯文件名: xxx.pdf  → xxx.pdf
    relative = _attachment_path(file_path)

    # 从路径末尾提取扩展名（如 xxx.PDF → pdf）
    ext = relative.rsplit('.', 1)[-1].lower() if '.' in relative else ''
    print(f"[VECTORIZE][READ] 请求文件: ext={ext or '?'} path={relative}")
    content, meta = get_file_content(relative)

    if not content:
        print(f"[VECTORIZE][READ] 读取失败: meta={meta} path={relative} detail=空响应")
        return None

    if not isinstance(content, bytes):
        preview = str(content)
        if len(preview) > 200:
            preview = preview[:200] + '...'
        print(f"[VECTORIZE][READ] 读取失败: meta={meta} path={relative} detail={preview}")
        return None

    print(f"[VECTORIZE][READ] 下载成功: bytes={len(content)} meta={meta} path={relative}")

    if ext == 'pdf':
        # 用 pdfplumber 解析 PDF，返回结构化数据（含页边界）
        # 返回 dict: {"text": 全文本, "pages": [{"page_num": int, "text": str}, ...]}
        try:
            bio = BytesIO(content)
            with pdfplumber.open(bio) as pdf:
                pages = []
                full_parts = []
                for idx, page in enumerate(pdf.pages, start=1):
                    page_text = ''.join(line['text'] + '\n' for line in plumber_lines(page))
                    pages.append({"page_num": idx, "text": page_text})
                    full_parts.append(page_text)
            text = ''.join(full_parts) or None
            text_len = len(text.strip()) if text else 0
            print(f"[VECTORIZE][READ] PDF解析完成: pages={len(pages)} text_len={text_len} path={relative}")
            return {"text": text, "pages": pages}
        except Exception as exc:
            print(f"[VECTORIZE][READ] PDF解析异常: path={relative} error={type(exc).__name__}: {exc}")
            return None

    if ext == 'txt':
        # 纯文本，尝试常见编码
        for enc in ('utf-8', 'gbk', 'latin-1'):
            try:
                text = content.decode(enc)
                print(f"[VECTORIZE][READ] TXT解析完成: encoding={enc} text_len={len(text.strip())} path={relative}")
                return text
            except Exception:
                continue
        print(f"[VECTORIZE][READ] TXT解析失败: 无法识别编码 path={relative}")
        return None

    if ext == 'docx':
        # 用 python-docx 解析
        try:
            bio = BytesIO(content)
            doc = Document(bio)
            text = '\n'.join(p.text for p in doc.paragraphs) or None
            text_len = len(text.strip()) if text else 0
            print(f"[VECTORIZE][READ] DOCX解析完成: paragraphs={len(doc.paragraphs)} text_len={text_len} path={relative}")
            return text
        except Exception as exc:
            print(f"[VECTORIZE][READ] DOCX解析异常: path={relative} error={type(exc).__name__}: {exc}")
            return None

    print(f"[VECTORIZE][READ] 不支持的文件类型: ext={ext or '?'} path={relative}")
    return None


def _blocks_to_chunks(blocks: list[dict], file_name: str, chunk_size: int = 512,
                      page_offset: int = 0, max_chunk_size: int = 768,
                      chunk_overlap: int = 64, min_chunk_size: int = 128) -> list[dict]:
    """单篇与批量索引共用 embedding token 预算，保留原文定位。"""
    from app.utils.document_chunking import blocks_to_chunks

    return blocks_to_chunks(
        blocks, file_name, chunk_size, page_offset,
        max_chunk_size, chunk_overlap, min_chunk_size,
    )


def _page_needs_ocr(text, images, width, height):
    """无文字的图片页，或大图配少量页眉文字的页面，需要 OCR。"""
    if not isinstance(images, (list, tuple)) or not images:
        return False
    if not text.strip():
        return True
    return len(text.strip()) < 80 and any(
        (box[2] - box[0]) * (box[3] - box[1]) >= width * height * 0.5
        for box in images
    )


def _complete_pdf_ocr(local_path, blocks, ocr_pages, page_count):
    """补齐需要 OCR 的原页，保留未识别出文字页面的原生文字。

    Args:
        local_path: 当前 PDF 本地路径。
        blocks: 原生解析得到的文字块。
        ocr_pages: 需 OCR 的零基页索引；全篇无文字且无候选时改为全部页。
        page_count: 原 PDF 页数。

    Returns:
        按页排序、block_id 连续的文字块；只有 OCR 返回文字的页替换原生块。

    Raises:
        RuntimeError: 合并后仍然没有任何可索引文字。
    """
    if not blocks and not ocr_pages:
        ocr_pages = list(range(page_count))
    if ocr_pages:
        from app.utils.qwen_ocr_utils import read_pdf_pages

        recognized = read_pdf_pages(local_path, ocr_pages)
        # 仅替换实际返回 OCR 文字的页。少量原生文字（如插图页标题）
        # 可能触发扫描页检测，OCR 无文字时仍保留这部分真实文字。
        pages = {b["page_idx"] for b in recognized}
        blocks = [b for b in blocks if b["page_idx"] not in pages] + recognized
        blocks.sort(key=lambda b: b["page_idx"])
        for index, block in enumerate(blocks):
            block["block_id"] = index
    if not blocks:
        raise RuntimeError("PDF 没有可索引文字")
    return blocks


def _read_pdf_blocks_with_pymupdf(local_path: str) -> list[dict]:
    """为 pdfminer 无法打开的 PDF 提取真实文字行，保持现有坐标契约。

    Args:
        local_path: 原始 PDF 的本地路径；图片页交由 qwen-vl-ocr 补充识别。

    Returns:
        携带 0-based 页码和 PDF 左下原点坐标的文字行。

    Raises:
        RuntimeError: 备用解析器也未找到可提取的文字层。
    """
    import fitz

    blocks, ocr_pages = [], []
    page_count = 0
    with fitz.open(local_path) as pdf:
        for page_idx, page in enumerate(pdf):
            page_count = page_idx + 1
            page_start = len(blocks)
            raw_blocks = page.get_text("dict")["blocks"]
            words = [{"x0": w[0], "top": w[1], "x1": w[2], "bottom": w[3], "text": w[4]}
                     for w in page.get_text("words")]
            tables = [table.bbox for table in page.find_tables().tables]
            for line in words_to_lines(words, tables):
                blocks.append({
                    "text": line["text"] + "\n", "page_idx": page_idx,
                    "bbox": [line["x0"], page.rect.height - line["bottom"],
                             line["x1"], page.rect.height - line["top"]],
                    "block_id": len(blocks),
                })
            images = [b["bbox"] for b in raw_blocks if b.get("type") == 1]
            if _page_needs_ocr("".join(b["text"] for b in blocks[page_start:]),
                               images, page.rect.width, page.rect.height):
                ocr_pages.append(page_idx)
    return _complete_pdf_ocr(local_path, blocks, ocr_pages, page_count)


def _read_blocks_locally(file_path: str, base_dir: str = None) -> list[dict]:
    """按真实扩展名解析本地附件，保留 PDF 的原始页码和行坐标。

    Args:
        file_path: 绝对路径或相对于挂载目录的原文件路径。
        base_dir: 可选挂载根目录，缺省读取 INDEX_FILE_BASE_DIR。

    Returns:
        原文文字块列表；DOCX、TXT 使用空页码与坐标，不伪造页面位置。

    Raises:
        RuntimeError: TXT 编码无法识别、格式不支持或 PDF 无可索引文字。
    """
    # 1. 解析真实本地路径和扩展名，不把普通文件名中的标点当 URL 参数
    local_base = base_dir or os.getenv("INDEX_FILE_BASE_DIR", "/foreign-files")
    local_path = file_path if os.path.isabs(file_path) else os.path.join(local_base, file_path)
    extension = os.path.splitext(local_path)[1].lower()

    blocks = []
    # 2. PDF 优先原生行解析，失败切换备用解析器，按页判定 OCR 并统一补齐
    if extension == ".pdf":
        try:
            document = pdfplumber.open(local_path)
        except Exception:
            # 某些有效 PDF 的目录结构不被 pdfminer 接受；备用解析同样检测扫描页。
            logging.getLogger(__name__).warning("PDF 主解析器无法打开原文，改用 PyMuPDF")
            return _read_pdf_blocks_with_pymupdf(local_path)
        ocr_pages = []
        with document as pdf:
            page_count = len(pdf.pages)
            for page_idx, page in enumerate(pdf.pages):
                page_start = len(blocks)
                for line in plumber_lines(page):
                    line_text = (line.get("text") or "").strip()
                    if not line_text:
                        continue
                    blocks.append({
                        "text": line_text + "\n",
                        "page_idx": page_idx,
                        "bbox": [line["x0"], page.height - line["bottom"],
                                 line["x1"], page.height - line["top"]],
                        "block_id": len(blocks),
                    })
                images = ([[im["x0"], im["top"], im["x1"], im["bottom"]]
                           for im in page.images]
                          if isinstance(page.images, (list, tuple)) else [])
                if _page_needs_ocr("".join(b["text"] for b in blocks[page_start:]),
                                   images, page.width, page.height):
                    ocr_pages.append(page_idx)
        return _complete_pdf_ocr(local_path, blocks, ocr_pages, page_count)

    # 3. 文本格式按段落或解码行保留顺序，不伪造不存在的 PDF 页码或坐标
    if extension == ".docx":
        lines = [paragraph.text for paragraph in Document(local_path).paragraphs]
    elif extension == ".txt":
        with open(local_path, "rb") as source:
            raw = source.read()
        for encoding in ("utf-8-sig", "utf-16", "gb18030"):
            try:
                lines = raw.decode(encoding).splitlines()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise RuntimeError("TXT 编码无法识别")
    else:
        raise RuntimeError(f"不支持索引附件格式: {extension or '无扩展名'}")

    for line in lines:
        if line.strip():
            blocks.append({"text": line.strip() + "\n", "page_idx": None,
                           "bbox": None, "block_id": len(blocks)})
    return blocks


def _read_index_blocks(file_path: str, base_dir: str, corpus_id: int) -> list[dict]:
    """优先读取本地原文，缺少挂载时从 Foreign 文件服务临时下载解析。

    Args:
        file_path: 数据库中的附件路径或批量脚本已下载的本地路径。
        base_dir: 本地附件目录。
        corpus_id: 文献 ID，用于调用方的索引任务上下文。

    Returns:
        带页码的原文块；临时下载文件在解析后立即清理。
    """
    local_base = base_dir or os.getenv("INDEX_FILE_BASE_DIR", "/foreign-files")
    local_path = file_path if os.path.isabs(file_path) else os.path.join(local_base, file_path)
    if os.path.isfile(local_path):
        return _read_blocks_locally(file_path, base_dir)

    from flask import current_app

    file_url = current_app.config.get("FILE_URL")
    if not file_url:
        raise RuntimeError("Foreign 文件服务地址未配置，无法读取索引原文")
    suffix = os.path.splitext(_attachment_path(file_path))[1] or ".pdf"
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            with requests.get(f"{file_url.rstrip('/')}/files",
                              params={"filename": file_path}, stream=True,
                              timeout=(10, 120)) as response:
                response.raise_for_status()
                for data in response.iter_content(chunk_size=1024 * 1024):
                    temp_file.write(data)
        return _read_blocks_locally(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def _chunk_text(full_text: str, file_name: str, pages=None, chunk_size=500, chunk_overlap=100) -> list[dict]:
    """
    使用 langchain RecursiveCharacterTextSplitter 将长文本切分为重叠的短片段。

    切分策略（按优先级尝试分割）：
        段落间（双换行） → 单换行 → 中文句号/感叹号/问号 → 空格 → 单字符

    Args:
        full_text: 完整文件文本
        file_name: 文件名（用于 chunk_id 命名）
        pages: PDF 解析后的页边界列表 [{"page_num": int, "text": str}, ...]，
              为 None 时（如 txt/docx）则不记录页码
        chunk_size: 每个 chunk 最大字符数（默认 500）
        chunk_overlap: 相邻 chunk 重叠字符数（默认 100，防止语义截断）

    Returns:
        list[dict]: 每个 chunk 一项，包含：
            - text: chunk 文本内容
            - metadata: {
                "file_name", "chunk_index",          # 标识
                "char_start", "char_end",            # 在原文中的字符位置（用于高亮定位）
                "page_start", "page_end",           # PDF 页码（PDF 时有效，其余为 None）
                "chunk_id"                          # 唯一 ID = f"{file_name}_{chunk_index}"
              }
    """
    # 构建字符偏移到页码的映射（仅 PDF 时有效）
    page_map = None
    if pages:
        page_map = []
        cursor = 0
        for p in pages:
            page_map.append({"page_num": p["page_num"], "char_start": cursor})
            cursor += len(p["text"]) + 1  # +1 模拟 join('\n') 时的换行符
        # 页码区间的闭区间映射：char -> page（每个字符落在哪一页）
        page_map_len = len(page_map)

    def char_to_page(char_pos):
        """给定字符偏移，返回对应的页码（1-based）。"""
        if page_map is None:
            return None
        for i in range(len(page_map) - 1):
            if page_map[i]["char_start"] <= char_pos < page_map[i + 1]["char_start"]:
                return page_map[i]["page_num"]
        return page_map[-1]["page_num"] if page_map else None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # 按优先级尝试在何处断开（更粗粒度优先，细粒度兜底）
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    result = []
    cursor = 0
    for i, chunk_text_content in enumerate(chunks):
        # 在原文中定位这个 chunk 的字符区间
        # 顺序扫描找到第一个匹配位置（overlap 可能导致重复，使用 cursor 避免错位）
        start = full_text.find(chunk_text_content, cursor)
        if start == -1:
            # 极端情况：overlap 区域文本差异导致 find 失败，用近似偏移
            start = cursor
        end = start + len(chunk_text_content)

        result.append({
            "text": chunk_text_content,
            "metadata": {
                "file_name": file_name,
                "chunk_index": i,
                "char_start": start,
                "char_end": end,
                "page_start": char_to_page(start),
                "page_end": char_to_page(end - 1),  # -1 保证 end 是闭区间
                "chunk_id": f"{file_name}_{i}",
            },
        })
        cursor = end  # 下一次搜索从当前位置继续

    return result


def index_corpus_chunks(corpus_id, attachments, corpus_meta=None, base_dir: str = None,
                        mark_vectorized: bool = True, expected_generation=None) -> dict:
    """兼容旧入口，统一交由双路索引服务执行加锁、重建与状态发布。"""
    from app.indexing.service import build_indexes

    return build_indexes(corpus_id, attachments, corpus_meta, base_dir, mark_vectorized,
                         expected_generation)


def _index_corpus_chunks(corpus_id, attachments, corpus_meta=None, base_dir: str = None,
                        mark_vectorized: bool = False, index_context=None) -> dict:
    """从同一组原文切块生成向量及词法文档，不独立发布文献就绪状态。

    Args:
        corpus_id: 当前文献 ID。
        attachments: 原附件或临时分片字典，source_file_path 保留真实原件引用。
        corpus_meta: 可选题录，缺失时读取当前文献补齐。
        base_dir: 本地附件挂载根目录。
        mark_vectorized: 此内部入口必须为 False，状态由外层双路服务统一发布。
        index_context: 外层提供的构建令牌和附件指纹，写入两路相同载荷。

    Returns:
        成功处理的附件数 indexed 和切块总数 total_chunks。

    Raises:
        ValueError: 未经外层服务调用，或附件和切块数量超出既有 point ID 编码范围。
        RuntimeError: 原文或分块为空，或向量数量与切块不一致。
    """
    from app.models import db, Corpus, CorpusCategory
    from app.indexing.opensearch_store import get_lexical_store

    # 1. 拒绝脱离双路服务的直接发布，固定当前构建上下文及题录
    if not index_context or mark_vectorized:
        raise ValueError("双路写入必须由索引服务提供构建版本并统一发布")

    client = get_qdrant_client()
    indexed = 0
    total_chunks = 0

    # 补全文献级元数据（若未传入则初始化空字典，后续查库填充）
    if corpus_meta is None:
        corpus_meta = {}
    if "corpus_id" not in corpus_meta:
        corpus_meta["corpus_id"] = corpus_id

    # 若元数据不完整（缺少 title_cn），查库补全
    if not corpus_meta.get("title_cn"):
        row = db.session.query(
            Corpus.title_cn, Corpus.title_en, Corpus.author,
            Corpus.publish_time, CorpusCategory.category_name
        ).outerjoin(
            CorpusCategory, Corpus.category_id == CorpusCategory.id
        ).filter(Corpus.id == corpus_id).first()
        if row:
            corpus_meta["title_cn"] = row.title_cn
            corpus_meta["title_en"] = row.title_en
            corpus_meta["author"] = row.author
            corpus_meta["publish_time"] = row.publish_time
            corpus_meta["category_name"] = row.category_name

    # 2. 按附件顺序解析、按 token 分块并生成向量，保留原件引用和页偏移
    for att_idx, att in enumerate(attachments):
        file_path = att.get('file_path')
        if not file_path:
            print(f"[VECTORIZE][SKIP] corpus_id={corpus_id} attachment_index={att_idx} 原因=缺少file_path")
            continue
        file_name = (att.get('file_name') or att.get('original_filename')
                     or _attachment_path(file_path).rsplit('/', 1)[-1] or 'unknown')
        chunk_name = att.get('chunk_name') or file_name
        page_offset = int(att.get('page_offset') or 0)

        # 临时解析路径只用于读文件；引用必须指向 corpus.attachment 里的原路径。
        stored_file_path = att.get('source_file_path') or _normalize_file_path(file_path, base_dir)

        # ① 本地原文解析 → blocks（PDF 含 bbox + page_idx）
        try:
            blocks = _read_index_blocks(file_path, base_dir, corpus_id)
        except Exception as exc:
            print(f"[VECTORIZE][PARSE-ERROR] corpus_id={corpus_id} file={file_name} → {type(exc).__name__}: {exc}")
            raise  # 不兜底，直接报错不入库

        if not blocks:
            raise RuntimeError(f"附件无可索引文字: {file_name}")
        print(f"[VECTORIZE][PARSE] corpus_id={corpus_id} file={file_name} blocks={len(blocks)}")

        # ② block → chunk（累积文本，携带 bbox_blocks）
        chunks = _blocks_to_chunks(blocks, chunk_name, page_offset=page_offset)
        if not chunks:
            raise RuntimeError(f"附件分块结果为空: {file_name}")
        if att_idx >= 100 or len(chunks) > 10000:
            raise ValueError("附件或切块数量超出既有 point ID 编码范围")

        print(f"[VECTORIZE][CHUNK] corpus_id={corpus_id} file={file_name} chunk_count={len(chunks)} first_chunk_len={len(chunks[0]['text']) if chunks else 0}")

        # ③ 向量化
        texts = [c["text"] for c in chunks]
        embeddings = _get_embedding(texts)
        if len(embeddings) != len(chunks):
            raise RuntimeError("向量数量与切块数量不一致")
        print(f"[VECTORIZE][EMBED] corpus_id={corpus_id} file={file_name} embedding_count={len(embeddings)}")

        # ④ 构建 Qdrant point 列表
        # 3. 为每块生成稳定 point ID 与共享版本载荷，正文哈希供两路召回交叉核验
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # 相邻 chunk 链：便于前端在上下文中展示前后段落
            prev_chunk_id = f"{chunk_name}_{i - 1}" if i > 0 else None
            next_chunk_id = f"{chunk_name}_{i + 1}" if i < len(chunks) - 1 else None

            file_ext = os.path.splitext(file_name)[-1].lower().lstrip('.')

            # locator_text：截断到 300 字符，用于前端直接展示（避免返回过长文本）
            locator_text = chunk["text"]
            if len(locator_text) > 300:
                locator_text = locator_text[:300] + "..."

            point_id = int(corpus_id) * 1_000_000 + att_idx * 10_000 + i
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        **index_context,
                        "content_hash": hashlib.sha256(chunk["text"].encode()).hexdigest(),
                        "embedding_model": embedding_config().model,
                        "embedding_dimension": VECTOR_SIZE,
                        "embedding_text_type": "document",
                        "ocr_models": sorted({b["ocr_model"] for b in blocks if b.get("ocr_model")}),
                        # 文献级元数据
                        "corpus_id": corpus_id,
                        "title_cn": corpus_meta.get("title_cn") or "",
                        "title_en": corpus_meta.get("title_en") or "",
                        "author": corpus_meta.get("author") or "",
                        "publish_time": corpus_meta.get("publish_time"),
                        "category_name": corpus_meta.get("category_name") or "",
                        # 文件级信息
                        "file_name": file_name,
                        "file_path": stored_file_path,
                        "file_ext": file_ext,
                        "attachment_index": att_idx,
                        # chunk 级定位
                        "chunk_id": chunk["metadata"]["chunk_id"],
                        "chunk_index": chunk["metadata"]["chunk_index"],
                        "text": chunk["text"],           # 完整 chunk 文本（检索结果用）
                        "chunk_char_start": chunk["metadata"]["char_start"],
                        "chunk_char_end": chunk["metadata"]["char_end"],
                        "chunk_length": chunk["metadata"]["char_end"] - chunk["metadata"]["char_start"],
                        "token_count": chunk["metadata"]["token_count"],
                        "token_count_kind": "chunking_estimate",
                        "tokenizer": chunk["metadata"]["tokenizer"],
                        "chunking_strategy": chunk["metadata"]["chunking_strategy"],
                        "page_start": chunk["metadata"].get("page_start"),
                        "page_end": chunk["metadata"].get("page_end"),
                        "prev_chunk_id": prev_chunk_id,
                        "next_chunk_id": next_chunk_id,
                        # PDF 行坐标高亮字段
                        "bbox_blocks": chunk["metadata"]["bbox_blocks"],
                        "page_idx_set": list({b["page_idx"] for b in chunk["metadata"]["bbox_blocks"]}),
                        # 前端高亮展示
                        "locator_text": locator_text,
                        # 版本标识（payload 结构升级时用于全量重建）
                        "version": PAYLOAD_VERSION,
                    },
                )
            )

        # ⑤ 批量写入 Qdrant（每批最多 100 条，避免单次请求过大）
        # 4. 每批先写向量再写词法；失败向外传播，由持锁服务清理残留并决定任务状态
        batch_size = 100
        for b in range(0, len(points), batch_size):
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[b:b + batch_size],
                wait=True,
            )
            get_lexical_store().write_points(points[b:b + batch_size])

        total_chunks += len(chunks)
        indexed += 1

    # ⑥ 返回统计，由外层服务复核原文版本后发布两路就绪状态。
    return {"indexed": indexed, "total_chunks": total_chunks}


def search_chunks(query: str, corpus_ids=None, top_k: int = 5) -> list[dict]:
    """
    用问题向量从 Qdrant 检索最相关的原文片段（余弦相似度）。

    流程：
        1. 将用户问题向量化（1024 维）
        2. 构造可选的 corpus_id 过滤条件
        3. 调用 Qdrant query_points，并从数据库复核审核及索引状态
        4. 整理通过复核的前 top_k 个 chunk 的完整引用信息

    Args:
        query: 用户问题文本
        corpus_ids: 可选，限定只在这些文献 ID 中检索
        top_k: 返回的 chunk 数量上限

    Returns:
        list[dict]: 每项包含 text / corpus_id / file_name / score / chunk_id /
                     chunk_char_start / chunk_char_end / locator_text /
                     prev_chunk_id / next_chunk_id / title_cn / title_en /
                     author / publish_time / category_name / file_path 等完整引用信息
    """
    # 1. 使用 query 角色生成查询向量，并按当前模型、维度及可选文献范围召回候选
    client = get_qdrant_client()

    # encode 查询文本
    query_vector = _get_embedding([query], text_type="query")[0]

    # 构造 corpus_id 过滤条件（若指定了文献范围）
    conditions = [
        models.FieldCondition(key="embedding_model", match=models.MatchValue(value=embedding_config().model)),
        models.FieldCondition(key="embedding_dimension", match=models.MatchValue(value=VECTOR_SIZE)),
    ]
    if corpus_ids:
        conditions.append(models.FieldCondition(key="corpus_id", match=models.MatchAny(any=corpus_ids)))
    filter_cond = models.Filter(must=conditions, must_not=[
        models.FieldCondition(key="personal_upload", match=models.MatchValue(value=True)),
    ])

    # 余弦相似度检索
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=filter_cond,
        limit=min(max(top_k * 5, top_k), 200),
        with_payload=True,
        score_threshold=0.0,  # 不过滤，允许返回低分结果
    )
    # 共享集合可能含旧片段；公开检索只返回当前已审核且索引有效的文献。
    from app.models import Corpus, db

    hit_ids = set()
    for hit in response.points:
        try:
            hit_ids.add(int((hit.payload or {}).get("corpus_id")))
        except (TypeError, ValueError):
            continue
    if not hit_ids:
        return []
    # 2. 回库复核当前审核、删除和就绪标记，不能只相信向量载荷的历史状态
    approved_ids = {
        row[0] for row in db.session.query(Corpus.id).filter(
            Corpus.id.in_(hit_ids), Corpus.status == 2, Corpus.owner_id.is_(None),
            Corpus.is_deleted == 0, Corpus.is_vectorized == True,
        ).all()
    }
    # 3. 按原召回顺序保留有效候选，取 top_k 并投影真实页框和字符偏移
    results = []
    for hit in response.points:
        try:
            hit_corpus_id = int((hit.payload or {}).get("corpus_id"))
        except (TypeError, ValueError):
            continue
        if hit_corpus_id in approved_ids:
            results.append(hit)
        if len(results) >= top_k:
            break

    return [
        {
            # 核心定位字段
            "text": hit.payload.get("text", ""),
            "corpus_id": hit.payload.get("corpus_id"),
            "file_name": hit.payload.get("file_name", ""),
            "file_path": hit.payload.get("file_path", ""),
            "file_ext": hit.payload.get("file_ext", ""),
            "score": hit.score,
            # chunk 定位
            "chunk_id": hit.payload.get("chunk_id", ""),
            "chunk_index": hit.payload.get("chunk_index"),
            "chunk_char_start": hit.payload.get("chunk_char_start"),
            "chunk_char_end": hit.payload.get("chunk_char_end"),
            "chunk_length": hit.payload.get("chunk_length"),
            "page_start": hit.payload.get("page_start"),
            "page_end": hit.payload.get("page_end"),
            "prev_chunk_id": hit.payload.get("prev_chunk_id"),
            "next_chunk_id": hit.payload.get("next_chunk_id"),
            # PDF 行坐标高亮字段
            "bbox_blocks": hit.payload.get("bbox_blocks", []),
            "page_idx_set": hit.payload.get("page_idx_set", []),
            # 高亮展示
            "locator_text": hit.payload.get("locator_text", ""),
            # 文献元数据
            "title_cn": hit.payload.get("title_cn", ""),
            "title_en": hit.payload.get("title_en", ""),
            "author": hit.payload.get("author", ""),
            "publish_time": hit.payload.get("publish_time"),
            "category_name": hit.payload.get("category_name", ""),
            # 版本
            "version": hit.payload.get("version", ""),
        }
        for hit in results
    ]


def search_literature_by_question(query: str, top_k_chunks: int = 30) -> list[dict]:
    """
    RAG 文献检索入口：将用户问题向量化后从 Qdrant 检索，
    按 corpus_id 聚合为知网式文献列表（每篇保留 best_score + 所有命中片段）。

    Args:
        query: 用户问题文本
        top_k_chunks: 从 Qdrant 拉取的总 chunk 上限（默认 30，返回约 5-15 篇不重复文献）

    Returns:
        list[dict]: 按 best_score 降序排列的文献列表，每项结构：
            {
                "corpus_id", "title_cn", "title_en", "author", "publish_time",
                "category_name", "source", "volume", "issue",
                "abstract_cn", "abstract_en", "keyword_cn", "keyword_en",
                "file_path", "file_name", "file_ext",
                "best_score": float,               # 该文献下最高 chunk 分数
                "matched_chunks": [                 # 该文献下所有命中片段
                    {
                        "chunk_id", "score", "text",
                        "chunk_char_start", "chunk_char_end",
                        "page_start", "page_end",
                        "bbox_blocks", "page_idx_set", "locator_text",
                    },
                    ...
                ]
            }
    """
    # 1. 复用已校验的片段召回结果，无命中时不查询题录
    chunks = search_chunks(query, top_k=top_k_chunks)

    if not chunks:
        return []

    # 按 corpus_id 聚合，同时收集所有文献 ID 一次性查库补全元数据
    corpus_ids = list({ch.get("corpus_id") for ch in chunks if ch.get("corpus_id") is not None})

    # 从 corpus 表补全 Qdrant payload 中缺失的字段，未找到时仅打日志，不阻断主流程
    # 2. 批量补全数据库题录，查询失败只使用已有载荷，不伪造缺失元数据
    db_meta = {}
    try:
        from app.models import db, Corpus, CorpusCategory
        rows = (
            db.session.query(Corpus, CorpusCategory.category_name)
            .outerjoin(CorpusCategory, Corpus.category_id == CorpusCategory.id)
            .filter(Corpus.id.in_(corpus_ids))
            .all()
        )
        db_meta = {
            r.Corpus.id: {
                "source": r.Corpus.source or "",
                "volume": r.Corpus.volume,
                "issue": r.Corpus.issue,
                "abstract_cn": r.Corpus.abstract_cn or "",
                "abstract_en": r.Corpus.abstract_en or "",
                "keyword_cn": r.Corpus.keyword_cn or "",
                "keyword_en": r.Corpus.keyword_en or "",
                "category_name": r.CorpusCategory.category_name or "",
            }
            for r in rows
        }
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(
            f"[search_literature_by_question] 数据库元数据查询失败，"
            f"仅使用 Qdrant payload 补全（corpus_ids={corpus_ids[:5]}...）: {e}"
        )

    # 3. 按文献聚合片段并以最高片段分排序，保留每块定位供引用展示
    bucket: dict[int, dict] = {}
    for ch in chunks:
        cid = ch.get("corpus_id")
        if cid is None:
            continue
        extra = db_meta.get(cid, {})
        if cid not in bucket:
            bucket[cid] = {
                "corpus_id": cid,
                "title_cn": ch.get("title_cn"),
                "title_en": ch.get("title_en"),
                "author": ch.get("author"),
                "publish_time": ch.get("publish_time"),
                "category_name": ch.get("category_name") or extra.get("category_name"),
                "source": extra.get("source"),
                "volume": extra.get("volume"),
                "issue": extra.get("issue"),
                "abstract_cn": extra.get("abstract_cn"),
                "abstract_en": extra.get("abstract_en"),
                "keyword_cn": extra.get("keyword_cn"),
                "keyword_en": extra.get("keyword_en"),
                "file_path": ch.get("file_path"),
                "file_name": ch.get("file_name"),
                "file_ext": ch.get("file_ext"),
                "best_score": ch.get("score", 0),
                "matched_chunks": [],
            }
        item = bucket[cid]
        item["matched_chunks"].append({
            "chunk_id": ch.get("chunk_id"),
            "score": ch.get("score"),
            "text": ch.get("text"),
            "chunk_char_start": ch.get("chunk_char_start"),
            "chunk_char_end": ch.get("chunk_char_end"),
            "page_start": ch.get("page_start"),
            "page_end": ch.get("page_end"),
            "bbox_blocks": ch.get("bbox_blocks", []),
            "page_idx_set": ch.get("page_idx_set", []),
            "locator_text": ch.get("locator_text", ""),
        })
        if ch.get("score", 0) > item["best_score"]:
            item["best_score"] = ch.get("score", 0)

    return sorted(bucket.values(), key=lambda x: x["best_score"], reverse=True)


def delete_corpus_chunks(corpus_id, file_names=None):
    """兼容旧删除入口，同时撤销状态并清理 Qdrant 与 OpenSearch。"""
    from app.indexing.service import delete_indexes

    return delete_indexes(corpus_id, file_names)


def _delete_qdrant_chunks(corpus_id, file_names=None):
    """
    从 Qdrant 删除指定文献的 chunk。

    Args:
        corpus_id: 文献 ID（必须，作为过滤条件）
        file_names: 可选，仅删除指定文件名对应的 chunk；
                    为 None 或空时删除整篇文献的所有 chunk
    """
    client = get_qdrant_client()

    conditions = [
        # 匹配 corpus_id
        models.FieldCondition(
            key="corpus_id",
            match=models.MatchValue(value=corpus_id),
        )
    ]
    # 若指定了文件名，叠加文件名校验（防止误删同名文件的不同文献）
    if file_names:
        conditions.append(
            models.FieldCondition(
                key="file_name",
                match=models.MatchAny(any=file_names),
            )
        )

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=conditions)
        ),
        wait=True,
    )


def get_source_snippet(file_path, char_start, char_end, window=200) -> dict:
    """
    根据文件路径和字符区间取原文片段，供前端高亮展示。

    Args:
        file_path: 附件文件路径
        char_start: chunk 在原文中的字符起始位置
        char_end: chunk 在原文中的字符结束位置
        window: 上下文窗口字符数（前后各扩展 window 个字符，丰富展示上下文）

    Returns:
        dict: {
            "full_text_available": bool,           # 文件是否可读
            "snippet_text": str,                   # 带上下文的片段文本
            "highlight_start": int,                # 高亮区在 snippet_text 中的起始位置
            "highlight_end": int,                  # 高亮区在 snippet_text 中的结束位置
            "char_start": int,                     # 原始 chunk 起始（不受 window 影响）
            "char_end": int,                       # 原始 chunk 结束
            "file_name": str,                      # 文件名
            "text": str,                           # 纯高亮文本（原 chunk 内容）
        }
    """
    text = _read_file_text(file_path)
    if isinstance(text, dict):
        text = text.get("text")
    if not text:
        return {
            "full_text_available": False,
            "snippet_text": "",
            "highlight_start": 0,
            "highlight_end": 0,
            "char_start": char_start,
            "char_end": char_end,
            "file_name": os.path.basename(file_path),
            "text": "",
        }

    # 扩展上下文窗口
    ctx_start = max(0, char_start - window)
    ctx_end = min(len(text), char_end + window)

    snippet_text = text[ctx_start:ctx_end]
    # 高亮区域在 snippet 中的相对偏移
    highlight_start = char_start - ctx_start
    highlight_end = char_end - ctx_start

    return {
        "full_text_available": True,
        "snippet_text": snippet_text,
        "highlight_start": highlight_start,
        "highlight_end": highlight_end,
        "char_start": char_start,
        "char_end": char_end,
        "file_name": os.path.basename(file_path),
        "text": text[char_start:char_end],
    }


def count_collection_points() -> dict:
    """
    返回 Qdrant collection 中当前 point 总数及状态，
    用于判断全量重建索引前的存量规模。
    """
    client = get_qdrant_client()
    try:
        info = client.get_collection(collection_name=COLLECTION_NAME)
        return {
            "points_count": info.points_count,
            "status": info.status,
        }
    except Exception:
        return {"points_count": 0, "status": "unknown", "error": str(Exception)}
