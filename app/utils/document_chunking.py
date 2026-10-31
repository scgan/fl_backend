"""按 embedding token 预算切分原文，保留自然边界和原始定位信息。"""

from bisect import bisect_left, bisect_right
import re
from app.utils.embedding_tokens import count_tokens, get_tokenizer, TOKENIZER_NAME, INPUT_OVERHEAD


def blocks_to_chunks(blocks, file_name, chunk_size=512, page_offset=0,
                     max_chunk_size=768, chunk_overlap=64, min_chunk_size=128):
    """按本地 token 估算预算切分原文，优先自然边界并保留真实定位及重叠。

    Args:
        blocks: 按阅读顺序排列的原文块，可含真实 page_idx、bbox 及 OCR 精度。
        file_name: 原件或分片显示名，用于构造本组 chunk_id。
        chunk_size: 目标 token 预算，包含接口开销，不是字符数或实际计费量。
        page_offset: PDF 分片相对原附件的零基页偏移。
        max_chunk_size: 切块 token 估算硬上限。
        chunk_overlap: 相邻块的目标重叠预算，必须小于目标预算。
        min_chunk_size: 优先避免过短切块的预算阈值。

    Returns:
        正文及 metadata 字典列表，字符偏移始终指向同一拼接原文，包含重叠区。
        PDF 被切开的文字行仍保留其完整真实 bbox，不伪造字符级或章节位置。

    Raises:
        ValueError: 预算关系无效，或预算不足以容纳接口开销及一个完整 Unicode 字符。
    """
    if not 0 <= chunk_overlap < chunk_size <= max_chunk_size:
        raise ValueError("必须满足 0 <= overlap < target <= maximum")
    if not 0 < min_chunk_size <= chunk_size:
        raise ValueError("最短块大小必须介于 1 与目标大小之间")
    if chunk_size <= INPUT_OVERHEAD:
        raise ValueError("目标预算必须容纳正文和接口开销")

    # 1. 校验预算后拼接原文，逐块记录字符区间及真实页框，所有偏移共用此文本基准
    parts, locations, starts, ends = [], [], [], []
    cursor = 0
    for block in blocks:
        value = block.get("text") or ""
        if not value:
            continue
        parts.append(value)
        starts.append(cursor)
        cursor += len(value)
        ends.append(cursor)
        page = block.get("page_idx")
        bbox = block.get("bbox")
        locations.append(
            {"page_idx": page + page_offset, "bbox": list(bbox),
             "block_id": block.get("block_id"),
             **({"location_precision": block["location_precision"], "ocr_model": block.get("ocr_model")}
                if block.get("location_precision") else {})}
            if isinstance(page, int) and isinstance(bbox, (list, tuple))
            and len(bbox) == 4 else None
        )
    text = "".join(parts)
    if not text.strip():
        return []

    # 句尾覆盖常见拉丁文、东亚文字、阿拉伯文和天城文标点。
    # 单个换行可能只是 PDF 折行，优先级低于句尾和空行。
    # 2. 建立自然边界候选；PDF 折行不是段落，DOCX/TXT 段落可作为优先切口
    natural = [m.end() for m in re.finditer(
        r'(?:[。！？!?؟।॥][”’"\x27»）)]*\s*|[.][”’"\x27»）)]*(?:\s+|$)|\n[ \t]*\n)', text
    )]
    paragraphs = [m.end() for m in re.finditer(r'\n[ \t]*\n', text)]
    # DOCX/TXT 的 block 是段落；PDF block 只是折行，不能当段落边界。
    paragraphs = sorted(set(paragraphs + [ends[i] for i, loc in enumerate(locations)
                                         if loc is None]))

    def budget_end(start, budget, stop=None):
        """用 token 原文偏移找窗口端点，再重新编码片段核验硬上限。

        BPE 在切口处可能重新组合，不能以全文 token 前缀差代替片段计数。
        偏移按 Unicode 字符处理，永不 decode 截断 token 造成乱码。
        """
        stop = len(text) if stop is None else stop
        window = min(stop, start + max(32, budget * 8))
        while True:
            encoded = get_tokenizer().encode(text[start:window], add_special_tokens=False)
            if len(encoded.ids) + INPUT_OVERHEAD > budget or window == stop:
                break
            window = min(stop, start + (window - start) * 2)
        if len(encoded.ids) + INPUT_OVERHEAD <= budget:
            return window
        end = start + encoded.offsets[budget - INPUT_OVERHEAD - 1][1]
        while end > start and count_tokens(text[start:end]) > budget:
            end -= 1
        if end <= start:
            raise ValueError("预算不足以容纳一个完整 Unicode 字符")
        return end

    def nearest(boundaries, low, high, target, start):
        """从指定区间内选离目标最近的边界，避免遍历全文。"""
        left, right = bisect_left(boundaries, low), bisect_right(boundaries, high)
        if left == right:
            return None
        pivot = bisect_left(boundaries, target, left, right)
        candidates = [value for value in boundaries[max(left, pivot - 1):min(right, pivot + 1)]
                      if count_tokens(text[start:value]) <= max_chunk_size]
        if not candidates:
            return None
        return min(candidates, key=lambda value: (abs(value - target), value))

    # 3. 逐窗验证 token 硬上限并选自然切口，映射真实定位后保留重叠向前推进
    chunks = []
    start = 0
    while start < len(text):
        limit = budget_end(start, max_chunk_size)
        target = budget_end(start, chunk_size)
        if limit == len(text) and count_tokens(text[start:]) < chunk_size + min_chunk_size:
            end = len(text)
        else:
            low = budget_end(start, max(INPUT_OVERHEAD + 1, min_chunk_size, chunk_size * 2 // 3))
            end = (nearest(natural, low, limit, target, start)
                   or nearest(paragraphs, low, limit, target, start) or target)
            # 尾块过短时在硬上限允许的情况下合并，保留自然边界。
            if end < len(text) and budget_end(end, max(2, min_chunk_size)) == len(text):
                if limit == len(text):
                    end = len(text)
        value = text[start:end]
        if value.strip():
            first = bisect_right(ends, start)
            last = bisect_left(starts, end)
            bboxes = [dict(item) for item in locations[first:last] if item is not None]
            pages = [item["page_idx"] for item in bboxes]
            index = len(chunks)
            chunks.append({"text": value, "metadata": {
                "char_start": start, "char_end": end,
                "token_count": count_tokens(value), "tokenizer": TOKENIZER_NAME,
                "token_count_kind": "chunking_estimate",
                "chunking_strategy": "embedding_tokens_v1",
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "bbox_blocks": bboxes, "file_name": file_name,
                "chunk_index": index, "chunk_id": f"{file_name}_{index}",
            }})
        if end == len(text):
            break
        if chunk_overlap:
            encoded = get_tokenizer().encode(value, add_special_tokens=False)
            # 优先从完整句子/段落的起点重叠，目标 64、可接受 32–96 token。
            left = bisect_right(natural, start)
            right = bisect_left(natural, end)
            candidates = natural[left:right]
            if not candidates:
                candidates = paragraphs[bisect_right(paragraphs, start):bisect_left(paragraphs, end)]
            candidates = [p for p in candidates
                          if chunk_overlap // 2 <= count_tokens(text[p:end]) <= chunk_overlap * 3 // 2]
            next_start = (min(candidates, key=lambda p: abs(count_tokens(text[p:end]) - chunk_overlap))
                          if candidates else start + encoded.offsets[max(0, len(encoded.ids) - chunk_overlap)][0])
            start = max(start + 1, next_start)
        else:
            start = end
    return chunks
