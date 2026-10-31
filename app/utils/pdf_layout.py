"""根据真实文字位置恢复双栏阅读顺序，不让整页同高文字合成一行。"""
from __future__ import annotations

import statistics


def _rows(items):
    """按垂直重叠聚合文字；允许同一行存在上标或字体大小差异。"""
    rows = []
    for item in sorted(items, key=lambda item: (item["top"], item["x0"])):
        height = item["bottom"] - item["top"]
        for row in reversed(rows[-3:]):
            overlap = min(row["bottom"], item["bottom"]) - max(row["top"], item["top"])
            if overlap >= min(height, row["bottom"] - row["top"]) * 0.5:
                row["items"].append(item)
                row["top"] = min(row["top"], item["top"])
                row["bottom"] = max(row["bottom"], item["bottom"])
                break
        else:
            rows.append({"items": [item], "top": item["top"], "bottom": item["bottom"]})
    return sorted(rows, key=lambda row: row["top"])


def _line(items):
    """按横坐标合并同一行词框，在明显字间距处补空格并保留真实联合边界。"""
    items = sorted(items, key=lambda item: item["x0"])
    parts = []
    previous = None
    for item in items:
        value = item["text"]
        if previous is not None:
            gap = item["x0"] - previous["x1"]
            if gap > max(0.5, (item["bottom"] - item["top"]) * 0.1):
                parts.append(" ")
        parts.append(value)
        previous = item
    return {"text": "".join(parts), "x0": min(i["x0"] for i in items),
            "x1": max(i["x1"] for i in items), "top": min(i["top"] for i in items),
            "bottom": max(i["bottom"] for i in items)}


def _gutter(rows, height):
    """只接受在多行同时出现的宽间隙，避免把普通单栏空格当成分栏。"""
    fragments, gaps = [], []
    for row in rows:
        groups = [[]]
        for item in sorted(row["items"], key=lambda item: item["x0"]):
            if groups[-1] and item["x0"] - groups[-1][-1]["x1"] >= max(12, height * 1.2):
                groups.append([])
            groups[-1].append(item)
        lines = [_line(group) for group in groups]
        fragments.append(lines)
        for left, right in zip(lines, lines[1:]):
            gaps.append((left["x1"], right["x0"], left, right))
    if not gaps:
        return None, fragments
    x0 = min(item["x0"] for row in rows for item in row["items"])
    x1 = max(item["x1"] for row in rows for item in row["items"])
    width = x1 - x0
    candidates = []
    for low, high, _, _ in gaps:
        cut = (low + high) / 2
        if not x0 + width * 0.25 <= cut <= x0 + width * 0.75:
            continue
        support = [(left, right) for start, stop, left, right in gaps if start <= cut <= stop
                   and min(left["x1"] - left["x0"], right["x1"] - right["x0"]) >= max(height * 6, width * 0.18)]
        if len(support) < max(4, len(rows) * 0.25):
            continue
        crossing = sum(any(line["x0"] < cut < line["x1"] for line in lines) for lines in fragments)
        candidates.append((len(support) - crossing * 0.25, len(support), cut))
    return (max(candidates)[2] if candidates else None), fragments


def words_to_lines(words, table_regions=()):
    """从原生词框恢复阅读顺序，以跨栏内容及有边框表格划分独立区域。

    Args:
        words: 含 text、x0、x1、top、bottom 的原生词框列表。
        table_regions: 与词框同一坐标系的表格矩形列表。

    Returns:
        保留真实几何边界的行字典；双栏区先左后右，表格区保留逐行顺序。
    """
    # 1. 去掉空白或无效词框，按真实垂直重叠聚行后检测稳定的分栏间隙
    words = [word for word in words if word.get("text", "").strip()
             and word["x1"] > word["x0"] and word["bottom"] > word["top"]]
    if not words:
        return []
    rows = _rows(words)
    height = statistics.median(word["bottom"] - word["top"] for word in words)
    cut, fragments = _gutter(rows, height)
    if cut is None:
        return [_line(row["items"]) for row in rows]

    # 跨栏标题/正文和表格按纵向位置分隔栏区，不能把它们当作左栏或右栏。
    # 2. 将跨栏标题、页眉页脚和表格合并为独立区域，避免把表格误按双栏读完
    anchors = []
    for row, lines in zip(rows, fragments):
        if any(line["x0"] < cut < line["x1"] for line in lines):
            anchors.append((row["top"], row["bottom"]))
    # 页眉/页脚可能由左右两段组成。与正文有明显纵向空白时整体保留位置，
    # 避免把右侧页眉排到左栏正文之后。只检查页首/页尾三行，勿切开正文段落。
    for index, (previous, following) in enumerate(zip(rows, rows[1:])):
        if following["top"] - previous["bottom"] >= height * 1.2:
            if index < 3:
                anchors.append((rows[0]["top"], previous["bottom"]))
            if len(rows) - index - 1 <= 3:
                anchors.append((following["top"], rows[-1]["bottom"]))
    anchors.extend((top, bottom) for _, top, _, bottom in table_regions)
    intervals = []
    for top, bottom in sorted(anchors):
        if intervals and top <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(bottom, intervals[-1][1]))
        else:
            intervals.append((top, bottom))

    # 3. 按区域顺序输出：普通正文先左后右，跨栏和表格保留逐行顺序
    result, remaining = [], list(words)

    def column_lines(items):
        """将当前普通正文区域按分栏线分为左右两侧，各自聚行后先左后右输出。"""
        return [_line(row["items"]) for side in (False, True)
                for row in _rows([item for item in items if (item["x0"] >= cut) == side])]

    for top, bottom in intervals:
        before = [word for word in remaining if (word["top"] + word["bottom"]) / 2 < top]
        result.extend(column_lines(before))
        inside = [word for word in remaining if top <= (word["top"] + word["bottom"]) / 2 <= bottom]
        result.extend(_line(row["items"]) for row in _rows(inside))
        remaining = [word for word in remaining if (word["top"] + word["bottom"]) / 2 > bottom]
    result.extend(column_lines(remaining))
    return result


def plumber_lines(page):
    # 有边框表格按行读取；普通正文没有必要运行表格检测。
    """从 pdfplumber 页面提取词框和有边框表格，用统一布局规则生成阅读行。"""
    tables = [table.bbox for table in page.find_tables()] if len(page.edges) >= 4 else []
    return words_to_lines(page.extract_words(), tables)


def order_positioned_lines(lines):
    """OCR 已返回独立行时，只恢复顺序，不改写文字或猜测缺失的坐标。"""
    if not lines or any(precision != "line" for _, _, precision in lines):
        return lines
    items = [{"text": text, "x0": box[0], "top": box[1], "x1": box[2], "bottom": box[3],
              "index": index} for index, (text, box, _) in enumerate(lines)]
    rows = _rows(items)
    cut, fragments = _gutter(rows, statistics.median(item["bottom"] - item["top"] for item in items))
    if cut is None:
        return sorted(lines, key=lambda line: (line[1][1], line[1][0]))
    anchors = [row for row, parts in zip(rows, fragments)
               if any(part["x0"] < cut < part["x1"] for part in parts)]
    ordered, remaining = [], items
    for anchor in anchors:
        before = [item for item in remaining if item["top"] < anchor["top"]]
        ordered.extend(sorted(before, key=lambda item: (item["x0"] >= cut, item["top"], item["x0"])))
        inside = [item for item in remaining if anchor["top"] <= item["top"] <= anchor["bottom"]]
        ordered.extend(sorted(inside, key=lambda item: (item["top"], item["x0"])))
        remaining = [item for item in remaining if item["top"] > anchor["bottom"]]
    ordered.extend(sorted(remaining, key=lambda item: (item["x0"] >= cut, item["top"], item["x0"])))
    return [lines[item["index"]] for item in ordered]
