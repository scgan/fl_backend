"""把 Quill 阶段产物组装为可下载的 Word 文档。

纯函数构建：输入产物行（stage/content/section_ref），输出 python-docx
Document。整篇导出时以最新 PLAN 产物为骨架，逐节插入同 section_ref 的
最新 DRAFT 草稿；单产物导出时只渲染该产物自身。
"""

from __future__ import annotations

from typing import Any

from docx import Document

ACADEMIC_NOTICE = "本结果由 AI 辅助生成，请遵守所在机构学术规范。"


def _latest(artifacts: list[dict[str, Any]], stage: str) -> dict[str, Any] | None:
    """取指定阶段版本号最高的产物。

    Args:
        artifacts: 产物行列表（按版本升序传入时后者自然覆盖前者）。
        stage: 阶段编码。

    Returns:
        最新产物行，或 None。
    """
    found: dict[str, Any] | None = None
    for artifact in artifacts:
        if artifact.get("stage") == stage:
            found = artifact
    return found


def _latest_drafts(artifacts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """按章节编号汇总各节最新草稿。

    Args:
        artifacts: 产物行列表。

    Returns:
        section_ref 到最新 DRAFT 产物行的映射。
    """
    drafts: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if artifact.get("stage") != "DRAFT":
            continue
        content = artifact.get("content") or {}
        ref = artifact.get("section_ref") or content.get("section_ref") or ""
        if ref:
            drafts[ref] = artifact
    return drafts


def _add_notice(document: Document) -> None:
    """在文档末尾追加学术诚信提示。

    Args:
        document: 目标 Word 文档。
    """
    document.add_paragraph(ACADEMIC_NOTICE)


def _add_runs(paragraph: Any, runs: list[dict[str, Any]]) -> None:
    """把行内片段写入段落，保留加粗/斜体/下划线/删除线/上下标。

    Args:
        paragraph: 目标段落。
        runs: 形如 [{"text": "…", "bold": true, "superscript": true}, …]。
    """
    for run in runs:
        text = str(run.get("text") or "")
        if not text:
            continue
        piece = paragraph.add_run(text)
        if run.get("bold"):
            piece.bold = True
        if run.get("italic"):
            piece.italic = True
        if run.get("underline"):
            piece.underline = True
        if run.get("strike"):
            piece.font.strike = True
        if run.get("superscript"):
            piece.font.superscript = True
        if run.get("subscript"):
            piece.font.subscript = True


def build_document_from_blocks(
    blocks: list[dict[str, Any]],
    title: str = "",
) -> Document:
    """把编辑器内容块组装为 Word 文档（左侧创作稿导出）。

    前端从 ProseMirror 文档树拍平成块列表（标题/段落/列表项），并带上行内样式
    runs；这里按块类型写样式、按 runs 还原加粗、上下标等格式，保证导出的
    Word 与编辑器所见一致。

    Args:
        blocks: 形如 [{"type": "h1|h2|h3|p|li", "text": "…", "runs": [...]}]；
            空文本块忽略，缺少 runs 时按 text 整体写入。
        title: 文档标题，作为 0 级标题；为空时使用默认标题。

    Returns:
        可直接保存为 .docx 的 Word 文档。

    Raises:
        ValueError: 没有任何可导出内容。
    """
    usable = [
        block
        for block in (blocks or [])
        if isinstance(block, dict) and str(block.get("text") or "").strip()
    ]
    if not usable:
        raise ValueError("编辑器暂无可导出的内容")

    document = Document()
    document.add_heading((title or "").strip() or "Quill 创作稿", level=0)
    for block in usable:
        kind = str(block.get("type") or "p").strip().lower()
        text = str(block.get("text") or "").strip()
        runs = block.get("runs") if isinstance(block.get("runs"), list) else []

        if kind in {"h1", "h2", "h3"}:
            heading = document.add_heading("", level=int(kind[1]))
            if runs:
                _add_runs(heading, runs)
            else:
                heading.add_run(text)
            continue

        style = "List Bullet" if kind == "li" else None
        lines = [line.strip() for line in text.split("\n") if line.strip()] or [text]
        if runs and len(lines) == 1:
            paragraph = document.add_paragraph(style=style) if style else document.add_paragraph()
            _add_runs(paragraph, runs)
            continue
        for line in lines:
            if style:
                document.add_paragraph(line, style=style)
            else:
                document.add_paragraph(line)
    _add_notice(document)
    return document


def _add_rq_section(document: Document, content: dict[str, Any]) -> None:
    """渲染研究问题产物。

    Args:
        document: 目标 Word 文档。
        content: RQ 产物快照。
    """
    document.add_heading("候选研究问题", level=1)
    for index, item in enumerate(content.get("research_questions") or [], start=1):
        document.add_heading(f"问题{index}：{item.get('question', '')}", level=2)
        for key, label in [
            ("proposed_title", "候选题目"),
            ("research_object", "研究对象（建议）"),
            ("research_scope", "研究范围（建议）"),
        ]:
            if item.get(key):
                document.add_paragraph(f"{label}：{item[key]}")
        document.add_paragraph(item.get("rationale", ""))
        hints = []
        if item.get("theory_hints"):
            hints.append("理论线索：" + "、".join(item["theory_hints"]))
        if item.get("method_hints"):
            hints.append("方法线索：" + "、".join(item["method_hints"]))
        if hints:
            document.add_paragraph("；".join(hints))


def _add_plan_section(
    document: Document,
    content: dict[str, Any],
    drafts: dict[str, dict[str, Any]],
) -> None:
    """渲染写作方案产物，并逐节插入已有草稿。

    Args:
        document: 目标 Word 文档。
        content: PLAN 产物快照。
        drafts: section_ref 到最新 DRAFT 产物的映射。
    """
    document.add_heading("论文大纲", level=1)
    if content.get("thesis_statement"):
        document.add_paragraph("中心论点：" + content["thesis_statement"])
    for section in content.get("sections") or []:
        level = min(2 + len(str(section.get("section_id", "")).split(".")) - 1, 4)
        document.add_heading(
            f"{section.get('section_id', '')} {section.get('title', '')}".strip(),
            level=level,
        )
        draft = drafts.get(str(section.get("section_id", "")))
        if draft:
            draft_content = draft.get("content") or {}
            document.add_paragraph((draft_content.get("draft_text") or "").strip())
        else:
            for point in section.get("writing_points") or []:
                document.add_paragraph(point, style="List Bullet")
            if section.get("suggested_citations"):
                markers = "".join(f"[{cid}]" for cid in section["suggested_citations"])
                document.add_paragraph(f"建议参考：{markers}")


def build_quill_document(artifacts: list[dict[str, Any]]) -> Document:
    """把产物列表组装为 Word 文档。

    Args:
        artifacts: 产物行列表，至少一项；每行含 stage/version/section_ref/content。

    Returns:
        可直接保存为 .docx 的 Word 文档。

    Raises:
        ValueError: 产物列表为空或包含未知阶段。
    """
    usable = [artifact for artifact in artifacts if artifact.get("content")]
    if not usable:
        raise ValueError("没有可导出的 Quill 产物")
    plan = _latest(usable, "PLAN")
    rq = _latest(usable, "RQ")
    drafts = _latest_drafts(usable)
    lead = plan or rq or usable[-1]
    topic = (lead.get("content") or {}).get("research_topic") or "Quill 研究写作"

    document = Document()
    document.add_heading(topic, level=0)
    if rq:
        _add_rq_section(document, rq.get("content") or {})
    if plan:
        _add_plan_section(document, plan.get("content") or {}, drafts)

    # 单产物导出 DRAFT/REVISE 时没有 PLAN 骨架，直接渲染其正文。
    if not plan and not rq:
        for artifact in usable:
            content = artifact.get("content") or {}
            document.add_heading(content.get("section_title") or artifact.get("stage", ""), level=1)
            document.add_paragraph(content.get("draft_text") or content.get("revised_text") or "")

    references = (plan or {}).get("content", {}).get("references") or []
    if not references:
        for artifact in usable:
            references = (artifact.get("content") or {}).get("references") or []
            if references:
                break
    if references:
        document.add_heading("建议参考文献", level=1)
        for reference in references:
            document.add_paragraph(reference, style="List Bullet")
    _add_notice(document)
    return document
