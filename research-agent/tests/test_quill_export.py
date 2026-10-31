"""覆盖 Quill 产物导出文档的组装逻辑。"""

from app.foreign_integration.docx_export import build_quill_document


def _artifact(stage, version, **content):
    """构造产物行。"""
    return {
        "stage": stage,
        "version": version,
        "section_ref": content.get("section_ref"),
        "content": {"research_topic": "生成式人工智能与教育评价", **content},
    }


def test_build_quill_document_assembles_full_paper() -> None:
    """整篇导出以 PLAN 为骨架，按章节插入最新 DRAFT 草稿与参考文献。"""
    artifacts = [
        _artifact(
            "RQ",
            1,
            research_questions=[
                {
                    "question": "智能教育如何影响师生角色分配？",
                    "rationale": "依据。[1]",
                    "theory_hints": ["活动理论"],
                }
            ],
            references=["张三. 智能教育研究[J]. 测试刊, 2024."],
        ),
        _artifact(
            "PLAN",
            1,
            thesis_statement="生成式人工智能倒逼评价体系系统性调适。",
            sections=[
                {"section_id": "1", "title": "引言", "writing_points": ["背景"],
                 "suggested_citations": [1]},
                {"section_id": "2", "title": "机制", "writing_points": ["范式转移"]},
            ],
            references=["张三. 智能教育研究[J]. 测试刊, 2024."],
        ),
        _artifact(
            "DRAFT",
            1,
            section_ref="1",
            section_title="引言",
            draft_text="旧版草稿。",
        ),
        _artifact(
            "DRAFT",
            2,
            section_ref="1",
            section_title="引言",
            draft_text="生成式人工智能降低了低阶内容加工的价值。[1]",
        ),
    ]
    document = build_quill_document(artifacts)
    texts = [paragraph.text for paragraph in document.paragraphs]
    joined = "\n".join(texts)
    assert "生成式人工智能与教育评价" in joined
    assert "候选研究问题" in joined
    assert "论文大纲" in joined
    assert "中心论点：生成式人工智能倒逼评价体系系统性调适。" in joined
    # 同章节取版本号最新的草稿。
    assert "生成式人工智能降低了低阶内容加工的价值。[1]" in joined
    assert "旧版草稿。" not in joined
    assert "张三. 智能教育研究[J]. 测试刊, 2024." in joined
    # 未起草的章节保留写作要点。
    assert "范式转移" in joined


def test_build_quill_document_rejects_empty() -> None:
    """空产物列表必须拒绝导出。"""
    import pytest

    with pytest.raises(ValueError):
        build_quill_document([])
