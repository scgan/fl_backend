#!/usr/bin/env python3
"""
为《服务器文件名_重复列表.xlsx》里的重复文件打“待删除”标记。

当前 Excel 结构（示例）：
  A 列：title_cn（归一化后文件名，仅作参考，本脚本不再用于匹配）
  B 列：corpus_id（如有）
  C 列：待删除（本脚本写入，若应删除则标记为 1）

新规则：
  - 只根据第二列的 corpus_id 去 corpus 表匹配：
      * 若 corpus_id 不为空：按 Corpus.id == corpus_id 查；
          - 查到记录（不论 is_deleted 值）：认为“库里有记录”，C 列留空；
          - 查不到记录：认为“仅服务器有文件”，在 C 列写入 1，表示待删除。
      * 若 corpus_id 为空：本行忽略，不写“待删除”标记。
"""

import os
import sys
from typing import Optional

from openpyxl import load_workbook


# 保证可以导入 app 包（与其它脚本保持一致）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------- 数据库连接控制 ----------------
# 这里显式指定脚本运行时应连接的数据库名。
# 注意：这只影响本脚本，不会改动其它进程中已加载的配置。
#
# 如果以后想改成别的库名，只需要修改下面这一行即可。
os.environ["FOREIGN_DBNAME"] = "foreign_language"


def strip_ext(title: str) -> str:
    """
    去掉末尾文件后缀（如 .pdf / .docx 等），只保留“标题本身”用于匹配 corpus.title_cn。
    例如：
      '中国教育思想史.pdf' -> '中国教育思想史'
      'abc.def.pdf'        -> 'abc.def'
    """
    if not title:
        return ""
    title = title.strip()
    # 若没有点，直接返回
    if "." not in title:
        return title
    # 从最后一个点分割，去掉扩展名
    return title.rsplit(".", 1)[0].strip()


def mark_delete_flag_for_row(session, Corpus, title_cn: str, corpus_id: Optional[str]) -> bool:
    """
    返回 True 表示“库中无记录，需要删除”（即 Excel 第三列应写 1）。
    返回 False 表示“库中有记录或无需处理”，第三列留空。

    新规则：
      1) corpus_id 不为空：
           - 仅按 (id == corpus_id) 查；
           - 查不到任何记录 → 认为“库中无记录，需要删除”（True）；
           - 查到记录（不论 is_deleted 值）→ 不删除（False）。
      2) corpus_id 为空：
           - 直接返回 False（不做处理）。
    """
    # corpus_id 为空：不做处理
    corpus_id = (corpus_id or "").strip()
    if not corpus_id:
        return False

    # corpus_id 不为空：仅按 id 匹配
    try:
        cid_int = int(corpus_id)
    except ValueError:
        # 非法 id，视为“库中无记录，需要删除”
        return True

    q = session.query(Corpus).filter(Corpus.id == cid_int)
    exists = session.query(q.exists()).scalar()
    return not bool(exists)


def main() -> int:
    # Excel 文件路径
    """对照数据库为重复文件清单设置第三列删除标记，再原地保存 Excel。

    Returns:
        输入缺失为 1，正常结束为 0；只改工作簿，不删除磁盘附件或文献记录。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "服务器文件名_重复列表.xlsx")

    if not os.path.exists(input_path):
        print(f"未找到输入文件: {input_path}")
        print("请确认《服务器文件名_重复列表.xlsx》在项目根目录，或调整脚本中的路径。")
        return 1

    # 延迟导入 Flask / 数据库相关模块，避免脚本在纯 Excel 操作场景下报错
    from flask import Flask
    from app.exts import db
    from app.models import Corpus

    app = Flask(__name__)
    app.config.from_object("app.config")
    db.init_app(app)

    # 1. 建立只供判断的数据库上下文，按工作簿原行顺序检查删除候选
    with app.app_context():
        # 读取 Excel
        wb = load_workbook(input_path)
        ws = wb.active  # 默认第一个工作表

        # 表头处理：确保第三列标题为“待删除”
        header_row = ws[1]
        if header_row is None or len(header_row) == 0:
            ws.cell(row=1, column=1, value="title cn")
            ws.cell(row=1, column=2, value="corpus_id（如有）")
            ws.cell(row=1, column=3, value="待删除")
        else:
            # 若第三列没标题，则补一个
            if header_row[2].value is None:
                header_row[2].value = "待删除"

        # 从第二行开始逐行处理
        marked_count = 0
        total_rows = 0
        for row in ws.iter_rows(min_row=2):
            total_rows += 1
            title_cell = row[0]  # A 列
            id_cell = row[1]     # B 列
            flag_cell = row[2] if len(row) >= 3 else ws.cell(row=title_cell.row, column=3)

            title_cn = (title_cell.value or "").strip() if title_cell.value else ""
            corpus_id = str(id_cell.value).strip() if id_cell and id_cell.value is not None else ""

            # 新规则：只看 corpus_id，title_cn 仅作参考
            if not corpus_id:
                # 没有 id 的行不做删除判断，清空标记
                flag_cell.value = None
                continue

            need_delete = mark_delete_flag_for_row(db.session, Corpus, title_cn, corpus_id)
            if need_delete:
                flag_cell.value = 1
                marked_count += 1
            else:
                # 确认有库中记录，则不标记删除
                flag_cell.value = None

        # 2. 仅把标记写回原工作簿，不在本命令删除原件或数据库记录
        wb.save(input_path)
        print(f"处理完成，共处理数据行 {total_rows} 行，其中标记待删除 {marked_count} 行。")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

