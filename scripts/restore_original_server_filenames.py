#!/usr/bin/env python3
"""
根据《服务器文件名_重复列表.xlsx》中的 corpus_id 与归一化标题，还原出“原始服务器文件名”。

当前 Excel 结构（示例）：
  A 列：title_cn（归一化后文件名，当前不带 id 前缀）
  B 列：corpus_id（如有）
  C 列：待删除（前一个脚本打的标记，本脚本不改）

本脚本逻辑：
  - 对于每一行：
      * 若第二列 corpus_id 有值：
            - 以 "corpus_id-原A列内容" 拼出“原始文件名”，写回 A 列；
              例如：A="How should I..."，B=195513 → A 改为 "195513-How should I..."。
      * 若第二列 corpus_id 为空：
            - 不改动 A 列（认为 A 本身就是原始文件名）。

  - 直接在原文件《服务器文件名_重复列表.xlsx》上修改，请在运行前自行备份（如有需要）。
"""

import os
from typing import Optional

from openpyxl import load_workbook


def normalize_corpus_id(raw_id: Optional[object]) -> str:
    """
    将 Excel 单元格里的 corpus_id 统一转为不带小数点的字符串：
      - 可能是 int、str 或 float（如 195513.0）。
    """
    if raw_id is None:
        return ""
    # 先按字符串处理
    s = str(raw_id).strip()
    if not s:
        return ""
    # 尝试当作数字转成整数，避免 "195513.0" 这种情况
    try:
        # 有些 Excel 会把数字列读成 float
        num = float(s)
        if num.is_integer():
            return str(int(num))
    except Exception:
        pass
    # 回退：直接返回原字符串
    return s


def main() -> int:
    """按文献 ID 为 Excel 第一列规范化文件名前缀，并原地保存工作簿。

    Returns:
        成功为 0，输入工作簿不存在为 1；不访问数据库，也不重命名磁盘附件。
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xlsx_path = os.path.join(project_root, "服务器文件名_重复列表.xlsx")

    if not os.path.exists(xlsx_path):
        print(f"未找到 Excel 文件: {xlsx_path}")
        return 1

    wb = load_workbook(xlsx_path)
    ws = wb.active  # 默认第一个工作表

    # 假设：
    #  A 列：title_cn（当前为“去掉前缀后的文件名”）
    #  B 列：corpus_id
    #  C 列：待删除（不改）

    updated_rows = 0
    total_rows = 0

    for row in ws.iter_rows(min_row=2):  # 跳过表头
        total_rows += 1
        title_cell = row[0]  # A 列
        id_cell = row[1]     # B 列

        title_val = title_cell.value
        corpus_id = normalize_corpus_id(id_cell.value if id_cell is not None else None)

        # corpus_id 为空：不改 A 列
        if not corpus_id:
            continue

        # A 原值可能已经带有前缀（多次运行脚本会变成 "id-id-标题" 这种），
        # 这里先把前面重复的 "id-" 前缀全部剥掉，再统一加回一次，保证幂等。
        title_str = "" if title_val is None else str(title_val)
        prefix = f"{corpus_id}-"
        while title_str.startswith(prefix):
            title_str = title_str[len(prefix) :]
        new_name = prefix + title_str

        if title_cell.value != new_name:
            title_cell.value = new_name
            updated_rows += 1

    wb.save(xlsx_path)
    print(f"处理完成，共扫描数据行 {total_rows} 行，其中更新了 {updated_rows} 行（写回原始文件名到 A 列）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

