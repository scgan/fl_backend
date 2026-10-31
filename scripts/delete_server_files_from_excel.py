#!/usr/bin/env python3
"""
根据《待删除.xlsx》中的文件名批量删除服务器上的文件。

使用说明（在服务器 Backend 项目根目录执行，例如 ~/Foreign/Backend）：

  python3 scripts/delete_server_files_from_excel.py --dry-run
  python3 scripts/delete_server_files_from_excel.py

默认约定：
  - Excel 文件名：项目根目录下的「待删除.xlsx」；
  - 文件名所在列：第 1 列（A 列），第 1 行是表头，从第 2 行开始为数据；
  - 服务器文件目录：相对项目根目录为 ../File/uploaded_files/corpus
      也就是你在服务器上看到的：
        ~/Foreign/File/uploaded_files/corpus

脚本行为：
  - 读取 Excel，从第 2 行开始按行读取 A 列的文件名（非空才处理）；
  - 对每个文件名，在 corpus 目录下拼出完整路径并检查是否存在；
  - 若存在：
      * --dry-run：只打印“将删除 ...”，不真正删除；
      * 非 dry-run：执行 os.remove 删除，并统计成功/失败数量。
  - 若不存在：记录“文件不存在”，方便你后续排查。
"""

import argparse
import os
from typing import List, Tuple

from openpyxl import load_workbook


def load_filenames_from_excel(xlsx_path: str, column_index: int = 1) -> List[str]:
    """
    从 Excel 指定列读取文件名列表。
    - column_index 从 1 开始（1=A 列，2=B 列...）
    - 跳过第 1 行表头，从第 2 行开始读取。
    """
    wb = load_workbook(xlsx_path)
    ws = wb.active

    names: List[str] = []
    for row in ws.iter_rows(min_row=2):
        if len(row) < column_index:
            continue
        cell = row[column_index - 1]
        val = cell.value
        if not val:
            continue
        name = str(val).strip()
        if not name:
            continue
        names.append(name)
    return names


def delete_files(
    base_dir: str, filenames: List[str], dry_run: bool = True
) -> Tuple[int, int, int]:
    """
    在 base_dir 下删除 filenames 指定的文件。
    返回 (success_count, not_found_count, fail_count)。
    """
    success = 0
    not_found = 0
    fail = 0

    for idx, name in enumerate(filenames, 1):
        full_path = os.path.join(base_dir, name)
        if not os.path.exists(full_path):
            print(f"[{idx}/{len(filenames)}] 文件不存在: {full_path}")
            not_found += 1
            continue

        if dry_run:
            print(f"[dry-run] [{idx}/{len(filenames)}] 将删除: {full_path}")
            success += 1
            continue

        try:
            os.remove(full_path)
            print(f"[{idx}/{len(filenames)}] 已删除: {full_path}")
            success += 1
        except Exception as e:
            print(f"[{idx}/{len(filenames)}] 删除失败: {full_path} -> {e}")
            fail += 1

    return success, not_found, fail


def main() -> int:
    """读取 Excel 第一列文件名并删除服务器附件目录中的匹配文件。

    必须显式给出 --dry-run 才仅预览，默认会执行磁盘删除；不更新数据库引用。

    Returns:
        输入目录、工作簿或候选缺失为 1；执行结束为 0，逐文件失败另在统计中报告。
    """
    parser = argparse.ArgumentParser(
        description="根据《待删除.xlsx》中的文件名删除服务器 corpus 目录下对应文件"
    )
    parser.add_argument(
        "--excel",
        type=str,
        default="待删除.xlsx",
        help="包含待删除文件名的 Excel 文件路径（相对项目根目录），默认=待删除.xlsx",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要删除的文件，不真正删除（推荐先跑一遍）",
    )
    args = parser.parse_args()

    # 1. 解析工作簿和真实附件目录，输入不完整时停止，避免基于错误路径删除
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = (
        args.excel
        if os.path.isabs(args.excel)
        else os.path.join(project_root, args.excel)
    )

    if not os.path.exists(excel_path):
        print(f"错误：未找到 Excel 文件: {excel_path}")
        return 1

    # 服务器文件目录：../File/uploaded_files/corpus
    corpus_dir = os.path.abspath(
        os.path.join(project_root, "..", "File", "uploaded_files", "corpus")
    )
    if not os.path.isdir(corpus_dir):
        print(f"错误：服务器文件目录不存在: {corpus_dir}")
        return 1

    print(f"项目根目录: {project_root}")
    print(f"Excel 文件: {excel_path}")
    print(f"服务器文件目录: {corpus_dir}")
    print(f"dry_run = {args.dry_run}")

    # 2. 读取并规范化候选文件名，再按显式 dry_run 决定是否删除磁盘文件
    filenames = load_filenames_from_excel(excel_path, column_index=1)
    if not filenames:
        print("Excel 中未读取到任何文件名（检查是否在第 1 列，从第 2 行开始）")
        return 1

    print(f"共读取到 {len(filenames)} 个待处理文件名（不含表头）。")

    success, not_found, fail = delete_files(
        corpus_dir, filenames, dry_run=args.dry_run
    )

    print(
        f"完成：成功处理 {success} 个，文件不存在 {not_found} 个，删除失败 {fail} 个。"
    )
    if args.dry_run:
        print("当前为 dry-run 模式，未真正删除任何文件。确认无误后去掉 --dry-run 再执行一次即可。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

