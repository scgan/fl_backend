"""
批量修复 corpus.attachment 中未带 <corpus_id>- 前缀的 file_path。

背景：
- 早期数据中，attachment.file_path 可能为：uploaded_files/corpus/原文件名
- 但磁盘文件已被重命名为：<corpus_id>-原文件名
  导致数据库路径与实际文件名不一致。

本脚本会将 corpus.attachment 的每个 item["file_path"] 更新为：
  uploaded_files/corpus/<corpus_id>-原文件名

默认 dry-run（不提交），加 --commit 才会真正写库。

用法示例：
  python scripts/update_attachment_name.py
  python scripts/update_attachment_name.py --commit
  python scripts/update_attachment_name.py --limit 100 --commit
  python scripts/update_attachment_name.py --prefix uploaded_files/corpus --commit
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any, Dict, List, Tuple


def _parse_attachment(attachment_text: str) -> List[Dict[str, Any]] | None:
    """解析附件 JSON 数组并仅保留字典项，无效输入返回 None。"""
    try:
        data = json.loads(attachment_text)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    out: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            out.append(item)
    return out


def _split_prefix_and_name(file_path: str) -> Tuple[str, str]:
    """
    将 'uploaded_files/corpus/xxx.pdf' 拆成 ('uploaded_files/corpus', 'xxx.pdf')
    若无斜杠则返回 ('', file_path)。
    """
    if not isinstance(file_path, str):
        return "", ""
    file_path = file_path.strip()
    if "/" not in file_path:
        return "", file_path
    prefix, name = file_path.rsplit("/", 1)
    return prefix, name


_LEADING_ID_RE = re.compile(r"^\d+-")


def _needs_id_prefix(corpus_id: int, file_name: str) -> bool:
    """判断非空文件名是否缺少当前文献 ID 前缀，已有其他 ID 前缀不视为匹配。"""
    if not file_name:
        return False
    return not file_name.startswith(f"{corpus_id}-")


def _with_id_prefix(corpus_id: int, file_name: str) -> str:
    # 保留原文件名内容（包括可能存在的前导空格等），只在最前面拼 corpus_id-
    """在当前文件名原样前添加文献 ID，不删除已有的其他前缀。"""
    return f"{corpus_id}-{file_name}"


def main() -> int:
    """为指定目录下的历史附件引用补充当前文献 ID 前缀，默认回滚预览。

    只改 corpus.attachment 中的路径，不重命名磁盘文件或调用文件服务，
    运行前应确认目标命名已在实际文件侧存在。

    Returns:
        完成为 0，显式提交失败并回滚时为 2。
    """
    parser = argparse.ArgumentParser(description="修复 corpus.attachment.file_path 缺少 <id>- 前缀的历史数据")
    parser.add_argument("--prefix", type=str, default="uploaded_files/corpus",
                        help="attachment.file_path 的目录前缀，默认 uploaded_files/corpus")
    parser.add_argument("--commit", action="store_true", help="真正提交数据库更新（默认 dry-run）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少条 corpus 记录（0 表示不限制）")
    parser.add_argument("--only-missing", action="store_true",
                        help="仅处理 file_name 不以 <id>- 开头的记录（默认就是该行为，仅用于显式说明）")
    args = parser.parse_args()

    # 复用项目 Flask app 与 SQLAlchemy 初始化（会读取 app/config.py 与 .env）
    from app.app import app
    from app.exts import db
    from app.models import Corpus

    changed_rows = 0
    scanned_rows = 0
    changed_items = 0

    # 1. 扫描有效附件引用，只处理目标远程目录内缺少当前文献前缀的项
    with app.app_context():
        q = db.session.query(Corpus).filter(Corpus.is_deleted == 0, Corpus.attachment.isnot(None))
        if args.limit and args.limit > 0:
            q = q.limit(int(args.limit))

        for corpus in q:
            scanned_rows += 1
            if not corpus.id or not corpus.attachment:
                continue

            att_list = _parse_attachment(corpus.attachment)
            if att_list is None:
                continue

            row_changed = False
            for item in att_list:
                fp = item.get("file_path")
                if not isinstance(fp, str) or not fp.strip():
                    continue

                prefix, file_name = _split_prefix_and_name(fp)
                if prefix != args.prefix:
                    # 只修复目标目录下的 attachment，避免误改其他目录
                    continue

                if not _needs_id_prefix(int(corpus.id), file_name):
                    continue

                # 更新为 <id>-原文件名（不去除 file_name 的前导空格，保持与磁盘一致）
                new_name = _with_id_prefix(int(corpus.id), file_name)
                new_fp = f"{args.prefix}/{new_name}"
                item["file_path"] = new_fp
                row_changed = True
                changed_items += 1

            if row_changed:
                corpus.attachment = json.dumps(att_list, ensure_ascii=False)
                changed_rows += 1

        # 2. 按执行开关统一提交或回滚；本命令不验证或修改真实附件文件
        if args.commit:
            try:
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                print(f"提交失败，已回滚：{e}")
                return 2
        else:
            db.session.rollback()

    mode = "COMMIT" if args.commit else "DRY-RUN"
    print(f"[{mode}] 扫描 corpus 行数={scanned_rows}，更新行数={changed_rows}，更新 attachment.item 数={changed_items}")
    if not args.commit:
        print("提示：当前为 dry-run，未写入数据库；如需提交请加 --commit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
