#!/usr/bin/env python3
"""
恢复 corpus.attachment='[]' 的语料记录。

背景：
- 部分已审核通过的语料（status=2）数据库中 attachment='[]'，但服务器磁盘上实际存在对应 PDF。
- 本脚本按 title_cn / title_en / corpus_id 前缀在服务器 corpus 目录中查找匹配文件，
  找到后将 corpus.attachment 恢复为指向该文件。

匹配策略（按优先级）：
1. 文件名以 {corpus_id}- 开头（精确匹配，最优先）
2. title_cn 或 title_en 与文件名（去掉 corpus_id 前缀和 .pdf 后）的子串匹配
3. title_cn 或 title_en 与文件名模糊匹配（相似度 >= 0.7）

用法（默认 dry-run，加 --commit 才写库）：
  python scripts/recover_empty_attachments.py --dry-run
  python scripts/recover_empty_attachments.py --commit
  python scripts/recover_empty_attachments.py --limit 50 --commit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except Exception:
    pass


def _load_exports_from_shell(script_path: str) -> None:
    """从 start.sh / start-dev.sh 中读取 export KEY=VALUE 写入 os.environ。"""
    try:
        if not script_path or not os.path.exists(script_path):
            return
        with open(script_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("export "):
                    continue
                kv = line[len("export ") :].strip()
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                k, v = k.strip(), v.strip().strip(";")
                if not k:
                    continue
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if k not in os.environ and v != "":
                    os.environ[k] = v
    except Exception:
        pass


_load_exports_from_shell(os.path.join(_PROJECT_ROOT, "start.sh"))
_load_exports_from_shell(os.path.join(_PROJECT_ROOT, "start-dev.sh"))

# 默认与 recover_corpus_from_files.py 保持一致
DEFAULT_CORPUS_DIR = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "File", "uploaded_files", "corpus"))
ATTACHMENT_PREFIX = "uploaded_files/corpus"


def _clean_filename_for_compare(name: str) -> str:
    """去掉扩展名、前后空格，用于模糊比较。"""
    name = name.strip()
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.strip()


def _strip_corpus_id_prefix(filename: str) -> str:
    """如果文件名以 '数字-' 开头，则去掉该前缀。"""
    m = re.match(r"^(\d+)-(.+)$", filename)
    if m:
        return m.group(2)
    return filename


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _score_match(corpus, filename: str) -> float:
    """
    返回 corpus 与 filename 的匹配分数。
    1.0 表示 corpus_id 前缀精确匹配；
    0.8~0.99 表示标题子串或高度相似匹配；
    0 表示不匹配。
    """
    title_cn = (corpus.title_cn or "").strip()
    title_en = (corpus.title_en or "").strip()

    # 1. corpus_id 前缀精确匹配，最优先
    if filename.startswith(f"{corpus.id}-"):
        return 1.0

    clean_name = _clean_filename_for_compare(_strip_corpus_id_prefix(filename))

    # 2. 子串匹配（标题在文件名中，或文件名在标题中）
    for title in (title_cn, title_en):
        if not title:
            continue
        t = title.lower()
        c = clean_name.lower()
        if len(t) >= 5 and (t in c or c in t):
            return 0.9

    # 3. 模糊相似度
    best = 0.0
    for title in (title_cn, title_en):
        if not title:
            continue
        sim = _similarity(title, clean_name)
        if sim > best:
            best = sim

    # 相似度 >= 0.7 才认为可能匹配
    return best if best >= 0.7 else 0.0


def find_matching_file(corpus, corpus_dir: str, threshold: float = 0.7) -> Optional[Tuple[str, float]]:
    """
    在 corpus_dir 中为 corpus 查找最匹配的 PDF 文件。
    返回 (filename, score) 或 None。
    """
    if not os.path.isdir(corpus_dir):
        return None

    candidates: List[Tuple[float, str]] = []
    for filename in os.listdir(corpus_dir):
        if not filename.lower().endswith(".pdf"):
            continue
        full_path = os.path.join(corpus_dir, filename)
        if not os.path.isfile(full_path):
            continue
        score = _score_match(corpus, filename)
        if score >= threshold:
            candidates.append((score, filename))

    if not candidates:
        return None

    # 按分数降序，同分数下优先文件名短的（更精确）
    candidates.sort(key=lambda x: (-x[0], len(x[1])))
    best_score, best_filename = candidates[0]
    return best_filename, best_score


def run(app, args) -> int:
    from app.exts import db
    from app.models import Corpus

    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.status == 2,
            Corpus.attachment == "[]",
            Corpus.comment.is_(None),
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    if not rows:
        print("没有 attachment='[]' 的语料记录")
        return 0

    print(f"找到 {len(rows)} 条 attachment='[]' 的记录")
    print(f"扫描目录: {args.corpus_dir}")

    updated = 0
    skipped = 0
    matched_files: set = set()

    for idx, corpus in enumerate(rows, 1):
        result = find_matching_file(corpus, args.corpus_dir, threshold=args.threshold)
        if not result:
            print(
                f"[{idx}/{len(rows)}] 未找到匹配文件 corpus_id={corpus.id} "
                f"title_cn={(corpus.title_cn or '')[:40]} "
                f"title_en={(corpus.title_en or '')[:40]}"
            )
            sys.stdout.flush()
            skipped += 1
            continue

        filename, score = result
        file_path = f"{args.attachment_prefix}/{filename}"

        if args.dry_run:
            print(
                f"[{idx}/{len(rows)}][dry-run] corpus_id={corpus.id} score={score:.2f} -> {file_path}\n"
                f"          title_cn={(corpus.title_cn or '')[:50]}"
            )
            sys.stdout.flush()
            continue

        corpus.attachment = json.dumps([{"file_path": file_path, "word": 0}], ensure_ascii=False)

        # 如果该记录之前是软删除状态，恢复时一并取消软删除
        was_deleted = False
        if getattr(corpus, "is_deleted", 0) != 0:
            corpus.is_deleted = 0
            was_deleted = True

        # 每更新一条就提交，方便实时看到数据库变化，也避免中途失败全部回滚
        try:
            db.session.commit()
            matched_files.add(filename)
            updated += 1
            deleted_tag = "（同时取消软删除）" if was_deleted else ""
            print(
                f"[{idx}/{len(rows)}] 已更新 corpus_id={corpus.id} score={score:.2f} -> {filename}{deleted_tag}\n"
                f"          title_cn={(corpus.title_cn or '')[:50]}"
            )
        except Exception as e:
            db.session.rollback()
            print(
                f"[{idx}/{len(rows)}] 更新失败 corpus_id={corpus.id}: {e}",
                file=sys.stderr,
            )
        sys.stdout.flush()

    if args.dry_run:
        print(f"\n[dry-run] 共 {len(rows)} 条，预计可更新 {len(rows) - skipped} 条，跳过 {skipped} 条")
        return 0

    print(f"\n完成：已更新 {updated} 条记录，跳过 {skipped} 条")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="恢复 attachment='[]' 的语料记录")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要更新的记录，不写库")
    parser.add_argument("--commit", action="store_true", help="真正提交数据库更新（默认 dry-run）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数，0 表示不限制")
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default=DEFAULT_CORPUS_DIR,
        help=f"服务器 corpus PDF 目录，默认 {DEFAULT_CORPUS_DIR}",
    )
    parser.add_argument(
        "--attachment-prefix",
        type=str,
        default=ATTACHMENT_PREFIX,
        help="attachment 中的路径前缀，默认 uploaded_files/corpus",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="模糊匹配相似度阈值，默认 0.7；建议先用默认值跑 dry-run 观察",
    )
    args = parser.parse_args()

    if not args.commit and not args.dry_run:
        print("提示：未加 --commit，默认以 dry-run 模式运行")
        args.dry_run = True

    if not os.path.isdir(args.corpus_dir):
        print(f"错误: corpus 目录不存在: {args.corpus_dir}", file=sys.stderr)
        return 2

    from flask import Flask
    from app.exts import db

    app = Flask(__name__)
    app.config.from_object("app.config")
    db.init_app(app)
    with app.app_context():
        return run(app, args)


if __name__ == "__main__":
    raise SystemExit(main())
