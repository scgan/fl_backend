#!/usr/bin/env python3
"""
交换 contributor=108、语言为英文但标题写反的语料的 title_cn 和 title_en。

使用方式（在项目根目录执行）：
  python scripts/swap_titles.py --dry-run
  python scripts/swap_titles.py --limit 100
  python scripts/swap_titles.py

说明：
  - 仅处理 contributor=108、未删除、语言为英文的语料。
  - 条件：title_cn 为中文（包含中文字符）且 title_en 为英文（不包含中文字符）。
  - 交换 title_cn 和 title_en。
"""

import argparse
import os
import sys
import re

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except Exception:
    pass


def _load_exports_from_shell(script_path: str) -> None:
    """从 start.sh / start-dev.sh 中读取 export KEY=VALUE 写入 os.environ，供数据库等配置使用。"""
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


def contains_chinese(text):
    """判断文本是否包含中文字符。"""
    if not text:
        return False
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def is_english_text(text):
    """判断文本是否为英文（不包含中文字符，但包含英文字母）。"""
    if not text:
        return False
    text = str(text)
    has_english = bool(re.search(r'[a-zA-Z]', text))
    has_chinese = contains_chinese(text)
    return has_english and not has_chinese


def run(app, args):
    """交换指定贡献者英文文献中放反的双语题名，按条提交。

    Args:
        app: 调用方已经建立上下文的 Flask 应用。
        args: limit 限制范围，dry_run 仅打印，缺省实际逐条写库。

    仅中文 title_cn 与英文 title_en 同时存在并符合检测规则时交换；
    前面已经提交的记录不会因后续某条失败整体回滚。
    """
    from app.exts import db
    from app.models import Corpus

    contributor_id = 108
    # 1. 固定贡献者与语言范围，仅候选两种题名都存在时进行后续判定
    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.contributor == contributor_id,
            Corpus.is_deleted == 0,
            Corpus.language == "英文",
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    total = len(rows)
    ok = 0
    skipped = 0

    # 2. 逐篇验证中英文放反条件；预览只打印，实际交换每条独立提交
    for corpus in rows:
        title_cn = (corpus.title_cn or "").strip()
        title_en = (corpus.title_en or "").strip()

        # 判断是否需要交换：title_cn 是中文，title_en 是英文
        if not title_cn or not title_en:
            skipped += 1
            continue

        if not (contains_chinese(title_cn) and is_english_text(title_en)):
            skipped += 1
            continue

        # 需要交换
        if args.dry_run:
            print(f"[dry-run] would swap corpus_id={corpus.id}")
            print(f"  title_cn (中文): {title_cn[:60]}...")
            print(f"  title_en (英文): {title_en[:60]}...")
            ok += 1
            continue

        # 执行交换
        corpus.title_cn, corpus.title_en = title_en, title_cn
        db.session.commit()
        print(f"[ok] swapped corpus_id={corpus.id}")
        ok += 1

    print(f"\n总计: {total}, 交换: {ok}, 跳过: {skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="仅预览不执行")
    parser.add_argument("--limit", type=int, default=0, help="限制处理数量")
    args = parser.parse_args()

    from flask import Flask
    from app.exts import db

    app = Flask(__name__)
    app.config.from_object("app.config")
    db.init_app(app)

    with app.app_context():
        run(app, args)
