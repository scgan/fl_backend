#!/usr/bin/env python3
"""
对上传者 contributor=26 的语料，将非中文文献的原文标题翻译成中文写入译文标题。

使用方式（在项目根目录执行）：
  python scripts/translate_recovery_titles.py --dry-run
  python scripts/translate_recovery_titles.py --limit 100
  python scripts/translate_recovery_titles.py

说明：
  - 仅处理 contributor=26、未删除的语料。
  - 若 language 为「中文」则跳过。
  - 非中文且 title_cn（原文）非空时，调用百度翻译 API 将 title_cn 译为中文，结果写入 title_en（译文）。
  - 翻译账号与 collector 中 translate_summary 一致，可按需在脚本内或环境变量中配置。
"""

import argparse
import os
import sys
import time
from datetime import datetime

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

# 百度翻译账号（与 collector translate_summary 一致）
BAIDU_ACCOUNTS = [
    {"app_id": "20250915002455002", "secret_key": "YG14pmLfoJqzfndudgml"},
]


def is_chinese_language(lang):
    """判断语料语言是否为中文（仅当 language 为「中文」两字时视为中文），若是则无需翻译标题。"""
    return lang is not None and str(lang).strip() == "中文"


def run(app, args):
    """翻译指定日期后恢复文献的非中文原题名，成功译文写入 title_en。

    Args:
        app: 已建立上下文的 Flask 应用。
        args: 含 limit 及 dry_run 的执行配置，缺省实际调用翻译服务。

    Returns:
        无失败为 0，单项翻译或最终提交失败为 1；已有失败不阻止其他成功译文提交。
    """
    from app.exts import db
    from app.models import Corpus
    from app.utils.utils import translate_text

    contributor_id = 26
    submit_time_threshold = datetime(2026, 3, 3)
    # 1. 按恢复贡献者和提交时间固定候选，跳过中文文献与空原题名
    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.contributor == contributor_id,
            Corpus.is_deleted == 0,
            Corpus.submit_time > submit_time_threshold,
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    total = len(rows)
    skipped_chinese = 0
    skipped_no_title = 0
    ok = 0
    fail = 0

    # 2. 逐篇调用既有账户翻译原题名，失败保留原数据，成功只更新译文列
    for corpus in rows:
        if is_chinese_language(corpus.language):
            skipped_chinese += 1
            continue
        title_cn_original = (corpus.title_cn or "").strip()  # 原文标题（可能为外文）
        if not title_cn_original:
            skipped_no_title += 1
            continue

        if args.dry_run:
            print(f"[dry-run] would translate corpus_id={corpus.id} title_cn(原文)={title_cn_original[:50]}...")
            ok += 1
            continue

        translated = None
        for acc in BAIDU_ACCOUNTS:
            translated = translate_text(
                acc["app_id"],
                acc["secret_key"],
                title_cn_original,
                "auto",
                "zh",
            )
            if translated:
                break
            time.sleep(0.2)

        if translated:
            corpus.title_en = translated  # 译文标题写入 title_en
            ok += 1
            print(f"corpus_id={corpus.id} 原文: {title_cn_original[:80]}{'...' if len(title_cn_original) > 80 else ''}")
            print(f"        译文: {translated}")
        else:
            fail += 1
            print(f"fail corpus_id={corpus.id} title_cn(原文)={title_cn_original[:50]}...", file=sys.stderr)
        time.sleep(0.15)

    # 3. 一次提交本轮成功译文，单项翻译失败仍在汇总中保留，不撤销其他成功项
    if not args.dry_run and (ok > 0 or fail > 0):
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1

    print(
        f"total={total} skipped_chinese={skipped_chinese} skipped_no_title={skipped_no_title} ok={ok} fail={fail}"
    )
    return 0 if fail == 0 else 1


def main():
    """读取预览和数量参数，建立最小 Flask 数据库上下文后执行题名翻译。"""
    parser = argparse.ArgumentParser(description="将 contributor=26 的非中文语料 title_cn(原文) 翻译成中文写入 title_en(译文)")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要处理的记录，不写库")
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数，0 表示不限制")
    args = parser.parse_args()

    from flask import Flask
    from app.exts import db

    app = Flask(__name__)
    app.config.from_object("app.config")
    db.init_app(app)
    with app.app_context():
        return run(app, args)


if __name__ == "__main__":
    sys.exit(main())
