#!/usr/bin/env python3
"""
将题目或关键词中含有「道」「儒」「中国」的待审核恢复语料预分配给 user_id=32 审核；
可选 --scale：将题目含「中国英语能力等级量表」的恢复语料关联到指定库并设为审核通过；
可选 --gudai：将 reviewer_id=32 的待审核文献中、题目含「中国古代」的语料放进 中国教育思想文献数据库（汉语）、中国古代教育思想文献 并设为审核通过。
可选 --jindai：将 reviewer_id=32 审核的历史文献中、标题含「中国近代」的语料放进 中国近现代教育思想文献 子库并设为审核通过。
可选 --caiyuanpei：将题目含「蔡元培」的文献放进 中国近现代教育思想文献 子库并设为审核通过。
可选 --lishi：历史文献审核规则，标题含「高校外语」→中国外语教育数据库；「日本」→国外教育文献数据库；
          「明代」→中国古代教育思想文献；「民国」「清代」「清末」→中国近现代教育思想文献；并设为审核通过。

使用方式（在项目根目录执行）：
  python scripts/assign_recovery_reviewer_32.py --dry-run
  python scripts/assign_recovery_reviewer_32.py --scale --dry-run
  python scripts/assign_recovery_reviewer_32.py --scale
  python scripts/assign_recovery_reviewer_32.py --gudai --dry-run
  python scripts/assign_recovery_reviewer_32.py --gudai
  python scripts/assign_recovery_reviewer_32.py --jindai --dry-run
  python scripts/assign_recovery_reviewer_32.py --jindai
  python scripts/assign_recovery_reviewer_32.py --caiyuanpei --dry-run
  python scripts/assign_recovery_reviewer_32.py --caiyuanpei
  python scripts/assign_recovery_reviewer_32.py --lishi --dry-run
  python scripts/assign_recovery_reviewer_32.py --lishi

说明：
  - 默认：contributor=26、待审核、reviewer_id 为空，且标题/关键词含「道」「儒」「中国」→ 设置 reviewer_id=32。
  - --scale：题目含「中国英语能力等级量表」→ 关联指定 4 库并设为审核通过。
  - --gudai：reviewer_id=32、待审核、题目含「中国古代」→ 关联 中国教育思想文献数据库（汉语）、中国古代教育思想文献。
  - --lishi：reviewer_id=32、contributor=26、待审核、标题含对应关键词→关联规则指定的库并设为审核通过。
"""

import argparse
import json
import os
import re
import sys

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

ASSIGNED_REVIEWER_ID = 32
RECOVERY_CONTRIBUTOR_ID = 26
# 题目/关键词含以下任一字符的语料分配给 ASSIGNED_REVIEWER_ID
KEYWORDS_DAO_RU_ZHONGGUO = ("道", "儒", "中国")
# 题目含「中国英语能力等级量表」的语料：关联到以下库并设为审核通过
SCALE_DATABASE_NAMES = (
    "中国当代教育思想文献",
    "中国外语教育理论文献库",
    "中国外语教育政策文献库",
    "中国外语教育实践数据库（教学模式、探索创新）",
)
# 题目含「中国古代」且 reviewer_id=32 的待审核语料：关联到以下库并设为审核通过
GUDAI_DATABASE_NAMES = (
    "中国教育思想文献数据库（汉语）",
    "中国古代教育思想文献",
)
# 题目含「中国近代」且 reviewer_id=32 审核的历史文献：关联到以下子库并设为审核通过
JINDAI_DATABASE_NAMES = ("中国近现代教育思想文献",)
# 题目含「蔡元培」的文献：关联到以下子库并设为审核通过
CAIYUANPEI_DATABASE_NAMES = ("中国近现代教育思想文献",)
# 题目含「习近平」的文献：关联到以下子库并设为审核通过
XIJINPING_DATABASE_NAMES = ("中国当代教育思想文献",)
# 题目含「中国口译」的文献：关联到以下子库并设为审核通过
ZHONGGUO_KOUYI_DATABASE_NAMES = ("中国外语教育实践数据库（教学模式、探索创新）",)
# 历史文献审核规则：标题关键词 → 目标库名
LISHI_RULES = (
    # (匹配标题关键词元组, 目标库名元组)
    (("高校外语",), ("中国外语教育数据库",)),
    (("日本",), ("国外教育文献数据库（汉语、英语、俄语、日语、法语、朝语、西班牙语）",)),
    (("明代",), ("中国古代教育思想文献",)),
    (("民国",), ("中国近现代教育思想文献",)),
    (("清代", "清末"), ("中国近现代教育思想文献",)),
)


def run_scale(app, args):
    """将题目含「中国英语能力等级量表」的恢复语料关联到指定库并设为审核通过。

    Args:
        app: 已由调用方建立上下文的 Flask 应用。
        args: dry_run 控制预览，limit 限制本次查询数量。

    Returns:
        完成或无候选时为 0，提交失败回滚后为 1。
    """
    from sqlalchemy import or_
    from app.exts import db
    from app.models import Corpus, Database, CorpusDatabase
    from app.utils.enums import CorpusStatus

    title_key = "中国英语能力等级量表"
    # 1. 按本专项的原有关键词、恢复身份和审核条件查询候选
    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.is_deleted == 0,
            Corpus.contributor == RECOVERY_CONTRIBUTOR_ID,
            or_(
                Corpus.title_cn.like(f"%{title_key}%"),
                Corpus.title_en.like(f"%{title_key}%"),
            ),
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    db_rows = (
        db.session.query(Database.id)
        .filter(Database.is_deleted == 0, Database.database_name.in_(SCALE_DATABASE_NAMES))
        .all()
    )
    target_db_ids = [r[0] for r in db_rows]
    if len(target_db_ids) != len(SCALE_DATABASE_NAMES):
        print(f"警告: 仅找到 {len(target_db_ids)} 个目标库，预期 {len(SCALE_DATABASE_NAMES)} 个", file=sys.stderr)

    # 2. 展示候选与实际找到的目标库；预览在关联和审核写入前结束
    if args.dry_run:
        print(f"[dry-run] 将把以下 {len(rows)} 条题目含「{title_key}」的语料关联到库 {target_db_ids} 并设为审核通过:")
        for c in rows[:20]:
            print(f"  corpus_id={c.id} title_cn={(c.title_cn or '')[:50]}...")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
        return 0

    # 3. 只补缺失的主题关联并更新审核状态，避免重复插入已有关系
    for corpus in rows:
        current_ids = [
            r[0]
            for r in db.session.query(CorpusDatabase.database_id)
            .filter(CorpusDatabase.corpus_id == corpus.id)
            .all()
        ]
        for db_id in target_db_ids:
            if db_id not in current_ids:
                db.session.add(CorpusDatabase(corpus_id=corpus.id, database_id=db_id))
        corpus.status = CorpusStatus.APPROVE_EXAMINE.value

    # 4. 统一提交本专项变更，失败回滚；此脚本不在此处登记索引任务
    if rows:
        try:
            db.session.commit()
            print(f"已将 {len(rows)} 条题目含「{title_key}」的语料关联到指定库并设为审核通过")
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有符合条件的语料")
    return 0


def run_gudai(app, args):
    """将 reviewer_id=32 的待审核文献中、题目含「中国古代」的语料关联到 中国教育思想文献数据库（汉语）、中国古代教育思想文献 并设为审核通过。

    Args:
        app: 已由调用方建立上下文的 Flask 应用。
        args: dry_run 控制预览，limit 限制本次查询数量。

    Returns:
        完成或无候选时为 0，提交失败回滚后为 1。
    """
    from sqlalchemy import or_
    from app.exts import db
    from app.models import Corpus, Database, CorpusDatabase
    from app.utils.enums import CorpusStatus

    title_key = "中国古代"
    # 1. 按本专项的原有关键词、恢复身份和审核条件查询候选
    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.is_deleted == 0,
            Corpus.contributor == RECOVERY_CONTRIBUTOR_ID,
            Corpus.reviewer_id == ASSIGNED_REVIEWER_ID,
            Corpus.status == CorpusStatus.WAITING_EXAMINE.value,
            or_(
                Corpus.title_cn.like(f"%{title_key}%"),
                Corpus.title_en.like(f"%{title_key}%"),
            ),
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    db_rows = (
        db.session.query(Database.id)
        .filter(Database.is_deleted == 0, Database.database_name.in_(GUDAI_DATABASE_NAMES))
        .all()
    )
    target_db_ids = [r[0] for r in db_rows]
    if len(target_db_ids) != len(GUDAI_DATABASE_NAMES):
        print(f"警告: 仅找到 {len(target_db_ids)} 个目标库，预期 {len(GUDAI_DATABASE_NAMES)} 个", file=sys.stderr)

    # 2. 展示候选与实际找到的目标库；预览在关联和审核写入前结束
    if args.dry_run:
        print(f"[dry-run] 将把以下 {len(rows)} 条（reviewer_id=32、题目含「{title_key}」）关联到库 {target_db_ids} 并设为审核通过:")
        for c in rows[:20]:
            print(f"  corpus_id={c.id} title_cn={(c.title_cn or '')[:50]}...")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
        return 0

    # 3. 只补缺失的主题关联并更新审核状态，避免重复插入已有关系
    for corpus in rows:
        current_ids = [
            r[0]
            for r in db.session.query(CorpusDatabase.database_id)
            .filter(CorpusDatabase.corpus_id == corpus.id)
            .all()
        ]
        for db_id in target_db_ids:
            if db_id not in current_ids:
                db.session.add(CorpusDatabase(corpus_id=corpus.id, database_id=db_id))
        corpus.status = CorpusStatus.APPROVE_EXAMINE.value

    # 4. 统一提交本专项变更，失败回滚；此脚本不在此处登记索引任务
    if rows:
        try:
            db.session.commit()
            print(f"已将 {len(rows)} 条题目含「{title_key}」的语料（reviewer_id=32）关联到 中国教育思想文献数据库（汉语）、中国古代教育思想文献 并设为审核通过")
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有符合条件的语料")
    return 0


def run_jindai(app, args):
    """将 reviewer_id=32 审核的历史文献中、标题含「中国近代」的语料关联到 中国近现代教育思想/文献 子库并设为审核通过。

    Args:
        app: 已由调用方建立上下文的 Flask 应用。
        args: dry_run 控制预览，limit 限制本次查询数量。

    Returns:
        完成或无候选时为 0，提交失败回滚后为 1。
    """
    from sqlalchemy import or_
    from app.exts import db
    from app.models import Corpus, Database, CorpusDatabase
    from app.utils.enums import CorpusStatus

    title_key = "中国近代"
    # 1. 按本专项的原有关键词、恢复身份和审核条件查询候选
    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.is_deleted == 0,
            Corpus.reviewer_id == ASSIGNED_REVIEWER_ID,
            or_(
                Corpus.title_cn.like(f"%{title_key}%"),
                Corpus.title_en.like(f"%{title_key}%"),
            ),
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    db_rows = (
        db.session.query(Database.id)
        .filter(Database.is_deleted == 0, Database.database_name.in_(JINDAI_DATABASE_NAMES))
        .all()
    )
    target_db_ids = [r[0] for r in db_rows]
    if len(target_db_ids) != len(JINDAI_DATABASE_NAMES):
        print(f"警告: 仅找到 {len(target_db_ids)} 个目标库，预期 {len(JINDAI_DATABASE_NAMES)} 个", file=sys.stderr)

    # 2. 展示候选与实际找到的目标库；预览在关联和审核写入前结束
    if args.dry_run:
        print(f"[dry-run] 将把以下 {len(rows)} 条（reviewer_id=32、标题含「{title_key}」）关联到库 {target_db_ids} 并设为审核通过:")
        for c in rows[:20]:
            print(f"  corpus_id={c.id} title_cn={(c.title_cn or '')[:50]}...")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
        return 0

    # 3. 只补缺失的主题关联并更新审核状态，避免重复插入已有关系
    for corpus in rows:
        current_ids = [
            r[0]
            for r in db.session.query(CorpusDatabase.database_id)
            .filter(CorpusDatabase.corpus_id == corpus.id)
            .all()
        ]
        for db_id in target_db_ids:
            if db_id not in current_ids:
                db.session.add(CorpusDatabase(corpus_id=corpus.id, database_id=db_id))
        corpus.status = CorpusStatus.APPROVE_EXAMINE.value

    # 4. 统一提交本专项变更，失败回滚；此脚本不在此处登记索引任务
    if rows:
        try:
            db.session.commit()
            print(f"已将 {len(rows)} 条标题含「{title_key}」的文献（reviewer_id=32）关联到 中国近现代教育思想/文献 子库并设为审核通过")
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有符合条件的语料")
    return 0


def run_caiyuanpei(app, args):
    """将题目含「蔡元培」的文献关联到 中国近现代教育思想文献 子库并设为审核通过。

    Args:
        app: 已由调用方建立上下文的 Flask 应用。
        args: dry_run 控制预览，limit 限制本次查询数量。

    Returns:
        完成或无候选时为 0，提交失败回滚后为 1。
    """
    from sqlalchemy import or_
    from app.exts import db
    from app.models import Corpus, Database, CorpusDatabase
    from app.utils.enums import CorpusStatus

    title_key = "蔡元培"
    # 1. 按本专项的原有关键词、恢复身份和审核条件查询候选
    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.is_deleted == 0,
            or_(
                Corpus.title_cn.like(f"%{title_key}%"),
                Corpus.title_en.like(f"%{title_key}%"),
            ),
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    db_rows = (
        db.session.query(Database.id)
        .filter(Database.is_deleted == 0, Database.database_name.in_(CAIYUANPEI_DATABASE_NAMES))
        .all()
    )
    target_db_ids = [r[0] for r in db_rows]
    if len(target_db_ids) != len(CAIYUANPEI_DATABASE_NAMES):
        print(f"警告: 仅找到 {len(target_db_ids)} 个目标库，预期 {len(CAIYUANPEI_DATABASE_NAMES)} 个", file=sys.stderr)

    # 2. 展示候选与实际找到的目标库；预览在关联和审核写入前结束
    if args.dry_run:
        print(f"[dry-run] 将把以下 {len(rows)} 条（题目含「{title_key}」）关联到库 {target_db_ids} 并设为审核通过:")
        for c in rows[:20]:
            print(f"  corpus_id={c.id} title_cn={(c.title_cn or '')[:50]}...")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
        return 0

    # 3. 只补缺失的主题关联并更新审核状态，避免重复插入已有关系
    for corpus in rows:
        current_ids = [
            r[0]
            for r in db.session.query(CorpusDatabase.database_id)
            .filter(CorpusDatabase.corpus_id == corpus.id)
            .all()
        ]
        for db_id in target_db_ids:
            if db_id not in current_ids:
                db.session.add(CorpusDatabase(corpus_id=corpus.id, database_id=db_id))
        corpus.status = CorpusStatus.APPROVE_EXAMINE.value

    # 4. 统一提交本专项变更，失败回滚；此脚本不在此处登记索引任务
    if rows:
        try:
            db.session.commit()
            print(f"已将 {len(rows)} 条题目含「{title_key}」的文献关联到 中国近现代教育思想文献 子库并设为审核通过")
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有符合条件的语料")
    return 0


def run_xijinping(app, args):
    """将题目含「习近平」的文献关联到 中国当代教育思想数据库 并设为审核通过。

    Args:
        app: 已由调用方建立上下文的 Flask 应用。
        args: dry_run 控制预览，limit 限制本次查询数量。

    Returns:
        完成或无候选时为 0，提交失败回滚后为 1。
    """
    from sqlalchemy import or_
    from app.exts import db
    from app.models import Corpus, Database, CorpusDatabase
    from app.utils.enums import CorpusStatus

    title_key = "习近平"
    # 1. 按本专项的原有关键词、恢复身份和审核条件查询候选
    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.is_deleted == 0,
            or_(
                Corpus.title_cn.like(f"%{title_key}%"),
                Corpus.title_en.like(f"%{title_key}%"),
            ),
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    db_rows = (
        db.session.query(Database.id)
        .filter(
            Database.is_deleted == 0,
            Database.database_name.in_(XIJINPING_DATABASE_NAMES),
        )
        .all()
    )
    target_db_ids = [r[0] for r in db_rows]
    if len(target_db_ids) != len(XIJINPING_DATABASE_NAMES):
        print(
            f"警告: 仅找到 {len(target_db_ids)} 个目标库，预期 {len(XIJINPING_DATABASE_NAMES)} 个",
            file=sys.stderr,
        )

    # 2. 展示候选与实际找到的目标库；预览在关联和审核写入前结束
    if args.dry_run:
        print(
            f"[dry-run] 将把以下 {len(rows)} 条（题目含「{title_key}」）关联到库 {target_db_ids} 并设为审核通过:"
        )
        for c in rows[:20]:
            print(f"  corpus_id={c.id} title_cn={(c.title_cn or '')[:50]}...")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
        return 0

    # 3. 只补缺失的主题关联并更新审核状态，避免重复插入已有关系
    for corpus in rows:
        current_ids = [
            r[0]
            for r in db.session.query(CorpusDatabase.database_id)
            .filter(CorpusDatabase.corpus_id == corpus.id)
            .all()
        ]
        for db_id in target_db_ids:
            if db_id not in current_ids:
                db.session.add(
                    CorpusDatabase(corpus_id=corpus.id, database_id=db_id)
                )
        corpus.status = CorpusStatus.APPROVE_EXAMINE.value

    # 4. 统一提交本专项变更，失败回滚；此脚本不在此处登记索引任务
    if rows:
        try:
            db.session.commit()
            print(
                f"已将 {len(rows)} 条题目含「{title_key}」的文献关联到 中国当代教育思想数据库 并设为审核通过"
            )
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有符合条件的语料")
    return 0


def run_zhongguo_kouyi(app, args):
    """将题目含「中国口译」的文献关联到 中国外语教育实践子库 并设为审核通过。

    Args:
        app: 已由调用方建立上下文的 Flask 应用。
        args: dry_run 控制预览，limit 限制本次查询数量。

    Returns:
        完成或无候选时为 0，提交失败回滚后为 1。
    """
    from sqlalchemy import or_
    from app.exts import db
    from app.models import Corpus, Database, CorpusDatabase
    from app.utils.enums import CorpusStatus

    title_key = "中国口译"
    # 1. 按本专项的原有关键词、恢复身份和审核条件查询候选
    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.is_deleted == 0,
            or_(
                Corpus.title_cn.like(f"%{title_key}%"),
                Corpus.title_en.like(f"%{title_key}%"),
            ),
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    db_rows = (
        db.session.query(Database.id)
        .filter(
            Database.is_deleted == 0,
            Database.database_name.in_(ZHONGGUO_KOUYI_DATABASE_NAMES),
        )
        .all()
    )
    target_db_ids = [r[0] for r in db_rows]
    if len(target_db_ids) != len(ZHONGGUO_KOUYI_DATABASE_NAMES):
        print(
            f"警告: 仅找到 {len(target_db_ids)} 个目标库，预期 {len(ZHONGGUO_KOUYI_DATABASE_NAMES)} 个",
            file=sys.stderr,
        )

    # 2. 展示候选与实际找到的目标库；预览在关联和审核写入前结束
    if args.dry_run:
        print(
            f"[dry-run] 将把以下 {len(rows)} 条（题目含「{title_key}」）关联到库 {target_db_ids} 并设为审核通过:"
        )
        for c in rows[:20]:
            print(f"  corpus_id={c.id} title_cn={(c.title_cn or '')[:50]}...")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
        return 0

    # 3. 只补缺失的主题关联并更新审核状态，避免重复插入已有关系
    for corpus in rows:
        current_ids = [
            r[0]
            for r in db.session.query(CorpusDatabase.database_id)
            .filter(CorpusDatabase.corpus_id == corpus.id)
            .all()
        ]
        for db_id in target_db_ids:
            if db_id not in current_ids:
                db.session.add(
                    CorpusDatabase(corpus_id=corpus.id, database_id=db_id)
                )
        corpus.status = CorpusStatus.APPROVE_EXAMINE.value

    # 4. 统一提交本专项变更，失败回滚；此脚本不在此处登记索引任务
    if rows:
        try:
            db.session.commit()
            print(
                f"已将 {len(rows)} 条题目含「{title_key}」的文献关联到 中国外语教育实践子库 并设为审核通过"
            )
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有符合条件的语料")
    return 0


def run_lishi(app, args):
    """历史文献审核规则：将 reviewer_id=32 的待审核恢复语料按标题关键词放入对应库并设为审核通过。

    Args:
        app: 已由调用方建立上下文的 Flask 应用。
        args: dry_run 控制预览，limit 限制本次查询数量。

    Returns:
        完成或无候选时为 0，提交失败回滚后为 1。
    """
    from sqlalchemy import or_
    from app.exts import db
    from app.models import Corpus, Database, CorpusDatabase
    from app.utils.enums import CorpusStatus

    # 1. 按名称解析全部目标库，只使用本次实际存在的节点
    all_db_names = set()
    for _, db_names in LISHI_RULES:
        all_db_names.update(db_names)

    db_name_to_id = {
        r[0]: r[1]
        for r in db.session.query(Database.database_name, Database.id)
        .filter(Database.is_deleted == 0, Database.database_name.in_(all_db_names))
        .all()
    }

    missing = all_db_names - set(db_name_to_id)
    if missing:
        print(f"警告: 未找到以下目标库: {missing}", file=sys.stderr)

    total_processed = 0
    processed_ids = set()

    # 2. 按规则先后处理候选，以已处理 ID 集合防止同篇在本轮重复分配
    for keywords, db_names in LISHI_RULES:
        valid_db_ids = [db_name_to_id[name] for name in db_names if name in db_name_to_id]
        if not valid_db_ids:
            continue

        cond_list = []
        for kw in keywords:
            cond_list.append(Corpus.title_cn.like(f"%{kw}%"))
            cond_list.append(Corpus.title_en.like(f"%{kw}%"))

        query = (
            db.session.query(Corpus)
            .filter(
                Corpus.is_deleted == 0,
                Corpus.contributor == RECOVERY_CONTRIBUTOR_ID,
                Corpus.reviewer_id == ASSIGNED_REVIEWER_ID,
                Corpus.status == CorpusStatus.WAITING_EXAMINE.value,
                or_(*cond_list),
            )
            .order_by(Corpus.id.asc())
        )
        if args.limit and args.limit > 0:
            query = query.limit(args.limit)
        rows = query.all()

        if args.dry_run:
            print(
                f"[dry-run] 标题含{keywords}的 {len(rows)} 条语料将关联到库{valid_db_ids}并设为审核通过"
            )
            for c in rows[:20]:
                print(f"  corpus_id={c.id} title_cn={(c.title_cn or '')[:50]}...")
            if len(rows) > 20:
                print(f"  ... 共 {len(rows)} 条")
            continue

        for corpus in rows:
            if corpus.id in processed_ids:
                continue
            current_ids = [
                r[0]
                for r in db.session.query(CorpusDatabase.database_id)
                .filter(CorpusDatabase.corpus_id == corpus.id)
                .all()
            ]
            for db_id in valid_db_ids:
                if db_id not in current_ids:
                    db.session.add(CorpusDatabase(corpus_id=corpus.id, database_id=db_id))
            corpus.status = CorpusStatus.APPROVE_EXAMINE.value
            processed_ids.add(corpus.id)
            total_processed += 1

    if args.dry_run:
        return 0

    # 3. 统一提交本轮主题关联及审核状态，失败整体回滚本轮数据库变更
    if total_processed:
        try:
            db.session.commit()
            print(f"已按历史文献规则处理 {total_processed} 条语料并设为审核通过")
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有符合历史文献规则的语料")
    return 0


def fix_empty_attachments(app, args):
    """将 attachment 为 '[]' 的 corpus 记录补上基于 title_cn 的单个 PDF 附件。

    Args:
        app: 已由调用方建立上下文的 Flask 应用。
        args: dry_run 控制预览，limit 限制本次查询数量。

    Returns:
        完成或无候选时为 0，提交失败回滚后为 1。
    """
    from app.exts import db
    from app.models import Corpus

    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.is_deleted == 0,
            Corpus.attachment == "[]",
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    if args.dry_run:
        print(f"[dry-run] 将为以下 {len(rows)} 条 attachment='[]' 的语料补全附件（基于 title_cn 生成 uploaded_files/corpus/<title_cn>.pdf）:")
        for c in rows[:20]:
            print(f"  corpus_id={c.id} title_cn={(c.title_cn or '')[:80]}...")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
        return 0

    updated = 0
    for corpus in rows:
        title_cn = (corpus.title_cn or "").strip()
        if not title_cn:
            # 没有中文标题的不自动补，避免写入错误路径
            print(f"  跳过 corpus_id={corpus.id}（title_cn 为空）")
            continue
        # 去除标题中的换行符，避免生成含 %0D%0A 的附件路径导致下载失败
        title_cn = re.sub(r'[/\x00\r\n]', '_', title_cn)
        file_path = f"uploaded_files/corpus/{title_cn}.pdf"
        attachment = [{"file_path": file_path, "word": 0}]
        corpus.attachment = json.dumps(attachment, ensure_ascii=False)
        updated += 1

    if updated:
        try:
            db.session.commit()
            print(f"已为 {updated} 条 attachment='[]' 的语料补全附件")
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有需要补全附件的语料")
    return 0


def run(app, args):
    """将匹配指定主题词且未分配的待审恢复文献统一分给预设审核人。

    Args:
        app: 调用方已建立上下文的 Flask 应用。
        args: limit 限定条数，dry_run 为真仅列出候选。

    Returns:
        完成或无候选为 0，提交失败回滚后为 1。
    """
    from sqlalchemy import or_
    from app.exts import db
    from app.models import Corpus
    from app.utils.enums import CorpusStatus

    # 构建「任一字段包含任一词」的条件（LIKE 匹配 JSON 或普通字符串均可）
    cond_list = []
    for word in KEYWORDS_DAO_RU_ZHONGGUO:
        cond_list.append(Corpus.title_cn.like(f"%{word}%"))
        cond_list.append(Corpus.title_en.like(f"%{word}%"))
        cond_list.append(Corpus.keyword_cn.like(f"%{word}%"))
        cond_list.append(Corpus.keyword_en.like(f"%{word}%"))
    match_condition = or_(*cond_list)

    query = (
        db.session.query(Corpus)
        .filter(
            Corpus.is_deleted == 0,
            Corpus.contributor == RECOVERY_CONTRIBUTOR_ID,
            Corpus.status == CorpusStatus.WAITING_EXAMINE.value,
            Corpus.reviewer_id.is_(None),
            match_condition,
        )
        .order_by(Corpus.id.asc())
    )
    if args.limit and args.limit > 0:
        query = query.limit(args.limit)
    rows = query.all()

    if args.dry_run:
        print(f"[dry-run] 将把以下 {len(rows)} 条语料设为 reviewer_id={ASSIGNED_REVIEWER_ID}:")
        for c in rows[:20]:
            print(f"  corpus_id={c.id} title_cn={ (c.title_cn or '')[:50] }...")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
        return 0

    for corpus in rows:
        corpus.reviewer_id = ASSIGNED_REVIEWER_ID
    if rows:
        try:
            db.session.commit()
            print(f"已将 {len(rows)} 条语料设置为 reviewer_id={ASSIGNED_REVIEWER_ID}")
        except Exception as e:
            db.session.rollback()
            print(f"commit error: {e}", file=sys.stderr)
            return 1
    else:
        print("没有符合条件的待分配语料")
    return 0


def main():
    """读取维护模式参数，按既有优先顺序选择预分配、专项入库或附件补全流程。"""
    parser = argparse.ArgumentParser(
        description="恢复语料预分配/批量入库：道儒中国→user32 或 中国英语能力等级量表→指定库并审核通过"
    )
    parser.add_argument("--dry-run", action="store_true", help="只列出将要更新的记录，不写库")
    parser.add_argument("--limit", type=int, default=0, help="最多处理条数，0 表示不限制")
    parser.add_argument("--scale", action="store_true", help="处理题目含「中国英语能力等级量表」的语料：关联指定库并设为审核通过")
    parser.add_argument("--gudai", action="store_true", help="reviewer_id=32 的待审核文献中题目含「中国古代」的语料→关联 中国教育思想文献数据库（汉语）、中国古代教育思想文献 并设为审核通过")
    parser.add_argument("--jindai", action="store_true", help="reviewer_id=32 审核的历史文献中标题含「中国近代」的语料→关联 中国近现代教育思想文献 子库并设为审核通过")
    parser.add_argument("--caiyuanpei", action="store_true", help="题目含「蔡元培」的文献→关联 中国近现代教育思想文献 子库并设为审核通过")
    parser.add_argument("--kouyi", action="store_true", help="题目含「中国口译」的文献→关联 中国外语教育实践子库 并设为审核通过")
    parser.add_argument("--xijinping", action="store_true", help="题目含「习近平」的文献→关联 中国当代教育思想数据库 并设为审核通过")
    parser.add_argument("--lishi", action="store_true", help="历史文献审核规则：高校外语/日本/明代/民国/清代/清末→对应库并设为审核通过")
    parser.add_argument(
        "--fix-empty-attachments",
        action="store_true",
        help="将 attachment='[]' 的 corpus 记录补全为基于 title_cn 的单个 PDF 附件",
    )
    args = parser.parse_args()

    from flask import Flask
    from app.exts import db

    app = Flask(__name__)
    app.config.from_object("app.config")
    db.init_app(app)
    with app.app_context():
        if args.fix_empty_attachments:
            return fix_empty_attachments(app, args)
        if args.scale:
            return run_scale(app, args)
        if args.gudai:
            return run_gudai(app, args)
        if args.jindai:
            return run_jindai(app, args)
        if args.caiyuanpei:
            return run_caiyuanpei(app, args)
        if args.kouyi:
            return run_zhongguo_kouyi(app, args)
        if args.xijinping:
            return run_xijinping(app, args)
        if args.lishi:
            return run_lishi(app, args)
        return run(app, args)


if __name__ == "__main__":
    sys.exit(main())
