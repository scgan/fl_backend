#!/usr/bin/env python3
"""
修复 corpus.attachment 中 file_path 包含换行符（%0D%0A / \r / \n）的历史数据。

背景：
- 早期上传时文件名中的 \r\n 被保留或编码为 %0D%0A，保存到磁盘后导致 HTTP 下载时协议解析失败。
- 本脚本会把命中文件下载到本地、用清理后的文件名重新上传到文件服务、更新数据库中的 file_path，
  并尝试删除文件服务上的旧文件。

用法（默认 dry-run，必须加 --commit 才写库/删旧文件）：
  python scripts/fix_attachment_newlines.py --dry-run
  python scripts/fix_attachment_newlines.py --commit
  python scripts/fix_attachment_newlines.py --limit 100 --commit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Set, Tuple

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


# 匹配需要修复的 file_path 特征：URL 编码的 CRLF 或字面量 CR/LF
_NEWLINE_RE = re.compile(r"%0D%0A|%0D|%0A|[\r\n]")


def _parse_attachment(attachment_text: str) -> List[Dict[str, Any]] | None:
    """解析附件 JSON 数组并保留字典项，非法 JSON 或非数组返回 None。"""
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


def _needs_repair(file_path: str) -> bool:
    """判断文件路径是否包含实际换行字符，供修复候选筛选使用。"""
    return isinstance(file_path, str) and _NEWLINE_RE.search(file_path) is not None


def _clean_filename(file_name: str) -> str:
    """将文件名中的 CRLF/CR/LF 及 URL 编码形式替换为单个下划线，并压缩连续下划线。"""
    cleaned = _NEWLINE_RE.sub("_", file_name)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned


def _split_prefix_and_name(file_path: str) -> Tuple[str, str]:
    """按最后一个正斜杠拆分远程目录和文件名，不处理 URL 语义。"""
    if "/" not in file_path:
        return "", file_path
    prefix, name = file_path.rsplit("/", 1)
    return prefix, name


def _collect_existing_paths() -> Set[str]:
    """收集当前所有 corpus 记录中在用的 file_path，用于检测清理后的文件名是否冲突。"""
    from app.exts import db
    from app.models import Corpus

    existing: Set[str] = set()
    rows = db.session.query(Corpus.attachment).filter(
        Corpus.is_deleted == 0, Corpus.attachment.isnot(None)
    ).all()
    for row in rows:
        att_list = _parse_attachment(row[0])
        if not att_list:
            continue
        for item in att_list:
            fp = item.get("file_path")
            if isinstance(fp, str) and fp:
                existing.add(fp)
    return existing


def _find_corrupted_attachments(limit: int = 0):
    """返回 (corpus, attachment_index, item) 列表。"""
    from app.exts import db
    from app.models import Corpus

    q = db.session.query(Corpus).filter(
        Corpus.is_deleted == 0, Corpus.attachment.isnot(None)
    ).order_by(Corpus.id.asc())
    if limit and limit > 0:
        q = q.limit(limit)

    result = []
    for corpus in q:
        att_list = _parse_attachment(corpus.attachment)
        if not att_list:
            continue
        for idx, item in enumerate(att_list):
            if _needs_repair(item.get("file_path", "")):
                result.append((corpus, idx, item))
    return result


def _download_to_temp(file_path: str) -> Tuple[str, str] | Tuple[None, str]:
    """下载文件到临时文件，返回 (temp_path, content_type) 或 (None, error_str)。"""
    from app.utils.utils import get_file_bytes

    result, code_or_type = get_file_bytes(file_path)
    # 失败时第二个返回值为状态码（int）；成功时为 content_type（str）
    if isinstance(code_or_type, int):
        return None, f"下载失败({code_or_type}): {result}"
    if not isinstance(result, bytes):
        return None, f"下载结果类型异常: {type(result)}"

    content_type = code_or_type if isinstance(code_or_type, str) else "application/octet-stream"
    fd, temp_path = tempfile.mkstemp(suffix=".pdf")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(result)
        return temp_path, content_type
    except Exception as e:
        os.close(fd)
        return None, f"写入临时文件失败: {e}"


def _upload_with_clean_name(
    temp_path: str,
    new_filename: str,
    content_type: str,
    word: int,
    username: str = "system_repair",
) -> Tuple[str | None, str]:
    """用清理后的文件名重新上传，返回 (new_file_path, error_str)。"""
    from app.utils.utils import upload_file

    f = None
    try:
        f = open(temp_path, "rb")
        file_obj = SimpleNamespace(
            filename=new_filename,
            stream=f,
            content_type=content_type,
        )
        upload_result, code = upload_file([(file_obj, word)], "corpus", username)
    except Exception as e:
        return None, f"上传异常: {e}"
    finally:
        try:
            if f is not None:
                f.close()
        except Exception:
            pass

    if code != 200:
        return None, f"上传失败({code}): {upload_result}"
    paths = upload_result
    if not paths or not isinstance(paths, list):
        return None, "上传返回路径为空或格式错误"
    return paths[0], ""


def run(app, args) -> int:
    """清理附件名中的换行，先上传替代文件，再提交引用，最后尽力删除旧文件。

    Args:
        app: 已初始化的 Flask 应用，调用方已进入其上下文。
        args: 含 limit 和 dry_run 的命令行配置。

    Returns:
        正常结束或预览为 0，数据库提交失败为 2；单文件失败在汇总中报告。

    提交失败会回滚数据库，但已上传的新远程文件不在本函数中自动补偿删除。
    """
    from app.exts import db
    from app.models import Corpus
    from app.utils.utils import delete_file

    # 1. 找出换行污染的附件引用，并读取已有路径集合用于避免新文件名冲突
    corrupted = _find_corrupted_attachments(args.limit)
    if not corrupted:
        print("未发现需要修复的附件")
        return 0

    print(f"发现 {len(corrupted)} 个需要修复的附件")

    existing_paths = _collect_existing_paths()
    old_paths_to_delete: List[str] = []
    updated_corpus_ids: Set[int] = set()
    success_count = 0
    fail_count = 0

    # 2. 逐附件清理名称并重新上传，保留真实返回路径；每次下载的临时副本独立清理
    for corpus, idx, item in corrupted:
        old_path = item.get("file_path", "")
        original_filename = item.get("original_filename", "")
        word = item.get("word") or 0

        prefix, file_name = _split_prefix_and_name(old_path)
        cleaned_name = _clean_filename(file_name)
        if not cleaned_name:
            print(f"  跳过 corpus_id={corpus.id}，清理后文件名为空: {old_path}")
            fail_count += 1
            continue

        # 处理文件名冲突
        new_path = f"{prefix}/{cleaned_name}" if prefix else cleaned_name
        base, ext = os.path.splitext(new_path)
        suffix = 1
        while new_path in existing_paths:
            new_path = f"{base}_{suffix}{ext}"
            suffix += 1

        if args.dry_run:
            print(f"[dry-run] corpus_id={corpus.id}: {old_path} -> {new_path}")
            continue

        # 下载原文件
        temp_path, err = _download_to_temp(old_path)
        if temp_path is None:
            print(f"  失败 corpus_id={corpus.id}: {err}")
            fail_count += 1
            continue

        try:
            # 上传新文件
            actual_new_path, err = _upload_with_clean_name(
                temp_path, os.path.basename(new_path), "application/pdf", word
            )
            if actual_new_path is None:
                print(f"  失败 corpus_id={corpus.id}: {err}")
                fail_count += 1
                continue

            # 更新数据库
            att_list = _parse_attachment(corpus.attachment)
            if att_list is None or idx >= len(att_list):
                print(f"  失败 corpus_id={corpus.id}: attachment 结构异常")
                fail_count += 1
                continue

            att_list[idx]["file_path"] = actual_new_path
            corpus.attachment = json.dumps(att_list, ensure_ascii=False)
            old_paths_to_delete.append(old_path)
            existing_paths.discard(old_path)
            existing_paths.add(actual_new_path)
            updated_corpus_ids.add(corpus.id)
            success_count += 1
            print(f"  成功 corpus_id={corpus.id}: {old_path} -> {actual_new_path}")
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

    if args.dry_run:
        return 0

    if not updated_corpus_ids:
        print("没有成功修复的语料")
        return 0

    # 提交数据库更新
    try:
        # 3. 全部成功项一起提交引用变更；新文件已在远程，提交失败并不会自动删除它们
        db.session.commit()
        print(f"已更新 {len(updated_corpus_ids)} 条语料记录，成功修复 {success_count} 个附件")
    except Exception as e:
        db.session.rollback()
        print(f"数据库提交失败，已回滚：{e}", file=sys.stderr)
        return 2

    # 删除文件服务上的旧文件（删除失败不影响主流程，仅打印警告）
    # 4. 只有新引用提交成功后才尽力删除旧文件，避免提交失败时原件先丢失
    if old_paths_to_delete:
        try:
            res, code = delete_file(old_paths_to_delete)
            print(f"旧文件删除结果({code}): {res}")
        except Exception as e:
            print(f"删除旧文件时发生异常：{e}", file=sys.stderr)

    if fail_count:
        print(f"注意：有 {fail_count} 个附件修复失败，可排查后重新运行脚本")
    return 0


def main() -> int:
    """解析修复范围和执行开关，默认预览，仅显式 --commit 才进行文件与数据库修改。"""
    parser = argparse.ArgumentParser(description="修复 corpus.attachment 中包含换行符的 file_path")
    parser.add_argument("--dry-run", action="store_true", help="只列出将要修复的记录，不写库、不删旧文件")
    parser.add_argument("--commit", action="store_true", help="真正执行修复（默认 dry-run）")
    parser.add_argument("--limit", type=int, default=0, help="最多扫描多少条 corpus 记录（0 表示不限制）")
    args = parser.parse_args()

    if not args.commit and not args.dry_run:
        print("提示：未加 --commit，默认以 dry-run 模式运行")
        args.dry_run = True

    from flask import Flask
    from app.exts import db

    app = Flask(__name__)
    app.config.from_object("app.config")
    db.init_app(app)

    with app.app_context():
        return run(app, args)


if __name__ == "__main__":
    raise SystemExit(main())
