"""批量向量化脚本：从 corpus 目录直接扫描 PDF 文件进行向量化。

用法：
    python scripts/batch_vectorize.py              # 全部待处理文献（并发数=3）
    python scripts/batch_vectorize.py --limit 10  # 仅处理前 10 条
    python scripts/batch_vectorize.py --corpus-id 123  # 仅处理指定文献
    python scripts/batch_vectorize.py --dry-run   # 仅预览，不写入
    python scripts/batch_vectorize.py --workers 5  # 并发数=5
"""

import os
import sys
import glob
import argparse
import json
import threading

# ── 索引文件基准目录（容器内 / 宿主机自适应） ─────────────────
_CONTAINER_BASE = "/foreign-files"
_HOST_BASE = "/home/ccnu/Foreign/FileDev"

def _resolve_index_base():
    """优先使用容器内路径，不存在则回退到宿主机路径"""
    if os.path.isdir(os.path.join(_CONTAINER_BASE, "uploaded_files", "corpus")):
        return _CONTAINER_BASE
    if os.path.isdir(os.path.join(_HOST_BASE, "uploaded_files", "corpus")):
        return _HOST_BASE
    raise RuntimeError(
        f"找不到 corpus 目录，请确认挂载路径：\n"
        f"  容器内: {_CONTAINER_BASE}/uploaded_files/corpus\n"
        f"  宿主机: {_HOST_BASE}/uploaded_files/corpus"
    )

INDEX_FILE_BASE_DIR = _resolve_index_base()
CORPUS_DIR = os.path.join(INDEX_FILE_BASE_DIR, "uploaded_files", "corpus")

# ── 硬编码环境变量（优先于 .env / start-dev.sh） ─────────────────
os.environ.setdefault("FOREIGN_SECRET_KEY", "THIS IS USED TO SIGN AND VERIFY JWT TOKENS, REPLACE IT WITH YOUR OWN SECRET, IT CAN BE ANY STRING")
os.environ.setdefault("FOREIGN_USERNAME", "root")
os.environ.setdefault("FOREIGN_PASSWORD", "ccnu#foreignlanguage")
os.environ.setdefault("FOREIGN_HOST", "10.222.9.151")
os.environ.setdefault("FOREIGN_DBNAME", "foreign_language_test")
os.environ.setdefault("FOREIGN_FILE_URL", "http://10.222.9.151:9101")

os.environ.setdefault("QWEN_API_KEY", "sk-2fb3bc7207c0413fb9dc6dacd2844ed8")
os.environ.setdefault("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
os.environ.setdefault("LLM_MODEL_NAME", "qwen3.8-flash")
os.environ.setdefault("LLM_TEMPERATURE", "0.7")
os.environ.setdefault("QA_MODEL_NAME", "qwen-long")
os.environ.setdefault("QA_TEMPERATURE", "0.3")

os.environ.setdefault("QDRANT_HOST", "qdrant")
os.environ.setdefault("QDRANT_PORT", "6333")
# ─────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.app import app as flask_app
from app.utils import qdrant_utils


def _scan_corpus_files():
    """扫描 corpus 目录，返回 {corpus_id: [(file_path, file_name), ...]}"""
    if not os.path.isdir(CORPUS_DIR):
        raise RuntimeError(f"目录不存在: {CORPUS_DIR}")

    pdf_files = glob.glob(os.path.join(CORPUS_DIR, "*.pdf"))
    result = {}

    for abs_path in pdf_files:
        file_name = os.path.basename(abs_path)
        # 文件名格式: corpus_id-原始文件名.pdf
        # 或者 corpus_id.pdf（无原始文件名）
        parts = file_name.rsplit("-", 1)
        if len(parts) == 2 and parts[0].isdigit():
            corpus_id = int(parts[0])
            original_name = parts[1]
        elif file_name.rsplit(".", 1)[0].isdigit():
            # 纯数字文件名，直接作为 corpus_id
            corpus_id = int(file_name.rsplit(".", 1)[0])
            original_name = file_name
        else:
            # 无法解析 corpus_id，用文件名 MD5 生成唯一 ID
            import hashlib
            corpus_id = int(hashlib.md5(file_name.encode()).hexdigest()[:12], 16) % (10 ** 12)
            original_name = file_name
            print(f"[WARN] 文件名无法解析 corpus_id，已用 hash 生成: {file_name} → {corpus_id}")

        if corpus_id not in result:
            result[corpus_id] = []
        result[corpus_id].append({"file_path": abs_path, "file_name": file_name, "abs_path": abs_path})

    return result


def _process_one(corpus_id: int, attachments: list, args, flask_app) -> dict:
    """处理单个 corpus，在线程内执行。"""
    from app.utils import qdrant_utils
    att_names = [a['file_name'] for a in attachments]
    # 预检：确认文件存在
    if not args.skip_verify:
        for a in attachments:
            if not os.path.isfile(a['abs_path']):
                raise RuntimeError(f"文件不存在: {a['abs_path']}")
    with flask_app.app_context():
        result = qdrant_utils.index_corpus_chunks(
            corpus_id, attachments,
            base_dir=INDEX_FILE_BASE_DIR,
        )
    return {"corpus_id": corpus_id, "files": att_names, "result": result, "error": None}


def main():
    """按目录扫描结果筛选文献，在线程池中重建索引并汇总逐篇结果。

    只有 --dry-run 才预览；实际并发由 --jobs 决定，保留旧 --workers 参数兼容。
    常规执行会访问数据库、文件和模型及双路索引服务，不是离线验证命令。
    """
    import concurrent.futures
    parser = argparse.ArgumentParser(description="批量向量化文献附件（从目录扫描）")
    parser.add_argument('--limit', type=int, default=0,
                        help='最多处理多少条（0 表示不限制）')
    parser.add_argument('--corpus-id', type=int, default=None,
                        help='仅处理指定文献 ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅预览，不实际写入 Qdrant')
    parser.add_argument('--skip-verify', action='store_true',
                        help='跳过文件存在性预检')
    parser.add_argument('--workers', type=int, default=3,
                        help='并发处理线程数（默认 3）')
    parser.add_argument('-j', '--jobs', type=int, default=4,
                        help='并发线程数（默认 4）')
    args = parser.parse_args()

    # 1. 扫描附件目录并按 ID 与数量限制固定本轮候选，预览分支只展示清单
    print(f"[SCAN] 扫描目录: {CORPUS_DIR}")
    corpus_files = _scan_corpus_files()
    all_ids = sorted(corpus_files.keys())
    print(f"[SCAN] 找到 {len(all_ids)} 个 corpus 目录")

    if args.corpus_id:
        target_ids = [args.corpus_id] if args.corpus_id in corpus_files else []
    else:
        target_ids = all_ids

    if args.limit > 0:
        target_ids = target_ids[:args.limit]

    total = len(target_ids)
    print(f"[{'DRY RUN' if args.dry_run else 'READY'}] 待处理 {total} 个 corpus，并发数={args.jobs}")

    if args.dry_run:
        for i, cid in enumerate(target_ids, 1):
            atts = corpus_files[cid]
            att_names = [a['file_name'] for a in atts]
            print(f"  [{i}/{total}] ID={cid} 附件: {att_names}")
        print(f"\n完成: 成功 {total}, 失败 0")
        return

    success, failed = 0, 0
    results = []

    # 2. 每篇在线程自己的应用上下文处理，按完成结果汇总，不要求按输入顺序结束
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(_process_one, cid, corpus_files[cid], args, flask_app): cid
            for cid in target_ids
        }
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            cid = futures[future]
            try:
                res = future.result()
                files = res['files']
                indexed = res['result']['indexed']
                chunks = res['result']['total_chunks']
                print(f"  [{i}/{total}] ID={cid} 附件: {files} -> 成功: {indexed} 个文件, {chunks} 个片段")
                success += 1
                results.append(res)
            except Exception as e:
                print(f"  [{i}/{total}] ID={cid} -> 失败: {str(e)}")
                failed += 1

    print(f"\n完成: 成功 {success}, 失败 {failed}")


if __name__ == '__main__':
    main()
