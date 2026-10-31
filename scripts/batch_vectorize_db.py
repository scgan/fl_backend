"""数据库驱动的批量向量化脚本：从数据库读取待向量化文献进行向量化。

用法：
    python scripts/batch_vectorize_db.py                        # 全部待处理文献（并发数=3）
    python scripts/batch_vectorize_db.py --limit 10            # 仅处理前 10 条
    python scripts/batch_vectorize_db.py --corpus-id 123      # 仅处理指定文献
    python scripts/batch_vectorize_db.py --dry-run            # 仅预览，不写入
    python scripts/batch_vectorize_db.py --workers 5          # 并发数=5
    python scripts/batch_vectorize_db.py --force              # 强制重新向量化已向量化的文献
"""

import os
import sys
import json
import argparse
import requests
import tempfile
import time

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
from app.models import Corpus
from app.utils.enums import CorpusStatus
from app.utils import qdrant_utils
from typing import List
from sqlalchemy import or_

MAX_PDF_PART_PAGES = 200

# ========== 拆分 PDF ==========
def split_pdf_if_needed(file_path: str, max_pages: int = MAX_PDF_PART_PAGES) -> List[str]:
    """把超过页数上限的 PDF 按原顺序拆分到同目录。

    Args:
        file_path: 本地 PDF 路径。
        max_pages: 每片页数上限，调用方须提供正数。

    Returns:
        分片路径列表；无需拆分、依赖缺失或拆分异常时返回原路径列表。
        临时分片清理由调用方管理，本函数不删除已生成文件。
    """
    try:
        from PyPDF2 import PdfReader, PdfWriter
        
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        
        if total_pages <= max_pages:
            return [file_path]
        
        print(f"[SPLIT] {os.path.basename(file_path)} 共 {total_pages} 页，超过限制 {max_pages} 页，正在拆分...")
        
        base_name = os.path.splitext(file_path)[0]
        output_files = []
        
        for start in range(0, total_pages, max_pages):
            writer = PdfWriter()
            end = min(start + max_pages, total_pages)
            
            for i in range(start, end):
                writer.add_page(reader.pages[i])
            
            output_path = f"{base_name}_part{start//max_pages + 1}.pdf"
            with open(output_path, 'wb') as f:
                writer.write(f)
            output_files.append(output_path)
            print(f"  ✅ 生成: {os.path.basename(output_path)} ({end-start} 页)")
        
        return output_files
        
    except ImportError:
        print(f"[WARN] PyPDF2 未安装，跳过拆分: {file_path}")
        return [file_path]
    except Exception as e:
        print(f"[WARN] 拆分失败，使用原文件: {e}")
        return [file_path]

def _get_pending_corpora(args):
    """从数据库查询待向量化的语料"""
    with flask_app.app_context():
        from app.exts import db
        
        # 基础过滤条件：未删除、审核通过、存在附件
        query = Corpus.query.filter(
            Corpus.is_deleted == 0,
            Corpus.status == CorpusStatus.APPROVE_EXAMINE.value,
        )
        
        # 如果不强制重处理，只查询未向量化的
        if not args.force:
            query = query.filter(or_(Corpus.is_vectorized.is_(None), Corpus.is_vectorized != True))
        
        # 如果指定了 corpus_id
        if args.corpus_id:
            query = query.filter(Corpus.id == args.corpus_id)
        
        # 获取结果
        corpora = query.order_by(Corpus.id).all()
        
        return corpora

def download_file_from_service(file_path: str) -> bytes:
    """通过文件服务获取文件内容"""
    file_url = os.environ.get("FOREIGN_FILE_URL", "http://10.222.9.151:9101")
    response = requests.get(f"{file_url.rstrip('/')}/files",
                            params={"filename": file_path}, timeout=(10, 120))
    if response.status_code == 200:
        return response.content
    raise RuntimeError(f"文件服务返回错误: {response.status_code}")

def _process_one(corpus_id: int, attachments: list, corpus_meta: dict, args) -> dict:
    """在线程应用上下文中下载、按需分片并重建单篇文献的双路索引。

    Args:
        corpus_id: 当前文献 ID。
        attachments: 附件路径与真实文件名列表。
        corpus_meta: 当前题录快照。
        args: 保留旧批处理调用参数，本函数当前不据此跳过校验或发布。

    Returns:
        文献 ID、附件名、构建统计或错误字典；退出时尽力清理下载及拆分临时文件。

    分片引用仍保存原附件 source_file_path，page_offset 按原文页数推进，
    不可把临时路径和分片页码写成用户可访问的原文位置。
    """
    with flask_app.app_context():
        temp_files = []
        try:
            temp_attachments = []
            # 1. 下载每个原附件并按需分片，同时保留原文件引用和原页偏移
            for a in attachments:
                file_path = a.get('file_path')
                file_name = a.get('file_name', 'unknown')
                
                if not file_path:
                    continue
                
                # 1. 从文件服务下载
                content = download_file_from_service(file_path)
                
                # 2. 保存为临时文件
                suffix = os.path.splitext(file_name)[1] or '.pdf'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                    temp_files.append(tmp_path)
                
                # 🔧 ========== 3. 新增：自动拆分大 PDF ==========
                split_paths = split_pdf_if_needed(tmp_path, max_pages=MAX_PDF_PART_PAGES)
                
                for i, split_path in enumerate(split_paths):
                    # 如果是拆分出来的，记录临时文件以便清理
                    if len(split_paths) > 1:
                        # 拆分生成的文件需要记录，最后一起清理
                        if split_path not in temp_files:
                            temp_files.append(split_path)
                        split_name = f"{os.path.splitext(file_name)[0]}_part{i+1}.pdf"
                    else:
                        split_name = file_name
                    
                    temp_attachments.append({
                        'file_path': split_path,
                        'source_file_path': file_path,
                        'file_name': file_name,
                        'chunk_name': split_name,
                        'page_offset': i * MAX_PDF_PART_PAGES if len(split_paths) > 1 else 0,
                    })

            if not temp_attachments:
                raise RuntimeError("文献没有可索引的附件")

            # 重试也可能留下部分旧片段；原文到位后统一清理整篇，失败时保持未向量化。
            from app.exts import db
            corpus = db.session.get(Corpus, corpus_id)
            # 2. 先提交不可检索状态，再清理旧的双路快照并从同一分片集重建
            corpus.is_vectorized = False
            db.session.commit()
            qdrant_utils.delete_corpus_chunks(corpus_id)
            
            # 4. 调用 qdrant_utils.index_corpus_chunks
            #    注意：同一个 corpus_id，多个附件（拆分后的文件）共享同一份 corpus_meta
            result = qdrant_utils.index_corpus_chunks(
                corpus_id, temp_attachments,
                corpus_meta=corpus_meta,
                base_dir=None,
            )
            if not result['indexed'] or not result['total_chunks']:
                raise RuntimeError("原文未生成可检索片段")
            return {
                "corpus_id": corpus_id,
                "files": [a.get('file_name', 'unknown') for a in attachments],
                "result": result,
                "error": None
            }
        except Exception as e:
            return {
                "corpus_id": corpus_id,
                "files": [a.get('file_name', 'unknown') for a in attachments],
                "result": None,
                "error": str(e)
            }
        finally:
            # 清理所有临时文件（包括拆分生成的文件）
            # 3. 无论构建成功或失败均尽力清理下载和分片副本，原始远程附件不受影响
            for tmp_path in temp_files:
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                except Exception:
                    pass

def _prepare_corpus_attachments(corpus: Corpus):
    """将文献 ORM 数据转换为线程可传递的附件引用和题录快照。

    Args:
        corpus: 含持久化附件 JSON 与题录的文献实例。

    Returns:
        (文献 ID, 附件列表, 题录字典)，缺失或损坏附件 JSON 对应空列表。
        abs_path 仅记录现有挂载命中位置，原 file_path 仍保留用于文件服务下载。
    """
    # 解析 attachment 字段
    # 附件循环只准备线程输入，不在此解析或提交索引；原始路径和显示名分别保留。
    attachments = []
    if corpus.attachment:
        try:
            attachment_list = json.loads(corpus.attachment)
            for item in attachment_list:
                file_path = item.get('file_path', '')
                original_filename = item.get('original_filename', '')
                
                if not file_path:
                    continue
                
                # 构造绝对路径（尝试多种基准目录）
                abs_path = None
                for base in [
                    "/home/ccnu/Foreign/FileDev",
                    "/foreign-files",
                    "/app",
                ]:
                    candidate = os.path.join(base, file_path.lstrip('/'))
                    if os.path.isfile(candidate):
                        abs_path = candidate
                        break
                
                # 也尝试直接使用 file_path
                if not abs_path and os.path.isfile(file_path):
                    abs_path = file_path
                
                attachments.append({
                    'file_path': file_path,  # 相对路径传给 qdrant_utils
                    'file_name': original_filename or file_path.split('/')[-1],
                    'abs_path': abs_path,
                })
        except json.JSONDecodeError:
            pass
    
    # 构建 corpus_meta
    corpus_meta = {
        'corpus_id': corpus.id,
        'title_cn': corpus.title_cn,
        'title_en': corpus.title_en,
        'author': corpus.author,
        'publish_time': corpus.publish_time,
        'category_name': None,  # 后续 qdrant_utils 会查库补全
    }
    
    return corpus.id, attachments, corpus_meta


def main():
    """从数据库筛选待处理文献，默认实际并发重建双路索引，--dry-run 仅打印候选。"""
    import concurrent.futures
    
    parser = argparse.ArgumentParser(description="批量向量化文献附件（从数据库读取）")
    parser.add_argument('--limit', type=int, default=0,
                        help='最多处理多少条（0 表示不限制）')
    parser.add_argument('--corpus-id', type=int, default=None,
                        help='仅处理指定文献 ID')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅预览，不实际写入 Qdrant')
    parser.add_argument('--skip-verify', action='store_true',
                        help='跳过文件存在性预检')
    parser.add_argument('-j', '--jobs', type=int, default=3,
                        help='并发处理线程数（默认 3）')
    parser.add_argument('--force', action='store_true',
                        help='强制重新向量化已向量化的文献（会先删除旧索引）')
    args = parser.parse_args()
    
    # 查询待处理语料
    print("[DB] 从数据库查询待向量化的语料...")
    # 1. 按当前数据库状态固定候选，预览在下载和模型调用前结束
    corpora = _get_pending_corpora(args)
    
    # 应用 limit
    target_corpora = corpora[:args.limit] if args.limit > 0 else corpora
    total = len(target_corpora)
    
    print(f"[READY] 待处理 {total} 个语料，并发数={args.jobs}")
    if args.force:
        print("[INFO] 强制模式：会先删除旧索引后重新向量化")
    
    if args.dry_run:
        for i, corpus in enumerate(target_corpora, 1):
            attachments_count = 0
            if corpus.attachment:
                try:
                    attachments_count = len(json.loads(corpus.attachment))
                except:
                    pass
            print(f"  [{i}/{total}] ID={corpus.id} 标题={corpus.title_cn or 'N/A'} 附件数={attachments_count} 已向量化={bool(corpus.is_vectorized)}")
        print(f"\n完成: 预览 {total} 条")
        return
    
    # 处理
    success, failed, skipped = 0, 0, 0
    results = []
    
    # 2. 将 ORM 记录转换为题录及附件快照后提交工作项，避免线程共享请求会话
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        for corpus in target_corpora:
            corpus_id, attachments, corpus_meta = _prepare_corpus_attachments(corpus)
            
            if not attachments:
                print(f"[SKIP] ID={corpus_id} 原因=无语料附件")
                skipped += 1
                continue
            
            future = executor.submit(_process_one, corpus_id, attachments, corpus_meta, args)
            futures[future] = corpus_id
        
        # 3. 按完成结果累计成功、失败与跳过；单篇异常不撤销已经完成的文献
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            corpus_id = futures[future]
            try:
                res = future.result()
                if res['error']:
                    print(f"  [{i}/{total}] ID={corpus_id} 附件: {res['files']} -> 失败: {res['error']}")
                    failed += 1
                else:
                    indexed = res['result']['indexed']
                    chunks = res['result']['total_chunks']
                    print(f"  [{i}/{total}] ID={corpus_id} 附件: {res['files']} -> 成功: {indexed} 个文件, {chunks} 个片段")
                    success += 1
                results.append(res)
            except Exception as e:
                print(f"  [{i}/{total}] ID={corpus_id} -> 异常: {str(e)}")
                failed += 1
    
    print(f"\n完成: 成功 {success}, 失败 {failed}, 跳过(无附件) {skipped}")


if __name__ == '__main__':
    main()
