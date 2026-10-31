"""在测试环境重建指定知识库，并核验双路索引和 token 硬上限。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.app import app
from app.exts import db
from app.models import Corpus, CorpusIndexJob, KnowledgeBase, KnowledgeBaseDocument
from app.indexing.jobs import enqueue_corpus_index, pending_status
from app.indexing.opensearch_store import get_lexical_store
from app.indexing.service import build_indexes, IndexBusy
from app.utils import qdrant_utils
from app.utils.embedding_tokens import count_tokens
from app.utils.bailian_client import embedding_config
from qdrant_client import models


def main():
    """限定测试库、知识库和索引目标后，按 mode 预览、入队、核验或直接重建。

    verify 核对双路构建、哈希、维度及 token 预算；repair/rebuild 会取消活动旧代次，
    并在线程各自的应用上下文中调用模型和索引服务，随完成结果更新 JSON 报告。

    Raises:
        AssertionError: 数据库、索引、文件服务、知识库或并发不满足预设测试目标。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan', 'enqueue', 'status', 'verify', 'repair', 'rebuild'])
    parser.add_argument('--base-id', type=int, required=True)
    parser.add_argument('--corpus-id', type=int, action='append',
                        help='只处理本测试知识库中的指定文献；可重复传入')
    parser.add_argument('--report', required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    # 所有写操作仅允许已经确认的测试部署。
    # 1. 强制核对数据库、索引、文件服务及知识库身份，只允许预设测试目标
    assert os.environ['FOREIGN_DBNAME'] == 'foreign_language_test'
    assert qdrant_utils.COLLECTION_NAME == 'corpus_chunks'
    assert os.environ['FOREIGN_OPENSEARCH_INDEX'] == 'foreign_chunks_v1'
    assert app.config['FILE_URL'].rstrip('/') == 'http://10.222.9.151:9101'
    assert 1 <= args.workers <= 8
    with app.app_context():
        base = db.session.get(KnowledgeBase, args.base_id)
        assert base and not base.is_deleted and base.name == '测试'
        rows = (db.session.query(Corpus).join(KnowledgeBaseDocument,
                    KnowledgeBaseDocument.corpus_id == Corpus.id)
                .filter(KnowledgeBaseDocument.knowledge_base_id == args.base_id,
                        KnowledgeBaseDocument.is_deleted == False).order_by(Corpus.id).all())
        if args.corpus_id:
            selected = set(args.corpus_id)
            assert selected <= {row.id for row in rows}, '指定文献不属于该测试知识库'
            rows = [row for row in rows if row.id in selected]
        config = embedding_config()
        report = {'base_id': base.id, 'name': base.name, 'embedding_model': config.model,
                  'embedding_dimension': config.dimension, 'ocr_model': os.getenv('FOREIGN_OCR_MODEL', 'qwen-vl-ocr'),
                  'documents': []}
        # 2. 固定候选状态，按模式登记任务或逐点核对双路构建、正文哈希和预算
        for corpus in rows:
            job = db.session.get(CorpusIndexJob, corpus.id)
            item = {'id': corpus.id, 'title': corpus.title_cn, 'status': job.status if job else None,
                    'chunks': job.chunk_count if job else 0,
                    'error': job.error_message if job else None}
            eligible = not corpus.is_deleted and corpus.status == 2 and bool(corpus.attachment)
            if not eligible:
                item['skipped'] = '未审核、已删除或缺少原文附件'
            # 此分支只在当前事务中登记新代次，循环结束后统一提交，其他模式不在此入队。
            elif args.mode == 'enqueue':
                item['previous_build_id'] = job.build_id if job else None
                enqueue_corpus_index(db.session, corpus)
                item['generation'] = job.generation if job else 1
                item['status'] = 'PENDING'
            elif args.mode in ('verify', 'repair') and job and job.status == 'SUCCESS':
                verify_build = job.build_id
                verify_ready = bool(corpus.is_vectorized and job.vector_status == job.lexical_status == 'SUCCESS')
                # 核验外部存储可能重试很久，不持有闲置 MySQL 事务和旧连接。
                db.session.rollback()
                points, offset = [], None
                client = qdrant_utils.get_qdrant_client()
                while True:
                    batch, offset = client.scroll(qdrant_utils.COLLECTION_NAME,
                        scroll_filter=models.Filter(must=[models.FieldCondition(
                            key='corpus_id', match=models.MatchValue(value=corpus.id))]),
                        limit=100, offset=offset, with_payload=True, with_vectors=False)
                    points.extend(batch)
                    if offset is None:
                        break
                ids = [str(p.id) for p in points]
                store = get_lexical_store()
                lexical = store.client.mget(index=store.index_name, body={'ids':ids})['docs'] if ids else []
                lmap = {p['_id']: p.get('_source') for p in lexical if p.get('found')}
                errors = []
                for point in points:
                    p = point.payload
                    if (p.get('embedding_model') != config.model
                            or p.get('embedding_dimension') != config.dimension
                            or p.get('embedding_text_type') != 'document'
                            or p.get('token_count_kind') != 'chunking_estimate'
                            or p.get('chunking_strategy') != 'embedding_tokens_v1'
                            or p.get('token_count') != count_tokens(p['text'])
                            or p['token_count'] > 768
                            or p.get('build_id') != verify_build
                            or hashlib.sha256(p['text'].encode()).hexdigest() != p.get('content_hash')):
                        errors.append(str(point.id))
                    lex = lmap.get(str(point.id), {})
                    if any(lex.get(k) != p.get(k) for k in ['build_id','content_hash','attachment_digest']):
                        errors.append(str(point.id))
                lexical_count = store.client.count(index=store.index_name,
                    body={'query':{'term':{'corpus_id':corpus.id}}})['count']
                sample = client.retrieve(qdrant_utils.COLLECTION_NAME, ids=[points[0].id],
                    with_vectors=True, with_payload=False) if points else []
                sample_dimension = len(sample[0].vector) if sample and isinstance(sample[0].vector, list) else 0
                if sample_dimension != config.dimension:
                    errors.append('vector_dimension')
                item.update(verified=bool(not errors and len(points) == len(lmap) == lexical_count == item['chunks']
                            and bool(points) and verify_ready),
                            invalid_points=errors,
                            sample_vector_dimension=sample_dimension,
                            max_tokens=max((count_tokens(p.payload['text']) for p in points), default=0))
            report['documents'].append(item)
        if args.mode == 'enqueue':
            db.session.commit()
        summary = {}
        for item in report['documents']:
            label = 'SKIPPED' if item.get('skipped') else item['status'] or 'NO_INDEX'
            summary[label] = summary.get(label, 0) + 1
        report['summary'] = summary
        if args.mode in ('verify', 'repair'):
            report['verified_documents'] = sum(x.get('verified', False) for x in report['documents'])
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in report.items() if k != 'documents'}, ensure_ascii=False), flush=True)
        # 3. 先写出核验报告；修复或重建时取消活动旧代次，再并发直接构建目标文献
        if args.mode in ('repair', 'rebuild'):
            targets = [x['id'] for x in report['documents'] if not x.get('skipped')
                       and (args.mode == 'rebuild' or not x.get('verified'))]
            # 撤销这些队列请求。随后直接持锁构建，不给旧版本消费者新的 PENDING 任务。
            for cid in targets:
                job = db.session.query(CorpusIndexJob).filter_by(corpus_id=cid).with_for_update().one_or_none()
                if job and job.status in ('PENDING', pending_status(), 'RUNNING'):
                    job.generation += 1
                    job.status = 'CANCELLED'
            db.session.commit()
            print(json.dumps({'direct_rebuild_ids':targets}), flush=True)
            def rebuild(cid):
                """持有文献锁重建指定测试文献，只有锁冲突才等待并重新建立会话。

                Args:
                    cid: 已校验属于测试知识库的文献 ID。

                Returns:
                    文献 ID 及构建统计或错误；锁冲突最多等待一百八十轮。
                """
                with app.app_context():
                    for attempt in range(180):
                        corpus = db.session.get(Corpus, cid)
                        attachments = json.loads(corpus.attachment)
                        if not isinstance(attachments, list):
                            attachments = [attachments]
                        try:
                            return {'id':cid, **build_indexes(cid, attachments)}
                        except IndexBusy:
                            db.session.remove()
                            time.sleep(5)
                        except Exception as error:
                            return {'id':cid, 'error':str(error)}
                    return {'id':cid, 'error':'仍有旧任务持有文献锁'}
            results = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(rebuild, cid) for cid in targets]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(json.dumps(result, ensure_ascii=False), flush=True)
                    report['direct_results'] = results
                    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
