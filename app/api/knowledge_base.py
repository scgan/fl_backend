"""提供私有知识库和已审核文献集合的管理接口。

知识库使用 knowledge_base/knowledge_base_document 两张表。所有查询先限定
当前登录用户，文献候选及关联操作只允许已审核且未删除的文献。
"""

import json
import os

from flask import Blueprint, g, jsonify, request
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from app.exts import db
from app.models import Corpus, CorpusIndexJob, KnowledgeBase, KnowledgeBaseDocument
from app.indexing.jobs import attachment_digest, enqueue_corpus_index
from app.utils.decorators import permission_required


bp = Blueprint('knowledge_base', __name__)
ALLOWED_ROLES = ['ADMIN', 'LEADER', 'COLLECTOR']
MAX_BATCH_SIZE = 100


def _owned_base(base_id):
    """读取当前用户尚未删除的知识库。"""
    return KnowledgeBase.query.filter_by(id=base_id, owner_id=g.current_user.id, is_deleted=False).first()


def _eligible_corpus_query():
    """允许所有用户收录公开检索中可见的已审核文献。"""
    return Corpus.query.filter(Corpus.is_deleted == 0, Corpus.status == 2)


def _requested_corpus_ids():
    """读取一次批量操作的文献 ID，保持顺序并去重。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return None, '请求体必须是 JSON 对象'
    raw_ids = body.get('corpus_ids')
    if not isinstance(raw_ids, list) or not 1 <= len(raw_ids) <= MAX_BATCH_SIZE:
        return None, '请选择 1–100 篇文献'
    if any(type(value) is not int or value < 1 for value in raw_ids):
        return None, '文献 ID 必须是正整数'
    return list(dict.fromkeys(raw_ids)), None


def _page_args():
    """返回边界明确的分页参数。"""
    page = max(request.args.get('page_num', 1, type=int), 1)
    size = request.args.get('page_size', 20, type=int)
    return page, min(max(size, 1), 100)


def _corpus_data(corpus):
    """只输出知识库表格需要的文献题录字段。"""
    return {
        'corpus_id': corpus.id,
        'title_cn': corpus.title_cn,
        'title_en': corpus.title_en,
        'author': corpus.author,
        'language': corpus.language,
        'publish_time': str(corpus.publish_time) if corpus.publish_time else None,
    }


def _base_data(base, count=None):
    """构造不包含权限或可见性字段的知识库响应。"""
    data = base.to_dict()
    if count is not None:
        data['corpus_count'] = count
    return data


def _has_index_source(corpus):
    """判断当前附件清单是否含可提交的原文件。"""
    try:
        attachments = json.loads(corpus.attachment or '[]')
    except (ValueError, TypeError):
        return False
    if not isinstance(attachments, list):
        attachments = [attachments]
    return any(isinstance(item, dict) and item.get('file_path') for item in attachments)


def _index_data(corpus, job):
    """把持久化任务投影为页面索引状态，不凭旧向量标记宣告双路就绪。

    Args:
        corpus: 当前文献，提供附件指纹与就绪标记。
        job: 当前索引任务，可为 None 或已删除记录。

    Returns:
        页面状态、原因、可重建标记和有效切块数；附件或索引版本不匹配时为 STALE。
    """
    source_available = _has_index_source(corpus)
    current = bool(job and not job.is_deleted)
    status = job.status if current else 'NOT_INDEXED'
    from app.indexing.jobs import pending_status
    if status == pending_status():
        status = 'PENDING'
    digest_matches = current and job.attachment_digest == attachment_digest(corpus.attachment)
    if current and (not digest_matches or (status == 'SUCCESS' and not (
        corpus.is_vectorized and job.vector_status == job.lexical_status == 'SUCCESS'
        and job.build_id and job.lexical_index == os.getenv('FOREIGN_OPENSEARCH_INDEX', 'foreign_chunks_v1')
    ))):
        status = 'STALE'
    descriptions = {
        'NOT_INDEXED': '尚未建立索引', 'PENDING': '任务已排队，完成后自动更新',
        'RUNNING': '正在处理文献，请稍候', 'SUCCESS': '索引已完成，可用于研究检索',
        'FAILED': '索引失败，可尝试重新构建', 'CANCELLED': '任务已取消，可重新构建',
        'STALE': '索引需要更新，请重新构建',
    }
    description = descriptions.get(status, '索引状态未知，请刷新')
    if status == 'EXTRACTING':
        from app.services.personal_metadata import metadata_processing_active
        if metadata_processing_active(job):
            status, description = 'RUNNING', '正在提取文献题录，请稍候'
        else:
            status, description = 'FAILED', '前次题录提取已中断，请点击重试'
    retry_pending = bool(status == 'PENDING' and job.attempts > 0 and job.error_message)
    if retry_pending:
        description = f'上次处理失败，等待自动重试（已尝试 {job.attempts} 次）'
    if status == 'FAILED':
        error = job.error_message or ''
        if '题录' in error:
            description = '题录提取失败，可点击重试'
        elif '404' in error:
            description = '原附件未找到，请先补齐文件再重建'
        elif '文字层' in error:
            description = 'PDF 缺少文字层，请提供可提取文字的文件'
    if not source_available:
        description = '暂无可索引的附件，请先补齐原文件'
    return {
        'status': status, 'description': description,
        'retry_pending': retry_pending,
        'chunk_count': job.chunk_count if current and status == 'SUCCESS' else 0,
        'can_rebuild': source_available and status not in ('PENDING', 'RUNNING'),
    }


@bp.route('/knowledge-bases', methods=['GET'])
@permission_required(ALLOWED_ROLES)
def list_bases():
    """分页列出当前用户创建的知识库。"""
    page, size = _page_args()
    query = KnowledgeBase.query.filter_by(owner_id=g.current_user.id, is_deleted=False)
    total = query.count()
    bases = query.order_by(KnowledgeBase.update_time.desc(), KnowledgeBase.id.desc()).offset(
        (page - 1) * size
    ).limit(size).all()
    counts = {}
    if bases:
        rows = db.session.query(KnowledgeBaseDocument.knowledge_base_id, func.count(Corpus.id)).join(
            Corpus, Corpus.id == KnowledgeBaseDocument.corpus_id
        ).filter(
            KnowledgeBaseDocument.knowledge_base_id.in_([base.id for base in bases]),
            KnowledgeBaseDocument.is_deleted == 0,
            Corpus.is_deleted == 0,
            Corpus.status == 2,
        )
        counts = dict(rows.group_by(KnowledgeBaseDocument.knowledge_base_id).all())
    return jsonify({
        'code': 200,
        'data': [_base_data(base, counts.get(base.id, 0)) for base in bases],
        'pagination': {'page_num': page, 'page_size': size, 'total': total},
    })


@bp.route('/knowledge-bases', methods=['POST'])
@permission_required(ALLOWED_ROLES)
def create_base():
    """为当前用户创建默认私有的知识库。"""
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({'code': 400, 'error': '请求体必须是 JSON 对象'}), 400
    name = body.get('name')
    description = body.get('description', '')
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 127:
        return jsonify({'code': 400, 'error': '知识库名称须为 1–127 个字符'}), 400
    if not isinstance(description, str) or len(description.strip()) > 255:
        return jsonify({'code': 400, 'error': '知识库说明最多 255 个字符'}), 400
    base = KnowledgeBase(owner_id=g.current_user.id, name=name.strip(), description=description.strip())
    db.session.add(base)
    db.session.commit()
    return jsonify({'code': 200, 'data': _base_data(base, 0)})


@bp.route('/knowledge-bases/<int:base_id>', methods=['GET'])
@permission_required(ALLOWED_ROLES)
def get_base(base_id):
    """返回本人知识库详情及有效文献数量。"""
    base = _owned_base(base_id)
    if not base:
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    query = db.session.query(func.count(Corpus.id)).join(
        KnowledgeBaseDocument, KnowledgeBaseDocument.corpus_id == Corpus.id
    ).filter(
        KnowledgeBaseDocument.knowledge_base_id == base_id,
        KnowledgeBaseDocument.is_deleted == 0,
        Corpus.is_deleted == 0,
        Corpus.status == 2,
    )
    return jsonify({'code': 200, 'data': _base_data(base, query.scalar() or 0)})


@bp.route('/knowledge-bases/<int:base_id>', methods=['PUT'])
@permission_required(ALLOWED_ROLES)
def update_base(base_id):
    """修改本人知识库的名称或说明。"""
    base = _owned_base(base_id)
    if not base:
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({'code': 400, 'error': '请求体必须是 JSON 对象'}), 400
    if 'name' in body:
        name = body['name']
        if not isinstance(name, str) or not 1 <= len(name.strip()) <= 127:
            return jsonify({'code': 400, 'error': '知识库名称须为 1–127 个字符'}), 400
        base.name = name.strip()
    if 'description' in body:
        description = body['description']
        if not isinstance(description, str) or len(description.strip()) > 255:
            return jsonify({'code': 400, 'error': '知识库说明最多 255 个字符'}), 400
        base.description = description.strip()
    db.session.commit()
    return jsonify({'code': 200, 'data': _base_data(base)})


@bp.route('/knowledge-bases/<int:base_id>', methods=['DELETE'])
@permission_required(ALLOWED_ROLES)
def delete_base(base_id):
    """软删除知识库，保留关联记录以便后续恢复。"""
    base = _owned_base(base_id)
    if not base:
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    base.is_deleted = True
    db.session.commit()
    return jsonify({'code': 200, 'message': '知识库已删除'})


@bp.route('/knowledge-bases/<int:base_id>/documents', methods=['GET'])
@permission_required(ALLOWED_ROLES)
def list_documents(base_id):
    """分页列出知识库中仍可访问的已审核文献。"""
    if not _owned_base(base_id):
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    page, size = _page_args()
    query = _eligible_corpus_query().join(
        KnowledgeBaseDocument, KnowledgeBaseDocument.corpus_id == Corpus.id
    ).filter(KnowledgeBaseDocument.knowledge_base_id == base_id, KnowledgeBaseDocument.is_deleted == 0)
    total = query.count()
    rows = query.order_by(KnowledgeBaseDocument.added_at.desc(), Corpus.id.desc()).offset(
        (page - 1) * size
    ).limit(size).all()
    jobs = {job.corpus_id: job for job in CorpusIndexJob.query.filter(
        CorpusIndexJob.corpus_id.in_([row.id for row in rows])
    ).all()} if rows else {}
    return jsonify({
        'code': 200,
        'data': [{**_corpus_data(row), 'index': _index_data(row, jobs.get(row.id))} for row in rows],
        'pagination': {'page_num': page, 'page_size': size, 'total': total},
    })


@bp.route('/knowledge-bases/<int:base_id>/documents/preview', methods=['POST'])
@permission_required(ALLOWED_ROLES)
def preview_documents(base_id):
    """预览所选文献中可新增、已在库中和不可收录的文献。"""
    if not _owned_base(base_id):
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    ids, error = _requested_corpus_ids()
    if error:
        return jsonify({'code': 400, 'error': error}), 400
    eligible = {row.id for row in _eligible_corpus_query().filter(Corpus.id.in_(ids)).all()}
    existing = {row.corpus_id for row in KnowledgeBaseDocument.query.filter(
        KnowledgeBaseDocument.knowledge_base_id == base_id,
        KnowledgeBaseDocument.corpus_id.in_(ids),
        KnowledgeBaseDocument.is_deleted == 0,
    ).all()}
    return jsonify({'code': 200, 'data': {
        'addable_ids': [corpus_id for corpus_id in ids if corpus_id in eligible and corpus_id not in existing],
        'already_added_ids': [corpus_id for corpus_id in ids if corpus_id in eligible and corpus_id in existing],
        'unavailable_ids': [corpus_id for corpus_id in ids if corpus_id not in eligible],
    }})


@bp.route('/knowledge-bases/<int:base_id>/candidates', methods=['GET'])
@permission_required(ALLOWED_ROLES)
def list_candidates(base_id):
    """不使用搜索条件，分页列出可加入的文献。"""
    if not _owned_base(base_id):
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    page, size = _page_args()
    active_ids = db.session.query(KnowledgeBaseDocument.corpus_id).filter(
        KnowledgeBaseDocument.knowledge_base_id == base_id, KnowledgeBaseDocument.is_deleted == 0
    )
    query = _eligible_corpus_query().filter(~Corpus.id.in_(active_ids))
    total = query.count()
    rows = query.order_by(Corpus.submit_time.desc(), Corpus.id.desc()).offset(
        (page - 1) * size
    ).limit(size).all()
    return jsonify({
        'code': 200,
        'data': [_corpus_data(row) for row in rows],
        'pagination': {'page_num': page, 'page_size': size, 'total': total},
    })


@bp.route('/knowledge-bases/<int:base_id>/documents', methods=['POST'])
@permission_required(ALLOWED_ROLES)
def add_documents(base_id):
    """批量加入文献；软删除过的关系复用原记录。"""
    if not _owned_base(base_id):
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    ids, error = _requested_corpus_ids()
    if error:
        return jsonify({'code': 400, 'error': error}), 400
    allowed = {row.id for row in _eligible_corpus_query().filter(Corpus.id.in_(ids)).all()}
    if len(allowed) != len(ids):
        return jsonify({'code': 400, 'error': '存在无权访问或未审核的文献'}), 400
    existing = {
        row.corpus_id: row for row in KnowledgeBaseDocument.query.filter(
            KnowledgeBaseDocument.knowledge_base_id == base_id, KnowledgeBaseDocument.corpus_id.in_(ids)
        ).all()
    }
    added = 0
    for corpus_id in ids:
        link = existing.get(corpus_id)
        if link and not link.is_deleted:
            continue
        if link:
            link.is_deleted = False
            link.added_at = func.now()
            link.added_by = g.current_user.id
        else:
            db.session.add(KnowledgeBaseDocument(
                knowledge_base_id=base_id, corpus_id=corpus_id, added_by=g.current_user.id
            ))
        added += 1
    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        return jsonify({'code': 409, 'error': '文献集合已变化，请刷新后重试'}), 409
    return jsonify({'code': 200, 'data': {'added_count': added}})


@bp.route('/knowledge-bases/<int:base_id>/documents/<int:corpus_id>', methods=['DELETE'])
@permission_required(ALLOWED_ROLES)
def remove_document(base_id, corpus_id):
    """软删除本人知识库中的一条文献关系。"""
    if not _owned_base(base_id):
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    link = KnowledgeBaseDocument.query.filter_by(
        knowledge_base_id=base_id, corpus_id=corpus_id, is_deleted=False
    ).first()
    if link:
        link.is_deleted = True
        db.session.commit()
    return jsonify({'code': 200, 'message': '文献已移除'})


@bp.route('/knowledge-bases/<int:base_id>/documents/<int:corpus_id>/reindex', methods=['POST'])
@permission_required(ALLOWED_ROLES)
def rebuild_document_index(base_id, corpus_id):
    """在本人知识库内提交异步重建；当前附件的活动任务幂等复用。"""
    base = KnowledgeBase.query.filter_by(
        id=base_id, owner_id=g.current_user.id, is_deleted=False
    ).with_for_update().first()
    if not base:
        return jsonify({'code': 404, 'error': '知识库不存在'}), 404
    corpus = _eligible_corpus_query().filter(Corpus.id == corpus_id).with_for_update().first()
    link = KnowledgeBaseDocument.query.filter_by(
        knowledge_base_id=base_id, corpus_id=corpus_id, is_deleted=False
    ).with_for_update().first()
    if not corpus or not link:
        return jsonify({'code': 404, 'error': '文献不在当前知识库中或不可访问'}), 404
    if not _has_index_source(corpus):
        return jsonify({'code': 400, 'error': '文献没有可索引的附件'}), 400
    job = CorpusIndexJob.query.filter_by(corpus_id=corpus_id).with_for_update().first()
    from app.indexing.jobs import pending_status
    reused = bool(job and not job.is_deleted and job.status in (pending_status(), 'RUNNING')
                  and job.attachment_digest == attachment_digest(corpus.attachment))
    if not reused:
        enqueue_corpus_index(db.session, corpus)
        db.session.flush()
        job = db.session.get(CorpusIndexJob, corpus_id)
    data = {'corpus_id': corpus_id, 'index': _index_data(corpus, job), 'reused': reused}
    db.session.commit()
    return jsonify({'code': 202, 'data': data, 'message': '任务已在处理中' if reused else '已提交索引重建'}), 202
