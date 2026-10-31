"""管理文献阅读的个人 PDF、索引状态和授权原文；支持与公共文献混合选择。"""

import json
import uuid
from pathlib import PurePosixPath

from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy import and_, or_

from app.api.knowledge_base import _index_data
from app.exts import db
from app.indexing.jobs import cancel_corpus_index, pending_status
from app.indexing.service import delete_indexes
from app.services.personal_metadata import (
    prepare_personal_index, extract_and_queue_personal_metadata,
    fail_personal_metadata, metadata_processing_active,
)
from app.models import Corpus, CorpusIndexJob
from app.utils.decorators import permission_required
from app.utils.utils import delete_file, get_file, upload_file


bp = Blueprint('literature_files', __name__, url_prefix='/literature')
# 第一版沿用 PDF 阅读器；每次只上传一个文件，多文件由前端逐个提交。
MAX_PDF_BYTES = 50 * 1024 * 1024
ROLES = ['COLLECTOR']


def _personal_query():
    """限定当前用户尚未删除的个人文件，不能用客户端所有者覆盖账号。"""
    return Corpus.query.filter(
        Corpus.status == 4, Corpus.owner_id == g.current_user.id, Corpus.is_deleted == 0,
    )


def _readable_query():
    """提供公共审核文献和本人个人文件的统一授权入口。"""
    return Corpus.query.filter(Corpus.is_deleted == 0, or_(
        and_(Corpus.status == 2, Corpus.owner_id.is_(None)),
        and_(Corpus.status == 4, Corpus.owner_id == g.current_user.id),
    ))


def _document_data(corpus, job):
    """返回题录、来源与双路索引状态，前端据此判断能否开始分析。"""
    data = corpus.to_dict()
    data['origin'] = 'PERSONAL' if corpus.status == 4 else 'LIBRARY'
    data['index'] = _index_data(corpus, job)
    return data


def _list_documents(query):
    """按标题查询并分页，批量读取任务避免每行单独访问数据库。"""
    page = max(request.args.get('page', 1, type=int), 1)
    size = min(max(request.args.get('page_size', 20, type=int), 1), 100)
    search = request.args.get('q', '').strip()[:200]
    if search:
        query = query.filter(or_(Corpus.title_cn.contains(search), Corpus.title_en.contains(search)))
    total = query.count()
    rows = query.order_by(Corpus.id.desc()).offset((page - 1) * size).limit(size).all()
    jobs = {job.corpus_id: job for job in CorpusIndexJob.query.filter(
        CorpusIndexJob.corpus_id.in_([row.id for row in rows]),
    ).all()} if rows else {}
    return jsonify(code=200, data={
        'items': [_document_data(row, jobs.get(row.id)) for row in rows],
        'total': total, 'page': page, 'page_size': size,
    })


@bp.route('/files', methods=['GET'])
@permission_required(ROLES)
def list_personal_files():
    """列出本人个人空间的文件和处理进度。"""
    return _list_documents(_personal_query())


@bp.route('/library', methods=['GET'])
@permission_required(ROLES)
def list_library_files():
    """提供混合选择弹窗的公共文献候选，绝不返回个人文件。"""
    return _list_documents(Corpus.query.filter(
        Corpus.status == 2, Corpus.owner_id.is_(None), Corpus.is_deleted == 0,
    ))


@bp.route('/files', methods=['POST'])
@permission_required(ROLES)
def upload_personal_file():
    """先持久化文件和准备任务，在当前 Backend 请求提取题录后才开放索引。

    Returns:
        提取成功返回 202；题录失败返回 502 并保留文件及失败任务供重试。
        无效格式/超限为 400/413，存储或登记失败为 502/500。
    """
    # 1. 限制文件大小并核对 PDF，题录与普通采集一样在 Backend 请求中同步提取。
    if request.content_length and request.content_length > MAX_PDF_BYTES + 1024 * 1024:
        return jsonify(code=413, error='PDF 文件不能超过 50 MB'), 413
    files = request.files.getlist('file')
    if len(files) != 1 or not files[0].filename:
        return jsonify(code=400, error='每次请选择一个 PDF 文件，字段名为 file'), 400
    file = files[0]
    original = PurePosixPath(file.filename.replace('\\', '/')).name
    original = ''.join(char for char in original if ord(char) >= 32)[:255]
    if not original.lower().endswith('.pdf'):
        return jsonify(code=400, error='目前仅支持 PDF 文件'), 400
    content = file.read(MAX_PDF_BYTES + 1)
    file.seek(0)
    if len(content) > MAX_PDF_BYTES:
        return jsonify(code=413, error='PDF 文件不能超过 50 MB'), 413
    if not content.startswith(b'%PDF-'):
        return jsonify(code=400, error='文件内容不是有效的 PDF'), 400
    uploaded_paths = []
    try:
        # 2. 所有者与状态由服务器赋值；flush 获取主键，文件服务名称使用随机值避免覆盖。
        corpus = Corpus(
            title_cn=original[:-4] or '未命名文件', contributor=g.current_user.id,
            owner_id=g.current_user.id, status=4, is_deleted=False, is_vectorized=False,
        )
        db.session.add(corpus)
        db.session.flush()
        file.filename = f'{corpus.id}-{uuid.uuid4().hex}.pdf'
        result, code = upload_file([(file, 0)], 'corpus', f'personal-{g.current_user.id}')
        if code != 200:
            db.session.rollback()
            current_app.logger.warning('个人 PDF 文件服务上传失败: %s', code)
            return jsonify(code=502, error='文件存储失败，请稍后重试'), 502
        attachments = result if isinstance(result, list) else []
        uploaded_paths = [item.get('file_path') for item in attachments if isinstance(item, dict)
                          and isinstance(item.get('file_path'), str) and item.get('file_path')]
        if len(uploaded_paths) != 1:
            raise ValueError('文件服务未返回单个附件路径')
        attachments = [{'file_path': uploaded_paths[0], 'original_filename': original}]
        corpus.attachment = json.dumps(attachments, ensure_ascii=False)
        # 3. 先持久化 EXTRACTING 准备任务，Worker 仅领取 pending_status()，不会抢跑。
        prepare_personal_index(corpus)
        job = db.session.get(CorpusIndexJob, corpus.id)
        snapshot = (corpus.id, job.attachment_digest, job.generation)
        db.session.commit()
    except Exception:
        db.session.rollback()
        current_app.logger.exception('个人 PDF 上传登记失败')
        # 数据库提交失败时尽力清理已上传的随机附件，不影响其他文献文件。
        if uploaded_paths:
            try:
                delete_file(uploaded_paths)
            except Exception:
                current_app.logger.exception('清理未登记的个人附件失败')
        return jsonify(code=500, error='文件登记失败，请重试'), 500
    # 4. 文件已持久化，模型失败不能进入上面的存储补偿，否则会留下失效附件引用。
    return _extract_and_reply(*snapshot)


def _extract_and_reply(corpus_id, digest, generation):
    """在当前 Backend 进程提取并排队；模型失败保留原件，返回可重试状态。"""
    try:
        completed = extract_and_queue_personal_metadata(corpus_id, digest, generation)
    except Exception:
        current_app.logger.exception('个人文件题录提取失败: %s', corpus_id)
        fail_personal_metadata(corpus_id, digest, generation, g.current_user.id)
        corpus = _personal_query().filter(Corpus.id == corpus_id).first()
        if corpus is None:
            return jsonify(code=404, error='文件不存在或不可访问'), 404
        return jsonify(code=502, error='题录提取失败，文件已保存，可点击重试',
                       data=_document_data(corpus, db.session.get(CorpusIndexJob, corpus_id))), 502
    corpus = _personal_query().filter(Corpus.id == corpus_id).first()
    if corpus is None:
        return jsonify(code=404, error='文件不存在或不可访问'), 404
    if not completed:
        return jsonify(code=409, error='文件或任务已更新，请刷新后重试'), 409
    return jsonify(code=202, data=_document_data(corpus, db.session.get(CorpusIndexJob, corpus_id))), 202


@bp.route('/documents/<int:corpus_id>', methods=['GET'])
@permission_required(ROLES)
def read_document(corpus_id):
    """统一读取已审核文献或本人个人文件，越权请求与不存在记录同样返回 404。"""
    corpus = _readable_query().filter(Corpus.id == corpus_id).first()
    if corpus is None:
        return jsonify(code=404, error='文件不存在或不可访问'), 404
    return jsonify(code=200, data=_document_data(corpus, db.session.get(CorpusIndexJob, corpus_id)))


@bp.route('/documents/<int:corpus_id>/file', methods=['GET'])
@permission_required(ROLES)
def read_document_file(corpus_id):
    """核对当前文献所有权和附件清单后代理原文，不接受任意路径下载。"""
    corpus = _readable_query().filter(Corpus.id == corpus_id).first()
    if corpus is None:
        return jsonify(code=404, error='文件不存在或不可访问'), 404
    path = request.args.get('attachment_path')
    attachments = json.loads(corpus.attachment or '[]')
    if path not in {item.get('file_path') for item in attachments if isinstance(item, dict)}:
        return jsonify(code=404, error='附件不存在'), 404
    response, status = get_file(path)
    if status != 200:
        return jsonify(code=502, error='原文件暂时不可用'), 502
    response.headers['Cache-Control'] = 'private, no-store'
    return response


@bp.route('/files/<int:corpus_id>/retry', methods=['POST'])
@permission_required(ROLES)
def retry_personal_file(corpus_id):
    """复用活动任务；题录缺失的文件由本次 Backend 请求提取后再交给 Worker。"""
    corpus = _personal_query().filter(Corpus.id == corpus_id).with_for_update().first()
    if corpus is None:
        return jsonify(code=404, error='文件不存在或不可访问'), 404
    job = db.session.get(CorpusIndexJob, corpus_id)
    active = bool(job and not job.is_deleted and (
        job.status in (pending_status(), 'RUNNING') or metadata_processing_active(job)))
    needs_extraction = False
    if not active:
        needs_extraction = prepare_personal_index(corpus)
    db.session.commit()
    if needs_extraction:
        job = db.session.get(CorpusIndexJob, corpus_id)
        return _extract_and_reply(corpus_id, job.attachment_digest, job.generation)
    return jsonify(code=202, data=_document_data(corpus, db.session.get(CorpusIndexJob, corpus_id))), 202


@bp.route('/files/<int:corpus_id>', methods=['DELETE'])
@permission_required(ROLES)
def delete_personal_file(corpus_id):
    """先撤销本人文件和任务版本，再尽力清理索引及原件，阻止在途 Worker 发布。

    Returns:
        删除结果及 cleanup_pending；清理失败仍保持文件不可访问，允许再次请求清理。
    """
    # 1. 不等待长耗时 OCR 持有的文献锁；先用行锁提交删除，发布前版本复核会拒绝旧结果。
    corpus = Corpus.query.filter(Corpus.id == corpus_id, Corpus.status == 4,
                                Corpus.owner_id == g.current_user.id).with_for_update().first()
    if corpus is None:
        return jsonify(code=404, error='文件不存在或不可访问'), 404
    paths = [item['file_path'] for item in json.loads(corpus.attachment or '[]')
             if isinstance(item, dict) and item.get('file_path')]
    corpus.is_deleted = True
    corpus.is_vectorized = False
    cancel_corpus_index(db.session, corpus_id)
    db.session.commit()
    # 2. 同时清理两路索引；锁忙或远程失败不恢复访问资格，由重复删除请求继续清理。
    cleanup_pending = False
    try:
        delete_indexes(corpus_id)
        result = delete_file(paths)
        if result and result[1] != 200:
            cleanup_pending = True
    except Exception:
        db.session.rollback()
        cleanup_pending = True
        current_app.logger.exception('个人文件已撤销，远程清理未完成: %s', corpus_id)
    return jsonify(code=200, data={'id': corpus_id, 'cleanup_pending': cleanup_pending})
