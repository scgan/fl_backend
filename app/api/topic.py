"""
专题管理 API：Topic CRUD + Topic-Corpus 关联

REST API:
  POST   /topic                 - 创建专题
  GET    /topic                 - 列出我的专题（支持分页）
  GET    /topic/<id>            - 获取专题详情（含文献列表）
  PUT    /topic/<id>            - 更新专题（name/description）
  DELETE /topic/<id>            - 删除专题（软删除）
  POST   /topic/<id>/corpus     - 向专题添加文献（支持批量）
  DELETE /topic/<id>/corpus/<corpus_id>  - 从专题移除文献（单个）
  POST   /topic/<id>/corpus/batch-remove  - 批量移除文献
"""
from flask import Blueprint, request, jsonify, g
from app.exts import db
from app.models import Topic, TopicCorpus, Corpus
from app.utils.decorators import permission_required

bp = Blueprint('topic', __name__)


# ── 辅助 ────────────────────────────────────────────────────────

def _get_owned_topic(topic_id):
    """返回 topic_id 对应且属于当前用户的 Topic，未找到或无权限返回 None。"""
    return Topic.query.filter_by(
        id=topic_id, user_id=g.current_user.id, is_deleted=False).first()


# ── Topic CRUD ─────────────────────────────────────────────────

@bp.route('/topic', methods=['POST'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def create_topic():
    """创建专题"""
    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    if not name:
        return jsonify({"code": 400, "error": "专题名称不能为空"}), 400
    if len(name) > 127:
        return jsonify({"code": 400, "error": "专题名称过长（最多127字符）"}), 400

    description = (data.get('description') or '').strip()
    if len(description) > 255:
        description = description[:255]

    topic = Topic(user_id=g.current_user.id, name=name, description=description)
    db.session.add(topic)
    db.session.commit()
    return jsonify({"code": 200, "data": topic.to_dict(include_stats=True)}), 200


@bp.route('/topic', methods=['GET'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def list_topics():
    """列出我的专题（支持分页）"""
    page_num = int(request.args.get('page_num', 1))
    page_size = int(request.args.get('page_size', 20))
    if page_num < 1:
        page_num = 1
    if page_size < 1 or page_size > 100:
        page_size = 20

    query = Topic.query.filter_by(
        user_id=g.current_user.id, is_deleted=False).order_by(Topic.create_time.desc())
    total = query.count()
    page = query.offset((page_num - 1) * page_size).limit(page_size).all()

    return jsonify({
        "code": 200,
        "data": [t.to_dict(include_stats=True) for t in page],
        "pagination": {"page_num": page_num, "page_size": page_size, "total": total}
    }), 200


@bp.route('/topic/<int:topic_id>', methods=['GET'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def get_topic(topic_id):
    """校验专题所有权，按关联顺序分页并仅展示当前有效已审核文献。

    Args:
        topic_id: 当前用户的专题 ID。

    Returns:
        专题、文献列表及分页信息；total 为有效关联数，可能大于实际可见文献数。
    """
    topic = _get_owned_topic(topic_id)
    if not topic:
        return jsonify({"code": 404, "error": "专题不存在或无权限"}), 404

    page_num = int(request.args.get('page_num', 1))
    page_size = int(request.args.get('page_size', 20))
    if page_num < 1:
        page_num = 1
    if page_size < 1 or page_size > 100:
        page_size = 20

    tc_query = TopicCorpus.query.filter_by(
        topic_id=topic_id, is_deleted=False).order_by(TopicCorpus.create_time.desc())
    total = tc_query.count()
    tc_page = tc_query.offset((page_num - 1) * page_size).limit(page_size).all()

    corpus_ids = [tc.corpus_id for tc in tc_page]
    corpus_map = {}
    if corpus_ids:
        rows = Corpus.query.filter(
            Corpus.id.in_(corpus_ids),
            Corpus.is_deleted == 0,
            Corpus.status == 2
        ).all()
        for row in rows:
            corpus_map[row.id] = {
                'id': row.id,
                'title_cn': row.title_cn,
                'title_en': row.title_en,
                'author': row.author,
                'publish_time': row.publish_time,
                'keyword_cn': row.keyword_cn,
                'keyword_en': row.keyword_en,
                'abstract_cn': row.abstract_cn,
                'abstract_en': row.abstract_en,
            }

    corpus_list = []
    for tc in tc_page:
        c = corpus_map.get(tc.corpus_id)
        if c:
            c['_added_at'] = tc.create_time.strftime("%Y-%m-%d %H:%M:%S") if tc.create_time else None
            corpus_list.append(c)

    result = topic.to_dict()
    result['corpus_list'] = corpus_list
    result['pagination'] = {"page_num": page_num, "page_size": page_size, "total": total}
    return jsonify({"code": 200, "data": result}), 200


@bp.route('/topic/<int:topic_id>', methods=['PUT'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def update_topic(topic_id):
    """更新专题（name/description）"""
    topic = _get_owned_topic(topic_id)
    if not topic:
        return jsonify({"code": 404, "error": "专题不存在或无权限"}), 404

    data = request.get_json() or {}
    name = (data.get('name') or '').strip()
    if name:
        if len(name) > 127:
            return jsonify({"code": 400, "error": "专题名称过长（最多127字符）"}), 400
        topic.name = name

    description = (data.get('description') or '').strip()
    if 'description' in data:
        topic.description = description[:255] if description else None

    db.session.commit()
    return jsonify({"code": 200, "data": topic.to_dict(include_stats=True)}), 200


@bp.route('/topic/<int:topic_id>', methods=['DELETE'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def delete_topic(topic_id):
    """删除专题（软删除）"""
    topic = _get_owned_topic(topic_id)
    if not topic:
        return jsonify({"code": 404, "error": "专题不存在或无权限"}), 404

    topic.is_deleted = True
    # 同时软删除所有关联记录
    TopicCorpus.query.filter_by(topic_id=topic_id, is_deleted=False).update(
        {'is_deleted': True})
    db.session.commit()
    return jsonify({"code": 200, "message": "专题已删除"}), 200


# ── Topic-Corpus 关联 ──────────────────────────────────────────

@bp.route('/topic/<int:topic_id>/corpus', methods=['POST'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def add_corpus_to_topic(topic_id):
    """向本人专题批量添加当前有效已审核文献，跳过无效 ID 或有效重复关联。

    Args:
        topic_id: 当前用户的专题 ID。

    Returns:
        新增数、跳过数和跳过 ID；新增记录在同一事务提交。
    """
    topic = _get_owned_topic(topic_id)
    if not topic:
        return jsonify({"code": 404, "error": "专题不存在或无权限"}), 404

    data = request.get_json() or {}
    raw_ids = data.get('corpus_ids', [])
    if isinstance(raw_ids, int):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, list) or not raw_ids:
        return jsonify({"code": 400, "error": "corpus_ids 必须是非空数组"}), 400

    added = []
    skipped = []
    for cid in raw_ids:
        cid_int = int(cid) if str(cid).isdigit() else None
        if cid_int is None:
            skipped.append(str(cid))
            continue
        # 检查文献是否存在且已审核
        exists = Corpus.query.filter_by(
            id=cid_int, is_deleted=0, status=2).first()
        if not exists:
            skipped.append(str(cid_int))
            continue
        # 检查是否已加入
        existing = TopicCorpus.query.filter_by(
            topic_id=topic_id, corpus_id=cid_int, is_deleted=False).first()
        if existing:
            skipped.append(str(cid_int))
            continue
        tc = TopicCorpus(topic_id=topic_id, corpus_id=cid_int)
        db.session.add(tc)
        added.append(cid_int)

    db.session.commit()
    return jsonify({
        "code": 200,
        "message": f"成功添加 {len(added)} 篇文献",
        "added_count": len(added),
        "skipped_count": len(skipped),
        "skipped_ids": skipped
    }), 200


@bp.route('/topic/<int:topic_id>/corpus/<int:corpus_id>', methods=['DELETE'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def remove_corpus_from_topic(topic_id, corpus_id):
    """从专题移除单篇文献"""
    topic = _get_owned_topic(topic_id)
    if not topic:
        return jsonify({"code": 404, "error": "专题不存在或无权限"}), 404

    tc = TopicCorpus.query.filter_by(
        topic_id=topic_id, corpus_id=corpus_id, is_deleted=False).first()
    if tc:
        tc.is_deleted = True
        db.session.commit()
    return jsonify({"code": 200, "message": "已从专题移除该文献"}), 200


@bp.route('/topic/<int:topic_id>/corpus/batch-remove', methods=['POST'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def batch_remove_corpus_from_topic(topic_id):
    """批量从专题移除文献"""
    topic = _get_owned_topic(topic_id)
    if not topic:
        return jsonify({"code": 404, "error": "专题不存在或无权限"}), 404

    data = request.get_json() or {}
    raw_ids = data.get('corpus_ids', [])
    if not isinstance(raw_ids, list) or not raw_ids:
        return jsonify({"code": 400, "error": "corpus_ids 必须是非空数组"}), 400

    removed = 0
    for cid in raw_ids:
        cid_int = int(cid) if str(cid).isdigit() else None
        if cid_int is None:
            continue
        tc = TopicCorpus.query.filter_by(
            topic_id=topic_id, corpus_id=cid_int, is_deleted=False).first()
        if tc:
            tc.is_deleted = True
            removed += 1

    db.session.commit()
    return jsonify({
        "code": 200,
        "message": f"已移除 {removed} 篇文献",
        "removed_count": removed
    }), 200


@bp.route('/topic/<int:topic_id>/corpus-ids', methods=['GET'])
@permission_required(['LEADER', 'ADMIN', 'COLLECTOR'])
def get_topic_corpus_ids(topic_id):
    """获取专题下所有文献的 ID 列表（供前端拼装 corpus_ids 调用 /corpus_qa）"""
    topic = _get_owned_topic(topic_id)
    if not topic:
        return jsonify({"code": 404, "error": "专题不存在或无权限"}), 404

    rows = TopicCorpus.query.filter_by(
        topic_id=topic_id, is_deleted=False).all()
    return jsonify({
        "code": 200,
        "data": {
            "topic_id": topic_id,
            "topic_name": topic.name,
            "corpus_ids": [tc.corpus_id for tc in rows]
        }
    }), 200
