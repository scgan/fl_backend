"""提供负责人文献审核、恢复文献审核分流、审核统计及下属用户查询接口。"""

import json
from flask import Blueprint, request, jsonify, g, current_app
from sqlalchemy import and_, or_

from app.exts import db
from app.models import Corpus, CorpusCategory, User, Database, CorpusDatabase
from app.services import corpus_ai_classify
from app.utils.decorators import permission_required
from app.utils.enums import CorpusStatus
from app.indexing.jobs import enqueue_corpus_index

bp = Blueprint('leader', __name__)

# 脚本恢复语料的上传者 id（待审核语料由二级管理员在此接口中重新审核）
RECOVERY_CONTRIBUTOR_ID = 26
# 题目/关键词含「道」「儒」「中国」的待审核语料预分配给该用户审核，仅此用户可见
ASSIGNED_REVIEWER_ID_DAO_RU_ZHONGGUO = 32

# 二级管理员：查看已审核的恢复语料（contributor=26, status=审核通过/驳回, reviewer_id非空）
@bp.route('/examine/recovery/reviewed', methods=['GET'])
@permission_required(['LEADER'])
def examine_recovery_reviewed():
    """按元数据筛选已分配审核人的恢复文献，返回通过或驳回的分页记录。

    未指定 status 时包含两种终态；该列表不按当前审核人进一步过滤。

    Returns:
        文献分页 JSON，非法状态参数返回 400。
    """
    data = request.args
    corpus_id = data.get('corpus_id')
    title_cn = data.get('title_cn')
    title_en = data.get('title_en')
    category_id = data.get('category_id')
    language = data.get('language')
    author = data.get('author')
    status = data.get('status', type=int)
    page_size = data.get('page_size', default=10, type=int)
    page_num = data.get('page_num', default=1, type=int)

    allowed_status = (
        CorpusStatus.APPROVE_EXAMINE.value,
        CorpusStatus.REJECT_EXAMINE.value,
    )
    if status is not None and status not in allowed_status:
        return jsonify({'code': 400, 'error': 'status 仅支持 2（审核通过）或 3（审核驳回）'}), 400

    filters = [
        Corpus.owner_id.is_(None), Corpus.status != 4,
        Corpus.is_deleted == 0,
        Corpus.contributor == RECOVERY_CONTRIBUTOR_ID,
        Corpus.reviewer_id.isnot(None),
    ]
    if status is not None:
        filters.append(Corpus.status == status)
    else:
        filters.append(Corpus.status.in_(allowed_status))
    if corpus_id:
        filters.append(Corpus.id == corpus_id)
    if title_cn:
        filters.append(Corpus.title_cn.like(f'%{title_cn}%'))
    if title_en:
        filters.append(Corpus.title_en.like(f'%{title_en}%'))
    if category_id:
        filters.append(Corpus.category_id == category_id)
    if language:
        filters.append(Corpus.language == language)
    if author:
        filters.append(or_(
            Corpus.author.like(f'%{author}%'),
            Corpus.translator.like(f'%{author}%'),
            Corpus.editor.like(f'%{author}%'),
        ))

    query = (
        db.session.query(
            Corpus.id,
            Corpus.title_cn,
            Corpus.title_en,
            Corpus.status,
            Corpus.language,
            CorpusCategory.category_name,
            Corpus.author,
            Corpus.keyword_cn,
            Corpus.keyword_en,
            Corpus.publish_time,
            Corpus.submit_time,
            User.name,
        )
        .outerjoin(CorpusCategory, Corpus.category_id == CorpusCategory.id)
        .outerjoin(User, User.id == Corpus.contributor)
        .filter(and_(*filters))
        .order_by(Corpus.submit_time.desc())
    )
    corpus_page = query.paginate(page=page_num, per_page=page_size, error_out=False)
    res = []
    for row in corpus_page.items:
        (cid, title_cn_val, title_en_val, status, language_val, category_name,
         author_val, keyword_cn, keyword_en, publish_time, submit_time, contributor_name) = row
        res.append({
            'corpus_id': cid,
            'title_cn': title_cn_val,
            'title_en': title_en_val,
            'status': status,
            'language': language_val,
            'category_name': category_name,
            'author': author_val,
            'keyword_cn': json.loads(keyword_cn) if keyword_cn else None,
            'keyword_en': json.loads(keyword_en) if keyword_en else None,
            'publish_time': publish_time,
            'submit_time': submit_time.strftime("%Y-%m-%d %H:%M:%S") if submit_time else submit_time,
            'contributor_name': contributor_name,
        })
    return jsonify({
        'code': 200,
        'data': {
            'total': corpus_page.total,
            'pages': corpus_page.pages,
            'data': res,
        },
    })

# 二级管理员：重新审核语料（筛选 status=待审核、上传者为 26 的文献）
@bp.route('/examine/recovery', methods=['GET'])
@permission_required(['LEADER'])
def examine_recovery():
    """按当前审核人的分流规则返回待审核恢复文献。

    指定专人查看分配给该专人的记录，其他负责人查看未分配记录。

    Returns:
        带贡献者和分类信息的文献分页 JSON。
    """
    data = request.args
    corpus_id = data.get('corpus_id')
    title = data.get('title')
    category_id = data.get('category_id')
    language = data.get('language')
    author = data.get('author')
    page_size = data.get('page_size', default=10, type=int)
    page_num = data.get('page_num', default=1, type=int)

    filters = [
        Corpus.owner_id.is_(None), Corpus.status != 4,
        Corpus.is_deleted == 0,
        Corpus.status == CorpusStatus.WAITING_EXAMINE.value,
        Corpus.contributor == RECOVERY_CONTRIBUTOR_ID,
    ]
    # 预分配给 user_id=32 的语料只对 32 展示；其他二级管理员只看到未分配（reviewer_id 为空）的
    if g.current_user.id == ASSIGNED_REVIEWER_ID_DAO_RU_ZHONGGUO:
        filters.append(Corpus.reviewer_id == ASSIGNED_REVIEWER_ID_DAO_RU_ZHONGGUO)
    else:
        filters.append(Corpus.reviewer_id.is_(None))
    if corpus_id:
        filters.append(Corpus.id == corpus_id)
    if title:
        filters.append(or_(
            Corpus.title_cn.like(f'%{title}%'),
            Corpus.title_en.like(f'%{title}%'),
        ))
    if category_id:
        filters.append(Corpus.category_id == category_id)
    if language:
        filters.append(Corpus.language == language)
    if author:
        filters.append(or_(
            Corpus.author.like(f'%{author}%'),
            Corpus.translator.like(f'%{author}%'),
            Corpus.editor.like(f'%{author}%'),
        ))

    query = (
        db.session.query(
            Corpus.id,
            Corpus.title_cn,
            Corpus.title_en,
            Corpus.status,
            Corpus.language,
            CorpusCategory.category_name,
            Corpus.author,
            Corpus.keyword_cn,
            Corpus.keyword_en,
            Corpus.publish_time,
            Corpus.submit_time,
            User.name,
        )
        .outerjoin(CorpusCategory, Corpus.category_id == CorpusCategory.id)
        .outerjoin(User, User.id == Corpus.contributor)
        .filter(and_(*filters))
        .order_by(Corpus.submit_time.desc())
    )
    corpus_page = query.paginate(page=page_num, per_page=page_size, error_out=False)
    res = []
    for row in corpus_page.items:
        (corpus_id, title_cn, title_en, status, language, category_name, author,
         keyword_cn, keyword_en, publish_time, submit_time, contributor_name) = row
        res.append({
            'corpus_id': corpus_id,
            'title_cn': title_cn,
            'title_en': title_en,
            'status': status,
            'language': language,
            'category_name': category_name,
            'author': author,
            'keyword_cn': json.loads(keyword_cn) if keyword_cn else None,
            'keyword_en': json.loads(keyword_en) if keyword_en else None,
            'publish_time': publish_time,
            'submit_time': submit_time.strftime("%Y-%m-%d %H:%M:%S") if submit_time else submit_time,
            'contributor_name': contributor_name,
        })
    return jsonify({
        'code': 200,
        'data': {
            'total': corpus_page.total,
            'pages': corpus_page.pages,
            'data': res,
        },
    })

# 二级管理员：审核通过 / 不通过，记录 reviewer_id
@bp.route('/examine/recovery/decision', methods=['POST'])
@permission_required(['LEADER'])
def examine_recovery_decision():
    """处理恢复文献审核，通过时同时登记当前附件的索引任务。

    通过前要求关联有效主题数据库；审核状态、审核人和任务登记在同一事务提交。

    Returns:
        审核结果 JSON，或参数、资源和事务错误响应。
    """
    # 1. 规范化审批参数并读取仍待审核的恢复文献，拒绝过期或无效目标
    data = request.get_json() or {}
    corpus_id = data.get('corpus_id')
    approve = data.get('approve')
    comment = data.get('comment')

    if corpus_id is None or approve is None:
        return jsonify({'code': 400, 'error': '缺少 corpus_id 或 approve 参数'}), 400

    # 兼容前端传 bool / int / str 的 approve，避免 bool("false") == True
    if isinstance(approve, bool):
        approve_flag = approve
    else:
        approve_flag = str(approve).strip().lower() in ('1', 'true', 'yes', 'y', 't')

    try:
        corpus_id = int(corpus_id)
    except (TypeError, ValueError):
        return jsonify({'code': 400, 'error': 'corpus_id 格式错误'}), 400

    corpus = Corpus.query.filter(
        Corpus.id == corpus_id,
        Corpus.is_deleted == 0,
        Corpus.status == CorpusStatus.WAITING_EXAMINE.value,
        Corpus.contributor == RECOVERY_CONTRIBUTOR_ID,
    ).first()
    if not corpus:
        return jsonify({'code': 404, 'error': '无该待审核的恢复语料记录'}), 404

    try:
        # 2. 通过时先验证主题归属，再登记索引任务；驳回时只写审核状态及意见
        if approve_flag:
            # 审核通过前要求必须关联所属数据库（通过 corpus_database 表）
            has_db = (
                db.session.query(CorpusDatabase.id)
                .join(Database, Database.id == CorpusDatabase.database_id)
                .filter(
                    CorpusDatabase.corpus_id == corpus_id,
                    Database.is_deleted == 0,
                )
                .first()
            )
            if not has_db:
                return jsonify({'code': 400, 'error': '该文献未关联所属数据库，无法审核通过，请先关联数据库'}), 400
            corpus.status = CorpusStatus.APPROVE_EXAMINE.value
            enqueue_corpus_index(db.session, corpus)
        else:
            corpus.status = CorpusStatus.REJECT_EXAMINE.value
            if comment:
                corpus.comment = comment
        corpus.reviewer_id = g.current_user.id
        # 3. 审核人、审核状态与索引登记同时提交，失败整体回滚以免出现通过但未入队
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'code': 400, 'error': str(e)}), 400

    msg = '审核通过成功' if approve_flag else '审核驳回成功'
    return jsonify({'code': 200, 'success': msg})


# 二级管理员：直接删除恢复语料（软删除，is_deleted=1），用于无需驳回、直接移除的记录
@bp.route('/examine/recovery/delete', methods=['POST'])
@permission_required(['LEADER'])
def examine_recovery_delete():
    """软删除待审核恢复文献；已分配给其他审核人的记录拒绝删除。"""
    data = request.get_json() or {}
    corpus_id = data.get('corpus_id')
    if corpus_id is None:
        return jsonify({'code': 400, 'error': '缺少 corpus_id 参数'}), 400
    try:
        corpus_id = int(corpus_id)
    except (TypeError, ValueError):
        return jsonify({'code': 400, 'error': 'corpus_id 格式错误'}), 400

    corpus = Corpus.query.filter(
        Corpus.id == corpus_id,
        Corpus.is_deleted == 0,
        Corpus.status == CorpusStatus.WAITING_EXAMINE.value,
        Corpus.contributor == RECOVERY_CONTRIBUTOR_ID,
    ).first()
    if not corpus:
        return jsonify({'code': 404, 'error': '无该待审核的恢复语料记录'}), 404
    if corpus.reviewer_id is not None and corpus.reviewer_id != g.current_user.id:
        return jsonify({'code': 403, 'error': '该文献已分配给其他审核人，您无法删除'}), 403

    try:
        corpus.is_deleted = 1
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'code': 400, 'error': str(e)}), 400
    return jsonify({'code': 200, 'success': '删除成功'})


# 二级管理员：统计审核工作量（按审核通过/不通过分开展示；is_recovery 区分恢复/非恢复语料）
@bp.route('/examine/recovery/stats', methods=['GET'])
@permission_required(['ADMIN', 'LEADER'])
def examine_recovery_stats():
    # is_recovery: 1=恢复语料(contributor=26)，0=非恢复语料
    """按恢复或普通文献范围，聚合各审核人已通过、已驳回的有效记录数。"""
    is_recovery = request.args.get('is_recovery', default=1, type=int)
    if is_recovery not in (0, 1):
        return jsonify({'code': 400, 'error': 'is_recovery 参数无效，请传 0 或 1'}), 400

    contributor_filter = (
        Corpus.contributor == RECOVERY_CONTRIBUTOR_ID
        if is_recovery == 1
        else Corpus.contributor != RECOVERY_CONTRIBUTOR_ID
    )

    approved_count_expr = db.func.sum(
        db.case(
            (Corpus.status == CorpusStatus.APPROVE_EXAMINE.value, 1),
            else_=0,
        )
    )
    rejected_count_expr = db.func.sum(
        db.case(
            (Corpus.status == CorpusStatus.REJECT_EXAMINE.value, 1),
            else_=0,
        )
    )
    rows = (
        db.session.query(
            User.id.label('reviewer_id'),
            User.name.label('reviewer_name'),
            approved_count_expr.label('approved_count'),
            rejected_count_expr.label('rejected_count'),
        )
        .join(User, User.id == Corpus.reviewer_id)
        .filter(
            Corpus.is_deleted == 0,
            contributor_filter,
            Corpus.status.in_([
                CorpusStatus.APPROVE_EXAMINE.value,
                CorpusStatus.REJECT_EXAMINE.value,
            ]),
        )
        .group_by(User.id, User.name)
        .order_by((approved_count_expr + rejected_count_expr).desc())
        .all()
    )

    data = [
        {
            'reviewer_id': reviewer_id,
            'reviewer_name': reviewer_name,
            'approved_count': int(approved_count or 0),
            'rejected_count': int(rejected_count or 0),
        }
        for reviewer_id, reviewer_name, approved_count, rejected_count in rows
    ]

    return jsonify({
        'code': 200,
        'data': data,
    })


# 二级管理员：启动 AI 批量归类（读取文献判断所属数据库，并按配置自动审核通过）
@bp.route('/examine/recovery/ai-classify/start', methods=['POST'])
@permission_required(['LEADER'])
def examine_recovery_ai_classify_start():
    """按当前筛选条件批量判定文献所属数据库并回写审核状态。

    任务在后台线程执行，进度由 /progress 轮询；同一时刻只允许一个任务运行。

    Returns:
        任务初始状态 JSON，或参数、并发、凭据错误响应。
    """
    data = request.get_json(silent=True) or {}
    try:
        limit = int(data.get('limit') or 0)
        min_confidence = float(data.get('min_confidence', 0))
    except (TypeError, ValueError):
        return jsonify({'code': 400, 'error': 'limit 或 min_confidence 格式错误'}), 400
    if limit < 0:
        return jsonify({'code': 400, 'error': 'limit 不能为负数'}), 400
    if not 0 <= min_confidence <= 1:
        return jsonify({'code': 400, 'error': 'min_confidence 需在 0—1 之间'}), 400

    scope = {
        'title': data.get('title') or '',
        'language': data.get('language') or '',
        'author': data.get('author') or '',
    }
    options = {
        'limit': limit,
        'auto_approve': bool(data.get('auto_approve', True)),
        'min_confidence': min_confidence,
        'model': (data.get('model') or '').strip() or None,
        'only_unassigned': bool(data.get('only_unassigned', True)),
    }
    options['model'] = options['model'] or None

    try:
        state = corpus_ai_classify.start_task(
            scope, options, g.current_user.id,
            ASSIGNED_REVIEWER_ID_DAO_RU_ZHONGGUO, RECOVERY_CONTRIBUTOR_ID,
        )
    except RuntimeError as error:
        return jsonify({'code': 409, 'error': str(error)}), 409
    except Exception as error:
        current_app.logger.exception('启动 AI 归类任务失败')
        return jsonify({'code': 500, 'error': f'启动失败：{error}'}), 500
    return jsonify({'code': 200, 'data': state})


# 二级管理员：查询 AI 批量归类进度
@bp.route('/examine/recovery/ai-classify/progress', methods=['GET'])
@permission_required(['LEADER'])
def examine_recovery_ai_classify_progress():
    """返回指定任务或最近一次任务的进度，供前端轮询。"""
    task_id = request.args.get('task_id')
    state = corpus_ai_classify.get_task_state(task_id or None, g.current_user.id)
    if not state:
        return jsonify({'code': 200, 'data': None})
    return jsonify({'code': 200, 'data': state})


# 二级管理员：停止 AI 批量归类
@bp.route('/examine/recovery/ai-classify/stop', methods=['POST'])
@permission_required(['LEADER'])
def examine_recovery_ai_classify_stop():
    """向运行中的任务发出停止信号，当前批次结束后生效。"""
    data = request.get_json(silent=True) or {}
    state = corpus_ai_classify.request_stop(
        (data.get('task_id') or '').strip() or None, g.current_user.id
    )
    if not state:
        return jsonify({'code': 404, 'error': '没有可停止的归类任务'}), 404
    return jsonify({'code': 200, 'data': state})


#TODO 查看自己待审核/已审核的语料
@bp.route('/examine',methods = ['GET'])
@permission_required(['LEADER'])
def examine():
    """返回分配给当前负责人的普通文献分页，排除恢复账号贡献的记录。"""
    page_size = request.args.get('page_size',default=10,type=int)
    page_num = request.args.get('page_num',default=1,type=int)
    status = request.args.get('status', type=int)
    reviewer_id = g.current_user.id

    filters = [
        Corpus.owner_id.is_(None), Corpus.status != 4,
        Corpus.is_deleted == 0,
        Corpus.reviewer_id == reviewer_id,
        Corpus.contributor != RECOVERY_CONTRIBUTOR_ID,
    ]
    
    if status == CorpusStatus.WAITING_EXAMINE.value:
        filters.append(Corpus.status == CorpusStatus.WAITING_EXAMINE.value)
    elif status == CorpusStatus.APPROVE_EXAMINE.value:
        filters.append(Corpus.status == CorpusStatus.APPROVE_EXAMINE.value)
    elif status == CorpusStatus.REJECT_EXAMINE.value:
        filters.append(Corpus.status == CorpusStatus.REJECT_EXAMINE.value)
    corpus_page = db.session.query(Corpus.id,Corpus.title_cn,Corpus.title_en,Corpus.status,Corpus.language,CorpusCategory.category_name,Corpus.attachment,User.name,Corpus.submit_time).outerjoin(User,User.id == Corpus.contributor).outerjoin(CorpusCategory,Corpus.category_id == CorpusCategory.id).filter(*filters).order_by(Corpus.submit_time.desc()).paginate(page=page_num, per_page=page_size, error_out=False)

    res = []
    for corpus_id,title_cn,title_en,status,language,category_name,attachment,name,submit_time in corpus_page.items:
        res.append({
            'corpus_id':corpus_id,
            'title_cn':title_cn,
            'title_en':title_en,
            'status':status,
            'language':language,
            'category_name':category_name,
            'name':name,
            'submit_time':submit_time.strftime("%Y-%m-%d %H:%M:%S") if submit_time else submit_time,
        })
    return jsonify({
        'code':200,
        'data':{
            'total':corpus_page.total,
            'pages':corpus_page.pages,
            'data':res
        }
    })

#TODO 审核语料通过
@bp.route('/examine/approve',methods = ['POST'])
@permission_required(['LEADER'])
def examine_approve():
    """将待审核文献改为通过，并在同一事务登记索引任务。

    Returns:
        审核结果 JSON；提交失败回滚，避免已通过但未入队的可见中间状态。
    """
    corpus_id = request.get_json().get('corpus_id')
    corpus = Corpus.query.filter(Corpus.id == corpus_id,Corpus.is_deleted == 0,Corpus.status == CorpusStatus.WAITING_EXAMINE.value).first()
    if not corpus: return jsonify({"code": 404, "error": "无该待审核语料记录"}),404

    try:
        corpus.status = CorpusStatus.APPROVE_EXAMINE.value
        enqueue_corpus_index(db.session, corpus)
        # corpus_category = CorpusCategory.query.filter(CorpusCategory.id == corpus.category_id,CorpusCategory.is_deleted == 0).first()
        # if corpus.number is not None:
        #     corpus.number = corpus_category.code + str(corpus_category.number)
        #     corpus_category.number = corpus_category.number + 1
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 400, "error": str(e)}), 400
    return jsonify({"code": 200, "success":"审核通过成功"})

#TODO 审核语料不通过（需要添加驳回意见）
@bp.route('/examine/reject',methods = ['POST'])
@permission_required(['LEADER'])
def examine_reject():
    """将待审核文献改为驳回并保存意见，在同一事务提交。"""
    data = request.get_json()
    corpus_id = data.get('corpus_id')
    comment = data.get('comment')
    corpus = Corpus.query.filter(Corpus.id == corpus_id,Corpus.is_deleted == 0,Corpus.status == CorpusStatus.WAITING_EXAMINE.value).first()
    if not corpus: return jsonify({"code": 404, "error": "无该待审核语料记录"}),404
    try:
        corpus.status = CorpusStatus.REJECT_EXAMINE.value
        corpus.comment = comment
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 400, "error": str(e)}), 400
    return jsonify({"code": 200, "success":"驳回成功"})

# 查看下属用户列表
@bp.route('/subordinates',methods = ['GET'])
@permission_required(['LEADER'])
def get_subordinates():
    """按当前负责人的用户 ID 分页查询其未删除下属账号。"""
    page_size = request.args.get('page_size', default=10, type=int)
    page_num = request.args.get('page_num', default=1, type=int)
    leader_id = g.current_user.id
    
    filters = [
        User.is_deleted == 0,
        User.leader_id == leader_id
    ]
    
    users_page = User.query.filter(*filters).paginate(
        page=page_num,
        per_page=page_size,
        error_out=False
    )
    
    res = []
    for user in users_page.items:
        res.append({
            'user_id': user.id,
            'username': user.username,
            'name': user.name,
            'email': user.email,
            'phone': user.phone,
            'role': user.role,
            'status': user.status
        })
    
    return jsonify({
        'code': 200,
        'data': {
            'total': users_page.total,
            'pages': users_page.pages,
            'data': res
        }
    })
