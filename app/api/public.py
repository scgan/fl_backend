"""提供公开已审核文献的筛选、附件访问和分类统计，并按可选身份标记默认审核人。"""

import json
import zipfile
from io import BytesIO
from urllib.parse import quote

from flask import Blueprint, request, jsonify, send_file, g
from sqlalchemy import or_, and_, func
from app.exts import db
from app.models import Corpus, CorpusCategory, Database, CorpusDatabase, User
from app.utils.enums import CorpusStatus
from app.utils.return_utils import success
from app.utils.utils import get_file, get_file_content
from queue import Queue
from elasticsearch import Elasticsearch
from app.utils.enums import UserRole
from datetime import datetime
from flask import url_for
bp = Blueprint('public', __name__)


#TODO 检索文献
@bp.route('/search',methods = ['GET'])
def search():
    """按元数据筛选未删除且审核通过的文献，返回公开字段分页。"""
    args = request.args
    title_cn = args.get('title_cn')
    title_en = args.get('title_en')
    category_id = args.get('category_id')
    language = args.get('language')
    author = args.get('author')
    keyword_cn = args.getlist('keyword_cn')
    keyword_en = args.getlist('keyword_en')
    page_size = args.get('page_size', default=10, type=int)
    page_num = args.get('page_num', default=1, type=int)
    filters = [Corpus.is_deleted == 0, Corpus.status == CorpusStatus.APPROVE_EXAMINE.value]
    if title_cn: filters.append(Corpus.title_cn.like(f'%{title_cn}%'))
    if title_en: filters.append(Corpus.title_en.like(f'%{title_en}%'))
    if category_id: filters.append(Corpus.category_id == category_id)
    if language: filters.append(Corpus.language == language)
    if author: filters.append(or_(Corpus.author.like(f"%{author}%"), Corpus.translator.like(f"%{author}%"),
                                  Corpus.editor.like(f"%{author}%")))
    if len(keyword_cn) > 0:
        filters.append(and_(Corpus.keyword_cn.like(f"%{keyword}%") for keyword in keyword_cn))
    if len(keyword_en) > 0:
        filters.append(and_(Corpus.keyword_en.like(f"%{keyword}%") for keyword in keyword_en))

    corpus_page = db.session.query(Corpus.id, Corpus.title_cn, Corpus.title_en, Corpus.language,CorpusCategory.category_name, Corpus.author, Corpus.keyword_cn,Corpus.keyword_en, Corpus.publish_time, Corpus.word_number).outerjoin(CorpusCategory,Corpus.category_id == CorpusCategory.id).filter(and_(*filters)).paginate(page=page_num, per_page=page_size, error_out=False)

    res = []
    for corpus_id, title_cn, title_en, language, category_name, author, keyword_cn, keyword_en, publish_time, word_number in corpus_page.items:
        res.append({
            'corpus_id': corpus_id,
            'title_cn': title_cn,
            'title_en': title_en,
            'language': language,
            'category_name': category_name,
            'author': author,
            'keyword_cn': json.loads(keyword_cn) if keyword_cn else None,
            'keyword_en': json.loads(keyword_en) if keyword_en else None,
            'publish_time': publish_time,
            'word_number': word_number
        })
    return success({
        'total': corpus_page.total,
        'pages': corpus_page.pages,
        'data': res
    })

@bp.route('/leaders', methods=['GET'])
def get_leaders():
    """列出有效负责人，可根据可选令牌标记当前账号的默认审核人。

    Returns:
        负责人列表 JSON；令牌缺失或无效时仍返回不带默认标记的公开列表。
    """
    leaders = db.session.query(User.id, User.name).filter(User.role.like(f'%{UserRole.LEADER.value}%'), User.is_deleted == 0).all()
    res = [{'id': uid, 'username': name, 'is_default': False} for uid, name in leaders]
    
    # 解析 token 获取当前用户
    current_user_id = None
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
        from flask import current_app
        import jwt
        try:
            payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
            username = payload.get('username')
            current_user = User.query.filter_by(username=username, is_deleted=0, status=1).first()
            if current_user:
                current_user_id = current_user.id
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            pass
    
    if current_user_id:
        default_reviewer_id = db.session.query(User.leader_id).filter(User.id == current_user_id, User.is_deleted == 0).first()
        if default_reviewer_id and default_reviewer_id[0]:
            default_id = default_reviewer_id[0]
            for leader in res:
                if leader['id'] == default_id:
                    leader['is_default'] = True
                    break
    
    return jsonify({'code': 200, 'data': res})

@bp.route('/file/corpus', methods=['GET'])
def get_corpus():
    # 支持单个语料ID和多个语料ID
    """返回公开文献的附件清单，或校验引用后代理单个附件下载。

    批量 corpus_id[] 分支返回文件信息而非 ZIP；单文件分支要求文献仍审核通过，
    且 attachment_path 属于当前附件清单。文件名优先保留原名和扩展名。

    Returns:
        文件列表 JSON、文件响应，或参数、文献和文件服务错误响应。
    """
    # 1. 区分批量清单和单文件下载输入，所有文献都限定为当前有效已审核
    corpus_ids = request.args.getlist('corpus_id[]')
    attachment_path = request.args.get('attachment_path')
    
    # 处理多语料ID批量下载 - 现在返回文件列表而不是打包
    if corpus_ids:
        try:
            corpus_ids = [int(cid) for cid in corpus_ids if cid]
            if not corpus_ids: 
                return jsonify({"code":400,"error":"无语料id"}),400
        except ValueError:
            return jsonify({"code":400,"error":"语料id格式错误"}),400
        
        # 查询所有语料
        corpora = Corpus.query.filter(
            Corpus.id.in_(corpus_ids), 
            Corpus.is_deleted == 0, 
            Corpus.status == CorpusStatus.APPROVE_EXAMINE.value
        ).all()
        
        if not corpora: 
            return jsonify({"code":400,"error":"不存在这些语料"}),400
        
        # 收集所有文件信息
        # 2. 批量分支展开真实附件，生成逐文件下载链接，不在后端隐式打包 ZIP
        files_info = []
        
        for corpus in corpora:
            current_attachment = json.loads(corpus.attachment) if corpus.attachment else []
            
            for item in current_attachment:
                file_path = item.get('file_path')
                original_filename = item.get('original_filename', '')
                
                if file_path:
                    # 生成安全的文件名
                    safe_corpus_title = "".join(
                        c for c in (corpus.title_cn or f'语料_{corpus.id}') 
                        if c.isalnum() or c in (' ', '-', '_')
                    ).rstrip()
                    
                    # 使用原始文件名或从路径提取
                    if original_filename:
                        display_name = original_filename
                    else:
                        display_name = file_path.split('/')[-1] if '/' in file_path else file_path
                    
                    # 构建下载URL
                    download_url = url_for(
                        '.get_corpus',
                        corpus_id=corpus.id,
                        attachment_path=file_path,
                        _external=False
                    )
                    
                    files_info.append({
                        'corpus_id': corpus.id,
                        'corpus_title': corpus.title_cn,
                        'file_path': file_path,
                        'display_name': f"{safe_corpus_title}_{display_name}",
                        'original_name': original_filename or display_name,
                        'download_url': download_url,
                        'file_size': item.get('file_size', 0)  # 如果有文件大小信息
                    })
        
        if not files_info:
            return jsonify({"code":400,"error":"所选语料无附件"}),400
        
        # 返回文件列表信息
        return jsonify({
            "code": 200,
            "message": f"找到{len(files_info)}个文件",
            "data": {
                "total_files": len(files_info),
                "files": files_info,
                "download_all": len(files_info) <= 10,  # 建议：10个以内可以打包下载
                "suggested_approach": "建议前端使用Promise.all或逐个下载文件"
            }
        })
    
    # 单个文件下载（保持原有逻辑不变）
    # 3. 单文件分支重查当前审核及附件归属，再代理返回并设置保留扩展名的下载名称
    elif attachment_path:
        corpus_id = request.args.get('corpus_id', type=int)
        if not corpus_id: 
            return jsonify({"code":400,"error":"无语料id"}),400

        corpus = Corpus.query.filter(
            Corpus.id == corpus_id,
            Corpus.is_deleted == 0,
            Corpus.status == CorpusStatus.APPROVE_EXAMINE.value
        ).first()
        
        if not corpus: 
            return jsonify({"code":400,"error":"不存在该语料"}),400

        current_attachment = json.loads(corpus.attachment) if corpus.attachment else []
        current_attachment_paths = [item['file_path'] for item in current_attachment]
        
        if attachment_path not in current_attachment_paths: 
            return jsonify({"code":400,"error":"语料无该附件"}),400

        # 调用get_file获取单个文件
        response, code = get_file(attachment_path)
        if code != 200:
            return jsonify({"code": code, "error": response}), code
        
        # 设置下载文件名
        fallback_filename = attachment_path.split('/')[-1] if '/' in attachment_path else attachment_path
        original_filename = ''
        for item in current_attachment:
            if item['file_path'] == attachment_path:
                original_filename = item.get('original_filename') or ''
                break
        # 部分历史附件记录缺少 original_filename；兜底用路径 basename，
        # 并确保保留扩展名，否则浏览器会存出无法打开的无后缀文件。
        if not original_filename or '.' not in original_filename:
            original_filename = fallback_filename
        
        safe_corpus_title = "".join(
            c for c in (corpus.title_cn or f'语料_{corpus.id}') 
            if c.isalnum() or c in (' ', '-', '_')
        ).rstrip()
        
        download_filename = f"{safe_corpus_title}_{original_filename}"

        # 修改响应头，添加文件名
        # 使用 RFC 6266 规范支持中文文件名
        response.headers['Content-Disposition'] = f"attachment; filename*=UTF-8''{quote(download_filename)}"
        
        return response
    
    else:
        return jsonify({"code":400,"error":"无有效参数"}),400


#按语料语言类别统计
@bp.route('/statistic/language', methods=['GET'])
def statistic_language():
    """按语言统计当前未删除且审核通过的文献数。"""
    language_count_tuples = db.session.query(Corpus.language,func.count().label('count')).filter(Corpus.is_deleted == 0,Corpus.status == CorpusStatus.APPROVE_EXAMINE.value).group_by(Corpus.language).all()
    res = []
    for language,count in language_count_tuples:
        res.append({
            'language': language,
            'count': count
        })

    return success(res)


#按语料分类统计
@bp.route('/statistic/category', methods=['GET'])
def statistic_category():
    """按有效文献类型返回审核通过数量，无文献的类型保留零计数。"""
    categories = db.session.query(CorpusCategory.id,CorpusCategory.category_name).filter(CorpusCategory.is_deleted == 0).all()
    res = {}
    for category_id,category_name in categories:
        res[category_id] = {
            "category_name": category_name,
            "count":0,
        }
    category_count_tuples = db.session.query(Corpus.category_id,func.count().label('count')).filter(Corpus.is_deleted == 0,Corpus.status == CorpusStatus.APPROVE_EXAMINE.value).group_by(Corpus.category_id).all()
    for category_id, count in category_count_tuples:
        if category_id in res:
            res[category_id]['count'] = count

    return success(list(res.values()))


@bp.route('/statistic/year', methods=['GET'])
def statistic_year():
    """按出版年份统计审核通过文献数量，供首页「文献年份分布」折线使用。

    Returns:
        形如 [{"year": 2016, "count": 12}] 的列表，按年份升序；无年份的文献不计入。
    """
    rows = db.session.query(Corpus.publish_time, func.count().label('count')).filter(
        Corpus.is_deleted == 0,
        Corpus.status == CorpusStatus.APPROVE_EXAMINE.value,
        Corpus.publish_time.isnot(None),
    ).group_by(Corpus.publish_time).order_by(Corpus.publish_time.asc()).all()
    return success([{'year': int(year), 'count': count} for year, count in rows if year])


#按语料主题分类统计
@bp.route('/statistic/database', methods=['GET'])
def statistic_database():
    """将已审核文献的主题关联计数向祖先累加，组装主题树。

    同一文献关联多个节点时按关联分别计数，因此祖先汇总不是跨节点去重文献数。

    Returns:
        含每个节点名称、计数和 children 的根节点列表。
    """
    # 1. 固定有效主题节点，并按文献与主题关联聚合已审核数量
    databases ={ database.id:{ "database_id":database.id,"database_name":database.database_name,"parent_id": database.parent_id,"count":0,"children":[]}  for database in db.session.query(Database).filter(Database.is_deleted == 0).all()}

    database_count_tuples = db.session.query(CorpusDatabase.database_id,func.count().label('count')).outerjoin(Corpus,Corpus.id == CorpusDatabase.corpus_id).outerjoin(Database,Database.id == CorpusDatabase.database_id).filter(Corpus.is_deleted == 0,Corpus.status == CorpusStatus.APPROVE_EXAMINE.value,Database.is_deleted == 0).group_by(CorpusDatabase.database_id).all()
    # 2. 将每项关联数量向祖先累加，保留现有按关联计数而非跨节点去重的口径
    for database_id,count in database_count_tuples:
        databases[database_id]['count'] = count
        parent_id = databases[database_id]['parent_id']
        while parent_id != -1:
            databases[parent_id]['count'] += count
            parent_id = databases[parent_id]['parent_id']

    # 3. 按父子关系构造树，根节点及子节点均保留有效的零计数主题
    items = list(databases.values())
    res = []
    queue = Queue()
    for item in items:
        if item['parent_id'] == -1:
            res.append(item)
            queue.put(item)

    while not queue.empty():
        current = queue.get()
        for item in items:
            if item['parent_id'] == current['database_id']:
                current['children'].append(item)
                queue.put(item)

    return success(res)


@bp.route('/database', methods=['GET'])
def database_roots():
    """返回未删除的主题库树（含各级子库），供前端菜单与子库筛选使用。

    Returns:
        根节点列表，每个节点含 database_id、database_name、children。
    """
    rows = db.session.query(Database.id, Database.database_name, Database.parent_id).filter(
        Database.is_deleted == 0
    ).order_by(Database.id.asc()).all()

    nodes = {
        node_id: {
            'database_id': node_id,
            'database_name': node_name,
            'parent_id': parent_id,
            'children': [],
        }
        for node_id, node_name, parent_id in rows
    }

    roots = []
    for node in nodes.values():
        parent = nodes.get(node['parent_id'])
        # 有效根节点：父节点为 -1 / 不存在，或自身成环
        if parent is None or node['parent_id'] == node['database_id']:
            roots.append(node)
        else:
            parent['children'].append(node)

    return success(roots)


@bp.route('/database/corpus', methods=['GET'])
def database_corpus():
    """按主题库检索已审核公开文献，支持子库与多维条件筛选。

    参数：
        database_id: 根主题库 ID（必填）。
        sub_database_id: 指定子库时只统计该子库，未指定则含其全部后代。
        title / author / language / category_id / year：可选筛选条件。
        page_num / page_size：分页参数。

    Returns:
        {"code": 200, "data": {"total": 总条数, "data": [文献简要信息]}}
    """
    args = request.args
    database_id = args.get('database_id', type=int)
    sub_database_id = args.get('sub_database_id', type=int)
    title = args.get('title')
    author = args.get('author')
    language = args.get('language')
    category_id = args.get('category_id', type=int)
    year = args.get('year', type=int)
    page_size = args.get('page_size', default=10, type=int)
    page_num = args.get('page_num', default=1, type=int)
    if not database_id:
        return jsonify({'code': 400, 'error': '缺少数据库 ID'}), 400

    # 1. 目标库集合：指定子库只取该库，否则取该库及其全部后代
    all_rows = db.session.query(Database.id, Database.parent_id).filter(Database.is_deleted == 0).all()
    children_map = {}
    for node_id, parent_id in all_rows:
        children_map.setdefault(parent_id, []).append(node_id)

    if sub_database_id:
        target_ids = [sub_database_id]
    else:
        target_ids = []
        queue = [database_id]
        while queue:
            current = queue.pop()
            target_ids.append(current)
            queue.extend(children_map.get(current, []))

    # 2. 筛选条件：只统计未删除且审核通过的文献
    filters = [
        Corpus.is_deleted == 0,
        Corpus.status == CorpusStatus.APPROVE_EXAMINE.value,
        CorpusDatabase.database_id.in_(target_ids),
    ]
    if title:
        filters.append(or_(Corpus.title_cn.like(f'%{title}%'), Corpus.title_en.like(f'%{title}%')))
    if author:
        filters.append(or_(Corpus.author.like(f'%{author}%'), Corpus.translator.like(f'%{author}%'),
                           Corpus.editor.like(f'%{author}%')))
    if language:
        filters.append(Corpus.language == language)
    if category_id:
        filters.append(Corpus.category_id == category_id)
    if year:
        filters.append(Corpus.publish_time == year)

    query = db.session.query(
        Corpus.id, Corpus.title_cn, Corpus.title_en, Corpus.author, Corpus.language,
        CorpusCategory.category_name, Corpus.publish_time, Corpus.keyword_cn,
        Corpus.keyword_en, Corpus.word_number, Corpus.original_source
    ).join(CorpusDatabase, CorpusDatabase.corpus_id == Corpus.id).outerjoin(
        CorpusCategory, Corpus.category_id == CorpusCategory.id
    ).filter(and_(*filters)).distinct()

    corpus_page = query.order_by(Corpus.id.desc()).paginate(page=page_num, per_page=page_size, error_out=False)

    res = []
    for (corpus_id, title_cn, title_en, author, lang, category_name, publish_time,
         keyword_cn, keyword_en, word_number, original_source) in corpus_page.items:
        res.append({
            'corpus_id': corpus_id,
            'title_cn': title_cn,
            'title_en': title_en,
            'author': author,
            'language': lang,
            'category_name': category_name,
            'publish_time': publish_time,
            'keyword_cn': json.loads(keyword_cn) if keyword_cn else None,
            'keyword_en': json.loads(keyword_en) if keyword_en else None,
            'word_number': word_number,
            'original_source': original_source,
        })

    return success({'total': corpus_page.total, 'data': res})



@bp.route('/corpus_detail',methods = ['GET'])
def get_corpus_detail():
    """返回有效已审核文献详情，补充附件显示名及所属主题数据库。"""
    corpus_id = request.args.get('corpus_id')
    corpus = Corpus.query.filter(Corpus.id == corpus_id, Corpus.is_deleted == 0,Corpus.status == CorpusStatus.APPROVE_EXAMINE.value).first()
    if not corpus: return jsonify({"code":400,"error":"无该语料"}),400
    corpus_databases = db.session.query(Database.id, Database.database_name).join(CorpusDatabase,CorpusDatabase.database_id == Database.id).filter(CorpusDatabase.corpus_id == corpus.id).all()
    res = corpus.to_dict()

    # 处理附件信息，返回文件名（去掉路径）
    if corpus.attachment:
        try:
            attachment_data = json.loads(corpus.attachment)
            file_info = []
            for item in attachment_data:
                file_path = item.get('file_path', '')
                original_filename = item.get('original_filename', '')
                # 从file_path中提取文件名（去掉路径）
                filename = original_filename or (file_path.split('/')[-1] if '/' in file_path else file_path)
                file_info.append({
                    'filename': filename,
                    'file_path': file_path,
                    'original_filename': original_filename
                })
            res['files'] = file_info
        except (json.JSONDecodeError, TypeError):
            res['files'] = []
    else:
        res['files'] = []

    res['database'] = [ {"database_id":database_id,"database_name":database_name} for database_id,database_name in corpus_databases]
    return jsonify({"code":200,"data":res})

# import traceback
# import pandas as pd
# @bp.route('/add_corpus_by_file',methods = ['POST'])
# def add_corpus_by_file():
#     file = request.files.get('file')
#     try:
#         df = pd.read_excel(file ,skiprows=1)
#         df.columns = ['author','original_title','translated_title','type','source','language','publish_time','ISBN','format','collector','category','other']
#         for index, row in df.iterrows():
#             a = row.to_dict()
#     except Exception as e:
#         print(e)
#         db.session.rollback()
#         return jsonify({
#             'code': 400,
#             'error': f'{str(e)}'
#         }),400
#     return jsonify({
#         'code': 200,
#         'success': '请求成功'
#     })
