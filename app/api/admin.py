"""提供管理员的账号审批、用户维护、文献类型、关键词和主题数据库管理接口。"""

from flask import Blueprint, request, jsonify, g

from app.utils.utils import delete_file, upload_file
from app.exts import db

from app.models import User, CorpusCategory, Database, Keywords, Corpus, CorpusDatabase
from app.utils.decorators import permission_required
from app.utils.enums import UserStatus, CorpusStatus, KeywordLanguage

bp = Blueprint('admin', __name__)

# 查看所有注册待审批人员
@bp.route('/users/waiting_examine', methods=['GET'])
@permission_required(['ADMIN'])
def get_waiting_examine_users():
    """按注册资料和角色筛选待审核账号，返回有效账号分页及总数。"""
    data = request.args
    username = data.get('username')
    name = data.get('name')
    phone = data.get('phone')
    role = data.get('role')
    page_num = data.get('page_num', default=1,type=int)
    page_size = data.get('page_size', default=10,type=int)
    filters = [User.status == UserStatus.WAITING_EXAMINE.value,User.is_deleted == 0]
    if username: filters.append(User.username.like(f'%{username}%'))
    if name: filters.append(User.name.like(f'%{name}%'))
    if phone: filters.append(User.phone.like(f'%{phone}%'))
    if role: filters.append(User.role.like(f'%"{role}"%'))

    users_pagination = db.session.query(User).filter(*filters).paginate(page=page_num, per_page=page_size, error_out=False)
    res = []
    for user in users_pagination.items:
        res.append(user.to_dict())
    return jsonify({
        'code': 200,
        'data': {
            'total': users_pagination.total,
            'pages': users_pagination.pages,
            'users': res
        }
    })

# 注册审批，0-通过，1-拒绝
@bp.route('/users/examine', methods=['GET'])
@permission_required(['ADMIN'])
def review_user():
    """处理账号注册审批，operation 为 0 时通过、为 1 时驳回并软删除。

    Returns:
        审批结果 JSON；查询或提交失败时返回错误响应并回滚当前事务。
    """
    data = request.args
    operation = data.get('operation',type=int)
    user_id = data.get('user_id')
    if operation is None or operation not in [0,1]: return jsonify({'code': 400,'error': '审批操作参数错误'}),400
    try:
        user = User.query.filter(User.id == user_id, User.is_deleted == 0,User.status == UserStatus.WAITING_EXAMINE.value).first()
        if not user: return jsonify({'code': 404,'error': '无该待审核的用户'}),404
        user.status = UserStatus.APPROVE_EXAMINE.value if operation == 0 else UserStatus.REJECT_EXAMINE.value
        if operation == 1:
            user.is_deleted = 1
        db.session.commit()
        return jsonify({'code': 200,'success': '审批完成'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'code': 400, 'error': str(e)}),400


# 超级管理员删改查用户
# @bp.route('/users/add', methods = ['POST'])
# @permission_required(['ADMIN'])
# def add_user():
#     data = request.form
#     username = data.get('username')
#     password = data.get('password')
#     email = data.get('email')
#     name = data.get('name')
#     phone = data.get('phone')
#     role = data.get('role')
#     picture = request.files.get('picture')
#     if not username or username == '' or not password or password == '':
#         return jsonify({"code": 400, "error": "账号密码格式错误或账号密码为空"}), 400
#     try:
#         if User.query.filter(User.username == username, User.is_deleted == 0).count() > 0: return jsonify({"code": 400, "error": "该账号已注册"}), 400
#         # password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
#         user = User(username=username, password=password, email=email, name=name, phone=phone, role=role, status=0)
#
#         files_dict = {}
#         for i, tmp_file in enumerate(picture):
#             # 将每个文件的 filename, stream 和 content_type 提取出来
#             files_dict[f'file{i}'] = (tmp_file.filename, tmp_file.stream, tmp_file.content_type)
#         params = {
#             "username": user.username,  # 查询参数
#             "upload_type": "picture"
#         }
#         response = requests.post('http://127.0.0.1:5001/upload', params=params, files=files_dict)
#         if response.status_code == 200:
#             # 解析返回的 JSON 数据
#             data = response.json()
#             # 获取文件路径列表
#             file_paths = data.get('file_paths', [])
#             user.picture = file_paths[0]
#
#         db.session.add(user)
#         db.session.commit()
#     except Exception as e:
#         db.session.rollback()
#         return jsonify({"code": 400, "error": f'添加用户失败: {str(e)}'})
#
#     return jsonify({'code': 200, 'success': '添加用户成功'})

#删除用户
@bp.route('/users/delete', methods=['POST'])
@permission_required(['ADMIN'])
def delete_user():
    """按 JSON 中的 user_id 软删除账号，不物理删除账号或其业务记录。"""
    user_id = request.get_json().get('user_id')
    if not user_id : return jsonify({'code': 400, 'error': '用户id错误'}),400
    try:
        user = User.query.filter_by(id=user_id, is_deleted=0).first()
        if user is None: return jsonify({'code': 404, 'error': '用户不存在'}),404,
        user.is_deleted = 1
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'code': 400, 'error': f'删除用户失败: {str(e)}'}),400
    return jsonify({'code': 200, 'success': '删除用户成功'})

#修改用户信息
@bp.route('/users/update', methods=['POST'])
@permission_required(['ADMIN'])
def update_user():
    """由管理员更新账号资料、角色列表及可选身份图片。

    文件替换先删除旧图片再上传新图，文件服务不参与数据库事务；
    失败时数据库回滚不能恢复已删除文件，保持现有调用顺序。

    Returns:
        更新结果 JSON，或参数、文件服务、数据库错误响应。
    """
    form_data = request.form
    user_id = form_data.get('user_id')
    username = form_data.get('username')
    # password = form_data.get('password')
    name = form_data.get('name')
    email = form_data.get('email')
    phone = form_data.get('phone')
    role = form_data.getlist('role')
    picture = request.files.get('picture')
    if user_id is None or int(user_id) <= 0: return jsonify({'code': 400, 'error': '用户id错误'}),400

    try:
        user = User.query.filter_by(id=user_id, is_deleted=0).first()
        if user is None: return jsonify({'code': 404, 'error': '用户不存在'}),404
        if username is not None: user.username = username
        # if password is not None: user.password = password
        if name is not None: user.name = name
        if email is not None: user.email = email
        if phone is not None: user.phone = phone
        if role is not None: user.role = User.role_list_to_json(role)
        if picture:
            # 删除旧文件
            if user.picture: delete_file(user.picture)
            # 上传图片
            upload_result, code = upload_file(picture, 'picture', user.username)
            if code != 200: return jsonify({"code": code, "error": upload_result}), code
            user.picture = upload_result[0]
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'code': 400, 'error': f'更新用户失败: {str(e)}'}),400
    return jsonify({'code': 200, 'success': '更新用户成功'})


# 超级管理员查询所有用户
@bp.route('/users/get', methods=['GET'])
@permission_required(['ADMIN'])
def list_all_users():
    """按姓名筛选未删除账号，返回公开资料分页，不包含密码哈希。"""
    data = request.args
    name = data.get('name')
    page_num = data.get('page_num', default=1, type=int)
    page_size = data.get('page_size', default=10, type=int)
    filters = [User.is_deleted == 0]
    if name:
        filters.append(User.name.like(f'%{name}%'))

    users_pagination = db.session.query(User).filter(*filters).paginate(page=page_num, per_page=page_size, error_out=False)
    res = [user.to_dict() for user in users_pagination.items]
    return jsonify({
        'code': 200,
        'data': {
            'total': users_pagination.total,
            'pages': users_pagination.pages,
            'users': res
        }
    })
  

# @bp.route('/users/search', methods=['GET'])
# @permission_required(['ADMIN'])
# def search_user():
#     page_num = int(request.args.get('page_num', 1))
#     page_size = int(request.args.get('page_size', 10))
#     name = request.args.get('name')
#
#     users_pagination = db.session.query(User).filter(User.name.like('%'+name+'%'), User.is_deleted == 0)\
#         .paginate(page=page_num, per_page=page_size, error_out=False)
#
#     waiting_examine_users = users_pagination.items
#     res = []
#     for user in waiting_examine_users:
#         res.append({
#             'id': user.id,
#             'username': user.username,
#             'name': user.name,
#             'email': user.email,
#             'phone': user.phone,
#             'picture': user.picture,
#             'role': user.role,
#             'status': user.status
#             })
#     return jsonify({
#         'code': 200,
#         'data': {
#             'total': users_pagination.total,
#             'pages': users_pagination.pages,
#             'users': res
#         }
#     })

# 语料类型增加
@bp.route('/category/add', methods=['POST'])
@permission_required(['ADMIN'])
def add_corpus_category():
    """校验名称和业务编码后新增文献类型，在提交失败时回滚。"""
    try:
        data = request.json
        category_name = data.get('category_name')
        code = data.get('code')
        if not category_name or len(category_name) == 0 or not code or len(code) == 0:
            return jsonify({"code":400,"error":"缺少必要参数"}),400
        if CorpusCategory.query.filter(CorpusCategory.code == code,CorpusCategory.is_deleted == 0).count() > 0: return jsonify({"code":400,"error":f"code为{code}的种类已存在"}),400

        # 创建新的CorpusCategory对象
        new_category = CorpusCategory(category_name=category_name,code=code,number=10001)

        db.session.add(new_category)
        db.session.commit()

        return jsonify({'code': 200, 'success': '添加语料类型成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'code': 400, 'error': f'添加语料类型失败: {str(e)}'}),400

# 语料类型删除
@bp.route('/category/delete', methods=['POST'])
@permission_required(['ADMIN'])
def delete_corpus_category():
    """拒绝删除仍有关联已审核文献的类型，否则设置软删除标记。"""
    try:
        category_id = request.get_json().get('category_id')
        if category_id is None or int(category_id) <= 0: return jsonify({'code': 400, 'error': '语料类型id错误'}),400

        category = CorpusCategory.query.filter_by(id=category_id, is_deleted=0).first()
        if category is None: return jsonify({'code': 404, 'error': '语料类型不存在'}),404
        if Corpus.query.filter(Corpus.category_id == category.id,Corpus.is_deleted == 0,Corpus.status == CorpusStatus.APPROVE_EXAMINE.value).count() > 0:
            return jsonify({'code': 400, 'error': '该语料类型下存在审核通过的语料，不能删除'}),400
        category.is_deleted = 1
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'code': 400, 'error': f'删除语料类型失败: {str(e)}'}),400
    return jsonify({'code': 200,'success': '删除语料类型成功'})

# 语料类型修改
@bp.route('/category/update', methods=['POST'])
@permission_required(['ADMIN'])
def update_corpus_category():
    """更新文献类型名称和编码；编码按现有有效记录进行重复校验。"""
    try:
        data = request.json
        category_id = data.get('category_id')
        category_name = data.get('category_name')
        code = data.get('code')
        if category_id is None or int(category_id) <= 0: return jsonify({'code': 400, 'error': '语料类型id错误'}),400

        category = CorpusCategory.query.filter_by(id=category_id, is_deleted=0).first()
        if category is None: return jsonify({'code': 404, 'error': '语料类型不存在'}),404

        if category_name is not None: category.category_name = category_name
        if code is not None:
            if CorpusCategory.query.filter(CorpusCategory.code == code,CorpusCategory.is_deleted == 0).count() > 0: return jsonify({'code': 400, 'error': f'code为{code}的语料类型已存在'}),400
            category.code = code

        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'code': 400, 'error': f'更新语料类型失败: {str(e)}'}),400

    return jsonify({'code': 200,'success': '更新语料类型成功'})

# 语料类型查询
# @bp.route('/corpus_category/search', methods=['GET'])
# @permission_required(['ADMIN'])
# def search_corpus_category():
#     category_name = request.args.get('category_name')
#     code = request.args.get('code')
#     filters = []
#     if category_name is not None:
#         filters.append(CorpusCategory.category_name.like('%'+category_name+'%'))
#     if code is not None:
#         filters.append(CorpusCategory.code == code)
#     categories = CorpusCategory.query.filter(CorpusCategory.is_deleted == 0, *filters).all()
#     res = []
#     for category in categories:
#         res.append({
#             'id': category.id,
#             'category_name': category.category_name,
#             'code': category.code,
#             'number': category.number,
#         })
#
#     return jsonify({'code': 200, 'data': {'categories': res}})


# 超级管理员添加关键词keyword
@bp.route('/keyword/add', methods=['POST'])
@permission_required(['ADMIN'])
def add_keyword():
    """新增管理员来源的关键词，按有效关键词文本去重。"""
    keyword = request.json.get('keyword')
    language = int(request.json.get('language'))
    if not keyword or len(keyword) == 0: return jsonify({"code":400,"error":"参数错误"}),400
    try:
        existing_keyword = Keywords.query.filter_by(keyword=keyword, is_deleted=0).first()

        if existing_keyword:
            return jsonify({"code":400,"error":"关键词已存在"}),400
        else:
            new_keyword = Keywords(keyword=keyword, flag=0, frequency=0,language=language)  # 将frequency设置为0
            db.session.add(new_keyword)
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code":400,"error":str(e)}),400
    return jsonify({"code": 200, "success":"关键词添加成功"})


# 超级管理员删除关键词keyword
@bp.route('/keyword/delete', methods=['GET'])
@permission_required(['ADMIN'])
def delete_keyword():
    """按查询参数 keyword_id 软删除关键词，不改写文献已有关键词文本。"""
    keyword_id = request.args.get('keyword_id')
    if not keyword_id: return jsonify({"code":400,"error":"缺少关键词id"}),400
    try:
        keyword = Keywords.query.filter(Keywords.id == keyword_id,Keywords.is_deleted == 0).first()
        if not keyword: return jsonify({"code":404,"error":"无该关键词"}),404
        keyword.is_deleted = 1
        db.session.commit()
        return jsonify({"code":200,"success":"删除关键词成功"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code":400,"error":str(e)}),400

# 超级管理员修改关键词keyword
@bp.route('/keyword/update', methods=['POST'])
@permission_required(['ADMIN'])
def update_keyword():
    """从表单更新有效关键词的文本及现有条件允许的语言编码。"""
    data = request.form
    keyword_id = data.get('keyword_id')
    keyword = data.get('keyword')
    language = data.get('language',type=int)
    try:
        current_keyword = Keywords.query.filter(Keywords.id == keyword_id,Keywords.is_deleted == 0).first()
        if not current_keyword: return jsonify({"code":404,"error":"无该关键词"}),404
        if keyword and len(keyword.strip()) > 0:
            if Keywords.query.filter(Keywords.keyword == keyword,Keywords.is_deleted == 0).count() > 0:
                return jsonify({"code": 400, "error": f"keyword为{keyword}的关键词已存在"}), 400
            current_keyword.keyword = keyword
        if language and language in [item.value for item in KeywordLanguage]: current_keyword.language = language
        db.session.commit()
        return jsonify({"code":200,"success":"修改关键词成功"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"code":400,"error":str(e)}),400

#TODO 文献库的增删改查（有子节点的文献库不能删除）
@bp.route('/database/add',methods = ['POST'])
@permission_required(['ADMIN'])
def database_add():
    """创建主题数据库节点；parent_id 为 -1 表示根，否则必须存在有效父节点。"""
    data = request.form
    database_name = data.get('database_name')
    parent_id = data.get('parent_id',default=-1,type=int)
    if not database_name or len(database_name) == 0: return jsonify({"code":400,"error":"参数错误，无数据库名"}),400
    if parent_id != -1 and Database.query.filter(Database.id == parent_id,Database.is_deleted == 0).count() == 0:
        return jsonify({"code":404,"error":"无该数据库"}),404
    try:
        database = Database(database_name=database_name,parent_id=parent_id)
        db.session.add(database)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code":400,"error":str(e)}),400
    return jsonify({"code": 200, "success":"数据库添加成功"})

@bp.route('/database/delete',methods = ['POST'])
@permission_required(['ADMIN'])
def database_delete():
    """仅在没有有效子节点和关联已审核文献时软删除主题节点。"""
    try:
        data = request.get_json()
        database_id = data.get('database_id')
        if not database_id or int(database_id) <= 0: return jsonify({"code":400,"error":"参数错误，无数据库id"}),400
        database = Database.query.filter_by(id=database_id,is_deleted=0).first()
        if database is None: return jsonify({"code":404,"error":"无该数据库"}),404
        # 判断是否有子节点
        children_count = Database.query.filter_by(parent_id=database_id,is_deleted=0).count()
        if children_count > 0 or db.session.query(CorpusDatabase).join(Corpus,Corpus.id == CorpusDatabase.corpus_id).filter(Corpus.is_deleted == 0,Corpus.status ==  CorpusStatus.APPROVE_EXAMINE.value,CorpusDatabase.database_id == database.id).count() > 0:
            return jsonify({"code":400,"error":"该数据库有子节点或数据库下存在审核通过的语料，不能删除"}),400

        database.is_deleted = 1
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code":400,"error":str(e)}),400
    return jsonify({"code": 200, "success":"数据库删除成功"})

@bp.route('/database/update',methods = ['POST'])
@permission_required(['ADMIN'])
def database_update():
    """从表单更新主题名称及父节点，提交失败时回滚并返回错误响应。"""
    data = request.form
    database_id = data.get('database_id')
    database_name = data.get('database_name')
    parent_id = data.get('parent_id', type=int)
    if not database_id or int(database_id) <= 0: return jsonify({"code":400,"error":"参数错误，无数据库id"}),400

    try:
        database = Database.query.filter_by(id=database_id, is_deleted=0).first()
        if database is None: return jsonify({"code":404, "error":"无该数据库"}),404
        if database_name is not None:
            if Database.query.filter(Database.name == database_name,Database.is_deleted == 0).count() > 0:
                return jsonify({"code": 400, "error": "该数据库名已存在"}), 400
            database.database_name = database_name
        if parent_id is not None:
            if parent_id != -1 and Database.query.filter(Database.id == parent_id,Database.is_deleted == 0).count() == 0:
                return jsonify({"code":404,"error":"无该数据库"}),404
            database.parent_id = parent_id
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code":400,"error":str(e)}),400
    return jsonify({"code": 200, "success":"数据库更新成功"})

