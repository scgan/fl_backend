"""提供登录用户的文献查询、分类数据、附件翻译提交和翻译结果回调接口。"""

import hmac
import hashlib
import base64 as b64
import json
import time
import uuid
import os
import re
import requests
from flask import request, jsonify, current_app
from werkzeug.utils import secure_filename
from flask import Blueprint, request, jsonify, g, current_app
from sqlalchemy import or_, and_
from werkzeug.utils import secure_filename

from app.exts import db
from app.models import Corpus, Keywords, CorpusCategory, Database, CorpusDatabase
from app.utils.decorators import permission_required, redis_client
from app.utils.enums import UserRole

# 翻译结果缓存 key 前缀，TTL 24 小时（百度链接带 auth_key 有时效）
BAIDU_TRANS_RESULT_KEY = "baidu_trans:result:"
BAIDU_TRANS_RESULT_TTL = 86400
BAIDU_TRANS_META_KEY = "baidu_trans:meta:"
BAIDU_TRANS_META_TTL = 86400
BAIDU_TRANS_ERROR_KEY = "baidu_trans:error:"
BAIDU_TRANS_ERROR_TTL = 86400
# 前端轮询超时阈值（秒）。超过这个时间仍无结果/失败回调，则认为超时。
BAIDU_TRANS_TIMEOUT_SEC = 20 * 60

bp = Blueprint('corpus', __name__)


#查询语料
@bp.route('/get',methods = ['GET'])
@permission_required(['LEADER','ADMIN','COLLECTOR'])
def get_corpus():
    """按文献元数据和贡献者筛选有效记录并分页；非草稿结果按提交时间倒序。"""
    data = request.args
    corpus_id = data.get('corpus_id')
    title_cn = data.get('title_cn')
    title_en = data.get('title_en')
    status = data.get('status')
    category_id = data.get('category_id')
    language = data.get('language')
    author = data.get('author')
    contributor = data.get('user_id')
    page_size = data.get('page_size', default=10, type=int)
    page_num = data.get('page_num', default=1, type=int)
    filters = [Corpus.owner_id.is_(None), Corpus.status != 4, Corpus.is_deleted == 0]
    if corpus_id: filters.append(Corpus.id == corpus_id)
    if title_cn: filters.append(Corpus.title_cn.like(f'%{title_cn}%'))
    if title_en: filters.append(Corpus.title_en.like(f'%{title_en}%'))
    if status: filters.append(Corpus.status == status)
    if category_id: filters.append(Corpus.category_id == category_id)
    if language: filters.append(Corpus.language == language)
    if author: filters.append(or_(Corpus.author.like(f"%{author}%"), Corpus.translator.like(f"%{author}%"),
                                  Corpus.editor.like(f"%{author}%")))
    if contributor: filters.append(Corpus.contributor ==contributor)

    query = db.session.query(Corpus.id, Corpus.title_cn, Corpus.title_en, Corpus.status, Corpus.language,
                            CorpusCategory.category_name,Corpus.author, Corpus.keyword_cn,
                            Corpus.keyword_en, Corpus.publish_time, Corpus.word_number,Corpus.submit_time)\
        .outerjoin(CorpusCategory, Corpus.category_id == CorpusCategory.id)\
        .filter(and_(*filters))
    
    # 只有当status不等于-1时才按submit_time降序排序
    if status != '-1':
        query = query.order_by(Corpus.submit_time.desc())
    corpus_page = query.paginate(page=page_num, per_page=page_size, error_out=False)

    res = []
    for corpus_id, title_cn, title_en, status, language, category_name, author, keyword_cn, keyword_en, publish_time, word_number,submit_time in corpus_page.items:
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
            'word_number': word_number,
            'submit_time': submit_time.strftime("%Y-%m-%d %H:%M:%S") if submit_time else submit_time,
        })
    return jsonify({
        'code': 200,
        'data': {
            'total': corpus_page.total,
            'pages': corpus_page.pages,
            'data': res
        }
    })

#TODO 查看语料详情
@bp.route('/get/detail',methods = ['GET'])
@permission_required(['COLLECTOR','LEADER','ADMIN'])
def corpus_get_detail():
    """返回文献元数据、附件显示名和主题关联；收集者只能查看本人记录。"""
    corpus_id = request.args.get('corpus_id')
    corpus = Corpus.query.filter(Corpus.id == corpus_id, Corpus.is_deleted == 0,
                                 Corpus.owner_id.is_(None), Corpus.status != 4).first()
    if not corpus: return jsonify({"code":404,"error":"无该语料"}),404
    user_role = g.role
    current_user = g.current_user
    if user_role == UserRole.COLLECTOR.value and corpus.contributor != current_user.id: return jsonify({"code":403,"error":"您只能查看自己的语料"}),403
    corpus_databases = db.session.query(Database.id, Database.database_name).join(CorpusDatabase,CorpusDatabase.database_id == Database.id).filter(CorpusDatabase.corpus_id == corpus.id).all()
    res = corpus.to_dict()

    # 处理附件信息，返回文件名
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


#搜索和获取关键词
@bp.route('/keywords',methods = ['GET'])
@permission_required(['COLLECTOR','LEADER','ADMIN'])
def get_keywords():
    """分页筛选有效关键词，按来源升序、频次降序返回候选项。"""
    data = request.args
    keyword_id = data.get('keyword_id')
    keyword = data.get('keyword')
    flag = data.get('flag')
    language = data.get('language')
    page_size = data.get('page_size',default=10,type=int)
    page_num = data.get('page_num',default=1,type=int)
    filters = [Keywords.is_deleted == 0]
    if keyword_id: filters.append(Keywords.id == keyword_id)
    if keyword: filters.append(Keywords.keyword.like(f'%{keyword}%'))
    if flag: filters.append(Keywords.flag == flag)
    if language: filters.append(Keywords.language == language)
    keywords_page = Keywords.query.filter(*filters).order_by(Keywords.flag.asc(),Keywords.frequency.desc()).paginate(page=page_num,per_page=page_size, error_out=False)
    res = []
    for keyword in keywords_page.items:
        res.append({
            "keyword_id": keyword.id,
            "keyword": keyword.keyword,
            "flag": keyword.flag,
            "frequency": keyword.frequency,
            "language": keyword.language,
        })
    return jsonify({
        'code':200,
        'data':{
            'total':keywords_page.total,
            'pages':keywords_page.pages,
            'data':res
        }
    })


#获取语料类别
@bp.route('/category/get',methods = ['GET'])
@permission_required(['COLLECTOR','LEADER','ADMIN'])
def get_category():
    """按 ID、名称和业务编码筛选有效文献类型。"""
    data = request.args
    category_id = data.get('category_id')
    category_name = data.get('category_name')
    code = data.get('code')
    filters = [CorpusCategory.is_deleted == 0]
    if category_id: filters.append(CorpusCategory.id == category_id)
    if category_name: filters.append(CorpusCategory.category_name.like(f'%{category_name}%'))
    if code: filters.append(CorpusCategory.code == code)
    corpus_category = CorpusCategory.query.filter(*filters).all()
    return jsonify({
        "code": 200,
        "data": [data.to_dict() for data in corpus_category]
    })

#获取数据库
@bp.route('/database/get',methods = ['GET'])
@permission_required(['COLLECTOR','LEADER','ADMIN'])
def database_get():
    """从 parent_id 为 -1 的有效根节点开始返回主题数据库树。"""
    databases = Database.query.filter(Database.is_deleted == 0,Database.parent_id == -1).all()
    return jsonify({
        "code": 200,
        "data": get_child_databases(databases,[])
    })

def get_child_databases(databases,res):
    """递归加载有效主题子节点并追加到当前层结果。

    Args:
        databases: 当前层的主题节点列表。
        res: 当前层结果累加列表；递归子层使用独立空列表。

    Returns:
        每项包含 children 的主题节点字典列表。
    """
    if len(databases) == 0: return []
    for database in databases:
        child_databases = Database.query.filter(Database.is_deleted == 0, Database.parent_id == database.id).all()
        res.append({
            "id": database.id,
            "database_name": database.database_name,
            "parent_id": database.parent_id,
            "children": get_child_databases(child_databases,[]),
        })
    return res


# 调用百度翻译异步文档翻译API
@bp.route('/translate_document', methods=['POST'])
@permission_required(['COLLECTOR', 'ADMIN'])
def translate_document():
    """同步提交远程文档翻译任务，并缓存查询所需元数据后立即返回任务 ID。

    异步的是远程翻译处理，不是本函数的 HTTP 提交。结果经服务端回调写入 Redis，
    客户端再查询 translation_result 或通过 download_translation 下载。

    Returns:
        任务 ID JSON 或格式、提交错误；元数据缓存失败只记日志，不撤销远程任务。
    """
    # 检查是否有文件上传
    # 1. 校验上传文件和支持格式，收集远程翻译参数
    if 'file' not in request.files:
        return jsonify({"code": 400, "msg": "未上传文件"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"code": 400, "msg": "文件名为空"}), 400

    # 获取语言参数
    source_language = request.form.get('source_language', 'auto')
    target_language = request.form.get('target_language', 'zh')

    # 获取其他可选参数
    trans_image = request.form.get('transImage', '1')
    need_intervene = request.form.get('needIntervene', '0')
    model_type = request.form.get('modelType', 'llm')
    output_format = request.form.get('output_format', 'docx')  # 输出格式
    output_mode = request.form.get('output_mode', '1')  # 1-单语译文，2-双语对照

    # 支持的文件格式（根据API文档）
    allowed_extensions = {
        '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
        '.pdf', '.txt', '.xml', '.html', '.htm'
    }
    file_ext = os.path.splitext(file.filename)[1].lower()

    if file_ext not in allowed_extensions:
        return jsonify({
            "code": 400,
            "msg": f"不支持的文件格式: {file_ext}。支持格式: {', '.join(allowed_extensions)}"
        }), 400

    # 配置信息
    account = {
        "app_id": "20250915002455002",
        "secret_key": "YG14pmLfoJqzfndudgml"
    }

    try:
        current_app.logger.info("开始异步文档翻译流程...")

        # 创建翻译任务
        current_app.logger.info("创建翻译任务")
        # 2. 同步提交远程异步任务，要求成功响应含 requestId 后才能向客户端确认
        create_result = create_translation_job_async(
            account["app_id"],
            account["secret_key"],
            file,
            file_ext,
            source_language,
            target_language,
            trans_image,
            need_intervene,
            model_type,
            output_format,
            output_mode
        )

        if not create_result:
            return jsonify({
                "code": 400,
                "msg": "创建翻译任务失败，无返回结果"
            }), 400

        # 检查返回的code，根据API文档，0表示成功
        if create_result.get("code") != 0:
            error_msg = create_result.get("msg", "创建翻译任务失败")
            current_app.logger.error(f"创建任务失败: code={create_result.get('code')}, msg={error_msg}")
            return jsonify({
                "code": create_result.get("code", 400),
                "msg": error_msg
            }), 400

        # 获取requestId
        request_id = create_result.get("data", {}).get("requestId")
        if not request_id:
            return jsonify({
                "code": 400,
                "msg": "未获取到任务ID"
            }), 400

        current_app.logger.info(f"翻译任务创建成功，requestId: {request_id}")

        # 记录任务元信息，便于轮询超时与排错（不依赖回调是否成功）
        try:
            # 3. 尽力缓存原文件名、格式与提交时间，供后续轮询及下载使用
            # 远程任务已经存在，元数据缓存失败只记录，不自动撤销已提交任务。
            meta_key = f"{BAIDU_TRANS_META_KEY}{request_id}"
            meta = {
                "requestId": str(request_id),
                "createdAt": int(time.time()),
                "filename": secure_filename(file.filename),
                "source_language": source_language,
                "target_language": target_language,
                "output_format": output_format,
                "output_mode": str(output_mode),
                "modelType": model_type,
            }
            redis_client.setex(meta_key, BAIDU_TRANS_META_TTL, json.dumps(meta, ensure_ascii=False))
        except Exception as e:
            current_app.logger.warning(f"写入翻译任务元信息失败: {str(e)}")

        # 返回任务ID，客户端通过回调获取结果
        return jsonify({
            "code": 200,
            "msg": "翻译任务已提交",
            "data": {
                "requestId": request_id,
                "message": "翻译任务已提交，请等待回调通知结果"
            }
        })

    except Exception as e:
        current_app.logger.error(f"文档翻译服务内部错误: {str(e)}", exc_info=True)
        return jsonify({
            "code": 500,
            "msg": f"文档翻译服务内部错误: {str(e)}"
        }), 500

def create_translation_job_async(app_id, secret_key, file, file_ext, source_lang, target_lang,
                                 trans_image='0', need_intervene='0', model_type='llm',
                                 output_format='', output_mode='1'):
    """按现有 HMAC 契约同步创建百度远程文档翻译任务。

    Args:
        app_id: 翻译账户标识。
        secret_key: 签名密钥。
        file: 可读写游标的上传文件对象。
        file_ext: 含点的原文档扩展名。
        source_lang: 源语言，可为 auto。
        target_lang: 目标语言代码。
        trans_image: 是否翻译图片文字的远程参数。
        need_intervene: 是否人工介入的远程参数。
        model_type: 远程翻译模型类型。
        output_format: 目标文档格式。
        output_mode: 可转为整数的远程输出模式。

    Returns:
        远程响应字典；超时或异常转换为含 code、msg 的失败字典。
    """
    try:
        # 获取当前时间戳（10位）
        # 1. 读取原文件并构造确定的 JSON 正文，签名和发送必须使用同一份序列化结果
        timestamp = str(int(time.time()))

        # 读取文件内容并转换为base64
        file_content = file.read()
        file_base64 = b64.b64encode(file_content).decode('utf-8')

        # 重置文件指针
        file.seek(0)

        # 构建请求体 - 完全按照API文档格式
        request_body = {
            "from": source_lang if source_lang != 'auto' else 'auto',
            "to": target_lang,
            "input": {
                "content": file_base64,
                "format": file_ext.lstrip('.').lower(),
                "filename": secure_filename(file.filename),
                "transImage": trans_image,  # 保持为字符串
                "needIntervene": need_intervene,  # 保持为字符串
                "modelType": model_type
            },
            "output": {
                "format": output_format,  # 输出格式
                "mode": int(output_mode)  # 1-单语译文，2-双语对照
            }
        }

        # 将请求体转换为JSON字符串（紧凑格式）
        request_body_str = json.dumps(request_body, separators=(',', ':'), ensure_ascii=False)

        # 计算签名
        # Step1: 字符串1 = 请求体原始JSON字符串
        # Step2: 字符串2 = APPID + 时间戳 + 字符串1
        # 2. 按账户、时间戳及正文生成 HMAC 签名，向远程提交任务
        string2 = app_id + timestamp + request_body_str

        # Step3: 对字符串2做hmac_sha256加密
        hmac_obj = hmac.new(
            key=secret_key.encode('utf-8'),
            msg=string2.encode('utf-8'),
            digestmod=hashlib.sha256
        )

        # Step4: 对加密结果做base64编码得到X-Sign
        signature = b64.b64encode(hmac_obj.digest()).decode('utf-8')

        current_app.logger.info(f"签名计算:")
        current_app.logger.info(f"  AppID: {app_id}")
        current_app.logger.info(f"  时间戳: {timestamp}")
        current_app.logger.info(f"  字符串2长度: {len(string2)}")
        current_app.logger.info(f"  签名结果: {signature}")

        # 准备请求URL
        url = "https://fanyi-api.baidu.com/transapi/doctrans/createjob/trans"

        # 设置请求头
        headers = {
            'Content-Type': 'application/json; charset=UTF-8',
            'X-Appid': app_id,
            'X-Timestamp': timestamp,
            'X-Sign': signature
        }

        current_app.logger.info(f"文件信息: {file.filename}, 大小: {len(file_content)} 字节")

        # 发送请求
        response = requests.post(
            url,
            headers=headers,
            data=request_body_str.encode('utf-8'),
            timeout=60  # 增加超时时间，因为文件可能较大
        )

        current_app.logger.info(f"API响应状态码: {response.status_code}")

        # 记录响应内容
        response_text = response.text
        current_app.logger.info(f"API响应内容: {response_text}")

        # 3. 区分 HTTP 成功和正文业务码，保留上游业务结果交给路由判定
        if response.status_code == 200:
            try:
                result = response.json()
                return result
            except json.JSONDecodeError:
                current_app.logger.error(f"响应不是有效的JSON: {response_text[:500]}")
                return {
                    "code": 500,
                    "msg": "API返回格式错误",
                    "raw_response": response_text[:500]
                }
        else:
            current_app.logger.error(f"API请求失败，状态码: {response.status_code}")
            try:
                error_result = response.json()
                return error_result
            except:
                return {
                    "code": response.status_code,
                    "msg": f"API请求失败: {response.status_code}",
                    "raw_response": response_text[:500]
                }

    except requests.exceptions.Timeout:
        current_app.logger.error("创建翻译任务超时")
        return {
            "code": 408,
            "msg": "请求超时"
        }
    except Exception as e:
        current_app.logger.error(f"创建翻译任务异常: {str(e)}", exc_info=True)
        return {
            "code": 500,
            "msg": f"系统异常: {str(e)}"
        }


def verify_callback_signature(callback_data):
    """
    验证回调参数的签名 - 按照百度官方文档实现
    Step1: 将返回的所有参数(除了sign)按照a-z顺序，升序排序，按照K=V形式拼接中间使用&分割 (V需要做urlencode) 字符串1。
    Step2: 对字符串1 做hmac_sha256加密，密钥使用平台分配的密钥 得到字符串2。
    Step3: 对字符串2 做base64编码得到sign参数。
    """
    try:
        # 配置信息
        secret_key = "YG14pmLfoJqzfndudgml"

        # 复制回调数据，排除sign字段
        data_copy = callback_data.copy()
        received_sign = data_copy.pop('sign', None)

        if not received_sign:
            current_app.logger.error("回调数据中缺少sign字段")
            return False

        # Step1: 除了sign之外的其他参数，按照a-z顺序排序
        sorted_keys = sorted(data_copy.keys())
        sorted_params = []
        for key in sorted_keys:
            value = str(data_copy[key])
            # V需要做urlencode
            encoded_value = requests.utils.quote(value, safe='')
            sorted_params.append(f"{key}={encoded_value}")

        # 拼接字符串
        string1 = '&'.join(sorted_params)

        current_app.logger.info(f"回调签名验证 - 排序后的参数: {string1}")

        # Step2: 对字符串1做hmac_sha256加密
        hmac_obj = hmac.new(
            key=secret_key.encode('utf-8'),
            msg=string1.encode('utf-8'),
            digestmod=hashlib.sha256
        )

        # Step3: 对加密结果做base64编码
        calculated_sign = b64.b64encode(hmac_obj.digest()).decode('utf-8')

        current_app.logger.info(f"回调签名验证 - 接收签名: {received_sign}")
        current_app.logger.info(f"回调签名验证 - 计算签名: {calculated_sign}")

        # 比较签名
        return hmac.compare_digest(received_sign, calculated_sign)

    except Exception as e:
        current_app.logger.error(f"回调签名验证异常: {str(e)}", exc_info=True)
        return False


# 下载翻译结果：从百度流式代理返回（不落盘），避免 auth_key 过期/暴露
@bp.route('/download_translation/<request_id>', methods=['GET'])
@permission_required(['COLLECTOR', 'ADMIN'])
def download_translation(request_id):
    """从回调缓存读取文件地址，构造代理下载响应，不将译文保存到磁盘。

    Args:
        request_id: 与翻译回调及元数据缓存一致的任务标识。

    Returns:
        以原名和目标格式命名的流响应，或未完成、远程下载和失败通知对应的错误响应。
    """
    try:
        # 1. 查询已完成链接与任务元数据，完成链接优先于失败或未完成状态
        request_id_str = str(request_id)
        result_key = f"{BAIDU_TRANS_RESULT_KEY}{request_id_str}"
        file_url = redis_client.get(result_key)
        if file_url:
            current_app.logger.info(f"开始代理下载翻译文件 requestId={request_id_str}")

            # 读取任务元信息，用于生成更友好的文件名
            meta_key = f"{BAIDU_TRANS_META_KEY}{request_id_str}"
            meta_raw = redis_client.get(meta_key)
            orig_name = None
            target_lang = None
            output_format = None
            if meta_raw:
                try:
                    meta = json.loads(meta_raw)
                    orig_name = meta.get("filename")
                    target_lang = meta.get("target_language") or meta.get("targetLanguage")
                    output_format = (meta.get("output_format") or "").lower()
                except Exception:
                    current_app.logger.warning(f"翻译任务元信息解析失败 requestId={request_id_str}")

            # 2. 代理获取远程译文，并依据目标格式或 MIME 推导下载后缀
            with requests.get(file_url, stream=True, timeout=120) as r:
                content_type = r.headers.get("Content-Type", "application/octet-stream")
                if r.status_code != 200:
                    # 透出百度侧返回，便于定位：auth_key过期/不可访问/404等
                    body_preview = ""
                    try:
                        body_preview = r.text[:500]
                    except Exception:
                        body_preview = "<non-text body>"
                    current_app.logger.error(
                        f"百度文件拉取失败 requestId={request_id_str}, status={r.status_code}, body={body_preview}"
                    )
                    return jsonify({
                        "code": 502,
                        "msg": "百度文件拉取失败（可能链接已过期或不可访问）",
                        "requestId": request_id,
                        "baidu_status": r.status_code,
                        "baidu_body": body_preview
                    }), 502

                # 计算文件扩展名：优先根据 output_format，其次 Content-Type
                ct = (content_type or "").lower()
                if output_format in ("pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx"):
                    ext = f".{output_format}"
                elif "pdf" in ct:
                    ext = ".pdf"
                elif "word" in ct or "docx" in ct:
                    ext = ".docx"
                elif "excel" in ct or "xlsx" in ct:
                    ext = ".xlsx"
                elif "powerpoint" in ct or "pptx" in ct:
                    ext = ".pptx"
                else:
                    ext = ".pdf"

                # 根据原文件名 + 目标语言生成友好文件名：
                #   原名: foo.docx, target_lang: zh -> foo_译.docx
                filename = None
                if orig_name:
                    base, orig_ext = os.path.splitext(orig_name)
                    # 目标语言简称映射
                    lang_map = {
                        'zh': '译', 'en': '译', 'ja': '译',
                        'zh-cn': '译', 'zh-tw': '译',
                        'auto': '', 'jp': '译', 'ko': '译'
                    }
                    lang_suffix = lang_map.get(str(target_lang).lower(), f"_{target_lang}") if target_lang else ""
                    # 保持原文件扩展名
                    final_ext = ext if ext else (orig_ext if orig_ext else '.pdf')
                    filename = f"{base}{lang_suffix}{final_ext}"
                    current_app.logger.info(f"生成文件名: {filename} (原名: {orig_name}, 目标语言: {target_lang})")
                else:
                    filename = f"translated_{request_id_str}{ext}"
                    current_app.logger.warning(f"无法获取原文件名，使用默认: {filename}")

                from flask import Response

                def generate():
                    """按 256 KB 分块转发翻译文件，跳过上游空块。

                    Yields:
                        上游文件流中的非空字节块。
                    """
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            yield chunk

                # 3. 创建惰性消费上游字节的响应，并补齐文件名与长度
                # 生成器引用当前上下文中的上游响应，维护时需要一起检查响应关闭与流消费的生命周期。
                resp = Response(generate(), mimetype=content_type)
                resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
                # 可选：把百度 content-length 透传
                if r.headers.get("Content-Length"):
                    resp.headers["Content-Length"] = r.headers["Content-Length"]
                return resp
        # 若已失败，返回更明确错误
        err_key = f"{BAIDU_TRANS_ERROR_KEY}{request_id_str}"
        err_raw = redis_client.get(err_key)
        if err_raw:
            try:
                err = json.loads(err_raw)
            except Exception:
                err = {"raw": err_raw}
            return jsonify({
                "code": 409,
                "msg": "翻译失败，无法下载",
                "requestId": request_id,
                "error": err
            }), 409
        return jsonify({
            "code": 404,
            "msg": "翻译文件不存在或链接已过期",
            "requestId": request_id
        }), 404
    except Exception as e:
        current_app.logger.error(f"下载翻译异常: {str(e)}", exc_info=True)
        return jsonify({"code": 500, "msg": str(e)}), 500


# 查询翻译结果（前端轮询此接口，完成时用返回的 file_url 直接下载）
@bp.route('/translation_result/<request_id>', methods=['GET'])
@permission_required(['COLLECTOR', 'ADMIN'])
def translation_result(request_id):
    """优先读取翻译成功链接，再读失败原因，否则返回处理中状态。

    Args:
        request_id: 远程翻译任务标识，与 Redis 元数据及回调键一致。

    Returns:
        结果 JSON；正文 code 为 202 时 HTTP 仍按现有接口保持 200。
        超过预期时限只增加耗时提示，不擅自把任务标记为失败。
    """
    try:
        # 1. 优先查成功结果链接，再查失败通知，避免把已有终态显示为仍在处理
        request_id_str = str(request_id)
        # 1) 已完成：直接返回下载链接
        key = f"{BAIDU_TRANS_RESULT_KEY}{request_id_str}"
        file_url = redis_client.get(key)
        if file_url:
            return jsonify({
                "code": 200,
                "msg": "翻译完成",
                "data": {
                    "requestId": request_id,
                    "file_url": file_url,
                    "download_url": f"/api/corpus/download_translation/{request_id}",
                    "status": "completed"
                }
            })

        # 2) 已失败：返回失败原因
        err_key = f"{BAIDU_TRANS_ERROR_KEY}{request_id_str}"
        err_raw = redis_client.get(err_key)
        if err_raw:
            try:
                err = json.loads(err_raw)
            except Exception:
                err = {"raw": err_raw}
            return jsonify({
                "code": 200,
                "msg": "翻译失败",
                "data": {
                    "requestId": request_id,
                    "status": "failed",
                    "error": err
                }
            })

        # 3) 超时：有元信息且超过阈值，但仍以 processing 返回，只附带更友好的提示
        # 2. 无终态时依据元信息计算耗时，只提示长时间等待，不擅自终止远程任务
        meta_key = f"{BAIDU_TRANS_META_KEY}{request_id_str}"
        meta_raw = redis_client.get(meta_key)
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                created_at = int(meta.get("createdAt", 0))
            except Exception:
                meta = {}
                created_at = 0
            elapsed = int(time.time()) - created_at if created_at else None
            if elapsed and elapsed > BAIDU_TRANS_TIMEOUT_SEC:
                # 注意这里仍返回 processing，只是 msg 和 data.hint 更具体
                return jsonify({
                    "code": 202,
                    "msg": "翻译任务耗时较长",
                    "data": {
                        "requestId": request_id,
                        "status": "processing",
                        "elapsedSeconds": elapsed,
                        "hint": f"已超过 {BAIDU_TRANS_TIMEOUT_SEC} 秒仍未收到翻译结果回调，可能文档较大或服务繁忙，建议稍后刷新或重试。"
                    }
                })

        return jsonify({
            "code": 202,
            "msg": "翻译任务正在处理中或还未完成",
            "data": {"requestId": request_id, "status": "processing"}
        })
    except Exception as e:
        current_app.logger.error(f"查询翻译结果异常: {str(e)}", exc_info=True)
        return jsonify({"code": 500, "msg": str(e)}), 500


@bp.route('/baidu_translate_callback', methods=['GET', 'POST'])
def baidu_translate_callback():
    """兼容 GET、JSON 和表单回调，按任务标识保存翻译结果或失败原因。

    签名验证受 SKIP_BAIDU_SIGNATURE_CHECK 控制，现有默认允许跳过，
    不能将此入口描述为始终强制验证签名。报价通知只确认接收，不发布结果。

    Returns:
        回调确认 JSON；结果缓存写入失败返回 500，失败通知记录异常仅写日志。
    """
    try:
        # 获取所有请求参数 - 支持GET参数和POST数据
        # 1. 兼容上游 GET、JSON 与表单编码，统一为回调参数字典
        if request.method == 'GET':
            # 尝试多种方式获取GET参数
            callback_data = request.args.to_dict()
            if not callback_data:
                # 如果args为空，尝试values
                callback_data = request.values.to_dict()
            if not callback_data:
                # 手动解析query string
                from urllib.parse import parse_qs
                query_string = request.query_string.decode('utf-8')
                if query_string:
                    parsed = parse_qs(query_string)
                    callback_data = {k: v[0] if isinstance(v, list) and len(v) == 1 else v for k, v in parsed.items()}
        else:
            callback_data = request.get_json() or request.form.to_dict()

        # 如果还是没有参数，返回错误
        if not callback_data:
            current_app.logger.error("无法获取回调参数！")
            return jsonify({"code": 400, "msg": "无法获取回调参数"}), 400

        # 验证签名（暂时跳过以便调试）
        # 2. 计算签名并按当前兼容配置决定是否强制拒绝，缺少任务 ID 始终拒绝
        skip_signature_check = current_app.config.get('SKIP_BAIDU_SIGNATURE_CHECK', True)  # 临时设为True
        signature_valid = verify_callback_signature(callback_data)

        if not signature_valid and not skip_signature_check:
            current_app.logger.error("回调签名验证失败")
            return jsonify({"code": 400, "msg": "签名验证失败"}), 400
        elif not signature_valid and skip_signature_check:
            current_app.logger.warning("回调签名验证失败，但配置为跳过验证，继续处理")
        else:
            current_app.logger.info("回调签名验证成功")

        # 解析回调参数 - 注意URL参数都是字符串类型
        try:
            code = int(callback_data.get('code', -1))
        except ValueError:
            code = -1

        msg = callback_data.get('msg')
        request_id = callback_data.get('requestId')
        notify_type = callback_data.get('notifyType')

        current_app.logger.info(f"解析后的参数类型: code={code}({type(code)}), notify_type='{notify_type}'")

        if not request_id:
            current_app.logger.error("回调中缺少requestId")
            return jsonify({"code": 400, "msg": "缺少requestId"}), 400

        # 处理不同类型的通知
        # 3. 报价只确认，翻译成功写入下载链接，失败通知保留诊断供客户端查询
        if notify_type == 'quote':
            # 报价回调
            amount = callback_data.get('amount', 0)
            char_count = callback_data.get('charCount', 0)
            last_free_char_count = callback_data.get('lastFreeCharCount', 0)

            current_app.logger.info(f"收到报价回调 - requestId: {request_id}, 金额: {amount}分, 字符数: {char_count}")

            # 这里可以保存报价信息到数据库或缓存
            # 暂时只记录日志
            return jsonify({"code": 200, "msg": "报价回调处理成功"}), 200

        elif notify_type == 'trans':
            # 翻译回调
            if code == 0:
                file_src_url = callback_data.get('fileSrcUrl')
                if not file_src_url:
                    current_app.logger.error("翻译回调缺少 fileSrcUrl")
                    return jsonify({"code": 400, "msg": "缺少 fileSrcUrl"}), 400

                current_app.logger.info(f"翻译成功 - requestId: {request_id}, 已写入缓存")

                # 只存链接到 Redis，不下载不落盘；前端通过 translation_result / download_translation 拿链接下载
                try:
                    key = f"{BAIDU_TRANS_RESULT_KEY}{request_id}"
                    redis_client.setex(key, BAIDU_TRANS_RESULT_TTL, file_src_url)
                except Exception as e:
                    current_app.logger.error(f"写入翻译结果缓存失败: {str(e)}", exc_info=True)
                    return jsonify({"code": 500, "msg": "缓存写入失败"}), 500

                return jsonify({"code": 200, "msg": "翻译完成"}), 200

            else:
                # 翻译失败
                current_app.logger.error(f"翻译失败 - requestId: {request_id}, code: {code}, msg: {msg}")
                # 写入失败原因，供前端轮询查看
                try:
                    err_key = f"{BAIDU_TRANS_ERROR_KEY}{request_id}"
                    err = {
                        "code": code,
                        "msg": msg,
                        "notifyType": notify_type,
                        "requestId": str(request_id),
                        "timestamp": int(callback_data.get("timestamp", int(time.time()))),
                    }
                    # 额外字段（若有）
                    for k in ("amount", "charCount", "lastFreeCharCount"):
                        if k in callback_data:
                            err[k] = callback_data.get(k)
                    redis_client.setex(err_key, BAIDU_TRANS_ERROR_TTL, json.dumps(err, ensure_ascii=False))
                except Exception as e:
                    current_app.logger.error(f"写入失败原因缓存异常: {str(e)}", exc_info=True)
                return jsonify({"code": 200, "msg": "已接收翻译失败通知"}), 200

        else:
            current_app.logger.warning(f"未知的通知类型: {notify_type}")
            return jsonify({"code": 400, "msg": f"未知的通知类型: {notify_type}"}), 400

    except Exception as e:
        current_app.logger.error(f"处理百度翻译回调异常: {str(e)}", exc_info=True)
        return jsonify({"code": 500, "msg": f"回调处理异常: {str(e)}"}), 500
