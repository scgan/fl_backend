"""按登录角色和文献归属校验文件引用，再代理独立文件服务返回图片或附件。"""

import json
from flask import Blueprint, request, jsonify, g

from app.models import User, Corpus
from app.utils.decorators import permission_required
from app.utils.enums import UserRole
from app.utils.utils import get_file

bp = Blueprint('file', __name__)

#获取教工一卡通图片
@bp.route('/picture', methods=['GET'])
@permission_required(['COLLECTOR', 'LEADER', 'ADMIN'])
def get_picture():
    """校验账号图片引用并代理返回；收集者只能访问本人的身份图片。"""
    user_id = request.args.get('user_id',type=int)
    if not user_id: return jsonify({"code":400,"error":"无用户id"}),400
    current_user = g.current_user
    user_role = g.role
    #语料收集者只能查看自己的教工一卡通图片
    if user_role == UserRole.COLLECTOR.value and user_id != current_user.id: return jsonify({"code":400,"error":"语料收集者只能查看自己的教工一卡通图片"}),400

    user = User.query.filter(User.id == user_id,User.is_deleted == 0).first()

    if not user or not user.picture or user.picture == '': return jsonify({"code":400,"error":"该用户不存在或用户未上传教工一卡通图片"}),400

    file, code = get_file(user.picture)
    if code != 200: return jsonify({"code": code, "error": file}), code

    return file

#获取语料文件
@bp.route('/corpus', methods=['GET'])
@permission_required(['COLLECTOR', 'LEADER', 'ADMIN'])
def get_corpus():
    """校验文献归属和当前附件清单后代理文件服务。

    Returns:
        文件响应或错误 JSON；收集者仅允许下载自己的文献，附件路径不能任意指定。
    """
    corpus_id = request.args.get('corpus_id',type=int)
    attachment_path = request.args.get('attachment_path')
    if not corpus_id: return jsonify({"code":400,"error":"无语料id"}),400
    current_user = g.current_user
    user_role = g.role
    corpus = Corpus.query.filter(Corpus.id == corpus_id,Corpus.is_deleted == 0,
                                 Corpus.owner_id.is_(None), Corpus.status != 4).first()
    if not corpus: return jsonify({"code":400,"error":"不存在该语料"}),400

    #语料收集者只能查看自己的语料文件
    if user_role == UserRole.COLLECTOR.value and corpus.contributor != current_user.id:
        return jsonify({"code":400,"error":"语料收集者只能查看自己的语料文件"}),400


    current_attachment = json.loads(corpus.attachment) if corpus.attachment else []
    current_attachment_path = [item['file_path'] for item in current_attachment]
    if attachment_path not in current_attachment_path: return jsonify({"code":400,"error":"语料无该附件"}),400

    file,code = get_file(attachment_path)
    if code != 200: return jsonify({"code": code, "error": file}), code

    return file
