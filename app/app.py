"""初始化 Flask、日志与业务蓝图，并在请求入口解析 JWT、加载有效账号和当前角色。"""

from flask import  Flask, request, jsonify, current_app, g
import logging
import jwt

from app.api.user import bp as user_bp
from app.api.collector import bp as collector_bp
from app.api.leader import bp as leader_bp
from app.api.admin import bp as admin_bp
from app.api.file import bp as file_bp
from app.api.corpus import bp as corpus_bp
from app.api.public import bp as public_bp
from app.api.AIsearch import bp as ai_search_bp
from app.api.topic import bp as topic_bp
from app.api.knowledge_base import bp as knowledge_base_bp
from app.api.terminology import bp as terminology_bp
from app.api.literature_files import bp as literature_files_bp
from app.exts import db
# from app.utils.utils import NLPUtils



app = Flask(__name__)
# 统一配置日志到 stdout/stderr，默认 INFO 级别，便于 docker logs 捕获
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
app.logger.setLevel(logging.INFO)
for handler in app.logger.handlers:
    handler.setLevel(logging.INFO)

# 同步设置 werkzeug 与 root logger，确保请求日志与自定义日志一致
logging.getLogger('werkzeug').setLevel(logging.INFO)
logging.getLogger().setLevel(logging.INFO)
app.config.from_pyfile("config.py")  # 加载配置
db.init_app(app)
app.register_blueprint(user_bp,url_prefix='/user')
app.register_blueprint(collector_bp,url_prefix='/collector')
app.register_blueprint(corpus_bp,url_prefix='/corpus')
app.register_blueprint(leader_bp,url_prefix='/leader')
app.register_blueprint(admin_bp,url_prefix ='/admin')
app.register_blueprint(file_bp,url_prefix ='/file')
app.register_blueprint(public_bp,url_prefix ='/public')
app.register_blueprint(ai_search_bp,url_prefix ='/ai_search')
app.register_blueprint(topic_bp, url_prefix='/topic')
app.register_blueprint(knowledge_base_bp)
app.register_blueprint(terminology_bp, url_prefix='/terminology')
app.register_blueprint(literature_files_bp)


@app.before_request
def load_user():
    """对受保护请求验证 JWT，并将有效账号和所选角色写入请求上下文。

    公开路由及认证相关白名单直接放行；其他请求必须提供 Bearer 令牌。
    账号须未删除且注册审核通过；无效令牌或账号直接返回错误，不进入业务蓝图。

    Returns:
        认证失败时的 JSON 响应及状态码；放行时返回 None。
    """
    if not request.endpoint or request.endpoint in ['user.login', 'user.leaders','user.register', 'user.get_captcha', 'user.reset_password', 'user.forget_password','corpus.baidu_translate_callback'] or request.endpoint.startswith("public."):
        return
    token = request.headers.get('Authorization')
    if token and token.startswith('Bearer '):
        token = token.split(' ')[1]
    else:
        g.current_user = None  # 如果没有 token，将当前用户设为 None
        return jsonify({"code":401,"error": "Token missing or invalid"}),401
    try:
        # 解析 token 并验证
        payload = jwt.decode(token, current_app.config['SECRET_KEY'], algorithms=['HS256'])
        username = payload.get('username')
        role = payload.get('role')
        g.role = role if role == '' else int(role)
        from app.models import User
        current_user = User.query.filter_by(username=username,is_deleted=0,status=1).first()
        if not current_user: return jsonify({"code":404,"error": "用户未注册或用户注册待审核"}),404
        g.current_user = current_user
        return
    except jwt.ExpiredSignatureError:
        return jsonify({"code":401,"error": "Token expired"}),401
    except jwt.InvalidTokenError:
        return jsonify({"code":401,"error": "Invalid token"}),401
    return




if __name__ == '__main__':
    # NLPUtils.initialize()  # 模型初始化
    app.run(host='0.0.0.0', port=5000)
