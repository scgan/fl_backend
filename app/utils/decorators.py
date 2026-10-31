"""提供基于当前登录角色的路由权限装饰器及 Redis 验证码的一次性校验。"""

from functools import wraps
from flask import g, jsonify

from app.utils.enums import UserRole
import redis

redis_client = redis.Redis(host='10.222.9.151', port=6379, db=0, password='admin@huwei123', decode_responses=True)

def permission_required(role):
    """构造角色权限装饰器；role 为 UserRole 成员名称列表，比较请求中的当前角色。"""
    def decorator(f):
        """包装目标视图并保留其名称，使 Flask 路由注册与白名单识别保持一致。"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            """校验请求上下文的登录状态及当前角色，通过后转发原视图参数，否则返回 401 或 403。"""
            if not hasattr(g, 'current_user'):
                return jsonify({"code": 401, "error": "未登录"}),401
            current_user_role = g.role  # 当前登录用户角色

            required_roles = [UserRole[r].value for r in role]
            if current_user_role not in required_roles:
                return jsonify({"code":403,"error": "无权限"}),403

            return f(*args, **kwargs)

        return decorated_function

    return decorator

def verify_captcha(captcha_id, user_input):
    """校验 Redis 验证码，成功后删除以避免重复使用。

    Args:
        captcha_id: 验证码缓存键中的唯一标识。
        user_input: 用户提交的文本，比较时忽略大小写。

    Returns:
        表示校验是否通过及原因的二元组。
    """
    real_captcha = redis_client.get(f"captcha:{captcha_id}")
    if not real_captcha:
        return False, "验证码已过期或不存在"
    if user_input.lower() != real_captcha.lower():
        return False, "验证码错误"
    # 校验通过后删除，防止重放
    redis_client.delete(f"captcha:{captcha_id}")
    return True, "验证通过"
