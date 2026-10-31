"""提供验证码、注册登录、角色切换、账号资料维护及邮件找回密码接口。"""

import json
import re
import hashlib
import urllib.parse
from datetime import datetime, timedelta

from flask import Blueprint, request, jsonify, current_app, g
import jwt

from app.utils.utils import upload_file, delete_file, send_email
from app.exts import bcrypt, db
from app.models import User, Corpus
from app.utils.decorators import permission_required,verify_captcha,redis_client
from app.utils.enums import UserStatus, UserRole, CorpusStatus
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import random
import string
import base64
import uuid

bp = Blueprint('user', __name__)

# 生成验证码
@bp.route("/captcha", methods=["GET"])
def get_captcha():
    # 生成验证码
    """生成含数字的四位图形验证码，缓存五分钟后返回图片和唯一标识。

    Returns:
        包含 captcha_id 和 Base64 PNG 的 JSON，验证码正文仅保存到 Redis。
    """
    # 1. 生成包含数字的验证码正文，限制随机重抽次数以保证请求能结束
    for _ in range(10):
        chars = random.choices(string.ascii_lowercase + string.digits, k=4)
        if any(c.isdigit() for c in chars):
            captcha_text = ''.join(chars)
            break
    else:
        captcha_text = 'a1b2'
    # 2. 绘制带干扰的图像，缺失字体时退回默认字体，正文不放入响应
    captcha_id = str(uuid.uuid4())

    # 生成图片
    width, height = 160, 60
    image = Image.new('RGB', (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font_path = "simsun.ttc"
    # 干扰线
    for _ in range(5):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        color = tuple(random.randint(60, 180) for _ in range(3))
        draw.line(((x1, y1), (x2, y2)), fill=color, width=2)

    # 干扰点
    for _ in range(100):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        color = tuple(random.randint(60, 180) for _ in range(3))
        draw.point((x, y), fill=color)

    # 绘制验证码字符
    char_count = len(captcha_text)
    char_centers = [int((i + 1) * width / (char_count + 1)) for i in range(char_count)]
    for i, char in enumerate(captcha_text):
        font_size = random.randint(int(height*0.6), int(height*0.8))
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
        char_color = tuple(random.randint(0, 80) for _ in range(3))
        angle = random.randint(-30, 30)
        char_img_size = font_size * 3
        char_img = Image.new('RGBA', (char_img_size, char_img_size), (255, 255, 255, 0))
        char_draw = ImageDraw.Draw(char_img)
        bbox = char_draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        char_draw.text(
            ((char_img_size - w) // 2, (char_img_size - h) // 2),
            char, font=font, fill=char_color
        )
        rotated = char_img.rotate(angle, resample=Image.BICUBIC, expand=1)
        rx, ry = rotated.size
        x = char_centers[i] - rx // 2
        y = (height - ry) // 2
        image.paste(rotated, (x, y), rotated)

    # 转base64
    buf = BytesIO()
    image.save(buf, format='PNG')
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # 存到Redis，5分钟过期
    # 3. 以唯一 ID 缓存五分钟，客户端仅获得 PNG 和 ID，成功校验后由校验器删除
    redis_client.setex(f"captcha:{captcha_id}", 300, captcha_text)

    return jsonify({
        "captcha_id": captcha_id,
        "captcha_image": image_base64
    })

#TODO 登录（返回token）
@bp.route('/login',methods = ['POST'])
def login():
    """验证一次性验证码和账号密码，签发尚未选择角色的登录令牌。

    Returns:
        token 及账号可选角色列表，或校验失败的 JSON 响应。
    """
    data = request.form
    captcha_id = data.get('captcha_id')
    captcha = data.get('captcha')
    if not captcha_id or not captcha:
        return jsonify({"code":400,"error": "缺少验证码参数"}), 400
    valid, msg = verify_captcha(captcha_id, captcha)
    if not valid:
        return jsonify({"code":400,"error": msg}), 400

    username = data.get('username')
    password = data.get('password')
    user = User.query.filter(User.username == username, User.is_deleted == 0).first()
    if not user: return jsonify({"code":400,"error":"账号未注册"}),400
    if user.status == 0: return jsonify({"code":400,"error":"账号待审批"}),400
    if not user.check_password(password): return jsonify({"code":400,"error":"账号密码错误"}),400
    token = jwt.encode({'username':user.username,'role':''}, current_app.config['SECRET_KEY'], algorithm='HS256')
    token_str = token.decode('utf-8')
    return jsonify({
        "code":200,
        "data": {
            "token":token_str,
            "role":User.role_json_to_list(user.role),
        }
    })

@bp.route('/role_select',methods = ['POST'])
def role_select():
    """校验所选角色属于当前账号后，重新签发包含当前角色的 JWT。"""
    data = request.form
    role = data.get('role')
    if not role or int(role) not in [r.value for r in UserRole]: return jsonify({"code":400,"error":"无role参数或role参数不合法"}),400
    current_user = g.current_user
    role_list = User.role_json_to_list(current_user.role)
    if int(role) not in role_list: return jsonify({"code":400,"error":"用户无该role"}),400
    token = jwt.encode({'username':current_user.username,'role':role}, current_app.config['SECRET_KEY'], algorithm='HS256')
    token_str = token.decode('utf-8')
    return jsonify({
        "code":200,
        "data": token_str,
    })

def is_strong_password(password):
    """检查密码是否至少八位，且同时含大小写字母、数字和允许的特殊字符。"""
    pattern = r'^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?`~])[A-Za-z\d!@#$%^&*()_+\-=\[\]{};\':"\\|,.<>/?`~]{8,}$'
    return re.match(pattern, password) is not None

#TODO 注册
@bp.route('/register',methods = ['POST'])
# @permission_required([''])
def register():
    """校验验证码与注册资料，上传身份图片后创建待审核账号。

    图片上传先于数据库提交，文件服务不参与数据库事务；提交失败只回滚数据库，
    不能在说明中将其描述为跨服务原子操作。

    Returns:
        注册结果 JSON 或参数、重复资料、文件服务和数据库错误响应。
    """
    # 1. 校验一次性验证码、密码规则及必填身份资料
    data = request.form
    captcha_id = data.get('captcha_id')
    captcha = data.get('captcha')
    if not captcha_id or not captcha:
        return jsonify({"code":400,"error": "缺少验证码参数"}), 400
    valid, msg = verify_captcha(captcha_id, captcha)
    if not valid:
        return jsonify({"code":400,"error": msg}), 400

    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    name = data.get('name')
    phone = data.get('phone')
    role = data.get('role')
    picture = request.files.get('picture')
    leader_id = data.get('reviewer_id')

    if not username or username == '' or not password or password == '': return jsonify({"code":400,"error":"账号密码格式错误或账号密码为空"}),400
    if not is_strong_password(password):
        return jsonify({"code":400,"error":"密码必须包含大写字母、小写字母、数字和特殊字符，且不少于8位"}),400
    if not picture:
        return jsonify({"code":400,"error":"未上传教工卡图片"}),400
    try:
        if User.query.filter(User.username == username, User.is_deleted == 0).count() > 0:
            return jsonify({"code":400,"error":"该账号已注册"})
        if email and User.query.filter(User.email == email, User.is_deleted == 0).count() > 0:
            return jsonify({"code":400,"error":"该邮箱已被绑定"}),400
        if phone and User.query.filter(User.phone == phone, User.is_deleted == 0).count() > 0:
            return jsonify({"code":400,"error":"该手机号已被绑定"}),400
        # 2. 在有效资料去重后上传身份图片，再创建待审核账号
        # 文件服务先产生副作用，数据库回滚不自动删除已上传图片。
        password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username,password=password_hash,email=email,name=name,phone=phone,role=User.role_list_to_json([role]),status=0,leader_id=leader_id)

        #上传图片
        upload_result,code = upload_file(picture,'picture',user.username)
        if code != 200: return jsonify({"code":code,"error":upload_result}),code
        user.picture = upload_result[0]

        db.session.add(user)
        # 3. 提交账号与图片引用，数据库失败回滚本次注册记录
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code":400,"error":str(e)}),400
    return jsonify({
        "code":200,
        "success":"注册成功"
    })

# 用户修改自身信息
@bp.route('/update',methods = ['POST'])
@permission_required(['ADMIN','LEADER','COLLECTOR'])
def update_user():
    """更新本人资料，可验证旧密码后改密或替换身份图片。

    图片删除和上传在提交前执行，数据库回滚不恢复远程文件。

    Returns:
        修改结果 JSON，或密码校验、上传和数据库错误响应。
    """
    data = request.form
    old_password = data.get('old_password')
    new_password = data.get('new_password')
    name = data.get('name')
    phone = data.get('phone')
    email = data.get('email')
    picture = request.files.get('picture')
    try:
        user = User.query.filter(User.id == g.current_user.id, User.is_deleted == 0).first()
        if name: user.name = name
        if phone: user.phone = phone
        if email: user.email = email
        if new_password:
            if not user.check_password(old_password): return jsonify({"code":400,"error":"原密码错误"}),400
            if not is_strong_password(new_password):
                return jsonify({"code":400,"error":"密码必须包含大写字母、小写字母、数字和特殊字符，且不少于8位"}),400
            user.set_password(new_password)
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
        return jsonify({"code": 400,"error":str(e)}),400
    return jsonify({"code":200,"success":"修改成功"})

#获取自身信息
@bp.route('/self',methods = ['GET'])
@permission_required(['ADMIN','LEADER','COLLECTOR'])
def get_self():
    """返回本人公开资料，并补充仍有效的上级审核人姓名。"""
    user_id = g.current_user.id
    user = User.query.filter(User.id == user_id, User.is_deleted == 0).first()
    leader = User.query.filter(User.id == user.leader_id, User.is_deleted == 0).first()
    leader_name = leader.name if leader else None
    return jsonify({
        "code":200,
        "data": {
            **user.to_dict(),
            "leader_name": leader_name
        }
    })

#获取用户信息
@bp.route('/get',methods = ['GET'])
@permission_required(['LEADER','ADMIN','COLLECTOR'])
def get_user():
    """筛选未删除账号，按其已审核文献数降序排序后进行内存分页。

    Returns:
        总账号数、总页数及附带 corpus_count 的当前页资料。
    """
    data = request.args
    user_id = data.get('user_id')
    username = data.get('username')
    name = data.get('name')
    phone = data.get('phone')
    role = data.get('role')
    status = data.get('status')
    page_size = data.get('page_size',default=10,type=int)
    page_num = data.get('page_num',default=1,type=int)
    filters = [User.is_deleted == 0]

    if user_id: filters.append(User.id == user_id)
    if username: filters.append(User.username.like(f'%{username}%'))
    if name: filters.append(User.name.like(f'%{name}%'))
    if phone: filters.append(User.phone.like(f'%{phone}%'))
    if role: filters.append(User.role.like(f'%"{role}"%'))
    if status: filters.append(User.status == status)

    all_users = User.query.filter(*filters).all()
    res = []
    for user in all_users:
        user_dict = user.to_dict()
        corpus_count = Corpus.query.filter(Corpus.contributor == user.id, Corpus.status == CorpusStatus.APPROVE_EXAMINE.value, Corpus.is_deleted == 0).count()
        user_dict['corpus_count'] = corpus_count
        res.append(user_dict)
    # 排序
    res.sort(key=lambda user_dict: user_dict['corpus_count'], reverse=True)
    # 分页
    start = (page_num - 1) * page_size
    end = start + page_size
    page_data = res[start:end]
    return jsonify({
        "code":200,
        "data":{
            "total":len(res),
            "page":(len(res) + page_size - 1) // page_size,
            "data":page_data
        }
    })

# 忘记密码 - 发送重置邮件
@bp.route('/forget_password', methods=['POST'])
def forget_password():
    """向绑定邮箱发送带签名及到期时间的重置链接，不直接修改密码。

    Returns:
        邮件发送结果；无效邮箱、未绑定账号或发送失败时返回错误响应。
    """
    data = request.get_json()
    email = data.get('email', '').strip()

    if not email or len(email) >= 50 or len(email) <= 3 or '@' not in email:
        return jsonify({'message': '错误的邮箱地址!'}), 400

    user = User.query.filter_by(email=email, is_deleted=0).first()
    if not user:
        return jsonify({'message': '该邮箱地址尚未绑定!'}), 400

    timeout = 10
    expires = (datetime.now() + timedelta(minutes=timeout)).strftime("%Y-%m-%d %H:%M")
    token = urllib.parse.quote(md5_encrypt(f"{expires}-{email}"))

    app_url = "https://dicce.ccnu.edu.cn"
    confirm_url = f"{app_url}/newpw?" + urllib.parse.urlencode(
        {
            "Email": email,
            "Token": token,
            "Expire": expires,
            "T": f"0.{datetime.now().strftime('%S%f')}",
        },
        safe=".",
    )

    email_content = f"""
    <html>
    <body style='background-color: rgb(242, 243, 245);'>
        <div style='margin:10% 10% 20% 20%;background-color:white;box-shadow: 1px 1px 1px #ccc;line-height: 200%;'>
            <div style='padding-top: 10px;padding-left: 10px;'>
                <div style='padding-top:5px;padding-left: 30px;min-height: 500px;'>
                    您正在使用密码找回功能，请在收到本邮件后2小时内，点击下面链接重置登录密码，密码设置成功后请妥善保存。<br>
                    用户名：{user.username}<br>
                    链接: <a href='{confirm_url}'>{confirm_url}</a><br>
                    如果以上连链接无法点击，可以复制以上链接在浏览器打开。<br><br><br><br>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    success = send_email(
        to_addresse=email,
        subject="重置密码",
        html_body=email_content
    )

    if success:
        return jsonify({'message': '邮件发送成功'}), 200
    else:
        return jsonify({'message': '邮件发送失败'}), 400

# 解析邮件链接中的过期时间，支持多种格式，返回 (datetime, 规范化字符串) 或 None
def _parse_expire(expire_str):
    """兼容 URL 编码、Unicode 横线及多种时间格式，解析重置链接的到期时间。

    Args:
        expire_str: 前端提交的原始到期时间，可为空。

    Returns:
        日期对象与分钟精度标准文本的二元组，无法解析时返回 None。
    """
    if expire_str is None:
        return None
    s = str(expire_str).strip()
    if not s:
        return None
    s = urllib.parse.unquote(s).replace("+", " ")
    # 统一连字符（避免前端传 unicode 减号等导致 strptime 失败）
    for char in ("\u2212", "\uff0d", "\u2010", "\u2011"):
        s = s.replace(char, "-")
    s = s.strip()
    if not s:
        return None
    formats = ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S")
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt, dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            continue
    return None


# 重置密码（兼容前端传小写或首字母大写的字段名，与链接里 Email/Token/Expire/T 一致）
@bp.route('/reset_password', methods=['POST'])
def reset_password():
    """验证邮箱、时间和签名后更新密码哈希。

    兼容前端参数大小写；现有有效性判断允许标记到期时间后额外三十分钟。
    签名校验使用规范到分钟的时间文本，避免 URL 编码差异导致误判。

    Returns:
        重置结果 JSON，或链接、账号和密码规则校验错误。
    """
    # 1. 兼容链接参数命名及到期格式，验证请求签名与当前有效账号
    data = request.get_json() or {}
    email = (data.get('email') or data.get('Email') or '').strip()
    token = data.get('token') or data.get('Token')
    expire = data.get('expire') or data.get('Expire')
    t = data.get('t') or data.get('T')
    new_password = data.get('new_password')

    if not all([email, token, expire, t]) or '@' not in email:
        return jsonify({'message': '错误的用户信息'}), 400

    parsed = _parse_expire(expire)
    if not parsed:
        return jsonify({'message': '错误的过期时间'}), 400
    expire_time, expire_normalized = parsed

    if expire_time + timedelta(minutes=30) < datetime.now():
        return jsonify({'message': '请求已过期，重新操作！'}), 400

    if not (t and isinstance(t, str) and t.startswith('0.') and len(t) == 10):
        return jsonify({'message': '错误的请求'}), 400

    compare_token = md5_encrypt(f"{expire_normalized}-{email}")
    token_normalized = (token or "").strip() if isinstance(token, str) else ""
    if not token_normalized or token_normalized != compare_token: 
        return jsonify({'message': '验证失败！'}), 400

    user = User.query.filter_by(email=email, is_deleted=0).first()
    if not user:
        return jsonify({'message': '该邮箱地址尚未绑定!'}), 400
    
    if not is_strong_password(new_password):
        return jsonify({"code":400,"error":"密码必须包含大写字母、小写字母、数字和特殊字符，且不少于8位"}),400
    
    # 2. 验证新密码强度后只更新密码哈希，提交成功才返回完成
    user.set_password(new_password)
    db.session.commit()
    return jsonify({'message': '密码修改成功'}), 200

# 超级管理员重置用户密码
@bp.route('/reset_user_password', methods=['POST'])
@permission_required(['ADMIN'])
def admin_reset_password():
    """由管理员将有效账号重置为当前预设密码，提交失败时回滚。"""
    data = request.get_json()
    user_id = data.get('user_id')
    DEFAULT_PASSWORD = 'Abcd123!'
    
    if not user_id:
        return jsonify({'message': '缺少user_id参数'}), 400
    
    try:
        user = User.query.filter_by(id=user_id, is_deleted=0).first()
        if not user:
            return jsonify({'message': '用户不存在'}), 400
            
        user.set_password(DEFAULT_PASSWORD)
        db.session.commit()
        return jsonify({
            'message': '密码已重置为默认密码',
            'default_password': DEFAULT_PASSWORD
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': str(e)}), 400

# MD5加密
def md5_encrypt(text):
    """计算旧版密码找回签名使用的 UTF-8 MD5 摘要，不用于存储登录密码。"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()