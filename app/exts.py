"""声明可由 Flask 应用绑定的数据库和密码哈希扩展，供模型与接口共同复用。"""

from flask_sqlalchemy import SQLAlchemy
import pytz
from flask_bcrypt import Bcrypt

db = SQLAlchemy()
tz = pytz.timezone('Asia/Shanghai')

bcrypt = Bcrypt()
