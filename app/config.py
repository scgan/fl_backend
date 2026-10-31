"""读取 Foreign 业务后端配置，提供数据库连接、JWT 密钥及独立文件服务地址。"""

import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
# load_dotenv()  # 加载.env文件中的变量

SECRET_KEY = os.getenv("FOREIGN_SECRET_KEY")
username = os.getenv("FOREIGN_USERNAME")
password = os.getenv("FOREIGN_PASSWORD")
hostname = os.getenv("FOREIGN_HOST")
port = '3306'
database = os.getenv("FOREIGN_DBNAME")
DB_URI = f"mysql+pymysql://{username}:{password}@{hostname}:{port}/{database}?charset=utf8mb4&collation=utf8mb4_unicode_ci"
SQLALCHEMY_DATABASE_URI = DB_URI
SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_ECHO = False
SQLALCHEMY_ENGINE_OPTIONS = {
    "pool_pre_ping": True,
}
FILE_URL = os.getenv("FOREIGN_FILE_URL")
