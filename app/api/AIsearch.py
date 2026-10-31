"""提供自然语言文献检索、流式回答、检索历史、时间线统计和索引任务管理接口。"""

import os
import json
import time
import tempfile
from uuid import uuid4
from pathlib import Path
from io import BytesIO
from flask import Blueprint, request, jsonify, current_app, g, Response, stream_with_context
from sqlalchemy import or_, text
from app.exts import db
from app.models import Corpus, CorpusCategory, Database, CorpusDatabase, User, CorpusQAHistory
from app.utils.utils import get_file_content
from app.utils.decorators import permission_required, redis_client
from redis.exceptions import RedisError
from app.utils.enums import UserRole
from app.utils.public_sql import public_corpus_sql
from openai import OpenAI
from dotenv import load_dotenv
import re
# 加载环境变量
load_dotenv()

bp = Blueprint('ai_search', __name__)

# 一次检索的 SQL 固定保留十五分钟；结果快照与之同寿命，翻页命中后不再查库。
AI_SEARCH_SQL_TTL = 15 * 60
# 结果快照同样保留十五分钟；只缓存规模可控的结果，避免大结果集长期占用 Redis 内存。
AI_SEARCH_RESULT_TTL = 15 * 60
AI_SEARCH_RESULT_LIMIT = 800

class NL2SQLManager:
    """封装旧业务检索与文献问答的同步模型调用，研究服务另有独立网关。

    Attributes:
        client: 百炼 OpenAI 兼容客户端，支持对话与远程文件操作。
        model_name: 自然语言检索及意图识别的模型名。
        temperature: 自然语言检索回答的采样温度。
        qa_model_name: 旧文献问答使用的文件理解模型。
        qa_temperature: 文献问答的采样温度。
    """
    
    def __init__(self):
        """读取旧业务的检索及问答模型配置，创建实例客户端，不修改研究网关配置。"""
        # 初始化OpenAI客户端（兼容Qwen API）
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY", "sk-2fb3bc7207c0413fb9dc6dacd2844ed8"),
            base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self.model_name = os.getenv("LLM_MODEL_NAME", "qwen3.8-flash")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        # 文献问答专用配置（更适合文本理解和生成任务）
        self.qa_model_name = os.getenv("QA_MODEL_NAME", "qwen-long")  # 使用qwen-long支持长文档和文件上传
        self.qa_temperature = float(os.getenv("QA_TEMPERATURE", "0.3"))  # 较低温度以获得更准确的回答
    
    def get_system_prompt(self):
        """获取MySQL系统提示模板"""
        return """
你是一位优秀的SQL编写专家，能够根据用户的中文问题和提供的表结构，生成准确的MySQL语句。

1. **字段用途说明**（必须按用途正确使用）：
   - title_cn / title_en：标题搜索，可匹配用户指定的“论文名”“文章标题”“研究主题”。
   - keyword_cn / keyword_en：关键词/主题搜索，当用户问“关于...的论文”“...方面的文献”时，必须优先搜索这两个字段。
   - abstract_cn / abstract_en：摘要搜索，用于辅助主题匹配。
   - author：作者搜索，用户提到“某人的文章”时必须使用此字段。
   - language：文献语言，用户指定“用XX语写的”“XX语言”等时使用。
   - publish_time：发布年份，用户指定“年份”“XX年前/后”“在XX年之后”等时使用。

2. **主题搜索策略**（重要）：
   当用户要求检索“关于XXX的文献/论文”时，必须在 title_cn、title_en、keyword_cn、keyword_en、abstract_cn、abstract_en 中**同时**使用 LIKE 进行模糊匹配，不可仅搜索标题。

3. **作者搜索策略**：
   当用户提到“作者是XXX”“XXX的文章”“XXX写的”“XXX的作品”时，必须对 author 字段使用 LIKE 匹配。如果姓名可能出现在作者列表中的任意位置，使用 LIKE。
   如果用户既给出了主题又给出了作者，两种条件都应出现。

4. **语言过滤**：language 字段的值可能是“中文”“英文”“俄语”等中文名称或英文缩写，请根据用户表述生成 LIKE 条件（如 language LIKE '%俄语%' 或 language = 'ru' 视实际数据而定，优先使用 LIKE '%俄语%' 防止值不精确）。

5. **年份过滤**：使用 publish_time < 2015 或 publish_time BETWEEN ... 等，不要使用字符串比较。

6. 查询 corpus 表时：必须查询 corpus 表所有字段，关联 corpus_category 表获取 category_name，并筛选 is_deleted=0 和 status=2 的文献。

7. 只返回 SQL 语句，不要添加任何解释。

8. 确保 SQL 语法正确，合理使用聚合、排序和分组。
"""

    def get_database_metadata(self):
        """构造供 NL2SQL 使用的静态表字段描述，不实际查询数据库元数据或样例记录。"""
        metadata = {}
        
        # 获取所有表名
        tables = ['corpus', 'corpus_category', 'database', 'corpus_database', 'user', 'keywords']
        
        for table in tables:
            # 获取表结构信息
            if table == 'corpus':
                columns = [
                    {"name": "id", "type": "BIGINT", "nullable": False},
                    {"name": "number", "type": "VARCHAR(31)", "nullable": True},
                    {"name": "title_cn", "type": "VARCHAR(1000)", "nullable": True},
                    {"name": "title_en", "type": "VARCHAR(1000)", "nullable": True},
                    {"name": "language", "type": "VARCHAR(31)", "nullable": True},
                    {"name": "author", "type": "VARCHAR(255)", "nullable": True},
                    {"name": "translator", "type": "VARCHAR(255)", "nullable": True},
                    {"name": "editor", "type": "VARCHAR(255)", "nullable": True},
                    {"name": "publish_time", "type": "YEAR", "nullable": True},
                    {"name": "ISSN", "type": "VARCHAR(127)", "nullable": True},
                    {"name": "source", "type": "VARCHAR(510)", "nullable": True},
                    {"name": "category_id", "type": "INT", "nullable": True},
                    {"name": "keyword_cn", "type": "VARCHAR(510)", "nullable": True},
                    {"name": "keyword_en", "type": "VARCHAR(1000)", "nullable": True},
                    {"name": "abstract_cn", "type": "TEXT", "nullable": True},
                    {"name": "abstract_en", "type": "TEXT", "nullable": True},
                    {"name": "original_name", "type": "VARCHAR(255)", "nullable": True},
                    {"name": "original_language", "type": "VARCHAR(31)", "nullable": True},
                    {"name": "original_time", "type": "YEAR", "nullable": True},
                    {"name": "original_source", "type": "VARCHAR(510)", "nullable": True},
                    {"name": "word_number", "type": "INT", "nullable": True},
                    {"name": "attachment", "type": "TEXT", "nullable": True},
                    {"name": "contributor", "type": "BIGINT", "nullable": False},
                    {"name": "status", "type": "INT", "nullable": True},
                    {"name": "comment", "type": "VARCHAR(255)", "nullable": True},
                    {"name": "is_deleted", "type": "BIT", "nullable": True},
                    {"name": "is_vectorized", "type": "BIT", "nullable": True},
                    {"name": "submit_time", "type": "DATETIME", "nullable": True},
                    {"name": "volume", "type": "SMALLINT", "nullable": True},
                    {"name": "issue", "type": "SMALLINT", "nullable": True},
                    {"name": "reviewer_id", "type": "BIGINT", "nullable": True}
                ]
            elif table == 'corpus_category':
                columns = [
                    {"name": "id", "type": "INT", "nullable": False},
                    {"name": "category_name", "type": "VARCHAR(31)", "nullable": True},
                    {"name": "code", "type": "VARCHAR(15)", "nullable": False},
                    {"name": "number", "type": "INT", "nullable": True},
                    {"name": "is_deleted", "type": "BIT", "nullable": True}
                ]
            elif table == 'database':
                columns = [
                    {"name": "id", "type": "INT", "nullable": False},
                    {"name": "database_name", "type": "VARCHAR(127)", "nullable": False},
                    {"name": "parent_id", "type": "INT", "nullable": True},
                    {"name": "is_deleted", "type": "BIT", "nullable": True}
                ]
            elif table == 'corpus_database':
                columns = [
                    {"name": "id", "type": "INT", "nullable": False},
                    {"name": "corpus_id", "type": "BIGINT", "nullable": False},
                    {"name": "database_id", "type": "INT", "nullable": False}
                ]
            elif table == 'user':
                columns = [
                    {"name": "id", "type": "BIGINT", "nullable": False},
                    {"name": "username", "type": "VARCHAR(31)", "nullable": False},
                    {"name": "email", "type": "VARCHAR(31)", "nullable": True},
                    {"name": "password", "type": "VARCHAR(63)", "nullable": False},
                    {"name": "name", "type": "VARCHAR(63)", "nullable": True},
                    {"name": "phone", "type": "VARCHAR(20)", "nullable": True},
                    {"name": "picture", "type": "VARCHAR(63)", "nullable": True},
                    {"name": "role", "type": "VARCHAR(31)", "nullable": False},
                    {"name": "status", "type": "INT", "nullable": False},
                    {"name": "is_deleted", "type": "BIT", "nullable": True},
                    {"name": "leader_id", "type": "BIGINT", "nullable": False}
                ]
            elif table == 'keywords':
                columns = [
                    {"name": "id", "type": "INT", "nullable": False},
                    {"name": "keyword", "type": "VARCHAR(31)", "nullable": False},
                    {"name": "flag", "type": "INT", "nullable": False},
                    {"name": "frequency", "type": "INT", "nullable": True},
                    {"name": "is_deleted", "type": "BIT", "nullable": True},
                    {"name": "language", "type": "INT", "nullable": False}
                ]
            
            metadata[table] = {
                "columns": columns,
                "sample_data": None  # 简化版本，不获取示例数据
            }
        
        return metadata
    
    def format_metadata(self, metadata):
        """格式化元数据为字符串"""
        metadata_str = ""
        
        for table_name, table_info in metadata.items():
            metadata_str += f"表名: {table_name}\n"
            metadata_str += "字段信息:\n"
            
            for column in table_info["columns"]:
                metadata_str += f"  - {column['name']} ({column['type']})"
                if column['nullable']:
                    metadata_str += " (可空)"
                metadata_str += "\n"
            
            metadata_str += "\n"
        
        return metadata_str
    
    def clean_sql(self, sql):
        """清理生成的SQL语句，去除不必要的标记和解释"""
        # 去除可能的代码块标记
        if sql.startswith("```sql"):
            sql = sql[7:]
        if sql.endswith("```"):
            sql = sql[:-3]
        
        # 去除首尾空格和换行符
        sql = sql.strip()
        
        # 去除可能的前缀如"SQL:"或"查询语句:"等
        if sql.lower().startswith("sql:"):
            sql = sql[4:].strip()
        elif sql.lower().startswith("查询语句:"):
            sql = sql[5:].strip()
        
        return sql
    
    def generate_query(self, user_question):
        """生成MySQL查询
        
        Args:
            user_question: 用户输入的问题
            
        Returns:
            str: 生成的MySQL查询语句
        """
        # 获取数据库元数据
        metadata = self.get_database_metadata()
        
        # 格式化元数据
        metadata_str = self.format_metadata(metadata)
        
        # 构建用户提示
        user_prompt = f"""
请根据以下表结构和字段信息，生成一个SQL查询语句，以回答用户的问题。

表结构信息：
{metadata_str}

用户问题：
{user_question}

请只返回SQL语句，不要添加任何其他内容。
"""
        
        # 调用大模型生成SQL
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": self.get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=self.temperature
            )
            
            # 提取生成的SQL语句
            sql = response.choices[0].message.content.strip()
            
            # 清理SQL语句
            sql = self.clean_sql(sql)
            
            return sql
        except Exception as e:
            raise RuntimeError(f"SQL生成失败: {str(e)}")
    
    def _convert_value_for_json(self, value):
        """将值转换为JSON可序列化的格式"""
        if value is None:
            return None
        elif isinstance(value, bytes):
            # 处理bytes类型，BIT字段通常返回b'\x00'或b'\x01'
            if value == b'\x00':
                return False
            elif value == b'\x01':
                return True
            else:
                # 其他bytes类型，尝试解码为字符串或转换为十六进制
                try:
                    return value.decode('utf-8')
                except UnicodeDecodeError:
                    return value.hex()
        elif isinstance(value, (int, float, str, bool)):
            return value
        elif hasattr(value, 'isoformat'):
            # 处理日期时间对象
            return value.isoformat()
        else:
            # 其他类型转换为字符串
            return str(value)
    
    def execute_query(self, sql):
        """执行限定公共文献的单条 SELECT，返回可序列化查询结果。"""
        try:
            # 使用SQLAlchemy执行原始SQL查询，使用text()包装
            result = db.session.execute(text(public_corpus_sql(sql)))
            
            # 获取列名
            columns = result.keys()
            
            # 获取所有结果
            rows = result.fetchall()
            
            # 转换为字典列表，处理不可序列化的数据类型
            result_list = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    # 转换值为JSON可序列化的格式
                    row_dict[col] = self._convert_value_for_json(row[i])
                result_list.append(row_dict)
            
            return result_list
        except Exception as e:
            raise RuntimeError(f"SQL执行失败: {str(e)}")
    
    def execute_query_with_params(self, sql, params):
        """执行带参数且限定公共文献的单条 SELECT，返回可序列化查询结果。"""
        try:
            # 使用SQLAlchemy执行带参数的SQL查询
            result = db.session.execute(text(public_corpus_sql(sql)), params)

            # 获取列名
            columns = result.keys()

            # 获取所有结果
            rows = result.fetchall()

            # 转换为字典列表，处理不可序列化的数据类型
            result_list = []
            for row in rows:
                row_dict = {}
                for i, col in enumerate(columns):
                    # 转换值为JSON可序列化的格式
                    row_dict[col] = self._convert_value_for_json(row[i])
                result_list.append(row_dict)

            return result_list
        except Exception as e:
            raise RuntimeError(f"SQL执行失败: {str(e)}")
    
    def generate_natural_language_response(self, user_question, sql, query_result):
        """根据SQL查询结果生成自然语言回答

        Args:
            user_question: 用户原始问题
            sql: 生成的SQL语句
            query_result: SQL查询结果

        Returns:
            str: 自然语言回答
        """
        # 构建系统提示词
        system_prompt = """
你是一位优秀的数据分析师，能够根据SQL查询结果生成自然、流畅的中文回答。

请遵循以下规则：
1. 根据用户的问题和查询结果，生成简洁明了的自然语言回答
2. 回答应该直接回应用户的问题，不要重复问题本身
3. 如果查询结果为空，如实告知用户没有找到相关数据
4. 如果查询结果包含多条记录，可以总结主要发现或提供关键信息
5. 保持回答的专业性和准确性
6. 不要提及SQL语句或技术细节，专注于用自然语言传达信息
7. 回答应该简洁明了，避免冗长
"""

        # 构建用户提示
        user_prompt = f"""
用户问题：{user_question}

生成的SQL查询：{sql}

查询结果（JSON格式）：
{query_result}

请根据以上信息，生成一个自然语言回答来回应用户的原始问题。
"""

        # 调用大模型生成自然语言回答
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                temperature=self.temperature
            )

            # 提取生成的回答
            answer = response.choices[0].message.content.strip()

            return answer
        except Exception as e:
            raise RuntimeError(f"自然语言回答生成失败: {str(e)}")

    def upload_file_to_qwen(self, file_content, file_path):
        """上传文件到通义千问服务器
        
        Args:
            file_content: 文件二进制内容
            file_path: 文件路径（用于获取文件名和扩展名）
            
        Returns:
            str: 文件ID，如果上传失败返回None
        """
        try:
            # 创建临时文件
            extension = file_path.split('.')[-1].lower() if '.' in file_path else ''
            suffix = f'.{extension}' if extension else ''
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # 上传文件到通义千问
                file_object = self.client.files.create(
                    file=Path(tmp_file_path),
                    purpose="file-extract"
                )
                return file_object.id
            finally:
                # 清理临时文件
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
        except Exception as e:
            current_app.logger.error(f"[QA] upload_file_to_qwen 失败: file_path={file_path}, error={str(e)}")
            return None

    def answer_corpus_question(self, corpus_list, question):
        """把题录与既有远程文件 ID 组合为消息，同步生成文献回答。

        Args:
            corpus_list: 含题录、摘要及可选 file_ids 的文献字典列表。
            question: 本轮用户问题。

        Returns:
            去除首尾空白的首候选回答文本。

        Raises:
            RuntimeError: 文件解析等待耗尽、模型请求失败或响应无法读取。
        """
        # 构建系统提示词
        system_prompt = """
你是一位优秀的文献分析专家，能够根据提供的文献内容回答用户的问题。

请遵循以下规则：
1. 仔细阅读和分析提供的文献内容（包括标题、摘要、关键词和完整的文献正文）
2. 根据用户的问题，从文献中提取相关信息并生成准确、专业的回答
3. 如果问题涉及多篇文献，需要综合分析多篇文献的内容
4. 回答应该：
   - 直接回应用户的问题，不要重复问题本身
   - 基于文献内容，不要编造信息
   - 如果文献中没有相关信息，如实告知
   - 保持专业性和准确性
   - 结构清晰，逻辑严谨
5. 对于常见问题类型：
   - 核心观点：总结文献的主要观点和结论
   - 文章大纲：梳理文献的结构和主要内容框架
   - 文献要点：提取文献的关键信息点
   - 其他问题：根据问题类型灵活回答
6. 如果涉及多篇文献，可以对比分析或综合总结
7. 使用与用户问题相同的语言回答（用户用中文问就中文答；用户用英文问就英文答），语言流畅自然
"""

        # 构建消息列表
        # 1. 保留既有问答模板，将各篇题录与已上传文件 ID 映射成模型消息
        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        
        # 添加文件引用和文献元数据信息
        corpus_info_text = ""
        for idx, corpus in enumerate(corpus_list, 1):
            corpus_info_text += f"\n【文献 {idx}】\n"
            corpus_info_text += f"文献ID: {corpus.get('id', 'N/A')}\n"
            if corpus.get('title_cn'):
                corpus_info_text += f"中文标题: {corpus.get('title_cn')}\n"
            if corpus.get('title_en'):
                corpus_info_text += f"英文标题: {corpus.get('title_en')}\n"
            if corpus.get('author'):
                corpus_info_text += f"作者: {corpus.get('author')}\n"
            if corpus.get('publish_time'):
                corpus_info_text += f"发表时间: {corpus.get('publish_time')}\n"
            if corpus.get('keyword_cn'):
                keywords = corpus.get('keyword_cn')
                if isinstance(keywords, list):
                    corpus_info_text += f"中文关键词: {', '.join(keywords)}\n"
                elif isinstance(keywords, str):
                    corpus_info_text += f"中文关键词: {keywords}\n"
            if corpus.get('keyword_en'):
                keywords = corpus.get('keyword_en')
                if isinstance(keywords, list):
                    corpus_info_text += f"英文关键词: {', '.join(keywords)}\n"
                elif isinstance(keywords, str):
                    corpus_info_text += f"英文关键词: {keywords}\n"
            if corpus.get('abstract_cn'):
                corpus_info_text += f"中文摘要: {corpus.get('abstract_cn')}\n"
            if corpus.get('abstract_en'):
                corpus_info_text += f"英文摘要: {corpus.get('abstract_en')}\n"
            
            # 添加文件引用
            file_ids = corpus.get('file_ids', [])
            if file_ids:
                for file_id in file_ids:
                    if file_id: 
                        messages.append({
                            "role": "system",
                            "content": f"fileid://{file_id}"
                        })
            corpus_info_text += "\n"

        # 构建用户提示
        user_prompt = f"""
用户问题：{question}

文献基本信息：
{corpus_info_text}

请根据以上文献的完整内容（包括上传的文件内容），回答用户的问题。如果文献中没有相关信息，请如实告知。
"""

        messages.append({
            "role": "user",
            "content": user_prompt
        })

        # 调用大模型生成回答（使用文件上传方式）
        try:
            # 等待文件解析完成（如果需要）
            # 2. 同步请求问答模型，仅对文件仍在解析的状态限次等待，其他异常交给外层转换
            max_wait_time = 120  # 最大等待时间（秒）
            wait_interval = 2  # 每次等待间隔（秒）
            max_retries = max_wait_time // wait_interval
            
            response = None
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.qa_model_name,
                        messages=messages,
                        temperature=self.qa_temperature,
                        max_tokens=8192
                    )
                    break  # 成功则跳出循环
                except Exception as api_error:
                    error_str = str(api_error)
                    # 检查是否是文件解析中的错误
                    if "File parsing in progress" in error_str or "parsing in progress" in error_str.lower():
                        if attempt < max_retries - 1:
                            time.sleep(wait_interval)
                            continue
                    # 其他错误直接抛出
                    raise

            # 3. 拒绝空响应并提取回答；错误统一转成旧业务 RuntimeError 契约
            if response is None:
                raise RuntimeError("文件解析超时或API调用失败")

            # 提取生成的回答
            answer = response.choices[0].message.content.strip()

            return answer
        except Exception as e:
            raise RuntimeError(f"文献问答生成失败: {str(e)}")

    def answer_corpus_question_stream(self, corpus_list, question):
        """按题录和已有远程文件引用构造输入，再转发问答模型流。

        Args:
            corpus_list: 题录、摘要及可选 file_ids 列表，此方法本身不上传原件。
            question: 用户本轮问题。

        Yields:
            模型回答的增量文本片段。
        """
        system_prompt = """
你是一位优秀的文献分析专家，能够根据提供的文献内容回答用户的问题。

请遵循以下规则：
1. 仔细阅读和分析提供的文献内容（包括标题、摘要、关键词和完整的文献正文）
2. 根据用户的问题，从文献中提取相关信息并生成准确、专业的回答
3. 如果问题涉及多篇文献，需要综合分析多篇文献的内容
4. 回答应该：
   - 直接回应用户的问题，不要重复问题本身
   - 基于文献内容，不要编造信息
   - 如果文献中没有相关信息，如实告知
   - 保持专业性和准确性
   - 结构清晰，逻辑严谨
5. 对于常见问题类型：
   - 核心观点：总结文献的主要观点和结论
   - 文章大纲：梳理文献的结构和主要内容框架
   - 文献要点：提取文献的关键信息点
   - 其他问题：根据问题类型灵活回答
6. 如果涉及多篇文献，可以对比分析或综合总结
7. 使用与用户问题相同的语言回答（用户用中文问就中文答；用户用英文问就英文答），语言流畅自然
"""

        messages = [{"role": "system", "content": system_prompt}]

        corpus_info_text = ""
        for idx, corpus in enumerate(corpus_list, 1):
            corpus_info_text += f"\n【文献 {idx}】\n"
            corpus_info_text += f"文献ID: {corpus.get('id', 'N/A')}\n"
            if corpus.get('title_cn'):
                corpus_info_text += f"中文标题: {corpus.get('title_cn')}\n"
            if corpus.get('title_en'):
                corpus_info_text += f"英文标题: {corpus.get('title_en')}\n"
            if corpus.get('author'):
                corpus_info_text += f"作者: {corpus.get('author')}\n"
            if corpus.get('publish_time'):
                corpus_info_text += f"发表时间: {corpus.get('publish_time')}\n"
            if corpus.get('keyword_cn'):
                keywords = corpus.get('keyword_cn')
                if isinstance(keywords, list):
                    corpus_info_text += f"中文关键词: {', '.join(keywords)}\n"
                elif isinstance(keywords, str):
                    corpus_info_text += f"中文关键词: {keywords}\n"
            if corpus.get('keyword_en'):
                keywords = corpus.get('keyword_en')
                if isinstance(keywords, list):
                    corpus_info_text += f"英文关键词: {', '.join(keywords)}\n"
                elif isinstance(keywords, str):
                    corpus_info_text += f"英文关键词: {keywords}\n"
            if corpus.get('abstract_cn'):
                corpus_info_text += f"中文摘要: {corpus.get('abstract_cn')}\n"
            if corpus.get('abstract_en'):
                corpus_info_text += f"英文摘要: {corpus.get('abstract_en')}\n"

            file_ids = corpus.get('file_ids', [])
            if file_ids:
                for file_id in file_ids:
                    messages.append({
                        "role": "system",
                        "content": f"fileid://{file_id}"
                    })
            corpus_info_text += "\n"

        user_prompt = f"""
用户问题：{question}

文献基本信息：
{corpus_info_text}

请根据以上文献的完整内容（包括上传的文件内容），回答用户的问题。如果文献中没有相关信息，请如实告知。
"""
        messages.append({"role": "user", "content": user_prompt})

        yield from self._stream_answer(messages)

    def build_qa_messages(self, corpus_list, question, history_messages=None, rag_context=None):
        """组合历史与当前来源，RAG 有片段时优先使用片段，不追加远程文件输入。

        Args:
            corpus_list: 题录及可选 file_ids 列表，用于非 RAG 模式。
            question: 当前用户问题。
            history_messages: 既有 user/assistant 消息列表，仅用于对话上下文。
            rag_context: 可选召回片段，正文按现有每项八百字符预算截取。

        Returns:
            按现有模板组装的模型消息列表，不在此调用模型。

        Raises:
            ValueError: 历史消息含 user、assistant 之外的角色。
        """
        # 1. 固定来源模式及已有模板，历史只保留允许的对话角色
        is_rag_mode = bool(rag_context)

        system_prompt = (
            "你是一个学术助手，必须严格根据提供的参考内容回答用户问题。\n\n"
            "重要规则：\n"
            "1. 回答必须完全基于参考内容，不得编造或添加参考内容中没有的信息。\n"
            "2. 如果参考内容不足以回答用户问题，请如实告知，不要自行补充。\n"
            "3. 引用参考内容时，使用编号如 [1] [2] 标注来源。\n"
            "4. 保持回答的专业性和准确性，语言流畅自然。"
        )

        messages = [{"role": "system", "content": system_prompt}]

        # 追加历史对话（多轮对话支持）
        if history_messages and isinstance(history_messages, list):
            for msg in history_messages:
                if isinstance(msg, dict) and msg.get("role") and msg.get("content"):
                    role = msg["role"].lower()  # 确保角色名称为小写
                    if role not in ["user", "assistant"]:
                        raise ValueError(f"无效的角色名称: {role}")
                    messages.append({"role": role, "content": msg["content"]})

        # 追加文献内容
        corpus_info_text = ""
        for idx, corpus in enumerate(corpus_list, 1):
            corpus_info_text += f"\n【文献 {idx}】\n"
            corpus_info_text += f"文献ID: {corpus.get('id', 'N/A')}\n"
            if corpus.get('title_cn'):
                corpus_info_text += f"中文标题: {corpus.get('title_cn')}\n"
            if corpus.get('title_en'):
                corpus_info_text += f"英文标题: {corpus.get('title_en')}\n"
            if corpus.get('author'):
                corpus_info_text += f"作者: {corpus.get('author')}\n"
            if corpus.get('publish_time'):
                corpus_info_text += f"发表时间: {corpus.get('publish_time')}\n"
            if corpus.get('keyword_cn'):
                keywords = corpus.get('keyword_cn')
                if isinstance(keywords, list):
                    corpus_info_text += f"中文关键词: {', '.join(keywords)}\n"
                elif isinstance(keywords, str):
                    corpus_info_text += f"中文关键词: {keywords}\n"
            if corpus.get('keyword_en'):
                keywords = corpus.get('keyword_en')
                if isinstance(keywords, list):
                    corpus_info_text += f"英文关键词: {', '.join(keywords)}\n"
                elif isinstance(keywords, str):
                    corpus_info_text += f"英文关键词: {keywords}\n"
            if corpus.get('abstract_cn'):
                corpus_info_text += f"中文摘要: {corpus.get('abstract_cn')}\n"
            if corpus.get('abstract_en'):
                corpus_info_text += f"英文摘要: {corpus.get('abstract_en')}\n"

            file_ids = corpus.get('file_ids', [])
            # 如果已有 RAG 原文片段，不再上传文件（file_ids 已被 rag_context 替代）
            if file_ids and not rag_context:
                for file_id in file_ids:
                    messages.append({
                        "role": "system",
                        "content": f"fileid://{file_id}"
                    })
            corpus_info_text += "\n"

        # ── RAG 溯源模式（优先） ───────────────────────────────
        # 2. 根据实际来源选择片段上下文或题录和远程文件，不把历史当作原文引用
        if rag_context:
            context_parts = []
            for i, chunk in enumerate(rag_context, 1):
                text = (chunk.get('text') or '').strip()[:800]
                title = chunk.get('title_cn') or chunk.get('title_en') or chunk.get('file_name') or f'文献{i}'
                context_parts.append(f"[{i}] {title}：{text}")

            context_text = '\n\n'.join(context_parts)

            user_prompt = (
                f"用户问题：{question}\n\n"
                f"参考内容：\n{context_text}\n\n"
                f"请根据上述参考内容回答用户问题，可引用编号如 [1] [2]。"
            )
        else:
            # 非 RAG 模式：基于摘要或上传全文回答
            corpus_info_text = ""
            for idx, corpus in enumerate(corpus_list, 1):
                corpus_info_text += f"\n【文献 {idx}】\n"
                corpus_info_text += f"文献ID: {corpus.get('id', 'N/A')}\n"
                if corpus.get('title_cn'):
                    corpus_info_text += f"中文标题: {corpus.get('title_cn')}\n"
                if corpus.get('title_en'):
                    corpus_info_text += f"英文标题: {corpus.get('title_en')}\n"
                if corpus.get('author'):
                    corpus_info_text += f"作者: {corpus.get('author')}\n"
                if corpus.get('publish_time'):
                    corpus_info_text += f"发表时间: {corpus.get('publish_time')}\n"
                if corpus.get('keyword_cn'):
                    keywords = corpus.get('keyword_cn')
                    if isinstance(keywords, list):
                        corpus_info_text += f"中文关键词: {', '.join(keywords)}\n"
                    elif isinstance(keywords, str):
                        corpus_info_text += f"中文关键词: {keywords}\n"
                if corpus.get('keyword_en'):
                    keywords = corpus.get('keyword_en')
                    if isinstance(keywords, list):
                        corpus_info_text += f"英文关键词: {', '.join(keywords)}\n"
                    elif isinstance(keywords, str):
                        corpus_info_text += f"英文关键词: {keywords}\n"
                if corpus.get('abstract_cn'):
                    corpus_info_text += f"中文摘要: {corpus.get('abstract_cn')}\n"
                if corpus.get('abstract_en'):
                    corpus_info_text += f"英文摘要: {corpus.get('abstract_en')}\n"

                file_ids = corpus.get('file_ids', [])
                if file_ids and not rag_context:
                    for file_id in file_ids:
                        messages.append({
                            "role": "system",
                            "content": f"fileid://{file_id}"
                        })
                corpus_info_text += "\n"

            user_prompt = f"""
用户问题：{question}

文献基本信息：
{corpus_info_text}

请根据以上文献的内容（包括上传的文件内容），回答用户的问题。如果文献中没有相关信息，请如实告知。
"""
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _stream_answer(self, messages):
        """内部方法：携带 messages 发起流式请求并 yield 内容块"""
        try:
            max_wait_time = 120
            wait_interval = 2
            max_retries = max_wait_time // wait_interval

            for attempt in range(max_retries):
                try:
                    stream = self.client.chat.completions.create(
                        model=self.qa_model_name,
                        messages=messages,
                        temperature=self.qa_temperature,
                        max_tokens=8192,
                        stream=True,
                        stream_options={"include_usage": True}
                    )
                    break
                except Exception as api_error:
                    error_str = str(api_error)
                    if "File parsing in progress" in error_str or "parsing in progress" in error_str.lower():
                        if attempt < max_retries - 1:
                            time.sleep(wait_interval)
                            continue
                    raise

            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                if chunk.usage:
                    current_app.logger.info(
                        f"[QA Stream] chunk_count={chunk_count} usage - "
                        f"prompt:{chunk.usage.prompt_tokens} "
                        f"completion:{chunk.usage.completion_tokens} "
                        f"total:{chunk.usage.total_tokens}"
                    )
                elif chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    current_app.logger.debug(
                        f"[QA Stream] chunk={chunk_count} content_preview={repr(content[:30])}"
                    )
                    yield content

        except Exception as e:
            current_app.logger.error(f"[_stream_answer] LLM 调用异常: {e}")
            yield f"[错误] 文献问答生成失败: {str(e)}"

    def answer_from_messages(self, corpus_list, question, history_messages=None, rag_context=None):
        """对外方法：传入 corpus_list、当前问题、历史消息、RAG 检索片段，直接流式返回"""
        messages = self.build_qa_messages(corpus_list, question, history_messages, rag_context)
        yield from self._stream_answer(messages)

    def detect_intent(self, question, global_search=False):
        """根据问题内容识别用户意图

        Args:
            question: 用户问题
            global_search: 前端按钮强制全局搜索标志，为True时直接返回search

        Returns:
            str: intent type — 'search' | 'timeline' | 'general'
        """
        # 前端按钮强制触发全库搜索
        if global_search:
            return 'search'

        # 全局搜库意图
        search_keywords = [
            '有哪些论文', '有哪些文献', '有哪些文章'
        ]

        # 可视化分析意图
        timeline_keywords = [
            '趋势', '分布', '按年', '按时间', '图', '可视化'
        ]

        q = question.lower()

        is_search = any(kw in q for kw in search_keywords)
        is_timeline = any(kw in q for kw in timeline_keywords)

        if is_search:
            return 'search'
        if is_timeline:
            return 'timeline'
        return 'general'

    def query_papers_by_nl(self, question, page_num=1, page_size=10):
        """自然语言查询论文列表"""
        sql = self.generate_query(question)
        full_result = self.execute_query(sql)
        if not full_result:
            return {'result': [], 'total': 0, 'sql': sql}

        seen_ids = set()
        unique_result = []
        for item in full_result:
            item_id = item.get('id')
            if item_id is not None and item_id not in seen_ids:
                seen_ids.add(item_id)
                unique_result.append(item)
            elif item_id is None:
                unique_result = full_result
                break

        total = len(unique_result)
        start = (page_num - 1) * page_size
        end = start + page_size
        return {
            'result': unique_result[start:end],
            'total': total,
            'total_pages': (total + page_size - 1) // page_size,
            'sql': sql
        }

    def query_timeline_by_ids(self, corpus_ids):
        """根据文献ID列表生成年度统计"""
        from app.models import Corpus
        from sqlalchemy import and_

        valid_ids = [int(_id) for _id in corpus_ids if str(_id).isdigit()]
        if not valid_ids:
            return None

        corpora = db.session.query(
            Corpus.publish_time
        ).filter(
            and_(
                Corpus.id.in_(valid_ids),
                Corpus.is_deleted == 0,
                Corpus.status == 2
            )
        ).all()

        if not corpora:
            return None

        count_dict = {}
        for (publish_time,) in corpora:
            if publish_time:
                try:
                    year = int(publish_time)
                    count_dict[year] = count_dict.get(year, 0) + 1
                except (ValueError, TypeError):
                    pass

        chart_data = [
            {"year": year, "count": count, "label": str(year)}
            for year, count in sorted(count_dict.items())
        ]
        return {
            'total_count': len(corpora),
            'chart_data': chart_data,
            'analysis': {
                'data_points': len(chart_data),
                'max_value': max([d['count'] for d in chart_data]) if chart_data else 0,
                'total_items': len(corpora),
            }
        }

    def truncate_messages(self, messages, max_pairs=20):
        """截断对话历史，保留最近 N 轮（每轮=user+assistant）

        Args:
            messages: 原始消息列表（含 system 消息）
            max_pairs: 保留的最大对话轮数，默认为20

        Returns:
            list: 截断后的消息列表
        """
        if not messages:
            return []

        # 分离 system 消息和对话消息
        system_msgs = [m for m in messages if m.get('role') == 'system']
        dialog_msgs = [m for m in messages if m.get('role') != 'system']

        # 保留最近 max_pairs 轮对话（每轮 = user + assistant）
        # dialog_msgs 结构：user / assistant / user / assistant / ...
        keep_count = max_pairs * 2
        if len(dialog_msgs) > keep_count:
            dialog_msgs = dialog_msgs[-keep_count:]

        return system_msgs + dialog_msgs
    
    def detect_intent_with_llm(self, question, corpus_list=None):
        """使用大模型判断用户意图
        
        Returns:
            dict: {
                'intent': 'qa' | 'search' | 'timeline',
                'confidence': 0.0-1.0,
                'keywords': []  # 提取的关键词，用于search或timeline
            }
        """
        # 构建上下文
        corpus_context = ""
        if corpus_list:
            corpus_context = "当前对话涉及的文献：\n"
            for idx, corpus in enumerate(corpus_list[:3], 1):
                corpus_context += f"{idx}. {corpus.get('title_cn', '')} - {corpus.get('author', '')}\n"
        
        system_prompt = """你是一个意图识别专家。根据用户问题和当前上下文，判断用户的真实意图。

    意图类型：
    1. qa: 针对当前文献内容的提问（如"这篇文章讲了什么"、"作者的观点是什么"、"研究方法是什么"）
    2. search: 需要检索其他文献（如"还有其他相关论文吗"、"马瑞还写过什么"、"有没有关于XX的文章"）
    3. timeline: 需要查看趋势分析（如"这个领域的研究趋势"、"近几年的发展情况"、"按年份分布"）

    判断标准：
    - 如果问题明确指向当前文献的具体内容，返回 qa
    - 如果问题要求查找其他文献或询问某个主题/作者的其他文章，返回 search
    - 如果问题涉及时间趋势、统计分析、年份分布，返回 timeline 

    返回JSON格式：
    {
        "intent": "qa|search|timeline",
        "confidence": 0.0-1.0,
        "keywords": ["提取的关键词"]
    }"""

        user_prompt = f"""
    当前上下文：
    {corpus_context}

    用户问题：{question}

    请判断用户意图并返回JSON。
    """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            intent_result = json.loads(response.choices[0].message.content)
            return intent_result
            
        except Exception as e:
            current_app.logger.error(f"[意图识别] LLM判断失败: {str(e)}")
            # 降级到关键词匹配
            return self._fallback_intent_detection(question)

    def _fallback_intent_detection(self, question):
        """降级方案：关键词匹配"""
        q = question.lower()
        
        # 可视化分析意图
        timeline_keywords = ['趋势', '分布', '按年', '按时间', '图', '可视化']
        if any(kw in q for kw in timeline_keywords):
            return {'intent': 'timeline', 'confidence': 0.7, 'keywords': [question]}
        
        # 全局搜库意图
        search_keywords = ['还有', '其他', '更多']
        if any(kw in q for kw in search_keywords):
            return {'intent': 'search', 'confidence': 0.7, 'keywords': [question]}
        
        # 默认qa
        return {'intent': 'qa', 'confidence': 0.8, 'keywords': []}

    def generate_query_with_context(self, user_question, history_messages=None, last_result=None):
        """生成带有对话上下文的SQL查询（已包含附件字段指导）"""
        metadata = self.get_database_metadata()
        metadata_str = self.format_metadata(metadata)

        # 构建上下文文本
        context_text = ""
        if history_messages:
            context_text += "## 历史对话\n"
            for msg in history_messages[-6:]:
                role = "用户" if msg['role'] == 'user' else "助手"
                context_text += f"{role}: {msg['content']}\n"
            context_text += "\n"

        if last_result:
            context_text += "## 上一次检索返回的文献列表\n"
            context_text += "| 序号 | 文献ID | 标题 | 作者 | 出版年份 |\n"
            context_text += "|------|--------|------|------|----------|\n"
            for idx, item in enumerate(last_result, 1):
                title = item.get('title_cn') or item.get('title_en') or '未知'
                author = item.get('author') or '未知'
                year = item.get('publish_time') or '未知'
                context_text += f"| {idx} | {item.get('id')} | {title} | {author} | {year} |\n"
            context_text += "\n"

        user_prompt = f"""
    请根据以下信息，生成一条MySQL查询语句来回答用户当前问题。

    数据库表结构：
    {metadata_str}

    {context_text if context_text else ""}

    用户当前问题：{user_question}

    重要规则：
    1. 如果问题中存在“第一篇”、“第二篇”、“这篇文章”、“该文献”等指代，请根据“上一次检索返回的文献列表”中的序号和ID，直接使用对应的文献ID进行查询。
    2. 如果用户要求分析、总结、提炼某篇文献的核心观点、大纲、方法论等内容，必须查询该文献的 **所有字段**（特别是 attachment 字段），以便后续深度分析。
    3. 如果问题是全新的检索需求，忽略历史结果，构造全新的查询。
    4. 只返回SQL语句，不要添加任何解释。
    """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature
            )
            sql = response.choices[0].message.content.strip()
            return self.clean_sql(sql)
        except Exception as e:
            raise RuntimeError(f"SQL生成失败: {str(e)}")

    def is_deep_analysis_intent(self, question, result_count):
        """判断用户问题是否属于需要原文深度分析的意图"""
        analysis_keywords = [
            '核心观点', '主要观点', '文章大纲', '文献要点', '总结', '提炼',
            '方法论', '研究方法', '创新点', '结论', '内容分析', '讲了什么',
            '梳理', '概括', '归纳', '评价', '对比分析', '异同点'
        ]
        q = question.lower()
        # 如果结果中只有一篇文章且出现分析关键词，则倾向深度分析
        if result_count <= 3 and any(kw in q for kw in analysis_keywords):
            return True
        return False

# 创建NL2SQL管理器实例
nl2sql_manager = NL2SQLManager()

def _dedupe_ai_search_rows(rows):
    """按 id 去重，保持原顺序；结果没有 id 字段时原样返回（沿用旧行为）。"""
    if not rows:
        return rows
    seen_ids = set()
    unique_rows = []
    for item in rows:
        item_id = item.get('id')
        if item_id is None:
            # 没有 id 字段时不做去重，直接使用原始结果。
            return rows
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        unique_rows.append(item)
    return unique_rows


@bp.route('/nl-search', methods=['POST'])
def nl_search():
    """首次生成 SQL 并缓存，后续按当前用户的 search_id 复用 SQL 与结果快照返回分页。

    请求可以携带首次检索的 question，或后续翻页使用的 search_id。
    翻页命中结果快照时不再查库，也不重复去重；快照缺失或过期时按缓存的 SQL 回源查询。

    Returns:
        含 search_id 的文献分页响应；参数、登录、缓存过期及不可用时返回明确错误。
    """
    try:
        # 1. 校验登录、分页和检索参数，缓存标识不能替代当前账号的认证。
        data = request.get_json()
        if not isinstance(data, dict):
            return jsonify(code=400, error="无效的检索参数"), 400
        user = getattr(g, 'current_user', None)
        if user is None:
            return jsonify(code=401, error="未登录"), 401
        try:
            page_num = int(data.get('page_num', 1))
            page_size = int(data.get('page_size', 10))
        except (TypeError, ValueError):
            return jsonify(code=400, error="无效的分页参数"), 400
        if page_num < 1 or page_size < 1:
            return jsonify(code=400, error="无效的分页参数"), 400

        # 2. 首次调用模型，后续读取本人缓存；缺失或过期不自动重新调用模型。
        search_id = data.get('search_id')
        is_new_search = search_id is None
        if is_new_search:
            question = data.get('question')
            if not isinstance(question, str) or not question.strip():
                return jsonify(code=400, error="问题不能为空"), 400
            sql = nl2sql_manager.generate_query(question.strip())
            search_id = uuid4().hex
        else:
            if not isinstance(search_id, str) or not re.fullmatch(r'[0-9a-f]{32}', search_id):
                return jsonify(code=400, error="无效的搜索标识"), 400
            sql = redis_client.get(f"ai_search:sql:{user.id}:{search_id}")
            if sql is None:
                return jsonify(code=410, error="搜索已过期，请重新搜索"), 410

        sql_key = f"ai_search:sql:{user.id}:{search_id}"
        result_key = f"ai_search:result:{user.id}:{search_id}"

        # 3. 优先读取结果快照：同一 search_id 翻页不再查库、不再全量去重；缺失时才回源执行 SQL。
        full_result = None
        if not is_new_search:
            cached_result = redis_client.get(result_key)
            if cached_result is not None:
                try:
                    full_result = json.loads(cached_result)
                except (TypeError, ValueError):
                    current_app.logger.warning("AI 检索结果快照解析失败，改为回源查询")
                    full_result = None

        if full_result is None:
            # execute_query 每次仍应用 public_corpus_sql 的校验和公共可见性约束。
            full_result = _dedupe_ai_search_rows(nl2sql_manager.execute_query(sql))
            # 只缓存规模可控的结果快照，避免大结果集长期占用 Redis 内存。
            if len(full_result) <= AI_SEARCH_RESULT_LIMIT:
                redis_client.setex(
                    result_key,
                    AI_SEARCH_RESULT_TTL,
                    json.dumps(full_result, ensure_ascii=False, default=str),
                )

        if is_new_search:
            redis_client.setex(sql_key, AI_SEARCH_SQL_TTL, sql)

        # 4. 按既有结构分页返回，保持关联查询及旧客户端的结果结构兼容。
        total = len(full_result)

        # 进行分页
        start_index = (page_num - 1) * page_size
        end_index = start_index + page_size
        paginated_result = full_result[start_index:end_index]

        # 计算总页数
        total_pages = (total + page_size - 1) // page_size

        # 生成自然语言回答（已注释）
        # natural_language_answer = nl2sql_manager.generate_natural_language_response(question, sql, result)

        # 返回结果
        return jsonify({
            "code": 200,
            "data": {
                "search_id": search_id,
                "sql": sql,
                "result": paginated_result,
                "pagination": {
                    "page_num": page_num,
                    "page_size": page_size,
                    "total": total,
                    "total_pages": total_pages
                }
                # "natural_language_answer": natural_language_answer  # 已注释
            }
        })

    except RedisError:
        current_app.logger.warning("AI 检索 SQL 缓存不可用")
        return jsonify(code=503, error="搜索缓存暂不可用，请稍后重试"), 503
    except Exception as e:
        return jsonify({
            "code": 500,
            "error": f"查询失败: {str(e)}"
        }), 500

def chinese_to_int(chinese_str):
    """中文数字转整数，支持 一 ~ 九十九"""
    if not chinese_str:
        return None

    num_map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4,
        '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10
    }

    # 单字数字
    if chinese_str in num_map:
        return num_map[chinese_str]

    # 十开头：十一、十二...十九
    if chinese_str.startswith('十'):
        if len(chinese_str) == 1:
            return 10
        last = chinese_str[1]
        if last in num_map:
            return 10 + num_map[last]

    # 十在中间：二十一、三十五...九十九
    if '十' in chinese_str:
        parts = chinese_str.split('十')
        left = parts[0]
        right = parts[1] if len(parts) > 1 else ''

        left_val = num_map.get(left, 0) if left else 1
        right_val = num_map.get(right, 0) if right else 0

        return left_val * 10 + right_val

    return None


def fuzzy_match_title(query_title, corpus):
    """模糊匹配标题：query_title 与 corpus 的 title_cn / title_en 逐字匹配"""
    if not query_title:
        return False
    q = query_title.lower().strip()
    titles = [
        (corpus.get('title_cn') or '').lower(),
        (corpus.get('title_en') or '').lower()
    ]
    # 精确包含
    if any(q in t for t in titles):
        return True
    # 关键词交叉：至少有一半字符能对上
    matched = sum(1 for ch in q if any(ch in t for t in titles))
    return matched >= len(q) * 0.6


def parse_sequence_reference(question, result_list):
    """解析问题中的“第X篇”指代，返回对应的文献字典"""
    if not result_list:
        return None

    # 阿拉伯数字：第123篇
    match = re.search(r'第(\d+)(?:篇|个|条|章|节)', question)
    if match:
        idx = int(match.group(1)) - 1
        if 0 <= idx < len(result_list):
            return result_list[idx]

    # 中文数字：第五篇、第二十一篇
    match_cn = re.search(r'第([零一二三四五六七八九十百千万]+)(?:篇|个|条|章|节)', question)
    if match_cn:
        num_str = match_cn.group(1)
        num = chinese_to_int(num_str)
        if num is not None and 0 <= num - 1 < len(result_list):
            return result_list[num - 1]

    # 默认指代（已废弃——不再默认返回第1篇）
    if re.search(r'这[篇个]|该文献|该文章|这篇论文', question) and result_list:
        return None  # 不再默认，返回 None 让上层决定

    return None


def resolve_reference(question, ordered_list, selected_corpus_id=None, last_mentioned_corpus_id=None):
    """可解释的指代解析器

    Args:
        question: 用户当前问题
        ordered_list: 最近一次搜索的完整结果（按UI顺序，跨页连续 1..N）
        selected_corpus_id: 前端传入的用户当前点选的文献ID
        last_mentioned_corpus_id: 后端记录的最近一次被成功指代的文献ID

    Returns:
        dict: {
            status: 'ok' | 'need_clarify' | 'not_found' | 'no_reference',
            corpus_id: int (仅 status=='ok' 时有),
            reason: str,
            candidates: list (仅 status=='need_clarify' 时有)
        }
    """
    # 1. 优先识别明确文献 ID，且必须在当前显示顺序快照内命中
    q = question.strip()

    # ── 1. 明确 ID ──────────────────────────────────────
    # id=123 / 文献123 / #123
    for pat in [r'id[=：:]\s*(\d+)', r'文献\s*(\d+)', r'#(\d+)']:
        m = re.search(pat, q)
        if m:
            target_id = int(m.group(1))
            for item in ordered_list:
                if item.get('id') == target_id:
                    return {'status': 'ok', 'corpus_id': target_id,
                            'reason': f'明确ID引用(id={target_id})'}
            # ID 明确但不在列表中
            return {'status': 'not_found', 'corpus_id': None,
                    'reason': f'指定的文献ID={target_id}不在当前检索结果中'}

    # ── 2. 明确序号 ──────────────────────────────────────
    # 第2篇 / 第二篇 / 2) / 2. / No.2
    # 2. 解析中文或数字序号，以当前 UI 结果顺序转换到真实文献 ID
    seq_match = re.search(
        r'(?:第\s*([零一二三四五六七八九十百千万\d]+)\s*(?:篇|个|条|章|节)|'
        r'(?:^|[^\d])(\d+)\s*[)\.。、])|'
        r'No\.?\s*(\d+)',
        q
    )
    if seq_match:
        num = None
        # 阿拉伯数字捕获组
        for g in seq_match.groups():
            if g and g.strip().isdigit():
                num = int(g.strip())
                break
            if g:
                cn = chinese_to_int(g.strip())
                if cn is not None:
                    num = int(cn)
                    break
        if num is not None:
            idx = num - 1  # 转为 0-index
            if 0 <= idx < len(ordered_list):
                target = ordered_list[idx]
                return {'status': 'ok', 'corpus_id': target['id'],
                        'reason': f'序号引用(第{num}篇 → id={target["id"]})'}
            elif idx >= len(ordered_list) and len(ordered_list) > 0:
                return {'status': 'need_clarify', 'corpus_id': None,
                        'reason': f'序号{num}超出当前检索结果范围(共{len(ordered_list)}篇)',
                        'candidates': ordered_list[:10]}
            else:
                return {'status': 'not_found', 'corpus_id': None,
                        'reason': '序号解析异常'}

    # ── 3. 标题片段匹配 ─────────────────────────────────
    # 《...》或"..."
    # 3. 再匹配题名；只有缺少明确引用时才尝试当前预览或最近指代，无法确定则请求澄清
    title_match = re.search(r'《([^》]+)》', q)
    if not title_match:
        title_match = re.search(r'"([^"]+)"', q)
    if title_match:
        query_title = title_match.group(1).strip()
        for item in ordered_list:
            if fuzzy_match_title(query_title, item):
                return {'status': 'ok', 'corpus_id': item['id'],
                        'reason': f'标题匹配(《{query_title}》 → id={item["id"]})'}
        return {'status': 'not_found', 'corpus_id': None,
                'reason': f'标题《{query_title}》未在当前结果中匹配到'}

    # ── 4. “这篇/该文献/上文那篇”等模糊指代 ─────────────
    if re.search(r'这[篇个条]|该文献|该文章|这篇|那篇|上面|上面那篇', q):
        # 优先级：前端传入的 selected_corpus_id > 后端记录的 last_mentioned_corpus_id
        for cid, label in [(selected_corpus_id, '前端传入的selected_corpus_id'),
                           (last_mentioned_corpus_id, '最近一次指代')]:
            if cid:
                for item in ordered_list:
                    if item.get('id') == cid:
                        return {'status': 'ok', 'corpus_id': int(cid),
                                'reason': f'“这篇”解析({label}) → id={cid}'}
                # selected/mentioned ID 不在当前列表：需要澄清
                return {'status': 'need_clarify', 'corpus_id': None,
                        'reason': f'“这篇”指向的文献(id={cid})不在当前检索结果中，请重新指明第几篇或标题',
                        'candidates': ordered_list[:10]}
        # 没有任何锚点：必须澄清
        if ordered_list:
            return {'status': 'need_clarify', 'corpus_id': None,
                    'reason': '无法确定您指的是哪篇文献，请明确说"第几篇"或提供标题',
                    'candidates': ordered_list[:10]}
        return {'status': 'no_reference', 'corpus_id': None,
                'reason': '当前没有可引用的文献列表，请先进行检索'}

    # ── 5. 无指代 ────────────────────────────────────────
    return {'status': 'no_reference', 'corpus_id': None, 'reason': '问题中未发现文献指代'}


def is_metadata_only_question(question):
    """
    判断是否只依赖元数据（id/title/author/year/keyword/abstract）即可回答。
    扩展关键词覆盖：作者、发表时间、期刊、关键词、语言、文献量、检索相关等。
    """
    q = question.lower()
    meta_keywords = [
        # 作者/机构
        '作者', '编者', '译者',
        # 发表/出版时间
        '发表时间', '发表年份', '出版年份', '出版时间', '哪一年', '什么时候发表',
        '发表期刊', '期刊名称', '期刊信息', '哪个期刊',
        # 关键词/分类
        '关键词', '分类', '类别', '属于哪类',
        # 语言/数量
        '语言', '语种', '英文', '中文', '收录数量', '文献数量', '共多少',
        # 检索/筛选相关
        '第几篇', '第几本', '结果有多少', '检索到多少'
    ]
    return any(kw in q for kw in meta_keywords)

@bp.route('/timeline', methods=['POST'])
def generate_timeline_chart():
    """按关键词检索文献，并按年份、类型、语言或作者生成统计图数据。

    年份趋势按有记录的连续数据点计算，不补齐缺失年份；作者统计沿用当前首段截取规则。

    Returns:
        聚合数据、最多五十条文献及可选趋势分析 JSON，或参数、查询错误响应。
    """
    try:
        # 获取请求数据
        # 1. 校验关键词和统计维度，再按现有公开文献条件查询候选
        data = request.get_json()
        if not data or 'keyword' not in data:
            return jsonify({
                "code": 400,
                "error": "缺少关键词参数"
            }), 400

        keyword = data['keyword'].strip()
        if not keyword:
            return jsonify({
                "code": 400,
                "error": "关键词不能为空"
            }), 400

        # 获取维度参数，默认为按年份统计
        dimension = data.get('dimension', 'year')
        # 处理空字符串或空白字符的情况
        if not dimension or not dimension.strip():
            dimension = 'year'

        supported_dimensions = ['year', 'category', 'language', 'author']
        if dimension not in supported_dimensions:
            return jsonify({
                "code": 400,
                "error": f"不支持的统计维度。可选维度: {', '.join(supported_dimensions)}"
            }), 400

        # 维度中文映射
        dimension_cn_map = {
            'year': '年份',
            'category': '类别',
            'language': '语言',
            'author': '作者'
        }
        dimension_cn = dimension_cn_map.get(dimension, dimension)

        # 生成检索文献的SQL查询
        # 搜索标题、关键词和摘要中的相关文献
        search_sql = f"""
        SELECT
            c.id,
            c.title_cn,
            c.title_en,
            c.publish_time,
            c.language,
            c.author,
            cc.category_name
        FROM corpus c
        LEFT JOIN corpus_category cc ON c.category_id = cc.id
        WHERE c.is_deleted = 0
          AND c.status = 2
          AND (
              c.title_cn LIKE '%{keyword}%'
              OR c.title_en LIKE '%{keyword}%'
              OR c.keyword_cn LIKE '%{keyword}%'
              OR c.keyword_en LIKE '%{keyword}%'
              OR c.abstract_cn LIKE '%{keyword}%'
              OR c.abstract_en LIKE '%{keyword}%'
          )
        ORDER BY c.publish_time DESC
        """

        # 执行查询
        search_results = nl2sql_manager.execute_query(search_sql)

        if not search_results:
            return jsonify({
                "code": 200,
                "data": {
                    "keyword": keyword,
                    "dimension": dimension,
                    "total_count": 0,
                    "chart_data": [],
                    "search_results": [],
                    "trend_analysis": {},
                    "analysis": {
                        "data_points": 0,
                        "max_value": 0,
                        "total_items": 0,
                        "dimension": dimension,
                    },
                    "message": f"未找到包含关键词'{keyword}'的文献"
                }
            })

        # 根据维度统计数据
        def analyze_by_dimension(results, dim):
            """将命中文献按指定维度计数，年份升序，分类和语言按数量降序。

            Args:
                results: 已检索到的文献字典列表。
                dim: year、category、language 或 author；作者分支仅取前十名。

            Returns:
                含维度值、count 和 label 的图表条目列表。
            """
            if dim == 'year':
                # 按年份统计
                count_dict = {}
                for item in results:
                    publish_time = item.get('publish_time')
                    if publish_time:
                        try:
                            year = int(publish_time)
                            count_dict[year] = count_dict.get(year, 0) + 1
                        except (ValueError, TypeError):
                            continue
                return [{"year": year, "count": count, "label": str(year)}
                       for year, count in sorted(count_dict.items())]

            elif dim == 'category':
                # 按类别统计
                count_dict = {}
                for item in results:
                    category = item.get('category_name') or '未分类'
                    count_dict[category] = count_dict.get(category, 0) + 1
                return [{"category": cat, "count": count, "label": cat}
                       for cat, count in sorted(count_dict.items(), key=lambda x: x[1], reverse=True)]

            elif dim == 'language':
                # 按语言统计
                count_dict = {}
                for item in results:
                    language = item.get('language') or '未知'
                    count_dict[language] = count_dict.get(language, 0) + 1
                return [{"language": lang, "count": count, "label": lang}
                       for lang, count in sorted(count_dict.items(), key=lambda x: x[1], reverse=True)]

            elif dim == 'author':
                # 按作者统计（取前10位主要作者）
                count_dict = {}
                for item in results:
                    author_str = item.get('author') or ''
                    if author_str:
                        # 简单处理多个作者的情况，取第一个
                        authors = author_str.split(';')[0].split(',')[0].split(' ')[0].strip()
                        if authors:
                            count_dict[authors] = count_dict.get(authors, 0) + 1
                # 只返回前10位作者
                top_authors = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)[:10]
                return [{"author": author, "count": count, "label": author}
                       for author, count in top_authors]

            return []

        # 分析数据
        # 2. 按指定维度计数；年份按值排序，其他维度按数量展示
        chart_data = analyze_by_dimension(search_results, dimension)

        # 趋势分析（仅对时间序列数据）
        # 3. 年份数据充足时计算相邻数据点增长率和三点均值，不补齐缺失年份
        trend_analysis = {}
        if dimension == 'year' and len(chart_data) >= 3:
            # 计算年度增长率
            growth_rates = []
            for i in range(1, len(chart_data)):
                prev_count = chart_data[i-1]['count']
                curr_count = chart_data[i]['count']
                if prev_count > 0:
                    growth_rate = ((curr_count - prev_count) / prev_count) * 100
                    growth_rates.append({
                        'year': chart_data[i]['year'],
                        'growth_rate': round(growth_rate, 2),
                        'prev_year': chart_data[i-1]['year'],
                        'prev_count': prev_count,
                        'curr_count': curr_count
                    })

            # 计算移动平均（3年期）
            moving_averages = []
            for i in range(2, len(chart_data)):
                avg_count = (chart_data[i-2]['count'] + chart_data[i-1]['count'] + chart_data[i]['count']) / 3
                moving_averages.append({
                    'year': chart_data[i]['year'],
                    'moving_average': round(avg_count, 2)
                })

            trend_analysis = {
                'growth_rates': growth_rates,
                'moving_averages': moving_averages,
                'total_trend': '上升' if growth_rates and sum(r['growth_rate'] for r in growth_rates[-3:]) > 0 else '下降',
                'avg_growth_rate': round(sum(r['growth_rate'] for r in growth_rates) / len(growth_rates), 2) if growth_rates else 0
            }

        # 返回结果（提供给前端 ECharts 使用的数据）
        return jsonify({
            "code": 200,
            "data": {
                "keyword": keyword,
                "dimension": dimension,
                "total_count": len(search_results),
                "chart_data": chart_data,
                "search_results": search_results[:50],  # 只返回前50条详细结果
                "trend_analysis": trend_analysis,
                "analysis": {
                    "data_points": len(chart_data),
                    "max_value": max([item.get('count', 0) for item in chart_data]) if chart_data else 0,
                    "total_items": len(search_results),
                    "dimension": dimension,
                }
            }
        })

    except Exception as e:
        return jsonify({
            "code": 500,
            "error": f"生成时间线图表失败: {str(e)}"
        }), 500


@bp.route('/timeline_by_ids', methods=['POST'])
def generate_timeline_chart_by_ids():
    """过滤所选 ID 中有效且审核通过的文献，按有效出版年份升序统计，不补齐空年份。"""
    try:
        # 1. 将所选 ID 规范化为有效整数，只查询当前有效且审核通过的文献
        data = request.get_json() or {}

        # 使用 corpus_ids 参数，默认为空列表
        corpus_ids = data.get('corpus_ids', [])
        if not isinstance(corpus_ids, list) or len(corpus_ids) == 0:
            return jsonify({
                "code": 400,
                "error": "文献ID列表(corpus_ids)必须为非空数组"
            }), 400

        # 转换为整数ID，过滤掉非法值
        valid_ids = []
        for _id in corpus_ids:
            try:
                valid_ids.append(int(_id))
            except (ValueError, TypeError):
                continue

        if not valid_ids:
            return jsonify({
                "code": 400,
                "error": "文献ID列表中没有有效的ID"
            }), 400

        # 使用 SQLAlchemy 查询对应文献的发表年份
        query = Corpus.query.filter(
            Corpus.is_deleted == 0,
            Corpus.status == 2,
            Corpus.id.in_(valid_ids)
        )
        corpus_list = query.all()

        if not corpus_list:
            return jsonify({
                "code": 200,
                "data": {
                    "total_count": 0,
                    "chart_data": [],
                    "analysis": {
                        "data_points": 0,
                        "max_value": 0,
                        "total_items": 0,
                    },
                    "message": "未找到对应文献"
                }
            })

        # 按年份统计数量
        # 2. 只对可解析的出版年份计数，缺失年份仍包含在查询总量中
        count_dict = {}
        for corpus in corpus_list:
            publish_time = corpus.publish_time
            if publish_time:
                try:
                    # YEAR 类型，直接转为 int
                    year = int(publish_time)
                    count_dict[year] = count_dict.get(year, 0) + 1
                except (ValueError, TypeError):
                    continue

        chart_data = [
            {"year": year, "count": count, "label": str(year)}
            for year, count in sorted(count_dict.items())
        ]

        return jsonify({
            "code": 200,
            "data": {
                "total_count": len(corpus_list),
                "chart_data": chart_data,
                "analysis": {
                    "data_points": len(chart_data),
                    "max_value": max([item.get('count', 0) for item in chart_data]) if chart_data else 0,
                    "total_items": len(corpus_list),
                }
            }
        })

    except Exception as e:
        return jsonify({
            "code": 500,
            "error": f"根据文献生成图表失败: {str(e)}"
        }), 500

@bp.route('/corpus_qa', methods=['POST'])
@permission_required(['LEADER','ADMIN','COLLECTOR'])
def corpus_qa():
    """校验选中文献或历史会话，准备文献元数据后返回多轮问答 SSE 流。

    Returns:
        保持请求上下文的流式响应，或问题、文献和历史归属校验错误。
    """
    data = request.get_json()
    if not data:
        return jsonify({
            "code": 400,
            "error": "缺少请求参数"
        }), 400

    question_type = data.get('question_type', 'custom')
    question = data.get('question')
    history_id = data.get('history_id')  # 多轮对话历史ID
    global_search = data.get('global_search', False)  # 前端全局搜索按钮

    # 加载或验证历史记录
    history_record = None
    history_messages = []
    is_new_session = True
    if history_id:
        history_record = CorpusQAHistory.query.filter_by(
            id=history_id,
            user_id=g.current_user.id,
            is_deleted=0
        ).first()
        if history_record:
            history_messages = history_record.load_messages()
            is_new_session = False

    # 首次请求必须有 corpus_ids
    corpus_ids = data.get('corpus_ids', [])
    if not isinstance(corpus_ids, list):
        corpus_ids = [corpus_ids]

    if is_new_session:
        if question_type == 'comparison':
            if len(corpus_ids) < 2:
                return jsonify({"code": 400, "error": "对比功能至少需要选择2篇文献"}), 400
        elif not global_search:
            if len(corpus_ids) < 1:
                return jsonify({"code": 400, "error": "请至少选择1篇文献"}), 400
        if len(corpus_ids) > 20:
            return jsonify({"code": 400, "error": "一次最多只能选择20篇文献"}), 400

    supported_types = [
        'custom', 'summary', 'comparison', 'core_idea', 'outline', 'highlights', 'methodology'
    ]
    if question_type not in supported_types:
        return jsonify({
            "code": 400,
            "error": f"不支持的问答类型。可选类型: {', '.join(supported_types)}"
        }), 400

    if question_type == 'custom':
        if not question or not question.strip():
            return jsonify({"code": 400, "error": "问题不能为空"}), 400
    else:
        preset_questions = {
            'summary': "请对这篇文献进行全面的综述，包括核心观点、研究方法、主要发现和结论。",
            'comparison': "请对比分析这些文献的异同点，包括研究主题、方法论、主要发现和结论。",
            'core_idea': "请提炼这篇文献的核心观点，并用简洁的语言进行概括。",
            'outline': "请根据这篇文献生成一个结构清晰的文章大纲，包含主要章节和关键小节。",
            'highlights': "请总结这篇文献的关键要点，包括研究背景、主要结论和创新点。",
            'methodology': "请说明这篇文献采用的研究方法，包括研究设计、数据来源与分析方法等。"
        }
        question = preset_questions.get(question_type, question)

    # 加载文献（多轮对话时复用首次请求的 corpus_ids）
    if is_new_session:
        from sqlalchemy import and_
        corpora = db.session.query(
            Corpus.id, Corpus.title_cn, Corpus.title_en, Corpus.author,
            Corpus.publish_time, Corpus.keyword_cn, Corpus.keyword_en,
            Corpus.abstract_cn, Corpus.abstract_en, Corpus.attachment,
            CorpusCategory.category_name
        ).outerjoin(
            CorpusCategory, Corpus.category_id == CorpusCategory.id
        ).filter(
            and_(
                Corpus.id.in_(corpus_ids),
                Corpus.is_deleted == 0,
                Corpus.status == 2
            )
        ).all()

        if not corpora:
            return jsonify({
                "code": 404,
                "error": "未找到符合条件的文献，请确保文献已审核通过"
            }), 404

        corpus_list = []
        for corpus in corpora:
            corpus_dict = {
                'id': corpus.id,
                'title_cn': corpus.title_cn,
                'title_en': corpus.title_en,
                'author': corpus.author,
                'publish_time': corpus.publish_time,
                'keyword_cn': corpus.keyword_cn,
                'keyword_en': corpus.keyword_en,
                'abstract_cn': corpus.abstract_cn,
                'abstract_en': corpus.abstract_en,
                'category_name': corpus.category_name,
                'file_ids': []
            }
            # 解析关键词 JSON
            for kw_field in ('keyword_cn', 'keyword_en'):
                val = corpus_dict.get(kw_field)
                if val and isinstance(val, str):
                    try:
                        corpus_dict[kw_field] = json.loads(val)
                    except (json.JSONDecodeError, TypeError):
                        pass

            # 注意：不在此处上传附件，改为按需上传（根据用户问题动态确定目标文献）
            # 附件上传逻辑移至 generate() 函数中

            corpus_list.append(corpus_dict)

        found_ids = {c['id'] for c in corpus_list}
        missing_ids = set(corpus_ids) - found_ids
        warning = f"注意：部分文献未找到或未审核通过（ID: {', '.join(map(str, missing_ids))}）" if missing_ids else None
    else:
        # 多轮对话：从历史记录的 corpus_ids 重建 corpus_list
        corpus_list = []
        from sqlalchemy import and_ as _and
        try:
            hist_corpus_ids = json.loads(history_record.corpus_ids) if history_record.corpus_ids else []
        except (json.JSONDecodeError, TypeError):
            hist_corpus_ids = []
        if hist_corpus_ids:
            corpora = db.session.query(
                Corpus.id, Corpus.title_cn, Corpus.title_en, Corpus.author,
                Corpus.publish_time, Corpus.keyword_cn, Corpus.keyword_en,
                Corpus.abstract_cn, Corpus.abstract_en, Corpus.attachment,
                CorpusCategory.category_name
            ).outerjoin(
                CorpusCategory, Corpus.category_id == CorpusCategory.id
            ).filter(
                _and(
                    Corpus.id.in_(hist_corpus_ids),
                    Corpus.is_deleted == 0,
                    Corpus.status == 2
                )
            ).all()
            for corpus in corpora:
                corpus_dict = {
                    'id': corpus.id, 'title_cn': corpus.title_cn, 'title_en': corpus.title_en,
                    'author': corpus.author, 'publish_time': corpus.publish_time,
                    'keyword_cn': corpus.keyword_cn, 'keyword_en': corpus.keyword_en,
                    'abstract_cn': corpus.abstract_cn, 'abstract_en': corpus.abstract_en,
                    'category_name': corpus.category_name, 'file_ids': []
                }
                for kw_field in ('keyword_cn', 'keyword_en'):
                    val = corpus_dict.get(kw_field)
                    if val and isinstance(val, str):
                        try:
                            corpus_dict[kw_field] = json.loads(val)
                        except (json.JSONDecodeError, TypeError):
                            pass
                # 注意：不在此处上传附件，改为按需上传
                corpus_list.append(corpus_dict)
        warning = None

    user_id = g.current_user.id


    def generate():
        """识别本轮意图，持久化问题后分流为检索跳转、时间线或文献问答。

        问答分支先解析指代并按需上传原文，再检索引用、流式生成并保存回答。
        检索失败可降级为摘要模式；指代不明确时发送澄清并提前结束。

        Yields:
            meta、jump、clarify、sources、chunk、error 或 done 等 SSE 事件帧。
        """
        nonlocal history_record, history_messages, corpus_list

        # 意图识别
        # 1. 对自由问题识别检索、时间线或问答意图，低置信度按现有策略降级问答
        intent_result = None
        if question_type == 'custom' and question:
            intent_result = nl2sql_manager.detect_intent_with_llm(question, corpus_list)
            current_app.logger.info(f"[意图识别] 结果: {intent_result}")
        else:
            intent_result = {'intent': 'qa', 'confidence': 1.0, 'keywords': []}

        # 根据意图处理
        intent = intent_result.get('intent', 'qa')
        confidence = intent_result.get('confidence', 1.0)
        keywords = intent_result.get('keywords', [])
        
        # 置信度阈值，低于0.6按qa处理
        if confidence < 0.6:
            intent = 'qa'
            current_app.logger.info(f"[意图识别] 置信度过低({confidence})，降级为qa")

        # 如果是新会话，先创建历史记录（初始 messages 为空）
        # 2. 在模型长调用前先保存用户问题和会话，向前端发布本轮元数据
        if is_new_session:
            try:
                def ensure_utf8(text):
                    """让字符串经 UTF-8 往返确认可编码，非字符串沿用原值。"""
                    if isinstance(text, str):
                        return text.encode('utf-8').decode('utf-8')
                    return text

                # 保存 corpus_ids
                saved_corpus_ids = [c['id'] for c in corpus_list if c.get('id')]

                history_record = CorpusQAHistory(
                    user_id=user_id,
                    question_type=question_type,
                    question=ensure_utf8(question),
                    corpus_ids=json.dumps(saved_corpus_ids),
                    corpus_count=len(saved_corpus_ids),
                    answer='',
                    messages=json.dumps([], ensure_ascii=False)
                )
                # 保存初始用户消息
                initial_messages = [{"role": "user", "content": ensure_utf8(question)}]
                history_record.messages = json.dumps(initial_messages, ensure_ascii=False)
                db.session.add(history_record)
                db.session.commit()
            except Exception as e:
                current_app.logger.warning(f"创建问答历史记录失败: {str(e)}")
                yield f"event: error\ndata: {json.dumps({'error': '创建历史记录失败'}, ensure_ascii=False)}\n\n"
                return

        else:
            # 多轮对话：加载历史，追加用户问题并立即保存
            history_messages = history_record.load_messages()
            history_messages.append({"role": "user", "content": question})
            # 持久化用户问题
            try:
                history_record.messages = json.dumps(history_messages, ensure_ascii=False)
                db.session.commit()
            except Exception as e:
                current_app.logger.warning(f"保存用户问题失败: {e}")

        # 截断历史（保留最近 10 轮）
        history_messages = nl2sql_manager.truncate_messages(history_messages, max_pairs=10)

        # 发送 meta 信息（包含意图识别结果）
        meta = {
            "question_type": question_type,
            "question": question,
            "corpus_count": len(corpus_list) if is_new_session else (history_record.corpus_count if history_record else 0),
            "corpus_ids": [c['id'] for c in corpus_list] if is_new_session else (
                json.loads(history_record.corpus_ids) if history_record and history_record.corpus_ids else []
            ),
            "history_id": history_record.id if history_record else None,
            "is_new_session": is_new_session,
            "intent": intent,  # 意图类型
            "confidence": confidence,  # 置信度
            "keywords": keywords  # 关键词
        }
        if warning:
            meta["warning"] = warning
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"

        # 根据意图执行不同逻辑
        # 3. 检索与时间线意图发送跳转数据并提前结束，不进入原文问答链路
        if intent == 'search':
            # 搜索意图：和 /query 接口逻辑一致
            search_query = ' '.join(keywords) if keywords else question
            try:
                page_data = nl2sql_manager.query_papers_by_nl(search_query, page_num=1, page_size=20)
                
                jump_event = {
                    "action": "redirect",
                    "target": "search",
                    "data": {
                        "code": 200,
                        "data": {
                            "sql": page_data.get('sql', ''),
                            "result": page_data.get('result', []),
                            "pagination": {
                                "page_num": 1,
                                "page_size": 20,
                                "total": page_data.get('total', 0),
                                "total_pages": page_data.get('total_pages', 0)
                            }
                        }
                    },
                    "message": f"为您在全库中找到 {page_data.get('total', 0)} 篇相关文献，正在跳转..."
                }
                yield f"event: jump\ndata: {json.dumps(jump_event, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': f'搜索失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            
            yield f"event: done\ndata: {json.dumps({}, ensure_ascii=False)}\n\n"
            return

        elif intent == 'timeline':
            # 可视化意图
            try:
                # 情况1：有当前选中的文献ID（从问答页面带来的）
                ids_to_query = [c['id'] for c in corpus_list if c.get('id')]
                
                timeline_data = None
                
                if ids_to_query:
                    # 直接用选中的文献ID生成时间线
                    timeline_data = nl2sql_manager.query_timeline_by_ids(ids_to_query)
                elif keywords:
                    # 情况2：没有选中文献，但有关键词，先搜索关键词获取文献ID
                    search_query = ' '.join(keywords)
                    # 先搜索相关文献
                    search_data = nl2sql_manager.query_papers_by_nl(search_query, page_num=1, page_size=50)
                    if search_data and search_data.get('result'):
                        found_ids = [item['id'] for item in search_data['result'] if item.get('id')]
                        if found_ids:
                            timeline_data = nl2sql_manager.query_timeline_by_ids(found_ids)
                
                if timeline_data:
                    keyword_text = ' '.join(keywords) if keywords else question
                    jump_event = {
                        "action": "redirect",
                        "target": "timeline",
                        "data": {
                            "code": 200,
                            "data": {
                                "keyword": keyword_text,
                                "total_count": timeline_data.get('total_count', 0),
                                "chart_data": timeline_data.get('chart_data', []),
                                "analysis": timeline_data.get('analysis', {})
                            }
                        },
                        "message": f"已为您生成 {timeline_data.get('total_count', 0)} 篇文献的年份分布图，正在跳转..."
                    }
                else:
                    jump_event = {
                        "action": "redirect",
                        "target": "timeline",
                        "data": None,
                        "message": "未找到相关文献，无法生成趋势图"
                    }
                
                yield f"event: jump\ndata: {json.dumps(jump_event, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': f'生成图表失败: {str(e)}'}, ensure_ascii=False)}\n\n"
            
            yield f"event: done\ndata: {json.dumps({}, ensure_ascii=False)}\n\n"
            return
        else:
            # qa意图：正常问答流程
            history_messages = nl2sql_manager.truncate_messages(history_messages, max_pairs=10)

            # ── 按需上传：解析用户问题，确定目标文献 ──────────────────────────
            # 4. 解析文献指代并决定是否需要原文；指代不明确时返回澄清而非任选一篇
            target_corpus_ids = None
            current_app.logger.info(f"[QA] question_type={question_type}, question={question[:50] if question else ''}")
            if question_type == 'custom' and question:
                # 解析用户问题中的文献指代
                reference_result = resolve_reference(
                    question,
                    corpus_list,  # ordered_list
                    selected_corpus_id=None,  # 前端可传入
                    last_mentioned_corpus_id=None
                )
                current_app.logger.info(f"[按需上传] 文献指代解析结果: {reference_result}")

                if reference_result.get('status') == 'ok' and reference_result.get('corpus_id'):
                    # 用户明确指定了某篇文献
                    target_corpus_ids = [reference_result['corpus_id']]
                elif reference_result.get('status') == 'need_clarify':
                    # 需要澄清，返回提示信息
                    candidates = reference_result.get('candidates', [])
                    if candidates:
                        clarify_msg = "请问您指的是哪篇文献？\n"
                        for i, c in enumerate(candidates[:5], 1):
                            title = c.get('title_cn') or c.get('title_en', '未知标题')
                            clarify_msg += f"{i}. {title}\n"
                        yield f"event: clarify\ndata: {json.dumps({'message': clarify_msg, 'candidates': [{'id': c.get('id'), 'title': c.get('title_cn') or c.get('title_en', '未知标题')} for c in candidates[:5]]}, ensure_ascii=False)}\n\n"
                        yield f"event: done\ndata: {json.dumps({}, ensure_ascii=False)}\n\n"
                        return
                    yield f"event: error\ndata: {json.dumps({'error': reference_result.get('reason', '无法确定目标文献')}, ensure_ascii=False)}\n\n"
                    return

            # 如果没有明确指代，检查是否需要全文上传
            # 预设问题类型（如 core_idea、summary）默认用摘要回答，不需要上传附件
            need_full_text = question_type not in ('custom', 'summary', 'core_idea', 'outline', 'highlights', 'methodology')
            if not need_full_text and question_type == 'custom':
                # 自定义问题：需要深度分析时才上传全文
                full_text_keywords = ['原文', '详细', '具体', '引用', '段落', '第', '核心观点', '方法论', '研究设计']
                need_full_text = any(kw in question for kw in full_text_keywords)

            if target_corpus_ids is None and not need_full_text:
                # 没有明确指代且不需要全文，只用摘要回答
                target_corpus_ids = []
            elif target_corpus_ids is None:
                # 需要全文但没有明确指代，上传所有文献
                target_corpus_ids = [c['id'] for c in corpus_list if c.get('id')]

            # ── 按需上传附件 ──────────────────────────────────────────────────
            # 5. 仅下载并上传目标文献的所需附件，远程文件 ID 供既有问答模型复用
            uploaded_count = 0
            for corpus_dict in corpus_list:
                if corpus_dict['id'] not in target_corpus_ids:
                    continue  # 跳过非目标文献
                if corpus_dict.get('file_ids'):
                    continue  # 已有 file_ids，跳过

                # 从数据库获取附件信息并上传
                corpus_record = Corpus.query.filter_by(id=corpus_dict['id']).first()
                if corpus_record and corpus_record.attachment:
                    try:
                        attachment_data = json.loads(corpus_record.attachment)
                        if isinstance(attachment_data, list):
                            for attachment in attachment_data:
                                file_path = attachment.get('file_path')
                                if file_path:
                                    current_app.logger.info(f"[按需上传] 下载文件: {file_path}")
                                    file_content, status_code = get_file_content(file_path)
                                    if file_content and isinstance(file_content, bytes) and len(file_content) > 0:
                                        ext = file_path.rsplit('.', 1)[-1].lower() if '.' in file_path else ''
                                        if ext in ('pdf', 'docx', 'doc', 'txt'):
                                            file_id = nl2sql_manager.upload_file_to_qwen(file_content, file_path)
                                            if file_id:
                                                corpus_dict['file_ids'].append(file_id)
                                                uploaded_count += 1
                                                current_app.logger.info(f"[按需上传] 成功: {file_path} -> file_id={file_id}")
                    except (json.JSONDecodeError, TypeError, Exception) as e:
                        current_app.logger.error(f"[按需上传] 处理附件失败: {str(e)}")

            current_app.logger.info(f"[按需上传] 完成，上传了 {uploaded_count} 个文件，目标文献: {target_corpus_ids}")

            # ── 从 Qdrant 检索相关原文片段（RAG） ───────────────────────────
            # 只有在有明确目标文献时才使用 RAG
            # 6. 在本轮文献范围内召回片段并建立引用，失败时沿用摘要降级
            corpus_ids_for_search = target_corpus_ids if target_corpus_ids else [c['id'] for c in corpus_list if c.get('id')]
            rag_chunks = []
            try:
                rag_chunks = corpus_indexer.search_chunks(question, corpus_ids=corpus_ids_for_search, top_k=5)
            except Exception as e:
                current_app.logger.warning(f"[RAG] Qdrant 检索失败，使用纯摘要模式: {str(e)}")

            # 先发送 sources 事件，把引用来源结构化推给前端
            citations = []
            for idx, ch in enumerate(rag_chunks, 1):
                citations.append({
                    "citation_id": f"C{idx}",
                    "chunk_id": ch.get("chunk_id", ""),
                    "corpus_id": ch.get("corpus_id"),
                    "title": ch.get("title_cn") or ch.get("title_en", "未知标题"),
                    "title_en": ch.get("title_en", ""),
                    "author": ch.get("author", ""),
                    "publish_time": ch.get("publish_time"),
                    "category_name": ch.get("category_name", ""),
                    "file_name": ch.get("file_name", ""),
                    "file_path": ch.get("file_path", ""),
                    "file_ext": ch.get("file_ext", ""),
                    "score": ch.get("score", 0),
                    "rank": idx,
                    "chunk_text": ch.get("text", ""),
                    "locator_text": ch.get("locator_text", ""),
                    "chunk_char_start": ch.get("chunk_char_start"),
                    "chunk_char_end": ch.get("chunk_char_end"),
                    "chunk_length": ch.get("chunk_length"),
                    "page_start": ch.get("page_start"),
                    "page_end": ch.get("page_end"),
                    "prev_chunk_id": ch.get("prev_chunk_id"),
                    "next_chunk_id": ch.get("next_chunk_id"),
                    # PDF 行坐标高亮字段
                    "bbox_blocks": ch.get("bbox_blocks", []),
                    "page_idx_set": ch.get("page_idx_set", []),
                    "version": ch.get("version", ""),
                })
            yield f"event: sources\ndata: {json.dumps({'citations': citations}, ensure_ascii=False)}\n\n"

            # ── 构建目标文献列表（只包含相关文献） ─────────────────────────────
            # 7. 使用本轮可用来源流式回答，完成后提交助手消息；错误通过 SSE 显式报告
            target_corpus_list = [c for c in corpus_list if c['id'] in target_corpus_ids] if target_corpus_ids else []
            current_app.logger.info(f"[按需上传] 目标文献列表: {[c.get('title_cn') or c.get('title_en', '')[:20] for c in target_corpus_list]}")

            try:
                full_answer = []
                for chunk in nl2sql_manager.answer_from_messages(target_corpus_list, question, history_messages, rag_context=rag_chunks):
                    full_answer.append(chunk)
                    yield f"event: chunk\ndata: {json.dumps({'content': chunk}, ensure_ascii=False)}\n\n"

                final_answer = ''.join(full_answer)

                # 追加 assistant 消息到历史
                if history_record:
                    try:
                        msgs = history_record.load_messages()
                        msgs.append({"role": "assistant", "content": final_answer})
                        history_record.messages = json.dumps(msgs, ensure_ascii=False)
                        history_record.answer = final_answer
                        db.session.commit()
                    except Exception as e:
                        current_app.logger.warning(f"更新问答历史失败: {str(e)}")

                yield f"event: done\ndata: {json.dumps({'history_id': history_record.id if history_record else None}, ensure_ascii=False)}\n\n"

            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )

@bp.route('/corpus_qa/history', methods=['GET'])
@permission_required(['LEADER','ADMIN','COLLECTOR'])
def get_corpus_qa_history():
    """按当前用户筛选未删除的文献问答历史，返回分页和记录摘要。"""
    try:
        # 获取查询参数
        page_num = int(request.args.get('page_num', 1))
        page_size = int(request.args.get('page_size', 10))
        question_type = request.args.get('question_type')  # 可选过滤条件

        # 验证分页参数
        if page_num < 1:
            page_num = 1
        if page_size < 1 or page_size > 50:
            page_size = 10

        # 构建查询
        query = CorpusQAHistory.query.filter(
            CorpusQAHistory.user_id == g.current_user.id,  # 只查询当前用户的记录
            CorpusQAHistory.is_deleted == 0
        )

        # 可选的类型过滤
        if question_type:
            supported_types = ['custom', 'summary', 'comparison']
            if question_type in supported_types:
                query = query.filter(CorpusQAHistory.question_type == question_type)

        # 按创建时间倒序排列
        query = query.order_by(CorpusQAHistory.create_time.desc())

        # 分页查询
        history_page = query.paginate(page=page_num, per_page=page_size, error_out=False)

        # 转换为字典列表
        history_list = [item.to_dict() for item in history_page.items]

        return jsonify({
            "code": 200,
            "data": {
                "total": history_page.total,
                "pages": history_page.pages,
                "page_num": page_num,
                "page_size": page_size,
                "data": history_list
            }
        })

    except Exception as e:
        return jsonify({
            "code": 500,
            "error": f"获取历史记录失败: {str(e)}"
        }), 500

@bp.route('/corpus_qa/history/<int:history_id>', methods=['GET'])
@permission_required(['LEADER','ADMIN','COLLECTOR'])
def get_corpus_qa_history_detail(history_id):
    """获取特定历史记录详情"""
    try:
        # 查询历史记录
        history = CorpusQAHistory.query.filter(
            CorpusQAHistory.id == history_id,
            CorpusQAHistory.user_id == g.current_user.id,  # 确保只能查看自己的记录
            CorpusQAHistory.is_deleted == 0
        ).first()

        if not history:
            return jsonify({
                "code": 404,
                "error": "未找到该历史记录"
            }), 404

        return jsonify({
            "code": 200,
            "data": history.to_dict()
        })

    except Exception as e:
        return jsonify({
            "code": 500,
            "error": f"获取历史记录详情失败: {str(e)}"
        }), 500

@bp.route('/corpus_qa/history/<int:history_id>', methods=['DELETE'])
@permission_required(['LEADER','ADMIN','COLLECTOR'])
def delete_corpus_qa_history(history_id):
    """删除历史记录（软删除）"""
    try:
        # 查询历史记录
        history = CorpusQAHistory.query.filter(
            CorpusQAHistory.id == history_id,
            CorpusQAHistory.user_id == g.current_user.id,  # 确保只能删除自己的记录
            CorpusQAHistory.is_deleted == 0
        ).first()

        if not history:
            return jsonify({
                "code": 404,
                "error": "未找到该历史记录"
            }), 404

        # 软删除
        history.is_deleted = True
        db.session.commit()

        return jsonify({
            "code": 200,
            "message": "历史记录已删除"
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            "code": 500,
            "error": f"删除历史记录失败: {str(e)}"
        }), 500


# ═════════════════════════════════════════════════════════════════════
# 内部管理接口：文献向量化和 RAG 检索
# ═════════════════════════════════════════════════════════════════════
from app.utils import qdrant_utils as corpus_indexer
from app.utils.utils import get_file_content


@bp.route('/internal/corpus/<int:corpus_id>/index', methods=['POST'])
def corpus_vectorize(corpus_id):
    """同步清理并重建指定已审核文献的双路索引，不是后台任务入队接口。

    Args:
        corpus_id: 有效且审核通过的文献 ID。

    Returns:
        完整双路构建统计 JSON，或附件、构建及空切块错误响应。
    """
    # 1. 校验当前审核记录及原附件，拒绝为无效原件启动外部索引操作
    corpus = db.session.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.is_deleted == 0,
        Corpus.status == 2
    ).first()
    if not corpus:
        return jsonify({"code": 404, "error": "文献不存在"}), 404

    if not corpus.attachment:
        return jsonify({"code": 400, "error": "文献无附件，无需向量化"}), 400

    try:
        attachments = json.loads(corpus.attachment)
    except Exception:
        return jsonify({"code": 400, "error": "附件格式解析失败"}), 400

    if not isinstance(attachments, list):
        attachments = [attachments]
    attachments = [{**item, "source_file_path": item.get("file_path")}
                   for item in attachments]

    try:
        # 2. 提前提交不可用状态，再受双路服务文献锁保护清理和重建，失败不恢复旧就绪
        corpus.is_vectorized = False
        db.session.commit()
        corpus_indexer.delete_corpus_chunks(corpus_id)
        result = corpus_indexer.index_corpus_chunks(corpus_id, attachments)
    except Exception:
        current_app.logger.exception("单篇文献索引失败: corpus_id=%s", corpus_id)
        return jsonify({"code": 500, "error": "向量化失败，请检查服务日志"}), 500
    if not result["indexed"] or not result["total_chunks"]:
        return jsonify({"code": 422, "error": "原文未生成可检索片段"}), 422
    return jsonify({
        "code": 200,
        "message": "向量化完成",
        "data": result
    })


@bp.route('/internal/corpus/<int:corpus_id>/reindex', methods=['POST'])
def corpus_reindex(corpus_id):
    """先撤销就绪并清理双路旧片段，再从当前附件同步重建。

    Args:
        corpus_id: 有效且审核通过的文献 ID。

    Returns:
        重建统计 JSON；无附件时仅清理，构建失败不恢复旧就绪状态。
    """
    # 1. 加载有效已审核文献并规范化当前附件引用
    corpus = db.session.query(Corpus).filter(
        Corpus.id == corpus_id,
        Corpus.is_deleted == 0,
        Corpus.status == 2,
    ).first()
    if not corpus:
        return jsonify({"code": 404, "error": "文献不存在"}), 404

    try:
        attachments = json.loads(corpus.attachment) if corpus.attachment else []
    except Exception:
        attachments = []

    if not isinstance(attachments, list):
        attachments = [attachments]
    attachments = [{**item, "source_file_path": item.get("file_path")}
                   for item in attachments]

    try:
        # 2. 先提交不可检索状态，再清理旧双路快照；无附件时只执行清理
        corpus.is_vectorized = False
        db.session.commit()
        corpus_indexer.delete_corpus_chunks(corpus_id)
        if attachments:
            result = corpus_indexer.index_corpus_chunks(corpus_id, attachments)
        else:
            result = {"indexed": 0, "total_chunks": 0}
    except Exception:
        current_app.logger.exception("单篇文献重建索引失败: corpus_id=%s", corpus_id)
        return jsonify({"code": 500, "error": "重新向量化失败，请检查服务日志"}), 500
    if attachments and (not result["indexed"] or not result["total_chunks"]):
        return jsonify({"code": 422, "error": "原文未生成可检索片段"}), 422

    return jsonify({
        "code": 200,
        "message": "重新向量化完成",
        "data": result
    })


@bp.route('/internal/corpus/reindex_all', methods=['POST'])
def corpus_reindex_all():
    """逐篇同步重建已有就绪标记的已审核文献，失败只计入本篇错误。

    读取请求中的 dry_run，为真只返回候选数；真正运行会清理旧的双路片段、
    调用解析与模型，并提交每篇文献的状态，不是只读验证操作。

    Returns:
        候选数或成功、失败统计及错误列表。
    """
    data = request.get_json() or {}
    dry_run = data.get('dry_run', False)  # dry_run=True 只返回数量，不实际写入

    # 1. 固定已有就绪标记的已审核候选，预览只返回数量
    corpora = db.session.query(Corpus).filter(
        Corpus.is_deleted == 0,
        Corpus.status == 2,
        Corpus.is_vectorized == True,
        Corpus.attachment.isnot(None),
    ).all()

    if dry_run:
        return jsonify({
            "code": 200,
            "message": "dry_run 模式",
            "data": {
                "total_to_reindex": len(corpora),
                "version": corpus_indexer.PAYLOAD_VERSION,
            }
        })

    success = 0
    failed = 0
    errors = []

    # 2. 每篇独立撤销就绪、清理及重建，累计错误不撤销已成功文献
    for corpus in corpora:
        try:
            attachments = json.loads(corpus.attachment) if corpus.attachment else []
            if not isinstance(attachments, list):
                attachments = [attachments]
            attachments = [{**item, "source_file_path": item.get("file_path")}
                           for item in attachments]
            corpus.is_vectorized = False
            db.session.commit()
            corpus_indexer.delete_corpus_chunks(corpus.id)
            if attachments:
                result = corpus_indexer.index_corpus_chunks(corpus.id, attachments)
                if not result["indexed"] or not result["total_chunks"]:
                    raise RuntimeError("原文未生成可检索片段")
            success += 1
        except Exception as e:
            failed += 1
            errors.append({"corpus_id": corpus.id, "error": str(e)})

    return jsonify({
        "code": 200,
        "message": f"批量重建完成，成功 {success}，失败 {failed}",
        "data": {
            "success": success,
            "failed": failed,
            "errors": errors[:20],  # 最多返回20条错误详情
            "version": corpus_indexer.PAYLOAD_VERSION,
        }
    })


@bp.route('/internal/corpus/<int:corpus_id>/vector_status', methods=['GET'])
def corpus_vector_status(corpus_id):
    """查询文献向量化状态"""
    corpus = db.session.query(Corpus.id, Corpus.is_vectorized).filter(
        Corpus.id == corpus_id,
        Corpus.is_deleted == 0
    ).first()
    if not corpus:
        return jsonify({"code": 404, "error": "文献不存在"}), 404

    from app.models import CorpusIndexJob
    job = db.session.get(CorpusIndexJob, corpus_id)
    return jsonify({
        "code": 200,
        "data": {
            "corpus_id": corpus.id,
            "is_vectorized": bool(corpus.is_vectorized),
            "index_status": job.status if job and not job.is_deleted else None,
            "index_attempts": job.attempts if job and not job.is_deleted else 0,
            "index_error": job.error_message if job and not job.is_deleted else None,
            "vector_status": job.vector_status if job and not job.is_deleted else None,
            "lexical_status": job.lexical_status if job and not job.is_deleted else None,
            "index_build_id": job.build_id if job and not job.is_deleted else None,
            "index_chunk_count": job.chunk_count if job and not job.is_deleted else 0,
        }
    })


@bp.route('/internal/rag/search', methods=['POST'])
def rag_search():
    """
    RAG 检索接口（供调试/测试用）。
    请求体: {"query": "...", "corpus_ids": [1,2], "top_k": 5}
    返回: 相关原文片段列表
    """
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"code": 400, "error": "缺少 query 参数"}), 400

    query = data['query'].strip()
    corpus_ids = data.get('corpus_ids')
    top_k = int(data.get('top_k', 5))

    chunks = corpus_indexer.search_chunks(query, corpus_ids=corpus_ids, top_k=top_k)
    return jsonify({
        "code": 200,
        "data": {
            "query": query,
            "chunks": chunks,
            "count": len(chunks)
        }
    })


@bp.route('/internal/rag/collection_stats', methods=['GET'])
def rag_collection_stats():
    """
    返回 Qdrant collection 的统计信息，用于判断是否需要全量重建索引。
    """
    try:
        stats = corpus_indexer.count_collection_points()
        return jsonify({"code": 200, "data": stats})
    except Exception as e:
        return jsonify({"code": 500, "error": str(e)}), 500


@bp.route('/internal/source_snippet', methods=['POST'])
def get_source_snippet():
    """
    根据文件路径和字符区间取原文片段，供前端高亮展示。
    请求体: {
        "file_path": "uploaded_files/corpus/xxx.pdf",
        "char_start": 1200,
        "char_end": 1500,
        "window": 200   // 可选，默认 200
    }
    """
    data = request.get_json()
    if not data or 'file_path' not in data:
        return jsonify({"code": 400, "error": "缺少 file_path 参数"}), 400

    file_path = data['file_path']
    # 原文路径不能作为授权凭据；必须属于当前可读文献的真实附件清单。
    from app.api.literature_files import _readable_query
    if not isinstance(file_path, str) or not file_path:
        return jsonify(code=400, error='无效文件路径'), 400
    # 旧附件可能使用 ASCII 转义 JSON；LIKE 按字面量匹配，保留中文、引号、百分号等文件名。
    encoded_paths = {json.dumps(file_path, ensure_ascii=ascii_only)[1:-1]
                     for ascii_only in (False, True)}
    candidates = _readable_query().filter(or_(
        *(Corpus.attachment.contains(path, autoescape=True) for path in encoded_paths)
    )).all()
    if not any(file_path in {item.get('file_path') for item in json.loads(corpus.attachment or '[]')
                             if isinstance(item, dict)} for corpus in candidates):
        return jsonify(code=404, error='原文不存在或不可访问'), 404
    char_start = int(data.get('char_start', 0))
    char_end = int(data.get('char_end', 0))
    window = int(data.get('window', 200))

    result = corpus_indexer.get_source_snippet(file_path, char_start, char_end, window=window)
    return jsonify({"code": 200, "data": result})

