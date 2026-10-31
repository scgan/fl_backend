"""定义 Foreign 账号、文献、审核、索引任务及用户分组模型，并提供接口序列化方法。"""

import json

from app.exts import db, bcrypt
from sqlalchemy.dialects.mysql import BIT, YEAR
from sqlalchemy import Column, Integer, String, BigInteger, DateTime, Text, SmallInteger, Boolean, UniqueConstraint, Index
from sqlalchemy.dialects.mysql import TEXT


class User(db.Model):
    """保存注册身份、密码哈希及可选角色，供登录、审核和上下级权限判断使用。

    Attributes:
        id: 账号主键。
        username: 登录名。
        email: 注册和密码找回邮箱。
        password: bcrypt 哈希，不作为接口字段返回。
        name: 用户姓名。
        phone: 联系电话。
        picture: 身份图片的文件服务路径。
        role: 角色编号字符串的 JSON 数组，接口解码为整数列表。
        status: 注册审核状态：0 待审核、1 通过、2 驳回。
        is_deleted: 账号软删除标记。
        leader_id: 上级审核人 ID。
    """
    __tablename__ = 'user'

    id = Column(BigInteger, primary_key=True,autoincrement=True)
    username = Column(String(31), nullable=False)
    email = Column(String(31), nullable=True)
    password = Column(String(63), nullable=False)
    name = Column(String(63),  nullable=True)
    phone = Column(String(20), nullable=True)
    picture = Column(String(63), nullable=True)
    role = Column(String(31), nullable=False)
    status = Column(Integer, nullable=False)
    is_deleted = Column(BIT, default=False, nullable=True)
    leader_id = Column(BigInteger, nullable=False)

    def set_password(self, password):
        """将明文密码编码为 bcrypt 哈希，只修改模型，提交由调用方负责。"""
        self.password = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        """对照已存 bcrypt 哈希校验候选密码，返回是否匹配。"""
        return bcrypt.check_password_hash(self.password, password)

    def to_dict(self):
        """返回账号公开资料并解码角色，排除密码哈希和内部删除标记。"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'name': self.name,
            'phone': self.phone,
            'picture': self.picture,
            'role': User.role_json_to_list(self.role),
            'status': self.status
        }
    @staticmethod
    def role_list_to_json(role_list):
        """将角色编号逐项转为字符串，按旧数据库契约存为 JSON 数组。"""
        return json.dumps(list(map(str, role_list)))

    @staticmethod
    def role_json_to_list(role_json):
        """将已存角色 JSON 数组转换为权限判断使用的整数列表。"""
        return list(map(int, json.loads(role_json)))


class Corpus(db.Model):
    """保存文献元数据、审核状态和附件清单；索引任务状态单独维护。

    Attributes:
        id: 文献主键。
        number: 业务文献编号。
        title_cn: 原文题名，保留历史字段名；不代表内容一定为中文。
        title_en: 译文题名，当前采集界面不将其限定为英文。
        language: 文献语言。
        author: 作者信息。
        translator: 译者信息。
        editor: 编者信息。
        publish_time: 出版年份。
        ISSN: 刊物或文献标识。
        source: 来源信息。
        category_id: 文献类型 ID。
        keyword_cn: 中文关键词 JSON 数组。
        keyword_en: 英文关键词 JSON 数组。
        abstract_cn: 中文摘要。
        abstract_en: 外文摘要。
        original_name: 原作名称。
        original_language: 原作语言。
        original_time: 原作年份。
        original_source: 原作来源。
        word_number: 统计得到的字数或词数。
        attachment: 附件对象 JSON 数组，含 file_path，可带原名等元数据。
        contributor: 提交文献的用户 ID。
        status: -1 草稿、0 编辑中、1 待审核、2 通过、3 驳回、4 个人上传。
        owner_id: 个人上传文件所有者；非个人文献为空，不从客户端接收归属。
        comment: 审核意见。
        is_deleted: 文献软删除标记。
        is_vectorized: 索引就绪提示，不能替代双路任务状态和版本复核。
        submit_time: 提交审核时间。
        volume: 卷号。
        issue: 期号。
        reviewer_id: 数据审核人 ID。
    """
    __tablename__ = 'corpus'

    id = Column(BigInteger, primary_key=True,autoincrement=True)
    number = Column(String(31), nullable=True)
    title_cn = Column(String(1000), nullable=True)
    title_en = Column(String(1000), nullable=True)
    language = Column(String(31), nullable=True)
    author = Column(String(255), nullable=True)
    translator = Column(String(255), nullable=True)
    editor = Column(String(255), nullable=True)
    publish_time = Column(YEAR, nullable=True)
    ISSN = Column(String(127), nullable=True)
    source = Column(String(510), nullable=True)
    category_id = Column(Integer, nullable=True)
    keyword_cn =Column(String(510), nullable=True)
    keyword_en = Column(String(1000), nullable=True)
    abstract_cn =Column(Text, nullable=True)
    abstract_en =Column(Text, nullable=True)
    original_name = Column(String(255), nullable=True)
    original_language = Column(String(31), nullable=True)
    original_time = Column(YEAR, nullable=True)
    original_source = Column(String(510), nullable=True)
    word_number = Column(Integer, nullable=True)
    attachment =Column(Text, nullable=True)
    contributor =Column(BigInteger, nullable=False)
    status = Column(Integer, nullable=True, default=-1)  # -1 草稿，0 编辑中，1 待审核，2 通过，3 驳回，4 个人上传
    owner_id = Column(BigInteger, nullable=True, index=True, comment="个人上传文件所有者，公共文献为空")
    comment = Column(String(255), nullable=True)
    is_deleted = Column(BIT, default=False, nullable=True)
    is_vectorized = Column(BIT, default=False, nullable=True)
    submit_time = Column(DateTime, nullable=True)
    volume =Column(SmallInteger, nullable=True)
    issue =Column(SmallInteger, nullable=True)
    reviewer_id = Column(BigInteger, nullable=True)  # 数据审核人（二级管理员）

    def to_dict(self, with_keyword_ids=False):
        """序列化文献，将关键词及附件 JSON 转为接口结构。

        Args:
            with_keyword_ids: 为真时按文本和语言查询有效关键词 ID，供编辑页回填。

        Returns:
            文献字典；attachment 只含路径，关键词 ID 仅在请求时补充。
        """
        data = {
            'id': self.id,
            'number': self.number,
            'title_cn': self.title_cn,
            'title_en': self.title_en,
            'language': self.language,
            'author': self.author,
            'translator': self.translator,
            'editor': self.editor,
            'publish_time': self.publish_time,
            'ISSN': self.ISSN,
            'source': self.source,
            'category_id': self.category_id,
            'keyword_cn': json.loads(self.keyword_cn) if self.keyword_cn else None,
            'keyword_en': json.loads(self.keyword_en) if self.keyword_en else None,
            'abstract_cn': self.abstract_cn,
            'abstract_en': self.abstract_en,
            'original_name': self.original_name,
            'original_language': self.original_language,
            'original_time': self.original_time,
            'original_source': self.original_source,
            'word_number': self.word_number,
            'attachment': [item['file_path'] for item in json.loads(self.attachment)] if self.attachment else None,
            'contributor': self.contributor,
            'status': self.status,
            'owner_id': self.owner_id,
            'comment': self.comment,
            'submit_time': self.submit_time.strftime("%Y-%m-%d %H:%M:%S") if self.submit_time else self.submit_time,
            'volume': self.volume,
            'issue': self.issue,
            'reviewer_id': self.reviewer_id,
        }
        if with_keyword_ids:
            from app.models import Keywords
            # keyword_cn_ids
            if self.keyword_cn:
                cn_keywords = json.loads(self.keyword_cn)
                cn_ids = [k.id for k in Keywords.query.filter(Keywords.keyword.in_(cn_keywords), Keywords.language==0, Keywords.is_deleted==0).all()]
                data['keyword_cn_ids'] = cn_ids
            else:
                data['keyword_cn_ids'] = []
            # keyword_en_ids
            if self.keyword_en:
                en_keywords = json.loads(self.keyword_en)
                en_ids = [k.id for k in Keywords.query.filter(Keywords.keyword.in_(en_keywords), Keywords.language==1, Keywords.is_deleted==0).all()]
                data['keyword_en_ids'] = en_ids
            else:
                data['keyword_en_ids'] = []
        return data


class CorpusIndexJob(db.Model):
    """每篇文献保留一个当前索引任务，以版本、指纹和双路状态防止旧结果发布。

    Attributes:
        corpus_id: 文献 ID，同时作为任务主键。
        attachment_digest: 附件清单 SHA-256 指纹。
        status: 任务总状态；EXTRACTING 为 Backend 题录准备态，待领取值由队列配置决定。
        generation: 任务代次，旧执行者发布前必须复核。
        vector_status: Qdrant 分路状态。
        lexical_status: OpenSearch 分路状态。
        build_id: 双路共享的构建令牌。
        lexical_index: 本次写入的词法索引名称。
        chunk_count: 完整切块数。
        attempts: 当前一轮的领取尝试次数。
        next_attempt_at: 允许再次领取的 UTC 时间。
        requested_at: 最近请求重建的 UTC 时间。
        started_at: 题录准备或最近索引领取的 UTC 时间，供各阶段过期判断使用。
        finished_at: 完成、取消或最终失败的 UTC 时间。
        error_message: 失败原因，或 SUCCESS 时因内容审核跳过 OCR 页的警告摘要。
        is_deleted: 任务软删除标记。
    """
    __tablename__ = 'corpus_index_job'
    __table_args__ = (
        Index('ix_corpus_index_job_due', 'status', 'next_attempt_at'),
        {'mysql_charset': 'utf8mb4', 'mysql_collate': 'utf8mb4_unicode_ci'},
    )

    corpus_id = Column(BigInteger, primary_key=True)
    attachment_digest = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False)
    generation = Column(Integer, nullable=False, default=1, server_default='1')
    vector_status = Column(String(16), nullable=False, default="PENDING", server_default="PENDING",
                           comment="向量阶段 PENDING/RUNNING/SUCCESS/FAILED")
    lexical_status = Column(String(16), nullable=False, default="PENDING", server_default="PENDING",
                            comment="BM25 阶段 PENDING/RUNNING/SUCCESS/FAILED")
    build_id = Column(String(32), nullable=True, comment="两路共享的构建令牌")
    lexical_index = Column(String(128), nullable=True, comment="词法索引版本名")
    chunk_count = Column(Integer, nullable=False, default=0, server_default='0', comment="完整切块数")
    attempts = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(DateTime, nullable=False, default=db.func.now())
    requested_at = Column(DateTime, nullable=False, default=db.func.now())
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    is_deleted = Column(Boolean, nullable=False, default=False, server_default='0')


class CorpusCategory(db.Model):
    """维护文献类型和业务编码，供筛选、统计和编辑使用。

    Attributes:
        id: 类型主键。
        category_name: 类型名称。
        code: 唯一业务编码。
        number: 分类文献流水计数，创建正式文献时锁行递增。
        is_deleted: 软删除标记。
    """
    __tablename__ = 'corpus_category'

    id = Column(Integer, primary_key=True,autoincrement=True)
    category_name = Column(String(31), nullable=True)
    code = Column(String(15), unique=True, nullable=False)
    number = Column(Integer, nullable=True)
    is_deleted = Column(BIT, default=False, nullable=True)

    def to_dict(self):
        """返回类型 ID、名称和编码，保留接口使用的 category_id 键名。"""
        return {
            'category_id': self.id,
            'category_name': self.category_name,
            'code': self.code,
        }


class Database(db.Model):
    """维护主题数据库树，文献通过 CorpusDatabase 关联到节点。

    Attributes:
        id: 主题节点 ID。
        database_name: 主题名称。
        parent_id: 父节点 ID，现有树接口以 -1 表示根节点。
        is_deleted: 节点软删除标记。
    """
    __tablename__ = 'database'

    id = Column(Integer, primary_key=True,autoincrement=True)
    database_name = Column(String(127), nullable=False)
    parent_id = Column(Integer, nullable=True)
    is_deleted = Column(BIT, default=False, nullable=True)

class CorpusDatabase(db.Model):
    """保存文献与主题数据库节点的多对多关系。

    Attributes:
        id: 关联主键。
        corpus_id: 文献 ID。
        database_id: 主题节点 ID。
    """
    __tablename__ = 'corpus_database'

    id = Column(Integer, primary_key=True,autoincrement=True)
    corpus_id = Column(BigInteger,  nullable=False)
    database_id = Column(Integer,  nullable=False)


class Keywords(db.Model):
    """维护双语关键词及其来源、频次，供采集和后台管理使用。

    Attributes:
        id: 关键词主键。
        keyword: 关键词文本。
        flag: 来源：0 管理员、1 收集者。
        frequency: 业务维护的使用频次。
        is_deleted: 软删除标记。
        language: 语言：0 中文、1 英文。
    """
    __tablename__ = 'keywords'

    id = Column(Integer, primary_key=True,autoincrement=True)
    keyword = Column(String(31), nullable=False)
    flag = Column(Integer,  nullable=False)
    frequency = Column(Integer, default=0,nullable=True)
    is_deleted = Column(BIT, default=False, nullable=True)
    language = Column(Integer,  nullable=False)


class CorpusRecoveryLog(db.Model):
    """语料恢复过程记录表：记录从服务器 PDF 重新提取并写回 corpus 的每条处理结果（成功/失败/跳过）。"""
    __tablename__ = 'corpus_recovery_log'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    file_name = Column(String(255), nullable=False)   # PDF 文件名，如 xxx.pdf
    file_path = Column(String(512), nullable=False)   # attachment 用路径，如 uploaded_files/corpus/xxx.pdf
    status = Column(String(20), nullable=False)       # success / fail / skip
    error_message = Column(Text, nullable=True)        # 失败时错误信息
    corpus_id = Column(BigInteger, nullable=True)     # 成功时写入的 corpus.id
    # 本条恢复结果记录的创建时间，供恢复过程审计，不作为文献提交时间。
    run_at = Column(DateTime, nullable=False, default=db.func.now())

    def to_dict(self):
        """返回恢复记录摘要，并将执行时间转为接口日期文本。"""
        return {
            'id': self.id,
            'file_name': self.file_name,
            'file_path': self.file_path,
            'status': self.status,
            'error_message': self.error_message,
            'corpus_id': self.corpus_id,
            'run_at': self.run_at.strftime("%Y-%m-%d %H:%M:%S") if self.run_at else None,
        }


class CorpusQAHistory(db.Model):
    """保存普通文献问答的选中文献、消息和最终回答，与研究会话表分开。

    Attributes:
        id: 历史主键。
        user_id: 发起者 ID。
        question_type: custom、summary 或 comparison。
        question: 本轮用户问题。
        corpus_ids: 参与文献 ID 的 JSON 数组。
        corpus_count: 参与文献数。
        answer: 最终助手回答。
        messages: 按顺序保存的消息 JSON。
        create_time: 历史创建时间。
        is_deleted: 软删除标记。
    """
    __tablename__ = 'corpus_qa_history'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)  # 创建问答的用户ID
    question_type = Column(String(20), nullable=False)  # 'custom', 'summary', 'comparison'
    question = Column(TEXT(charset='utf8mb4'), nullable=False)  # 问题内容
    corpus_ids = Column(TEXT(charset='utf8mb4'), nullable=False)  # 涉及的文献ID列表（JSON格式）
    corpus_count = Column(Integer, nullable=False)  # 文献数量
    answer = Column(TEXT(charset='utf8mb4'), nullable=False)  # AI回答内容
    messages = Column(TEXT(charset='utf8mb4'), nullable=True)  # 完整对话历史（JSON格式）
    create_time = Column(DateTime, nullable=False, default=db.func.now())
    is_deleted = Column(BIT, default=False, nullable=True)

    def to_dict(self):
        """返回问答历史，将缺失或损坏的文献及消息 JSON 降级为空列表。"""
        try:
            corpus_ids_list = json.loads(self.corpus_ids) if self.corpus_ids else []
        except (json.JSONDecodeError, TypeError):
            corpus_ids_list = []

        try:
            messages_list = json.loads(self.messages) if self.messages else []
        except (json.JSONDecodeError, TypeError):
            messages_list = []

        return {
            'id': self.id,
            'user_id': self.user_id,
            'question_type': self.question_type,
            'question': self.question,
            'corpus_ids': corpus_ids_list,
            'corpus_count': self.corpus_count,
            'answer': self.answer,
            'messages': messages_list,
            'create_time': self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None
        }

    def load_messages(self):
        """读取持久化消息，历史 JSON 缺失或损坏时返回空列表。"""
        if not self.messages:
            return []
        try:
            return json.loads(self.messages)
        except (json.JSONDecodeError, TypeError):
            return []

    def append_user_message(self, role, content):
        """以调用方给定角色追加消息并更新 JSON，不在此提交事务。"""
        msgs = self.load_messages()
        msgs.append({"role": role, "content": content})
        self.messages = json.dumps(msgs, ensure_ascii=False)

    def append_assistant_message(self, content):
        """追加 assistant 消息并更新 JSON，不在此提交事务。"""
        msgs = self.load_messages()
        msgs.append({"role": "assistant", "content": content})
        self.messages = json.dumps(msgs, ensure_ascii=False)

class Topic(db.Model):
    """保存用户专题分组，供收藏与文献整理使用。

    Attributes:
        id: 专题主键。
        user_id: 所有者 ID。
        name: 专题名称。
        description: 专题说明。
        create_time: 创建时间。
        update_time: 更新时间。
        is_deleted: 软删除标记。
    """
    __tablename__ = 'topic'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    name = Column(String(127), nullable=False)
    description = Column(String(255), nullable=True)
    create_time = Column(DateTime, nullable=False, default=db.func.now())
    update_time = Column(DateTime, nullable=False, default=db.func.now(), onupdate=db.func.now())
    is_deleted = Column(Boolean, default=False, nullable=True)

    def to_dict(self, include_stats=False):
        """返回专题资料，可附带未软删除的文献关联数。

        Args:
            include_stats: 为真时查询关联数，不额外复核文献审核或删除状态。

        Returns:
            专题字典，可选含 corpus_count。
        """
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'create_time': self.create_time.strftime("%Y-%m-%d %H:%M:%S") if self.create_time else None,
            'update_time': self.update_time.strftime("%Y-%m-%d %H:%M:%S") if self.update_time else None,
        }
        if include_stats:
            data['corpus_count'] = TopicCorpus.query.filter_by(
                topic_id=self.id, is_deleted=0).count()
        return data


class TopicCorpus(db.Model):
    """保存专题与文献关系，唯一键支持软删除后恢复同一关联。

    Attributes:
        id: 关联主键。
        topic_id: 所属专题。
        corpus_id: 关联文献。
        create_time: 首次关联时间。
        is_deleted: 关联软删除标记。
    """
    __tablename__ = 'topic_corpus'
    __table_args__ = (
        UniqueConstraint('topic_id', 'corpus_id', name='uq_topic_corpus'),
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    topic_id = Column(BigInteger, nullable=False, index=True)
    corpus_id = Column(BigInteger, nullable=False, index=True)
    create_time = Column(DateTime, nullable=False, default=db.func.now())
    is_deleted = Column(Boolean, default=False, nullable=True)


class KnowledgeBase(db.Model):
    """保存用户私有知识库，研究入口通过所有者校验控制访问范围。

    Attributes:
        id: 知识库主键。
        owner_id: 所有者 ID。
        name: 知识库名称。
        description: 知识库说明。
        create_time: 创建时间。
        update_time: 更新时间。
        is_deleted: 软删除标记。
    """
    __tablename__ = 'knowledge_base'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    owner_id = Column(BigInteger, nullable=False, index=True)
    name = Column(String(127), nullable=False)
    description = Column(String(255), nullable=True)
    create_time = Column(DateTime, nullable=False, default=db.func.now())
    update_time = Column(DateTime, nullable=False, default=db.func.now(), onupdate=db.func.now())
    is_deleted = Column(Boolean, nullable=False, default=False)

    def to_dict(self):
        """序列化知识库资料及日期，所有者和删除标记由接口权限层处理。"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'create_time': self.create_time.strftime('%Y-%m-%d %H:%M:%S') if self.create_time else None,
            'update_time': self.update_time.strftime('%Y-%m-%d %H:%M:%S') if self.update_time else None,
        }


class KnowledgeBaseDocument(db.Model):
    """连接私有知识库与文献，以复合主键支持软删除后恢复。

    Attributes:
        knowledge_base_id: 知识库 ID，与 corpus_id 共同组成主键。
        corpus_id: 文献 ID。
        added_by: 执行关联操作的用户 ID。
        added_at: 最近加入时间。
        is_deleted: 关联软删除标记。
    """
    __tablename__ = 'knowledge_base_document'

    knowledge_base_id = Column(BigInteger, primary_key=True)
    corpus_id = Column(BigInteger, primary_key=True)
    added_by = Column(BigInteger, nullable=False)
    added_at = Column(DateTime, nullable=False, default=db.func.now())
    is_deleted = Column(Boolean, nullable=False, default=False)


class Terminology(db.Model):
    """术语库条目：术语名、双语定义与出处；附件以关联文献（corpus_id）形式保存。"""
    __tablename__ = 'terminology'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, comment='术语名')
    category = Column(BigInteger, nullable=False, comment='所属大类（database.id）')
    source_mla = Column(String(1020), nullable=False, comment='术语出处（MLA 格式）')
    definition_cn = Column(TEXT, nullable=False, comment='术语定义（中文）')
    definition_cn_source = Column(String(510), nullable=True, comment='中文定义出处')
    definition_en = Column(TEXT, nullable=False, comment='术语定义（英文）')
    definition_en_source = Column(String(510), nullable=True, comment='英文定义出处')
    source_attachment_corpus_id = Column(BigInteger, nullable=True, comment='术语出处附件文献 id')
    definition_cn_attachment_corpus_id = Column(BigInteger, nullable=True, comment='中文定义附件文献 id')
    definition_en_attachment_corpus_id = Column(BigInteger, nullable=True, comment='英文定义附件文献 id')
    # 多条目结构（JSON 文本，兼容旧列之后的新契约）：
    #   sources_json        {"mlas": ["MLA…"], "attachments": [{"corpus_id": 1, "name": "a.pdf"}]}
    #   definitions_cn_json [{"text": "…", "mlas": ["MLA…"], "attachments": [{"corpus_id": 1}]}]
    #   definitions_en_json 同上；附件入库后由后台线程回填 corpus_id
    sources_json = Column(TEXT, nullable=True, comment='术语出处多条目JSON（多条MLA+模块附件）')
    definitions_cn_json = Column(TEXT, nullable=True, comment='中文定义多条目JSON（多条定义，每条含多条MLA与附件）')
    definitions_en_json = Column(TEXT, nullable=True, comment='英文定义多条目JSON')
    attachment_process = Column(TEXT, nullable=True, comment='附件后台处理状态JSON：{附件字段:{state:processing|done|reused|failed, corpus_id, error}}')
    contributor = Column(BigInteger, nullable=False, comment='提交人（数据收集者）')
    status = Column(Integer, nullable=False, default=1, comment='1:待审核, 2:审核通过, 3:审核驳回')
    reviewer_id = Column(BigInteger, nullable=True, comment='审核人')
    # 术语提交审核的时间，与后台附件处理进度分开保存。
    submit_time = Column(DateTime, nullable=True)
    is_deleted = Column(BIT, default=False, nullable=True)

    def to_dict(self):
        """序列化术语旧版基础字段，保留出处和双语定义的文献引用 ID。"""
        return {
            'id': self.id,
            'name': self.name,
            'category': self.category,
            'source_mla': self.source_mla,
            'definition_cn': self.definition_cn,
            'definition_cn_source': self.definition_cn_source,
            'definition_en': self.definition_en,
            'definition_en_source': self.definition_en_source,
            'source_attachment_corpus_id': self.source_attachment_corpus_id,
            'definition_cn_attachment_corpus_id': self.definition_cn_attachment_corpus_id,
            'definition_en_attachment_corpus_id': self.definition_en_attachment_corpus_id,
            'contributor': self.contributor,
            'status': self.status,
            'submit_time': self.submit_time.strftime('%Y-%m-%d %H:%M:%S') if self.submit_time else None,
        }
