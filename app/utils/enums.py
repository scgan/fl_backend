"""约定账号角色、注册审核、文献审核及关键词来源和语言的整数编码。"""

from enum import Enum

class UserRole(Enum):
    """账号角色编码：0 管理员、1 负责人、2 收集者、3 公众。"""
    ADMIN = 0
    LEADER = 1
    COLLECTOR = 2
    PUBLIC = 3

class UserStatus(Enum):
    """注册审核状态：0 待审核、1 通过、2 驳回；与文献审核编码不同。"""
    WAITING_EXAMINE = 0
    APPROVE_EXAMINE = 1
    REJECT_EXAMINE = 2

class CorpusStatus(Enum):
    """文献状态：-1 草稿、0 编辑中、1 待审核、2 通过、3 驳回、4 个人上传。"""
    DRAFT = -1
    EDITING = 0
    WAITING_EXAMINE = 1
    APPROVE_EXAMINE = 2
    REJECT_EXAMINE = 3
    PERSONAL_UPLOAD = 4

class KeywordFlag(Enum):
    """关键词来源标记：0 管理员添加，1 收集者添加。"""
    ADMIN_ADDED = 0
    COLLECTOR_ADDED = 1

class KeywordLanguage(Enum):
    """关键词语言编码：0 中文，1 英文。"""
    CN = 0
    EN = 1
