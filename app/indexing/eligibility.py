"""统一索引资格：公开审核文献及具有所有者的个人材料可以构建，读取权限另行校验。"""


def can_index_corpus(corpus):
    """判断当前未删除文献是否具备有效的公共或个人身份，不授予任何读取权限。"""
    return bool(corpus and not corpus.is_deleted and (
        (corpus.status == 2 and corpus.owner_id is None)
        or (corpus.status == 4 and corpus.owner_id is not None and corpus.owner_id > 0)
    ))
