"""由 Backend 请求同步提取个人题录，先保存准备任务、提取成功后才向 Worker 开放索引。"""

import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta

import requests
from flask import current_app

from app.exts import db
from app.indexing.eligibility import can_index_corpus
from app.indexing.jobs import attachment_digest, enqueue_corpus_index, pending_status
from app.models import Corpus, CorpusIndexJob


# Backend 同步题录阶段；与 Worker 消费的 pending_status() 隔离，防止题录未完成就建索引。
METADATA_STATUS = 'EXTRACTING'
# Backend 意外退出后允许用户手动重试；不新增后台线程或任务消费者。
METADATA_TIMEOUT = timedelta(minutes=15)


def metadata_processing_active(job):
    """判断 Backend 题录请求是否仍在有效时间内，重复重试不并发调用模型。"""
    return bool(job and not job.is_deleted and job.status == METADATA_STATUS and job.started_at
                and datetime.utcnow() - job.started_at < METADATA_TIMEOUT)


def prepare_personal_index(corpus):
    """登记当前原件任务；缺题录时先置 EXTRACTING，完整材料可直接排队。

    Args:
        corpus: 已核对所有者的有效个人文献，事务由 Backend 调用方提交。

    Returns:
        是否需要同步提取题录。
    """
    enqueue_corpus_index(db.session, corpus)
    job = db.session.get(CorpusIndexJob, corpus.id)
    needs_extraction = not has_personal_metadata(corpus)
    if needs_extraction:
        job.status = METADATA_STATUS
        job.started_at = datetime.utcnow()
    return needs_extraction


def fail_personal_metadata(corpus_id, digest, generation, owner_id):
    """只将仍属当前原件的准备任务标记失败，撤销或新版本任务保持原状态。"""
    db.session.rollback()
    corpus = db.session.query(Corpus).filter(Corpus.id == corpus_id).with_for_update().one_or_none()
    job = db.session.query(CorpusIndexJob).filter(
        CorpusIndexJob.corpus_id == corpus_id).with_for_update().one_or_none()
    if (corpus and job and corpus.status == 4 and corpus.owner_id == owner_id
            and not corpus.is_deleted and not job.is_deleted
            and job.status == METADATA_STATUS and job.generation == generation
            and job.attachment_digest == digest and attachment_digest(corpus.attachment) == digest):
        job.status = 'FAILED'
        job.error_message = '题录提取失败，请重试'
        job.finished_at = datetime.utcnow()
    db.session.commit()


def keyword_json(value, limit):
    """将模型关键词规范为去重 JSON 数组，按数据库列容量截取完整词项。"""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            value = re.split(r'[;；,，、\n]', value)
    words = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, str):
            continue
        word = item.strip()[:80]
        if not word or word in words:
            continue
        if len(json.dumps(words + [word], ensure_ascii=False)) <= limit:
            words.append(word)
    return json.dumps(words, ensure_ascii=False) if words else None


def normalize_metadata(data):
    """校验个人文献必需的题名、关键词与摘要，只接受题录字段。

    Args:
        data: 模型返回的原始对象，不信任其中的所有者、审核状态等业务字段。

    Returns:
        符合数据库字段容量的题录赋值字典。

    Raises:
        ValueError: 模型结果缺少可用于阅读和推荐的基本信息。
    """
    if not isinstance(data, dict):
        raise ValueError('个人文件题录必须是 JSON 对象')
    values = {}
    for field, limit in (('title_cn', 1000), ('title_en', 1000), ('author', 255),
                         ('source', 510), ('language', 31), ('translator', 255),
                         ('editor', 255),
                         ('abstract_cn', 10000), ('abstract_en', 10000)):
        value = data.get(field)
        values[field] = value.strip()[:limit] if isinstance(value, str) else None
    values['keyword_cn'] = keyword_json(data.get('keyword_cn'), 510)
    values['keyword_en'] = keyword_json(data.get('keyword_en'), 1000)
    year = re.search(r'\b(19\d{2}|20\d{2}|21[0-5]\d)\b', str(data.get('publish_time') or ''))
    values['publish_time'] = int(year.group()) if year and 1901 <= int(year.group()) <= 2155 else None
    for field in ('volume', 'issue'):
        number = str(data.get(field) or '').strip()
        values[field] = int(number) if number.isascii() and number.isdigit() and len(number) <= 5 and int(number) <= 32767 else None
    if not (values['title_cn'] or values['title_en']):
        raise ValueError('个人文件未提取到文献标题')
    if not (values['keyword_cn'] or values['keyword_en']):
        raise ValueError('个人文件未提取到关键词')
    if not (values['abstract_cn'] or values['abstract_en']):
        raise ValueError('个人文件未提取到摘要')
    return values


def _current_source(corpus, job, digest, generation, owner):
    """核对个人身份、所有者和任务附件版本，拒绝已撤销或替换的输入。"""
    return bool(can_index_corpus(corpus) and corpus.status == 4 and corpus.owner_id == owner
                and job and not job.is_deleted and job.status == METADATA_STATUS
                and job.generation == generation and job.attachment_digest == digest
                and attachment_digest(corpus.attachment) == digest)


def has_personal_metadata(corpus):
    """判断个人题录是否已具备题名、摘要和有效关键词，供重试与补齐脚本共用。"""
    return bool((corpus.title_cn or corpus.title_en) and (corpus.abstract_cn or corpus.abstract_en)
                and (keyword_json(corpus.keyword_cn, 510) or keyword_json(corpus.keyword_en, 1000)))


def extract_and_queue_personal_metadata(corpus_id, digest, generation):
    """在 Backend 中同步提取题录，并在同一回填事务里将任务开放给 Worker。

    Args:
        corpus_id: 已持久化个人文件的主键。
        digest: 准备任务的附件指纹。
        generation: 准备任务代次。

    Returns:
        题录保存且任务已排队时为 True；原件、所有者或任务失效时为 False。

    Raises:
        RuntimeError: 文件格式不符或模型提取失败，由 Backend 返回可重试状态。
        ValueError: 提取结果缺少基本题录。
    """
    # 1. 只消费 Backend 的题录准备状态，Worker 不会领取该状态。
    corpus = db.session.get(Corpus, corpus_id)
    if not corpus or corpus.status != 4:
        return False
    job = db.session.get(CorpusIndexJob, corpus_id)
    owner = corpus.owner_id
    if not _current_source(corpus, job, digest, generation, owner):
        return False
    attachments = json.loads(corpus.attachment)
    if not isinstance(attachments, list) or len(attachments) != 1:
        raise RuntimeError('个人文件题录提取要求单个 PDF 附件')
    path = attachments[0].get('file_path') if isinstance(attachments[0], dict) else None
    if not isinstance(path, str) or not path.lower().endswith('.pdf'):
        raise RuntimeError('个人文件题录提取要求有效的 PDF 路径')
    # 不让下载和模型请求占用数据库行锁或长事务；原件版本在回填前重查。
    db.session.commit()

    # 2. 在本次 Backend 调用中复用普通采集提取，失败保留已落库文件供用户重试。
    from app.utils.literature_metadata import extract_personal_pdf_metadata

    with personal_source_path(path) as local_path:
        values = normalize_metadata(extract_personal_pdf_metadata(local_path))

    # 3. 长耗时模型调用后锁行复核，只有同一所有者和原件版本才能回填题录。
    db.session.expire_all()
    corpus = db.session.query(Corpus).filter(Corpus.id == corpus_id).with_for_update().one_or_none()
    job = db.session.query(CorpusIndexJob).filter(
        CorpusIndexJob.corpus_id == corpus_id).with_for_update().one_or_none()
    if not _current_source(corpus, job, digest, generation, owner):
        db.session.rollback()
        return False
    for field, value in values.items():
        if value is not None:
            setattr(corpus, field, value)
    job.status = pending_status()
    job.started_at = None
    job.error_message = None
    job.next_attempt_at = datetime.utcnow()
    db.session.commit()
    return True


@contextmanager
def personal_source_path(file_path):
    """仅为 Backend 题录请求读取个人 PDF，下载的临时原件在退出时删除。

    Args:
        file_path: 已由当前个人文献附件清单确认的 PDF 路径。

    Yields:
        可交给普通采集提取函数的本地 PDF 路径。

    Raises:
        RuntimeError: 文件未挂载且未配置文件服务。
    """
    base_dir = os.getenv('INDEX_FILE_BASE_DIR', '/foreign-files')
    local_path = file_path if os.path.isabs(file_path) else os.path.join(base_dir, file_path)
    if os.path.isfile(local_path):
        yield local_path
        return
    file_url = current_app.config.get('FILE_URL')
    if not file_url:
        raise RuntimeError('文件服务地址未配置，无法读取个人 PDF')
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as target:
            temporary_path = target.name
            with requests.get(f"{file_url.rstrip('/')}/files", params={'filename': file_path},
                              stream=True, timeout=(10, 120)) as response:
                response.raise_for_status()
                for content in response.iter_content(chunk_size=1024 * 1024):
                    target.write(content)
        yield temporary_path
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.unlink(temporary_path)
