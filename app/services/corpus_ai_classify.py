"""按文献元数据与附件内容调用大模型判定所属分类库，并批量回写审核状态。

该模块只服务于「历史数据审核」页的批量归类：读取仍待审核的文献，让模型在真实库树
的末级节点中选择归属，写回 corpus_database；命中自动通过条件时同时登记索引任务。
任务在守护线程内串行写库、并发调用模型，进度写入临时目录下的 JSON 供前端轮询，
因此不依赖进程内状态，重启或多 worker 部署下仍可读到最近一次任务。
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from flask import current_app
from openai import OpenAI
import requests
from sqlalchemy import or_

from app.exts import db
from app.models import Corpus, CorpusDatabase, Database
from app.utils.enums import CorpusStatus

try:
    from app.indexing.jobs import enqueue_corpus_index
except ImportError:  # master 分支尚未包含索引任务模块，通过时跳过入队，与该分支既有审核行为一致
    enqueue_corpus_index = None

LOGGER = logging.getLogger(__name__)

# 默认归类模型；flash 响应快、成本低，需要更稳时可切到 qwen-plus
DEFAULT_MODEL = (os.getenv('FOREIGN_AI_CLASSIFY_MODEL') or 'qwen3.8-flash').strip()

# 每批送入模型的文献条数；实测关闭思考后 12 条仍稳定全量返回，单条摊薄的固定开销更低
BATCH_SIZE = 12
# 分类阶段并发调用数；实测 4 路并发时模型返回不稳定（近半批次漏答或空结果），2 路最稳
CONCURRENCY = 2
# 单条文献摘要送入模型的最大字符数，超出部分截断
ABSTRACT_CHARS = 320
# 元数据信息不足时补读附件正文的最大字符数
ATTACHMENT_CHARS = 4000
# 中英文摘要合计低于该字符数视为信息不足，需要补读附件
LOW_INFO_CHARS = 80
# 进度中保留的最近处理明细条数
RECENT_ITEM_LIMIT = 40
# 置信度低于该值的记录即使通过也计入「建议抽检」，仅作统计口径
REVIEW_HINT_CONFIDENCE = 0.6
# 任务超过该秒数未更新即视为已失联，允许启动新任务
STALE_SECONDS = 300
# 模型调用失败后的重试次数（单批）
MAX_RETRY = 2
# 单次模型调用超时秒数；不设超时会在模型端无响应时挂满默认的 600 秒
LLM_TIMEOUT = 90
# 单次调用输出上限：每条约 100 token，留足余量避免整批结果被截断
MAX_OUTPUT_TOKENS = 4096
# 单批等待上限，超过即放弃该批并计入失败，避免整轮任务被一组卡死
BATCH_TIMEOUT = 200
# 连续多少轮整批都没有模型返回结果时终止任务：一轮 18 条全无返回即可判定模型不可用
MAX_CONSECUTIVE_MODEL_FAILURE = 1

_CATEGORY_RULES = """【归类规则】
1. 先定主题大区：教育思想（思想、学说、观念、学术史）/ 教育政策（政策文本、法规、规划、治理）/ 外语教育（外语教学、外语学科、外语人才培养、国际中文教育之外的语种教学）。
2. 再定地域：研究对象、作者机构、政策主体在中国境内为中国；研究他国或跨国比较的为国外。国外文献只能归入「国外……」开头的库。
3. 再看语种：文献语言为汉语、且内容为中文原创的进「汉语」库；文献语言非汉语（英、俄、日、法、韩、西等），或内容为对中文成果的翻译、介绍、评价，进「多语种」库。
4. 时代划分：1840 年以前为古代；1840—1949 年为近现代；1949 年以后为当代。以文献的研究对象所处时代为准，而非出版年。
5. 原文与译文并列对照的双语文本归平行语料库；单语种的著译作品不归平行语料库。
6. 外语教育大区里，进一步区分理论文献、政策文献、实践（教学模式、探索创新）、事件（会议、交流、报道）、人物（学者、名师访谈与生平）、外译成果、国外研究文献。
7. 一条文献最多选 2 个库，且必须是最贴切的末级库。
8. 每条都必须给出 database_ids，**不得返回空数组**；确实无法精确判断时，选最接近的末级库并把 confidence 降到 0.3—0.5，由人工复核。"""


def _now_text():
    """返回秒级本地时间字符串，供进度与日志展示。"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _progress_dir():
    """返回存放任务进度的目录，缺失时创建。"""
    path = os.path.join(tempfile.gettempdir(), 'foreign_ai_classify')
    os.makedirs(path, exist_ok=True)
    return path


def _progress_path(task_id):
    """按任务号拼接进度文件路径。"""
    safe_id = re.sub(r'[^0-9a-zA-Z_-]', '', task_id or '')
    return os.path.join(_progress_dir(), f'{safe_id}.json')


def _write_state(state):
    """原子写入任务状态，避免轮询方读到半截 JSON。"""
    path = _progress_path(state.get('task_id'))
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as handle:
        json.dump(state, handle, ensure_ascii=False)
    os.replace(tmp_path, path)


def _load_state(task_id):
    """读取指定任务状态，文件不存在或损坏时返回 None。"""
    path = _progress_path(task_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _state_age(state):
    """返回任务状态距现在的秒数，用于识别已失联的后台任务。"""
    try:
        updated_at = datetime.strptime(state.get('updated_at', ''), '%Y-%m-%d %H:%M:%S')
    except (TypeError, ValueError):
        return STALE_SECONDS + 1
    return (datetime.now() - updated_at).total_seconds()


def latest_task_state(owner_id=None):
    """返回最近一次任务的进度；传入 owner_id 时只看该用户发起的任务。"""
    task_dir = _progress_dir()
    candidates = []
    for name in os.listdir(task_dir):
        if not name.endswith('.json'):
            continue
        state = _load_state(name[:-5])
        if not state:
            continue
        if owner_id is not None and state.get('owner_id') != owner_id:
            continue
        candidates.append((state.get('started_at', ''), state))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def get_task_state(task_id=None, owner_id=None):
    """返回指定任务或该用户最近一次任务的进度，供接口层调用。"""
    return _load_state(task_id) if task_id else latest_task_state(owner_id)


def active_task_state(owner_id=None):
    """返回仍在运行且未失联的任务状态，无则返回 None。"""
    state = latest_task_state(owner_id)
    if not state or state.get('status') != 'running':
        return None
    if _state_age(state) > STALE_SECONDS:
        return None
    return state


def any_active_task():
    """返回全局任意正在运行的任务，用于避免并发任务相互覆盖进度文件。"""
    for name in os.listdir(_progress_dir()):
        if not name.endswith('.json'):
            continue
        state = _load_state(name[:-5])
        if state and state.get('status') == 'running' and _state_age(state) <= STALE_SECONDS:
            return state
    return None


def load_database_tree():
    """从数据库读取有效库树，返回展示文本与末级可选节点。

    Returns:
        (树形文本, 可选节点 id 集合, id 到名称的映射, id 到父节点的映射)；
        可选节点为无子节点的末级库，父节点仅用于展示层级与回填祖先链。
    """
    rows = (
        db.session.query(Database.id, Database.database_name, Database.parent_id)
        .filter(Database.is_deleted == 0)
        .order_by(Database.id)
        .all()
    )
    nodes = {}
    children = {}
    for database_id, name, parent_id in rows:
        clean_name = (name or '').strip()
        nodes[database_id] = {'name': clean_name, 'parent_id': parent_id}
        children.setdefault(parent_id, []).append(database_id)

    # 1. 末级节点即可选节点；父级只用于展示层级，避免把文献挂到主题大区上
    selectable = set()
    for database_id, node in nodes.items():
        if children.get(database_id):
            continue
        if not node['name']:
            continue
        selectable.add(database_id)

    # 2. 按 parent_id 递归展开为缩进文本，并标出可选节点
    lines = []

    def walk(parent_id, depth):
        for database_id in children.get(parent_id, []):
            node = nodes[database_id]
            mark = '【可选】' if database_id in selectable else ''
            lines.append(f'{"  " * depth}{database_id} {node["name"]}{mark}')
            walk(database_id, depth + 1)

    roots = [database_id for database_id, node in nodes.items() if node['parent_id'] in (None, -1, 0)]
    for root_id in roots:
        node = nodes[root_id]
        mark = '【可选】' if root_id in selectable else ''
        lines.append(f'{root_id} {node["name"]}{mark}')
        walk(root_id, 1)

    id_to_name = {database_id: node['name'] for database_id, node in nodes.items()}
    parent_map = {database_id: node['parent_id'] for database_id, node in nodes.items()}
    return '\n'.join(lines), selectable, id_to_name, parent_map


def expand_with_ancestors(database_ids, parent_map):
    """把末级库展开为「末级 + 全部上级」，与页面勾选子节点后上级同样勾选的规则一致。

    Args:
        database_ids: 模型判定的末级库 id 列表。
        parent_map: 库 id 到父节点 id 的映射。

    Returns:
        去重后的库 id 列表，先末级后各级祖先；根节点的 -1 不参与。
    """
    expanded = []
    for database_id in database_ids:
        current = database_id
        guard = 0
        while current not in (None, -1, 0) and guard < 32:
            if current not in expanded:
                expanded.append(current)
            current = parent_map.get(current)
            guard += 1
    return expanded


def _text(value, limit=None):
    """把任意字段安全转为去空白文本；年份等数值字段可能不是字符串。"""
    if value is None:
        return ''
    text = str(value).strip()
    return text[:limit] if limit else text


def _parse_keywords(raw):
    """把关键词字段的 JSON 文本转为用顿号连接的短语，异常时原样返回。"""
    if not raw:
        return ''
    try:
        values = json.loads(raw)
        if isinstance(values, list):
            return '、'.join(str(item) for item in values if item)
    except (TypeError, ValueError):
        pass
    return str(raw)


def _download_attachment(path):
    """从文件服务拉取附件字节，带连接与读取超时，失败返回 None。

    Args:
        path: 附件在文件服务中的相对路径。

    Returns:
        文件字节；服务不可用、超时或非 200 时返回 None。
    """
    file_url = (current_app.config.get('FILE_URL') or '').rstrip('/')
    if not file_url:
        return None
    try:
        response = requests.get(f'{file_url}/files', params={'filename': path}, timeout=(5, 20))
    except requests.RequestException as error:
        LOGGER.warning('下载附件失败 path=%s err=%s', path, type(error).__name__)
        return None
    if response.status_code != 200:
        return None
    return response.content


def _attachment_text(corpus):
    """按附件清单读取第一份可解析文件的前若干字符，失败返回空串。"""
    try:
        attachments = json.loads(corpus.attachment or '[]')
    except (TypeError, ValueError):
        return ''
    for item in attachments:
        path = ''
        if isinstance(item, dict):
            path = item.get('file_path') or item.get('path') or ''
        elif isinstance(item, str):
            path = item
        if not path:
            continue
        content = _download_attachment(path)
        if not content:
            continue
        lower = path.lower()
        try:
            if lower.endswith('.pdf'):
                import fitz

                with fitz.open(stream=content, filetype='pdf') as doc:
                    text = ''.join(page.get_text() for page in doc)
            elif lower.endswith('.docx'):
                from io import BytesIO

                from docx import Document

                document = Document(BytesIO(content))
                text = '\n'.join(paragraph.text for paragraph in document.paragraphs)
            elif lower.endswith('.txt'):
                text = content.decode('utf-8', errors='ignore')
            else:
                continue
        except Exception as error:  # 单个附件解析失败不影响整体归类
            LOGGER.warning('读取附件正文失败 corpus=%s err=%s', corpus.id, error)
            continue
        text = re.sub(r'\s+', ' ', text or '').strip()
        if text:
            return text[:ATTACHMENT_CHARS]
    return ''


def _document_payload(corpus, index):
    """把一条文献整理成送入模型的精简字典。"""
    abstract = ' '.join(
        part.strip() for part in (_text(corpus.abstract_cn), _text(corpus.abstract_en)) if part
    )
    payload = {
        'corpus_index': index,
        'corpus_id': corpus.id,
        '标题': _text(corpus.title_cn or corpus.title_en, 200),
        '副标题': _text(corpus.title_en, 200) if corpus.title_cn else '',
        '语言': _text(corpus.language),
        '作者': _text(corpus.author, 80),
        '译者': _text(corpus.translator, 80),
        '来源': _text(corpus.source, 120),
        '出版年': _text(corpus.publish_time, 20),
        '关键词': _text(_parse_keywords(corpus.keyword_cn) or _parse_keywords(corpus.keyword_en), 160),
        '摘要': abstract[:ABSTRACT_CHARS],
    }
    # 1. 题名与摘要都不足时补读附件正文，提升无法从题录判断的文献准确率
    if len(abstract) < LOW_INFO_CHARS and corpus.attachment:
        excerpt = _attachment_text(corpus)
        if excerpt:
            payload['正文节选'] = excerpt
    return payload


def _client():
    """用环境变量中的百炼凭据创建同步客户端。

    Returns:
        OpenAI 兼容客户端。

    Raises:
        RuntimeError: 未配置任何可用凭据。
    """
    key = (os.getenv('DASHSCOPE_API_KEY') or os.getenv('QWEN_API_KEY') or '').strip()
    if not key:
        raise RuntimeError('未配置 DASHSCOPE_API_KEY 或 QWEN_API_KEY，无法调用归类模型')
    base_url = (os.getenv('QWEN_BASE_URL') or 'https://dashscope.aliyuncs.com/compatible-mode/v1').strip()
    # 显式设置超时与不重试：SDK 默认 600 秒超时会让后台任务在模型无响应时长时间挂起
    return OpenAI(api_key=key, base_url=base_url, timeout=LLM_TIMEOUT, max_retries=0)


def _extract_json(text):
    """从模型回复中取出 JSON 对象，兼容代码块包裹与前后缀说明文字。"""
    if not text:
        return None
    fenced = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.S)
    candidate = fenced.group(1) if fenced else text
    start = candidate.find('{')
    end = candidate.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(candidate[start:end + 1])
    except ValueError:
        return None


def _classify_batch(client, model, tree_text, payloads):
    """对一批文献调用模型判定归属库。

    Returns:
        corpus_index 到判定结果的映射，模型整体失败时抛出异常由调用方降级。
    """
    system_prompt = (
        '你是「中国教育思想与外语教育多语种数据库」的文献分类专家。'
        '请依据给定的末级库清单判定每条文献的归属。'
        '只能输出 JSON，不要解释、不要 Markdown。'
    )
    user_prompt = (
        '【末级库清单】（格式：库 id 库名，缩进表示层级，仅【可选】为合法取值）\n'
        f'{tree_text}\n\n'
        f'{_CATEGORY_RULES}\n\n'
        '【待判定文献】\n'
        f'{json.dumps(payloads, ensure_ascii=False)}\n\n'
        '【输出格式】\n'
        '{"results":[{"corpus_index":1,"database_ids":[201],"confidence":0.86,"reason":"不超过40字的依据"}]}\n'
        'results 必须与输入条数一一对应，corpus_index 原样返回；'
        '每条都要有 database_ids 且不能为空数组，没有十足把握时选最接近的末级库并把 confidence 调低。'
    )
    kwargs = dict(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        temperature=0.1,
        response_format={'type': 'json_object'},
        # 必须显式放宽输出上限：默认上限会让整批结果被截断，表现为 JSON 解析失败或只返回前几条
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    # 关闭思考模式：qwen3.8-flash 默认先长推理，实测单批 88 秒降到 4 秒，判定结果不受影响
    try:
        completion = client.chat.completions.create(**kwargs, extra_body={'enable_thinking': False})
    except Exception as error:
        # 个别模型不认识该参数时退回普通调用，不影响功能可用性
        if 'enable_thinking' in str(error).lower() or getattr(error, 'status_code', None) == 400:
            LOGGER.info('当前模型不支持 enable_thinking，按默认模式调用: %s', error)
            completion = client.chat.completions.create(**kwargs)
        else:
            raise
    parsed = _extract_json(completion.choices[0].message.content)
    if not parsed or not isinstance(parsed.get('results'), list):
        raise ValueError('模型返回内容无法解析为结果数组')
    result_map = {}
    for item in parsed['results']:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get('corpus_index'))
        except (TypeError, ValueError):
            continue
        raw_ids = item.get('database_ids')
        if isinstance(raw_ids, (int, str)):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list):
            raw_ids = []
        clean_ids = []
        for value in raw_ids:
            try:
                clean_ids.append(int(value))
            except (TypeError, ValueError):
                continue
        try:
            confidence = float(item.get('confidence'))
        except (TypeError, ValueError):
            confidence = 0.0
        result_map[index] = {
            'database_ids': clean_ids,
            'confidence': max(0.0, min(1.0, confidence)),
            'reason': str(item.get('reason') or '')[:80],
        }
    return result_map


def _fallback_single(client, model, tree_text, payload):
    """单条重试，用于整批解析失败时逐条兜底。"""
    try:
        return _classify_batch(client, model, tree_text, [payload]).get(payload['corpus_index'])
    except Exception as error:
        LOGGER.warning('单条归类失败 corpus=%s err=%s', payload.get('corpus_id'), error)
        return None


def _save_result(corpus, database_ids, auto_approve, min_confidence, confidence, reviewer_id):
    """写回判定库并按条件推进审核状态。

    Args:
        database_ids: 已展开的完整归属集合，含末级库及其各级上级。

    Returns:
        动作标识：approved / classified / low_confidence；写库失败抛异常由调用方计数。
    """
    # 1. 归属库整体替换，保证与模型判定结果一致；上级节点一并写入，与页面勾选规则一致
    CorpusDatabase.query.filter(CorpusDatabase.corpus_id == corpus.id).delete(synchronize_session=False)
    for database_id in database_ids:
        db.session.add(CorpusDatabase(corpus_id=corpus.id, database_id=database_id))

    action = 'classified'
    if auto_approve and confidence >= min_confidence:
        corpus.status = CorpusStatus.APPROVE_EXAMINE.value
        corpus.reviewer_id = reviewer_id
        # 索引任务模块仅在包含该机制的分支存在，缺失时只推进审核状态
        if enqueue_corpus_index is not None:
            enqueue_corpus_index(db.session, corpus)
        action = 'approved'
    elif confidence < min_confidence:
        action = 'low_confidence'
    db.session.commit()
    return action


def _select_corpus_ids(filters, limit, only_unassigned):
    """按审核页筛选条件取出本轮要处理的文献主键。"""
    query = db.session.query(Corpus.id).filter(*filters)
    # 1. 默认跳过已关联库的文献，避免重复判定覆盖人工结果
    if only_unassigned:
        query = query.filter(
            ~db.session.query(CorpusDatabase.id).filter(CorpusDatabase.corpus_id == Corpus.id).exists()
        )
    query = query.order_by(Corpus.id)
    if limit and limit > 0:
        query = query.limit(limit)
    return [row[0] for row in query.all()]


def _update_state(state, **changes):
    """合并进度字段并落盘。"""
    state.update(changes)
    state['updated_at'] = _now_text()
    _write_state(state)


def _run_task(app, state, filters, options):
    """后台任务主体：分批分类、串行写库、持续刷新进度。"""
    with app.app_context():
        try:
            model = options['model']
            auto_approve = options['auto_approve']
            min_confidence = options['min_confidence']
            reviewer_id = options['reviewer_id']
            client = _client()
            tree_text, selectable, id_to_name, parent_map = load_database_tree()
            if not selectable:
                _update_state(state, status='error', message='未读取到可选的末级分类库')
                return

            corpus_ids = _select_corpus_ids(filters, options['limit'], options['only_unassigned'])
            _update_state(state, total=len(corpus_ids))
            if not corpus_ids:
                _update_state(state, status='finished', message='没有符合条件的待处理文献')
                return

            # 1. 每轮取 BATCH_SIZE*CONCURRENCY 条，先并发分类再串行写库
            step = BATCH_SIZE * CONCURRENCY
            consecutive_model_failure = 0
            for start in range(0, len(corpus_ids), step):
                # 停止信号由外部写入进度文件，因此每轮都要重新读取
                latest = _load_state(state['task_id']) or {}
                if latest.get('status') in ('stopping', 'stopped'):
                    _update_state(state, status='stopped', message='已按请求停止')
                    return
                window = corpus_ids[start:start + step]
                corpora = (
                    db.session.query(Corpus)
                    .filter(Corpus.id.in_(window), Corpus.status == CorpusStatus.WAITING_EXAMINE.value)
                    .order_by(Corpus.id)
                    .all()
                )
                # 单条元数据异常只跳过该条，不中断整批任务
                payloads = []
                for corpus in corpora:
                    try:
                        payloads.append(_document_payload(corpus, len(payloads) + 1))
                    except Exception as error:
                        LOGGER.warning('组装文献元数据失败 corpus=%s err=%s', corpus.id, error)
                        state['failed'] += 1
                        state['processed'] += 1
                if not payloads:
                    continue
                _update_state(state, message=f'正在判定第 {start + 1}—{start + len(payloads)} 条')

                groups = [payloads[i:i + BATCH_SIZE] for i in range(0, len(payloads), BATCH_SIZE)]
                result_map = {}
                pool = ThreadPoolExecutor(max_workers=CONCURRENCY)
                try:
                    futures = [
                        pool.submit(_classify_batch_retry, client, model, tree_text, group)
                        for group in groups
                    ]
                    for future in futures:
                        try:
                            # 单组超时不等待其余组，避免一组无响应把整轮拖住
                            result_map.update(future.result(timeout=BATCH_TIMEOUT))
                        except Exception as error:
                            LOGGER.warning('分批归类超时或失败，转逐条重试: %s', error)
                finally:
                    # 不 join：卡住的调用由模型超时自行结束，主流程继续推进
                    pool.shutdown(wait=False)

                # 2. 整批失败的条目再逐条补判；整轮一条都没返回时按模型故障处理，不再逐条重试
                if result_map:
                    for payload in payloads:
                        if payload['corpus_index'] not in result_map:
                            single = _fallback_single(client, model, tree_text, payload)
                            if single:
                                result_map[payload['corpus_index']] = single

                # 3. 串行写库，逐条提交，单条异常不影响后续文献
                state.setdefault('items', [])
                model_missing = 0
                last_flush = time.time()
                for payload in payloads:
                    corpus = next((item for item in corpora if item.id == payload['corpus_id']), None)
                    if corpus is None:
                        continue
                    judged = result_map.get(payload['corpus_index'])
                    if judged is None:
                        model_missing += 1
                    raw_ids = (judged or {}).get('database_ids', [])
                    valid_ids = [db_id for db_id in raw_ids if db_id in selectable]
                    if not valid_ids:
                        # 模型落到父级节点时直接采用该节点兜底，避免整条被误判为无库可用
                        valid_ids = [db_id for db_id in raw_ids if db_id in parent_map][:1]
                    # 归属库仍最多两个，与既有录入习惯保持一致
                    valid_ids = list(dict.fromkeys(valid_ids))[:2]
                    title = (corpus.title_cn or corpus.title_en or f'文献 {corpus.id}')[:40]
                    full_ids = []
                    if not valid_ids:
                        # 模型没给出可用库记「待人工」，与写库异常导致的失败分开统计
                        if judged is None:
                            state['failed'] += 1
                            action = 'failed'
                        else:
                            state['pending'] += 1
                            action = 'pending'
                        names = []
                    else:
                        try:
                            # 勾选末级库时上级同步挂上，与页面多选树的勾选结果一致
                            full_ids = expand_with_ancestors(valid_ids, parent_map)
                            action = _save_result(
                                corpus, full_ids, auto_approve, min_confidence,
                                (judged or {}).get('confidence', 0.0), reviewer_id,
                            )
                            names = [id_to_name.get(db_id, str(db_id)) for db_id in valid_ids]
                            confidence = (judged or {}).get('confidence', 0.0)
                            # 低置信度单独计数，无论是否已通过，都便于事后抽检
                            if confidence < REVIEW_HINT_CONFIDENCE:
                                state['low_confidence'] += 1
                            if action == 'approved':
                                state['approved'] += 1
                            elif action == 'low_confidence':
                                pass
                            else:
                                state['classified'] += 1
                        except Exception as error:
                            db.session.rollback()
                            LOGGER.error('写回归类结果失败 corpus=%s err=%s', corpus.id, error)
                            state['failed'] += 1
                            action, names = 'failed', []
                    state['processed'] += 1
                    state['items'].insert(0, {
                        'corpus_id': corpus.id,
                        'title': title,
                        'databases': names,
                        'linked_count': len(full_ids),
                        'confidence': round((judged or {}).get('confidence', 0.0), 2),
                        'action': action,
                        'reason': (judged or {}).get('reason', '') if judged else '模型未返回该条结果',
                    })
                    del state['items'][RECENT_ITEM_LIMIT:]
                    # 按时间节流写盘，让前端能看到进度推进，而不是整批结束才跳一次
                    if time.time() - last_flush >= 1.5:
                        _update_state(state, message=f'已处理 {state["processed"]}/{state["total"]} 条')
                        last_flush = time.time()
                _update_state(state)
                # 模型侧整轮都无返回时快速失败，避免把上万条逐一记成失败
                if payloads and model_missing >= len(payloads):
                    consecutive_model_failure += 1
                    if consecutive_model_failure >= MAX_CONSECUTIVE_MODEL_FAILURE:
                        _update_state(
                            state, status='error',
                            message='模型连续无响应，任务已终止：请检查 QWEN_API_KEY 以及服务器到 dashscope 的网络连通性',
                        )
                        return
                else:
                    consecutive_model_failure = 0
                db.session.remove()

            _update_state(state, status='finished', message='处理完成')
        except Exception as error:
            LOGGER.exception('AI 归类任务异常')
            db.session.rollback()
            _update_state(state, status='error', message=str(error)[:200])


def _classify_batch_retry(client, model, tree_text, group):
    """带重试与退避的批量分类，全部失败时返回空映射交由上层逐条兜底。"""
    last_error = None
    for attempt in range(MAX_RETRY):
        try:
            return _classify_batch(client, model, tree_text, group)
        except Exception as error:
            last_error = error
            if attempt + 1 < MAX_RETRY:
                # 限流或瞬时抖动时短暂退避，再接下一轮
                time.sleep(2 * (attempt + 1))
    LOGGER.warning('批量归类连续失败: %s', last_error)
    return {}


def build_scope_filters(scope, current_user_id, assigned_reviewer_id, recovery_contributor_id):
    """复用审核列表的筛选与分流规则，构造待处理文献的查询条件。

    Args:
        scope: 前端传入的筛选条件，含 title / language / author。
        current_user_id: 当前操作人 id，用于恢复语料的分流。
        assigned_reviewer_id: 被预分配审核的用户 id。
        recovery_contributor_id: 历史恢复语料的提交者 id。

    Returns:
        供查询使用的条件列表。
    """
    filters = [
        Corpus.status == CorpusStatus.WAITING_EXAMINE.value,
        Corpus.is_deleted == 0,
        Corpus.contributor == recovery_contributor_id,
    ]
    title = (scope.get('title') or '').strip()
    if title:
        filters.append(or_(Corpus.title_cn.like(f'%{title}%'), Corpus.title_en.like(f'%{title}%')))
    language = (scope.get('language') or '').strip()
    if language:
        filters.append(Corpus.language == language)
    author = (scope.get('author') or '').strip()
    if author:
        filters.append(or_(
            Corpus.author.like(f'%{author}%'),
            Corpus.translator.like(f'%{author}%'),
            Corpus.editor.like(f'%{author}%'),
        ))
    # 1. 与 examine_recovery 一致：预分配给专人的记录只由该专人处理
    if current_user_id == assigned_reviewer_id:
        filters.append(Corpus.reviewer_id == assigned_reviewer_id)
    else:
        filters.append(Corpus.reviewer_id.is_(None))
    return filters


def start_task(scope, options, reviewer_id, assigned_reviewer_id, recovery_contributor_id):
    """校验并启动后台归类任务。

    Args:
        scope: 页面筛选条件。
        options: limit / auto_approve / min_confidence / model / only_unassigned。
        reviewer_id: 审核人 id，通过时写入 reviewer_id。
        assigned_reviewer_id: 预分配审核人 id，用于分流。
        recovery_contributor_id: 历史恢复语料的提交者 id。

    Returns:
        任务状态字典。

    Raises:
        RuntimeError: 已有任务在运行或模型凭据缺失。
    """
    running = any_active_task()
    if running:
        same_owner = running.get('owner_id') == reviewer_id
        raise RuntimeError(
            f'已有归类任务在运行（{running.get("processed")}/{running.get("total")}），请先等待或停止'
            if same_owner else '其他审核人正在运行归类任务，请稍后再试'
        )
    _client()  # 提前暴露凭据缺失，避免任务启动后立刻失败

    filters = build_scope_filters(scope, reviewer_id, assigned_reviewer_id, recovery_contributor_id)
    model = options.get('model') or DEFAULT_MODEL
    state = {
        'task_id': uuid.uuid4().hex[:12],
        'owner_id': reviewer_id,
        'status': 'running',
        'total': 0,
        'processed': 0,
        'approved': 0,
        'classified': 0,
        'low_confidence': 0,
        'pending': 0,
        'failed': 0,
        'items': [],
        'message': '任务已启动',
        'model': model,
        'auto_approve': options['auto_approve'],
        'min_confidence': options['min_confidence'],
        'limit': options['limit'],
        'only_unassigned': options['only_unassigned'],
        'started_at': _now_text(),
        'updated_at': _now_text(),
    }
    _write_state(state)
    app = current_app._get_current_object()
    thread = threading.Thread(target=_run_task, args=(app, state, filters, {
        'model': model,
        'auto_approve': options['auto_approve'],
        'min_confidence': options['min_confidence'],
        'reviewer_id': reviewer_id,
        'limit': options['limit'],
        'only_unassigned': options['only_unassigned'],
    }), daemon=True)
    thread.start()
    return state


def request_stop(task_id=None, owner_id=None):
    """请求停止任务，返回更新后的状态。

    Args:
        task_id: 目标任务号；为空时取该用户最近一次任务。
        owner_id: 当前用户 id，用于限制只能停止自己发起的任务。
    """
    state = _load_state(task_id) if task_id else latest_task_state(owner_id)
    if not state:
        return None
    if owner_id is not None and state.get('owner_id') != owner_id:
        return None
    if state.get('status') == 'running':
        _update_state(state, status='stopping', message='正在停止')
    return state
