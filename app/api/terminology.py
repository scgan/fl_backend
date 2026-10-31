"""术语库接口：术语提交与检索。

提供两种提交方式：
- POST /terminology/upload：既兼容旧方式（前端已完成提取，只传关联文献 corpus_id），
  也支持新契约（前端用 TERM_ATTACHMENT_BACKEND_READY=true 直传附件原始文件，
  字段名 source_attachment / definition_cn_attachment / definition_en_attachment），
  附件原始文件由后台线程完成「大模型提取 → 文献入库 → 关联」，处理状态记在
  terminology.attachment_process（JSON），可通过 /terminology/get 查看。
- POST /terminology/upload_async：与上面新契约等价的独立入口（术语+附件文件一起提交），
  保留兼容。
"""

import os
import re
import threading
import tempfile
from datetime import datetime
import json

from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.datastructures import FileStorage

from app.exts import db
from app.models import Corpus, CorpusCategory, CorpusDatabase, Database, Terminology, User
from app.utils.decorators import permission_required
from app.utils.enums import CorpusStatus
from app.utils.utils import get_file
from app.api.collector import (
    extract_year,
    find_duplicate_title_cn,
    process_file_upload,
    robust_extract_pdf_info,
    setup_qwen_client,
)

bp = Blueprint('terminology', __name__)

# 三个附件字段对应的关联文献 id（multipart 表单字段名）
TERM_ATTACHMENT_CORPUS_FIELDS = (
    'source_attachment_corpus_id',
    'definition_cn_attachment_corpus_id',
    'definition_en_attachment_corpus_id',
)

# 附件文件字段名（upload_async 用，与前端表单一致）
TERM_ATTACHMENT_FILE_FIELDS = (
    'source_attachment',
    'definition_cn_attachment',
    'definition_en_attachment',
)

REQUIRED_FIELDS = ('name', 'category', 'source_mla', 'definition_cn', 'definition_en')

# 附件大模型提取的必填字段：任一缺失则文献转草稿状态（DRAFT），全部齐全才进待审核。
# (提取JSON键, 展示名)；其中 category（文献类型）还要求能匹配到文献大类，匹配不到视同缺失。
_REQUIRED_EXTRACT_FIELDS = [
    ('title_cn', '中文标题'),
    ('title_en', '英文标题'),
    ('language', '语言'),
    ('author', '作者'),
    ('publish_time', '出版时间'),
    ('source', '来源'),
    ('category', '文献类型'),
]


def _attachment_info(corpus):
    """取语料首个附件的展示名与下载链接（相对 /api 前缀，走公共文件接口）。

    Args:
        corpus: Corpus 模型实例（可能为 None）。

    Returns:
        (展示文件名, 下载链接)；无附件或解析失败时返回 (None, None)。
    """
    if not corpus or not corpus.attachment:
        return None, None
    try:
        attachments = json.loads(corpus.attachment)
        item = attachments[0] if isinstance(attachments, list) and attachments else None
    except (TypeError, ValueError):
        current_app.logger.warning(f"解析语料附件失败 corpus_id={corpus.id}")
        return None, None
    if not isinstance(item, dict):
        return None, None

    file_path = item.get('file_path') or ''
    display_name = item.get('original_filename') or (
        file_path.rsplit('/', 1)[-1] if '/' in file_path else file_path
    )
    # 术语附件多为待审核状态，公共文件接口（仅放行审核通过语料）打不开；
    # 统一走需登录的术语附件接口。
    file_url = f"/terminology/attachment_file?corpus_id={corpus.id}"
    return display_name, file_url


def _attachment_payload(item):
    """附件记录 {corpus_id, name} → 前端期望的 {name, url, corpus_id}。"""
    corpus_id = item.get('corpus_id') if isinstance(item, dict) else None
    stored_name = (item.get('name') if isinstance(item, dict) else '') or ''
    corpus = Corpus.query.filter(Corpus.id == corpus_id).first() if corpus_id else None
    display_name, file_url = _attachment_info(corpus)
    url = None if (corpus and corpus.is_deleted) else file_url
    return {
        'name': display_name or stored_name,
        'url': url or '',
        'corpus_id': corpus_id,
    }


def _definition_entries_payload(raw, prefix, row, process):
    """定义条目返回结构：[{text, sources, attachments, process}]；无 JSON 时兼容旧扁平列。"""
    doc = _load_json_doc(raw, [])
    entries = []
    if isinstance(doc, list):
        for index, item in enumerate(doc):
            if not isinstance(item, dict):
                continue
            field = prefix + '_attachment' if index == 0 else f'{prefix}_attachment_{index}'
            entries.append({
                'text': item.get('text') or '',
                'sources': [
                    str(mla).strip() for mla in (item.get('mlas') or []) if str(mla or '').strip()
                ],
                'attachments': [
                    _attachment_payload(att) for att in (item.get('attachments') or [])
                ],
                'process': process.get(field),
            })
    if entries:
        return entries
    # 旧数据：单条扁平列
    legacy_corpus_id = getattr(row, f'{prefix}_attachment_corpus_id', None)
    return [{
        'text': getattr(row, prefix) or '',
        'sources': _split_mla(getattr(row, f'{prefix}_source')),
        'attachments': (
            [_attachment_payload({'corpus_id': legacy_corpus_id})] if legacy_corpus_id else []
        ),
        'process': process.get(f'{prefix}_attachment'),
    }]


def _term_multi_payload(row, process):
    """多条目返回字段：sources / source_attachments / definitions_cn(en)。

    结构与前端 normalizeTermEntries 的约定一致：多条 MLA 出处、中英文多条定义，
    每条附件给出 {name, url} 与后台处理状态；没有多条目数据时回退旧单值列。
    """
    sources_doc = _load_json_doc(row.sources_json, {'mlas': [], 'attachments': []})
    if not isinstance(sources_doc, dict):
        sources_doc = {'mlas': [], 'attachments': []}
    mlas = [
        str(mla).strip() for mla in (sources_doc.get('mlas') or []) if str(mla or '').strip()
    ]
    if not mlas:
        mlas = _split_mla(row.source_mla)
    attachments = list(sources_doc.get('attachments') or [])
    if not attachments and row.source_attachment_corpus_id:
        attachments = [{'corpus_id': row.source_attachment_corpus_id}]
    return {
        'sources': mlas,
        'source_attachments': [_attachment_payload(item) for item in attachments],
        'source_attachment_process': process.get('source_attachment'),
        'definitions_cn': _definition_entries_payload(
            row.definitions_cn_json, 'definition_cn', row, process
        ),
        'definitions_en': _definition_entries_payload(
            row.definitions_en_json, 'definition_en', row, process
        ),
    }


# ---------------- 多条目结构（多条 MLA / 多条定义 / 每处多个附件）----------------
# 存储形态（JSON 文本列，旧列继续镜像第 1 条，老接口不受影响）：
#   sources_json        {"mlas": ["MLA…"], "attachments": [{"corpus_id": 1, "name": "a.pdf"}]}
#   definitions_cn_json [{"text": "…", "mlas": ["MLA…"], "attachments": [...]}]
#   definitions_en_json 同上
# 附件组字段名（与前端 buildTermFormData 约定一致，每组可多个文件）：
#   术语出处模块     source_attachment
#   第 1 个定义      definition_cn_attachment
#   第 i 个定义(i≥1) definition_cn_attachment_{i}
# 旧客户端只发这 3 个固定字段（各 1 个文件），同一套代码天然兼容。


def _split_mla(value):
    """把旧单字段 MLA 文本按换行拆成多条。"""
    return [line.strip() for line in str(value or '').split('\n') if line.strip()]


# 旧单值附件列 → 附件组字段名（改绑时定位到多条目结构的对应位置）
_LEGACY_CORPUS_TO_GROUP = {
    'source_attachment_corpus_id': 'source_attachment',
    'definition_cn_attachment_corpus_id': 'definition_cn_attachment',
    'definition_en_attachment_corpus_id': 'definition_en_attachment',
}


def _load_json_doc(raw, default):
    """解析 JSON 文本列；空值或非法内容返回默认结构的副本。"""
    if isinstance(raw, (dict, list)):
        return raw
    if raw in (None, ''):
        return json.loads(json.dumps(default))
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return json.loads(json.dumps(default))
    if not isinstance(parsed, type(default)):
        return json.loads(json.dumps(default))
    return parsed


def _json_list(value):
    """解析表单里的 JSON 数组字段（非法时返回空列表）。"""
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _build_sources_doc(data):
    """术语出处：新字段 sources(JSON) 优先，缺失时回退旧字段 source_mla。"""
    mlas = [str(item).strip() for item in _json_list(data.get('sources')) if str(item or '').strip()]
    if not mlas:
        mlas = _split_mla(data.get('source_mla'))
    return {'mlas': mlas, 'attachments': []}


def _build_definitions_doc(data, key):
    """中/英文定义多条目：definitions_cn/en(JSON) 优先，缺失时回退旧扁平字段。"""
    entries = []
    for item in _json_list(data.get('definitions_cn' if key == 'definition_cn' else 'definitions_en')):
        if not isinstance(item, dict):
            continue
        raw_sources = item.get('sources')
        if isinstance(raw_sources, str):
            mlas = _split_mla(raw_sources)
        else:
            mlas = [str(mla).strip() for mla in (raw_sources or []) if str(mla or '').strip()]
        entries.append({
            'text': str(item.get('text') or '').strip(),
            'mlas': mlas,
            'attachments': [],
        })
    if not entries:
        entries = [{
            'text': str(data.get(key) or '').strip(),
            'mlas': _split_mla(data.get(f'{key}_source')),
            'attachments': [],
        }]
    return entries


def _definition_counts(data):
    """定义条目数：以 JSON 长度为准，旧客户端按 *_count（缺省 1）。"""
    counts = {}
    for prefix, json_key in (('definition_cn', 'definitions_cn'), ('definition_en', 'definitions_en')):
        entries = _json_list(data.get(json_key))
        if entries:
            counts[prefix] = len(entries)
            continue
        try:
            counts[prefix] = max(int(data.get(f'{prefix}_count') or 1), 1)
        except (TypeError, ValueError):
            counts[prefix] = 1
    return counts


def _attachment_group_fields(counts):
    """本次请求涉及的附件组字段名（出处模块 + 每个定义，下标与 JSON 条目对齐）。"""
    fields = ['source_attachment']
    for prefix in ('definition_cn', 'definition_en'):
        for index in range(max(int(counts.get(prefix, 1)), 1)):
            fields.append(prefix + '_attachment' if index == 0 else f'{prefix}_attachment_{index}')
    return fields


def _attachment_slot(field):
    """附件组字段名 → (JSON 列名, 定义条目下标)；出处模块下标为 None。"""
    if field == 'source_attachment':
        return 'sources_json', None
    for column, prefix in (
        ('definitions_cn_json', 'definition_cn'),
        ('definitions_en_json', 'definition_en'),
    ):
        if field == f'{prefix}_attachment':
            return column, 0
        if field.startswith(f'{prefix}_attachment_'):
            tail = field[len(f'{prefix}_attachment_'):]
            if tail.isdigit():
                return column, int(tail)
    return None, None


def _mirror_column(field):
    """附件组字段名 → 旧的单值 corpus_id 列（只镜像第 1 个定义的附件）。"""
    if field == 'source_attachment':
        return 'source_attachment_corpus_id'
    if field == 'definition_cn_attachment':
        return 'definition_cn_attachment_corpus_id'
    if field == 'definition_en_attachment':
        return 'definition_en_attachment_corpus_id'
    return None


def _attach_corpus_to_docs(corpus_id, legacy_field, sources_doc, definitions_cn_doc, definitions_en_doc):
    """把关联文献 id（旧 *_corpus_id 字段）记入对应条目的附件列表，已存在则跳过。"""
    group_field = _LEGACY_CORPUS_TO_GROUP.get(legacy_field)
    if not group_field:
        return
    column, slot = _attachment_slot(group_field)
    if column == 'sources_json':
        attachments = sources_doc.setdefault('attachments', [])
    else:
        doc = definitions_cn_doc if column == 'definitions_cn_json' else definitions_en_doc
        while len(doc) <= slot:
            doc.append({'text': '', 'mlas': [], 'attachments': []})
        entry = doc[slot]
        if not isinstance(entry, dict):
            entry = {'text': '', 'mlas': [], 'attachments': []}
            doc[slot] = entry
        attachments = entry.setdefault('attachments', [])
    if any(
        isinstance(item, dict) and item.get('corpus_id') == corpus_id for item in attachments
    ):
        return
    attachments.append({'corpus_id': corpus_id})


def _merge_multi_docs(term, data):
    """修改术语时合并多条目结构：按下标替换文本/MLA，保留各条目已有附件。

    Args:
        term: 现有术语记录。
        data: 请求表单。

    Returns:
        需要写入的列与兼容镜像字段；本次没传多条目字段时返回空 dict。
    """
    updates = {}
    if data.get('sources') is not None:
        sources_doc = _build_sources_doc(data)
        existing = _load_json_doc(term.sources_json, {'mlas': [], 'attachments': []})
        sources_doc['attachments'] = list((existing or {}).get('attachments') or [])
        updates['sources_json'] = json.dumps(sources_doc, ensure_ascii=False)
        if sources_doc['mlas']:
            updates['source_mla'] = sources_doc['mlas'][0]

    for key, json_key, column in (
        ('definition_cn', 'definitions_cn', 'definitions_cn_json'),
        ('definition_en', 'definitions_en', 'definitions_en_json'),
    ):
        if data.get(json_key) is None:
            continue
        entries = _build_definitions_doc(data, key)
        existing = _load_json_doc(getattr(term, column), [])
        if not isinstance(existing, list):
            existing = []
        for index, entry in enumerate(entries):
            old = existing[index] if index < len(existing) else None
            entry['attachments'] = list((old or {}).get('attachments') or [])
        updates[column] = json.dumps(entries, ensure_ascii=False)
        if entries and entries[0]['text']:
            updates[key] = entries[0]['text']
        if entries and entries[0]['mlas']:
            updates[f'{key}_source'] = entries[0]['mlas'][0]
    return updates


def _merge_rebound_corpora(term, rebind):
    """旧接口 *_corpus_id 改绑：把文献 id 记入对应条目并回写三列 JSON。"""
    sources_doc = _load_json_doc(term.sources_json, {'mlas': [], 'attachments': []})
    if not isinstance(sources_doc, dict):
        sources_doc = {'mlas': [], 'attachments': []}
    definitions_cn_doc = _load_json_doc(term.definitions_cn_json, [])
    definitions_en_doc = _load_json_doc(term.definitions_en_json, [])
    definitions_cn_doc = definitions_cn_doc if isinstance(definitions_cn_doc, list) else []
    definitions_en_doc = definitions_en_doc if isinstance(definitions_en_doc, list) else []
    for legacy_field, corpus_id in rebind.items():
        _attach_corpus_to_docs(
            corpus_id, legacy_field, sources_doc, definitions_cn_doc, definitions_en_doc
        )
    return {
        'sources_json': json.dumps(sources_doc, ensure_ascii=False),
        'definitions_cn_json': json.dumps(definitions_cn_doc, ensure_ascii=False),
        'definitions_en_json': json.dumps(definitions_en_doc, ensure_ascii=False),
    }


@bp.route('/upload', methods=['POST'])
@permission_required(['COLLECTOR'])
def terminology_upload():
    """提交一条术语（附件已作为文献入库，这里只接收关联的 corpus_id）。

    Returns:
        新建术语的 id。
    """
    data = request.form
    values = {field: (data.get(field) or '').strip() for field in REQUIRED_FIELDS}
    for field in REQUIRED_FIELDS:
        if not values[field]:
            return jsonify({"code": 400, "error": "术语名、所属类别、术语出处与中英文定义不能为空"}), 400

    try:
        category_id = int(values['category'])
    except ValueError:
        return jsonify({"code": 400, "error": "所属类别无效"}), 400

    user = g.current_user
    saved_files = {}  # 附件文件（字段名 -> (落盘路径, 原始文件名)），成功提交后交给后台线程

    try:
        # 1. 所属类别必须是一个真实存在的大类（database）
        database = Database.query.filter(
            Database.id == category_id, Database.is_deleted == 0
        ).first()
        if not database:
            return jsonify({"code": 404, "error": "无该所属类别"}), 404

        # 2. 同一大类下不允许重复术语名
        duplicate = Terminology.query.filter(
            Terminology.name == values['name'],
            Terminology.category == category_id,
            Terminology.is_deleted == 0,
        ).first()
        if duplicate:
            return jsonify({"code": 400, "error": "该类别下已存在同名术语"}), 400

        # 3. 多条目结构：出处多条 MLA + 中英文多条定义（旧扁平字段作为第 1 条镜像）
        counts = _definition_counts(data)
        sources_doc = _build_sources_doc(data)
        definitions_cn_doc = _build_definitions_doc(data, 'definition_cn')
        definitions_en_doc = _build_definitions_doc(data, 'definition_en')

        # 4. 关联文献必须真实存在且未被删除；直接记入对应条目的附件列表
        corpus_columns = {}
        for field in TERM_ATTACHMENT_CORPUS_FIELDS:
            raw = data.get(field)
            if raw in (None, ''):
                continue
            try:
                corpus_id = int(raw)
            except ValueError:
                return jsonify({"code": 400, "error": f"{field} 无效"}), 400
            corpus = Corpus.query.filter(
                Corpus.id == corpus_id, Corpus.is_deleted == 0
            ).first()
            if not corpus:
                return jsonify({"code": 404, "error": f"关联文献不存在（id={corpus_id}）"}), 404
            corpus_columns[field] = corpus_id
            _attach_corpus_to_docs(
                corpus_id, field, sources_doc, definitions_cn_doc, definitions_en_doc
            )

        # 5. 可选：附件原始文件随表单一起提交（TERM_ATTACHMENT_BACKEND_READY=true 的新契约），
        #    后台线程完成「大模型提取 → 文献入库 → 关联」；旧前端只传 corpus_id，此处为空
        saved_files = _collect_attachment_files(counts)

        reviewer_id = db.session.query(User.leader_id).filter(
            User.id == user.id, User.is_deleted == 0
        ).scalar()

        terminology = Terminology(
            name=values['name'],
            category=category_id,
            source_mla=values['source_mla'],
            definition_cn=values['definition_cn'],
            definition_cn_source=data.get('definition_cn_source'),
            definition_en=values['definition_en'],
            definition_en_source=data.get('definition_en_source'),
            sources_json=json.dumps(sources_doc, ensure_ascii=False),
            definitions_cn_json=json.dumps(definitions_cn_doc, ensure_ascii=False),
            definitions_en_json=json.dumps(definitions_en_doc, ensure_ascii=False),
            contributor=user.id,
            status=1,  # 待审核，与文献上传一致
            submit_time=datetime.now(),
            attachment_process=(
                json.dumps(_processing_state(saved_files), ensure_ascii=False)
                if saved_files else None
            ),
            **corpus_columns,
        )
        db.session.add(terminology)
        db.session.commit()

        if saved_files:
            _spawn_attachment_workers(
                current_app._get_current_object(), terminology.id, saved_files,
                user.username, user.id, reviewer_id, category_id,
            )
        message = (
            "提交成功，附件正在后台识别，完成后可在术语查找中查看"
            if saved_files else "提交成功"
        )
        return jsonify({
            "code": 200,
            "success": message,
            "terminology_id": terminology.id,
            "processing": list(saved_files.keys()),
        }), 200
    except SQLAlchemyError:
        db.session.rollback()
        # 术语没建成，清掉已落盘的附件文件
        _discard_saved_files(saved_files)
        current_app.logger.error("术语提交数据库失败", exc_info=True)
        return jsonify({"code": 503, "error": "系统繁忙，请稍后重试"}), 503


# ---------------- 附件异步识别（后台线程） ----------------

# 不同附件线程会并发更新同一术语的状态 JSON，加锁避免互相覆盖
_PROCESS_STATE_LOCK = threading.Lock()


def _load_process_state(term):
    """读取术语附件处理状态 JSON；缺失、损坏或非对象值降级为空字典。"""
    try:
        state = json.loads(term.attachment_process) if term.attachment_process else {}
    except (TypeError, ValueError):
        state = {}
    return state if isinstance(state, dict) else {}


def _update_process_state(app, term_id, field, entry):
    """后台线程安全地更新术语的附件处理状态（带锁 + 独立事务）。"""
    with _PROCESS_STATE_LOCK:
        with app.app_context():
            try:
                term = Terminology.query.filter(
                    Terminology.id == term_id, Terminology.is_deleted == 0
                ).first()
                if not term:
                    return
                state = _load_process_state(term)
                state[field] = entry
                term.attachment_process = json.dumps(state, ensure_ascii=False)
                db.session.commit()
            except Exception:
                db.session.rollback()
                app.logger.error(
                    f"更新附件处理状态失败 term_id={term_id} field={field}", exc_info=True
                )


def _process_term_attachment(
    app, term_id, field, file_path, original_filename, username, user_id,
    reviewer_id, database_id, group_total=1,
):
    """在后台线程提取单个术语附件题录，复用或创建文献，再回填术语关联及组进度。

    Args:
        app: 可在线程建立独立上下文的 Flask 应用。
        term_id: 附件所属术语 ID。
        field: 出处或具体定义条目的附件组字段名。
        file_path: 请求阶段保存的本地 PDF 副本，退出时清理。
        original_filename: 上传原名，回填展示和生成文献文件名时使用。
        username: 文件服务中的归属名称。
        user_id: 文献贡献者及题名去重所使用的用户 ID。
        reviewer_id: 文献审核人 ID。
        database_id: 为新建文献挂接的主题数据库 ID。
        group_total: 同组本次提交的附件总数，供终态聚合。

    提取失败仍可创建待补全草稿；组状态使用旧名称 editing，但文献实际写 DRAFT。
    新文献先提交再上传，上传失败尝试删除主记录补偿，不保证远程文件和关系原子回滚。
    """
    # 1. 在独立线程上下文提取题录，提取失败保留原件及缺失原因用于补全草稿
    corpus_id = None
    reused = False
    extraction_failed = False
    extract_error = None
    try:
        with app.app_context():
            # 1. 大模型提取（内部带 MD5 缓存与 429 退避重试）
            api_key = os.getenv('QWEN_API_KEY')
            if not api_key:
                raise RuntimeError("未配置通义千问API密钥")
            client = setup_qwen_client(api_key)
            result = robust_extract_pdf_info(client, file_path)
            # 提取失败不放弃：仍建文献兜底（标题用文件名），必填字段视为全部缺失，
            # 文献转草稿状态，学生可在「文献管理」里编辑补全后再提交审核
            extraction_failed = not result.get("success")
            extract_error = result.get("error") or "大模型提取失败"
            if extraction_failed:
                app.logger.warning(
                    f"术语附件提取失败，转EDITING文献 term_id={term_id} field={field}: {extract_error}"
                )

            data = result.get("extracted_data") if not extraction_failed else {}
            data = data if isinstance(data, dict) else {}
            # 与文献上传页一致：提取结果里的中英文标题、ISBN 顺序相反，这里对调一次
            # 沿用提取结果到采集表单的既有映射：模型 title_en 写入文献 title_cn，反向亦然。
            # 字段名带历史语言后缀，不能仅按 cn/en 命名推断当前界面的题名语义。
            title_cn = (data.get('title_en') or '').strip()  # 原文标题
            title_en = (data.get('title_cn') or '').strip()  # 译文标题
            # 提取完整度：必填字段任一缺失（如文献类型未识别）则文献转草稿，
            # 全部齐全才进待审核。中文标题按入库对调规则取值，须在文件名兜底前判定。
            extracted_values = {'title_cn': title_cn, 'title_en': title_en}
            missing_fields = [
                label
                for key, label in _REQUIRED_EXTRACT_FIELDS
                if not str(extracted_values.get(key, data.get(key)) or '').strip()
            ]
            if not title_cn:
                base = re.sub(r'[/\x00\r\n]', '_', os.path.splitext(original_filename)[0])
                title_cn = base.strip() or '未命名文献'

            # 2. 同标题文献直接复用（与前端原复用逻辑一致，不重复上传文件）；
            # 提取失败时标题只是文件名兜底，不做复用判断
            # 2. 成功提取后按既有题名规则尝试复用；不命中时分配文献编号、主题关联和审核状态
            duplicate = (
                find_duplicate_title_cn(title_cn, user_id) if not extraction_failed else None
            )
            if duplicate:
                corpus_id = duplicate.id
                reused = True
                app.logger.info(
                    f"术语附件复用已有文献 term_id={term_id} field={field} corpus_id={corpus_id}"
                )
            else:
                # 3. 文献类型：按提取结果的类型名称匹配 CorpusCategory，匹配不到则置空（审核时可补）
                category = None
                category_name = (data.get('category') or '').strip()
                if category_name:
                    category = CorpusCategory.query.filter(
                        CorpusCategory.category_name == category_name,
                        CorpusCategory.is_deleted == 0,
                    ).first()

                number = None
                if category:
                    category = CorpusCategory.query.filter(
                        CorpusCategory.id == category.id
                    ).with_for_update(of=CorpusCategory).first()
                    category.number = (category.number or 0) + 1
                    number = f"{category.code}{category.number}"

                volume = int(data['volume']) if str(data.get('volume') or '').strip().isdigit() else None
                issue = int(data['issue']) if str(data.get('issue') or '').strip().isdigit() else None

                corpus = Corpus(
                    language=data.get('language'), author=data.get('author'),
                    translator=data.get('translator'), editor=data.get('editor'),
                    publish_time=extract_year(data.get('publish_time')),
                    ISSN=data.get('ISBN'), source=data.get('source'),
                    category_id=category.id if category else None,
                    abstract_cn=data.get('abstract_cn'), abstract_en=data.get('abstract_en'),
                    contributor=user_id, title_cn=title_cn, title_en=title_en,
                    number=number, volume=volume, issue=issue,
                    reviewer_id=reviewer_id, submit_time=datetime.now(),
                )
                # 状态判定：必填字段任一缺失转草稿，全部齐全才进待审核；
                # 文献类型提取到了但未匹配到文献大类，同样视同缺失。
                if extraction_failed:
                    missing_fields = [label for _, label in _REQUIRED_EXTRACT_FIELDS]
                if category is None and '文献类型' not in missing_fields:
                    missing_fields.append('文献类型（未匹配到大类）')
                corpus.status = (
                    CorpusStatus.WAITING_EXAMINE.value
                    if not missing_fields
                    else CorpusStatus.DRAFT.value
                )
                db.session.add(corpus)
                db.session.flush()

                # 所属大类取术语的所属类别
                db.session.add(CorpusDatabase(corpus_id=corpus.id, database_id=database_id))

                for key in ('keyword_cn', 'keyword_en'):
                    keywords = data.get(key)
                    if isinstance(keywords, list) and keywords:
                        setattr(corpus, key, json.dumps(keywords, ensure_ascii=False))

                db.session.commit()
                corpus_id = corpus.id

                # 4. 附件文件上传到文件服务（事务外），失败则删除刚建的文献记录
                # 3. 主记录提交后再上传附件并单独保存引用，上传失败走现有删除补偿
                with open(file_path, 'rb') as stream:
                    storage = FileStorage(
                        stream=stream, filename=original_filename, content_type='application/pdf'
                    )
                    file_data = process_file_upload(storage, title_cn, username, corpus_id)
                if file_data['code'] != 200:
                    Corpus.query.filter(Corpus.id == corpus_id).delete()
                    db.session.commit()
                    raise RuntimeError(f"附件文件上传失败: {file_data.get('error')}")

                corpus.attachment = json.dumps(file_data['result'], ensure_ascii=False)
                corpus.word_number = file_data['words']
                db.session.commit()
    except Exception as e:
        app.logger.error(
            f"术语附件后台识别失败 term_id={term_id} field={field}: {e}", exc_info=True
        )
        _record_group_result(
            app, term_id, field, {"state": "failed", "error": str(e)}, group_total
        )
    else:
        # 4. 向附件组聚合本篇终态和关联，提取不足与上传失败分别保留不同原因
        entry = {
            "state": "reused" if reused else ("editing" if extraction_failed else "done"),
            "corpus_id": corpus_id,
            # 提取完整度：缺失的必填字段列表；复用已有文献时不做完整度判定
            "missing_fields": [] if reused else missing_fields,
        }
        if extraction_failed:
            entry["error"] = extract_error
        _record_group_result(
            app, term_id, field, entry, group_total,
            corpus_writer=(
                (lambda term: _write_attachment_corpus(term, field, corpus_id, original_filename))
                if corpus_id else None
            ),
        )
    finally:
        try:
            # 5. 无论复用、新建还是失败都清理请求阶段留下的本地副本
            os.remove(file_path)
        except OSError:
            pass


def _collect_attachment_files(counts):
    """收集出处模块与各定义的全部附件（每组可多个文件），落盘供后台线程使用。

    Args:
        counts: {定义前缀: 条目数}，用于推导需要读取哪些动态附件字段。

    Returns:
        {附件组字段名: [(落盘路径, 原始文件名), ...]}；没有附件时为空 dict。

    Raises:
        ValueError: 文件类型或大小不符合要求。
    """
    saved = {}
    for field in _attachment_group_fields(counts):
        entries = []
        for file in request.files.getlist(field):
            if not file or not file.filename:
                continue
            if not file.filename.lower().endswith('.pdf'):
                raise ValueError(f"{field} 仅支持PDF文件")
            file.seek(0, 2)
            if file.tell() > 200 * 1024 * 1024:
                raise ValueError(f"{field} 文件大小不能超过200MB")
            file.seek(0)
            fd, saved_path = tempfile.mkstemp(suffix='.pdf', prefix='term_async_')
            os.close(fd)
            file.save(saved_path)
            entries.append((saved_path, file.filename))
        if entries:
            saved[field] = entries
    return saved


def _discard_saved_files(saved_files):
    """术语没建成时，清掉已落盘的附件文件。"""
    for entries in (saved_files or {}).values():
        for path, _ in entries:
            try:
                os.remove(path)
            except OSError:
                pass


def _processing_state(saved_files):
    """附件提交时的初始处理状态：每组一个计数条目（全部文件完成才给终态）。"""
    return {
        field: {"state": "processing", "total": len(entries), "finished": 0, "corpus_ids": []}
        for field, entries in saved_files.items()
    }


def _merge_processing_state(app, term_id, saved_files):
    """把新一批附件的初始处理状态合并进 attachment_process（覆盖同组旧计数）。"""
    with _PROCESS_STATE_LOCK:
        with app.app_context():
            try:
                term = Terminology.query.filter(
                    Terminology.id == term_id, Terminology.is_deleted == 0
                ).first()
                if not term:
                    return
                state = _load_process_state(term)
                state.update(_processing_state(saved_files))
                term.attachment_process = json.dumps(state, ensure_ascii=False)
                db.session.commit()
            except Exception:
                db.session.rollback()
                app.logger.error(
                    f"初始化附件处理状态失败 term_id={term_id}", exc_info=True
                )


def _final_group_state(states):
    """一组附件全部处理完后的终态：失败优先，其次待补全，全复用为 reused，否则 done。"""
    if any(state == "failed" for state in states):
        return "failed"
    if any(state == "editing" for state in states):
        return "editing"
    if states and all(state == "reused" for state in states):
        return "reused"
    return "done"


def _write_attachment_corpus(term, field, corpus_id, display_name):
    """把单个附件的入库结果写进多条目 JSON（组内追加），并镜像旧单值列。"""
    column, slot = _attachment_slot(field)
    if not column:
        return
    if column == 'sources_json':
        doc = _load_json_doc(term.sources_json, {'mlas': [], 'attachments': []})
        attachments = doc.setdefault('attachments', [])
        attachments.append({'corpus_id': corpus_id, 'name': display_name})
        term.sources_json = json.dumps(doc, ensure_ascii=False)
    else:
        doc = _load_json_doc(getattr(term, column), [])
        if not isinstance(doc, list):
            doc = []
        while len(doc) <= slot:
            doc.append({'text': '', 'mlas': [], 'attachments': []})
        entry = doc[slot]
        if not isinstance(entry, dict):
            entry = {'text': '', 'mlas': [], 'attachments': []}
            doc[slot] = entry
        entry.setdefault('attachments', []).append(
            {'corpus_id': corpus_id, 'name': display_name}
        )
        setattr(term, column, json.dumps(doc, ensure_ascii=False))

    # 旧单值列镜像组内第 1 个附件，保证老前端/老接口仍能看到附件
    mirror = _mirror_column(field)
    if mirror and not getattr(term, mirror, None):
        setattr(term, mirror, corpus_id)


def _record_group_result(app, term_id, field, result_entry, group_total, corpus_writer=None):
    """合并单附件处理结果，并将术语关联和组进度在同一事务提交。

    Args:
        app: 后台线程使用的 Flask 应用。
        term_id: 当前术语 ID，已删除记录不再回填。
        field: 当前附件组字段名。
        result_entry: 单附件状态、文献 ID、缺失字段或错误摘要。
        group_total: 本次组总数，用于判断何时形成组终态。
        corpus_writer: 可选关联回填回调，与进度写入共用事务。

    锁仅协调本进程内线程，不提供跨进程互斥；每次调用都会增加 finished，
    未按附件 ID 做重复回调去重。失败回滚后只记日志，不伪造组完成状态。
    """
    # 1. 进程内串行合并同组更新，线程各自进入应用上下文读取最新术语
    with _PROCESS_STATE_LOCK:
        with app.app_context():
            try:
                term = Terminology.query.filter(
                    Terminology.id == term_id, Terminology.is_deleted == 0
                ).first()
                if not term:
                    return
                # 2. 在同一事务回填文献关联，并累积本篇的完成数、状态与错误
                if corpus_writer is not None:
                    corpus_writer(term)
                state = _load_process_state(term)
                group = state.get(field) if isinstance(state.get(field), dict) else {}
                finished = int(group.get("finished") or 0) + 1
                corpus_ids = list(group.get("corpus_ids") or [])
                if result_entry.get("corpus_id"):
                    corpus_ids.append(result_entry["corpus_id"])
                states = list(group.get("states") or [])
                states.append(result_entry.get("state") or "done")
                errors = [item for item in (group.get("errors") or []) if item]
                if result_entry.get("error"):
                    errors.append(result_entry["error"])
                total = max(int(group.get("total") or 0), int(group_total or 1), finished)
                group.update({
                    "total": total,
                    "finished": finished,
                    "corpus_ids": corpus_ids,
                    # 兼容旧前端：仍读单值 corpus_id
                    "corpus_id": corpus_ids[0] if corpus_ids else None,
                    "states": states,
                    "state": "processing" if finished < total else _final_group_state(states),
                })
                if result_entry.get("missing_fields"):
                    group["missing_fields"] = result_entry["missing_fields"]
                if errors:
                    group["error"] = "；".join(errors)
                # 3. 未全部完成保留 processing，全部完成按组规则确定终态后统一提交
                state[field] = group
                term.attachment_process = json.dumps(state, ensure_ascii=False)
                db.session.commit()
            except Exception:
                db.session.rollback()
                app.logger.error(
                    f"更新附件组处理状态失败 term_id={term_id} field={field}", exc_info=True
                )


def _spawn_attachment_workers(app, term_id, saved_files, username, user_id, reviewer_id, category_id):
    """为每个附件文件启动一个后台线程；daemon=True，进程重启后未完成的任务丢弃。

    Args:
        saved_files: {附件组字段名: [(落盘路径, 原始文件名), ...]}；同一组内多个文件
            各启动一个线程，处理状态按组计数推进，全部完成才给终态。
    """
    for field, entries in saved_files.items():
        total = len(entries)
        for index, (path, original_name) in enumerate(entries):
            worker = threading.Thread(
                target=_process_term_attachment,
                args=(app, term_id, field, path, original_name,
                      username, user_id, reviewer_id, category_id, total),
                daemon=True,
                name=f"term-extract-{term_id}-{field}-{index}",
            )
            worker.start()


@bp.route('/upload_async', methods=['POST'])
@permission_required(['COLLECTOR'])
def terminology_upload_async():
    """提交术语并异步识别附件。

    术语立即入库（status=1 待审核），附件文件落盘后交给后台线程执行
    「大模型提取 → 文献入库 → 关联」，立即返回术语 id 与正在处理的附件字段。
    处理状态存于 terminology.attachment_process，可通过 /terminology/get 查看。
    """
    data = request.form
    values = {field: (data.get(field) or '').strip() for field in REQUIRED_FIELDS}
    for field in REQUIRED_FIELDS:
        if not values[field]:
            return jsonify({"code": 400, "error": "术语名、所属类别、术语出处与中英文定义不能为空"}), 400

    try:
        category_id = int(values['category'])
    except ValueError:
        return jsonify({"code": 400, "error": "所属类别无效"}), 400

    user = g.current_user

    try:
        # 1. 所属类别必须是真实存在的大类
        database = Database.query.filter(
            Database.id == category_id, Database.is_deleted == 0
        ).first()
        if not database:
            return jsonify({"code": 404, "error": "无该所属类别"}), 404

        # 2. 同一大类下不允许重复术语名
        duplicate = Terminology.query.filter(
            Terminology.name == values['name'],
            Terminology.category == category_id,
            Terminology.is_deleted == 0,
        ).first()
        if duplicate:
            return jsonify({"code": 400, "error": "该类别下已存在同名术语"}), 400
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.error("术语提交数据库失败", exc_info=True)
        return jsonify({"code": 503, "error": "系统繁忙，请稍后重试"}), 503

    # 3. 收集附件文件（选填，仅 PDF，≤200MB；每组可多个），落盘供后台线程使用
    sources_doc = _build_sources_doc(data)
    definitions_cn_doc = _build_definitions_doc(data, 'definition_cn')
    definitions_en_doc = _build_definitions_doc(data, 'definition_en')
    counts = _definition_counts(data)
    try:
        saved_files = _collect_attachment_files(counts)
    except ValueError as e:
        return jsonify({"code": 400, "error": str(e)}), 400

    reviewer_id = db.session.query(User.leader_id).filter(
        User.id == user.id, User.is_deleted == 0
    ).scalar()

    try:
        terminology = Terminology(
            name=values['name'],
            category=category_id,
            source_mla=values['source_mla'],
            definition_cn=values['definition_cn'],
            definition_cn_source=data.get('definition_cn_source'),
            definition_en=values['definition_en'],
            definition_en_source=data.get('definition_en_source'),
            sources_json=json.dumps(sources_doc, ensure_ascii=False),
            definitions_cn_json=json.dumps(definitions_cn_doc, ensure_ascii=False),
            definitions_en_json=json.dumps(definitions_en_doc, ensure_ascii=False),
            contributor=user.id,
            status=1,  # 待审核，与文献上传一致
            submit_time=datetime.now(),
            attachment_process=(
                json.dumps(_processing_state(saved_files), ensure_ascii=False)
                if saved_files else None
            ),
        )
        db.session.add(terminology)
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        # 术语没建成，清掉已落盘的附件文件
        _discard_saved_files(saved_files)
        current_app.logger.error("术语提交数据库失败", exc_info=True)
        return jsonify({"code": 503, "error": "系统繁忙，请稍后重试"}), 503

    # 4. 每个附件一个后台线程处理
    _spawn_attachment_workers(
        current_app._get_current_object(), terminology.id, saved_files,
        user.username, user.id, reviewer_id, category_id,
    )

    message = (
        "提交成功，附件正在后台识别，完成后可在术语查找中查看"
        if saved_files else "提交成功"
    )
    return jsonify({
        "code": 200,
        "success": message,
        "terminology_id": terminology.id,
        "processing": list(saved_files.keys()),
    }), 200


@bp.route('/update', methods=['POST'])
@permission_required(['COLLECTOR', 'LEADER'])
def terminology_update():
    """修改术语（与 /collector/update 风格一致，只更新传上来的字段与附件关联）。

    - 文本字段：传了才更新（前端 buildTermFormData 会过滤空值）；name 更新时做同类查重，
      category 更新时验证大类存在。
    - 附件，三种情况（同一次请求只会出现其中一种）：
      a) 传 *_corpus_id：验证后直接改绑；
      b) 传附件原始文件（TERM_ATTACHMENT_BACKEND_READY=true 新契约）：落盘后交给后台线程
         「提取 → 入库 → 改绑」，替换中的字段状态记入 attachment_process；
      c) 都不传：保持原有绑定不变。
    - 任何情况都不会删除或修改原有文献记录。
    - 权限：数据收集者只能改自己提交的术语；已审核通过（status=2）的术语不可修改。
    """
    data = request.form
    terminology_id = data.get('terminology_id', type=int)
    if not terminology_id:
        return jsonify({"code": 400, "error": "缺少 terminology_id"}), 400

    user = g.current_user
    saved_files = {}

    try:
        term = Terminology.query.filter(
            Terminology.id == terminology_id, Terminology.is_deleted == 0
        ).first()
        if not term:
            return jsonify({"code": 404, "error": "无该术语"}), 404

        # 权限：只能管理自己提交的术语；组长/管理员不受限
        from app.utils.enums import UserRole
        if term.contributor != user.id and g.role not in (
            UserRole.ADMIN.value, UserRole.LEADER.value
        ):
            return jsonify({"code": 403, "error": "只能管理自己提交的术语"}), 403
        if term.status == 2:
            return jsonify({"code": 400, "error": "该术语已审核通过，不能再修改"}), 400

        updates = {}

        # 1. 文本字段（只更新传上来的）
        for field in ('name', 'source_mla', 'definition_cn', 'definition_cn_source',
                      'definition_en', 'definition_en_source'):
            value = (data.get(field) or '').strip()
            if value:
                updates[field] = value

        # name 查重：同一大类下不允许与其他术语重名（排除自身）
        if 'name' in updates:
            dup = Terminology.query.filter(
                Terminology.name == updates['name'],
                Terminology.category == term.category,
                Terminology.id != terminology_id,
                Terminology.is_deleted == 0,
            ).first()
            if dup:
                return jsonify({"code": 400, "error": "该类别下已存在同名术语"}), 400

        # 2. 所属类别
        category_id = term.category
        raw_category = (data.get('category') or '').strip()
        if raw_category:
            try:
                new_category = int(raw_category)
            except ValueError:
                return jsonify({"code": 400, "error": "所属类别无效"}), 400
            if new_category != term.category:
                if not Database.query.filter(
                    Database.id == new_category, Database.is_deleted == 0
                ).first():
                    return jsonify({"code": 404, "error": "无该所属类别"}), 404
                updates['category'] = new_category
            category_id = new_category

        # 2.5 多条目结构（前端新契约）：sources / definitions_cn / definitions_en(JSON)
        #     按下标替换文本与 MLA，各条目已有附件保留；旧扁平字段同步镜像第 1 条
        updates.update(_merge_multi_docs(term, data))

        # 3. 附件关联（*_corpus_id 直接改绑；同时记入多条目结构）
        rebind = {}
        for field in TERM_ATTACHMENT_CORPUS_FIELDS:
            raw = data.get(field)
            if raw in (None, ''):
                continue
            try:
                corpus_id = int(raw)
            except ValueError:
                return jsonify({"code": 400, "error": f"{field} 无效"}), 400
            if not Corpus.query.filter(
                Corpus.id == corpus_id, Corpus.is_deleted == 0
            ).first():
                return jsonify({"code": 404, "error": f"关联文献不存在（id={corpus_id}）"}), 404
            updates[field] = corpus_id
            rebind[field] = corpus_id
        if rebind:
            updates.update(_merge_rebound_corpora(term, rebind))

        # 4. 新附件文件（新契约，每组可多个）：先落盘校验，通过后再更新数据库
        counts = _definition_counts(data)
        try:
            saved_files = _collect_attachment_files(counts)
        except ValueError as e:
            return jsonify({"code": 400, "error": str(e)}), 400

        if updates:
            Terminology.query.filter(Terminology.id == terminology_id).update(updates)
            db.session.commit()

        if saved_files:
            app = current_app._get_current_object()
            _merge_processing_state(app, terminology_id, saved_files)
            reviewer_id = db.session.query(User.leader_id).filter(
                User.id == user.id, User.is_deleted == 0
            ).scalar()
            _spawn_attachment_workers(
                app, terminology_id, saved_files,
                user.username, user.id, reviewer_id, category_id,
            )
    except SQLAlchemyError:
        db.session.rollback()
        # 术语没改成，清掉已落盘的附件文件
        _discard_saved_files(saved_files)
        current_app.logger.error("术语更新数据库失败", exc_info=True)
        return jsonify({"code": 503, "error": "系统繁忙，请稍后重试"}), 503

    message = "保存成功，新附件正在后台识别" if saved_files else "保存成功"
    return jsonify({"code": 200, "success": message, "terminology_id": terminology_id}), 200


@bp.route('/delete', methods=['GET'])
@permission_required(['COLLECTOR', 'LEADER'])
def terminology_delete():
    """删除术语（逻辑删除，与 /collector/delete 风格一致）。

    只删除术语条目本身，已入库的关联文献与附件文件都保留在文献库中。
    权限：数据收集者只能删自己提交的术语；已审核通过（status=2）的术语不可删除。
    """
    terminology_id = request.args.get('terminology_id', type=int)
    if not terminology_id:
        return jsonify({"code": 400, "error": "缺少 terminology_id"}), 400

    user = g.current_user
    try:
        term = Terminology.query.filter(
            Terminology.id == terminology_id, Terminology.is_deleted == 0
        ).first()
        if not term:
            return jsonify({"code": 404, "error": "无该术语"}), 404

        from app.utils.enums import UserRole
        if term.contributor != user.id and g.role not in (
            UserRole.ADMIN.value, UserRole.LEADER.value
        ):
            return jsonify({"code": 403, "error": "只能管理自己提交的术语"}), 403
        if term.status == 2:
            return jsonify({"code": 400, "error": "该术语已审核通过，不能再删除"}), 400

        # BIT 列只在 SQL 侧赋值，不 select 进 Python（BIT 铁律）
        Terminology.query.filter(Terminology.id == terminology_id).update(
            {Terminology.is_deleted: 1}
        )
        db.session.commit()
        return jsonify({"code": 200, "success": "删除成功"}), 200
    except SQLAlchemyError:
        db.session.rollback()
        current_app.logger.error("术语删除数据库失败", exc_info=True)
        return jsonify({"code": 503, "error": "系统繁忙，请稍后重试"}), 503


@bp.route('/attachment_file', methods=['GET'])
def terminology_attachment_file():
    """获取术语关联文献的附件文件流。

    与公共文件接口的区别：登录即可访问，不受语料审核状态限制——
    术语附件入库后处于待审核状态，学生需要立刻查看自己上传的附件。
    """
    corpus_id = request.args.get('corpus_id', type=int)
    if not corpus_id:
        return jsonify({"code": 400, "error": "缺少文献 id"}), 400
    corpus = Corpus.query.filter(
        Corpus.id == corpus_id, Corpus.is_deleted == 0
    ).first()
    if not corpus or not corpus.attachment:
        return jsonify({"code": 404, "error": "文献不存在或无附件"}), 404
    try:
        attachments = json.loads(corpus.attachment)
    except (TypeError, ValueError):
        return jsonify({"code": 400, "error": "附件信息异常"}), 400
    items = attachments if isinstance(attachments, list) else []
    if not items or not items[0].get('file_path'):
        return jsonify({"code": 404, "error": "语料无该附件"}), 404

    response, code = get_file(items[0]['file_path'])
    if code != 200:
        return jsonify({"code": code, "error": response}), code
    response.headers['Content-Disposition'] = "attachment; filename=term_attachment.pdf"
    return response


@bp.route('/get', methods=['GET'])
def terminology_get():
    """按术语名 / 所属类别分页检索术语。

    Query:
        page_num: 从 1 开始的页号（默认 1）。
        page_size: 每页数量（默认 10，最大 100）。
        name: 术语名模糊匹配（可选）。
        category: 所属大类 id（可选）。

    Returns:
        术语列表（含附件展示名与下载链接）与总数。
    """
    try:
        page_num = max(int(request.args.get('page_num', 1)), 1)
        page_size = min(max(int(request.args.get('page_size', 10)), 1), 100)
    except ValueError:
        return jsonify({"code": 400, "error": "分页参数无效"}), 400

    name = (request.args.get('name') or '').strip()
    category = request.args.get('category')
    terminology_id = request.args.get('terminology_id', type=int)

    filters = [Terminology.is_deleted == 0]
    if name:
        filters.append(Terminology.name.like(f"%{name}%"))
    if category not in (None, ''):
        try:
            filters.append(Terminology.category == int(category))
        except ValueError:
            return jsonify({"code": 400, "error": "所属类别无效"}), 400
    if terminology_id:
        # 精确查询单条：前端轮询附件后台识别状态用
        filters.append(Terminology.id == terminology_id)

    try:
        query = Terminology.query.filter(*filters)
        total = query.count()
        rows = (
            query.order_by(Terminology.submit_time.desc(), Terminology.id.desc())
            .offset((page_num - 1) * page_size)
            .limit(page_size)
            .all()
        )

        items = []
        for row in rows:
            item = row.to_dict()
            # 附件后台处理状态（upload_async 异步识别的进度：processing/done/reused/failed）
            item['attachment_process'] = _load_process_state(row)
            # 附件展示名与下载链接：来自关联文献的首个附件
            for field in TERM_ATTACHMENT_CORPUS_FIELDS:
                corpus_id = getattr(row, field)
                corpus = Corpus.query.filter(Corpus.id == corpus_id).first() if corpus_id else None
                display_name, file_url = _attachment_info(corpus)
                # source_attachment_corpus_id -> source_attachment / source_attachment_url
                short_field = field.replace('_corpus_id', '')
                item[short_field] = display_name
                item[f"{short_field}_url"] = (
                    file_url if not (corpus and corpus.is_deleted) else None
                )
            # 多条目结构：多条 MLA 出处 / 中英文多条定义 / 每处多个附件（新契约）
            item.update(_term_multi_payload(row, item['attachment_process']))
            items.append(item)

        return jsonify(
            {
                "code": 200,
                "data": {
                    "items": items,
                    "total": total,
                    "page_num": page_num,
                    "page_size": page_size,
                },
            }
        ), 200
    except SQLAlchemyError:
        current_app.logger.error("术语查询数据库失败", exc_info=True)
        return jsonify({"code": 503, "error": "系统繁忙，请稍后重试"}), 503
