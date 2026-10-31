"""提供收集者文献采集、草稿维护、提交审核及模型辅助提取、摘要与翻译接口。"""

import os
from datetime import datetime
import re
import csv
import io
import json

from flask import Blueprint, request, jsonify, g, current_app
from sqlalchemy import and_, func, or_
from sqlalchemy.exc import SQLAlchemyError
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment

from app.utils.utils import *
from app.exts import db
from app.models import Corpus, CorpusCategory, Database, CorpusDatabase, Keywords, User
from app.utils.decorators import permission_required
from app.utils.enums import CorpusStatus, KeywordFlag, KeywordLanguage
from app.indexing.jobs import cancel_corpus_index
from app.utils.literature_metadata import (
    setup_qwen_client, robust_extract_pdf_info, extract_pdf_info_turbo, extract_pdf_info_long,
)
from sqlalchemy import case
bp = Blueprint('collector', __name__)

# 导出用户上传数量详情
@bp.route('/export_corpus_count', methods=['GET'])
@permission_required(['COLLECTOR'])
def export_corpus_count():
    """按时间范围导出所有有效用户的已审核文献数量及类型、语言分布。

    日期按提交时间过滤，用户按总量降序；每个类型生成中文和外文两列。
    未给开始时间时不添加下界过滤，表格展示的默认日期仅为历史显示约定。

    Returns:
        内存生成的 XLSX 下载响应，或参数、统计与文件生成错误。
    """
    try:
        # 获取时间参数
        # 1. 解析统计时间范围，以文献提交时间作为筛选口径
        start_time_str = request.args.get('start_time')
        end_time_str = request.args.get('end_time')

        # 处理时间参数
        start_time = None
        end_time = datetime.now()
        if start_time_str:
            try:
                start_time = datetime.strptime(start_time_str, '%Y-%m-%d')
            except ValueError:
                return jsonify({"code": 400, "error": "开始时间格式错误，请使用 YYYY-MM-DD 格式"}), 400
        if end_time_str:
            try:
                end_time = datetime.strptime(end_time_str, '%Y-%m-%d')
            except ValueError:
                return jsonify({"code": 400, "error": "结束时间格式错误，请使用 YYYY-MM-DD 格式"}), 400

        # 2. 读取有效类型和用户，聚合总量与类型语言计数；无数据用户仍保留零值
        categories = db.session.query(CorpusCategory.id, CorpusCategory.category_name).filter(
            CorpusCategory.is_deleted == 0
        ).order_by(CorpusCategory.id.asc()).all()
        category_ids = [cid for cid, _ in categories]
        category_names = [cname for _, cname in categories]

        users = db.session.query(User.id, User.name).filter(User.is_deleted == 0).all()
        user_ids = [uid for uid, _ in users]

        # 构建时间过滤条件
        time_filters = [
            Corpus.is_deleted == 0,
            Corpus.status == CorpusStatus.APPROVE_EXAMINE.value,
            Corpus.contributor.in_(user_ids)
        ]
        if start_time:
            time_filters.append(Corpus.submit_time >= start_time)
        if end_time:
            time_filters.append(Corpus.submit_time <= end_time)

        # 统计每位用户审核通过语料的总量（不变）
        total_counts = dict(
            db.session.query(Corpus.contributor, func.count(Corpus.id))
            .filter(*time_filters)
            .group_by(Corpus.contributor)
            .all()
        )

        # 按corpus_count总量降序排序用户列表
        users_with_counts = [(uid, name, total_counts.get(uid, 0)) for uid, name in users]
        users_with_counts.sort(key=lambda x: x[2], reverse=True)

        # ========== 按 用户 + 类别 分组，SQL条件聚合区分中英文 ==========
        cn_expr = case(
            (Corpus.language == '中文', 1),
            else_=0
        )
        foreign_expr = case(
            (Corpus.language == '中文', 0),
            else_=1
        )

        per_category_rows = (
            db.session.query(
                Corpus.contributor,
                Corpus.category_id,
                func.sum(cn_expr).label('cn_count'),
                func.sum(foreign_expr).label('foreign_count')
            )
            .filter(
                *time_filters,
                Corpus.category_id.in_(category_ids)
            )
            .group_by(Corpus.contributor, Corpus.category_id)
            .all()
        )

        # 组装为: user_id -> {category_id: {'cn': count, 'foreign': count}}
        # 每个 (uid, cid) 只有一行结果，不存在覆盖/遗漏问题
        per_category_counts = {}
        for uid, cid, cn_cnt, foreign_cnt in per_category_rows:
            if uid not in per_category_counts:
                per_category_counts[uid] = {}
            per_category_counts[uid][cid] = {
                'cn': int(cn_cnt or 0),
                'foreign': int(foreign_cnt or 0)
            }
        # 创建Excel工作簿
        # 3. 按固定类型列顺序生成工作簿，逐用户映射中文与外文数量
        wb = Workbook()
        ws = wb.active
        ws.title = "用户上传语料数量统计"

        # ========== 表头每个类别拆分为 中文 + 外文 两列 ==========
        header = ['姓名', '上传总量']
        for cname in category_names:
            header.append(f'{cname}(中文)')
            header.append(f'{cname}(外文)')
        header.extend(['开始时间', '结束时间'])

        # 写入表头并设置样式
        for col, header_text in enumerate(header, 1):
            cell = ws.cell(row=1, column=col, value=header_text)
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal='center')

        # ========== 数据写入适配新的两列结构 ==========
        for row_idx, (uid, name, corpus_count) in enumerate(users_with_counts, 2):
            ws.cell(row=row_idx, column=1, value=name or '')
            ws.cell(row=row_idx, column=2, value=int(corpus_count))

            user_cat_map = per_category_counts.get(uid, {})
            col_idx = 3
            for cid in category_ids:
                lang_data = user_cat_map.get(cid, {'cn': 0, 'foreign': 0})
                ws.cell(row=row_idx, column=col_idx, value=int(lang_data['cn']))
                ws.cell(row=row_idx, column=col_idx + 1, value=int(lang_data['foreign']))
                col_idx += 2

            # 写入时间信息
            start_time_col = col_idx
            end_time_col = start_time_col + 1
            if start_time:
                ws.cell(row=row_idx, column=start_time_col, value=start_time.strftime('%Y-%m-%d'))
            else:
                ws.cell(row=row_idx, column=start_time_col, value='2025-07-21')
            ws.cell(row=row_idx, column=end_time_col, value=end_time.strftime('%Y-%m-%d'))

        # 自动调整列宽
        def get_display_width(text):
            """估算表格列宽，ASCII 字符计一格，其他字符计两格。"""
            if not text:
                return 0
            width = 0
            for char in str(text):
                if ord(char) > 127:
                    width += 2
                else:
                    width += 1
            return width

        for column in ws.columns:
            max_width = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    cell_width = get_display_width(cell.value)
                    if cell_width > max_width:
                        max_width = cell_width
                except:
                    pass
            adjusted_width = max(8, min(max_width + 3, 60))
            ws.column_dimensions[column_letter].width = adjusted_width

        # 保存到内存
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)

        from flask import Response
        now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"user_corpus_by_category_{now_str}.xlsx"
        return Response(
            output.getvalue(),
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            headers={
                'Content-Disposition': f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        return jsonify({"code": 500, "error": str(e)}), 500

@bp.route('/check_title', methods=['POST'])
@permission_required(['COLLECTOR'])
def check_title():
    """规范化题名后检查当前用户适用范围内的重复文献，空题名直接放行。"""
    data = request.form
    title_cn = (data.get('title_cn') or '').strip()
    
    # 如果title_cn为空，直接返回成功
    if not title_cn:
        return jsonify({"code": 200, "msg": "success"}), 200
    
    # 获取当前用户ID
    current_user_id = g.current_user.id
    
    # 查询条件：排除当前用户的草稿；忽略空白差异做同名检查
    # 1. 如果记录是当前用户的草稿，不算重复
    # 2. 其他所有情况都算重复（包括：其他用户的草稿、其他用户的非草稿、当前用户的非草稿）
    duplicate = find_duplicate_title_cn(title_cn, current_user_id)
    if duplicate:
        # 附带已有文献 id：术语附件等场景可直接复用已有文献，不必重复上传
        return jsonify({"code": 400, "error": "原文标题重复", "corpus_id": duplicate.id}), 400
    
    return jsonify({"code": 200, "msg": "success"}), 200

# 只保留年份
def extract_year(val):
    """按当前正则尝试从输入开头提取年份，空值或不匹配时返回 None。"""
    if val is None or val == '':
        return None
    match = re.match(r"(\d{4})", str(val))
    if match:
        return int(match.group(1))
    return None


def normalize_title_cn(title):
    """去掉首尾空白，并移除标题中全部空白字符（含全角空格），用于同名比对。"""
    if not title:
        return ""
    return re.sub(r'\s+', '', str(title).strip(), flags=re.UNICODE)


def title_cn_normalized_expr():
    """SQL 侧标题归一化：trim + 去掉普通空格/制表符/全角空格。"""
    return func.replace(
        func.replace(
            func.replace(func.trim(Corpus.title_cn), ' ', ''),
            '\t', ''
        ),
        '\u3000', ''
    )


def find_duplicate_title_cn(title_cn, current_user_id, exclude_corpus_id=None):
    """按归一化标题查重；排除当前用户自己的草稿。"""
    normalized = normalize_title_cn(title_cn)
    if not normalized:
        return None
    query = Corpus.query.filter(
        title_cn_normalized_expr() == normalized,
        Corpus.is_deleted == 0,
        ~(
            (Corpus.contributor == current_user_id) &
            (Corpus.status == CorpusStatus.DRAFT.value)
        )
    )
    if exclude_corpus_id is not None:
        query = query.filter(Corpus.id != exclude_corpus_id)
    return query.first()

# TODO 上传语料（status=0）
@bp.route('/upload', methods=['POST'])
@permission_required(['COLLECTOR'])
def upload_corpus():
    """保存或更新本人草稿，或创建待审核文献并上传附件。

    草稿元数据先提交，再上传附件；正式记录先分配分类流水号并提交主表及关联，
    再调用文件服务。正式附件上传失败时尝试删除已提交主记录作为补偿，
    数据库回滚不会撤销先前提交或远程文件操作。

    Returns:
        保存结果及可选 corpus_id，或校验、文件服务和数据库错误响应。
    """
    # 1. 读取采集表单并规范化元数据，区分可缺附件的草稿与正式待审提交
    data = request.form
    # 新增：支持草稿保存
    try:
        status = int(data.get('status', 0))  # 默认为0（编辑中），-1为草稿
    except Exception:
        status = 0
    is_draft = (status == CorpusStatus.DRAFT.value)
    language = data.get('language')
    title_cn = (data.get('title_cn') or '').strip()
    title_en = data.get('title_en')
    author = data.get('author')
    translator = data.get('translator')
    editor = data.get('editor')
    publish_time = data.get('publish_time')
    ISSN = data.get('ISSN')
    source = data.get('source')
    category_id = data.get('category_id')
    keyword_cn = [keyword_cn.strip() for keyword_cn in data.getlist('keyword_cn') if keyword_cn.strip() != ""]
    keyword_en = [keyword_en.strip() for keyword_en in data.getlist('keyword_en') if keyword_en.strip() != ""]
    abstract_cn = data.get('abstract_cn')
    abstract_en = data.get('abstract_en')
    original_name = data.get('original_name')
    original_language = data.get('original_language')
    original_time = data.get('original_time')
    publish_time = extract_year(publish_time)
    original_time = extract_year(original_time)
    original_source = data.get('original_source')
    database_ids = data.getlist('database_id', type=int)
    volume = data.get('volume')
    issue = data.get('issue')

    user = g.current_user
    # 支持前端传入 reviewer_id，如果未传则使用默认绑定的审核者
    reviewer_id = data.get('reviewer_id', type=int)
    if not reviewer_id:
        reviewer_id = db.session.query(User.leader_id).filter(User.id == user.id, User.is_deleted == 0).first()[0]
    
    # 获取当前用户ID
    # 2. 按当前用户检查题名及附件约束，避免无效请求进入有副作用的保存流程
    current_user_id = user.id

    # 正式上传必须填写原文标题；草稿仍允许暂时为空
    if not is_draft and not title_cn:
        return jsonify({"code": 400, "error": "原文标题不能为空"}), 400

    # 查询条件：排除当前用户的草稿；忽略空白差异做同名检查
    if title_cn:
        duplicate = find_duplicate_title_cn(title_cn, current_user_id)
        if duplicate:
            # 附带已有文献 id：术语附件链路可直接复用已有文献，不必重复上传
            return (
                jsonify({"code": 400, "error": "原文标题重复", "corpus_id": duplicate.id}),
                400,
            )

    files = request.files.getlist('file')
    
    # 检查文件上传限制（提前检查，避免不必要的数据库操作）
    if not is_draft and (not files or len(files) == 0):
        return jsonify({"code": 400, "error": "缺少语料文件"}), 400
    if files and len(files) > 1:
        return jsonify({"code": 400, "error": "仅支持上传单个文件"}), 400

    try:
        # 开始数据库事务
        current_app.logger.info(f"开始保存语料: {title_cn}")
        # 3. 在草稿或正式分支持久化元数据与关联，先获得文献 ID 再命名上传附件
        # 后续分支存在多次提交；远程上传与这些数据库检查点不构成跨服务原子事务。
        db.session.begin_nested()

        if is_draft:
            # 草稿保存逻辑
            # 如果所有主要字段和文件都为空，则不保存草稿，直接返回成功
            # 草稿分支复用本人最新草稿，并锁定记录防止并发自动保存相互覆盖。
            has_content = any([
                language, title_cn, title_en, author, translator, editor,
                publish_time, ISSN, source, category_id,
                abstract_cn, abstract_en,
                original_name, original_language, original_time,
                original_source, database_ids, keyword_cn, keyword_en,
                volume, issue
            ])
            if not has_content and (not files or len(files) == 0):
                db.session.rollback()
                return jsonify({"code": 200, "success": "草稿未保存（内容为空）"})

            # 查找当前用户最新的一条草稿记录，做更新操作
            corpus = db.session.query(Corpus).filter(
                Corpus.is_deleted == 0,
                Corpus.contributor == user.id,
                Corpus.status == CorpusStatus.DRAFT.value
            ).order_by(Corpus.id.desc()).with_for_update(of=Corpus).first()

            if corpus:
                # 更新已有草稿
                corpus.language = language
                corpus.author = author
                corpus.translator = translator
                corpus.editor = editor
                corpus.publish_time = publish_time
                corpus.ISSN = ISSN
                corpus.source = source
                corpus.category_id = category_id
                corpus.abstract_cn = abstract_cn
                corpus.abstract_en = abstract_en
                corpus.original_name = original_name
                corpus.original_language = original_language
                corpus.original_time = original_time
                corpus.original_source = original_source
                corpus.title_cn = title_cn
                corpus.title_en = title_en
                corpus.volume = volume
                corpus.issue = issue
                corpus.reviewer_id = reviewer_id

                # 更新所属数据库：先清空再重建
                CorpusDatabase.query.filter(
                    CorpusDatabase.corpus_id == corpus.id
                ).delete(synchronize_session=False)
                for db_id in database_ids:
                    corpus_database = CorpusDatabase(corpus_id=corpus.id, database_id=db_id)
                    db.session.add(corpus_database)

                # 更新关键词（仅保存在 Corpus 表中，不影响关键词统计表）
                corpus.keyword_cn = json.dumps(keyword_cn, ensure_ascii=False) if keyword_cn else None
                corpus.keyword_en = json.dumps(keyword_en, ensure_ascii=False) if keyword_en else None

            else:
                # 原逻辑：不存在草稿时新建一条
                corpus = Corpus(
                    language=language, author=author, translator=translator, editor=editor,
                    publish_time=publish_time, ISSN=ISSN, source=source, category_id=category_id,
                    abstract_cn=abstract_cn, abstract_en=abstract_en, original_name=original_name,
                    original_language=original_language, original_time=original_time,
                    original_source=original_source, contributor=user.id, title_cn=title_cn, 
                    title_en=title_en, number=None, volume=volume, issue=issue, reviewer_id=reviewer_id
                )
                corpus.status = CorpusStatus.DRAFT.value
                db.session.add(corpus)
                # 先取得数据库分配的文献主键，供主题关联及后续附件命名使用；此处尚未提交。
                db.session.flush()

                # 保存所属数据库
                for db_id in database_ids:
                    corpus_database = CorpusDatabase(corpus_id=corpus.id, database_id=db_id)
                    db.session.add(corpus_database)

                # 保存关键词
                for keywords, attr_name, lang in [
                    (keyword_cn, 'keyword_cn', KeywordLanguage.CN.value),
                    (keyword_en, 'keyword_en', KeywordLanguage.EN.value)
                ]:
                    if keywords:
                        setattr(corpus, attr_name, json.dumps(keywords, ensure_ascii=False))
            
            db.session.commit()
            
            # 文件上传（在事务外）
            if files and len(files) > 0:
                # 元数据已提交；附件失败时草稿保留，便于后续继续补充。
                file_data = process_file_upload(files[0], title_cn, user.username, corpus.id)
                if file_data['code'] != 200:
                    return jsonify(file_data), file_data['code']
                
                # 更新文件信息到数据库（新事务）
                try:
                    corpus.attachment = json.dumps(file_data['result'], ensure_ascii=False)
                    corpus.word_number = file_data['words']
                    db.session.commit()
                except Exception as e:
                    current_app.logger.error(f"更新文件信息失败: {str(e)}")
                    # 文件上传成功但数据库更新失败，仍返回成功但记录日志
                    
            return jsonify({"code": 200, "success": "草稿保存成功"})
        else:
            if not database_ids or len(database_ids) == 0: 
                return jsonify({"code": 400, "error": "缺少语料所属的数据库id"}), 400
            if not category_id: 
                return jsonify({"code": 400, "error": "缺少语料类别id"}), 400 
            corpus_category = CorpusCategory.query.filter(
                CorpusCategory.id == category_id,
                CorpusCategory.is_deleted == 0
            ).with_for_update(of=CorpusCategory).first()
            
            if not corpus_category: 
                return jsonify({"code": 400, "error": "无该语料类别"}), 400
                
            # 正式分支锁定类型计数器后分配编号，避免同类型并发提交取得相同流水号。
            new_category_number = corpus_category.number + 1
            corpus_category.number = new_category_number

            current_app.logger.info(f"保存语料: {title_cn}")
            corpus = Corpus(
                language=language, author=author, translator=translator, editor=editor,
                publish_time=publish_time, ISSN=ISSN, source=source, category_id=category_id,
                abstract_cn=abstract_cn, abstract_en=abstract_en, original_name=original_name,
                original_language=original_language, original_time=original_time,
                original_source=original_source, contributor=user.id, title_cn=title_cn, 
                title_en=title_en, number=f"{corpus_category.code}{new_category_number}",
                volume=volume, issue=issue, reviewer_id=reviewer_id, submit_time=datetime.now()
            )
            corpus.status = CorpusStatus.WAITING_EXAMINE.value
            db.session.add(corpus)
            db.session.flush()

            # 保存所属数据库
            current_app.logger.info(f"保存语料所属数据库: {database_ids}")
            for db_id in database_ids:
                if not Database.query.filter(Database.id == db_id, Database.is_deleted == 0).first():
                    db.session.rollback()
                    return jsonify({"code": 404, "error": f'无id为{db_id}的数据库'}), 404
                corpus_database = CorpusDatabase(corpus_id=corpus.id, database_id=db_id)
                db.session.add(corpus_database)

            # 保存关键词
            current_app.logger.info(f"保存语料关键词: {keyword_cn}, {keyword_en}")  
            for keywords, attr_name, lang in [
                (keyword_cn, 'keyword_cn', KeywordLanguage.CN.value),
                (keyword_en, 'keyword_en', KeywordLanguage.EN.value)
            ]:
                if keywords:
                    setattr(corpus, attr_name, json.dumps(keywords, ensure_ascii=False))

            db.session.commit()
            
            # 文件上传（在事务外）
            current_app.logger.info(f"开始上传语料文件: {files[0].filename}")
            file_data = process_file_upload(files[0], title_cn, user.username, corpus.id)
            if file_data['code'] != 200:
                # 文件上传失败，删除已保存的数据库记录，避免数据不一致
                try:
                    Corpus.query.filter(Corpus.id == corpus.id).delete()
                    db.session.commit()
                    current_app.logger.warning(f"文件上传失败，已删除语料记录 id={corpus.id}")
                except Exception as e:
                    current_app.logger.error(f"删除失败记录时出错: {str(e)}")
                return jsonify(file_data), file_data['code']
            
            # 更新文件信息到数据库（新事务）
            try:
                corpus.attachment = json.dumps(file_data['result'], ensure_ascii=False)
                corpus.word_number = file_data['words']
                db.session.commit()
            except Exception as e:
                current_app.logger.error(f"更新文件信息失败: {str(e)}")
                # 文件上传成功但数据库更新失败，仍返回成功但记录日志

            # 删除同名草稿（新事务），并同时删除服务器上的草稿文件
            try:
                # 正式上传完成后尽力清理同规范题名的草稿和远程附件；当前查询不另按贡献者过滤。
                draft_query = Corpus.query.filter(
                    title_cn_normalized_expr() == normalize_title_cn(title_cn),
                    Corpus.status == CorpusStatus.DRAFT.value,
                )
                draft_list = draft_query.all()

                # 收集草稿附件对应的文件路径
                draft_files = []
                for draft in draft_list:
                    if not draft.attachment:
                        continue
                    try:
                        att_list = json.loads(draft.attachment)
                        for item in att_list:
                            if isinstance(item, dict) and item.get("file_path"):
                                draft_files.append(item["file_path"])
                    except Exception as e:
                        current_app.logger.warning(f"解析草稿附件失败 corpus_id={draft.id}: {e}")

                # 先请求文件服务删除这些草稿文件（忽略返回值，仅记录错误日志）
                if draft_files:
                    try:
                        delete_file(draft_files)
                    except Exception as e:
                        current_app.logger.error(f"删除草稿文件失败: {str(e)}")

                # 再删除数据库中的草稿记录
                draft_query.delete(synchronize_session=False)
                db.session.commit()
            except Exception as e:
                current_app.logger.error(f"删除草稿及草稿文件失败: {str(e)}")

            # 返回新建语料 id：术语库等功能需要把附件关联到文献
            return jsonify({"code": 200, "success": "保存成功", "corpus_id": corpus.id}), 200

    except requests.exceptions.RequestException as e:
        db.session.rollback()
        return jsonify({"code": 400, "error": str(e)}), 400
    except SQLAlchemyError as e:
        db.session.rollback()
        current_app.logger.error(f"数据库操作失败: {str(e)}")
        return jsonify({"code": 503, "error": "系统繁忙，请稍后重试"}), 503
 
def process_file_upload(file, title_cn, username, corpus_id=None):
    """生成带文献 ID 的文件名，统计字数并将单个附件上传到文件服务。

    Args:
        file: 可读写游标及 filename 的上传文件对象。
        title_cn: 用于生成文件名的题名，空值使用 file。
        username: 文件服务中的用户归属标识。
        corpus_id: 可选文献主键，正式记录先 flush 获取后传入。

    Returns:
        含 code、附件记录 result 和 words 的字典，失败时包含 error。
    """
    try:
        file_list = []
        base_name = title_cn if title_cn else 'file'
        # 在文件名前增加 corpus_id 前缀，形如：1234-文件名.pdf
        if corpus_id is not None:
            base_name = f"{corpus_id}-{base_name}"
        ext = file.filename.rsplit('.', 1)[-1] if '.' in file.filename else ''
        new_filename = f"{base_name}.{ext}" if ext else base_name
        # 去除文件名中的换行符等控制字符，避免 %0D%0A 进入保存路径导致下载时 HTTP 解析失败
        new_filename = re.sub(r'[/\x00\r\n]', '_', new_filename)
        file.filename = new_filename
        current_app.logger.info(f"统计字数: {file.filename}")
        original_pos = file.tell()
        words = count_words(file)
        file.seek(original_pos)
        file_list.append((file, words))
        file.seek(original_pos)

        upload_result, code = upload_file(file_list, 'corpus', username)
        if code != 200:
            return {"code": code, "error": upload_result}
    except Exception as e:
        current_app.logger.error(f"文件上传失败: {str(e)}")
        return {"code": 500, "error": "文件上传失败"}
    return {"code": 200, "result": upload_result, "words": words}



# TODO 修改语料
@bp.route('/update', methods=['POST'])
@permission_required(['COLLECTOR', 'LEADER'])
def update_corpus():
    """更新文献元数据及关联，附件变化时使旧索引任务失效并重新待审。

    表单中的真值才覆盖元数据；主题和关键词按差集更新。文件服务操作先于提交，
    附件新清单与任务取消在同一数据库事务提交，之后尽力清理旧索引。

    Returns:
        修改结果 JSON，或参数、资源和文件服务错误响应。
    """
    # 1. 读取修改表单并校验卷期与目标记录，保持现有真值覆盖字段的规则
    data = request.form
    corpus_id = data.get('corpus_id')
    language = data.get('language')
    title_cn = data.get('title_cn')
    title_en = data.get('title_en')
    author = data.get('author')
    translator = data.get('translator')
    editor = data.get('editor')
    publish_time = data.get('publish_time')
    ISSN = data.get('ISSN')
    source = data.get('source')
    category_id = data.get('category_id')
    keyword_cn = [keyword_cn.strip() for keyword_cn in data.getlist('keyword_cn') if keyword_cn.strip() != ""]
    keyword_en = [keyword_en.strip() for keyword_en in data.getlist('keyword_en') if keyword_en.strip() != ""]
    abstract_cn = data.get('abstract_cn')
    abstract_en = data.get('abstract_en')
    original_name = data.get('original_name')
    original_language = data.get('original_language')
    original_time = data.get('original_time')
    publish_time = extract_year(publish_time)
    original_time = extract_year(original_time)
    original_source = data.get('original_source')
    database_ids = data.getlist('database_id', type=int)
    user = g.current_user
    volume = data.get('volume')
    issue = data.get('issue')
    reviewer_id = data.get('reviewer_id')

    # volume 和 issue 必须为整数
    try:
        if volume is not None and volume != '':
            volume = int(volume)
        else:
            volume = None
        if issue is not None and issue != '':
            issue = int(issue)
        else:
            issue = None
    except ValueError:
        return jsonify({"code": 400, "error": "卷号和期号必须为整数"}), 400

    corpus = Corpus.query.filter(Corpus.id == corpus_id, Corpus.is_deleted == 0,
                                 Corpus.owner_id.is_(None), Corpus.status != 4).first()
    if not corpus: return jsonify({"code": 404, "error": "无该语料"}), 404
    
    if category_id and CorpusCategory.query.filter(CorpusCategory.id == category_id,
                                                   CorpusCategory.is_deleted == 0).count() == 0:
        return jsonify({"code": 404, "error": "无该语料类别"}), 404

    try:
        # 定义需要检查和可能更新的属性列表
        # 2. 合并元数据、主题关联及双语关键词差集，并同步已有关键词引用频次
        attributes = [
            'title_cn', 'title_en', 'language', 'author', 'translator', 'editor',
            'publish_time', 'ISSN', 'source', 'category_id',
            'abstract_cn', 'abstract_en',
            'original_name', 'original_language', 'original_time',
            'original_source' ,'volume', 'issue', 'reviewer_id'
        ]
        # 遍历属性列表，仅当变量值不为空时更新corpus对象的属性
        for attr in attributes:
            value = locals().get(attr)
            if value: setattr(corpus, attr, value)

        # 修改语料所属的数据库
        if database_ids and len(database_ids) > 0:
            current_corpus_database_ids = [data[0] for data in
                                           CorpusDatabase.query.with_entities(CorpusDatabase.database_id).filter(
                                               CorpusDatabase.corpus_id == corpus.id).all()]
            intersection = set(current_corpus_database_ids) & set(database_ids)
            current_corpus_database_ids = [num for num in current_corpus_database_ids if num not in intersection]
            database_ids = [num for num in database_ids if num not in intersection]
            CorpusDatabase.query.filter(CorpusDatabase.database_id.in_(current_corpus_database_ids),
                                        CorpusDatabase.corpus_id == corpus.id).delete(
                synchronize_session=False)
            corpus_databases = []
            for db_id in database_ids:
                if Database.query.filter(Database.id == db_id, Database.is_deleted == 0).count() == 0: return jsonify(
                    {"code": 404, "error": f'无id为{db_id}的数据库'}), 404
                corpus_database = CorpusDatabase(corpus_id=corpus.id, database_id=db_id)
                corpus_databases.append(corpus_database)
            db.session.add_all(corpus_databases)

        # 关键词
        for keywords, attr_name, language in [(keyword_cn, 'keyword_cn', KeywordLanguage.CN.value),
                                              (keyword_en, 'keyword_en', KeywordLanguage.EN.value)]:
            # if keywords and len(keywords) > 0:
            # 旧的关键词
            current_keywords = json.loads(getattr(corpus, attr_name)) if getattr(corpus, attr_name) else []
            # 旧关键词和新关键词共有的部分
            intersection = set(keywords) & set(current_keywords)
            # 旧关键词除公共部分外的关键词
            current_keywords = [keyword for keyword in current_keywords if keyword not in intersection]
            # 新关键词除公共部分外的关键词
            keywords = [keyword for keyword in keywords if keyword not in intersection]
            # 若频率为0，则直接设置is_deleted=1
            Keywords.query.filter(Keywords.is_deleted == 0, Keywords.keyword.in_(current_keywords),
                                  Keywords.language == language, Keywords.frequency == 0).update(
                {Keywords.is_deleted: 1})
            # 将取出的旧关键词频率-1
            Keywords.query.filter(Keywords.is_deleted == 0, Keywords.keyword.in_(current_keywords),
                                  Keywords.language == language).update(
                {Keywords.frequency: func.greatest(Keywords.frequency - 1, 0)})
            new_keywords = []
            cite_keyword_ids = []
            for keyword in keywords:
                current_keyword = Keywords.query.filter(Keywords.keyword == keyword, Keywords.is_deleted == 0,
                                                        Keywords.language == language).first()
                if current_keyword:
                    cite_keyword_ids.append(current_keyword.id)
                else:
                    new_keyword = Keywords(keyword=keyword, flag=KeywordFlag.COLLECTOR_ADDED.value, language=language)
                    new_keywords.append(new_keyword)
            db.session.add_all(new_keywords)
            # 新关键词的频率+1
            Keywords.query.filter(Keywords.is_deleted == 0, Keywords.id.in_(cite_keyword_ids)).update(
                {Keywords.frequency: Keywords.frequency + 1})
            # 设置关键字
            saved_keywords = keywords + list(intersection)
            setattr(corpus, attr_name, json.dumps(saved_keywords, ensure_ascii=False))

        # 3. 校验保留附件属于当前文献，再删除移除项并上传新文件
        # 这些远程操作不可随数据库回滚撤销，失败不能描述为自动恢复原附件。
        attachment = data.getlist('attachment')
        # TODO 删除文件后要重新计算字数
        # 删除旧文件
        current_attachment = json.loads(corpus.attachment) if corpus.attachment else []
        current_attachment_path = [item['file_path'] for item in current_attachment]
        for file_path in attachment:
            if file_path not in current_attachment_path: return jsonify({"code": 400, "error": "语料旧附件地址参数错误"})
        removed_attachment = []
        preserved_attachment = []
        # 减去被删除文件的字数
        for item in current_attachment:
            if item['file_path'] not in attachment:
                removed_attachment.append(item['file_path'])
                corpus.word_number -= int(item['word'])
            else: preserved_attachment.append(item)
        # removed_attachment = [item for item in current_attachment if item not in attachment]
        delete_file(removed_attachment)

        # if removed_attachment:
        #     try:
        #         delete_by_file(
        #             corpus_id=corpus.id,
        #             file_names=[os.path.basename(path) for path in removed_attachment]
        #         )
        #     except Exception as e:
        #         db.session.rollback()
        #         return jsonify({"code": 500, "error": str(e)}), 500

        # 上传新文件
        files = request.files.getlist('file')
        upload_result = []
        if files and len(files) > 0:
            file_list = []
            # 统计语料字
            for file in files:
                original_pos = file.tell()
                words = count_words(file)
                file.seek(original_pos)
                corpus.word_number = corpus.word_number + words if corpus.word_number else words
                file_list.append((file, words))
                file_content = get_content(file)
                file.seek(original_pos)
                # try:
                #     index_documents(
                #         corpus_id=corpus.id,
                #         file_content=file_content,
                #         file_name=file.filename,
                #     )
                # except Exception as e:
                #     db.session.rollback()
                #     return jsonify({"code": 500, "error": str(e)}), 500

            upload_result, code = upload_file(file_list, 'corpus', user.username)
            if code != 200: return jsonify({"code": code, "error": upload_result}), code

        # 修改附件路径
        # 4. 将附件新快照、索引失效及重新待审状态在同一数据库事务中提交
        # 先推进旧任务版本，再让新附件可见，防止在途执行者发布旧原件的索引。
        new_attachment = preserved_attachment + upload_result
        attachment_changed = current_attachment_path != [item['file_path'] for item in new_attachment]
        corpus.attachment = json.dumps(new_attachment, ensure_ascii=False)
        if attachment_changed:
            corpus.is_vectorized = False
            cancel_corpus_index(db.session, corpus.id)
        # 草稿状态修改后变为待审核（前提是已关联数据库）
        if corpus.status == CorpusStatus.DRAFT.value:
            if CorpusDatabase.query.filter(CorpusDatabase.corpus_id == corpus.id).first():
                corpus.status = CorpusStatus.WAITING_EXAMINE.value
        # 审核通过的语料编辑后需要重新审核
        elif corpus.status == CorpusStatus.APPROVE_EXAMINE.value:
            corpus.status = CorpusStatus.WAITING_EXAMINE.value
        db.session.commit()
        if attachment_changed:
            try:
                # 5. 提交后尽力清理远程旧索引，清理失败记录日志，不撤销已保存的附件修改
                from app.utils.qdrant_utils import delete_corpus_chunks
                delete_corpus_chunks(corpus.id)
            except Exception:
                current_app.logger.exception("附件已变更但旧向量片段清理失败: corpus_id=%s", corpus.id)
    except requests.exceptions.RequestException as e:
        db.session.rollback()
        return jsonify({"code": 400, "error": str(e)}), 400
    return jsonify({"code": 200, "success": "修改语料成功"})


# TODO 上传语料审核（status=1）
@bp.route('/examine', methods=['GET'])
@permission_required(['COLLECTOR'])
def corpus_review():
    """将本人编辑中、驳回或草稿文献提交待审核；要求已关联主题数据库。"""
    corpus_id = request.args.get('corpus_id')
    user = g.current_user
    try:
        corpus = Corpus.query.filter(Corpus.id == corpus_id, Corpus.is_deleted == 0,
                                     or_(Corpus.status == CorpusStatus.EDITING.value,
                                         Corpus.status == CorpusStatus.REJECT_EXAMINE.value,
                                         Corpus.status == CorpusStatus.DRAFT.value),
                                     Corpus.contributor == user.id).first()
        if not corpus: return jsonify({"code": 404, "error": "您无该编辑中/审核驳回/草稿语料记录"}), 404
        if not CorpusDatabase.query.filter(CorpusDatabase.corpus_id == corpus.id).first():
            return jsonify({"code": 400, "error": "文献未关联数据库，无法提交审核"}), 400
        corpus.status = CorpusStatus.WAITING_EXAMINE.value
        corpus.submit_time = datetime.now()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 400, "error": str(e)}), 400

    return jsonify({"code": 200, "success": "成功请求语料审核"})


# TODO 撤销语料审核（status=0）
@bp.route('/withdraw', methods=['GET'])
@permission_required(['COLLECTOR'])
def corpus_withdraw():
    """将本人待审核文献退回编辑中，不改变已经通过或驳回的记录。"""
    corpus_id = request.args.get('corpus_id')
    user = g.current_user
    try:
        corpus = Corpus.query.filter(Corpus.id == corpus_id, Corpus.is_deleted == 0,
                                     Corpus.status == CorpusStatus.WAITING_EXAMINE.value,
                                     Corpus.contributor == user.id).first()
        if not corpus: return jsonify({"code": 404, "error": "您无该待审核语料记录"}), 404
        corpus.status = CorpusStatus.EDITING.value
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 400, "error": str(e)}), 400

    return jsonify({"code": 200, "success": "撤销语料审核成功"})


# TODO 查询自己的语料（可添加筛选条件）
@bp.route('/get', methods=['GET'])
@permission_required(['COLLECTOR'])
def corpus_get():
    """按本人身份和元数据筛选文献，按提交时间倒序返回分页。"""
    data = request.args
    corpus_id = data.get('corpus_id')
    title_cn = data.get('title_cn')
    title_en = data.get('title_en')
    status = data.get('status')
    category_id = data.get('category_id')
    language = data.get('language')
    author = data.get('author')
    page_size = data.get('page_size', default=10, type=int)
    page_num = data.get('page_num', default=1, type=int)
    filters = [Corpus.owner_id.is_(None), Corpus.status != 4, Corpus.is_deleted == 0, Corpus.contributor == g.current_user.id]
    if corpus_id: filters.append(Corpus.id == corpus_id)
    if title_cn: filters.append(Corpus.title_cn.like(f'%{title_cn}%'))
    if title_en: filters.append(Corpus.title_en.like(f'%{title_en}%'))
    if status: filters.append(Corpus.status == status)
    if category_id: filters.append(Corpus.category_id == category_id)
    if language: filters.append(Corpus.language == language)
    if author: filters.append(or_(Corpus.author.like(f"%{author}%"), Corpus.translator.like(f"%{author}%"),
                                  Corpus.editor.like(f"%{author}%")))

    corpus_page = db.session.query(Corpus.id, Corpus.title_cn, Corpus.title_en, Corpus.status, Corpus.language,
                                   CorpusCategory.category_name,Corpus.author, Corpus.keyword_cn,
                                   Corpus.keyword_en, Corpus.publish_time, Corpus.word_number).outerjoin(CorpusCategory,
                                                                                                         Corpus.category_id == CorpusCategory.id).filter(
        and_(*filters)).order_by(Corpus.submit_time.desc()).paginate(page=page_num, per_page=page_size, error_out=False)

    res = []
    for corpus_id, title_cn, title_en, status, language, category_name, author, keyword_cn, keyword_en, publish_time, word_number in corpus_page.items:
        res.append({
            'corpus_id': corpus_id,
            'title_cn': title_cn,
            'title_en': title_en,
            'status': status,
            'language': language,
            'category_name': category_name,
            'author': author,
            'keyword_cn': json.loads(keyword_cn) if keyword_cn else None,
            'keyword_en': json.loads(keyword_en) if keyword_en else None,
            'publish_time': publish_time,
            'word_number': word_number
        })
    return jsonify({
        'code': 200,
        'data': {
            'total': corpus_page.total,
            'pages': corpus_page.pages,
            'data': res
        }
    })


# TODO 删除语料（审核通过的材料就不能删除）
@bp.route('/delete', methods=['GET'])
@permission_required(['COLLECTOR'])
def corpus_delete():
    """删除本人尚未通过审核的文献，先尽力删除附件，再提交软删除。

    附件解析或远程删除失败仅记录日志，不阻止逻辑删除；文件操作不能由数据库回滚。

    Returns:
        删除结果 JSON 或记录、事务错误响应。
    """
    corpus_id = request.args.get('corpus_id')
    user = g.current_user
    try:
        corpus = Corpus.query.filter(Corpus.id == corpus_id, Corpus.is_deleted == 0, Corpus.status.in_(
            [CorpusStatus.DRAFT.value, CorpusStatus.EDITING.value, CorpusStatus.WAITING_EXAMINE.value, CorpusStatus.REJECT_EXAMINE.value]),
                                     Corpus.contributor == user.id).first()
        if not corpus: return jsonify({"code": 404, "error": "您无该可删除的语料记录"}), 404
        # 优先删除文件服务器上的附件
        if corpus.attachment:
            try:
                att_list = json.loads(corpus.attachment)
                file_paths = []
                for item in att_list:
                    if isinstance(item, dict) and item.get("file_path"):
                        file_paths.append(item["file_path"])
                if file_paths:
                    try:
                        delete_file(file_paths)
                    except Exception as e:
                        current_app.logger.error(f"删除语料附件文件失败 corpus_id={corpus.id}: {str(e)}")
            except Exception as e:
                current_app.logger.warning(f"解析语料附件失败 corpus_id={corpus.id}: {str(e)}")

        corpus.is_deleted = 1
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"code": 400, "error": str(e)}), 400

    return jsonify({"code": 200, "success": "删除语料成功"})


# 获取当前用户最新草稿
@bp.route('/get_draft', methods=['GET'])
@permission_required(['COLLECTOR'])
def get_latest_draft():
    """返回本人 ID 最大的有效草稿，并补齐关键词 ID 和所属主题 ID 供表单恢复。"""
    user = g.current_user
    draft = db.session.query(Corpus).filter(
        Corpus.is_deleted == 0,
        Corpus.contributor == user.id,
        Corpus.status == CorpusStatus.DRAFT.value
    ).order_by(Corpus.id.desc()).first()
    if not draft:
        return jsonify({'code': 200, 'data': None})
    # 获取所有关联的database_id
    database_ids = [row.database_id for row in db.session.query(CorpusDatabase.database_id).filter(CorpusDatabase.corpus_id == draft.id).all()]
    data = draft.to_dict(with_keyword_ids=True)
    data['database_ids'] = database_ids
    return jsonify({'code': 200, 'data': data})


# 生成摘要和关键词
@bp.route('/generate_summary', methods=['POST'])
@permission_required(['COLLECTOR', 'ADMIN'])
def generate_summary():
    """读取单个上传文档，逐段生成摘要后汇总，拆出摘要正文与关键词。

    Returns:
        包含 abstract 和按数量截取 keywords 的 JSON，或文件及模型错误响应。
    """
    data = request.form
    abstract_words_num = data.get('abstract_words_num',  type=int, default=800)
    keywords_num = data.get('keywords_num',  type=int, default=6)
    file = request.files.get('file')

    if not file:
        return jsonify({"code": 400, "error": "请上传一个文件"}), 400
    if len(request.files) != 1:
        return jsonify({"code": 400, "error": "只能上传一个文件"}), 400

    client = init_openai_client("sk-99b22e1a511b4d14b2a7ed12cf690b18")

    try:
        # 语料内容
        paper_content = get_content(file)
        if not paper_content:
            return jsonify({"code": 400, "error": "未获取到文件内容"}), 400

        # 分段
        segments = split_text_into_segments(paper_content)

        # 生成分段摘要
        segment_summaries = generate_summary_for_segments(client, segments)

        # 生成总摘要
        total_summary_result = generate_total_summary(client, segment_summaries, abstract_words_num, keywords_num)

    except Exception as e:
        return jsonify({"code": 400, "error": str(e)}), 400

    # 分割摘要和关键词
    if "关键词：" in total_summary_result:
        abstract_part, keyword_part = total_summary_result.split("关键词：", 1)
        keywords = [k.strip() for k in keyword_part.split('；') if k.strip()]
    else:
        abstract_part = total_summary_result
        keywords = []
    return jsonify({
        "code": 200,
        "data": {
            "abstract": abstract_part.strip(),
            "keywords": keywords[:keywords_num]  # 确保数量符合要求
        }
    })


# 调用百度翻译API翻译
@bp.route('/translate_summary', methods=['POST'])
@permission_required(['COLLECTOR', 'ADMIN'])
def translate_summary():
    """调用既有翻译账户将非空摘要翻译为目标语言，服务无结果时返回错误响应。"""
    data = request.json
    text = data.get('text')
    source_language = data.get('source_language', 'en')
    target_language = data.get('target_language', 'zh')

    # 参数校验
    if not text or not text.strip():
        return jsonify({"code": 400, "error": "翻译文本不能为空"}), 400
    
    # 配置信息 - 多个账号配置
    accounts = [
        {
            "app_id": "20250915002455002",
            "secret_key": "YG14pmLfoJqzfndudgml"
        }
    ]

    try:
        # 依次尝试各个账号进行翻译
        translated_text = None
        for account in accounts:
            translated_text = translate_text(
                account["app_id"], 
                account["secret_key"], 
                text, 
                source_language, 
                target_language
            )
            # 如果翻译成功，跳出循环
            if translated_text is not None:
                break

        # 处理结果
        if translated_text is not None:
            return jsonify({"code": 200, "data": {"译文": translated_text}})
        else:
            return jsonify({"code": 400, "error": "翻译失败，请检查百度翻译API"}), 400
    except Exception as e:
        return jsonify({
            "code": 500, 
            "error": "翻译服务内部错误",
            "detail": str(e)
        }), 500


# 关键词翻译接口
@bp.route('/keywords_translate', methods=['POST'])
@permission_required(['COLLECTOR', 'ADMIN'])
def keywords_translate():
    """逐个翻译英文逗号分隔的关键词，返回按中文逗号拆分的译文数组。

    单个词翻译失败时辅助函数保留原词；整体输入或服务失败返回错误响应。

    Returns:
        含译文数组的 JSON 或参数、服务错误。
    """
    data = request.json
    text = data.get('text')
    source_language = data.get('source_language', 'en')
    target_language = data.get('target_language', 'zh')

    # 参数校验
    if not text or not text.strip():
        return jsonify({"code": 400, "error": "翻译文本不能为空"}), 400
    
    # 配置信息
    accounts = [
        {
            "app_id": "20250915002455002",
            "secret_key": "YG14pmLfoJqzfndudgml"
        }
    ]

    try:
        # 依次尝试各个账号进行翻译
        translated_text = None
        for account in accounts:
            translated_text = translate_keywords(
                account["app_id"], 
                account["secret_key"], 
                text, 
                source_language, 
                target_language
            )
            if translated_text is not None:
                break

        # 处理结果
        if translated_text is not None:
            # 将逗号分隔的结果转为数组
            translated_array = translated_text.split('，') if translated_text else []
            return jsonify({"code": 200, "data": {"译文": translated_array}})
        else:
            return jsonify({"code": 400, "error": "翻译失败，请检查百度翻译API"}), 400
    except Exception as e:
        return jsonify({
            "code": 500, 
            "error": "翻译服务内部错误",
            "detail": str(e)
        }), 500

# 配置通义千问API客户端
# PDF信息提取接口（单文件版本，保持向后兼容）
@bp.route('/extract_pdf_info', methods=['POST'])
@permission_required(['COLLECTOR'])
def extract_pdf_info():
    """校验单份 PDF 的类型与大小，临时落盘后调用带缓存及重试的题录提取。

    Returns:
        提取数据及原模型响应 JSON；refresh 可绕过缓存读取，文件退出上下文后清理。
    """
    import tempfile
    import gc
    import contextlib
    
    @contextlib.contextmanager
    def safe_temp_file(file_obj):
        """临时保存上传 PDF，处理结束后尽力删除本地副本。

        Args:
            file_obj: 支持 save 的上传文件对象。

        Yields:
            仅在当前上下文内有效的 PDF 路径。
        """
        tmp_file_path = None
        try:
            # 使用delete=True确保自动清理
            with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf') as tmp_file:
                file_obj.save(tmp_file.name)
                tmp_file_path = tmp_file.name
                yield tmp_file_path
        finally:
            # 双重保险：手动清理
            if tmp_file_path and os.path.exists(tmp_file_path):
                try:
                    os.unlink(tmp_file_path)
                except Exception as e:
                    current_app.logger.error(f"清理临时文件失败: {e}")
    
    try:
        # 检查是否有文件上传
        # 1. 校验 PDF、文件大小与模型凭据，读取显式刷新开关
        if 'file' not in request.files:
            return jsonify({"code": 400, "error": "请上传PDF文件"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"code": 400, "error": "未选择文件"}), 400
        
        # 检查文件类型
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"code": 400, "error": "只支持PDF文件"}), 400
        
        # 检查文件大小（限制为200MB）
        file.seek(0, 2)  # 移动到文件末尾
        file_size = file.tell()
        file.seek(0)  # 重置到文件开头
        
        if file_size > 200 * 1024 * 1024:  # 200MB
            return jsonify({"code": 400, "error": "文件大小不能超过200MB"}), 400
        
        # 获取API密钥（从环境变量或配置中获取）
        qwen_api_key = os.getenv('QWEN_API_KEY')
        if not qwen_api_key:
            return jsonify({"code": 500, "error": "未配置通义千问API密钥"}), 500
        
        # 初始化客户端
        client = setup_qwen_client(qwen_api_key)

        # refresh=1：强制重新提取（跳过缓存读取，提取成功后覆盖缓存）
        force_refresh = str(request.form.get('refresh', '')).strip().lower() in ('1', 'true', 'yes')

        # 使用安全的临时文件管理
        # 2. 在可清理的本地副本上执行题录提取，结果按 success 契约映射为接口响应
        with safe_temp_file(file) as tmp_file_path:
            # 提取PDF信息
            result = robust_extract_pdf_info(client, tmp_file_path, use_cache=not force_refresh)
            
            # 强制垃圾回收
            gc.collect()
            
            if result["success"]:
                return jsonify({
                    "code": 200,
                    "data": result["extracted_data"],
                    "raw_response": result.get("raw_response", "")
                })
            else:
                error_message = result.get("error") or "通义千问API调用失败"
                current_app.logger.error(f"阿里云API错误: {error_message}")
                return jsonify({
                    "code": 400,
                    "error": error_message
                }), 400
                
    except Exception as e:
        current_app.logger.error(f"PDF信息提取失败: {str(e)}")
        gc.collect()
        return jsonify({
            "code": 500,
            "error": f"系统错误: {str(e)}"
        }), 500


# 批量PDF信息提取接口（支持最多20篇同时上传）
@bp.route('/extract_pdf_info_batch', methods=['POST'])
@permission_required(['COLLECTOR'])
def extract_pdf_info_batch():
    """并发提取多份 PDF，逐篇保存草稿并返回按输入顺序排列的结果。

    每个线程使用自己的 Flask 应用上下文和数据库会话；失败只影响对应文件。
    refresh 控制题录缓存读取，不改变现有提示词或索引 OCR 配置。

    Returns:
        成功与失败汇总、逐文件结果及草稿 ID 列表，或整批校验错误响应。
    """
    import tempfile
    import gc
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    MAX_FILES = 20
    MAX_WORKERS = 10  # 并行线程数
    
    def process_single_file(args):
        """在线程独立上下文中完成题录提取、题名去重、草稿创建与附件上传。

        Args:
            args: 应用、文件、输入序号、用户及审核人、凭据、用户名称和刷新开关的元组。

        Returns:
            保留输入序号及文件名的结果字典；失败回滚本篇事务，成功返回草稿 ID。
        """
        # 1. 在线程内绑定独立应用上下文，准备本篇文件及其用户、审核人和输入序号
        app, file_obj, file_idx, user_id, reviewer_id, api_key, username, force_refresh = args
        from app.models import db, Corpus
        from app.utils.enums import CorpusStatus
        import json, re
        import logging
        
        logger = logging.getLogger('werkzeug')
        
        def extract_year(val):
            """按当前正则解析题录年份，空值或不匹配时返回 None。"""
            if val is None or val == '':
                return None
            match = re.match(r"(\d{4})", str(val))
            if match:
                return int(match.group(1))
            return None
        
        result = {'index': file_idx, 'filename': file_obj.filename, 'success': False}
        
        with app.app_context():
            try:
                # 先记录文件流的当前位置，以便恢复
                original_pos = file_obj.tell()
                
                client = setup_qwen_client(api_key)
                
                # 2. 将本篇 PDF 临时落盘进行题录提取；退出后清理副本并恢复上传流游标
                with tempfile.NamedTemporaryFile(delete=True, suffix='.pdf') as tmp:
                    tmp.write(file_obj.read())
                    tmp.flush()
                    tmp_path = tmp.name
                    try:
                        extract_result = robust_extract_pdf_info(client, tmp_path, use_cache=not force_refresh)
                    except Exception as e:
                        extract_result = {"success": False, "error": str(e)}
                
                # 恢复文件流位置，以便 process_file_upload 中的 count_words 能从头读取
                file_obj.seek(original_pos)
                
                if extract_result.get("success"):
                    data = extract_result.get("extracted_data", {})
                    title_cn = data.get('title_cn', '')
                    
                    if title_cn and title_cn.strip() != "":
                        duplicate_query = Corpus.query.filter(
                            Corpus.title_cn == title_cn,
                            Corpus.is_deleted == 0
                        )
                        if duplicate_query.count() > 0:
                            result['error'] = "原文标题重复"
                            return result
                    
                    # 3. 去重后创建草稿并 flush 取得 ID，关键词及附件引用与草稿一起提交
                    corpus = Corpus(
                        language=data.get('language', ''),
                        author=data.get('author', ''),
                        translator=data.get('translator', ''),
                        editor=data.get('editor', ''),
                        publish_time=extract_year(data.get('publish_time')),
                        ISSN=data.get('ISBN', ''),
                        source=data.get('source', ''),
                        category_id=None,
                        abstract_cn=data.get('abstract_cn', ''),
                        abstract_en=data.get('abstract_en', ''),
                        contributor=user_id,
                        title_cn=title_cn,
                        title_en=data.get('title_en', ''),
                        reviewer_id=reviewer_id,
                        status=CorpusStatus.DRAFT.value
                    )
                    db.session.add(corpus)
                    db.session.flush()
                    
                    kw_cn = data.get('keyword_cn', [])
                    kw_en = data.get('keyword_en', [])
                    if kw_cn:
                        corpus.keyword_cn = json.dumps(kw_cn, ensure_ascii=False)
                    if kw_en:
                        corpus.keyword_en = json.dumps(kw_en, ensure_ascii=False)
                    
                    # 上传文件到文件服务器（保存附件信息）
                    # 4. 上传本篇附件，失败回滚草稿；远程文件不受数据库事务回滚管理
                    upload_res = process_file_upload(file_obj, title_cn, username, corpus.id)
                    if upload_res.get("code") != 200:
                        db.session.rollback()
                        result['error'] = upload_res.get("error", "文件上传失败")
                        return result
                    
                    corpus.attachment = json.dumps(upload_res["result"], ensure_ascii=False)
                    corpus.word_number = upload_res.get("words", 0)
                    
                    db.session.commit()
                    result['success'] = True
                    result['extracted_data'] = data
                    result['draft_id'] = corpus.id
                else:
                    result['error'] = extract_result.get("error", "解析失败")
            
            except Exception as e:
                db.session.rollback()
                result['error'] = str(e)
            finally:
                gc.collect()
        
        return result
    
    try:
        # 检查文件上传
        # 1. 校验整批数量、格式、文件大小及模型配置，拒绝不完整批次输入
        if 'files' not in request.files:
            return jsonify({"code": 400, "error": "请上传PDF文件"}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({"code": 400, "error": "未选择文件"}), 400
        
        if len(files) > MAX_FILES:
            return jsonify({"code": 400, "error": f"最多支持{MAX_FILES}篇文献"}), 400
        
        # 验证文件
        file_list = []
        for file in files:
            if not file.filename:
                return jsonify({"code": 400, "error": "存在空文件名"}), 400
            if not file.filename.lower().endswith('.pdf'):
                return jsonify({"code": 400, "error": f"{file.filename} 不是PDF"}), 400
            file.seek(0, 2)
            if file.tell() > 200 * 1024 * 1024:
                return jsonify({"code": 400, "error": f"{file.filename} 超过200MB"}), 400
            file.seek(0)
            file_list.append(file)
        
        qwen_api_key = os.getenv('QWEN_API_KEY')
        if not qwen_api_key:
            return jsonify({"code": 500, "error": "未配置API密钥"}), 500
        
        user = g.current_user
        user_id = user.id
        username = user.username
        reviewer_id = db.session.query(User.leader_id).filter(
            User.id == user_id, User.is_deleted == 0
        ).scalar()
        
        current_app.logger.info(f"开始批量处理{len(file_list)}篇文献（并行{MAX_WORKERS}线程）...")
        
        # 构建线程参数（file 已 seek(0)，可重复读取）
        app = current_app._get_current_object()
        # refresh=1：强制重新提取（跳过缓存读取，提取成功后覆盖缓存）
        force_refresh = str(request.form.get('refresh', '')).strip().lower() in ('1', 'true', 'yes')
        # 2. 固定用户上下文参数，为每份文件创建独立工作项，禁止共享请求内数据库会话
        work_args = [
            (app, file, idx, user_id, reviewer_id, qwen_api_key, username, force_refresh)
            for idx, file in enumerate(file_list)
        ]
        
        results = []
        draft_ids = []
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_file, args): args for args in work_args}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as e:
                    args = futures[future]
                    result = {'index': -1, 'filename': args[1].filename, 'success': False, 'error': str(e)}
                results.append(result)
                if result.get('success'):
                    draft_ids.append({'draft_id': result.get('draft_id'), 'filename': result['filename']})
        
        # 按原始顺序排列
        # 3. 将完成顺序恢复为上传顺序，分别汇总草稿与失败文件，不让单篇失败覆盖整批
        results.sort(key=lambda x: x['index'])
        
        success_count = len(draft_ids)
        failed_count = len(results) - success_count
        failed_list = [
            {"filename": r["filename"], "error": r.get("error", "未知错误")}
            for r in results if not r.get("success")
        ]

        return jsonify({
            "code": 200,
            "summary": {"total": len(file_list), "success": success_count, "failed": failed_count},
            "results": results,
            "drafts": {"draft_ids": draft_ids},
            "failed_list": failed_list
        })
        
    except Exception as e:
        current_app.logger.error(f"批量处理失败: {str(e)}", exc_info=True)
        return jsonify({"code": 500, "error": str(e)}), 500
