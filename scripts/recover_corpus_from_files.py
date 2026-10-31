#!/usr/bin/env python3
"""
从服务器上的 corpus PDF 文件重新提取信息并写回 corpus 表（补全误删记录）。

使用方式（在项目根目录执行，如 cd ~/Foreign/Backend；服务器多为 python3）：
  python3 scripts/recover_corpus_from_files.py --dry-run --limit 2
  python3 scripts/recover_corpus_from_files.py --skip-existing

默认：contributor=26，PDF 目录=项目同级 ../File/uploaded_files/corpus；不统计字数（word_number 与 attachment.word 记为 0）。
通义千问 API 密钥从项目根目录 .env 中的 QWEN_API_KEY 读取。

注意：
  - 程序直接访问 --dir 指定目录（服务器上的 uploaded_files/corpus），无需手动下载文件。
  - 每次处理（成功/失败/跳过）都会写入 corpus_recovery_log 表，便于排查和重跑失败项。
  - 默认将恢复出的语料写入 corpus_database，database_id 固定为 198，后续再人工审核调整。
  - category_id 会尽量按提取的「文献类型」匹配 corpus_category，匹配不到则留空由人工填。
  - 若某 PDF 的 attachment 路径或文件名在 corpus 表中已存在，可用 --skip-existing 跳过，避免重复。
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import func

# 项目根目录加入 path，并加载项目 .env（避免 os.getenv 取不到）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except Exception:
    pass


def _load_exports_from_shell(script_path: str) -> None:
    """
    从 start.sh / start-dev.sh 这类脚本中读取 export KEY=VALUE 并写入 os.environ。
    仅作为 .env / 环境变量缺失时的兜底，避免你在服务器上额外手工设置。
    """
    try:
        if not script_path or not os.path.exists(script_path):
            return
        with open(script_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("export "):
                    continue
                kv = line[len("export ") :].strip()
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                k = k.strip()
                v = v.strip().strip(";")
                if not k:
                    continue
                # 去掉最外层引号
                if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                    v = v[1:-1]
                if k not in os.environ and v != "":
                    os.environ[k] = v
    except Exception:
        return


# 若未通过 .env/环境变量拿到配置，则从项目的 start.sh 兜底读取（“使用项目中的账号”）
_load_exports_from_shell(os.path.join(_PROJECT_ROOT, "start.sh"))


def extract_year(val):
    """按当前正则尝试从题录值开头提取年份，空值或不匹配时返回 None。"""
    if val is None or val == '':
        return None
    match = re.match(r"(\d{4})", str(val))
    if match:
        return int(match.group(1))
    return None


def get_category_id_by_name(db_session, category_name):
    """根据文献类型名称查询 corpus_category.id，未找到返回 None。"""
    if not category_name or not str(category_name).strip():
        return None
    from app.models import CorpusCategory
    c = db_session.query(CorpusCategory.id).filter(
        CorpusCategory.category_name == str(category_name).strip(),
        CorpusCategory.is_deleted == 0
    ).first()
    return c[0] if c else None


def normalized_keywords(val):
    """将提取结果中的 keyword_cn/keyword_en 转为 list of str。"""
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        return [x.strip() for x in val.replace("；", ";").split(";") if x.strip()]
    return []


def safe_str(val, max_len=None):
    """将非空值转为去空白文本，可按数据库列长度截断，空结果返回 None。"""
    if val is None:
        return None
    s = str(val).strip()
    if max_len and len(s) > max_len:
        return s[:max_len]
    return s if s else None


def setup_qwen_client(api_key: str):
    """配置通义千问兼容模式客户端（不依赖 collector.py，避免引入 redis 等无关依赖）。"""
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


def _extract_message_content(completion):
    """取首个模型候选的正文，返回 (正文, None) 或 (None, 错误原因)。"""
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return None, "通义千问API返回空的choices列表"
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None) if message else None
    if not content:
        return None, "通义千问API返回空的message内容"
    return content, None


def _parse_json_from_text(result_text: str):
    """从大模型文本响应中提取 JSON 内容；失败时返回 raw_output。"""
    if not result_text:
        return False, {"raw_output": ""}
    try:
        if "```json" in result_text:
            json_str = result_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in result_text:
            json_str = result_text.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            json_str = result_text.strip()

        # 清理无效控制字符（保留 \n, \r, \t）
        json_str = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", json_str)
        return True, json.loads(json_str)
    except Exception:
        return False, {"raw_output": result_text}


def extract_pdf_info_long(client, pdf_path: str, extraction_prompt: str = None):
    """上传恢复原件，以文件引用调用 qwen-long 并解析题录。

    Args:
        client: 已配置的模型客户端。
        pdf_path: 当前恢复文件的本地路径。

    Returns:
        含 success、extracted_data 和 raw_response 或 error 的结果字典。
        获得响应后尽力删除远程临时文件，异常返回失败供恢复日志记录。
    """
    if extraction_prompt is None:
        extraction_prompt = """
请从这篇学术论文中仅提取并返回一个严格的JSON对象，包含以下键，并遵循每项规则：
- 所有值用字符串类型；无法确定则用空字符串""，不要返回 null/None/[]。
- 仅输出JSON本体，不要包含解释、前后缀文字或Markdown代码块。

字段与规则：
- title_en: 中文标题（一律翻译为中文）
- language: 语言（如中文、英文、日语、韩语、俄语、西班牙语等；不要使用 zh/en 等缩写）
- author: 作者姓名（多个作者用中文逗号分隔）
- translator: 译者（无则空字符串）
- editor: 编者（无则空字符串）
- ISBN: 无则空字符串
- category: 文献类型，包括期刊文章, 学位论文, 会议论文, 专著, 报告, 专利, 标准, 报纸文章, 电子文献，如果不是以上类型则空字符串
- publish_time: 发表日期（仅年份，如"2023"，无法确定则空字符串）
- volume: 卷号（无则空字符串）
- issue: 期号（无则空字符串）
- source: 来源
- keyword_en: 英文关键词（将原文提取的关键词翻译为英文，以数组形式返回；数量不少于3个、不超过8个）
- keyword_cn: 中文关键词（若原文关键词非中文则翻译为中文，以数组形式返回；数量不少于3个、不超过8个）
- abstract_cn: 中文摘要（若原文非中文则翻译为中文；若无摘要则基于全文生成不超过300字中文摘要）
- abstract_en: 原文摘要（保持原文语言；若仅有中文摘要，则翻译为与原文相同语言；仍无法获得则空字符串）

输出要求：
- 仅输出包含上述所有键的JSON对象；键不可缺省。
""".strip()

    try:
        # 1. 上传原件取得远程文件 ID，使用既有恢复提示词构造模型输入
        file_object = client.files.create(file=Path(pdf_path), purpose="file-extract")

        # 单个文件的最长等待时间控制在约 60 秒内，避免长时间卡死
        max_wait_time = 60
        wait_interval = 3
        max_retries = max_wait_time // wait_interval

        messages = [
            {"role": "system", "content": "你是专业的信息分析助手，擅长从学术PDF中提取结构化信息。请准确、完整地提取所需信息。"},
            {"role": "system", "content": f"fileid://{file_object.id}"},
            {"role": "user", "content": extraction_prompt},
        ]

        # 2. 同步调用模型提取题录，网络或结构错误统一转为失败字典
        completion = None
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model="qwen-long",
                    messages=messages,
                    temperature=0.1,
                    stream=False,
                )
                break
            except Exception as api_error:
                error_str = str(api_error)
                if "File parsing in progress" in error_str or "parsing in progress" in error_str.lower():
                    if attempt < max_retries - 1:
                        time.sleep(wait_interval)
                        continue
                    raise Exception("文件解析超时，请稍后重试或使用较小的文件")
                raise

        if completion is None:
            return {"success": False, "error": "API调用失败：无法获取响应", "file_path": pdf_path}

        # 3. 验证首候选内容，随后尽力清理远程文件并解析返回 JSON
        result_text, err = _extract_message_content(completion)
        if err:
            return {"success": False, "error": err, "raw_response": getattr(completion, "__dict__", str(completion))}

        # 清理上传文件（失败不影响主流程）
        try:
            client.files.delete(file_object.id)
        except Exception:
            pass

        _, result_data = _parse_json_from_text(result_text)
        return {"success": True, "extracted_data": result_data, "raw_response": result_text}

    except Exception as e:
        return {"success": False, "error": str(e), "file_path": pdf_path}


def robust_extract_pdf_info(client, pdf_path: str, max_retries: int = 1):
    """
    带重试机制的 PDF 信息提取（仅走 qwen-long）。
    为了避免单个文件长时间阻塞，这里将重试次数控制为 1，
    每次调用内部的 extract_pdf_info_long 也有约 60 秒的超时控制，
    故单个文件理论最长处理时间约为 1 分钟多一点。
    """
    last = None
    for attempt in range(max_retries):
        try:
            last = extract_pdf_info_long(client, pdf_path)
            if last.get("success"):
                return last
            break
        except Exception as e:
            last = {"success": False, "error": str(e), "file_path": pdf_path}
            break
    return last or {"success": False, "error": "未知错误", "file_path": pdf_path}


def _extract_one_worker(full_path: str) -> dict:
    """
    供多进程调用：在子进程中创建 client 并做一次提取，仅做 API 调用，不碰 DB。
    必须在模块顶层定义以便 multiprocessing 可 pickle。
    """
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        load_dotenv(os.path.join(_root, ".env"))
    except Exception:
        pass
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        return {"success": False, "error": "QWEN_API_KEY not set", "file_path": full_path}
    client = setup_qwen_client(api_key)
    return robust_extract_pdf_info(client, full_path)


def run(app, args):
    """从既有 PDF 恢复题录、主题归属和审核状态，主进程逐篇提交记录与日志。

    Args:
        app: 调用方已建立上下文的最小 Flask 应用。
        args: 目录、贡献者、预览、数量、顺序及并行配置。

    Returns:
        配置失败时为 1，预览完成为 0，正常实际处理完成沿用隐式 None。

    启动阶段会先修复历史重命名路径，当前调用位置不受 dry_run 保护；
    因此预览仅跳过后续提取和新增，不保证整个命令没有数据库写入。
    反向遍历没有跨进程互斥，也不保证与正序运行互不重叠。
    实际写库按篇提交，batch_size 参数当前不改变提交频率。
    """
    from app.exts import db
    from app.models import Corpus, CorpusCategory, CorpusRecoveryLog, CorpusDatabase
    from app.utils.enums import CorpusStatus

    #
    # ---------------------- 分类与规则相关的辅助函数 ----------------------
    #

    # 规则中涉及的关键字统一在这里维护，方便后期扩展 / 修改
    FOCUS_PERSON_KEYWORDS_CN = ["陶行知", "杜威"]  # 规则1：聚焦陶行知和杜威的中文文献
    EDU_THOUGHT_KEYWORDS_CN = ["杜威", "书院", "毛泽东", "陶行知", "孔子", "蒙学"]  # 规则2

    CONFUCIUS_KEYWORDS_EN = ["confucius", "confucian", "confucianism"]  # 规则3

    # 规则4：俄文和韩文（朝鲜语）文献要放入的子库 id（见“database表内容”）
    FOREIGN_EDU_DB_IDS = {
        207,  # 国外教育文献数据库（汉语、英语、俄语、日语、法语、朝语、西班牙语）
        221,  # 国外外语教育数据库（汉语、英语、俄语、日语、法语、朝鲜语、西语）
        222,  # 国外外语教育理论文献库
        229,  # 国外研究文献库
    }

    # 规则5：涉及基督教、种族、港澳台等的敏感关键词，需要强制设为“待审核”
    SENSITIVE_REVIEW_KEYWORDS = [
        "基督教", "基督徒", "基督宗教",
        "宗教", "伊斯兰", "佛教",  # 可按需增减
        "种族", "族群", "族裔", "种族主义", "反种族主义",
        "港澳台", "香港", "澳门", "台湾", "台独", "港独",
    ]

    # 规则6：与“中国教育政策国际传播多语种数据库”相关的关键词，从外部文件导入
    def load_policy_related_keywords():
        """
        从项目根目录的“ 中国教育政策国际传播多语种数据库相关关键词 ”文件中读取关键词列表，
        忽略空行和“关键词：”等说明行。
        """
        keywords = set()
        try:
            # 文件已在项目根目录下，通过相对 _PROJECT_ROOT 定位
            keyword_file = os.path.join(_PROJECT_ROOT, "中国教育政策国际传播多语种数据库相关关键词")
            if not os.path.exists(keyword_file):
                return keywords
            with open(keyword_file, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip()
                    if not w or w.startswith("关键词"):
                        continue
                    keywords.add(w.lower())
        except Exception:
            # 关键词文件缺失或解析异常时，直接忽略，不影响主流程
            return keywords
        return keywords

    POLICY_DB_KEYWORDS = load_policy_related_keywords()

    # “中国教育思想文献数据库（汉语）” id（规则1/2）
    CN_EDU_THOUGHT_DB_ID = 199
    # “中国教育思想国际传播多语种数据库（汉语、英语、俄语、日语、法语、韩语、西语）” id（规则3）
    CN_EDU_THOUGHT_INTL_DB_ID = 203
    # “中国教育政策国际传播多语种数据库（汉语、英语、俄语、日语、法语、朝语、西语）” id（规则6 推荐）
    CN_POLICY_INTL_DB_ID = 211
    # 规则7：中国外语教育数据库（汉语）id=215
    CN_FL_EDU_DB_ID = 215
    # 规则8：中国外语教育国际传播数据库（汉语、英语、俄语、日语、法语、朝鲜语、西语）id=227
    CN_FL_EDU_INTL_DB_ID = 227

    # 规则7：子库“中国外语教育数据库（汉语）”中文关键词，命中则归入 215，直接审核通过
    CN_FL_EDU_KEYWORDS_CN = [
        "续理论", "续\"理论", "写长法", "产出导向法", "英语课程标准", "教学指南",
        "英语口语", "英语写作", "英语阅读", "英语听力", "英语学习者", "任务型教学", "英语学习活动观",
    ]
    # 规则8：子库“中国外语教育国际传播数据库”关键词（英文），命中则归入 227，直接审核通过
    CN_FL_EDU_INTL_KEYWORDS_EN = [
        "xu theory", "product-oriented approach", "product-orientedapproach",
        "chinese efl learners",
    ]

    def _to_lower_list(items):
        """忽略空值，将剩余项转换为小写文本列表，供恢复规则匹配使用。"""
        return [str(x).lower() for x in items if x]

    def extract_all_text_fields(title_cn, title_en, keyword_cn_list, keyword_en_list, abstract_cn, abstract_en):
        """合并双语题名、关键词和摘要，为顺序规则提供统一匹配文本。

        Args:
            title_cn: 原文题名字段。
            title_en: 译文题名字段。
            keyword_cn_list: 中文关键词列表。
            keyword_en_list: 外文关键词列表。
            abstract_cn: 中文摘要。
            abstract_en: 外文摘要。

        Returns:
            中文字段合并文本、外文字段合并文本、两者合并小写文本的三元组；第三项不是集合。
        """
        cn_parts = []
        en_parts = []
        if title_cn:
            cn_parts.append(title_cn)
        if abstract_cn:
            cn_parts.append(abstract_cn)
        if keyword_cn_list:
            cn_parts.extend(keyword_cn_list)

        if title_en:
            en_parts.append(title_en)
        if abstract_en:
            en_parts.append(abstract_en)
        if keyword_en_list:
            en_parts.extend(keyword_en_list)

        all_cn = " ".join([str(x) for x in cn_parts if x])
        all_en = " ".join([str(x) for x in en_parts if x])
        # 为了方便中英文混合场景，同时准备一个全部小写的 token 聚合字符串
        all_text = " ".join([all_cn, all_en]).lower()
        return all_cn, all_en, all_text

    def match_policy_db(all_text_lower):
        """判断合并小写文本是否命中政策主题词；这里只返回布尔值，审核终态由完整规则链决定。"""
        if not POLICY_DB_KEYWORDS:
            return False
        for kw in POLICY_DB_KEYWORDS:
            if kw and kw in all_text_lower:
                return True
        return False

    def contains_any(text, keywords, case_insensitive=True):
        """简单的“包含任一关键词”判断。"""
        if not text or not keywords:
            return False
        t = text.lower() if case_insensitive else text
        for kw in keywords:
            if not kw:
                continue
            k = kw.lower() if case_insensitive else kw
            if k in t:
                return True
        return False

    def classify_databases_and_status(language, title_cn, title_en, keyword_cn_list, keyword_en_list,
                                      abstract_cn, abstract_en):
        """依次应用恢复分类规则，合并主题集合并由后续命中规则覆盖审核状态。

        Args:
            language: 文献语言。
            title_cn: 原文题名字段。
            title_en: 译文题名字段。
            keyword_cn_list: 中文关键词列表。
            keyword_en_list: 外文关键词列表。
            abstract_cn: 中文摘要。
            abstract_en: 外文摘要。

        Returns:
            主题 ID 集合与最终审核整数状态。政策关键词先设待审，但后续语言或主题规则
            可能改为通过；敏感词最后强制待审。无归属时回退到默认库并保持待审。
        """
        # 1. 从待审核和空归属开始，合并双语匹配文本供后续规则依次判断
        database_ids = set()
        status = CorpusStatus.WAITING_EXAMINE.value  # 默认待审核（相比人工上传更安全）

        lang = (language or "").strip()
        all_cn, all_en, all_text_lower = extract_all_text_fields(
            title_cn, title_en, keyword_cn_list, keyword_en_list, abstract_cn, abstract_en
        )

        # ---------- 规则6：中国教育政策国际传播多语种数据库 ----------
        # 2. 依次执行主题和语言规则；归属集合累计，审核状态可能被后续规则覆盖
        if match_policy_db(all_text_lower):
            database_ids.add(CN_POLICY_INTL_DB_ID)
            status = CorpusStatus.WAITING_EXAMINE.value

        # ---------- 规则1/2：陶行知、杜威、书院、孔子、蒙学等（中文文献） ----------
        if lang == "中文":
            # 规则1：聚焦“陶行知、杜威”的中文文献 -> 教育思想文献子库，且可直接审核通过
            if contains_any(all_cn, FOCUS_PERSON_KEYWORDS_CN, case_insensitive=False):
                database_ids.add(CN_EDU_THOUGHT_DB_ID)
                status = CorpusStatus.APPROVE_EXAMINE.value
            # 规则2：标题/关键词中含有指定关键词的中文文献 -> 教育思想文献子库
            if contains_any(all_cn, EDU_THOUGHT_KEYWORDS_CN, case_insensitive=False):
                database_ids.add(CN_EDU_THOUGHT_DB_ID)
                # 特殊：毛泽东教育思想的文献需要再审核
                if "毛泽东" in all_cn:
                    status = CorpusStatus.WAITING_EXAMINE.value
                else:
                    # 若之前未被其他规则设置为更“严格”的状态，可以直接审核通过
                    if status != CorpusStatus.WAITING_EXAMINE.value:
                        status = CorpusStatus.APPROVE_EXAMINE.value

        # ---------- 规则3：Confucius/Confucian* 等英文文献 ----------
        if lang and lang.lower() in ["英文", "英语", "english"]:
            if contains_any(all_en, CONFUCIUS_KEYWORDS_EN, case_insensitive=True):
                database_ids.add(CN_EDU_THOUGHT_INTL_DB_ID)
                status = CorpusStatus.APPROVE_EXAMINE.value

        # ---------- 规则4：俄文和韩文（朝鲜语）文献 ----------
        if lang in ["俄语", "俄文", "俄文文献", "俄文资料"] or lang.lower().startswith("russian"):
            database_ids.update(FOREIGN_EDU_DB_IDS)
            status = CorpusStatus.APPROVE_EXAMINE.value
        if lang in ["韩语", "朝鲜语", "朝语"] or lang.lower().startswith("korean"):
            database_ids.update(FOREIGN_EDU_DB_IDS)
            status = CorpusStatus.APPROVE_EXAMINE.value

        # ---------- 规则7：中国外语教育数据库（汉语）— 中文关键词命中则归入 215，直接审核通过 ----------
        if contains_any(all_cn, CN_FL_EDU_KEYWORDS_CN, case_insensitive=False):
            database_ids.add(CN_FL_EDU_DB_ID)
            status = CorpusStatus.APPROVE_EXAMINE.value

        # ---------- 规则8：中国外语教育国际传播数据库 — 英文关键词命中则归入 227，直接审核通过 ----------
        if contains_any(all_text_lower, CN_FL_EDU_INTL_KEYWORDS_EN, case_insensitive=True):
            database_ids.add(CN_FL_EDU_INTL_DB_ID)
            status = CorpusStatus.APPROVE_EXAMINE.value

        # ---------- 规则5：涉及基督教、种族、港澳台的敏感主题 ----------
        # 3. 最后应用敏感主题限制，再为没有归属的记录设置默认库及待审状态
        if contains_any(all_text_lower, [k.lower() for k in SENSITIVE_REVIEW_KEYWORDS], case_insensitive=False):
            status = CorpusStatus.WAITING_EXAMINE.value

        # ---------- 默认规则：若前述规则均未命中任何子库，则挂到 198，状态保持待审核 ----------
        if not database_ids:
            database_ids.add(198)
            status = CorpusStatus.WAITING_EXAMINE.value

        return database_ids, status

    def log_recovery(file_name, file_path, status, error_message=None, corpus_id=None):
        """写入恢复过程记录（成功/失败/跳过）。"""
        if args.dry_run:
            return
        rec = CorpusRecoveryLog(
            file_name=file_name,
            file_path=file_path,
            status=status,
            error_message=error_message,
            corpus_id=corpus_id,
        )
        db.session.add(rec)

    def is_file_already_processed(file_name: str) -> bool:
        """
        判断文件是否已经在 corpus_recovery_log 中出现过。

        注意：
        - 历史数据中部分 file_name 前面带有空格，因此这里统一使用 TRIM 之后再比较；
        - 只要表中存在同名记录（无论 success/fail/skip），都视为“已经处理过”，本次跳过。
        """
        if not file_name:
            return False
        normalized_name = file_name.strip()
        if not normalized_name:
            return False
        try:
            exists = (
                db.session.query(CorpusRecoveryLog.id)
                .filter(func.trim(CorpusRecoveryLog.file_name) == normalized_name)
                .first()
            )
            return exists is not None
        except Exception:
            # 查询异常时，不影响主流程，按“未处理过”继续后续逻辑
            return False

    def is_file_already_in_corpus(file_name: str, file_path: str) -> bool:
        """按去扩展名的文件名、贡献者及有效记录判断是否已有恢复文献。

        Args:
            file_name: 当前 PDF 文件名，作为题名去重依据。
            file_path: 保留的旧调用参数，当前实现没有查询附件路径。

        Returns:
            命中已有题名时为 True，未命中或查询异常为 False；不提供并发互斥保证。
        """
        if not file_name:
            return False
        base_title = file_name[:-4] if file_name.lower().endswith(".pdf") else file_name
        try:
            # 依据标题与贡献者匹配
            exists_by_title = (
                db.session.query(Corpus.id)
                .filter(
                    Corpus.is_deleted == 0,
                    Corpus.contributor == contributor_id,
                    Corpus.title_cn == base_title,
                )
                .first()
            )
            if exists_by_title:
                return True
        except Exception:
            # 任何异常都不影响主流程，按“未在 corpus 中出现过”处理
            return False
        return False

    def fix_existing_attachments_for_renamed_files(corpus_dir, attachment_path_prefix):
        """
        修复之前已经写入 corpus 但 attachment 仍然指向“旧文件名”的记录。

        早期版本逻辑是：
        - attachment.file_path = uploaded_files/corpus/原文件名
        - 磁盘上的文件被重命名为：<corpus_id>-原文件名
        这会导致数据库路径与实际文件名不一致。

        这里根据 corpus_recovery_log 中的成功记录进行修复：
        - 若磁盘上存在 <corpus_id>-原文件名
        - 且 corpus.attachment 中仍然出现旧路径，则替换为新路径。
        """
        try:
            logs = (
                db.session.query(CorpusRecoveryLog)
                .filter(
                    CorpusRecoveryLog.status == "success",
                    CorpusRecoveryLog.corpus_id.isnot(None),
                )
                .all()
            )
        except Exception:
            return

        # 1. 只处理成功日志中能够在磁盘验证新文件名的记录，避免猜测性改写路径
        updated_count = 0
        for rec in logs:
            corpus_id = rec.corpus_id
            file_name = rec.file_name
            if not corpus_id or not file_name:
                continue

            old_rel_path = rec.file_path or f"{attachment_path_prefix}/{file_name}"
            new_file_name = f"{corpus_id}-{file_name}"
            new_rel_path = f"{attachment_path_prefix}/{new_file_name}"

            # 物理文件是否存在（避免误改）
            new_full_path = os.path.join(corpus_dir, new_file_name)
            if not os.path.exists(new_full_path):
                continue

            corpus = db.session.query(Corpus).filter(
                Corpus.id == corpus_id,
                Corpus.is_deleted == 0,
            ).first()
            if not corpus or not corpus.attachment:
                continue

            try:
                att_list = json.loads(corpus.attachment)
                changed = False
                for item in att_list:
                    if (
                        isinstance(item, dict)
                        and item.get("file_path") == old_rel_path
                    ):
                        item["file_path"] = new_rel_path
                        changed = True

                if changed:
                    corpus.attachment = json.dumps(att_list, ensure_ascii=False)
                    # 同步更新日志表中的 file_path，保持一致
                    rec.file_path = new_rel_path
                    updated_count += 1
            except Exception:
                continue

        # 2. 文献引用和恢复日志路径同时提交，失败时仅回滚本次数据库修复
        if updated_count > 0:
            try:
                db.session.commit()
                print(f"已修复 {updated_count} 条 attachment 与重命名文件不一致的记录")
            except Exception:
                db.session.rollback()

    # 默认使用项目指定账号 26，命令行可覆盖
    # 1. 解析恢复目标及模型配置，固定贡献者与本地原件目录
    contributor_id = args.contributor if args.contributor is not None else 26
    contributor_id = int(contributor_id)
    corpus_dir = os.path.abspath(args.dir or os.path.join(_PROJECT_ROOT, "..", "File", "uploaded_files", "corpus"))
    if not os.path.isdir(corpus_dir):
        print(f"错误: 目录不存在: {corpus_dir}")
        return 1

    # 从项目 .env 加载后的环境变量读取（脚本开头已 load_dotenv 项目根 .env）
    qwen_api_key = os.getenv("QWEN_API_KEY")
    if not qwen_api_key and not args.dry_run:
        print("错误: 项目 .env 中未配置 QWEN_API_KEY，无法调用通义千问提取")
        return 1

    # 在正式处理前，先修复一遍历史数据中“attachment 路径与物理文件名不一致”的问题
    # （仅对之前已经写入成功且磁盘上存在 <corpus_id>-文件名 的记录做修复）
    attachment_path_prefix = "uploaded_files/corpus"
    # 2. 修复历史数据库引用与已重命名原件的不一致，形成独立提交检查点
    # 当前调用位于 dry_run 分支判断之前，因此预览模式也可能执行此处写库。
    fix_existing_attachments_for_renamed_files(corpus_dir, attachment_path_prefix)

    # 列出所有 PDF
    pdf_files = [f for f in os.listdir(corpus_dir) if f.lower().endswith(".pdf")]
    pdf_files.sort()
    total = len(pdf_files)
    if args.limit and args.limit > 0:
        pdf_files = pdf_files[: args.limit]
    workers = getattr(args, "workers", 1) or 1
    if getattr(args, "reverse", False):
        pdf_files = list(reversed(pdf_files))
        print("已反转列表，从末尾往前处理（与正序脚本并行时互不重叠）")
    print(f"目录中共 {total} 个 PDF，本次处理 {len(pdf_files)} 个 (dry_run={args.dry_run}, skip_existing={args.skip_existing}, reverse={getattr(args, 'reverse', False)}, workers={workers})")

    # 相对路径前缀（用于生成 file_path）：与服务器一致为 uploaded_files/corpus
    client = None
    if qwen_api_key and not args.dry_run and workers <= 1:
        client = setup_qwen_client(qwen_api_key)

    success_count = 0
    skip_count = 0
    fail_count = 0

    # 先扫一遍，得到待处理列表（用于顺序或批量并行提取）
    # 3. 按文件名、贡献者和恢复日志排除已处理项，生成本轮提取队列
    # 反向排序仅改变顺序；这些查询没有跨进程领取锁，并发重复运行仍可能竞争同一文件。
    to_process = []
    for idx, filename in enumerate(pdf_files, 1):
        full_path = os.path.join(corpus_dir, filename)
        file_path_for_attachment = (
            f"{attachment_path_prefix}/{filename}"
            if not attachment_path_prefix.endswith("/")
            else attachment_path_prefix + filename
        )

        # 规则1：以“6位数字-”为前缀的文件名，一律视为已按 corpus.id 重命名过的旧文件，直接跳过且不写入日志。
        if re.match(r"^\d{6}-", filename):
            skip_count += 1
            if idx <= 3 or idx % 500 == 0:
                print(f"  [{idx}/{len(pdf_files)}] 跳过(已按id重命名): {filename}")
            continue

        # 此处 dry-run 只列出新增候选；此前的历史路径修复仍可能访问并提交数据库。
        if args.dry_run:
            print(f"  [{idx}/{len(pdf_files)}] {filename}")
            continue

        # 额外保护：若 corpus 表中已存在该文件对应记录，则直接跳过，避免重复解析 / 写入。
        if is_file_already_in_corpus(filename, file_path_for_attachment):
            skip_count += 1
            if idx <= 3 or idx % 500 == 0:
                print(f"  [{idx}/{len(pdf_files)}] 跳过(corpus 已有记录): {filename}")
            continue

        # 规则2/3：其余文件需要检查 corpus_recovery_log，若表中已存在同名记录（注意去掉前后空格再比较），则视为“已处理过”直接跳过。
        if is_file_already_processed(filename):
            skip_count += 1
            if idx <= 3 or idx % 500 == 0:
                print(f"  [{idx}/{len(pdf_files)}] 跳过(已在 corpus_recovery_log 中记录): {filename}")
            continue

        # 规则3：仅当在日志中未出现过时，才加入本次提取队列并在后续成功/失败时写入 corpus_recovery_log。
        to_process.append((idx, filename, full_path, file_path_for_attachment))

    if args.dry_run:
        print(f"完成: 成功 0, 跳过 {skip_count}, 失败 0")
        return 0

    def process_one(item, result):
        """把单篇模型结果映射为文献、主题关联及恢复日志，并按篇提交。

        Args:
            item: 输入序号、原文件名、本地路径和原附件引用组成的元组。
            result: 提取阶段返回的成功数据或错误字典。

        失败结果记录日志后提交，成功结果 flush 获取主键再尝试重命名真实文件。
        磁盘重命名不能随数据库事务回滚；批次整体也不具备原子性。
        """
        nonlocal success_count, fail_count
        idx, filename, full_path, file_path_for_attachment = item
        # 1. 拒绝提取失败或无法解析的结果，记录失败日志并提交，以免下次无条件重复提取
        if not result.get("success"):
            fail_count += 1
            err_msg = (result.get("error") or "unknown")[:2000]
            # 失败日志与之前未提交变更共用当前事务，提交点并非整批事务边界。
            log_recovery(filename, file_path_for_attachment, "fail", error_message=err_msg)
            print(f"  [{idx}/{len(pdf_files)}] 提取失败: {filename} -> {result.get('error', 'unknown')}")
            db.session.commit()
            return
        data = result.get("extracted_data") or {}
        if data.get("raw_output") is not None and "title_en" not in data:
            fail_count += 1
            log_recovery(filename, file_path_for_attachment, "fail", error_message="无法解析为结构化数据")
            print(f"  [{idx}/{len(pdf_files)}] 无法解析为结构化数据: {filename}")
            db.session.commit()
            return
        # 2. 规范化题录列长度、卷期和关键词，锁定分类计数器生成业务编号
        title_cn = filename[:-4] if filename.lower().endswith(".pdf") else filename
        title_en = safe_str(data.get("title_en"), 1000)
        category_name = safe_str(data.get("category"))
        category_id = get_category_id_by_name(db.session, category_name)
        publish_time = extract_year(data.get("publish_time"))
        volume = data.get("volume")
        issue = data.get("issue")
        try:
            volume = int(volume) if volume is not None and str(volume).strip() else None
        except (TypeError, ValueError):
            volume = None
        try:
            issue = int(issue) if issue is not None and str(issue).strip() else None
        except (TypeError, ValueError):
            issue = None
        corpus_number = None
        if category_id is not None:
            try:
                corpus_category = (
                    db.session.query(CorpusCategory)
                    .filter(CorpusCategory.id == category_id, CorpusCategory.is_deleted == 0)
                    .with_for_update(of=CorpusCategory)
                    .first()
                )
                if corpus_category:
                    current_num = corpus_category.number or 0
                    new_category_number = current_num + 1
                    corpus_category.number = new_category_number
                    corpus_number = f"{corpus_category.code}{new_category_number}"
            except Exception:
                corpus_number = None
        keyword_cn_list = normalized_keywords(data.get("keyword_cn"))
        keyword_en_list = normalized_keywords(data.get("keyword_en"))
        # 3. 按顺序规则确定主题集合及最终审核状态，创建当前恢复文献
        auto_database_ids, auto_status = classify_databases_and_status(
            language=safe_str(data.get("language"), 31),
            title_cn=title_cn,
            title_en=title_en,
            keyword_cn_list=keyword_cn_list,
            keyword_en_list=keyword_en_list,
            abstract_cn=safe_str(data.get("abstract_cn")),
            abstract_en=safe_str(data.get("abstract_en")),
        )
        attachment = [{"file_path": file_path_for_attachment, "word": 0}]
        corpus = Corpus(
            number=corpus_number,
            title_cn=title_cn,
            title_en=title_en,
            language=safe_str(data.get("language"), 31),
            author=safe_str(data.get("author"), 255),
            translator=safe_str(data.get("translator"), 255),
            editor=safe_str(data.get("editor"), 255),
            publish_time=publish_time,
            ISSN=safe_str(data.get("ISBN") or data.get("ISSN"), 127),
            source=safe_str(data.get("source"), 510),
            category_id=category_id,
            keyword_cn=json.dumps(keyword_cn_list, ensure_ascii=False) or None,
            keyword_en=json.dumps(keyword_en_list, ensure_ascii=False) or None,
            abstract_cn=safe_str(data.get("abstract_cn")),
            abstract_en=safe_str(data.get("abstract_en")),
            original_name=None,
            original_language=None,
            original_time=None,
            original_source=None,
            word_number=0,
            attachment=json.dumps(attachment, ensure_ascii=False),
            contributor=contributor_id,
            status=auto_status,
            comment=None,
            is_deleted=False,
            is_vectorized=False,
            submit_time=datetime.now(),
            volume=volume,
            issue=issue,
            reviewer_id=None,
        )
        db.session.add(corpus)
        # 4. 先 flush 获得文献 ID 再尝试重命名原件，磁盘操作不受事务回滚保护
        db.session.flush()
        try:
            new_filename = f"{corpus.id}-{filename}"
            new_full_path = os.path.join(corpus_dir, new_filename)
            if not os.path.exists(new_full_path):
                os.rename(full_path, new_full_path)
            new_rel_path = f"{attachment_path_prefix}/{new_filename}"
            corpus.attachment = json.dumps([{"file_path": new_rel_path, "word": 0}], ensure_ascii=False)
            file_path_for_attachment = new_rel_path
        except Exception:
            pass
        # 5. 写入主题关联及成功日志后按篇提交，已提交记录不会被后续文件失败撤销
        for db_id in auto_database_ids:
            try:
                db.session.add(CorpusDatabase(corpus_id=corpus.id, database_id=db_id))
            except Exception:
                pass
        log_recovery(filename, file_path_for_attachment, "success", corpus_id=corpus.id)
        success_count += 1
        if idx <= 5 or idx % 100 == 0 or idx == len(pdf_files):
            print(f"  [{idx}/{len(pdf_files)}] 已写入: {filename} -> title_cn={title_cn[:40] if title_cn else ''}...")
        db.session.commit()

    # 4. 多进程仅并行题录提取，主进程按结果顺序执行单篇入库、重命名及日志提交
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i in range(0, len(to_process), workers):
                batch = to_process[i : i + workers]
                full_paths = [x[2] for x in batch]
                results = list(pool.map(_extract_one_worker, full_paths))
                for item, result in zip(batch, results):
                    process_one(item, result)
    else:
        for item in to_process:
            result = robust_extract_pdf_info(client, item[2])
            process_one(item, result)

    if not args.dry_run:
        db.session.commit()
    print(f"完成: 成功 {success_count}, 跳过 {skip_count}, 失败 {fail_count}")
    return 0


def main():
    """解析恢复目录及提取并行参数，在最小 Flask 上下文中执行恢复；默认实际处理。"""
    parser = argparse.ArgumentParser(description="从服务器 PDF 目录恢复 corpus 表记录")
    # 默认目录：项目在 Foreign/Backend，PDF 在 Foreign/File/uploaded_files/corpus
    _default_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "File", "uploaded_files", "corpus"))
    parser.add_argument("--dir", type=str, default=_default_dir,
                        help="存放 corpus PDF 的目录，默认 ../File/uploaded_files/corpus（相对项目根）")
    parser.add_argument("--contributor", type=int, default=26,
                        help="贡献者用户 ID，默认 26")
    parser.add_argument("--dry-run", action="store_true", help="仅列出文件，不调用 API、不写库")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个 PDF，0 表示不限制")
    parser.add_argument("--skip-existing", action="store_true",
                        help="文件名已为 6位数字- 开头则跳过，不写 recovery_log")
    parser.add_argument("--reverse", action="store_true",
                        help="从列表末尾往前处理（与正序脚本并行时互不重叠）")
    parser.add_argument("--workers", type=int, default=1,
                        help="并行提取进程数，>1 时仅提取阶段并行，写库与重命名仍在主进程顺序执行")
    parser.add_argument("--batch-size", type=int, default=50, help="每 N 条提交一次事务，0 表示最后统一提交")
    args = parser.parse_args()

    # 仅创建最小 Flask app + db，不注册任何蓝图，避免引入 redis 等无关依赖
    from flask import Flask
    from app.exts import db
    app = Flask(__name__)
    app.config.from_object("app.config")
    db.init_app(app)
    with app.app_context():
        return run(app, args)


if __name__ == "__main__":
    sys.exit(main())
