"""提供文件服务代理、文档内容与字数提取、摘要翻译及邮件发送等同步业务工具。"""

import json
import os
import re

# from PyPDF2 import PdfReader
import pdfplumber
from docx import Document
import re
from io import BytesIO
import requests
from flask import send_file
import random
import string
import hashlib
from openai import OpenAI
from flask import current_app
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import fitz
import unicodedata
import os
import ssl
import tempfile
import logging
from multiprocessing import get_context
from email.utils import formatdate, make_msgid

_logger = logging.getLogger(__name__)

# PDF 字数统计子进程超时（秒）；异常 PDF 卡住时由父进程终止
_PDF_WORDCOUNT_TIMEOUT = 120

def count_valid_chars(text):
    """统计 Unicode 字母、数字和标点字符，排除空白与其他符号类别。"""
    count = 0
    for ch in text:
        cat = unicodedata.category(ch)
        # 统计所有字母（L开头类别）、数字（N开头类别）、标点符号（P开头类别）
        if cat.startswith(('L', 'N', 'P')):
            count += 1
    return count


def extract_text_length_filtered(page):
    """提取 PyMuPDF 页面原生文字，并按统一有效字符口径统计。"""
    text = page.get_text()
    return count_valid_chars(text)


def _pdf_wordcount_worker(pdf_path, conn):
    """在子进程中统计 PDF 字数。MuPDF segfault 只会杀掉本进程。"""
    try:
        doc = fitz.open(pdf_path)
        total = 0
        for page in doc:
            total += count_valid_chars(page.get_text())
        doc.close()
        conn.send(('ok', total))
    except Exception as e:
        conn.send(('err', str(e)))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _count_pdf_words_isolated(file_content, timeout=_PDF_WORDCOUNT_TIMEOUT):
    """在 spawn 子进程统计 PDF，隔离 MuPDF 崩溃并限制处理时间。

    Args:
        file_content: 从上传流读出的 PDF 字节。
        timeout: 等待子进程完成的秒数，超时后终止并按需强制结束。

    Returns:
        正常统计的字符数，子进程崩溃、超时或未返回有效结果时降级为零。
    """
    # 1. 把上传字节写入临时 PDF，使隔离进程不继承 Flask 对象或数据库连接
    fd, pdf_path = tempfile.mkstemp(suffix='.pdf')
    try:
        os.write(fd, file_content)
        os.close(fd)
        fd = None

        # spawn：不继承父进程状态，比 fork 更安全（Flask/DB 连接等）
        # 2. 使用 spawn 启动原生解析子进程，经单向管道取回统计，避免段错误杀死请求进程
        ctx = get_context('spawn')
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_pdf_wordcount_worker, args=(pdf_path, child_conn))
        proc.start()
        child_conn.close()
        proc.join(timeout)

        # 3. 超时先终止、再按需强杀；崩溃或无结果时按既有契约降级为零
        if proc.is_alive():
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
            _logger.error('PDF字数统计超时（>%ss），已终止子进程', timeout)
            return 0

        if proc.exitcode != 0:
            # Linux 上 SIGSEGV 通常为 -11
            _logger.error('PDF字数统计子进程异常退出: exitcode=%s', proc.exitcode)
            return 0

        if parent_conn.poll(1):
            status, value = parent_conn.recv()
            if status == 'ok':
                return int(value)
            _logger.error('PDF字数统计失败: %s', value)
            return 0

        _logger.error('PDF字数统计子进程未返回结果')
        return 0
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(pdf_path)
        except OSError:
            pass


def count_words(file):
    """按 PDF、TXT 或 DOCX 格式统计有效字符数，并将文件游标重置到开头。

    Args:
        file: 带 filename 的可读取、可 seek 上传文件。

    Returns:
        有效字符数；格式不支持或隔离 PDF 统计失败时返回零。
    """
    extension_name = file.filename.split('.')[-1].lower()
    if extension_name not in ['pdf', 'txt', 'docx']:
        return 0

    file_content = file.read()
    file.seek(0)
    file_stream = BytesIO(file_content)

    total_length = 0

    if extension_name == 'txt':
        text = file_content.decode('utf-8')
        total_length = count_valid_chars(text)

    elif extension_name == 'pdf':
        # 必须在子进程中跑：MuPDF 原生崩溃无法被 try/except 捕获
        total_length = _count_pdf_words_isolated(file_content)

    elif extension_name == 'docx':
        doc = Document(file_stream)
        for para in doc.paragraphs:
            total_length += count_valid_chars(para.text)

    file_stream.close()
    return total_length



# 用于批量下载文件
def get_file_content(filename):
    """获取文件字节及类型；失败时同一返回位置改为错误内容和 HTTP 状态。

    Args:
        filename: 文件服务识别的文件路径，使用当前旧接口拼接到查询串。

    Returns:
        成功为 (bytes, MIME)，失败为 (错误内容, 状态码)，调用方须区分两种契约。
    """
    try:
        response = requests.get(current_app.config['FILE_URL'] + '/files?filename=' + filename)

        if response.status_code == 200:
            # 返回文件内容和MIME类型
            return response.content, response.headers.get('Content-Type', 'application/octet-stream')
        else:
            # 返回错误信息和状态码
            try:
                error_data = response.json()
                return (error_data, response.status_code)
            except:
                return (response.text, response.status_code)
    except requests.exceptions.RequestException as e:
        return (str(e), 500)


# 用于单个文件下载
def get_file(filename):
    """以查询参数传递真实文件路径，返回 (Flask 文件响应或错误内容, HTTP 状态)。"""
    try:
        response = requests.get(current_app.config['FILE_URL'] + '/files', params={'filename': filename})

        if response.status_code == 200:
            # 读取文件内容到内存
            file_content = BytesIO(response.content)
            file_content.seek(0)  # 将指针移动到文件开头
            mimetype = response.headers.get('Content-Type', 'application/octet-stream')
            return send_file(file_content, mimetype=mimetype), 200
        else:
            try:
                error_data = response.json()
                return error_data, response.status_code
            except:
                return response.text, response.status_code
    except requests.exceptions.RequestException as e:
        return str(e), 500


# 用于批量修复等场景，直接返回文件字节
# 成功返回 (bytes, content_type)；失败返回 (error_data, status_code)
def get_file_bytes(filename):
    """以查询参数获取原始文件，成功返回 (bytes, MIME)，失败返回 (错误内容, HTTP 状态)。"""
    try:
        response = requests.get(current_app.config['FILE_URL'] + '/files', params={'filename': filename})
        if response.status_code == 200:
            return response.content, response.headers.get('Content-Type', 'application/octet-stream')
        else:
            try:
                return response.json(), response.status_code
            except Exception:
                return response.text, response.status_code
    except requests.exceptions.RequestException as e:
        return str(e), 500


# 上传文件
def upload_file(files, upload_type, username):
    """将身份图片或文献附件按 multipart 契约上传到独立文件服务。

    Args:
        files: picture 类型为单个上传对象；其他类型为 (文件, 字数) 列表。
        upload_type: 文件服务的业务分类，例如 picture、corpus。
        username: 文件服务中的用户归属标识。

    Returns:
        (附件引用列表, 200)，或 (错误内容, 状态码)。不提交任何数据库事务。
    """
    if upload_type == 'picture':
        if files.filename.rsplit('.', 1)[-1].lower() not in ['jpg', 'jpeg', 'png']:
            return "图片上传只支持jpg,jpeg,png类型", 400
        # 统一格式
        files = [(files, 0)]
    try:
        files_dict = {}
        files_words = {}
        for i, (tmp_file, words) in enumerate(files):
            # 将每个文件的 filename, stream 和 content_type 提取出来
            files_dict[f'file{i}'] = (tmp_file.filename, tmp_file.stream, tmp_file.content_type)
            files_words[tmp_file.filename] = words
        params = {
            "username": username,  # 查询参数
            "upload_type": upload_type,
            # 文件的字数，存在路径里方便减去的时候计算
            "files_words": json.dumps(files_words)
        }
        response = requests.post(current_app.config['FILE_URL'] + '/upload', params=params, files=files_dict)
        if response.status_code == 200:
            # 解析返回的 JSON 数据
            data = response.json()
            # 获取文件路径列表
            file_paths = data.get('file_paths', [])
            return file_paths, 200
        else:
            return response.json(), response.status_code
    except Exception as e:
        return str(e), 400


# 删除文件
def delete_file(filename_list):
    """删除一个或多个远程文件引用，不参与数据库事务。

    Args:
        filename_list: 单一路径或路径列表。

    Returns:
        非空输入返回 (服务结果, HTTP 状态)，空列表直接返回 None。
    """
    if not isinstance(filename_list, list):
        filename_list = [filename_list]
    if len(filename_list) == 0: return
    try:
        response = requests.post(current_app.config['FILE_URL'] + '/delete', json={"filename_list": filename_list})

        return response.json(), response.status_code
    except requests.exceptions.RequestException as e:
        return str(e), 400


# 初始化OpenAI客户端
def init_openai_client(api_key):
    """用显式凭据创建百炼 OpenAI 兼容客户端，不设置 SDK 全局配置。"""
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

def generate_summary(client, content, is_segment=False):
    """调用现有摘要模板生成整篇或单段总结。

    Args:
        client: 已配置的同步模型客户端。
        content: 待总结正文。
        is_segment: 为真选择分段总结模板，否则选择整篇摘要及关键词模板。

    Returns:
        模型首条候选消息正文，调用异常由上层决定如何返回。
    """
    if is_segment:
        # 单个段落的提示词
        prompt = (
            "请总结以下学术文章段落的主要内容。\n"
            "要求：300字-500字\n"
            "论文内容：\n"
            f"{content}"
        )
    else:
        # 整篇论文的提示词
        prompt = (
            "请为以下学术论文生成摘要和关键词。\n"
            "要求：\n"
            "1. 摘要需包含研究目的、方法和结论，300字以内\n"
            "2. 关键词5-8个，用中文分号分隔\n"
            "论文内容：\n"
            f"{content}"
        )

    completion = client.chat.completions.create(
        model="qwen-long", # 通义千问-Long
        messages=[
            {"role": "system", "content": "你是一个专业的学术助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )

    return completion.choices[0].message.content


def generate_summary_for_segments(client, segments):
    """按输入顺序同步生成各分段摘要，保留顺序供最终汇总。"""
    summaries = []
    for segment in segments:
        result = generate_summary(client, segment, is_segment=True)
        summaries.append(result)
    return summaries

def generate_total_summary(client, segment_summaries, abstract_words_num, keywords_num):
    # 将所有分段摘要拼接成一个字符串
    """汇总分段摘要并检查输出长度是否达到目标的八成。

    Args:
        client: 同步模型客户端。
        segment_summaries: 按原文顺序排列的分段摘要。
        abstract_words_num: 期望摘要字符数量。
        keywords_num: 期望关键词数量，传入现有提示词模板。

    Returns:
        模型生成的总摘要与关键词文本。

    Raises:
        ValueError: 输出长度不足目标的八成。
    """
    combined_summaries = "\n".join(segment_summaries)

    prompt = (
        "请整合以下分段摘要，生成详细的总摘要：\n"
        f"{combined_summaries}\n\n"
        "要求：\n"
        "1. 摘要需{abstract_words_num}字左右，全面覆盖研究背景、方法、结果和意义\n"
        "2. 避免直接复制分段内容，需重新组织语言\n"
        "3. 最后用'关键词：'开头列出{keywords_num}个关键词，格式示例：\n"
        "   关键词：A；B；C；D；E；F\n\n"
        "输出时无需包含'摘要：'前缀，直接以正文开始。"
    ).format(abstract_words_num=abstract_words_num, keywords_num=keywords_num)

    completion = client.chat.completions.create(
        model="qwen-long",
        messages=[
            {"role": "system", "content": "你是一个专业的学术助手。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=2048
    )
    output_text = completion.choices[0].message.content
    required_length = int((abstract_words_num or 800) * 0.8)  # 默认取800字的90%
    if len(output_text) < required_length:
        raise ValueError("摘要长度不足，请调整prompt或max_tokens")
    return output_text


def get_content(file):
    """从上传 PDF、UTF-8 TXT 或 DOCX 提取正文，不在此执行扫描件 OCR。

    Args:
        file: 带 filename 的可读上传流，读取后先将游标重置到开头。

    Returns:
        拼接正文；格式不支持时按现有契约返回提示文本而非抛出异常。
    """
    extension_name = file.filename.split('.')[-1].lower()
    if extension_name not in ['pdf', 'txt', 'docx']:
        return '文件类型不支持'
    # 创建文件内容副本
    file_content = file.read()
    file.seek(0)  # 重置原始文件指针
    # 创建新的内存流用于处理
    file_stream = BytesIO(file_content)
    text = ''
    if extension_name == 'txt':
        text = file_content.decode('utf-8')
    elif extension_name == 'pdf':
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
    elif extension_name == 'docx':
        doc = Document(file_stream)
        for para in doc.paragraphs:
            text += para.text
    return text


def split_text_into_segments(text, max_chars=18000):
    """按 max_chars 字符宽度顺序切段，此处不是索引 token 预算分块器。"""
    segments = []
    for i in range(0, len(text), max_chars):
        segment = text[i:i + max_chars]
        segments.append(segment)
    return segments


# 调用百度翻译API
def translate_text(api_key, secret_key, text, source_lang, target_lang):
    """请求百度文本翻译，失败时返回 None 供调用方决定降级方式。

    Args:
        api_key: 百度翻译 appid。
        secret_key: 对应账户的签名密钥。
        text: 待翻译原文。
        source_lang: 源语言代码。
        target_lang: 目标语言代码。

    Returns:
        首个翻译条目的 dst 文本，无结果或请求失败时为 None。
    """
    url = "https://fanyi-api.baidu.com/api/trans/vip/translate"
    salt = generate_salt(8)
    sign = compute_sign(api_key, text, salt, secret_key)

    # 构建请求参数
    params = {
        "q": text,
        "from": source_lang,
        "to": target_lang,
        "appid": api_key,
        "salt": salt,
        "sign": sign
    }

    try:
        response = requests.post(url, params=params)
        if response.status_code == 200:
            result = response.json()
            # 检查是否存在错误码字段
            if 'error_code' in result:
                return None
            # 没有错误码，则获取翻译结果
            trans_result = result.get('trans_result', [])
            if trans_result:
                return trans_result[0].get('dst')  # 只获取译文
            return None
        else:
            return None
    except Exception as e:
        return None

# 生成指定长度的随机盐值
def generate_salt(length=8):
    """生成指定长度的 ASCII 字母数字随机串，用于旧翻译接口请求签名。"""
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


# 计算百度翻译API的签名
def compute_sign(api_key, text, salt, secret_key):
    """按 appid、原文、salt、密钥的顺序拼接并计算百度翻译 MD5 签名。"""
    sign_str = api_key + text + salt + secret_key
    md5 = hashlib.md5()
    md5.update(sign_str.encode('utf-8'))
    return md5.hexdigest()

def translate_keywords(api_key, secret_key, keywords_str, source_lang, target_lang):
    """
    翻译逗号分隔的关键词字符串
    :param api_key: 百度API key
    :param secret_key: 百度API secret
    :param keywords_str: 逗号分隔的关键词字符串
    :param source_lang: 源语言
    :param target_lang: 目标语言
    :return: 逗号分隔的翻译结果字符串
    """
    if not keywords_str or not isinstance(keywords_str, str):
        return None
    
    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
    translated = []
    
    for keyword in keywords:
        result = translate_text(api_key, secret_key, keyword, source_lang, target_lang)
        translated.append(result if result else keyword)
    
    return '，'.join(translated)


def send_email(to_addresse, subject, html_body):
    """组合纯文本与 HTML 邮件，按既有 SMTP 配置发送并关闭连接。

    Args:
        to_addresse: 单个收件人地址或地址列表。
        subject: 邮件主题。
        html_body: 邮件 HTML 正文，纯文本备选内容从此剥离标签生成。

    Returns:
        发送成功为 True；连接、认证或发送失败记录日志并返回 False。
    """
    try:
        smtp_server = "smtpdm.aliyun.com"
        smtp_port = 80
        smtp_use_ssl = False       # 端口 80 不直接启用 SSL
        smtp_use_starttls = True

        sender_email = "accounts@mail.guojiayuwei.com"
        sender_password = "GUojiayuwei001"
        # QQ 邮箱对“中文显示名/纯 HTML/可疑 Reply-To”更容易风控，这里做更保守的默认
        reply_to = None
        
        # 创建邮件对象
        # 1. 组合纯文本和 HTML 备选正文，保留既有邮件模板内容
        msg = MIMEMultipart("alternative")
        # 最保守：From 不带显示名，降低“特殊字符/诱导”风险提示
        msg['From'] = sender_email
        if reply_to:
            msg['Reply-To'] = reply_to
        msg['Subject'] = subject
        msg['Date'] = formatdate(localtime=True)
        msg['Message-ID'] = make_msgid()
        # 同时提供纯文本和 HTML（multipart/alternative）
        text_body = re.sub(r"<[^>]+>", "", html_body or "")
        text_body = re.sub(r"\n{3,}", "\n\n", text_body).strip()
        msg.attach(MIMEText(text_body or "请在浏览器中打开邮件查看完整内容。", "plain", "utf-8"))
        msg.attach(MIMEText(html_body or "", "html", "utf-8"))
        
        # 确保to_addresse是列表格式
        if isinstance(to_addresse, str):
            to_addresse = [to_addresse]
        msg['To'] = ", ".join(to_addresse)

        # 2. 按当前配置建立 TLS 会话后认证发送，finally 负责正常或异常关闭连接
        context = ssl.create_default_context()
        smtp_client = None
        try:
            if smtp_use_ssl:
                smtp_client = smtplib.SMTP_SSL(host=smtp_server, port=smtp_port, timeout=15, context=context)
                smtp_client.ehlo()
            else:
                smtp_client = smtplib.SMTP(host=smtp_server, port=smtp_port, timeout=15)
                smtp_client.ehlo()
                if smtp_use_starttls:
                    smtp_client.starttls(context=context)
                    smtp_client.ehlo()

            smtp_client.login(sender_email, sender_password)
            # 用 sendmail 明确 envelope-from / rcpt-to，便于投递追踪
            smtp_client.sendmail(sender_email, to_addresse, msg.as_string())
        finally:
            if smtp_client:
                try:
                    smtp_client.quit()
                except Exception:
                    try:
                        smtp_client.close()
                    except Exception:
                        pass
        
        return True
        
    except Exception as e:
        try:
            current_app.logger.exception("邮件发送失败")
        except Exception:
            print(f"邮件发送失败: {str(e)}")
        return False
