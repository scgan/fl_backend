"""复用采集与个人上传的同步 PDF 题录提取，集中维护模型调用、文件回退和内容缓存。"""

import json
import os
import time
from pathlib import Path

import pdfplumber
from flask import current_app
from openai import OpenAI

def setup_qwen_client(api_key):
    """创建采集与个人文件共用的百炼兼容客户端。"""
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return client


def _extract_message_content(completion):
    """
    通义千问响应解析辅助函数，确保 choices 不为空并返回首条消息内容
    """
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return None, "通义千问API返回空的choices列表"
    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    content = getattr(message, "content", None) if message else None
    if not content:
        return None, "通义千问API返回空的message内容"
    return content, None


def _parse_json_from_text(result_text):
    """
    从大模型文本响应中提取JSON内容；失败时返回原始文本
    """
    logger = current_app.logger

    if not result_text:
        return False, {"raw_output": ""}
    try:
        if "```json" in result_text:
            json_str = result_text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in result_text:
            json_str = result_text.split("```", 1)[1].split("```", 1)[0].strip()
        else:
            json_str = result_text.strip()

        # 清理无效的控制字符（保留合法的空白字符如 \n, \t, \r）
        import re
        # 移除除了 \n, \r, \t 之外的控制字符（ASCII 0-31，除了 9, 10, 13）
        json_str = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', json_str)

        return True, json.loads(json_str)
    except (IndexError, ValueError, json.JSONDecodeError) as e:
        logger.warning(f"大模型响应无法解析为JSON: {e}，返回原始文本")
        return False, {"raw_output": result_text}


# 默认的论文信息提取提示词：精简版，去掉完整示例JSON，减少每次调用的输入token与诱导出的长输出
_DEFAULT_EXTRACTION_PROMPT = """请从这篇学术论文中提取结构化信息，仅返回一个严格的JSON对象：
- 所有值为字符串；无法确定用空字符串""，不要返回 null/None/[]。
- 仅输出JSON本体，不要任何解释、前后缀文字或Markdown代码块。
- keyword_en、keyword_cn 为字符串数组且必须给出；原文无关键词时根据标题与摘要自行提炼。
- abstract_cn：中文摘要，不超过150字；abstract_en：保持原文语言的摘要（原文仅有中文摘要时翻译为中文）。
- title_cn 一律为中文；title_en 保持原文标题；language 用中文全称（如"英文""韩语"），不要用 zh/en 等缩写。
- author 多个作者用中文逗号分隔；publish_time 仅年份如"2023"。
- category 从以下类型判断：期刊文章/学位论文/会议论文/专著/报告/专利/标准/报纸文章/电子文献，无法判断则空字符串。

JSON键（全部不可缺省）：title_en, title_cn, language, author, translator, editor, ISBN, category, publish_time, volume, issue, source, keyword_en, keyword_cn, abstract_cn, abstract_en
"""


def extract_pdf_info_turbo(client, pdf_path, extraction_prompt=None):
    """先读取 PDF 前五页、最多五千字符，调用快速模型提取题录。

    Args:
        client: 已配置的模型客户端。
        pdf_path: 本地 PDF 路径。
        extraction_prompt: 可选调用方提示词；缺省使用现有题录模板。

    Returns:
        包含 success、提取数据和原响应或错误的字典；无法读取原生文字或
        命中内容审核失败时转到文件上传提取流程。
    """
    import gc
    logger = current_app.logger

    # 默认的论文信息提取提示词（模块级精简常量，两个函数共用，避免重复维护）
    if extraction_prompt is None:
        extraction_prompt = _DEFAULT_EXTRACTION_PROMPT

    try:
        # 1. 使用pdfplumber提取PDF文本内容（优化内存使用）
        logger.info(f"正在读取文件: {pdf_path}")

        # 1. 优先提取前五页原生文字，解析失败或无文字时转到整文件上传模式
        file_content = ""
        # 标题/作者/摘要/关键词等元数据通常都在前几页；最终也只会截取前5000字符，
        # 解析更多页纯属浪费CPU，设为5页即可（扫描件提取不到文本时仍会回退到文件上传模式）
        max_pages = 5
        page_count = 0

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"PDF总页数: {total_pages}")

                # 限制处理的页数
                pages_to_process = min(total_pages, max_pages)

                for i, page in enumerate(pdf.pages[:pages_to_process]):
                    try:
                        text = page.extract_text()
                        if text:
                            file_content += text + "\n"
                            page_count += 1

                            # 每处理10页进行一次垃圾回收
                            if page_count % 10 == 0:
                                gc.collect()

                    except Exception as e:
                        logger.warning(f"处理第{i+1}页时出错: {e}")
                        continue
        except Exception as e:
            # 兼容所有底层解析异常（如 Invalid dictionary construct 等），统一切换到 qwen-long 模式
            logger.warning(
                f"pdfplumber解析异常（{type(e).__name__}），切换到qwen-long模式: {e}"
            )
            gc.collect()
            return extract_pdf_info_long(client, pdf_path, extraction_prompt)

        # 清理PDF对象
        del pdf
        gc.collect()

        if not file_content.strip():
            logger.warning("无法从PDF中提取文本内容，切换为qwen-long文件上传模式...")
            return extract_pdf_info_long(client, pdf_path, extraction_prompt)

        if total_pages > max_pages:
            logger.info(f"PDF文本提取成功，总页数{total_pages}页，为优化性能只分析了前{page_count}页")
        else:
            logger.info(f"PDF文本提取成功，处理了{page_count}页")

        # 2. 截取前5000个字符（控制token数量）
        # 2. 限制本次题录输入字符预算，使用既有提示词和配置的快速模型
        file_content = file_content[:5000]
        logger.info(f"截断后的文本内容长度：{len(file_content)} 字符")

        # 3. 构建对话消息
        messages = [
            {
                "role": "system",
                "content": "你是专业的信息分析助手，擅长从PDF文本中提取结构化信息。请准确、完整地提取所需信息。"
            },
            {
                "role": "user",
                "content": f"请分析以下PDF文本内容：\n\n{file_content}\n\n{extraction_prompt}"
            }
        ]

        # 清理文本内容变量
        del file_content
        gc.collect()

        # 4. 调用通义千问API进行信息提取
        # 短文本提取走速度优先模型（环境变量 EXTRACT_MODEL_NAME 可覆盖，默认 qwen-plus），
        # qwen-long 仅保留给 extract_pdf_info_long 的长文档文件上传兜底路径
        extract_model = os.getenv('EXTRACT_MODEL_NAME', 'qwen-plus')
        logger.info(f"正在调用通义千问API分析论文内容（模型：{extract_model}）...")
        try:
            completion = client.chat.completions.create(
                model=extract_model,
                messages=messages,
                temperature=0.1,  # 低随机性确保结果稳定
            )
        except Exception as api_error:
            error_str = str(api_error)
            # 检测内容审核失败错误，自动切换到qwen-long模式
            if "data_inspection_failed" in error_str or "inappropriate content" in error_str.lower():
                logger.warning(f"qwen-long内容审核失败，切换到qwen-long模式: {error_str}")
                # 清理消息变量
                del messages
                gc.collect()
                return extract_pdf_info_long(client, pdf_path, extraction_prompt)
            else:
                # 其他错误直接抛出
                raise

        # 5. 处理返回结果
        # 3. 提取候选消息并解析结构化结果，异常统一返回失败字典供上层重试
        result_text, err = _extract_message_content(completion)
        if err:
            logger.error(err)
            return {
                "success": False,
                "error": err,
                "raw_response": getattr(completion, "__dict__", str(completion))
            }

        # 清理消息变量
        del messages
        gc.collect()

        # 尝试解析JSON结果
        parsed, result_data = _parse_json_from_text(result_text)
        if not parsed or not isinstance(result_data, dict) or 'raw_output' in result_data:
            return {"success": False, "error": "模型未返回有效的题录 JSON 对象"}

        return {
            "success": True,
            "extracted_data": result_data,
            "raw_response": result_text
        }

    except Exception as e:
        logger.error(f"提取失败: {e}")
        # 确保垃圾回收
        gc.collect()
        return {
            "success": False,
            "error": str(e),
            "file_path": pdf_path
        }


def extract_pdf_info_long(client, pdf_path, extraction_prompt=None):
    """上传整份 PDF，等待远程文件解析后调用 qwen-long 提取题录。

    Args:
        client: 支持文件上传、删除及对话调用的模型客户端。
        pdf_path: 当前本地 PDF。
        extraction_prompt: 可选调用方提示词，缺省沿用现有题录模板。

    Returns:
        包含 success、结构化数据及原响应或错误的字典。只在得到有效响应后
        尝试删除远程临时文件，清理失败不会覆盖已获得的提取结果。
    """
    import gc

    # 默认的论文信息提取提示词（模块级精简常量，两个函数共用，避免重复维护）
    if extraction_prompt is None:
        extraction_prompt = _DEFAULT_EXTRACTION_PROMPT

    try:
        current_app.logger.info(f"正在上传PDF文件: {pdf_path}")

        # 1. 上传PDF文件到通义千问服务器
        # 1. 上传完整原件，以远程文件 ID 作为模型输入，不把本地路径当作远程引用
        file_object = client.files.create(
            file=Path(pdf_path),
            purpose="file-extract"
        )
        current_app.logger.info(f"文件上传成功，文件ID: {file_object.id}")

        # 2. 等待文件解析完成（使用轮询机制）
        max_wait_time = 120  # 最大等待时间（秒）
        wait_interval = 3  # 每次等待间隔（秒）
        max_retries = max_wait_time // wait_interval  # 最大重试次数

        # 3. 构建包含文件引用的消息
        messages = [
            {
                "role": "system",
                "content": "你是专业的信息分析助手，擅长从学术PDF中提取结构化信息。请准确、完整地提取所需信息。"
            },
            {
                "role": "system",
                "content": f"fileid://{file_object.id}"  # 引用上传的文件
            },
            {
                "role": "user",
                "content": extraction_prompt
            }
        ]

        # 4. 调用通义千问API进行信息提取（带重试机制）
        current_app.logger.info("正在调用通义千问API分析论文内容...")

        completion = None
        # 2. 仅对远程文件仍在解析的状态等待后重试，其他调用异常交给外层失败处理
        # 次数限制只覆盖解析等待，不是包含上传和模型请求耗时的严格总超时。
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model="qwen-long",  # 支持超长上下文
                    messages=messages,
                    temperature=0.1,
                    stream=False
                )
                # 如果成功，跳出循环
                break
            except Exception as api_error:
                error_str = str(api_error)
                # 检查是否是文件解析中的错误
                if "File parsing in progress" in error_str or "parsing in progress" in error_str.lower():
                    if attempt < max_retries - 1:
                        current_app.logger.info(f"文件仍在解析中，等待{wait_interval}秒后重试（第{attempt+1}/{max_retries}次）...")
                        time.sleep(wait_interval)
                        continue
                    else:
                        # 最后一次重试也失败
                        error_msg = "文件解析超时，请稍后重试或使用较小的文件"
                        current_app.logger.error(f"文件解析超时: {error_msg}")
                        raise Exception(error_msg)
                else:
                    # 其他错误直接抛出
                    raise

        # 检查 completion 是否成功获取
        if completion is None:
            error_msg = "API调用失败：无法获取响应"
            current_app.logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "file_path": pdf_path
            }

        # 5. 处理返回结果
        # 3. 校验响应后尽力删除远程临时文件，再解析 JSON；清理失败保留提取结果
        result_text, err = _extract_message_content(completion)
        if err:
            current_app.logger.error(err)
            return {
                "success": False,
                "error": err,
                "raw_response": getattr(completion, "__dict__", str(completion))
            }
        current_app.logger.info("API调用成功，开始解析结果...")

        # 6. 清理上传的文件
        try:
            client.files.delete(file_object.id)
            current_app.logger.info("临时文件已清理")
        except Exception as e:
            current_app.logger.error(f"文件清理失败: {e}")

        # 强制垃圾回收
        gc.collect()

        # 尝试解析JSON结果
        parsed, result_data = _parse_json_from_text(result_text)
        if not parsed or not isinstance(result_data, dict) or 'raw_output' in result_data:
            return {"success": False, "error": "模型未返回有效的题录 JSON 对象"}
        if parsed:
            current_app.logger.info("成功解析为JSON")
        else:
            current_app.logger.error("无法解析为JSON，已返回原始文本")

        return {
            "success": True,
            "extracted_data": result_data,
            "raw_response": result_text,
            "file_id": file_object.id
        }

    except Exception as e:
        current_app.logger.error(f"提取失败: {e}")
        # 确保垃圾回收
        gc.collect()
        return {
            "success": False,
            "error": str(e),
            "file_path": pdf_path
        }


# 提取结果内存缓存：按文件内容MD5命中，同一PDF重复上传时跳过大模型调用。
# 条目带时间戳，超过 _EXTRACT_CACHE_TTL 自动过期；请求带 refresh=1 可立即强制重提（结果覆盖缓存）。
# 只缓存成功结果，FIFO淘汰；单文件与批量接口都经过 robust_extract_pdf_info，一处挂钩两处生效。
_EXTRACT_CACHE = {}
_EXTRACT_CACHE_MAX = 64
_EXTRACT_CACHE_TTL = 3600  # 缓存有效期（秒），过期后自动重新提取


def _file_md5(pdf_path):
    """按一 MB 分块计算 PDF 内容 MD5，作为题录提取的进程内缓存键。"""
    import hashlib
    h = hashlib.md5()
    with open(pdf_path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def robust_extract_pdf_info(client, pdf_path, max_retries=3, use_cache=True):
    """按内容缓存和有限重试提取题录，不属于索引 OCR 缓存。

    Args:
        client: 调用题录提取模型的同步客户端。
        pdf_path: 本地 PDF 路径。
        max_retries: 最大尝试次数，调用方应提供正数。
        use_cache: 是否读取有效 MD5 缓存；为假仍会在成功后更新缓存。

    Returns:
        包含 success 和 extracted_data 或 error 的字典；缓存命中时含 cached。
    """
    logger = current_app.logger

    # 1. 计算内容缓存键并检查 TTL，哈希失败时仅跳过缓存，不阻止实际提取
    md5 = None
    try:
        md5 = _file_md5(pdf_path)
    except Exception as e:
        logger.warning(f"计算文件MD5失败，跳过缓存: {e}")
    if use_cache and md5 and md5 in _EXTRACT_CACHE:
        entry = _EXTRACT_CACHE[md5]
        if time.time() - entry['ts'] <= _EXTRACT_CACHE_TTL:
            logger.info(f"命中提取结果缓存（MD5={md5[:12]}...），跳过大模型调用")
            return {
                "success": True,
                "extracted_data": entry['data'],
                "raw_response": "",
                "cached": True
            }
        logger.info(f"提取缓存已过期（MD5={md5[:12]}...），重新提取")
        _EXTRACT_CACHE.pop(md5, None)

    # 2. 按错误类型有限重试；成功结果更新进程内题录缓存
    # 限流和超时返回按次数退避，普通业务失败尽早结束，异常分支沿用原恢复策略。
    for attempt in range(max_retries):
        result = None
        try:
            # 使用快速模型提取（内部在无法抽取文本时会自动切换到文件上传模式）
            result = extract_pdf_info_turbo(client, pdf_path)
            if result["success"]:
                if md5:
                    if len(_EXTRACT_CACHE) >= _EXTRACT_CACHE_MAX:
                        _EXTRACT_CACHE.pop(next(iter(_EXTRACT_CACHE)))
                    _EXTRACT_CACHE[md5] = {
                        'data': result.get("extracted_data", {}),
                        'ts': time.time(),
                    }
                return result
            else:
                logger.warning(f"提取失败: {result['error']}")
                err_lower = str(result['error']).lower()
                # 超时或限流(429 limit_requests)按阶梯等待后重试；其余错误直接放弃。
                # 429 常见于共用 API key 的多服务并发场景，退避重试通常可恢复。
                retryable = (
                    "timeout" in err_lower
                    or "429" in err_lower
                    or "limit_requests" in err_lower
                    or "throttl" in err_lower
                    or "rate limit" in err_lower
                )
                if retryable:
                    wait_time = (attempt + 1) * 10
                    logger.info(f"等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    break

        except Exception as e:
            logger.error(f"第{attempt+1}次尝试异常: {e}")
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "error": f"所有重试均失败: {str(e)}",
                    "file_path": pdf_path
                }
            time.sleep(5)
            result = {"success": False, "error": str(e)}

    return result


def extract_personal_pdf_metadata(pdf_path):
    """在 Backend 请求中同步复用普通采集题录提取，错误只暴露可安全持久化的说明。

    Args:
        pdf_path: 经附件清单确认的本地 PDF 原件。

    Returns:
        尚待业务层校验和规范化的模型题录字典。

    Raises:
        RuntimeError: 凭据缺失或提取失败，由 Backend 保留失败状态供用户重试。
    """
    key = (os.getenv('QWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY', '')).strip()
    if not key:
        raise RuntimeError('个人文件题录提取缺少 QWEN_API_KEY 或 DASHSCOPE_API_KEY')
    with setup_qwen_client(key) as client:
        # Backend 以数据库完整题录作为重试复用条件，不读取可能缺字段的旧内存缓存。
        result = robust_extract_pdf_info(client, pdf_path, max_retries=1, use_cache=False)
    if not result or not result.get('success'):
        raise RuntimeError('个人文件题录提取失败，请重试或检查模型服务')
    return result.get('extracted_data')
