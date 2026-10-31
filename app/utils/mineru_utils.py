"""MinerU v4 文件上传客户端；输出统一为原文页码、PDF 左下原点坐标。

仅在解析层发现扫描页时调用，每次重新识别，不保存或复用 OCR 缓存。
API/结果异常向上传播，由现有 worker 决定重试和失败状态。
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import time
import zipfile
from html.parser import HTMLParser
from uuid import uuid4
from urllib.parse import quote, urlparse

import requests


LOGGER = logging.getLogger(__name__)
MAX_FILE_BYTES = 200 * 1024 * 1024
MAX_RESULT_BYTES = 256 * 1024 * 1024


class MinerUError(RuntimeError):
    """可由索引任务记录的 OCR 错误，不包含 Token 或签名下载链接。"""


class _TableText(HTMLParser):
    """将 MinerU 的 HTML 表格转为保留行列分隔的文本。

    Attributes:
        parts: 解析中累积的单元格文本、制表符和换行。
    """
    def __init__(self):
        """初始化解析器及本张表格独立的文本累加列表。"""
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        """保留单元格中的文本，由结束标签决定行列分隔。"""
        self.parts.append(data)

    def handle_endtag(self, tag):
        """在单元格末尾补制表符，在行或段落末尾补换行。"""
        if tag in {"td", "th"}:
            self.parts.append("\t")
        elif tag in {"tr", "p"}:
            self.parts.append("\n")


def _block_text(raw):
    """按内容字段顺序合并正文、列表、表格和图注，并将 HTML 表格转为纯文本。"""
    parts = []
    for field in ("text", "list_items", "table_caption", "table_body",
                  "table_footnote", "image_caption", "image_footnote"):
        value = raw.get(field)
        if isinstance(value, list):
            value = "\n".join(item for item in value if isinstance(item, str))
        if isinstance(value, str) and value.strip():
            if field == "table_body":
                parser = _TableText()
                parser.feed(value)
                value = "".join(parser.parts)
            parts.append(value.strip())
    return "\n".join(parts)


def parse_content_list(content, page_geometry):
    """V4 content_list 的 bbox 为 0..1000 左上原点；映射回上传页对应的原页。

    page_geometry: [(原文 0-based 页码, 页面宽度, 页面高度), ...]。
    不接受没有可靠页码/坐标的文字，避免部分内容被静默丢弃或错页引用。
    完成的 OCR 结果可以不包含空白页/纯图片页，甚至整个分片没有文字；
    是否可索引由调用方合并本地文字和全部 OCR 分片后判断。
    """
    if not isinstance(content, list):
        raise MinerUError("MinerU content_list 格式错误")
    blocks = []
    for raw in content:
        if not isinstance(raw, dict):
            raise MinerUError("MinerU 内容块格式错误")
        value = _block_text(raw)
        if not value:
            continue
        page = raw.get("page_idx")
        bbox = raw.get("bbox")
        if type(page) is not int or not 0 <= page < len(page_geometry):
            raise MinerUError("MinerU 返回无效页码")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise MinerUError("MinerU 文字块缺少坐标")
        try:
            x0, top, x1, bottom = map(float, bbox)
        except (TypeError, ValueError):
            raise MinerUError("MinerU 坐标格式错误") from None
        if not all(math.isfinite(v) for v in (x0, top, x1, bottom)) or not (
                0 <= x0 < x1 <= 1000 and 0 <= top < bottom <= 1000):
            raise MinerUError("MinerU 坐标超出 0..1000 范围")
        original_page, width, height = page_geometry[page]
        blocks.append({"text": value + "\n", "page_idx": original_page,
                       "bbox": [x0 * width / 1000, height * (1 - bottom / 1000),
                                x1 * width / 1000, height * (1 - top / 1000)],
                       "block_id": len(blocks)})
    return blocks


class MinerUClient:
    """在同一截止时间内完成扫描页上传、任务轮询和结果下载。

    Attributes:
        token: 当前请求使用的 MinerU 凭据，不写入错误信息。
        base: v4 API 基础地址。
        model: vlm 或 pipeline 模型版本。
        language: 识别语言代码。
        timeout: 本客户端生命周期的总处理预算秒数。
        max_pages: 每个上传分片的页数上限，范围 1..600。
        deadline: 使用单调时钟计算的截止时间，多个分片共享。
    """
    def __init__(self):
        """读取并校验 OCR 配置，建立覆盖上传及轮询的统一时间预算。

        Raises:
            MinerUError: Token 缺失，模型或超时、页数配置不符合约束。
        """
        self.token = os.getenv("MINERU_API_TOKEN", "").strip()
        if not self.token:
            raise MinerUError("扫描 PDF 需要 OCR，请配置有效的 MINERU_API_TOKEN")
        self.base = os.getenv("MINERU_BASE_URL", "https://mineru.net/api/v4").rstrip("/")
        self.model = os.getenv("MINERU_MODEL_VERSION", "vlm")
        self.language = os.getenv("MINERU_LANGUAGE", "ch")
        try:
            self.timeout = int(os.getenv("MINERU_TIMEOUT", "1200"))
            self.max_pages = int(os.getenv("MINERU_MAX_PAGES", "200"))
        except ValueError:
            raise MinerUError("MinerU 超时和分片页数必须为整数") from None
        if self.timeout <= 0 or not 1 <= self.max_pages <= 600:
            raise MinerUError("MinerU 超时必须大于 0，分片页数必须为 1..600")
        if self.model not in {"vlm", "pipeline"}:
            raise MinerUError("PDF OCR 的 MINERU_MODEL_VERSION 必须为 vlm 或 pipeline")
        self.deadline = time.monotonic() + self.timeout

    def _remaining(self):
        """返回共享截止时间前剩余秒数。

        Returns:
            单调时钟下的正数剩余秒数。

        Raises:
            MinerUError: 本客户端的 OCR 总预算已经耗尽。
        """
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise MinerUError("MinerU OCR 总处理时间超时")
        return remaining

    def _request(self, method, url, **kwargs):
        """按剩余总预算发送 HTTP 请求，不将链接或凭据带入错误信息。

        Args:
            method: HTTP 方法。
            url: API、上传或签名下载地址，仅接受 HTTP/HTTPS。
            **kwargs: 转发给 requests 的请求参数。

        Returns:
            由调用方关闭的成功响应。

        Raises:
            MinerUError: 链接无效、总预算耗尽、网络错误或非成功 HTTP 状态。
        """
        if urlparse(url).scheme not in {"https", "http"}:
            raise MinerUError("MinerU 返回无效服务链接")
        try:
            response = requests.request(method, url,
                                        timeout=min(180, self._remaining()), **kwargs)
        except requests.RequestException as error:
            raise MinerUError(f"MinerU 网络请求失败 ({type(error).__name__})") from None
        if not 200 <= response.status_code < 300:
            status = response.status_code
            response.close()
            hint = "，请更新 MINERU_API_TOKEN" if status in (401, 403) else ""
            raise MinerUError(f"MinerU HTTP {status}{hint}")
        return response

    def _api(self, method, endpoint, **kwargs):
        """调用带认证的 MinerU API 并校验 HTTP 之外的业务状态。

        Args:
            method: HTTP 方法。
            endpoint: 相对于 v4 基础地址的路径。
            **kwargs: 请求体等附加参数。

        Returns:
            业务成功响应中的 data 字典。

        Raises:
            MinerUError: 请求失败、JSON 无效、业务码失败或 data 缺失。
        """
        with self._request(method, self.base + endpoint, headers={
                "Authorization": f"Bearer {self.token}"}, **kwargs) as response:
            try:
                body = response.json()
            except ValueError:
                raise MinerUError("MinerU 返回非 JSON 响应") from None
        if not isinstance(body, dict) or body.get("code") != 0:
            code = body.get("code") if isinstance(body, dict) else "invalid"
            raise MinerUError(f"MinerU API 业务失败 (code={code})")
        if not isinstance(body.get("data"), dict):
            raise MinerUError("MinerU 响应缺少 data")
        return body["data"]

    def extract(self, pdf_bytes, data_id, geometry):
        """上传一个扫描页分片，等待匹配任务完成并恢复原 PDF 页码与坐标。

        Args:
            pdf_bytes: 已按选定页面组成的 PDF 字节，不超过 200 MB。
            data_id: 本分片唯一标识，用于防止取到批次中其他任务结果。
            geometry: 按上传页顺序保存的原页索引、宽度和高度列表。

        Returns:
            坐标原点为 PDF 左下角、页索引指向原附件的文字块列表。

        Raises:
            MinerUError: 文件过大、响应不完整、任务不匹配、失败、超时或结果无效。
        """
        # 1. 校验分片大小并申请上传地址，签名上传链接不附加 API 认证头
        if len(pdf_bytes) > MAX_FILE_BYTES:
            raise MinerUError("OCR 分片超过 200MB，请调小 MINERU_MAX_PAGES")
        data = self._api("POST", "/file-urls/batch", json={
            "files": [{"name": "scan.pdf", "data_id": data_id, "is_ocr": True}],
            "model_version": self.model, "language": self.language,
            "enable_formula": True, "enable_table": True,
        })
        urls, batch = data.get("file_urls"), data.get("batch_id")
        if not isinstance(urls, list) or len(urls) != 1 or not isinstance(urls[0], str) or not batch:
            raise MinerUError("MinerU 未返回上传链接或 batch_id")
        # 上传和下载请求不携带 API Token。
        with self._request("PUT", urls[0], data=pdf_bytes):
            pass
        # 2. 在共享截止时间内轮询匹配 data_id 的任务，拒绝串用其他分片结果
        while True:
            data = self._api("GET", "/extract-results/batch/" + quote(str(batch), safe=""))
            results = data.get("extract_result") or []
            if not isinstance(results, list):
                raise MinerUError("MinerU 批量结果格式错误")
            matches = [item for item in results if isinstance(item, dict)
                       and item.get("data_id") == data_id]
            if len(matches) > 1 or (results and not matches):
                raise MinerUError("MinerU 返回的任务标识不匹配")
            if matches:
                item = matches[0]
                state = item.get("state")
                if state == "failed":
                    raise MinerUError("MinerU 扫描件解析失败")
                # 3. 下载唯一结果清单并映射回原附件几何信息，任一步失败都交由 Worker 重试
                if state == "done":
                    url = item.get("full_zip_url")
                    if not isinstance(url, str) or not url:
                        raise MinerUError("MinerU 完成结果缺少 ZIP 链接")
                    return parse_content_list(self._download_content(url), geometry)
                if state not in {"waiting-file", "pending", "running", "converting"}:
                    raise MinerUError("MinerU 返回未知任务状态")
            time.sleep(min(3, self._remaining()))

    def _download_content(self, url):
        """限量下载 ZIP，只读取其中唯一且大小受限的内容清单。

        Args:
            url: 任务完成后返回的结果下载地址。

        Returns:
            内容清单解析后的 JSON 值，结构由后续解析器进一步校验。

        Raises:
            MinerUError: 下载超时、ZIP 或成员超限、成员不唯一或 JSON/ZIP 无效。
        """
        buffer = io.BytesIO()
        try:
            with self._request("GET", url, stream=True) as response:
                for part in response.iter_content(1024 * 1024):
                    self._remaining()
                    if buffer.tell() + len(part) > MAX_RESULT_BYTES:
                        raise MinerUError("MinerU 结果 ZIP 超过大小限制")
                    buffer.write(part)
            with zipfile.ZipFile(buffer) as archive:
                names = [n for n in archive.namelist()
                         if n.endswith("_content_list.json") or n == "content_list.json"]
                if len(names) != 1 or archive.getinfo(names[0]).file_size > 64 * 1024 * 1024:
                    raise MinerUError("MinerU ZIP 缺少唯一且大小有效的 content_list.json")
                return json.loads(archive.read(names[0]))
        except (requests.RequestException, ValueError, zipfile.BadZipFile, OSError):
            raise MinerUError("MinerU 结果下载或 JSON/ZIP 解析失败") from None


def read_pdf_pages(local_path, page_indices):
    """按配置页数分片上传所选页，恢复原始页索引，不读写 OCR 缓存。

    Args:
        local_path: 当前附件的本地 PDF 路径。
        page_indices: 从零开始的原附件页索引，内部去重并升序处理。

    Returns:
        按页面和内容顺序排列、block_id 重新连续编号的文字块列表。

    Raises:
        MinerUError: 配置或页索引无效，或任一分片识别失败。
    """
    import fitz

    # 1. 固定本次识别配置和总时限，再校验原附件页索引
    client = MinerUClient()
    blocks = []
    with fitz.open(local_path) as source:
        pages = sorted(set(page_indices))
        if not pages or any(type(p) is not int or not 0 <= p < len(source) for p in pages):
            raise MinerUError("OCR 待处理页码无效")
        # 2. 按上限生成分片并保存上传页到原页的映射，不让不连续页面重新从零编号
        for start in range(0, len(pages), client.max_pages):
            client._remaining()
            selected = pages[start:start + client.max_pages]
            geometry = [(p, source[p].rect.width, source[p].rect.height) for p in selected]
            with fitz.open() as upload:
                for page in selected:
                    upload.insert_pdf(source, from_page=page, to_page=page)
                pdf_bytes = upload.tobytes(garbage=3, deflate=True)
            LOGGER.info("MinerU OCR: pages=%s", ",".join(str(p + 1) for p in selected))
            result = client.extract(pdf_bytes, uuid4().hex, geometry)
            blocks.extend(result)
    # 3. 合并分片结果后统一块编号，页索引保持原文不变
    for index, block in enumerate(blocks):
        block["block_id"] = index
    return blocks
