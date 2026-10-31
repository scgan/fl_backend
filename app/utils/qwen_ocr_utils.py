"""按页识别扫描 PDF，保留真实页码、PDF 坐标和明确的定位精度。"""
from __future__ import annotations

import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from functools import partial
import io
import logging
import math
import os

import fitz
from PIL import Image

from app.utils.bailian_client import BailianError, post_json
from app.utils.ocr_errors import OCRContentRejected, record_ocr_warning
from app.utils.pdf_layout import order_positioned_lines

LOGGER = logging.getLogger(__name__)


class OCRTruncated(BailianError):
    """识别输出不完整，必须分区域恢复；恢复失败时不得发布不完整索引。"""


class OCRPositionOnly(BailianError):
    """该模型返回检测框但未同时返回识别文字。"""


def _rectangle(word):
    """将八点多边形或中心式旋转框转为像素轴对齐矩形。

    Args:
        word: OCR 返回的文字项，含 location 或 rotate_rect。

    Returns:
        像素坐标的 x0、y0、x1、y1 四元组。

    Raises:
        BailianError: 坐标缺失、含非有限数或旋转框尺寸无效。
    """
    location = word.get("location")
    if isinstance(location, list) and len(location) == 8:
        points = list(zip(location[::2], location[1::2]))
    else:
        rect = word.get("rotate_rect")
        if not isinstance(rect, list) or len(rect) != 5:
            raise BailianError("OCR 文字行缺少可靠坐标")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in rect):
            raise BailianError("OCR 旋转框坐标无效")
        cx, cy, width, height, angle = rect
        if width <= 0 or height <= 0:
            raise BailianError("OCR 旋转框大小无效")
        cosine, sine = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        points = [(cx + x * cosine - y * sine, cy + x * sine + y * cosine)
                  for x, y in ((-width / 2, -height / 2), (width / 2, -height / 2),
                               (width / 2, height / 2), (-width / 2, height / 2))]
    if any(type(v) not in (int, float) or not math.isfinite(v) for p in points for v in p):
        raise BailianError("OCR 文字坐标无效")
    xs, ys = zip(*points)
    return min(xs), min(ys), max(xs), max(ys)


def _words(body):
    """校验高精 OCR 完成原因，展开同时含文字和坐标的结果项。

    Args:
        body: 百炼原生 OCR 响应字典。

    Returns:
        所有 content 分段的 words_info 合并列表。

    Raises:
        OCRTruncated: 输出达到长度上限。
        OCRPositionOnly: 缺少可配对的文字与位置，需要降级识别。
        BailianError: 响应结构无效或未正常完成。
    """
    try:
        choice = body["output"]["choices"][0]
        reason = choice["finish_reason"]
        if reason == "length":
            raise OCRTruncated("OCR 输出截断，需要分区域识别")
        if reason != "stop":
            raise BailianError("OCR 未正常完成识别")
        results = [part["ocr_result"]["words_info"]
                   for part in choice["message"]["content"]
                   if isinstance(part, dict) and isinstance(part.get("ocr_result"), dict)
                   and "words_info" in part["ocr_result"]]
    except (KeyError, IndexError, TypeError):
        raise BailianError("OCR 响应缺少高精识别结果") from None
    if not results or any(not isinstance(words, list) for words in results):
        raise OCRPositionOnly("OCR 响应未包含可用的文字与位置组合")
    return [word for words in results for word in words]


def _split_row(image):
    """在中部附近寻找最少墨迹的行，降低区域切口截断文字的概率。"""
    gray = image.convert("L").resize((min(256, image.width), image.height))
    start, stop = int(image.height * 0.43), int(image.height * 0.57)
    return min(range(max(1, start), max(2, stop)),
               key=lambda y: (sum(v < 220 for v in gray.crop((0, y, gray.width, y + 1)).getdata()),
                              abs(y - image.height / 2)))


def _located_lines(words, image, origin):
    """验证图像内真实行框并叠加分区偏移，不推测无法验证的位置。

    Args:
        words: 高精 OCR 文字项列表。
        image: 本次输入图像，用于边界校验。
        origin: 当前分区相对整页的像素左上角。

    Returns:
        文字、整页像素框、line 精度组成的三元组列表。

    Raises:
        OCRPositionOnly: 文字和位置不配对，或坐标无法可靠验证。
    """
    lines = []
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("text"), str):
            raise OCRPositionOnly("高精识别未同时返回文字和位置")
        text = word["text"].strip()
        if not text:
            continue
        try:
            x0, y0, x1, y1 = _rectangle(word)
            if not (0 <= x0 < x1 <= image.width + 2 and 0 <= y0 < y1 <= image.height + 2):
                raise BailianError("OCR 文字坐标超出输入图像范围")
        except BailianError:
            raise OCRPositionOnly("高精识别位置无效，使用可靠的页级文字识别") from None
        lines.append((text, [x0 + origin[0], y0 + origin[1],
                             min(x1, image.width) + origin[0], min(y1, image.height) + origin[1]], "line"))
    return lines


def _recognize(image, model, timeout, *, origin=(0, 0), depth=0):
    """识别百炼图像，按可靠行坐标、页级文字、分区恢复的顺序处理异常。

    Args:
        image: 整页或分区 Pillow 图像。
        model: 本文档固定使用的 OCR 模型。
        timeout: 单次请求超时秒数。
        origin: 当前图像相对整页的像素偏移。
        depth: 分区递归深度，决定精度标记及恢复上限。

    Returns:
        文字、整页像素框和实际定位精度组成的三元组列表。

    Raises:
        BailianError: 模型调用或响应无效。
        OCRTruncated: 分区恢复仍无法获得完整文本。
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{
            "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "enable_rotate": False,
        }]}]},
        "parameters": {"ocr_options": {"task": "advanced_recognition"}, "max_tokens": 8192},
    }
    # 1. 请求高精文字及行框，并验证返回的内容与几何位置能够配对
    body = post_json("/services/aigc/multimodal-generation/generation", payload, timeout=timeout)
    try:
        words = _words(body)
        return _located_lines(words, image, origin)
    except OCRPositionOnly:
        LOGGER.info("Qwen OCR: 高精识别未返回可靠行位置，改用同模型文字识别和页/区域定位")
        # 2. 无可靠行框时用同模型文字识别降级，只声明整页或分区边界，绝不伪造行框
        payload["parameters"]["ocr_options"]["task"] = "text_recognition"
        body = post_json("/services/aigc/multimodal-generation/generation", payload, timeout=timeout)
        try:
            choice = body["output"]["choices"][0]
            if choice["finish_reason"] == "length":
                raise OCRTruncated("OCR 输出截断，需要分区域识别")
            if choice["finish_reason"] != "stop":
                raise BailianError("OCR 未正常完成识别")
            parts = choice["message"]["content"]
            values = [part.get("ocr_result", {}).get("processed_text", part.get("text")) for part in parts]
            if not values or any(not isinstance(value, str) for value in values):
                raise BailianError("通用 OCR 未返回识别文字")
            text = "\n".join(values).strip()
            if not text:
                return []
            # 识别文字可靠但缺少行位置时，仅提供真实的页/区域边界，不猜测文字坐标。
            return [(text, [origin[0], origin[1], origin[0] + image.width, origin[1] + image.height],
                     "page" if depth == 0 else "region")]
        except OCRTruncated:
            # 3. 文字响应截断时递归分区恢复，超过恢复上限向上传播失败
            return _recognize_split(image, model, timeout, origin, depth)
        except (KeyError, IndexError, TypeError, AttributeError):
            raise BailianError("通用 OCR 响应格式错误") from None
    except OCRTruncated:
        return _recognize_split(image, model, timeout, origin, depth)


def _recognize_aicp(image, model, timeout, *, client, origin=(0, 0), depth=0):
    """用独立 AICP 客户端识别图像，仅声明真实的页或分区定位精度。

    Args:
        image: 当前图像。
        model: 固定模型名称，向分区恢复传递。
        timeout: 单次请求超时。
        client: 当前文档共享且由外层关闭的 AICP 客户端。
        origin: 当前分区相对整页的像素偏移。
        depth: 分区递归深度。

    Returns:
        文字及页或分区边界列表；空白图像返回空列表。
    """
    from app.utils.aicp_ocr_client import AICPOCRTruncated

    try:
        text = client.recognize(image, timeout)
    except AICPOCRTruncated:
        return _recognize_split(image, model, timeout, origin, depth,
                                recognize=partial(_recognize_aicp, client=client))
    if not text:
        return []
    # 兼容接口只验证过纯文字输出；边界来自实际输入，不推算文字行框。
    return [(text, [origin[0], origin[1], origin[0] + image.width, origin[1] + image.height],
             "page" if depth == 0 else "region")]


def _recognize_split(image, model, timeout, origin, depth, recognize=None):
    """在较少墨迹处上下分区识别，保留整页偏移，限制递归避免无限恢复。

    Args:
        image: 被截断的输入图像。
        model: 本次 OCR 模型。
        timeout: 单次识别超时。
        origin: 当前分区相对整页的像素偏移。
        depth: 已执行的分区层数。
        recognize: 可注入 AICP 识别器，缺省使用百炼识别器。

    Returns:
        上下区域按顺序拼接的识别项列表。

    Raises:
        OCRTruncated: 深度达到四层或图像过小，不能继续可靠恢复。
    """
    if depth >= 4 or image.height < 128:
        raise OCRTruncated("OCR 分区域后仍截断，未发布不完整索引")
    split = _split_row(image)
    LOGGER.info("PDF OCR split: depth=%s height=%s", depth, image.height)
    recognize = recognize or _recognize
    return (recognize(image.crop((0, 0, image.width, split)), model, timeout,
                       origin=origin, depth=depth + 1)
            + recognize(image.crop((0, split, image.width, image.height)), model, timeout,
                         origin=(origin[0], origin[1] + split), depth=depth + 1))


def _valid_blocks(value, page, width, height):
    """检查文字块均属目标原页、文字非空且 PDF 坐标有限并位于页面内；允许空列表。"""
    return isinstance(value, list) and all(
        isinstance(b, dict) and b.get("page_idx") == page
        and isinstance(b.get("text"), str) and b["text"].strip()
        and isinstance(b.get("bbox"), list) and len(b["bbox"]) == 4
        and all(type(v) in (float, int) and math.isfinite(v) for v in b["bbox"])
        and 0 <= b["bbox"][0] < b["bbox"][2] <= width
        and 0 <= b["bbox"][1] < b["bbox"][3] <= height for b in value)


def read_pdf_pages(local_path, page_indices):
    """按当前 provider 识别指定页，保持原页索引、PDF 坐标及真实定位精度。

    Args:
        local_path: 待处理 PDF 的本地路径。
        page_indices: 从零开始的原附件页索引，内部去重升序处理。

    Returns:
        按原页顺序排列且 block_id 连续的文字块列表，不读写 OCR 缓存。
        内容审核明确拒绝的页不返回 OCR 文字，其原页号及原因作为构建警告记录。

    Raises:
        BailianError: provider、模型、DPI、超时、并发或页索引无效，或坐标校验失败。
    """
    # 1. 固定本次 provider、模型和请求预算；不同 provider 的凭据与结果精度分别处理
    provider = os.getenv("FOREIGN_OCR_PROVIDER", "qwen")
    if provider == "mineru":
        from app.utils.mineru_utils import read_pdf_pages as legacy_read
        return legacy_read(local_path, page_indices)
    if provider == "aicp":
        from app.utils.aicp_ocr_client import AICPOCRClient, OCRSettings

        settings = OCRSettings.from_env()
        model = settings.model
    elif provider == "qwen":
        model = os.getenv("FOREIGN_OCR_MODEL", "qwen-vl-ocr")
        if model != "qwen-vl-ocr":
            raise BailianError("百炼 OCR 适配器仅支持 qwen-vl-ocr")
    else:
        raise BailianError("FOREIGN_OCR_PROVIDER 必须为 aicp、qwen 或 mineru")
    try:
        dpi = int(os.getenv("FOREIGN_OCR_DPI", "150"))
        timeout = int(os.getenv("FOREIGN_OCR_TIMEOUT", "120"))
        concurrency = int(os.getenv("FOREIGN_OCR_CONCURRENCY", "4"))
    except ValueError:
        raise BailianError("OCR DPI 和超时必须为整数") from None
    if not 72 <= dpi <= 300 or not 1 <= timeout <= 600 or not 1 <= concurrency <= 8:
        raise BailianError("OCR DPI 必须为 72..300，超时为 1..600 秒，并发为 1..8")
    recognized_pages = {}
    pending = {}

    def finish(futures):
        """收集页面结果，记录审核拒绝并跳页，其余结果转换为原页 PDF 坐标。

        Args:
            futures: 已完成的识别 Future 集合；其页面几何信息来自 pending。

        Raises:
            BailianError: 除明确内容审核拒绝外的识别失败，或文字块不满足边界约束。
        """
        for future in futures:
            page_idx, width, height, image_width, image_height = pending.pop(future)
            try:
                lines = order_positioned_lines(future.result())
            except OCRContentRejected as error:
                # 即使只有分区被拒绝，也舍弃整页 OCR，避免发布半页识别结果。
                message = (f'PDF OCR 已跳过: file={os.path.basename(local_path)} '
                           f'page={page_idx + 1} provider={provider} model={model} '
                           f'image_pixels={image_width}x{image_height}: {error}')
                LOGGER.warning('%s', message)
                record_ocr_warning(message)
                recognized_pages[page_idx] = []
                continue
            except BailianError as error:
                # 保留 OCRTruncated 等异常类型，让原有恢复与失败流程继续生效。
                raise type(error)(
                    f'PDF OCR 失败: page={page_idx + 1} provider={provider} model={model} '
                    f'image_pixels={image_width}x{image_height}: {error}'
                ) from None
            result = [{"text": text + "\n", "page_idx": page_idx,
                       "bbox": [x0 * width / image_width, height - y1 * height / image_height,
                                x1 * width / image_width, height - y0 * height / image_height],
                       "block_id": index, "ocr_model": model, "location_precision": precision}
                      for index, (text, (x0, y0, x1, y1), precision) in enumerate(lines)]
            if not _valid_blocks(result, page_idx, width, height):
                raise BailianError("OCR 坐标转换无效，未发布索引")
            recognized_pages[page_idx] = result

    # PyMuPDF 仅在当前线程解析和渲染；线程池只负责 HTTP，避免并发访问 PDF 对象。
    # 2. 主线程逐页渲染，线程池只并发 HTTP 识别，不跨线程共享 PDF 页面对象
    client_context = AICPOCRClient(settings, timeout) if provider == "aicp" else nullcontext()
    with client_context as client, fitz.open(local_path) as pdf, ThreadPoolExecutor(max_workers=concurrency) as pool:
        recognize = partial(_recognize_aicp, client=client) if provider == "aicp" else _recognize
        pages = sorted(set(page_indices))
        if not pages or any(type(p) is not int or not 0 <= p < len(pdf) for p in pages):
            raise BailianError("OCR 待处理页码无效")
        for page_idx in pages:
            page = pdf[page_idx]
            # PDF.js 的 bbox 使用未旋转 PDF 坐标；识别图像也使用该坐标系。
            page.set_rotation(0)
            width, height = page.rect.width, page.rect.height
            # 控制图像总像素，提前对齐 32 像素，避免模型再次缩放造成坐标歧义。
            scale = min(dpi / 72, math.sqrt(4_000_000 / (width * height)))
            pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
            image = image.resize((max(32, round(image.width / 32) * 32),
                                  max(32, round(image.height / 32) * 32)))
            LOGGER.info("PDF OCR: provider=%s page=%s pixels=%sx%s", provider, page_idx + 1, image.width, image.height)
            future = pool.submit(recognize, image, model, timeout)
            pending[future] = (page_idx, width, height, image.width, image.height)
            if len(pending) >= concurrency:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                finish(done)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            finish(done)
    # 3. 按原页顺序恢复异步完成结果并统一编号，避免网络完成顺序影响原文定位
    blocks = [block for page_idx in pages for block in recognized_pages[page_idx]]
    for index, block in enumerate(blocks):
        block["block_id"] = index
    return blocks
