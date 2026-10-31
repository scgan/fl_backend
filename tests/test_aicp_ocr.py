"""真实 SDK + 内存 HTTP 传输验证新 OCR，混合 PDF 验证原文定位契约。"""
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import fitz
import httpx
from openai import OpenAI as RealOpenAI
from PIL import Image

from app.utils import aicp_ocr_client as gateway
from app.utils import qwen_ocr_utils as ocr, qdrant_utils
from app.utils.bailian_client import BailianError
from app.utils.ocr_errors import collect_ocr_warnings


def response(text="Recognized evidence", reason="stop", **choice_fields):
    """构造可交给真实 SDK 解析的离线 OCR 响应，支持注入结束原因和候选字段。"""
    return {"id": "ocr-test", "object": "chat.completion", "created": 1,
            "model": "qwen-vl-ocr-2025-11-20", "choices": [
                {"index": 0, "finish_reason": reason,
                 "message": {"role": "assistant", "content": text}, **choice_fields}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class AICPOCRTests(unittest.TestCase):
    """隔离真实 SDK 的 HTTP 传输，验证独立配置、完整文字、定位精度及错误脱敏。"""
    def setUp(self):
        """建立本例临时目录、独立 OCR 环境和请求/客户端记录列表，退出后恢复环境。"""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = mock.patch.dict(os.environ, {
            "FOREIGN_OCR_PROVIDER": "aicp", "FOREIGN_OCR_API_KEY": "offline-ocr-secret",
            "FOREIGN_OCR_BASE_URL": gateway.DEFAULT_BASE_URL, "FOREIGN_OCR_MODEL": gateway.DEFAULT_MODEL,
            "FOREIGN_OCR_DPI": "150", "FOREIGN_OCR_TIMEOUT": "120", "FOREIGN_OCR_CONCURRENCY": "4",
            "FOREIGN_OCR_CACHE_DIR": self.folder.name + "/cache", "QWEN_API_KEY": "unrelated-embedding-key"})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.calls = []
        self.clients = []

    def sdk(self, handler):
        """将真实 SDK 接到 MockTransport，保留凭据、重试配置及连接关闭契约的断言。"""
        def serve(request):
            """记录离线请求，再交给本例注入的响应处理函数。"""
            self.calls.append(request)
            return handler(request)

        def factory(**kwargs):
            """确认使用独立 OCR 凭据且 SDK 不重试，创建可跟踪关闭状态的离线客户端。"""
            self.assertEqual(kwargs["api_key"], "offline-ocr-secret")
            self.assertEqual(kwargs["max_retries"], 0)
            client = RealOpenAI(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(serve)))
            self.clients.append(client)
            return client

        return mock.patch.object(gateway, "OpenAI", side_effect=factory)

    def make_pdf(self, scanned_only=False, rotation=0, scanned_pages=1):
        """创建纯扫描或混合 PDF，用可选旋转及多页图像验证页序恢复。

        Args:
            scanned_only: 是否不添加原生文字页及空白页。
            rotation: 每个扫描页的旋转角度。
            scanned_pages: 附加的扫描页数量。

        Returns:
            临时 PDF 的路径。
        """
        path = Path(self.folder.name) / "mixed.pdf"
        with fitz.open() as page_pdf:
            page = page_pdf.new_page(width=200, height=300)
            page.insert_text((20, 50), "Scanned words")
            raster = page.get_pixmap().tobytes("png")
        with fitz.open() as pdf:
            if not scanned_only:
                page = pdf.new_page(width=200, height=300)
                page.insert_text((20, 50), "Native text")
                pdf.new_page(width=200, height=300)
            for _ in range(scanned_pages):
                page = pdf.new_page(width=200, height=300)
                page.insert_image(page.rect, stream=raster)
                page.set_rotation(rotation)
            pdf.save(path)
        return str(path)

    def test_request_and_repeated_pdf_preserve_page_and_precision(self):
        """验证重复 OCR 每次发请求，保留原页序号、真实整页边界和页级精度。"""
        def serve(request):
            """检查实际 HTTP 端点、模型、既有提示词及图像尺寸，返回固定 OCR 文本。"""
            self.assertEqual(str(request.url), gateway.DEFAULT_BASE_URL + "/chat/completions")
            self.assertEqual(request.headers["Authorization"], "Bearer offline-ocr-secret")
            payload = json.loads(request.content)
            self.assertEqual(payload["model"], gateway.DEFAULT_MODEL)
            self.assertFalse(payload["stream"])
            self.assertEqual(payload["max_tokens"], 8192)
            parts = payload["messages"][0]["content"]
            self.assertEqual(parts[1]["text"], gateway.TEXT_PROMPT)
            image = Image.open(io.BytesIO(base64.b64decode(parts[0]["image_url"]["url"].split(",", 1)[1])))
            self.assertEqual(image.width % 32, 0)
            self.assertEqual(image.height % 32, 0)
            return httpx.Response(200, json=response())

        path = self.make_pdf(rotation=90)
        with self.sdk(serve):
            blocks = qdrant_utils._read_blocks_locally(path)
            self.assertEqual([b["page_idx"] for b in blocks], [0, 2])
            self.assertEqual(blocks[-1]["bbox"], [0, 0, 200, 300])
            self.assertEqual(blocks[-1]["location_precision"], "page")
            self.assertEqual(blocks[-1]["ocr_model"], gateway.DEFAULT_MODEL)
            self.assertEqual(qdrant_utils._read_blocks_locally(path), blocks)
            self.assertEqual(len(self.calls), 2)
            chunks = qdrant_utils._blocks_to_chunks(blocks, "mixed.pdf")
            locations = [b for c in chunks for b in c["metadata"]["bbox_blocks"] if b["page_idx"] == 2]
            self.assertTrue(locations)
            self.assertTrue(all(b["location_precision"] == "page" for b in locations))
        self.assertTrue(all(c.is_closed() for c in self.clients))

    def test_existing_cache_is_ignored_and_each_call_returns_fresh_text(self):
        """验证旧缓存内容不被读取，每次识别都取得当前服务文字。"""
        path = self.make_pdf(scanned_only=True)
        cache = Path(self.folder.name) / 'cache'
        cache.mkdir()
        legacy = cache / 'legacy.json'
        legacy.write_text('[{"text": "stale evidence"}]', encoding='utf-8')
        before = legacy.read_bytes()
        responses = iter([response('First fresh result'), response('Second fresh result')])
        with self.sdk(lambda _: httpx.Response(200, json=next(responses))):
            first = ocr.read_pdf_pages(path, [0])
            second = ocr.read_pdf_pages(path, [0])
        self.assertEqual(first[0]['text'], 'First fresh result\n')
        self.assertEqual(second[0]['text'], 'Second fresh result\n')
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(list(cache.iterdir()), [legacy])
        self.assertEqual(legacy.read_bytes(), before)

    def test_truncation_splits_and_preserves_actual_region_offsets(self):
        """验证截断触发分区识别，结果叠加真实分区偏移。"""
        responses = iter([response("partial", "length"), response("Top"), response("Bottom")])
        with self.sdk(lambda _: httpx.Response(200, json=next(responses))):
            with gateway.AICPOCRClient(gateway.OCRSettings.from_env(), 120) as client:
                lines = ocr._recognize_aicp(Image.new("RGB", (320, 320), "white"), gateway.DEFAULT_MODEL,
                                           120, client=client)
        self.assertEqual(lines, [("Top", [0, 0, 320, 160], "region"),
                                 ("Bottom", [0, 160, 320, 320], "region")])
        self.assertEqual(len(self.calls), 3)

    def test_unrecoverable_truncation_does_not_publish_or_cache(self):
        """验证超过分区恢复能力的截断会失败，不返回不完整文字。"""
        path = self.make_pdf(scanned_only=True)
        with self.sdk(lambda _: httpx.Response(200, json=response("partial", "length"))):
            with self.assertRaises(ocr.OCRTruncated):
                ocr.read_pdf_pages(path, [0])
        self.assertFalse(list((Path(self.folder.name) / "cache").glob("*.json")))

    def test_repeated_blank_page_preserves_native_text_but_empty_document_fails(self):
        """验证 OCR 空白时保留原生文字，全篇无文字则重复执行仍失败。"""
        path = self.make_pdf()
        with self.sdk(lambda _: httpx.Response(200, json=response("  "))):
            self.assertEqual([b["page_idx"] for b in qdrant_utils._read_blocks_locally(path)], [0])
            self.assertEqual([b["page_idx"] for b in qdrant_utils._read_blocks_locally(path)], [0])
            self.assertEqual(len(self.calls), 2)
            with self.assertRaisesRegex(RuntimeError, "没有可索引文字"):
                qdrant_utils._read_blocks_locally(self.make_pdf(scanned_only=True))

    def test_each_call_uses_current_endpoint_model_prompt_and_provider(self):
        """验证每次调用读取当前 OCR 配置与已有请求模板，未复用旧识别结果。"""
        path = self.make_pdf()
        with self.sdk(lambda _: httpx.Response(200, json=response())):
            ocr.read_pdf_pages(path, [2])
            with mock.patch.dict(os.environ, {"FOREIGN_OCR_BASE_URL": "https://other.example/v1"}):
                ocr.read_pdf_pages(path, [2])
            with mock.patch.dict(os.environ, {"FOREIGN_OCR_MODEL": "qwen/other-ocr"}):
                ocr.read_pdf_pages(path, [2])
            with mock.patch.object(gateway, "TEXT_PROMPT", "A revised text-only prompt"):
                ocr.read_pdf_pages(path, [2])
            with mock.patch.dict(os.environ, {"FOREIGN_OCR_PROVIDER": "qwen", "FOREIGN_OCR_MODEL": "qwen-vl-ocr"}), \
                    mock.patch.object(ocr, "_recognize", return_value=[("Old OCR", [0, 0, 32, 32], "line")]) as old:
                ocr.read_pdf_pages(path, [2])
                old.assert_called_once()
            ocr.read_pdf_pages(path, [2])
        self.assertEqual(len(self.calls), 5)
        self.assertFalse((Path(self.folder.name) / "cache").exists())

    def test_transient_status_retries_once_sdk_does_not_retry(self):
        """验证暂时性状态由适配器重试，SDK 重试关闭避免倍增请求。"""
        for status in (408, 409, 429, 500):
            with self.subTest(status=status):
                self.calls.clear()
                responses = iter([httpx.Response(status, json={"error": {"message": "temporary"}},
                                                 headers={"Retry-After": "3"}),
                                  httpx.Response(200, json=response())])
                with self.sdk(lambda _: next(responses)), mock.patch.object(gateway.time, "sleep") as sleep:
                    with gateway.AICPOCRClient(gateway.OCRSettings.from_env(), 120) as client:
                        self.assertEqual(client.recognize(Image.new("RGB", (32, 32)), 120), "Recognized evidence")
                self.assertEqual(len(self.calls), 2)
                sleep.assert_called_once_with(3)

    def test_network_retries_are_bounded_and_errors_do_not_leak_credentials(self):
        """验证网络异常最多请求三次，最终错误不包含独立密钥。"""
        def serve(request):
            """模拟包含测试密钥的连接异常，检验适配器不会透传底层错误。"""
            raise httpx.ConnectError("offline-ocr-secret remote failure", request=request)
        with self.sdk(serve), mock.patch.object(gateway.time, "sleep") as sleep:
            with gateway.AICPOCRClient(gateway.OCRSettings.from_env(), 120) as client:
                with self.assertRaises(BailianError) as error:
                    client.recognize(Image.new("RGB", (32, 32)), 120)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertNotIn("offline-ocr-secret", str(error.exception))

    def test_bad_request_records_page_and_safe_reason_without_retry(self):
        """验证不可重试的请求错误包含页码和安全原因，且不重复请求。"""
        body = {'error': {'type': 'invalid_request_error', 'code': 'bad_input',
                         'param': 'max_tokens', 'message': 'input rejected; offline-ocr-secret'}}
        with self.sdk(lambda _: httpx.Response(400, json=body, headers={'x-request-id': 'req-400'})), \
                mock.patch.object(gateway.time, 'sleep') as sleep, \
                self.assertLogs(gateway.LOGGER.name, level='WARNING') as logs:
            with self.assertRaises(BailianError) as error:
                qdrant_utils._read_blocks_locally(self.make_pdf())
        message = str(error.exception)
        for expected in ('page=3', 'provider=aicp', 'image_pixels=', 'HTTP 400',
                         'bad_input', 'param=max_tokens', 'input rejected', 'req-400'):
            self.assertIn(expected, message)
        self.assertNotIn('offline-ocr-secret', message)
        self.assertNotIn('offline-ocr-secret', '\n'.join(logs.output))
        self.assertIn('max_tokens=8192', '\n'.join(logs.output))
        self.assertEqual(len(self.calls), 1)
        sleep.assert_not_called()

    def test_content_rejection_skips_one_page_and_continues_without_retry(self):
        """中间页被审核拒绝时仍处理后续页，保留原页码及脱敏跳页摘要。"""
        path = self.make_pdf(scanned_only=True, scanned_pages=3)
        responses = iter([
            httpx.Response(200, json=response("Before")),
            httpx.Response(400, json={"error": {
                "code": "data_inspection_failed", "type": "data_inspection_failed",
                "message": "Input rejected; offline-ocr-secret",
            }}, headers={"x-request-id": "req-rejected-page"}),
            httpx.Response(200, json=response("After")),
        ])
        with self.sdk(lambda _: next(responses)), \
                mock.patch.dict(os.environ, {"FOREIGN_OCR_CONCURRENCY": "1"}), \
                mock.patch.object(gateway.time, "sleep") as sleep, \
                collect_ocr_warnings() as warnings:
            blocks = qdrant_utils._read_blocks_locally(path)
        self.assertEqual([block["page_idx"] for block in blocks], [0, 2])
        self.assertEqual([block["text"] for block in blocks], ["Before\n", "After\n"])
        self.assertEqual([block["block_id"] for block in blocks], [0, 1])
        self.assertEqual(len(warnings), 1)
        for expected in ("page=2", "mixed.pdf", "data_inspection_failed", "req-rejected-page"):
            self.assertIn(expected, warnings[0])
        self.assertNotIn("offline-ocr-secret", warnings[0])
        self.assertEqual(len(self.calls), 3)
        sleep.assert_not_called()
        self.assertTrue(all(client.is_closed() for client in self.clients))

    def test_rejected_only_page_without_text_still_fails(self):
        """只有被拒绝的扫描页时仍因无正文失败，不把空文献判为成功。"""
        body = {"error": {"code": "data_inspection_failed"}}
        with self.sdk(lambda _: httpx.Response(400, json=body)), collect_ocr_warnings() as warnings:
            with self.assertRaisesRegex(RuntimeError, "没有可索引文字"):
                qdrant_utils._read_blocks_locally(self.make_pdf(scanned_only=True))
        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(self.calls), 1)

    def test_rejected_region_discards_entire_page_ocr(self):
        """分区恢复中被拒绝时舍弃该页已经识别的其他区域，避免半页结果。"""
        responses = iter([
            httpx.Response(200, json=response("partial", "length")),
            httpx.Response(200, json=response("Top region")),
            httpx.Response(400, json={"error": {"code": "data_inspection_failed"}}),
        ])
        with self.sdk(lambda _: next(responses)), collect_ocr_warnings() as warnings:
            blocks = ocr.read_pdf_pages(self.make_pdf(scanned_only=True), [0])
        self.assertEqual(blocks, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("page=1", warnings[0])
        self.assertEqual(len(self.calls), 3)

    def test_auth_failure_is_not_retried_and_never_cached(self):
        """验证认证失败不重试且不会生成可复用的识别结果。"""
        for status in (401, 403):
            with self.subTest(status=status):
                self.calls.clear()
                with self.sdk(lambda _: httpx.Response(status, text="offline-ocr-secret raw error")), \
                        mock.patch.object(gateway.time, "sleep") as sleep:
                    with self.assertRaises(BailianError) as error:
                        ocr.read_pdf_pages(self.make_pdf(), [2])
                self.assertIn(str(status), str(error.exception))
                self.assertNotIn("offline-ocr-secret", str(error.exception))
                self.assertEqual(len(self.calls), 1)
                sleep.assert_not_called()
                self.assertFalse(list((Path(self.folder.name) / "cache").glob("*.json")))

    def test_incomplete_or_nontext_responses_are_rejected(self):
        """验证不完整结束、缺少候选或非文字消息不能被当作识别成功。"""
        bodies = [{"choices": []}, response("partial", "content_filter"),
                  response(None), response("text", message={"role": "assistant", "content": "text", "refusal": "blocked"})]
        for body in bodies:
            with self.subTest(body=body), self.sdk(lambda _: httpx.Response(200, json=body)):
                with gateway.AICPOCRClient(gateway.OCRSettings.from_env(), 120) as client:
                    with self.assertRaises(BailianError):
                        client.recognize(Image.new("RGB", (32, 32)), 120)

    def test_configuration_never_falls_back_to_other_model_credentials(self):
        """验证缺失独立密钥时失败，不借用其他模型凭据。"""
        with mock.patch.dict(os.environ, {"FOREIGN_OCR_API_KEY": ""}):
            with self.assertRaisesRegex(BailianError, "FOREIGN_OCR_API_KEY"):
                gateway.OCRSettings.from_env()
        for url in ("http://example/v1", "https://example/v1/chat/completions", "https://key@example/v1",
                    "https://example/v1?key=secret", "https://example/v1#fragment"):
            with self.subTest(url=url), mock.patch.dict(os.environ, {"FOREIGN_OCR_BASE_URL": url}):
                with self.assertRaises(BailianError):
                    gateway.OCRSettings.from_env()
        settings = gateway.OCRSettings.from_env()
        self.assertNotIn(settings.api_key, repr(settings))


if __name__ == "__main__":
    unittest.main()
