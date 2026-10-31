"""真实 PDF + 模拟 HTTP 验证扫描页解析，不访问 MinerU 或生产数据库。"""
import io
import json
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import fitz
import requests

from app.utils import mineru_utils as mineru
from app.utils import qdrant_utils


def response(body=None, status=200, content=b""):
    """构造支持上下文管理和分块下载的 requests 响应替身。"""
    result = mock.MagicMock(status_code=status)
    result.__enter__.return_value = result
    result.json.return_value = body
    result.iter_content.return_value = [content]
    return result


def result_zip(blocks):
    """将给定内容块打包为唯一 content_list JSON 的内存 ZIP。"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("scan_content_list.json", json.dumps(blocks))
    return buffer.getvalue()


class MinerUTests(unittest.TestCase):
    """用离线 HTTP 和真实 PDF 验证 MinerU 分片、原页映射、完整结果及失败传播。"""
    def setUp(self):
        """为每例配置临时 PDF 目录和独立 MinerU 环境，退出时自动恢复。"""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = mock.patch.dict(os.environ, {
            "FOREIGN_OCR_PROVIDER": "mineru",
            "MINERU_API_TOKEN": "offline-test-token", "MINERU_MODEL_VERSION": "vlm",
            "MINERU_BASE_URL": "https://mineru.example/api/v4", "MINERU_TIMEOUT": "30",
            "MINERU_LANGUAGE": "ch", "MINERU_MAX_PAGES": "200",
            "MINERU_CACHE_DIR": os.path.join(self.folder.name, "cache"),
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_pdf(self, kinds):
        """按 text、scan、header 或空白类型生成真实多页 PDF。"""
        path = os.path.join(self.folder.name, "paper.pdf")
        with fitz.open() as text_pdf:
            page = text_pdf.new_page(width=200, height=300)
            page.insert_text((20, 50), "Scanned words")
            raster = page.get_pixmap().tobytes("png")
        with fitz.open() as pdf:
            for kind in kinds:
                page = pdf.new_page(width=200, height=300)
                if kind == "text":
                    page.insert_text((20, 50), "Native text")
                elif kind in {"scan", "header"}:
                    page.insert_image(page.rect, stream=raster)
                    if kind == "header":
                        page.insert_text((20, 20), "Header")
            pdf.save(path)
        return path

    def fake_service(self, method, url, **kwargs):
        """模拟上传申请、PUT、匹配任务完成及 ZIP 下载，并记录真实上传页数。"""
        if method == "POST":
            self.payload = kwargs["json"]
            return response({"code": 0, "data": {"batch_id": "batch",
                            "file_urls": ["https://upload.example/file"]}})
        if method == "PUT":
            self.assertNotIn("headers", kwargs)
            with fitz.open(stream=kwargs["data"], filetype="pdf") as pdf:
                self.uploaded_pages = len(pdf)
            return response()
        if "extract-results/batch/" in url:
            return response({"code": 0, "data": {"extract_result": [{
                "data_id": self.payload["files"][0]["data_id"], "state": "done",
                "full_zip_url": "https://download.example/result.zip"}]}})
        self.assertNotIn("headers", kwargs)
        return response(content=result_zip([
            {"text": f"OCR page {p}", "page_idx": p, "bbox": [100, 100, 900, 200]}
            for p in range(self.uploaded_pages)
        ]))

    def test_mixed_pdf_preserves_native_and_original_ocr_page_on_repeat(self):
        """验证混合 PDF 重复处理保留原生文字和扫描页的原始页码。"""
        path = self.make_pdf(["text", "blank", "scan", "header"])
        with mock.patch.object(mineru.requests, "request", side_effect=self.fake_service) as http:
            blocks = qdrant_utils._read_blocks_locally(path)
            self.assertEqual(self.uploaded_pages, 2)
            self.assertTrue(self.payload["files"][0]["is_ocr"])
            self.assertNotIn("is_ocr", self.payload)
            self.assertEqual([b["page_idx"] for b in blocks], [0, 2, 3])
            self.assertEqual(blocks[0]["text"], "Native text\n")
            self.assertEqual(blocks[1]["bbox"], [20, 240, 180, 270])
            self.assertEqual([b["block_id"] for b in blocks], [0, 1, 2])
            calls = http.call_count
            self.assertEqual(qdrant_utils._read_blocks_locally(path), blocks)
            self.assertEqual(http.call_count, calls * 2)
        chunks = qdrant_utils._blocks_to_chunks(blocks, "original.pdf")
        self.assertEqual({b["page_idx"] for c in chunks for b in c["metadata"]["bbox_blocks"]},
                         {0, 2, 3})

    def test_native_pdf_needs_neither_token_nor_http(self):
        """验证纯原生文字 PDF 无需 OCR Token 或远程请求。"""
        path = self.make_pdf(["text", "blank"])
        with mock.patch.dict(os.environ, {"MINERU_API_TOKEN": ""}), \
                mock.patch.object(mineru.requests, "request") as http:
            self.assertEqual(qdrant_utils._read_blocks_locally(path)[0]["text"], "Native text\n")
            http.assert_not_called()

    def test_missing_token_does_not_silently_publish_only_native_pages(self):
        """验证混合 PDF 缺 OCR 凭据时整体失败，不只发布原生页。"""
        path = self.make_pdf(["text", "scan"])
        with mock.patch.dict(os.environ, {"MINERU_API_TOKEN": ""}), \
                self.assertRaisesRegex(mineru.MinerUError, "MINERU_API_TOKEN"):
            qdrant_utils._read_blocks_locally(path)

    def test_pymupdf_fallback_also_recognizes_scans(self):
        """验证主解析器失败后，备用解析仍识别扫描页。"""
        path = self.make_pdf(["text", "scan"])
        with mock.patch.object(qdrant_utils.pdfplumber, "open", side_effect=ValueError), \
                mock.patch.object(mineru.requests, "request", side_effect=self.fake_service):
            blocks = qdrant_utils._read_blocks_locally(path)
        self.assertEqual([b["page_idx"] for b in blocks], [0, 1])

    def test_repeated_split_keeps_noncontiguous_original_pages(self):
        """验证多分片与重复执行不改变不连续扫描页的原页索引。"""
        path = self.make_pdf(["scan", "text", "scan"])
        with mock.patch.dict(os.environ, {"MINERU_MAX_PAGES": "1"}), \
                mock.patch.object(mineru.requests, "request", side_effect=self.fake_service) as http:
            result = mineru.read_pdf_pages(path, [0, 2])
            self.assertEqual([b["page_idx"] for b in result], [0, 2])
            self.assertEqual(sum(c.args[0] == "POST" for c in http.call_args_list), 2)
            mineru.read_pdf_pages(path, [0, 2])
            self.assertEqual(sum(c.args[0] == "POST" for c in http.call_args_list), 4)
            with mock.patch.dict(os.environ, {"MINERU_LANGUAGE": "en"}):
                mineru.read_pdf_pages(path, [0, 2])
            self.assertEqual(sum(c.args[0] == "POST" for c in http.call_args_list), 6)

    def test_poll_waits_for_matching_item_done(self):
        """验证只等待当前 data_id 的完成结果，处理中状态继续轮询。"""
        client = mineru.MinerUClient()
        pending = {"data_id": "key", "state": "running"}
        done = {"data_id": "key", "state": "done", "full_zip_url": "https://cdn.example/x"}
        with mock.patch.object(client, "_api", side_effect=[
            {"batch_id": "batch", "file_urls": ["https://upload.example/x"]},
            {"extract_result": [pending]}, {"extract_result": [done]},
        ]) as api, mock.patch.object(client, "_request", return_value=response()), \
                mock.patch.object(client, "_download_content", return_value=[
                    {"text": "recognized", "page_idx": 0, "bbox": [0, 0, 500, 500]}]), \
                mock.patch.object(mineru.time, "sleep") as sleep:
            result = client.extract(b"pdf", "key", [(9, 200, 300)])
            self.assertEqual(result[0]["page_idx"], 9)
            self.assertEqual(api.call_count, 3)
            sleep.assert_called_once()

    def test_failed_and_mismatched_results_raise(self):
        """验证失败任务、其他分片结果及未知状态都拒绝发布。"""
        for item in ({"data_id": "key", "state": "failed"},
                     {"data_id": "another", "state": "done"}):
            with self.subTest(item=item):
                client = mineru.MinerUClient()
                with mock.patch.object(client, "_api", side_effect=[
                    {"batch_id": "batch", "file_urls": ["https://upload.example/x"]},
                    {"extract_result": [item]},
                ]), mock.patch.object(client, "_request", return_value=response()), \
                        self.assertRaises(mineru.MinerUError):
                    client.extract(b"pdf", "key", [(0, 200, 300)])

    def test_timeout_and_http_errors_do_not_leak_credentials(self):
        """验证总预算和 HTTP 错误向上传播，消息不泄露 Token 或签名链接。"""
        client = mineru.MinerUClient()
        with mock.patch.object(mineru.requests, "request", return_value=response(status=401)), \
                self.assertRaisesRegex(mineru.MinerUError, "MINERU_API_TOKEN"):
            client._api("GET", "/extract-results/batch/x")
        with mock.patch.object(mineru.requests, "request",
                               side_effect=requests.Timeout("secret-signed-url")), \
                self.assertRaisesRegex(mineru.MinerUError, r"网络请求失败 \(Timeout\)") as error:
            client._request("GET", "https://example.com/secret")
        self.assertNotIn("secret", str(error.exception))
        client.deadline = time.monotonic() - 1
        with self.assertRaisesRegex(mineru.MinerUError, "超时"):
            client._request("GET", "https://example.com")

    def test_table_list_formula_and_geometry_validation(self):
        """验证表格、列表与公式正文的提取，以及原页和坐标范围校验。"""
        content = [
            {"type": "table", "table_body": "<table><tr><td>A</td><td>B</td></tr></table>",
             "page_idx": 0, "bbox": [0, 0, 500, 500]},
            {"type": "list", "list_items": ["one", "two"], "page_idx": 0,
             "bbox": [0, 500, 500, 800]},
            {"type": "equation", "text": "x^2", "page_idx": 0,
             "bbox": [0, 800, 500, 1000]},
        ]
        blocks = mineru.parse_content_list(content, [(4, 200, 300)])
        self.assertEqual([b["text"] for b in blocks], ["A\tB\n", "one\ntwo\n", "x^2\n"])
        for invalid in ({"page_idx": -1}, {"bbox": [0, 0, 1001, 1000]},
                        {"bbox": [0, 0, float("nan"), 1000]}):
            with self.subTest(invalid=invalid), self.assertRaises(mineru.MinerUError):
                mineru.parse_content_list([{**content[2], **invalid}], [(4, 200, 300)])
        self.assertEqual(mineru.parse_content_list(content, [(4, 200, 300), (7, 200, 300)]),
                         blocks)

    def test_blank_page_in_ocr_result_keeps_original_page_number_on_repeat(self):
        """验证 OCR 结果中的空白页不压缩后续原始页码。"""
        path = self.make_pdf(["scan", "scan"])
        content = [{"type": "image", "page_idx": 0, "bbox": [0, 0, 1000, 1000]},
                   {"text": "second page", "page_idx": 1, "bbox": [100, 100, 900, 200]}]
        with mock.patch.object(mineru.requests, "request", side_effect=self.fake_service) as http, \
                mock.patch.object(mineru.MinerUClient, "_download_content", return_value=content):
            blocks = qdrant_utils._read_blocks_locally(path)
            self.assertEqual([b["page_idx"] for b in blocks], [1])
            self.assertEqual(blocks[0]["bbox"], [20, 240, 180, 270])
            calls = http.call_count
            self.assertEqual(qdrant_utils._read_blocks_locally(path), blocks)
            self.assertEqual(http.call_count, calls * 2)
        chunks = qdrant_utils._blocks_to_chunks(blocks, "paper.pdf")
        self.assertEqual(chunks[0]["metadata"]["page_start"], 1)

    def test_empty_ocr_batch_and_later_batch_are_recognized_on_each_call(self):
        """验证空分片与后续分片每次都重新识别，不形成缓存命中。"""
        path = self.make_pdf(["scan", "scan"])
        content = [{"text": "later batch", "page_idx": 0, "bbox": [100, 100, 900, 200]}]
        with mock.patch.dict(os.environ, {"MINERU_MAX_PAGES": "1"}), \
                mock.patch.object(mineru.requests, "request", side_effect=self.fake_service) as http, \
                mock.patch.object(mineru.MinerUClient, "_download_content", side_effect=[[], content, [], content]):
            blocks = qdrant_utils._read_blocks_locally(path)
            self.assertEqual([b["page_idx"] for b in blocks], [1])
            self.assertEqual(sum(c.args[0] == "POST" for c in http.call_args_list), 2)
            calls = http.call_count
            self.assertEqual(qdrant_utils._read_blocks_locally(path), blocks)
            self.assertEqual(http.call_count, calls * 2)

    def test_empty_ocr_preserves_native_text_including_scan_page_header(self):
        """验证 OCR 无正文时仍保留原生页及扫描页已有页眉。"""
        path = self.make_pdf(["text", "header", "scan"])
        with mock.patch.object(mineru.requests, "request", side_effect=self.fake_service), \
                mock.patch.object(mineru.MinerUClient, "_download_content", return_value=[]):
            blocks = qdrant_utils._read_blocks_locally(path)
        self.assertEqual([b["page_idx"] for b in blocks], [0, 1])
        self.assertEqual([b["text"] for b in blocks], ["Native text\n", "Header\n"])

    def test_entire_attachment_without_text_still_fails_on_repeated_ocr(self):
        """验证全篇没有可用文字时重复识别仍然失败。"""
        path = self.make_pdf(["scan", "blank"])
        with mock.patch.object(mineru.requests, "request", side_effect=self.fake_service) as http, \
                mock.patch.object(mineru.MinerUClient, "_download_content", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "PDF 没有可索引文字"):
                qdrant_utils._read_blocks_locally(path)
            calls = http.call_count
            with self.assertRaisesRegex(RuntimeError, "PDF 没有可索引文字"):
                qdrant_utils._read_blocks_locally(path)
            self.assertEqual(http.call_count, calls * 2)

    def test_malformed_result_remains_an_error(self):
        """验证损坏或非法结果不被降级为可成功发布的空内容。"""
        for content in ({}, [None], [{"text": "text", "page_idx": 0}]):
            with self.subTest(content=content), self.assertRaises(mineru.MinerUError):
                mineru.parse_content_list(content, [(0, 200, 300)])

    def test_failed_ocr_is_not_cached(self):
        """验证失败 OCR 不会阻止下一次重新请求。"""
        path = self.make_pdf(["scan"])
        with mock.patch.object(mineru.MinerUClient, "extract", side_effect=mineru.MinerUError("failed")), \
                self.assertRaises(mineru.MinerUError):
            mineru.read_pdf_pages(path, [0])
        self.assertFalse(list(Path(os.environ["MINERU_CACHE_DIR"]).glob("*.json")))


if __name__ == "__main__":
    unittest.main()
