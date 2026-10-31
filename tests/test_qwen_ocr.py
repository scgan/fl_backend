"""真实 PDF 渲染与模拟百炼响应验证页码、坐标、缓存和截断恢复。"""
import base64
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import fitz
from PIL import Image

from app.utils import qwen_ocr_utils as ocr, qdrant_utils
from app.utils.bailian_client import BailianError


def result(words, reason="stop"):
    """构造百炼高精 OCR 的结构化候选响应，支持指定结束原因。"""
    return {"output": {"choices": [{"finish_reason": reason,
        "message": {"content": [{"ocr_result": {"words_info": words}}]}}]}}


class QwenOCRTests(unittest.TestCase):
    """用本地图像和请求替身验证百炼 OCR 的原页、旋转、分区与真实精度契约。"""
    def setUp(self):
        """建立临时目录并固定百炼 OCR 配置，保持每例环境相互隔离。"""
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = mock.patch.dict(os.environ, {"FOREIGN_OCR_PROVIDER": "qwen",
            "FOREIGN_OCR_CACHE_DIR": self.folder.name + "/cache", "FOREIGN_OCR_MODEL": "qwen-vl-ocr",
            "FOREIGN_OCR_DPI": "150"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_pdf(self, rotation=0):
        """生成原生文字、空白及扫描页混合 PDF，可旋转扫描页。"""
        path = self.folder.name + "/mixed.pdf"
        with fitz.open() as scan:
            page = scan.new_page(width=200, height=300)
            page.insert_text((20, 50), "Scanned words")
            raster = page.get_pixmap().tobytes("png")
        with fitz.open() as pdf:
            page = pdf.new_page(width=200, height=300)
            page.insert_text((20, 50), "Native text")
            pdf.new_page(width=200, height=300)
            page = pdf.new_page(width=200, height=300)
            page.insert_image(page.rect, stream=raster)
            page.set_rotation(rotation)
            pdf.save(path)
        return path

    def fake(self, endpoint, payload, **kwargs):
        """检查高精识别参数并按输入图像大小返回真实范围内的文字框。"""
        self.assertEqual(payload["model"], "qwen-vl-ocr")
        self.assertEqual(payload["parameters"]["ocr_options"]["task"], "advanced_recognition")
        content = payload["input"]["messages"][0]["content"][0]
        image = Image.open(io.BytesIO(base64.b64decode(content["image"].split(",", 1)[1])))
        width, height = image.size
        self.assertFalse(content["enable_rotate"])
        return result([{"text": "OCR evidence", "location": [width * .1, height * .1,
            width * .9, height * .1, width * .9, height * .2, width * .1, height * .2]}])

    def test_mixed_pdf_original_pages_bbox_and_repeat_ocr(self):
        """验证混合 PDF 的原始页码和边界在重复 OCR 后保持一致。"""
        path = self.make_pdf()
        with mock.patch.object(ocr, "post_json", side_effect=self.fake) as http:
            blocks = qdrant_utils._read_blocks_locally(path)
            self.assertEqual([b["page_idx"] for b in blocks], [0, 2])
            self.assertEqual(blocks[-1]["bbox"], [20, 240, 180, 270])
            self.assertEqual(blocks[-1]["ocr_model"], "qwen-vl-ocr")
            self.assertEqual(qdrant_utils._read_blocks_locally(path), blocks)
            self.assertEqual(http.call_count, 2)
            self.assertFalse((Path(self.folder.name) / "cache").exists())
            chunks = qdrant_utils._blocks_to_chunks(blocks, "mixed.pdf")
            self.assertEqual({b["page_idx"] for c in chunks for b in c["metadata"]["bbox_blocks"]}, {0, 2})

    def test_rotation_is_mapped_to_unrotated_pdf_coordinates(self):
        """验证旋转页的图像坐标恢复到未旋转 PDF 坐标。"""
        with mock.patch.object(ocr, "post_json", side_effect=self.fake):
            blocks = ocr.read_pdf_pages(self.make_pdf(rotation=90), [2])
        self.assertEqual(blocks[0]["bbox"], [20, 240, 180, 270])

    def test_blank_pages_are_recognized_again_and_invalid_results_fail(self):
        """验证空白页仍重新识别，非法响应不会视为正常空白。"""
        path = self.make_pdf()
        with mock.patch.object(ocr, "post_json", return_value=result([])) as http:
            self.assertEqual(ocr.read_pdf_pages(path, [2]), [])
            self.assertEqual(ocr.read_pdf_pages(path, [2]), [])
            self.assertEqual(http.call_count, 2)
        with mock.patch.dict(os.environ, {"FOREIGN_OCR_CACHE_DIR": self.folder.name + "/bad-cache"}):
            for body in ({}, result([{"text": "missing location"}]), result([], "error")):
                with mock.patch.object(ocr, "post_json", return_value=body), self.assertRaises(BailianError):
                    ocr.read_pdf_pages(path, [2])
            self.assertFalse(list(Path(os.environ["FOREIGN_OCR_CACHE_DIR"]).glob("*.json")))

    def test_truncation_splits_image_and_restores_tile_offset(self):
        """验证截断分区后恢复整页像素偏移。"""
        image = Image.new("RGB", (320, 320), "white")
        word = {"text": "line", "location": [10, 20, 40, 20, 40, 40, 10, 40]}
        with mock.patch.object(ocr, "post_json", side_effect=[result([], "length"), result([word]), result([word])]):
            lines = ocr._recognize(image, "qwen-vl-ocr", 120)
        self.assertEqual(lines, [("line", [10, 20, 40, 40], "line"), ("line", [10, 180, 40, 200], "line")])

    def test_position_only_response_uses_text_recognition_and_honest_page_boundary(self):
        """验证只有检测框时降级文字识别，并只标记可靠整页边界。"""
        recognized = {"output": {"choices": [{"finish_reason": "stop", "message": {
            "content": [{"ocr_result": {"processed_text": "Actual scanned evidence"}}]}}]}}
        with mock.patch.object(ocr, "post_json", side_effect=[result([{"pos_list": []}]), recognized]) as http:
            lines = ocr._recognize(Image.new("RGB", (320, 640), "white"), "qwen-vl-ocr", 120)
        self.assertEqual(lines, [("Actual scanned evidence", [0, 0, 320, 640], "page")])
        self.assertEqual(http.call_args.args[1]["parameters"]["ocr_options"]["task"], "text_recognition")

    def test_truncated_text_fallback_is_split_instead_of_published(self):
        """验证文字降级仍截断时必须分区，不发布部分正文。"""
        truncated = {"output": {"choices": [{"finish_reason": "length", "message": {"content": []}}]}}
        word = {"text": "line", "location": [10, 20, 40, 20, 40, 40, 10, 40]}
        with mock.patch.object(ocr, "post_json", side_effect=[result([{"pos_list": []}]), truncated,
                result([word]), result([word])]):
            lines = ocr._recognize(Image.new("RGB", (320, 320), "white"), "qwen-vl-ocr", 120)
        self.assertEqual(len(lines), 2)

    def test_out_of_bounds_location_falls_back_without_fabricating_line_coordinates(self):
        """验证越界检测框触发降级，不生成猜测的文字行框。"""
        bad = result([{"text": "detected", "location": [0, 0, 1000, 0, 1000, 1000, 0, 1000]}])
        text = {"output": {"choices": [{"finish_reason": "stop", "message": {
            "content": [{"text": "Actual evidence"}]}}]}}
        with mock.patch.object(ocr, "post_json", side_effect=[bad, text]):
            self.assertEqual(ocr._recognize(Image.new("RGB", (320, 640)), "qwen-vl-ocr", 120),
                             [("Actual evidence", [0, 0, 320, 640], "page")])

    def test_rotation_rect_and_invalid_polygon(self):
        """验证旋转矩形换算及无效多边形的拒绝行为。"""
        rect = ocr._rectangle({"rotate_rect": [100, 100, 20, 40, 90]})
        for actual, expected in zip(rect, (80, 90, 120, 110)):
            self.assertAlmostEqual(actual, expected)
        with self.assertRaises(BailianError):
            ocr._rectangle({"location": [0, 0, float("nan"), 0, 20, 20, 0, 20]})


if __name__ == "__main__":
    unittest.main()
