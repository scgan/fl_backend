"""保留可定位的错误原因，隔离凭据、图片和未经筛选的响应正文。"""
import unittest

import httpx

from app.utils.ocr_errors import (
    collect_ocr_warnings, is_content_rejection, record_ocr_warning, safe_http_error,
)


class OCRErrorTests(unittest.TestCase):
    """验证上游错误只保留有限排障字段，剔除密钥、图像、输入和原响应正文。"""
    def test_only_explicit_content_rejection_code_is_skippable(self):
        """仅接受 HTTP 400 的审核错误码，其他状态及同文案错误仍不可跳页。"""
        cases = [
            (400, {"error": {"code": "data_inspection_failed"}}, True),
            (400, {"code": "DataInspectionFailed"}, True),
            (400, {"error": {"type": "data_inspection_failed"}}, True),
            (500, {"error": {"code": "data_inspection_failed"}}, False),
            (403, {"error": {"code": "data_inspection_failed"}}, False),
            (400, {"error": {"code": "bad_input", "message": "data_inspection_failed"}}, False),
            (400, {"error": "data_inspection_failed"}, False),
            (400, [], False),
        ]
        for status, body, expected in cases:
            with self.subTest(status=status, body=body):
                self.assertEqual(is_content_rejection(httpx.Response(status, json=body)), expected)
        self.assertFalse(is_content_rejection(httpx.Response(400, text="not JSON")))

    def test_warning_collection_is_scoped_and_resets_after_failure(self):
        """嵌套构建及失败退出均隔离警告，后续文献不能沿用前一篇的缺页信息。"""
        record_ocr_warning("outside")
        with collect_ocr_warnings() as outer:
            record_ocr_warning("first")
            with self.assertRaises(RuntimeError):
                with collect_ocr_warnings() as inner:
                    record_ocr_warning("nested")
                    raise RuntimeError("aborted")
            record_ocr_warning("last")
        self.assertEqual(outer, ["first", "last"])
        self.assertEqual(inner, ["nested"])
        with collect_ocr_warnings() as fresh:
            self.assertEqual(fresh, [])

    def test_structured_error_preserves_reason_and_redacts_sensitive_values(self):
        """验证结构化原因和请求 ID 可见，敏感值被替换。"""
        response = httpx.Response(400, headers={'x-request-id': 'request-123'}, json={
            'error': {'type': 'invalid_request_error', 'code': 'bad_input',
                      'param': 'max_tokens', 'message': 'maximum exceeded; key=offline-secret '
                      'Bearer another-secret data:image/png;base64,' + 'A' * 200},
            'request_body': 'arbitrary sensitive body',
        })
        summary = safe_http_error(response, secrets=('offline-secret',))
        for expected in ('HTTP 400', 'invalid_request_error', 'bad_input', 'max_tokens',
                         'maximum exceeded', 'request-123'):
            self.assertIn(expected, summary)
        for forbidden in ('offline-secret', 'another-secret', 'data:image', 'A' * 100,
                          'arbitrary sensitive body'):
            self.assertNotIn(forbidden, summary)

    def test_validation_error_does_not_include_input_or_context(self):
        """验证参数错误只保留消息与位置，不包含 input 或 ctx。"""
        response = httpx.Response(400, json={'detail': [
            {'msg': 'Invalid image', 'loc': ['body', 'messages'],
             'input': 'private request image', 'ctx': {'secret': 'private context'}}]})
        summary = safe_http_error(response)
        self.assertIn('Invalid image', summary)
        self.assertIn('messages', summary)
        self.assertNotIn('private', summary)

    def test_non_json_response_preserves_request_id_without_raw_body(self):
        """验证非 JSON 错误保留请求 ID，不泄露未筛选正文。"""
        response = httpx.Response(400, headers={'x-request-id': 'req-html'},
                                  text='<html>arbitrary server body with a secret</html>')
        self.assertEqual(safe_http_error(response), 'AICP OCR HTTP 400 [request_id=req-html]')

    def test_string_error_is_bounded_and_single_line(self):
        """验证文本错误被压平为单行且截断到约定长度。"""
        response = httpx.Response(400, json={'error': 'Invalid image\n' + 'x ' * 2000})
        summary = safe_http_error(response)
        self.assertIn('Invalid image', summary)
        self.assertNotIn('\n', summary)
        self.assertLess(len(summary), 800)
