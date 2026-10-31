"""覆盖主库 BIT(1) 布尔列的安全转换。"""

from app.foreign_integration.api import _flag_value


def test_flag_value_handles_bit_bytes() -> None:
    """b"\\x00" 是非空字节，直接 bool() 会误判为真；必须按位解析。"""
    assert _flag_value(b"\x00") is False
    assert _flag_value(b"\x01") is True
    assert _flag_value(bytearray(b"\x00")) is False
    assert _flag_value(0) is False
    assert _flag_value(1) is True
    assert _flag_value(True) is True
    assert _flag_value(False) is False
    assert _flag_value(None) is False
