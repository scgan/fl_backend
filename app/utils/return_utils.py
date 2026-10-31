"""生成业务接口使用的成功或失败 JSON 信封，并设置失败响应的 HTTP 状态。"""

from flask import jsonify


def success(data=None,msg="请求成功"):
    """生成成功信封；data 为真值时返回数据，否则返回 success 消息。

    Args:
        data: 可 JSON 序列化的数据；空列表、空字典等假值走消息分支。
        msg: 不返回数据时使用的成功提示。

    Returns:
        正文 code 为 200 的 Flask JSON 响应。
    """
    if data:
        return jsonify({
            "code": 200,
            "data": data,
        })
    else:
        return jsonify({
            "code": 200,
            "success": msg,
        })


def error(msg="请求失败",code=400):
    """返回包含 error 消息的 JSON 信封，以 code 同时设置正文和 HTTP 状态。"""
    return jsonify({
        "code": code,
        "error": msg,
    }),code