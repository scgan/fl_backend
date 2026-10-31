"""提供统一 API 响应信封、业务异常及其 FastAPI 异常处理器。"""

from __future__ import annotations

from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def envelope(
    *,
    request: Request,
    data: Any,
    code: str = "OK",
    message: str = "success",
) -> dict[str, Any]:
    """构造规范要求的统一 API 响应结构。

    Args:
        request: 包含请求追踪标识的 FastAPI 请求。
        data: 成功数据或错误详情。
        code: 稳定的机器可读业务码。
        message: 面向调用方的结果说明。

    Returns:
        包含 ``code``、``message``、``data`` 和 ``request_id`` 的字典。
    """
    return {
        "code": code,
        "message": message,
        "data": data,
        "request_id": request.state.request_id,
    }


class AppError(Exception):
    """可被统一处理器转换为标准响应的业务异常。"""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        """保存稳定错误码、用户提示和 HTTP 状态码。

        Args:
            code: 稳定的机器可读错误码。
            message: 可返回客户端的错误说明。
            status_code: 对应的 HTTP 状态码。
        """
        # 供客户端分支处理和审计检索的稳定业务错误码。
        self.code = code
        # 已脱敏、允许返回客户端的错误说明。
        self.message = message
        # 统一异常处理器应返回的 HTTP 状态码。
        self.status_code = status_code
        super().__init__(message)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """将业务异常转换为统一错误信封。

    Args:
        request: 当前请求。
        exc: 路由或服务层抛出的业务异常。

    Returns:
        使用异常状态码的 JSON 响应。
    """
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope(request=request, data=None, code=exc.code, message=exc.message),
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """将 Pydantic/FastAPI 参数错误转换为统一校验错误。

    Args:
        request: 当前请求。
        exc: FastAPI 收集的请求参数校验异常。

    Returns:
        HTTP 422 标准错误响应。
    """
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=envelope(
            request=request,
            # Starlette/FastAPI 新版本的 ValidationException 不再接受 Pydantic
            # 专有关键字；使用跨版本稳定接口，确保 422 不会被二次转换成 500。
            data={"errors": exc.errors()},
            code="VALIDATION_ERROR",
            message="请求参数校验失败",
        ),
    )
