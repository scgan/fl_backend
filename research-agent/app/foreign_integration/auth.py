"""校验 Foreign 已签发的 JWT，并从共用数据库重新加载用户。"""

from __future__ import annotations

import json
from dataclasses import dataclass

import jwt
from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.foreign_integration.config import get_foreign_settings
from app.foreign_integration.database import get_session
from app.foreign_integration.schema import foreign_user


@dataclass(frozen=True)
class ForeignActor:
    """记录本次请求中经过 JWT 与数据库双重校验的用户身份。"""

    id: int
    username: str
    role: int


def current_actor(
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> ForeignActor:
    """验证 Foreign Bearer Token、账号状态和当前选择的角色。

    Args:
        authorization: Foreign 前端当前使用的 Bearer Token。
        session: 共用数据库短事务会话。

    Returns:
        已认证且未删除的 Foreign 用户。

    Raises:
        HTTPException: Token 无效、账号不可用、角色不属于该用户或当前角色不是数据收集者。
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少 Foreign 登录凭证")
    try:
        payload = jwt.decode(
            authorization.removeprefix("Bearer ").strip(),
            get_foreign_settings().jwt_secret,
            algorithms=["HS256"],
        )
        username = payload["username"]
        role = int(payload["role"])
        if not isinstance(username, str) or role not in {0, 1, 2, 3}:
            raise ValueError("角色无效")
    except (jwt.PyJWTError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Foreign 登录凭证无效") from exc
    row = (
        session.execute(
            select(foreign_user).where(
                foreign_user.c.username == username,
                foreign_user.c.status == 1,
                foreign_user.c.is_deleted == 0,
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        raise HTTPException(status_code=401, detail="Foreign 用户已停用或不存在")
    try:
        roles = {int(value) for value in json.loads(row["role"] or "[]")}
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=403, detail="Foreign 用户角色无效") from exc
    if role not in roles:
        raise HTTPException(status_code=403, detail="当前角色无权使用此账号")
    if role != 2:
        raise HTTPException(status_code=403, detail="智能研究仅对数据收集者开放")
    return ForeignActor(id=int(row["id"]), username=username, role=role)
