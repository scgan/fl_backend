"""保存模型网关和智能体质量门槛；数据库与索引连接由 Foreign 配置负责。"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境加载四个智能体仍使用的模型与证据质量配置。"""

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )
    # 网关密钥只从环境读取，不进入研究运行记录。
    openai_api_key: SecretStr | None = None
    # 评估脚本使用的模型入口，正式 Run 固定数据库中的模型配置。
    openai_base_url: str | None = None
    model_name: str | None = None
    # 多节点研究默认关闭思考；调用者可按节点显式覆盖。
    model_thinking_enabled: bool = False
    # 主研究流程的单次 HTTP 请求等待秒数；生产启动脚本为 Max 提供更长预算。
    research_model_timeout_seconds: int = Field(default=60, ge=1)
    # 比较两侧分别需要达到的证据与来源文献下限。
    zhijian_min_side_evidence: int = 1
    zhijian_min_side_documents: int = 1


@lru_cache
def get_settings() -> Settings:
    """返回进程内缓存的智能体配置实例。"""
    return Settings()
