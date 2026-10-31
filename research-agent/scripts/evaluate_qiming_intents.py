"""用真实兼容模型评测启明当前意图规则与人工标注问法。

从环境读取模型连接配置，逐条调用与生产相同的结构化网关和分类指令；
只输出用例 ID、预期与实际标签，不输出问题全文或模型响应。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.agents.catalog import QIMING_SYSTEM_PROMPT
from app.agents.gateway import get_model_gateway
from app.agents.qiming_graph import build_intent_instruction
from app.agents.schemas import QimingIntentClassification
from app.common.config import get_settings

GOLD_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "qiming_intent_gold.json"


async def evaluate() -> int:
    """用当前真实模型逐条分类，并返回不匹配或失败的用例数。

    Returns:
        分类不匹配及调用失败的总数。

    Raises:
        RuntimeError: 环境缺少真实兼容模型的必要配置。
    """
    settings = get_settings()
    if not (settings.openai_api_key and settings.openai_base_url and settings.model_name):
        raise RuntimeError("需配置 OPENAI_API_KEY、OPENAI_BASE_URL 和 MODEL_NAME")
    gateway = get_model_gateway(
        "OPENAI_COMPATIBLE",
        model_name=settings.model_name,
        api_base=settings.openai_base_url,
        temperature=0.0,
    )
    cases = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    failures = 0
    for case in cases:
        try:
            result = await gateway.structured(
                instruction=build_intent_instruction(case["query"]),
                schema=QimingIntentClassification,
                system_prompt=QIMING_SYSTEM_PROMPT,
                thinking=False,
            )
            actual = result.intent
        except Exception as exc:
            # 只打印异常类型，避免供应商错误正文把用户问题或密钥带入日志。
            actual = f"ERROR:{type(exc).__name__}"
        passed = actual == case["intent"]
        failures += not passed
        print(
            f"{case['id']} {'PASS' if passed else 'FAIL'} "
            f"expected={case['intent']} actual={actual}"
        )
    print(f"total={len(cases)} passed={len(cases) - failures} failed={failures}")
    return failures


def main() -> None:
    """运行异步评测，并以退出码表示是否全部命中。"""
    raise SystemExit(1 if asyncio.run(evaluate()) else 0)


if __name__ == "__main__":
    main()
