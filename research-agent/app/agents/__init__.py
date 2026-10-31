"""汇集启明与各专业 Agent 的领域实现。

本包保留 Prompt 版本、LangGraph 状态和节点编排、检索契约及模型网关。
Run/Step/Evidence 持久化和 API 投影由 foreign_integration 负责，
图只通过受控工具获取证据，保持授权与 Evidence First 边界。
"""
