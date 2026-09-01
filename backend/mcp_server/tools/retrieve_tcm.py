#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
retrieve_tcm MCP 工具：查 knowledge_chunks WHERE domain='tcm' 的向量检索。
签名对齐 docs/ARCHITECTURE.md §2.2；混合检索(filters)的具体语义见
backend/mcp_server/tools/_retrieval_common.py 的模块文档。

调用方：TCM SubAgent(事实查询/候选评估/单领域/完整推荐分支均可能调用，§2.2)。

状态：检索逻辑本身可用；MCP 协议层的权限声明(§2.3，"TCM SubAgent 看得到这个
工具、Nutrition SubAgent 看不到")还没做，见 backend/mcp_server/server.py。

2026-08-30：默认打开 MQE + 混合检索(dense+sparse)，见
_retrieval_common.py 模块文档"检索评分方法优化"一节——这是"检索工具内部的
增强"，SubAgent 不需要知道、也看不到 `use_mqe`/`use_hybrid`/`complete` 这几
个参数（不在 MCP 工具的 JSON Schema 里，registry.py 只声明了 query/top_k/
filters）。
"""
from __future__ import annotations

from backend.llm.adapter import CompleteFn
from backend.mcp_server.tools._retrieval_common import (
    RetrievedChunk,
    search_knowledge_chunks,
)


def retrieve_tcm(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
    dsn: str | None = None,
    *,
    use_mqe: bool = True,
    use_hybrid: bool = True,
    complete: CompleteFn | None = None,
    locale: str | None = None,
) -> list[RetrievedChunk]:
    return search_knowledge_chunks(
        "tcm",
        query,
        top_k=top_k,
        filters=filters,
        dsn=dsn,
        use_mqe=use_mqe,
        use_hybrid=use_hybrid,
        complete=complete,
        locale=locale,
    )
