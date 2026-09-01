#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSE 事件格式化的两个底层工具函数——`backend/agents/{log_write,log_review,
dispatch}.py` 和 `api/main.py` 的顶层编排都要把最终文本变成 `event: ...\\n
data: ...\\n\\n` 这种格式，独立成一个没有其他依赖的小文件，避免这几个模块
互相 import 对方只为了借用这两个函数。

2026-08-28：从 api/main.py 拆出（原来的 `_sse_event`/`_chunk_text`）。放在
`backend/agents/` 而不是 `api/` 下：这两个函数被多个 `backend/agents/*.py`
模块直接调用（它们各自产出的就是 SSE 格式的字符串，不是先返回中立结果再由
api 层格式化——这是本来就有的设计,不是这次改的），放在 `api/` 下会让
`backend/` 反过来依赖 `api/`,方向反了。
"""
from __future__ import annotations

import json

# 切一份"已经算好的完整文本"成多块依次吐出——不是模型级流式，见 api/main.py
# 模块文档"诚实说明流式这两个字目前的真实程度"。
STREAM_CHUNK_SIZE = 40


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def chunk_text(text: str, size: int = STREAM_CHUNK_SIZE) -> list[str]:
    if not text:
        return [""]
    return [text[i : i + size] for i in range(0, len(text), size)]
