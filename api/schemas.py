#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pydantic 请求/响应模型。

设计依据：docs/ARCHITECTURE.md §10.1

阶段4任务11("FastAPI 最小闭环")当时只做了 `/api/chat`(`ChatRequest`)——`/api/profile`、
`/api/onboarding/*` 依赖的首次引导流程(§11)/CCMQ 计分/用户画像读写当时都还没实现，
装一个只有形状没有实现的 schema 容易让人以为"这个端点已经有了"，所以当时明确不做。

2026-08-26 补：CCMQ 计分(`backend/onboarding/ccmq_scoring.py`)、引导对话步骤
(`backend/onboarding/flow.py`)、`user_profile` 读写(`backend/agents/user_context.py`、
`write_memory` 的 `category="critical"` 分支)都已经补上，这三个模型跟着补齐。

`OnboardingAnswerRequest` 比 §10.1 原文的 `OnboardingAnswer{step_id, answer}` 多了
一个 `state` 字段——原文没有回答"多轮 `/api/onboarding/answer` 请求之间的状态怎么
传"这个问题(没有 session/state 字段，也没有为此设计过数据库表)，`state` 是
`backend/onboarding/flow.py` 返回、要求客户端原样带回的不透明 dict，服务端本身
不维护会话态，见该模块文档顶部的说明。
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# 2026-08-30 补：真实多用户支持——`user_id` 有默认值(不是必填)，旧客户端/
# 现有单测不传这个字段时仍然落到 `"default_user"`，不是破坏性变更。真实值
# 由前端的用户切换器决定，见 backend/agents/user_context.py `list_users()`/
# `create_user()`；后端从这里(而不是从 session_id)判断"这是哪个用户"——
# session_id 只是压缩/归档的记账单位，同一个用户可以有很多个 session_id。
_DEFAULT_USER_ID = "default_user"


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    user_id: str = Field(default=_DEFAULT_USER_ID, min_length=1)
    # Follows the user (frontend toggle), not Accept-Language. Default zh so
    # existing clients/tests keep Chinese copy and prompts.
    locale: Literal["zh", "en"] = "zh"

    @field_validator("locale", mode="before")
    @classmethod
    def _normalize_locale(cls, value: object) -> object:
        if value is None or value == "":
            return "zh"
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("zh", "en"):
                return lowered
        return value


class OnboardingAnswerRequest(BaseModel):
    step_id: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    state: dict[str, Any] = Field(default_factory=dict)
    user_id: str = Field(default=_DEFAULT_USER_ID, min_length=1)
    locale: Literal["zh", "en"] = "zh"

    @field_validator("locale", mode="before")
    @classmethod
    def _normalize_locale(cls, value: object) -> object:
        if value is None or value == "":
            return "zh"
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("zh", "en"):
                return lowered
        return value


class CriticalFactDecisionRequest(BaseModel):
    """Confirm or revoke a scanner hit that is waiting in pending_critical_facts."""

    pending_id: str = Field(..., min_length=1)


class ProfileUpdateRequest(BaseModel):
    """对应 §10.1 `ProfileUpdate{field, value, confirmed: true}`——`confirmed`
    没有默认值为 True：不带这个字段/带 false 都会被拒绝，因为通过这条端点直接
    改画像(不经过 §11.2 的对话式引导)没有"引导流程本身就是确认过程"这层前提，
    需要调用方显式声明"这是用户确认过的"(PRD §10.2 人在环)。"""

    field: str = Field(..., min_length=1)
    value: Any
    confirmed: bool
    user_id: str = Field(default=_DEFAULT_USER_ID, min_length=1)


class CreateUserRequest(BaseModel):
    """`POST /api/users`——前端"新用户"表单，纯展示名字，不是登录凭证。"""

    name: str = Field(..., min_length=1)
