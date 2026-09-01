#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User-facing locale helpers.

Locale follows the user (ChatRequest.locale / onboarding locale), default ``zh``.
Do not infer from Accept-Language. Knowledge-base chunks, conflict rules, and
most skill markdown stay in their original language; this module only covers
hardcoded user-visible copy and the one LLM language instruction.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Any

SUPPORTED_LOCALES = ("zh", "en")
DEFAULT_LOCALE = "zh"

_locale_var: ContextVar[str] = ContextVar("diet_expert_locale", default=DEFAULT_LOCALE)

LANGUAGE_INSTRUCTION_EN = (
    "User-facing language: English. Write all user-visible text in English. "
    "Retrieved chunks may be Chinese; translate conclusions, keep [source: chunk_id] unchanged."
)

# Omitted on zh to avoid prompt drift vs the current Chinese-only behavior.
LANGUAGE_INSTRUCTION_ZH = ""

_MQE_CHINESE_REWRITE_NOTE = (
    "The knowledge base is in Chinese. If the user question is in English, "
    "rewrite retrieval queries in Chinese so they match Chinese chunks. "
    "Keep [source: chunk_id] style ids out of the rewrite; output search phrases only."
)


def normalize_locale(value: str | None) -> str:
    """Return ``zh`` or ``en``. Anything else (including None/blank) is ``zh``."""
    if not value:
        return DEFAULT_LOCALE
    loc = str(value).strip().lower()
    if loc in SUPPORTED_LOCALES:
        return loc
    return DEFAULT_LOCALE


def current_locale() -> str:
    return _locale_var.get()


def set_current_locale(locale: str | None) -> None:
    """Bind locale for this request so tools (weather, MQE) can read it."""
    _locale_var.set(normalize_locale(locale))


def language_instruction(locale: str | None = None) -> str:
    """One user-facing instruction for complete() system prompts. Empty on zh."""
    loc = normalize_locale(locale if locale is not None else current_locale())
    if loc == "en":
        return LANGUAGE_INSTRUCTION_EN
    return LANGUAGE_INSTRUCTION_ZH


def apply_language_instruction(system_prompt: str, locale: str | None = None) -> str:
    inst = language_instruction(locale)
    if not inst:
        return system_prompt
    return f"{system_prompt}\n\n{inst}"


def mqe_rewrite_instruction() -> str:
    """Always-on MQE hint: English questions must still retrieve Chinese chunks."""
    return _MQE_CHINESE_REWRITE_NOTE


def t(key: str, locale: str | None = None, **kwargs: Any) -> str:
    """Look up a user-visible string. Missing keys fall back to zh, then the key."""
    loc = normalize_locale(locale if locale is not None else current_locale())
    table = _MESSAGES.get(loc) or _MESSAGES[DEFAULT_LOCALE]
    template = table.get(key) or _MESSAGES[DEFAULT_LOCALE].get(key) or key
    if kwargs:
        return template.format(**kwargs)
    return template


_MEAL_TYPE_KEYS = {
    "早餐": "meal.breakfast",
    "午餐": "meal.lunch",
    "晚餐": "meal.dinner",
    "夜宵": "meal.late_night",
    "下午茶": "meal.afternoon_tea",
    "加餐": "meal.snack",
    "未知": "meal.unknown",
}


def meal_type_label(meal_type: str | None, locale: str | None = None) -> str:
    """Translate an internal meal-type value without changing stored data."""
    value = str(meal_type or "未知")
    return t(_MEAL_TYPE_KEYS.get(value, "meal.unknown"), locale)


# ---------------------------------------------------------------------------
# Hardcoded user-visible copy. Keys are stable ids; comments stay English.
# ---------------------------------------------------------------------------

_MESSAGES: dict[str, dict[str, str]] = {
    "zh": {
        "onboarding.start": (
            "我会先了解一些基本情况(过敏原/体质/口味/所在城市)，几个问题，"
            "随时可以说「跳过」跳过当前问题，或者说「全部跳过」结束引导，"
            "不影响基础功能。\n\n"
            "首先，您对什么食物过敏或者有忌口吗？没有的话可以直接说「没有」。"
        ),
        "onboarding.preferences": (
            "了解了。您在饮食口味或用餐场景上有什么偏好或限制吗？"
            "(比如「不吃香菜」「能吃辣」「办公室没法热饭」，没有可以说「没有」)"
        ),
        "onboarding.city": "您现在在哪个城市/地区？这会用来给天气相关的建议和判断饮食记录的时间。",
        "onboarding.tz_confirm_known": (
            "看起来您在 {tz} 时区，对吗？"
            "(直接回复「对」确认，或者回复正确的时区/城市名，也可以说「跳过」)"
        ),
        "onboarding.tz_confirm_unknown": (
            "没能识别出对应的时区，方便告诉我一个 IANA 时区名吗(比如 Asia/Shanghai)？"
            "不确定的话可以说「跳过」。"
        ),
        "onboarding.constitution_known": (
            "最后是体质。您知道自己的中医体质类型吗？"
            "(知道的话直接告诉我是哪一类；不知道/不确定可以说「不知道」，我们做个小问卷)"
        ),
        "onboarding.ccmq_intro": (
            "接下来几道小题。每题打 1-5 的数字（{scale}），"
            "不确定就打 3。用逗号分隔，例如 4,2,5"
            "（第 {start}-{end} 题，共 {total} 题）：\n{items}"
        ),
        "onboarding.ccmq_confirm": (
            "根据您的回答，体质判定结果是：{summary}。"
            "回复「确认」采用这个结果，或者告诉我正确的体质，也可以说「跳过」不使用这次结果。"
        ),
        "onboarding.ccmq_no_bias": "未见明显偏颇体质倾向",
        "onboarding.done_followup": "{summary}\n之后可以直接继续提问。",
        "onboarding.finish_prefix": "首次引导完成：",
        "onboarding.likert_unrecognized": (
            "无法识别的量表回答：{answer!r}，请打 1-5 的数字（{scale}）"
        ),
        "onboarding.likert_scale": "1完全不像 2不太像 3有些像 4比较符合 5非常符合",
        "dispatch.clarification_unresolved": "这次仍然信息不足，没有得到有效回答，建议换个方式重新描述",
        "dispatch.verification_rejected": "回答未通过核查，已拦截",
        "dispatch.subagent_timeout": "这一侧分析超时，未能给出结论",
        "dispatch.both_subagents_failed": "两侧分析均失败",
        "dispatch.partial_failure_tcm": "中医侧分析失败，仅展示营养学侧结论",
        "dispatch.partial_failure_nutrition": "营养学侧分析失败，仅展示中医侧结论",
        "dispatch.verification_fallback": (
            "抱歉，这次没能给出有可靠依据支持的具体建议。可以换个问法，"
            "或者补充更具体的信息，我再试一次。"
        ),
        "dispatch.allergen_fallback": (
            "安全提示：上面的内容提到了你标记的过敏原（{allergens}）或其潜在来源，"
            "可能只是描述或举例，不一定是在推荐你食用。请自行核对配料表和过敏原标识"
            "后再判断；如果成分不明或无法确认，请先不要尝试，必要时咨询医生或过敏专科人员。"
        ),
        "dispatch.unknown_allergen": "相关过敏原",
        "dispatch.model_knowledge_unverified": (
            "模型通用知识，未经过当前知识库核验，可能不完全准确。"
        ),
        "dispatch.rejected_item": "建议未通过核查，已移除（检查项 {check_number}）",
        # 2026-09-01：过程可见性——`stage` SSE 事件的文案(backend/agents/sse.py
        # 没有"模型级流式"，这几条 detail 是在派发/调和/核查真正跑的时候即时
        # 吐出来的，填补"路由完成到第一条 token 之间用户干等"的空白，见
        # backend/agents/dispatch.py `_stage_event()`。同一条文案在 start/done
        # 两种状态下复用，前端靠事件里的 `status` 字段区分，不需要"开始中/已
        # 完成"两套措辞。
        "dispatch.stage_routing": "已确定处理方式",
        "dispatch.stage_subagent_tcm": "中医侧分析",
        "dispatch.stage_subagent_nutrition": "营养学侧分析",
        "dispatch.stage_reconcile": "两侧结论调和",
        "dispatch.stage_verify": "核查",
        "log_write.decompose_failed": "菜品拆解失败，请重试",
        "log_write.not_recognized": "还是没能识别出具体的食物，这次先不记录了",
        "log_write.clarification": "没能识别出具体吃了什么，能再具体说一下吗？比如吃了什么菜、喝了什么？",
        "log_write.allergen_warning": "注意：这条记录含有你标记过敏的「{allergen}」",
        "log_write.write_failed": "记录写入失败，请重试",
        "log_write.already_recorded": "这条记录之前已经记过了，没有重复写入",
        "log_write.recorded": "已记录",
        "log_write.unknown_dish": "（未能识别出具体菜品）",
        "log_write.unknown_ingredient": "未知食材",
        "log_write.llm_note": "（模型推测，仅供参考，未在知识库/个人记录中找到）",
        "log_review.empty": "没有找到「{time_range}」的饮食记录。",
        "log_review.header": "「{time_range}」的饮食记录，共 {count} 条：",
        "api.chain_timeout": "请求超时，已停止后续分析",
        "api.internal_error": "处理失败，请重试",
        "api.medical_intent_detail": "涉及疾病/用药咨询，转入受限模式：仅通用信息 + 免责声明",
        "api.medical_disclaimer": (
            "你提到的情况涉及具体疾病/用药，这超出了本工具的范围。\n"
            "我可以提供通用的膳食参考，但不能替代主治医生的建议，"
            "尤其是在药物与食物可能存在相互作用的情况下。"
        ),
        "api.create_user_failed": "创建用户失败，请重试",
        "pending.allergen": "过敏原「{names}」",
        "pending.supplement": "补剂「{names}」",
        "pending.generic": "一条关键事实",
        "pending.detail": "检测到{joined}。确认后才会写入画像；本轮建议尚未使用这条信息。",
        "api.profile_unconfirmed": "未确认的修改不会被写入（PRD §10.2 人在环）",
        "api.profile_field_unsupported": "不支持的字段：{field}，可选：{fields}",
        "api.pending_not_found": "未找到待确认的关键事实",
        "meal.breakfast": "早餐",
        "meal.lunch": "午餐",
        "meal.dinner": "晚餐",
        "meal.late_night": "夜宵",
        "meal.afternoon_tea": "下午茶",
        "meal.snack": "加餐",
        "meal.unknown": "未知",
    },
    "en": {
        "onboarding.start": (
            "I'll ask a few questions first (allergens / constitution / taste / city). "
            "You can say \"skip\" to skip the current question, or \"skip all\" to end "
            "onboarding — basic features still work.\n\n"
            "First, do you have any food allergies or foods you avoid? If none, just say \"none\"."
        ),
        "onboarding.preferences": (
            "Got it. Any taste preferences or meal-context limits? "
            "(e.g. \"no cilantro\", \"ok with spicy food\", \"can't reheat lunch at the office\". "
            "If none, say \"none\".)"
        ),
        "onboarding.city": (
            "Which city/region are you in? This is used for weather-aware advice "
            "and for dating diet-log entries."
        ),
        "onboarding.tz_confirm_known": (
            "Looks like you're in the {tz} timezone — is that right? "
            "(Reply \"yes\" to confirm, or send the correct timezone/city, or say \"skip\".)"
        ),
        "onboarding.tz_confirm_unknown": (
            "I couldn't map that to a timezone. Could you send an IANA timezone "
            "(e.g. Asia/Shanghai)? If you're not sure, say \"skip\"."
        ),
        "onboarding.constitution_known": (
            "Last, constitution. Do you know your TCM constitution type? "
            "(If yes, tell me which one; if not / unsure, say \"unknown\" and we'll do a short questionnaire.)"
        ),
        "onboarding.ccmq_intro": (
            "A few short items. Score each 1-5 ({scale}); "
            "if unsure, use 3. Separate with commas, e.g. 4,2,5 "
            "(items {start}-{end} of {total}):\n{items}"
        ),
        "onboarding.ccmq_confirm": (
            "Based on your answers, the constitution result is: {summary}. "
            "Reply \"confirm\" to use this, tell me the correct type, or say \"skip\" to discard it."
        ),
        "onboarding.ccmq_no_bias": "No clear biased-constitution tendency",
        "onboarding.done_followup": "{summary}\nYou can keep asking questions as usual.",
        "onboarding.finish_prefix": "Onboarding complete: ",
        "onboarding.likert_unrecognized": (
            "Unrecognized scale answer: {answer!r}. Please reply with a number 1-5 ({scale})"
        ),
        "onboarding.likert_scale": "1 not at all  2 slightly  3 somewhat  4 quite a bit  5 very much",
        "dispatch.clarification_unresolved": (
            "Still not enough information for a useful answer — try describing it another way"
        ),
        "dispatch.verification_rejected": "The answer did not pass verification and was blocked",
        "dispatch.subagent_timeout": "That side of the analysis timed out and could not finish",
        "dispatch.both_subagents_failed": "Both analyses failed",
        "dispatch.partial_failure_tcm": "TCM analysis failed; showing the nutrition conclusion only",
        "dispatch.partial_failure_nutrition": "Nutrition analysis failed; showing the TCM conclusion only",
        "dispatch.verification_fallback": (
            "Sorry, I could not produce a specific recommendation with reliable support this time. "
            "Try asking another way or add more detail, and I will try again."
        ),
        "dispatch.allergen_fallback": (
            "Safety notice: the text above mentions an allergen you marked "
            "({allergens}) or a possible hidden source. It may be descriptive or "
            "illustrative, not necessarily a recommendation to eat it. Check the "
            "product's ingredient list and allergen statement before deciding. If "
            "the ingredients are unclear, do not try it until you can confirm them; "
            "consult a clinician or allergy specialist if needed."
        ),
        "dispatch.unknown_allergen": "the relevant allergen",
        "dispatch.model_knowledge_unverified": (
            "General model knowledge; not verified against the current knowledge base "
            "and may be inaccurate."
        ),
        "dispatch.rejected_item": "A suggestion was removed during verification (check {check_number}).",
        "dispatch.stage_routing": "Routing decided",
        "dispatch.stage_subagent_tcm": "TCM-side analysis",
        "dispatch.stage_subagent_nutrition": "Nutrition-side analysis",
        "dispatch.stage_reconcile": "Reconciling both sides",
        "dispatch.stage_verify": "Verifying",
        "log_write.decompose_failed": "Could not parse the dish; please try again",
        "log_write.not_recognized": "Still could not recognize a specific food, so this was not logged",
        "log_write.clarification": (
            "I couldn't tell what you ate — could you be more specific? For example which dish or drink?"
        ),
        "log_write.allergen_warning": "Note: this entry contains an allergen you marked: 「{allergen}」",
        "log_write.write_failed": "Failed to save the log; please try again",
        "log_write.already_recorded": "This entry was already logged; not written again",
        "log_write.recorded": "Logged",
        "log_write.unknown_dish": "(could not recognize a specific dish)",
        "log_write.unknown_ingredient": "unknown ingredients",
        "log_write.llm_note": "(model guess, for reference only; not found in the knowledge base / personal log)",
        "log_review.empty": "No diet-log entries found for 「{time_range}」.",
        "log_review.header": "Diet log for 「{time_range}」, {count} entries:",
        "api.chain_timeout": "Request timed out; remaining analysis was stopped",
        "api.internal_error": "Something went wrong; please try again",
        "api.medical_intent_detail": (
            "This looks like a disease/medication question. Switching to restricted mode: "
            "general information + disclaimer only"
        ),
        "api.medical_disclaimer": (
            "What you described involves a specific illness or medication, which is outside this tool's scope.\n"
            "I can share general dietary information, but it is not a substitute for your clinician's advice, "
            "especially when food–drug interactions may be involved."
        ),
        "api.create_user_failed": "Failed to create user; please try again",
        "pending.allergen": "allergen(s) \"{names}\"",
        "pending.supplement": "supplement(s) \"{names}\"",
        "pending.generic": "a critical fact",
        "pending.detail": (
            "Detected {joined}. It will be written to your profile only after you confirm; "
            "this turn's advice has not used it yet."
        ),
        "api.profile_unconfirmed": "Unconfirmed changes are not written (human review is required).",
        "api.profile_field_unsupported": "Unsupported field: {field}. Available: {fields}",
        "api.pending_not_found": "Pending critical fact not found",
        "meal.breakfast": "breakfast",
        "meal.lunch": "lunch",
        "meal.dinner": "dinner",
        "meal.late_night": "late-night snack",
        "meal.afternoon_tea": "afternoon tea",
        "meal.snack": "snack",
        "meal.unknown": "unspecified meal",
    },
}
