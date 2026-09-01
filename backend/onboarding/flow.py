#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首次使用渐进式引导对话流程(§11.2)：确定性步骤机，不经过 LLM。

设计依据：docs/ARCHITECTURE.md §11.1/§11.2
决策依据：docs/DECISIONS.md D28(渐进式引导+主/次体质)、D30(city→timezone 只建议
不地理编码，显式确认才写入)

⚠️ **两处文档没有回答、这里做出的具体选择(照现有风格延伸，不是发明新架构)**：

1. **多轮 `/api/onboarding/*` 请求之间的状态怎么传**——ARCHITECTURE §10.1 给的
   `OnboardingAnswer{step_id, answer}` 没有 session/state 字段，这是文档本身留白，
   不是这次故意跳过。选择：服务端不新建一张 `onboarding_state` 表(§1.2 没有为此
   设计过表)，状态是一个不透明 dict，每次响应里带给客户端，客户端下一轮原样带
   回来——`api/main.py` 对应的路由会这样接。
2. **步骤1(过敏原)/步骤2(偏好)的自由文本抽取**——用比 §4.2 简单得多的规则(按
   常见分隔符切词+几个否定词判断"没有")，不是 §4.2 给"记录一顿饭"设计的菜品/
   配料拆解管线(`dish_decomposition.py` 仍是 `NotImplementedError`，套用不了，
   套用也没必要——这里要处理的是"列出过敏原"这种简单得多的任务)。

**CCMQ 问卷环节走确定性代码，不调用 LLM**——45 道题都是固定的 5 级量表选择题，
不需要生成式能力去理解自由文本(比 D25 说的"一次性情境信息"简单得多)；这也
是为什么这里没有引用 `backend/skills/ccmq_questionnaire.md` 那个 Skill(它是
给"引导对话本身由 LLM 驱动"这条从未真正实现的假设登记的元数据，当前这个确定性
实现不需要在某次 LLM 调用里加载它)——`.md` 文件本身仍然是给人看的题库/计分口径
文档，本模块的 `ITEM_TEXTS` 是同一份内容的可执行副本，两处保持一致，改一处要
同步改另一处(尚无自动化机制强制这一点,是已知的小份手工同步负担)。

**只有"确定性可判定"的自由文本才在这里处理**；无法从固定选项/简单规则里解析
出结果时(比如量表题回答了一句无关的话)，本模块返回原样的问题 + 报错提示，
不猜测、不用 LLM 兜底替用户做决定。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from backend.i18n import normalize_locale, t
from backend.onboarding.ccmq_scoring import CONSTITUTIONS, ITEMS_PER_CONSTITUTION, score_ccmq
from backend.onboarding.session_store import ActiveOnboarding, OnboardingSessionStore

# Keep in sync with backend/skills/ccmq_questionnaire.md (see module docstring).
# Stems must not use Likert option words (没有/很少/有时/经常/总是) — they collide
# with the answer scale. Prefer plain language; put TCM terms in parentheses.
ITEM_TEXTS: dict[str, tuple[str, ...]] = {
    "ping_he": (
        "您平时精力充沛吗？",
        "您睡觉踏实、夜里睡得安稳吗？",
        "您换季时身体适应得好、不容易感冒吗？",
        "您情绪比较平稳，遇事不容易烦或慌吗？",
        "您觉得自己身体状况总体不错吗？",
    ),
    "qi_xu": (
        "您容易疲乏，稍微活动就觉得喘气费劲(气短)吗？",
        "您说话声音偏小、不太想多说话吗？",
        "您容易出汗，稍微活动就出汗明显吗？",
        "您容易感冒，而且感冒后恢复得慢吗？",
        "您比别人更容易喘气费劲、感觉气接不上来吗？",
    ),
    "yang_xu": (
        "您手脚发凉，比别人更怕冷吗？",
        "您吃(喝)凉的东西会不舒服，或者容易拉肚子(腹泻)吗？",
        "您比别人更容易着凉、容易感冒吗？",
        "您提不起精神，做事缺少热情吗？",
        "您比一般人更怕冷，需要比别人穿更多衣服吗？",
    ),
    "yin_xu": (
        "您感觉手心、脚心发热吗？",
        "您皮肤或嘴唇容易干燥吗？",
        "您容易口干、嗓子发干、想喝水吗？",
        "您大便干燥，不容易排出吗？",
        "您容易失眠、晚上睡不安稳吗？",
    ),
    "tan_shi": (
        "您感觉身体沉重、不轻松吗？",
        "您觉得肚子摸上去软软的、偏胖(腹部肥满松软)吗？",
        "您额头或脸上出油比较多吗？",
        "您舌头上那层苔又厚又黏、嘴里不清爽(舌苔厚腻)吗？",
        "您皮肤油腻，出的汗黏黏的、不太干爽吗？",
    ),
    "shi_re": (
        "您脸上或鼻子感觉油油的、亮亮的吗？",
        "您容易长痘(痤疮)或皮肤上起疖子、疙瘩(疮疖)吗？",
        "您感觉嘴里发苦，或有异味吗？",
        "您大便黏、解不干净吗？",
        "您小便颜色发黄、气味比较重吗？",
    ),
    "xue_yu": (
        "您皮肤容易不明原因地出现淤青吗？",
        "您脸颊靠近鼻子两侧(两颧)有细细的红血丝吗？",
        "您身上某个地方会刺痛，而且总在同一个位置吗？",
        "您嘴唇颜色偏暗吗？",
        "您容易忘事、刚做过的事也记不住吗？",
    ),
    "qi_yu": (
        "您觉得闷闷不乐、情绪低落吗？",
        "您容易紧张、感到焦虑不安吗？",
        "您胸口两侧或肋骨附近(胸胁)、或者乳房有胀痛感吗？",
        "您容易无缘无故叹气吗？",
        "您嗓子眼总感觉有东西堵着吗？",
    ),
    "te_bing": (
        "不感冒的时候，也会打喷嚏、鼻塞、流鼻涕吗？",
        "您容易对药物、食物、气味或花粉等过敏吗？",
        "您的皮肤容易起风团、风疙瘩(荨麻疹)吗？",
        "您皮肤会因为过敏出现紫红色的小点或青紫斑吗？",
        "您的皮肤只要一抓就红，并留下抓痕吗？",
    ),
}

# English stems, same 9×5 order/scoring as ITEM_TEXTS. Avoid Likert option
# words (never/rarely/sometimes/often/always) so they do not collide with
# the 1-5 aliases. Executable copy lives here; ccmq_questionnaire.md is
# the human-readable twin.
ITEM_TEXTS_EN: dict[str, tuple[str, ...]] = {
    "ping_he": (
        "Do you usually feel energetic?",
        "Do you sleep soundly and stay asleep through the night?",
        "Do you adapt well to seasonal changes and catch colds infrequently?",
        "Is your mood fairly even, without getting easily annoyed or anxious?",
        "Do you feel your overall health is pretty good?",
    ),
    "qi_xu": (
        "Do you tire easily, and get short of breath after light activity (qi shortness)?",
        "Is your voice on the quiet side, and do you prefer not to talk much?",
        "Do you sweat easily, with noticeable sweating after light activity?",
        "Do you catch colds easily, and recover slowly afterward?",
        "Are you more prone than others to feeling short of breath, as if you cannot catch your breath?",
    ),
    "yang_xu": (
        "Are your hands and feet cold, and do you feel the cold more than others?",
        "Do cold foods or drinks make you uncomfortable, or do you tend to have loose stools (diarrhea)?",
        "Do you catch chills and colds more easily than others?",
        "Do you lack energy and enthusiasm for getting things done?",
        "Are you more sensitive to cold than most people, needing extra layers?",
    ),
    "yin_xu": (
        "Do your palms and soles feel warm?",
        "Is your skin or lips prone to dryness?",
        "Do you get dry mouth or throat and want to drink water?",
        "Is your stool dry and hard to pass?",
        "Do you have trouble sleeping or staying asleep at night?",
    ),
    "tan_shi": (
        "Does your body feel heavy and not light?",
        "Does your abdomen feel soft and a bit plump (soft, full belly)?",
        "Do your forehead or face get oily?",
        "Is the coating on your tongue thick and sticky, with a sticky feeling in the mouth (thick greasy tongue coating)?",
        "Is your skin oily, with sticky sweat that does not dry easily?",
    ),
    "shi_re": (
        "Does your face or nose feel oily and shiny?",
        "Are you prone to acne or boils/bumps on the skin?",
        "Do you notice a bitter taste or bad taste in your mouth?",
        "Is your stool sticky and hard to pass completely?",
        "Is your urine dark yellow with a strong odor?",
    ),
    "xue_yu": (
        "Do you bruise easily without a clear reason?",
        "Do you have fine red veins on the cheeks near the sides of the nose?",
        "Do you get a stabbing pain that always stays in the same spot?",
        "Are your lips on the darker side?",
        "Do you forget things easily, even things you just did?",
    ),
    "qi_yu": (
        "Do you feel down or low in mood?",
        "Do you get tense or anxious easily?",
        "Do you feel distending pain in the chest sides/ribs (flanks) or breasts?",
        "Do you sigh for no particular reason?",
        "Does your throat feel blocked, as if something is stuck there?",
    ),
    "te_bing": (
        "Even when you do not have a cold, do you sneeze, get a stuffy nose, or have a runny nose?",
        "Are you prone to allergies to medicine, food, odors, or pollen?",
        "Does your skin easily develop hives or welts (urticaria)?",
        "Does your skin show purplish spots from allergies?",
        "Does your skin turn red and leave marks as soon as you scratch it?",
    ),
}

# 45 道题铺平成固定顺序的 (体质, 题号) 列表，供分批呈现时按索引切片。
_FLAT_ITEMS: tuple[tuple[str, int], ...] = tuple(
    (constitution, idx) for constitution in CONSTITUTIONS for idx in range(ITEMS_PER_CONSTITUTION)
)
_TOTAL_ITEMS = len(_FLAT_ITEMS)
_CCMQ_BATCH_SIZE = 3  # ARCHITECTURE §11.2 步骤3b："每轮 2-3 题"

# User-facing answers are 1-5. Words stay as aliases (agreement + official frequency).
_LIKERT_WORDS = {
    "完全不像": 1, "完全不符合": 1, "根本不": 1, "没有": 1,
    "不太像": 2, "有一点像": 2, "有一点": 2, "很少": 2,
    "有些像": 3, "有些": 3, "有时": 3,
    "比较符合": 4, "相当符合": 4, "相当": 4, "经常": 4,
    "非常符合": 5, "完全符合": 5, "非常": 5, "总是": 5,
    "不确定": 3, "不知道": 3, "跳过": 3,  # midpoint, see ccmq_questionnaire.md
    # English aliases (longer phrases first; matched case-insensitively).
    "not at all": 1, "never": 1,
    "a little": 2, "slightly": 2, "rarely": 2,
    "somewhat": 3, "sometimes": 3, "unsure": 3, "unknown": 3,
    "don't know": 3, "dont know": 3, "skip": 3,
    "quite a bit": 4, "frequently": 4, "often": 4,
    "very much": 5, "completely": 5, "always": 5,
}

_FULLWIDTH_DIGITS = str.maketrans("１２３４５", "12345")


def _likert_scale_hint(locale: str) -> str:
    return t("onboarding.likert_scale", locale)


def _parse_likert(answer: str, locale: str = "zh") -> int:
    a = answer.strip().translate(_FULLWIDTH_DIGITS)
    if a.isdigit() and 1 <= int(a) <= 5:
        return int(a)
    lowered = a.lower()
    # Longer aliases first so "not at all" wins over a stray "not".
    for word, score in sorted(_LIKERT_WORDS.items(), key=lambda kv: -len(kv[0])):
        needle = word.lower() if word.isascii() else word
        haystack = lowered if word.isascii() else a
        if needle in haystack:
            return score
    raise ValueError(
        t(
            "onboarding.likert_unrecognized",
            locale,
            answer=answer,
            scale=_likert_scale_hint(locale),
        )
    )


_NEGATION_ONLY = re.compile(
    r"^(没有|无|不|没什么|没啥|无过敏|不过敏|none|no|nothing|nope|n/?a|no allergies)$",
    re.IGNORECASE,
)
_SPLIT_RE = re.compile(r"[，,、;；/和及\s]+")


def _extract_list(answer: str) -> list[str]:
    """比 §4.2 菜品拆解简单得多的确定性抽取：按常见分隔符切词，识别"没有"类
    否定整句。见模块文档顶部关于为什么不套用 §4.2 管线的说明。"""
    a = answer.strip()
    if not a or _NEGATION_ONLY.match(a):
        return []
    return [p for p in _SPLIT_RE.split(a) if p]


# D30：只建议，不做地理编码——常见城市给个候选，其余一律要求用户自己确认/输入，
# 不静默套用默认值。
_CITY_TZ_HINTS: dict[str, str] = {
    **{
        c: "Asia/Shanghai"
        for c in (
            "北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉",
            "西安", "重庆", "天津", "苏州", "香港", "澳门", "青岛", "厦门",
        )
    },
    "台北": "Asia/Taipei",
    "高雄": "Asia/Taipei",
    "东京": "Asia/Tokyo",
    "大阪": "Asia/Tokyo",
    "首尔": "Asia/Seoul",
    "纽约": "America/New_York",
    "洛杉矶": "America/Los_Angeles",
    "旧金山": "America/Los_Angeles",
    "西雅图": "America/Los_Angeles",
    "伦敦": "Europe/London",
    "巴黎": "Europe/Paris",
    "柏林": "Europe/Berlin",
    "悉尼": "Australia/Sydney",
    "新加坡": "Asia/Singapore",
    # English names (matched case-insensitively in _suggest_timezone).
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "guangzhou": "Asia/Shanghai",
    "shenzhen": "Asia/Shanghai",
    "hangzhou": "Asia/Shanghai",
    "nanjing": "Asia/Shanghai",
    "chengdu": "Asia/Shanghai",
    "wuhan": "Asia/Shanghai",
    "hong kong": "Asia/Shanghai",
    "taipei": "Asia/Taipei",
    "tokyo": "Asia/Tokyo",
    "osaka": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    "new york": "America/New_York",
    "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "sydney": "Australia/Sydney",
    "singapore": "Asia/Singapore",
}


def _suggest_timezone(city: str) -> str | None:
    lowered = city.casefold()
    for name, tz in _CITY_TZ_HINTS.items():
        needle = name.casefold() if name.isascii() else name
        haystack = lowered if name.isascii() else city
        if needle in haystack:
            return tz
    return None


@dataclass(frozen=True)
class OnboardingStep:
    step_id: str
    prompt: str
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OnboardingResult:
    step_id: str
    profile_updates: dict[str, Any]
    summary: str


def should_trigger(profile: Any | None) -> bool:
    """Ask until the intro has actually finished, including「全部跳过」.

    `POST /api/users` inserts a stub row so the switcher can list them; that
    is not "already onboarded". Empty constitution / skipped questions also
    must not hide the intro. The latch is `onboarding_done` (written on
    finish). Users can still call `POST /api/onboarding/start` themselves.
    """
    if profile is None:
        return True
    return not bool(getattr(profile, "onboarding_done", False))


# Per-question skip (continue to the next step with an empty value). Distinct
# from aborting the rest of onboarding — see `_ABORT_ALL` in `apply_answer`.
_STEP_SKIP = frozenset({"跳过", "先跳过", "skip"})
_ABORT_ALL = frozenset({"全部跳过", "先不用", "不用了", "skip all", "skipall"})
_TZ_SKIP = frozenset({"跳过", "不确定", "不知道", "skip", "unknown", "unsure", "don't know", "dont know"})
_CONFIRM_YES = frozenset({"对", "是", "确认", "没错", "yes", "confirm", "y"})
_CONFIRM_SKIP = frozenset({"跳过", "不准", "不太准", "算了", "skip"})
# D28 enum: finished the flow without a constitution (skip / abort).
_SOURCE_UNCONFIRMED = "unconfirmed"


@dataclass(frozen=True)
class ChatOnboardingTurn:
    """One `/api/chat` reply that is still inside the onboarding state machine."""

    prompt: str
    done: bool = False
    started: bool = False
    profile_updates: dict[str, Any] | None = None


def _in_alias(answer: str, aliases: frozenset[str]) -> bool:
    a = answer.strip()
    return a in aliases or a.lower() in aliases


def _item_texts_for(locale: str) -> dict[str, tuple[str, ...]]:
    return ITEM_TEXTS_EN if normalize_locale(locale) == "en" else ITEM_TEXTS


def advance_chat_onboarding(
    message: str,
    profile: Any | None,
    store: OnboardingSessionStore,
    *,
    user_id: str,
    locale: str = "zh",
) -> ChatOnboardingTurn | None:
    """Drive onboarding from a chat turn. None = this is normal chat, not onboarding.

    The first triggering message is not consumed as an answer (the user usually
    asked a real question). Later messages in the same first conversation are
    answers to the current step.
    """
    loc = normalize_locale(locale)
    active = store.get(user_id)
    if active is not None:
        result = apply_answer(active.step_id, message, active.state, locale=loc)
        if isinstance(result, OnboardingResult):
            store.clear(user_id)
            return ChatOnboardingTurn(
                prompt=t("onboarding.done_followup", loc, summary=result.summary),
                done=True,
                profile_updates=result.profile_updates,
            )
        store.put(user_id, ActiveOnboarding(step_id=result.step_id, state=result.state))
        return ChatOnboardingTurn(prompt=result.prompt)

    if not should_trigger(profile):
        return None

    step = start_onboarding(locale=loc)
    store.put(user_id, ActiveOnboarding(step_id=step.step_id, state=step.state))
    return ChatOnboardingTurn(prompt=step.prompt, started=True)


def start_onboarding(locale: str = "zh") -> OnboardingStep:
    loc = normalize_locale(locale)
    return OnboardingStep(
        step_id="allergens",
        prompt=t("onboarding.start", loc),
        state={},
    )


def _next_ccmq_batch_step(state: dict[str, Any], locale: str = "zh") -> OnboardingStep:
    loc = normalize_locale(locale)
    idx = state.get("ccmq_flat_index", 0)
    batch = _FLAT_ITEMS[idx : idx + _CCMQ_BATCH_SIZE]
    texts = _item_texts_for(loc)
    lines = [f"{i + 1}. {texts[c][item_idx]}" for i, (c, item_idx) in enumerate(batch)]
    prompt = t(
        "onboarding.ccmq_intro",
        loc,
        scale=_likert_scale_hint(loc),
        start=idx + 1,
        end=idx + len(batch),
        total=_TOTAL_ITEMS,
        items="\n".join(lines),
    )
    return OnboardingStep(step_id="ccmq_batch", prompt=prompt, state=state)


def apply_answer(
    step_id: str, answer: str, state: dict[str, Any], locale: str = "zh"
) -> OnboardingStep | OnboardingResult:
    """按当前 `step_id` 解释 `answer`，返回下一步(`OnboardingStep`)或最终结果
    (`OnboardingResult`，`step_id="done"`)。`state` 是上一次响应原样带回的值，
    调用方(api/main.py)不需要、也不应该解读它的内部结构。"""

    loc = normalize_locale(locale)
    a = answer.strip()
    if _in_alias(a, _ABORT_ALL):
        return _finish_skipped(state, locale=loc)

    if step_id == "allergens":
        allergens = [] if _in_alias(a, _STEP_SKIP) else _extract_list(answer)
        new_state = {**state, "allergens": allergens}
        return OnboardingStep(
            step_id="preferences",
            prompt=t("onboarding.preferences", loc),
            state=new_state,
        )

    if step_id == "preferences":
        notes = answer.strip()
        preferences = (
            {}
            if _in_alias(notes, _STEP_SKIP) or _NEGATION_ONLY.match(notes or "")
            else {"notes": notes}
        )
        new_state = {**state, "preferences": preferences}
        return OnboardingStep(
            step_id="city",
            prompt=t("onboarding.city", loc),
            state=new_state,
        )

    if step_id == "city":
        city = "" if _in_alias(a, _STEP_SKIP) else answer.strip()
        suggested = _suggest_timezone(city) if city else None
        new_state = {**state, "city": city or None}
        if suggested:
            return OnboardingStep(
                step_id="timezone_confirm",
                prompt=t("onboarding.tz_confirm_known", loc, tz=suggested),
                state={**new_state, "timezone_suggested": suggested},
            )
        return OnboardingStep(
            step_id="timezone_confirm",
            prompt=t("onboarding.tz_confirm_unknown", loc),
            state={**new_state, "timezone_suggested": None},
        )

    if step_id == "timezone_confirm":
        a = answer.strip()
        suggested = state.get("timezone_suggested")
        if _in_alias(a, _TZ_SKIP):
            timezone = None
        elif _in_alias(a, _CONFIRM_YES) and suggested:
            timezone = suggested
        else:
            resuggested = _suggest_timezone(a)
            timezone = resuggested or (a if "/" in a else None)
        new_state = {**state, "timezone": timezone}
        return OnboardingStep(
            step_id="constitution_known",
            prompt=t("onboarding.constitution_known", loc),
            state=new_state,
        )

    if step_id == "constitution_known":
        guess = answer.strip()
        if _in_alias(guess, _STEP_SKIP):
            return _finish_skipped(state, locale=loc)
        matched = _match_constitution(guess)
        if matched:
            return _finish(
                state,
                constitution=matched,
                constitution_secondary=[],
                constitution_source="self_reported",
                locale=loc,
            )
        new_state = {**state, "ccmq_flat_index": 0, "ccmq_answers": {c: [] for c in CONSTITUTIONS}}
        return _next_ccmq_batch_step(new_state, locale=loc)

    if step_id == "ccmq_batch":
        if _in_alias(a, _STEP_SKIP):
            return _finish_skipped(state, locale=loc)
        idx = state["ccmq_flat_index"]
        batch = _FLAT_ITEMS[idx : idx + _CCMQ_BATCH_SIZE]
        raw_scores = [s for s in re.split(r"[，,、;；\s]+", answer.strip()) if s]
        if len(raw_scores) != len(batch):
            return _next_ccmq_batch_step(state, locale=loc)
        try:
            scores = [_parse_likert(s, locale=loc) for s in raw_scores]
        except ValueError:
            return _next_ccmq_batch_step(state, locale=loc)
        ccmq_answers = {c: list(v) for c, v in state["ccmq_answers"].items()}
        for (constitution, item_idx), score in zip(batch, scores):
            ccmq_answers[constitution].append(score)
        new_idx = idx + len(batch)
        new_state = {**state, "ccmq_flat_index": new_idx, "ccmq_answers": ccmq_answers}
        if new_idx < _TOTAL_ITEMS:
            return _next_ccmq_batch_step(new_state, locale=loc)

        result = score_ccmq(ccmq_answers)
        primary = result.primary
        secondary = list(result.secondary)
        verdicts = (
            {"是": "yes", "基本是": "basically yes", "倾向是": "leaning yes", "否": "no"}
            if loc == "en"
            else {}
        )

        def display_verdict(constitution: str) -> str:
            verdict = result.scores[constitution].verdict
            return verdicts.get(verdict, verdict)

        summary_lines = (
            [f"{_display_name(primary, loc)} ({display_verdict(primary)})"] if primary else []
        )
        summary_lines += [
            f"{_display_name(c, loc)} ({display_verdict(c)})" for c in secondary
        ]
        summary = (", " if loc == "en" else "、").join(summary_lines) if summary_lines else t("onboarding.ccmq_no_bias", loc)
        return OnboardingStep(
            step_id="constitution_confirm",
            prompt=t("onboarding.ccmq_confirm", loc, summary=summary),
            state={**new_state, "ccmq_primary": primary, "ccmq_secondary": secondary},
        )

    if step_id == "constitution_confirm":
        a = answer.strip()
        if _in_alias(a, _CONFIRM_SKIP):
            return _finish_skipped(state, locale=loc)
        if _in_alias(a, _CONFIRM_YES):
            return _finish(
                state,
                constitution=state.get("ccmq_primary"),
                constitution_secondary=state.get("ccmq_secondary") or [],
                constitution_source="ccmq_computed",
                locale=loc,
            )
        matched = _match_constitution(a)
        return _finish(
            state,
            constitution=matched,
            constitution_secondary=[],
            constitution_source="self_reported" if matched else _SOURCE_UNCONFIRMED,
            locale=loc,
        )

    raise ValueError(f"unknown onboarding step_id: {step_id!r}")


_ZH_NAMES = {
    "ping_he": "平和质", "qi_xu": "气虚质", "yang_xu": "阳虚质", "yin_xu": "阴虚质",
    "tan_shi": "痰湿质", "shi_re": "湿热质", "xue_yu": "血瘀质", "qi_yu": "气郁质",
    "te_bing": "特禀质",
}

_EN_NAMES = {
    "ping_he": "Balanced",
    "qi_xu": "Qi deficiency",
    "yang_xu": "Yang deficiency",
    "yin_xu": "Yin deficiency",
    "tan_shi": "Phlegm-dampness",
    "shi_re": "Damp-heat",
    "xue_yu": "Blood stasis",
    "qi_yu": "Qi stagnation",
    "te_bing": "Special constitution",
}


def _zh_name(constitution: str | None) -> str:
    return _ZH_NAMES.get(constitution, "") if constitution else ""


def _display_name(constitution: str | None, locale: str = "zh") -> str:
    if not constitution:
        return ""
    if normalize_locale(locale) == "en":
        return _EN_NAMES.get(constitution, constitution)
    return _ZH_NAMES.get(constitution, constitution)


def _match_constitution(guess: str) -> str | None:
    """Match a free-text answer to a CCMQ id (zh name, en name, or slug)."""
    g = guess.strip()
    if not g:
        return None
    gl = g.lower()
    for c in CONSTITUTIONS:
        if c in gl or c.replace("_", " ") in gl:
            return c
        zh = _ZH_NAMES.get(c, "")
        if zh and zh in g:
            return c
        en = _EN_NAMES.get(c, "")
        if en and en.lower() in gl:
            return c
    return None


def _finish_skipped(state: dict[str, Any], locale: str = "zh") -> OnboardingResult:
    return _finish(
        state,
        constitution=None,
        constitution_secondary=[],
        constitution_source=_SOURCE_UNCONFIRMED,
        locale=locale,
    )


def _finish(
    state: dict[str, Any],
    *,
    constitution: str | None,
    constitution_secondary: list[str],
    constitution_source: str | None,
    locale: str = "zh",
) -> OnboardingResult:
    loc = normalize_locale(locale)
    updates: dict[str, Any] = {
        "allergens": state.get("allergens", []),
        "preferences": state.get("preferences", {}),
        "city": state.get("city"),
        "timezone": state.get("timezone"),
        "constitution": constitution,
        "constitution_secondary": constitution_secondary,
        "constitution_source": constitution_source,
        "onboarding_done": True,
        "locale": loc,
    }
    separator = ", " if loc == "en" else "、"
    summary = t("onboarding.finish_prefix", loc) + separator.join(
        f"{k}={v}"
        for k, v in updates.items()
        if k != "onboarding_done" and v not in (None, [], {})
    )
    return OnboardingResult(step_id="done", profile_updates=updates, summary=summary)
