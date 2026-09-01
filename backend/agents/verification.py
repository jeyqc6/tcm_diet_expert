#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核查 pass：调和层之后、输出之前的独立验证（ARCHITECTURE §5.2 步骤 6 · D15）。

硬约束：
  - **第 1/4/6/8 项硬安全检查只拒绝，不改写**——缺 source_id、过敏原命中、
    ED 防护、诊断性表述这四类由确定性代码判定，绝不能「补上一个看起来像
    的 id」或悄悄改写成合规版本再放行（BUILD_PLAN 完成判据；PRD §10.1）。
  - **第 2/3/5 项软判定允许有限改写**（2026-08-31 起，见 `_parse_llm_verdicts`/
    `verification_checklist.md` Output format）——LLM 可以把条目标记为
    `annotate`：删掉引用支持不上的具体陈述，或给「有用但不确定」的内容加
    免责标注，但不能新增事实性论断、不能编造 `source_id`。当前策略是在初次
    硬检查通过后直接使用 annotate 文本；`_recheck_annotated_items` 保留为可选
    的防御性工具，但不由默认 verify 流程调用。加这条口子的原因：D15 原本
    "只拒绝不改写"在实测中导致过度拦截——check 2 命中率不低，多数情况下只是
    "引用位置张冠李戴"而不是"整条结论都是编的"，直接整条移除经常把真正有用的
    内容也一起扔掉，用户看到的是空回复而不是打了折扣的回复。
  - **不做规划、不调工具、不多轮**（D15）——至多一次 LLM 调用做软判定。
  - **Skill 按需拼入本次调用**，不常驻中枢 system prompt（D22 / §6.2）。
  - 确定性检查能做的先做（ENGINEERING §7.3）：source_id 有无/是否在本次
    available 集合里；幻觉引用同样移除。

分支差异（D25）：
  - 默认（完整推荐/单领域/事实查询）：每条建议必须有至少一个合法 source_id。
  - 候选评估：结论本身可不挂 id，但必须能拆出至少一条带合法 source_id 的支持理由。
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable, Literal

from backend.agents.citation import (
    extract_cited_ids,
    strip_invalid_citation_markers,
    validate_citations,
)
from backend.guardrails import ed_protection
from backend.guardrails.output_filters import check_allergens, check_diagnostic_statement
from backend.i18n import apply_language_instruction, t
from backend.llm import adapter as llm_adapter
from backend.llm.adapter import LLMResult
from backend.observability.tracing import observation, stage_log, update_current
from backend.skills.registry import compose_prompt_with_skills

logger = logging.getLogger("diet_expert.agents.verification")

CompleteFn = Callable[..., Awaitable[LLMResult]]

BranchMode = Literal[
    "full_recommend",
    "single_domain",
    "fact_query",
    "candidate_eval",
    "log_review",
]

_BASE_SYSTEM = (
    "你是核查 pass：独立验证饮食建议。第 1/4/6/8 项硬安全检查由确定性代码完成，"
    "你不参与、也不能推翻。对第 2/3/5 项的软判定，每条给出 accept/annotate/reject "
    "三选一（annotate=只能删减引用支持不上的具体陈述、或给不确定但有用的内容加"
    "免责标注，不能新增事实性论断、不能编造 source_id；reject=确实无法挽救才整条移除）。"
    "不要规划多步，不要调用工具。"
    "最终输出必须是英文键名的 JSON 对象（见 Skill 的 Output format），"
    "不要 Markdown，不要中文小节标题。"
)


@dataclass
class SuggestionItem:
    """一条待核查建议。source_ids 可显式传入；若为空则从 text 里抽 [source: …]。"""

    text: str
    item_id: str = ""
    source_ids: list[str] = field(default_factory=list)

    def resolved_source_ids(self) -> list[str]:
        if self.source_ids:
            return list(self.source_ids)
        return extract_cited_ids(self.text)


@dataclass
class RejectedItem:
    item: SuggestionItem
    check_number: int
    reason: str
    action: str = "remove"  # remove | hard_block | degrade
    allergen_names: tuple[str, ...] = ()


@dataclass
class VerificationResult:
    accepted: list[SuggestionItem]
    rejected: list[RejectedItem]
    needs_reconciliation_retry: bool = False
    branch: BranchMode = "full_recommend"
    skill_in_prompt: bool = False
    system_prompt: str = ""
    llm_raw: str | None = None


def _item_label(item: SuggestionItem, index: int) -> str:
    return item.item_id or f"item_{index}"


def apply_deterministic_source_check(
    items: list[SuggestionItem],
    available_source_ids: Iterable[str],
    *,
    branch: BranchMode = "full_recommend",
) -> tuple[list[SuggestionItem], list[RejectedItem]]:
    """PRD §10.1 #1（及 D25 候选评估变体）：无合法 source_id → 移除，不补全。"""
    available = set(available_source_ids)
    accepted: list[SuggestionItem] = []
    rejected: list[RejectedItem] = []

    if branch == "candidate_eval":
        # 整包判定：至少一条支持理由带真实 source_id
        any_valid = False
        for item in items:
            cited = item.resolved_source_ids()
            check = validate_citations(
                " ".join(f"[source: {c}]" for c in cited) if cited else item.text,
                available,
            )
            if cited and check.ok and check.has_any_citation:
                any_valid = True
                break
            # also accept if text itself has valid citations
            text_check = validate_citations(item.text, available)
            if text_check.has_any_citation and text_check.ok:
                any_valid = True
                break

        if any_valid:
            # Still drop individual reasons that cite hallucinated ids
            for item in items:
                cited = item.resolved_source_ids()
                text_check = validate_citations(item.text, available)
                cited_check = validate_citations(
                    " ".join(f"[source: {source_id}]" for source_id in cited),
                    available,
                )
                if cited and not cited_check.ok or not text_check.ok:
                    rejected.append(
                        RejectedItem(
                            item=item,
                            check_number=1,
                            reason=(
                                "支持理由引用了不存在的 source_id: "
                                f"{list(dict.fromkeys([*cited_check.missing_ids, *text_check.missing_ids]))}"
                                "（移除，不补全）"
                            ),
                            action="remove",
                        )
                    )
                else:
                    accepted.append(item)
        else:
            for item in items:
                rejected.append(
                    RejectedItem(
                        item=item,
                        check_number=1,
                        reason=(
                            "候选评估：结论缺少至少一条带真实 source_id 的支持理由"
                            "（移除相关条目，不补全 id）"
                        ),
                        action="remove",
                    )
                )
        return accepted, rejected

    for i, item in enumerate(items):
        cited = item.resolved_source_ids()
        if not cited:
            rejected.append(
                RejectedItem(
                    item=item,
                    check_number=1,
                    reason=f"{_item_label(item, i)} 无 source_id，按 PRD §10.1 移除（不补全）",
                    action="remove",
                )
            )
            continue
        text_check = validate_citations(item.text, available)
        missing = list(
            dict.fromkeys(
                [
                    *[
                        source_id
                        for source_id in cited
                        if source_id not in available
                        or not validate_citations(
                            f"[source: {source_id}]", available
                        ).ok
                    ],
                    *text_check.missing_ids,
                ]
            )
        )
        if missing:
            rejected.append(
                RejectedItem(
                    item=item,
                    check_number=1,
                    reason=(
                        f"{_item_label(item, i)} 引用了不存在的 source_id: {missing}"
                        "（幻觉引用，移除，不改写成合法 id）"
                    ),
                    action="remove",
                )
            )
            continue
        accepted.append(item)

    return accepted, rejected


def apply_deterministic_ed_check(
    items: list[SuggestionItem],
) -> tuple[list[SuggestionItem], list[RejectedItem]]:
    """PRD §10.1 #6：ED 防护规则。复用 `backend/guardrails/ed_protection.py`
    的 `scan_model_output()`(规则1数值化表述 + 规则2极端限制性表述，均为系统
    即将发出的建议文本这一侧)——之前这里内联过一份更窄的正则(只挡"数字+kcal/
    大卡"这一种形态)，现在改用阶段 5 guardrails 那份、经过 THREAT_MODEL.md E3
    穷举打磨过的实现(覆盖千分位数字、中文数字大写、BMI/体脂率、"两位数的千卡"
    这类绕开数字本身的换皮表述)，不再维护两份规则。
    """
    accepted: list[SuggestionItem] = []
    rejected: list[RejectedItem] = []
    for i, item in enumerate(items):
        result = ed_protection.scan_model_output(item.text)
        if result.blocked:
            hit = result.primary
            rejected.append(
                RejectedItem(
                    item=item,
                    check_number=6,
                    reason=(
                        f"{_item_label(item, i)} 命中 ED 防护规则({hit.rule.value}):"
                        f"{hit.matched!r}，硬拦截（移除，不改写成定性版）"
                    ),
                    action="hard_block",
                )
            )
        else:
            accepted.append(item)
    return accepted, rejected


def apply_deterministic_allergen_check(
    items: list[SuggestionItem],
    user_allergens: Iterable[str] | None,
) -> tuple[list[SuggestionItem], list[RejectedItem]]:
    """PRD §10.1 #4：过敏原交叉检查(含隐藏成分，如蚝油→甲壳类)。复用
    `backend/guardrails/output_filters.py` 的 `check_allergens()`——ENGINEERING
    §7.3 点名的三类 100%-覆盖确定性检查之一(THREAT_MODEL.md E2，之前是"真空"，
    核查 pass 完全不查这一项)。

    ⚠️ `user_allergens` 为空(默认)时这个检查等于没跑——不是把"用户没有过敏原"
    和"我们不知道用户有没有过敏原"混为一谈，而是调用方(目前是 `api/main.py`)
    还没有从 `user_profile` 读到真实过敏原列表(BUILD_PLAN 已知缺口)。这里不
    编造数据，如实传空。
    """
    allergens = list(user_allergens or [])
    if not allergens:
        return items, []
    accepted: list[SuggestionItem] = []
    rejected: list[RejectedItem] = []
    for i, item in enumerate(items):
        findings = check_allergens(item.text, allergens)
        if findings:
            matched = "、".join(f"{f.matched_term}→{f.allergen}" for f in findings)
            rejected.append(
                RejectedItem(
                    item=item,
                    check_number=4,
                    reason=(
                        f"{_item_label(item, i)} 命中用户过敏原({matched})，硬阻断"
                        "（移除，不重生成）"
                    ),
                    action="hard_block",
                    allergen_names=tuple(dict.fromkeys(f.allergen for f in findings)),
                )
            )
        else:
            accepted.append(item)
    return accepted, rejected


def apply_deterministic_diagnostic_check(
    items: list[SuggestionItem],
) -> tuple[list[SuggestionItem], list[RejectedItem]]:
    """ARCHITECTURE §5.4"输出拦截(诊断性表述/过敏原/无 source_id) | 核查
    pass(步骤6)"——诊断性表述这一项不在 PRD §10.1 的 7 项编号清单里(那 7 项是
    "核查 pass"这个子表自己的编号)，是 §5.4 更大的"输出拦截"表格里单列的一行，
    这里延用编号 8 而不是硬凑进 1-7 的某一项，保持"这条拒绝对应哪一条规则"
    可审计，不含糊。"""
    accepted: list[SuggestionItem] = []
    rejected: list[RejectedItem] = []
    for i, item in enumerate(items):
        finding = check_diagnostic_statement(item.text)
        if finding:
            rejected.append(
                RejectedItem(
                    item=item,
                    check_number=8,
                    reason=(
                        f"{_item_label(item, i)} 含诊断性表述({finding.matched_text!r})，"
                        "输出拦截（移除，替换为免责模板由上游模板负责，本 pass 不代写）"
                    ),
                    action="hard_block",
                )
            )
        else:
            accepted.append(item)
    return accepted, rejected


def build_verification_system_prompt(*, branch: BranchMode, locale: str = "zh") -> str:
    """必然加载 verification_checklist Skill（§6.2 确定性加载）。"""
    branch_note = (
        f"当前分支标注：{branch}。"
        + (
            "请使用 Skill 中「候选评估分支专用规则」替代第 1 条的默认解法。"
            if branch == "candidate_eval"
            else "请使用 Skill 中完整 7 项检查（生成式建议路径）。"
        )
    )
    base = f"{_BASE_SYSTEM}\n{branch_note}"
    return apply_language_instruction(
        compose_prompt_with_skills(base, ["verification_checklist"]), locale
    )


_VALID_VERDICT_ACTIONS = {"accept", "annotate", "reject"}


@dataclass
class _Verdict:
    action: str  # accept | annotate | reject
    text: str | None  # only meaningful for action="annotate"
    check_number: int
    reason: str


def _parse_llm_verdicts(text: str) -> tuple[dict[str, _Verdict], bool]:
    """Parse the English-key JSON the prompt requires.

    Expected: {items:[{item_id, action, text?, check_number?, reason?}],
    retry_reconciliation: bool}. Chinese keys are ignored on purpose so
    extraction matches the English output contract. An `item_id` that's
    missing, has an invalid `action`, or is `annotate` without `text` is
    dropped from the result — the caller treats a missing verdict as
    `accept` (2026-08-31：宽松容错，解析边界情况不该错杀一条本来合格的建议，
    同旧版"不在 reject 列表里就是 accept"的既有精神)。
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # 同 router.py `_strip_json_fences` 的既有做法——单独观察到的解析健壮性
    # 问题：模型有时会在 JSON 后面附加解释文字，即便 prompt 要求只输出 JSON。
    # 去围栏只处理整段被 ```包住这一种情况，这里再定位第一个括号平衡的
    # {...} 片段，避免尾部文字让 json.loads 直接判失败。
    start = text.find("{")
    if start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    text = text[start : i + 1]
                    break
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("verification LLM returned non-JSON; ignoring soft verdicts")
        return {}, False
    verdicts: dict[str, _Verdict] = {}
    for row in data.get("items") or []:
        iid = row.get("item_id")
        action = row.get("action")
        if not iid or action not in _VALID_VERDICT_ACTIONS:
            continue
        row_text = row.get("text")
        if action == "annotate" and not (isinstance(row_text, str) and row_text.strip()):
            continue
        check_number = row.get("check_number")
        verdicts[str(iid)] = _Verdict(
            action=action,
            text=row_text.strip() if isinstance(row_text, str) else None,
            check_number=check_number if isinstance(check_number, int) else 2,
            reason=str(row.get("reason") or ""),
        )
    return verdicts, bool(data.get("retry_reconciliation"))


def _recheck_annotated_items(
    to_recheck: list[tuple[SuggestionItem, "_Verdict"]],
    available_source_ids: list[str],
    *,
    branch: BranchMode,
    user_allergens: Iterable[str] | None,
    skip_allergen_check: bool,
    rejected: list[RejectedItem],
) -> list[SuggestionItem]:
    """Optionally recheck annotate text with the deterministic safety checks.

    The helper is retained for callers that explicitly want defense in depth.
    The default ``verify`` flow intentionally sends annotate text directly after
    the initial deterministic checks, according to the current output policy.
    """
    passed, source_rejected = apply_deterministic_source_check(
        [item for item, _ in to_recheck], available_source_ids, branch=branch
    )
    passed, ed_rejected = apply_deterministic_ed_check(passed)
    if skip_allergen_check:
        allergen_rejected: list[RejectedItem] = []
    else:
        passed, allergen_rejected = apply_deterministic_allergen_check(passed, user_allergens)
    passed, diagnostic_rejected = apply_deterministic_diagnostic_check(passed)

    newly_rejected = [*source_rejected, *ed_rejected, *allergen_rejected, *diagnostic_rejected]
    if newly_rejected:
        logger.warning(
            "annotate downgraded to reject after deterministic recheck · count=%d · checks=%s",
            len(newly_rejected), [r.check_number for r in newly_rejected],
        )
    rejected.extend(newly_rejected)
    return passed


_EVIDENCE_REPAIR_SYSTEM_PROMPT = (
    "你是饮食助手的证据修复 pass。你没有工具，也不能检索。请只根据给你的"
    "原始草稿和核查失败原因，输出一版可直接给用户看的自然语言回答。保留原稿中"
    "有用且已有真实依据的内容；删除核查原因指出的、没有足够依据支持的具体断言，"
    "不要为了修复而新增事实、数字、建议或推断。如果仍保留的是知识库没有核验的"
    "有限通用知识，必须明确加上「模型通用知识，未经过当前知识库核验，可能不完全"
    "准确」这类用户可见标注。"
    "绝对不要编造引用。`[source: chunk_id]`、`[source: source_id]` 等是占位符，"
    "不是实际 source_id；只能原样保留原稿中已经出现且在允许列表里的真实标记，"
    "不能新增任何 source_id。只输出修复后的正文，不要 Markdown 代码围栏、解释"
    "过程或 JSON。"
)


def _evidence_repair_fallback(
    text: str, locale: str, available_source_ids: Iterable[str] = ()
) -> SuggestionItem | None:
    """Keep a bounded, explicitly labelled draft if the repair call fails."""
    available = list(available_source_ids)
    cleaned = strip_invalid_citation_markers(text, available)
    if not cleaned:
        return None
    source_ids = [
        source_id
        for source_id in extract_cited_ids(cleaned)
        if validate_citations(f"[source: {source_id}]", available).ok
    ]
    if not source_ids:
        cleaned = f"{cleaned}\n\n{t('dispatch.model_knowledge_unverified', locale)}"
    return SuggestionItem(
        text=cleaned,
        source_ids=list(dict.fromkeys(source_ids)),
    )


async def repair_insufficient_evidence(
    draft: str,
    failure_reasons: Iterable[str],
    *,
    available_source_ids: Iterable[str] = (),
    complete: CompleteFn | None = None,
    locale: str = "zh",
) -> SuggestionItem | None:
    """Rewrite an evidence-failed draft once, without tools or retrieval.

    This is a recovery path, not a second verification pass. Provenance is
    reconstructed only from citation markers that are already present in the
    repaired text and known to be real retrieval ids.
    """
    if not draft.strip():
        return None
    complete = complete or llm_adapter.complete
    available = list(available_source_ids)
    reasons = [reason for reason in failure_reasons if reason]
    user_message = json.dumps(
        {
            "draft": draft,
            "failure_reasons": reasons,
            "allowed_existing_source_ids": available,
            "instruction": (
                "删除不受支持的具体陈述；能保留的原文尽量保留。没有本地依据但"
                "仍有必要保留的有限通用知识，必须加明确的模型通用知识未核验标注。"
                "不要新增事实或引用。"
            ),
        },
        ensure_ascii=False,
    )
    messages = [
        {
            "role": "system",
            "content": apply_language_instruction(_EVIDENCE_REPAIR_SYSTEM_PROMPT, locale),
        },
        {"role": "user", "content": user_message},
    ]
    try:
        result = await complete(messages, force_prod_tier=False)
        repaired_text = result.text.strip() if isinstance(result.text, str) else ""
    except Exception as exc:  # noqa: BLE001 - recovery must not erase usable output
        logger.warning("evidence repair call failed; using labelled draft fallback: %s", exc)
        return _evidence_repair_fallback(draft, locale, available)
    if not repaired_text:
        return _evidence_repair_fallback(draft, locale, available)

    repaired_text = strip_invalid_citation_markers(repaired_text, available)
    valid_ids = [
        source_id
        for source_id in extract_cited_ids(repaired_text)
        if validate_citations(f"[source: {source_id}]", available).ok
    ]
    if not valid_ids and "[" in repaired_text and "source:" in repaired_text:
        logger.warning("evidence repair returned no valid citation markers")
    if not valid_ids:
        label = t("dispatch.model_knowledge_unverified", locale)
        if label not in repaired_text:
            repaired_text = f"{repaired_text}\n\n{label}"
    return SuggestionItem(text=repaired_text, source_ids=list(dict.fromkeys(valid_ids)))


async def verify(
    items: list[SuggestionItem],
    *,
    available_source_ids: Iterable[str],
    branch: BranchMode = "full_recommend",
    run_llm_soft_checks: bool = True,
    complete: CompleteFn | None = None,
    user_profile_summary: str = "",
    user_allergens: Iterable[str] | None = None,
    skip_allergen_check: bool = False,
    locale: str = "zh",
) -> VerificationResult:
    """跑核查 pass。

    顺序：确定性 source_id → 确定性 ED 防护(四条) → 确定性过敏原交叉检查 →
    确定性诊断性表述拦截 →（可选）一次 LLM 软判定。
    LLM 只能再拒绝，不能把已拒绝条目改写后塞回 accepted，也不能发明 source_id。

    `user_allergens` 默认为空——目前 `/api/chat` 还没有从 `user_profile` 读到
    真实过敏原列表(BUILD_PLAN 已知缺口，THREAT_MODEL.md E2)，传空时过敏原检查
    等于跳过，不是"确认用户没有过敏原"。

    `skip_allergen_check`：调和层已经按过敏原反馈重试过一轮之后跳过第 4 条。
    其余检查（ED / 诊断性表述 / source_id）照跑。
    """
    t0 = time.perf_counter()
    with observation(
        "verify",
        as_type="guardrail",
        metadata={"branch": branch, "item_count": len(items)},
    ):
        result = await _verify_inner(
            items,
            available_source_ids=available_source_ids,
            branch=branch,
            run_llm_soft_checks=run_llm_soft_checks,
            complete=complete,
            user_profile_summary=user_profile_summary,
            user_allergens=user_allergens,
            skip_allergen_check=skip_allergen_check,
            locale=locale,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        update_current(
            output={
                "accepted": len(result.accepted),
                "rejected": [
                    {"check_number": r.check_number, "reason": r.reason, "action": r.action}
                    for r in result.rejected
                ],
                "needs_reconciliation_retry": result.needs_reconciliation_retry,
            },
            metadata={"latency_ms": round(latency_ms, 1)},
            level="WARNING" if result.rejected and not result.accepted else None,
        )
        stage_log(
            logger,
            "verify",
            latency_ms=round(latency_ms, 1),
            accepted=len(result.accepted),
            rejected=len(result.rejected),
            branch=branch,
            intercept_reasons=[r.reason for r in result.rejected] or None,
            intercept_checks=[r.check_number for r in result.rejected] or None,
        )
        return result


async def _verify_inner(
    items: list[SuggestionItem],
    *,
    available_source_ids: Iterable[str],
    branch: BranchMode = "full_recommend",
    run_llm_soft_checks: bool = True,
    complete: CompleteFn | None = None,
    user_profile_summary: str = "",
    user_allergens: Iterable[str] | None = None,
    skip_allergen_check: bool = False,
    locale: str = "zh",
) -> VerificationResult:
    system_prompt = build_verification_system_prompt(branch=branch, locale=locale)
    skill_in_prompt = "核查 pass 检查清单" in system_prompt or "source_id" in system_prompt
    # Normalize once so all initial deterministic checks receive a reusable collection.
    available_source_ids = list(available_source_ids)

    accepted, rejected = apply_deterministic_source_check(
        items, available_source_ids, branch=branch
    )
    accepted, ed_rejected = apply_deterministic_ed_check(accepted)
    rejected.extend(ed_rejected)
    if not skip_allergen_check:
        accepted, allergen_rejected = apply_deterministic_allergen_check(accepted, user_allergens)
        rejected.extend(allergen_rejected)
    accepted, diagnostic_rejected = apply_deterministic_diagnostic_check(accepted)
    rejected.extend(diagnostic_rejected)

    needs_retry = False
    llm_raw = None

    if run_llm_soft_checks and accepted:
        complete = complete or llm_adapter.complete
        # Assign stable ids for LLM reference
        labeled = []
        for i, item in enumerate(accepted):
            iid = item.item_id or f"item_{i}"
            labeled.append((iid, item))

        payload = {
            "branch": branch,
            "user_profile_summary": user_profile_summary,
            "items": [
                {
                    "item_id": iid,
                    "text": item.text,
                    "source_ids": item.resolved_source_ids(),
                }
                for iid, item in labeled
            ],
            "instruction": (
                "对上述已通过确定性 source_id/ED 检查的条目，按 Skill 做第 2/3/5/7 项软判定，"
                "对每个 item_id 给出 accept/annotate/reject 三选一的结论。"
                "只输出这一个 JSON 对象本身，前后不要有任何文字或解释："
                '{"items":[{"item_id":"...","action":"accept|annotate|reject",'
                '"text":"（仅 action=annotate 时提供：改写后的完整条目全文）",'
                '"check_number":2,"reason":"..."}],"retry_reconciliation":false}。'
                "annotate 只能删减引用支持不上的具体陈述、或加免责标注，禁止新增事实性论断、"
                "禁止编造 source_id；没出现在列表里的 item_id 默认按 accept 处理。"
            ),
        }
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        result = await complete(messages, force_prod_tier=False)
        llm_raw = result.text
        verdicts, needs_retry = _parse_llm_verdicts(result.text or "")
        still_accepted: list[SuggestionItem] = []
        for iid, item in labeled:
            verdict = verdicts.get(iid)
            if verdict is None or verdict.action == "accept":
                still_accepted.append(item)
            elif verdict.action == "reject":
                rejected.append(
                    RejectedItem(
                        item=item,
                        check_number=verdict.check_number,
                        reason=verdict.reason or f"LLM 软判定拒绝 {iid}",
                        action="remove",
                    )
                )
            else:  # annotate
                # The initial hard checks have already passed; current policy
                # sends the LLM-generated annotation directly.
                annotated_source_ids = [
                    source_id
                    for source_id in extract_cited_ids(verdict.text or "")
                    if validate_citations(
                        f"[source: {source_id}]", available_source_ids
                    ).ok
                ]
                still_accepted.append(
                    SuggestionItem(
                        text=verdict.text or "",
                        item_id=item.item_id,
                        # Keep provenance metadata limited to markers present
                        # in the rewritten text and valid for this request.
                        source_ids=list(dict.fromkeys(annotated_source_ids)),
                    )
                )
        accepted = still_accepted

    logger.info(
        "verification done · branch=%s · accepted=%d · rejected=%d · retry=%s",
        branch, len(accepted), len(rejected), needs_retry,
    )
    return VerificationResult(
        accepted=accepted,
        rejected=rejected,
        needs_reconciliation_retry=needs_retry,
        branch=branch,
        skill_in_prompt=skill_in_prompt,
        system_prompt=system_prompt,
        llm_raw=llm_raw,
    )
