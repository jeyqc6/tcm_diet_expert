#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
溯源(citation grounding)——TCM/Nutrition SubAgent 共用的引用格式与校验逻辑。

planning/roadmap.md 阶段2 Step4 把 citation grounding 拆成三步：
  1. chunk 入库时带 source_id —— 已有，knowledge_chunks.chunk_id 就是
     (backend/mcp_server/tools/_retrieval_common.py 的 RetrievedChunk.source_id)
  2. prompt 里要求每条结论引用 source_id —— 本模块 build_citation_instruction()
     + format_retrieved_context()
  3. 生成后校验：
     a. id 真实存在 —— 确定性代码，本模块 validate_citations() 做这个
     b. 引用内容确实支持该结论 —— 这不是规则能判定的事，需要 LLM-as-judge 或
        人工抽检，属于 eval 范畴(对应 PRD 里的溯源类指标)，不是本模块的职责，
        更不应该伪装成"看起来确定性"的正则去蒙混过去(呼应 ENGINEERING §7.3
        "确定性优先"——优先是指"能做确定性检查的地方必须做"，不是"所有检查
        都要装成确定性的")

引用格式约定(本次新定，此前设计文档没有钉死具体语法，写在这里备查)：
    每条结论性陈述后紧跟一个行内标记 `[source: <chunk_id>]`，
    例如："阳虚质忌生冷 [source: tcm_000123]"。
选这个格式的原因：足够简单，一个正则就能抽取；不需要模型切到结构化输出模式
(json mode 等)就能用，对现有"生成自然语言回答"的 SubAgent prompt 侵入最小。
局限：如果模型经常把引用标记放错句子边界，逐句核对会不准；真出现这个问题，
下一步是换成"结尾统一列引用 chunk_id 列表"的格式，但那样就没法做"这句话
到底有没有依据"的逐句检查了——两种格式是有取舍的，不是纯粹的选择偏好。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

CITATION_PATTERN = re.compile(r"\[source:\s*([A-Za-z0-9_\-]+)\]")
# These tokens are prompt placeholders, never retrieval identifiers.
_PLACEHOLDER_SOURCE_IDS = frozenset(
    {"chunk_id", "source_id", "citation_id", "your_source_id"}
)


def build_citation_instruction() -> str:
    """拼进 SubAgent system prompt 里的引用要求。"""
    return (
        "你的每一条结论性陈述后面必须紧跟一个引用标记，格式是 [source: chunk_id]，"
        "chunk_id 必须是下面提供的检索结果里真实出现过的 id，不能编造、不能凭记忆写一个。"
        "没有检索依据支持的内容不要说；如果确实需要给出你自己的推断，"
        "要明确说明这是推断而非知识库依据，不要给它配引用标记。"
    )


CLARIFICATION_MARKER = "[NEED_CLARIFICATION]"


def build_clarification_instruction() -> str:
    """拼进 SubAgent system prompt 里的"信息不足时反问"约定(D20 五处 agent
    行为点第3条,ARCHITECTURE.md 新增小节,2026-08-27 实现)。

    格式选择和引用标记同一个原则：足够简单，一个字符串前缀判断就能识别，
    不需要模型切到结构化输出模式。约定是"整条回复严格以这个标记开头"，
    不是"回复里某处出现这个标记"——避免和正常结论文本混在一起时难以判断
    这条回复到底是不是在反问。

    真实测试(2026-08-27，本地真实 Anthropic Haiku)发现：不加范围限定时，
    模型会把这条指令过度泛化——"红烧肉能不能吃"这种已经点名具体菜品的问题，
    也会因为"没说明血脂/血糖情况、没说明用糖腌制与否"这类外围细节而触发反问，
    体验上等同于对每个候选评估都追问一轮。收窄成"只在判断对象本身是谁/是
    什么都不确定时才触发"，并明确要求外围细节用合理假设兜底(可在结论里注明
    假设，但不能反问)——这才是 PRD §11"输入模糊"真正想覆盖的场景(指代不明，
    不是信息不够全面)。"""
    return (
        f"只有在你连判断对象本身是什么都无法确定时(比如用户说"
        f"「这个」「这道菜」却没有指明具体是哪种食物)，才需要反问——"
        f"这种情况下不要猜测或编造一个假设，回复严格以 {CLARIFICATION_MARKER} "
        f"开头，后面紧跟需要用户补充的具体问题，不要输出其他内容"
        f"(不要引用标记、不要部分结论)。\n"
        f"判断对象已经明确时(用户已经点名具体食物/菜品)，不要因为其他外围细节"
        f"(比如具体的健康状况、这次打算吃多少、菜的具体做法)不够全就触发反问——"
        f"这些信息按已经提供给你的用户画像信息结合常规做法/常规食用量给出合理"
        f"判断即可，需要的话可以在结论里注明「判断所依据的假设是什么」，"
        f"但不要把这些当成必须先问清楚才能回答的前提。"
    )


def extract_clarification_question(final_text: str) -> str | None:
    """命中约定返回需要追问的问题文本(已去掉标记本身)；没命中返回 None。"""
    stripped = final_text.strip()
    if stripped.startswith(CLARIFICATION_MARKER):
        return stripped[len(CLARIFICATION_MARKER):].strip()
    return None


# 2026-08-30(检索评分方法优化)：`retrieve_tcm`/`retrieve_nutrition` 返回的
# 每个 chunk 都带一个 `score` 字段(工具调用结果的原始 JSON 里就有，见
# backend/agents/agent_loop.py `_json_default` 用 dataclasses.asdict() 序列化
# `RetrievedChunk`)，但之前没有任何地方告诉模型这个数字什么意思、该怎么用——
# 算出来了，等于没算。这条指令补上使用指引。低于 SCORE_LOW_RELEVANCE_THRESHOLD
# 只是一个粗略的经验值(不是拿真实数据校准过的精确阈值)，够用来提醒模型"这批
# 结果整体偏弱，别硬凑"，不是一个需要精确调参的业务规则。
SCORE_LOW_RELEVANCE_THRESHOLD = 0.3


def build_score_guidance_instruction(threshold: float = SCORE_LOW_RELEVANCE_THRESHOLD) -> str:
    """拼进 SubAgent system prompt——告诉模型检索结果里的 `score` 字段是什么、
    该怎么用。刻意不在后端按这个阈值静默过滤掉低分 chunk(比如直接不返回给
    模型)——那样会让模型完全看不到"知识库其实没有强相关资料"这个信号本身,
    只会导致它更容易在信息不足时还是想办法凑一个回答；交给模型自己判断,
    和 D15"只拒绝不改写"的精神一致：宁可让模型自己看到弱证据然后诚实declined,
    也不要后端替它做"这些证据不够好所以我先偷偷藏起来"的隐性决定。"""
    return (
        f"每个检索结果都带一个 score 字段(0-1 的相似度分数，越接近 1 代表这条"
        f"资料和你的问题越匹配)。如果这次检索结果里最高的 score 明显偏低"
        f"(比如都低于 {threshold})，说明知识库里可能没有直接覆盖这个问题——"
        f"不要为了凑出一个答案而牵强引用弱相关的资料，应该如实说明知识库里没有"
        f"找到直接相关的资料，而不是勉强给出一个引用不牢靠的结论。"
    )


def format_retrieved_context(chunks: Iterable) -> str:
    """把检索结果格式化成 prompt 里能直接用的文本块，每条都标出它的 source_id，
    模型才有确切的 id 可以引用（而不是自己瞎编一个看起来像 id 的字符串）。"""
    return "\n\n".join(f"[id: {c.source_id}] {c.text}" for c in chunks)


def extract_cited_ids(text: str) -> list[str]:
    return CITATION_PATTERN.findall(text)


@dataclass
class CitationCheckResult:
    ok: bool
    cited_ids: list[str] = field(default_factory=list)
    # 引用了、但不在这次检索到的结果集合里的 id —— 幻觉引用
    missing_ids: list[str] = field(default_factory=list)
    has_any_citation: bool = False


_CITATION_WITH_LEADING_SPACE = re.compile(
    r"[ \t]*\[source:\s*[A-Za-z0-9_\-]+"
    r"(?:\s*[,，]\s*[A-Za-z0-9_\-]+)*\]"
)


def strip_citation_markers(text: str) -> str:
    """去掉 `[source: chunk_id]` 引用标记本身，只留自然语言正文——供**展示给
    用户**的那份文本用(`backend/agents/dispatch.py` `_stream_verification_result`
    的 `token` 事件)，不是给核查 pass 用的。

    引用标记的作用是让核查 pass(`validate_citations()`)能确定性验证"这句话
    是不是真有检索依据"，以及给前端一份可展开溯源的 `source_id` 列表——但那份
    列表已经通过独立的 `source` SSE 事件单独吐给前端了(§5.2 步骤8"溯源可
    展开")，`token` 事件里再把 `[source: tcm_001410]` 这种机器可读标记原样
    暴露给用户，只是重复信息 + 看起来像未渲染完的调试文本，不是真实产品会给
    用户看的东西。**必须在核查 pass 跑完之后才调用这个函数**——核查 pass 依赖
    这些标记本身做幻觉引用检测，先剥离了就没有可验证的东西了。
    """
    if not text:
        return text
    stripped = _CITATION_WITH_LEADING_SPACE.sub("", text)
    # 标记原来在句中(两侧都有空格)时，去掉标记后会留下"文本  ，"这类双空格/
    # 标点前多余空格，清理掉，不留视觉上的"这里少了点什么"的痕迹。
    stripped = re.sub(r"[ \t]{2,}", " ", stripped)
    stripped = re.sub(r"[ \t]+([，。！？；：、,.!?;:])", r"\1", stripped)
    return stripped.strip()


def strip_invalid_citation_markers(
    text: str, available_source_ids: Iterable[str]
) -> str:
    """Remove only citation markers that are not real retrieval ids.

    This is output sanitation for recovery text, not a replacement for
    ``validate_citations`` in the verification pass.
    """
    available = set(available_source_ids)

    def replace(match: re.Match[str]) -> str:
        leading = match.group(0)[: len(match.group(0)) - len(match.group(0).lstrip())]
        ids = re.findall(r"[A-Za-z0-9_\-]+", match.group(1))
        valid = [
            source_id
            for source_id in ids
            if source_id in available and source_id not in _PLACEHOLDER_SOURCE_IDS
        ]
        return f"{leading}[source: {', '.join(valid)}]" if valid else ""

    pattern = re.compile(
        r"[ \t]*\[source:\s*([A-Za-z0-9_\-]+"
        r"(?:\s*[,，]\s*[A-Za-z0-9_\-]+)*)\]"
    )
    cleaned = pattern.sub(replace, text)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([，。！？；：、,.!?;:])", r"\1", cleaned)
    return cleaned.strip()


def validate_citations(text: str, available_source_ids: Iterable[str]) -> CitationCheckResult:
    """确定性检查：文本里引用的每个 id 是不是真的在这次检索到的结果里。

    不判断"引用的内容是否真的支持这句话"——见模块文档，那是 eval 的职责。
    `ok=False` 只代表"存在幻觉引用(id 不存在)"，不代表"这条建议内容有问题"。
    """
    cited = extract_cited_ids(text)
    available = set(available_source_ids)
    missing = [
        cid for cid in cited if cid in _PLACEHOLDER_SOURCE_IDS or cid not in available
    ]
    return CitationCheckResult(
        ok=len(missing) == 0,
        cited_ids=cited,
        missing_ids=missing,
        has_any_citation=len(cited) > 0,
    )
