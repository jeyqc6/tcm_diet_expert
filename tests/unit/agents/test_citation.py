"""
测试目标：backend/agents/citation.py ——溯源(citation grounding)的确定性部分
（引用标记抽取、幻觉引用检测）。不测"引用内容是否真的支持结论"，那部分不是
规则能判定的事，属于 eval 范畴，见模块文档。
对应实现：backend/agents/citation.py
"""
from backend.agents.citation import (
    SCORE_LOW_RELEVANCE_THRESHOLD,
    build_citation_instruction,
    build_score_guidance_instruction,
    extract_cited_ids,
    format_retrieved_context,
    strip_invalid_citation_markers,
    strip_citation_markers,
    validate_citations,
)


class _FakeChunk:
    def __init__(self, source_id, text):
        self.source_id = source_id
        self.text = text


def test_extract_cited_ids_single():
    text = "阳虚质忌生冷 [source: tcm_000123]。"
    assert extract_cited_ids(text) == ["tcm_000123"]


def test_extract_cited_ids_multiple():
    text = "第一句 [source: tcm_000001]，第二句 [source: nutrition_000042]。"
    assert extract_cited_ids(text) == ["tcm_000001", "nutrition_000042"]


def test_extract_cited_ids_none():
    assert extract_cited_ids("这句话完全没有引用标记。") == []


def test_validate_citations_all_valid():
    text = "结论一 [source: a_1]，结论二 [source: a_2]。"
    result = validate_citations(text, available_source_ids=["a_1", "a_2", "a_3"])
    assert result.ok is True
    assert result.cited_ids == ["a_1", "a_2"]
    assert result.missing_ids == []
    assert result.has_any_citation is True


def test_validate_citations_hallucinated_id():
    # a_99 不在这次检索到的结果集合里——幻觉引用，必须被抓出来
    text = "结论一 [source: a_1]，结论二 [source: a_99]。"
    result = validate_citations(text, available_source_ids=["a_1", "a_2"])
    assert result.ok is False
    assert result.missing_ids == ["a_99"]


def test_validate_citations_no_citation_at_all():
    result = validate_citations("完全没有引用的一段话。", available_source_ids=["a_1"])
    assert result.has_any_citation is False
    assert result.ok is True  # 没引用不等于"引用错了"，是两件事，分别由不同信号发现


def test_validate_citations_never_accepts_placeholder_source_id():
    result = validate_citations("结论 [source: chunk_id]", ["chunk_id"])
    assert result.ok is False
    assert result.missing_ids == ["chunk_id"]


def test_format_retrieved_context_includes_source_id():
    chunks = [_FakeChunk("tcm_000123", "阳虚质忌生冷。"), _FakeChunk("tcm_000456", "气虚质宜补中益气。")]
    formatted = format_retrieved_context(chunks)
    assert "[id: tcm_000123]" in formatted
    assert "[id: tcm_000456]" in formatted
    assert "阳虚质忌生冷" in formatted


def test_build_citation_instruction_mentions_format():
    instruction = build_citation_instruction()
    assert "[source:" in instruction


def test_build_score_guidance_instruction_mentions_score_field_and_threshold():
    instruction = build_score_guidance_instruction()
    assert "score" in instruction
    assert str(SCORE_LOW_RELEVANCE_THRESHOLD) in instruction


def test_build_score_guidance_instruction_uses_custom_threshold():
    instruction = build_score_guidance_instruction(threshold=0.5)
    assert "0.5" in instruction


def test_strip_citation_markers_removes_trailing_marker():
    text = "阳虚质忌生冷 [source: tcm_000123]。"
    assert strip_citation_markers(text) == "阳虚质忌生冷。"


def test_strip_citation_markers_removes_multiple_markers():
    text = "第一句 [source: a_1]，第二句 [source: a_2]。"
    assert strip_citation_markers(text) == "第一句，第二句。"


def test_strip_citation_markers_removes_comma_separated_ids():
    text = "建议参考以下资料 [source: tcm_000001, nutrition_000042]。"
    assert strip_citation_markers(text) == "建议参考以下资料。"


def test_strip_citation_markers_cleans_up_mid_sentence_spacing():
    text = "综合建议：红烧肉可以适量食用 [source: t1] [source: n1]，注意控制量。"
    assert strip_citation_markers(text) == "综合建议：红烧肉可以适量食用，注意控制量。"


def test_strip_citation_markers_no_marker_returns_unchanged_content():
    text = "这句话完全没有引用标记。"
    assert strip_citation_markers(text) == text


def test_strip_citation_markers_empty_text():
    assert strip_citation_markers("") == ""


def test_strip_invalid_citation_markers_keeps_only_real_ids():
    text = "结论 [source: real_1, chunk_id, fake_2]。"
    assert strip_invalid_citation_markers(text, ["real_1"]) == "结论 [source: real_1]。"


def test_strip_citation_markers_marker_at_very_end_no_trailing_punctuation():
    text = "维生素C含量很高 [source: n1]"
    assert strip_citation_markers(text) == "维生素C含量很高"
