"""Exception hierarchy: concrete types keep their names; they now share a root."""
from __future__ import annotations

from backend.agents.agent_loop import AgentLoopResourceLimitError
from backend.exceptions import (
    AuthorizationError,
    ChainTimeoutError,
    DegradedResultError,
    DietExpertError,
    NonRetryableError,
    ResourceLimitError,
    RetryableError,
    SubAgentTimeoutError,
)
from backend.exceptions import LLMCallError
from backend.llm.adapter import LLMCallError as AdapterLLMCallError
from backend.llm.adapter import NonRetryableError as AdapterNonRetryableError
from backend.llm.providers.replay import ReplayFixtureMissing
from backend.mcp_server.exceptions import ToolNotDeclaredError
from backend.mcp_server.roles import CallerRole


def test_adapter_reexports_the_same_non_retryable_class():
    assert AdapterNonRetryableError is NonRetryableError


def test_exceptions_module_reexports_llm_call_error():
    assert LLMCallError is AdapterLLMCallError


def test_concrete_types_are_diet_expert_errors():
    assert issubclass(NonRetryableError, DietExpertError)
    assert issubclass(LLMCallError, DietExpertError)
    assert issubclass(AgentLoopResourceLimitError, ResourceLimitError)
    assert issubclass(AgentLoopResourceLimitError, DietExpertError)
    assert issubclass(ToolNotDeclaredError, AuthorizationError)
    assert issubclass(ToolNotDeclaredError, DietExpertError)
    assert issubclass(ReplayFixtureMissing, DietExpertError)
    assert issubclass(RetryableError, DietExpertError)
    assert issubclass(DegradedResultError, DietExpertError)
    assert issubclass(SubAgentTimeoutError, DegradedResultError)
    assert issubclass(ChainTimeoutError, DietExpertError)


def test_except_diet_expert_error_catches_concrete_types():
    raised = []
    for exc in (
        NonRetryableError("no retry"),
        LLMCallError("exhausted"),
        AgentLoopResourceLimitError("cap"),
        ToolNotDeclaredError(role=CallerRole.TCM_SUBAGENT, tool_name="write_memory"),
        ReplayFixtureMissing("missing"),
    ):
        try:
            raise exc
        except DietExpertError:
            raised.append(type(exc).__name__)
    assert raised == [
        "NonRetryableError",
        "LLMCallError",
        "AgentLoopResourceLimitError",
        "ToolNotDeclaredError",
        "ReplayFixtureMissing",
    ]


def test_bare_exception_is_not_a_known_error():
    assert not isinstance(RuntimeError("bug"), DietExpertError)
