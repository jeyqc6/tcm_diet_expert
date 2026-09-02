#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared exception hierarchy.

Call sites keep raising the original concrete types
(`NonRetryableError`, `AgentLoopResourceLimitError`, …). Those classes now
subclass one of the types below so an outer layer can write:

    except DietExpertError:
        # expected failure — degrade / fallback / structured 4xx
    except Exception:
        # unexpected bug — logger.exception and a generic 500

The concrete classes stay in the modules that raise them so existing
`from backend.llm.adapter import NonRetryableError` imports keep working.
"""
from __future__ import annotations


class DietExpertError(Exception):
    """Root of expected, application-defined failures.

    Distinguishes "we planned for this" from a genuine bug caught by a bare
    `except Exception`. The latter is what should page; the former is a
    normal degrade / fallback path.
    """

    http_status: int = 400
    error_type: str = "diet_expert_error"


class RetryableError(DietExpertError):
    """429 / 5xx / timeout — ENGINEERING §1.2 says back off and retry."""

    http_status = 503
    error_type = "retryable_error"


class NonRetryableError(DietExpertError):
    """400 / 401 / content policy — retrying cannot help.

    Also re-exported from `backend.llm.adapter` so existing imports keep
    working. Adapter wraps provider failures of this class before they
    leave `complete()`.
    """

    http_status = 400
    error_type = "non_retryable_error"


class DegradedResultError(DietExpertError):
    """One side timed out / failed — the pipeline continues with a partial result.

    "Degraded" is not "failed": ENGINEERING §2 pit 1 (unilateral SubAgent
    output) is the intended path, not an HTTP error. Defined here so that
    path can be typed without inventing a one-off exception later.
    """

    error_type = "degraded_result"


class SubAgentTimeoutError(DegradedResultError):
    """ENGINEERING §1.1: one SubAgent exceeded 45s. Dual dispatch treats this
    as a unilateral failure (`return_exceptions=True`); single-domain emits a
    `subagent_timeout` guardrail. Not an HTTP error on the SSE path.
    """

    error_type = "subagent_timeout"


class ChainTimeoutError(DietExpertError):
    """ENGINEERING §1.1: the whole request exceeded 90s.

    SSE already started, so `/api/chat` emits `chain_timeout` + `done` rather
    than changing the HTTP status. `http_status` is for any non-streaming
    caller that surfaces the same type.
    """

    http_status = 504
    error_type = "chain_timeout"


class ResourceLimitError(DietExpertError):
    """Safety cap hit (Agent Loop `max_tool_calls`, stall guard, …)."""

    http_status = 429
    error_type = "resource_limit_error"


class LLMCallError(DietExpertError):
    """Exhausted retries on an LLM call — callers should degrade per PRD §11."""

    http_status = 503
    error_type = "llm_call_error"


class AuthorizationError(DietExpertError):
    """Caller invoked something outside its declared permission set."""

    http_status = 403
    error_type = "authorization_error"
