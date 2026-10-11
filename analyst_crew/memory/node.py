"""The Crew-level Memory resolution step (ADR-0005). Not an Agent.

Runs once per run, before the Fetcher: deterministically reads the user's Typed
Preferences, resolves them into analysis parameters (a question-level instruction
wins) and records what was applied. The Analyst and Reporter read the resolved
parameters through ``effective_parameters``; nothing else sees preferences.
A storage outage degrades to defaults and is recorded, never raised.
"""
from __future__ import annotations

from typing import Any

from langgraph.config import get_store

from analyst_crew.memory.models import AppliedMemory
from analyst_crew.memory.resolve import resolve_preferences
from analyst_crew.memory.store import MemoryStore, MemoryUnavailable


def effective_parameters(artifacts: dict[str, Any], configurable: dict[str, Any]) -> dict[str, Any]:
    """The analysis parameters a Step runs with: memory-resolved ones, then the caller's explicit ones."""
    return {**(artifacts.get("memory_parameters") or {}), **dict(configurable.get("analysis_parameters", {}))}


def applied_memories(artifacts: dict[str, Any]) -> list[AppliedMemory]:
    return [item if isinstance(item, AppliedMemory) else AppliedMemory.model_validate(item)
            for item in artifacts.get("applied_memories") or []]


def memory_node(state: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    configurable = (config or {}).get("configurable", {})
    user_id = configurable.get("user_id", "default")
    # The question the Analyst's Handoff carries (the Plan's), so the override matches what the Tool will parse.
    plan = state.get("artifacts", {}).get("plan")
    messages = state.get("messages", [])
    question = getattr(plan, "user_question", None) or (messages[0].content if messages else "")
    artifacts: dict[str, Any] = {"memory_parameters": {}, "applied_memories": []}
    try:
        store = get_store()
    except RuntimeError:
        return {"current_step": "memory_node", "artifacts": artifacts}
    try:
        records = MemoryStore(store).read_for_role("Analyst", user_id)
    except MemoryUnavailable:
        artifacts["memory_status"] = "unavailable"
        return {"current_step": "memory_node", "artifacts": artifacts}
    resolution = resolve_preferences(records, question)
    artifacts["memory_parameters"] = resolution.parameters
    artifacts["applied_memories"] = resolution.applied
    return {"current_step": "memory_node", "artifacts": artifacts}
