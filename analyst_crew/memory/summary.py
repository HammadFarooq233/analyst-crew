"""Past-run Summaries: written by the Crew's Memory Update step, never by an Agent (ADR-0005).

The summary text is a fixed template over run records only: the Analyst Tool's Findings, the
Plan's Steps and the recorded Step outcomes. No Agent prose (Report text, Critic notes,
Planner objectives) is read. It is tied to the source-data content hash and is stale, and not
applied, as soon as the data on disk no longer hashes to that value.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.config import get_store

from analyst_crew.memory.models import AppliedMemory, InferenceResult, MemoryRecord
from analyst_crew.memory.store import MemoryStore, MemoryUnavailable
from analyst_crew.tools.analysis import BaselineFindings


def current_source_hashes(data_sources: Mapping[str, Any] | None) -> set[str]:
    """Content hashes of the CSV Data Sources this run would read (the same hash Findings record)."""
    from analyst_crew.agents.resume import _source_content_hash

    hashes = (_source_content_hash(reference) for reference in (data_sources or {}).values())
    return {value for value in hashes if value}


def fresh_summaries(records: list[MemoryRecord], source_hashes: set[str]) -> list[MemoryRecord]:
    """The records still valid for this data: a changed source file makes a summary stale."""
    return [r for r in records if r.kind != "summary" or (r.source_hash is not None and r.source_hash in source_hashes)]


def applied_summaries(records: list[MemoryRecord]) -> list[AppliedMemory]:
    return [AppliedMemory(memory_id=r.id, kind="summary", origin=r.origin) for r in records if r.kind == "summary"]


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.1f}%"


def build_summary(artifacts: Mapping[str, Any]) -> tuple[str, str] | None:
    """(content, source hash) for a successful run, or None when the run did not succeed."""
    findings = artifacts.get("findings")
    if artifacts.get("terminal_failure") or not isinstance(findings, BaselineFindings):
        return None
    # The source file's byte hash, the one the Planner can recompute before any fetch.
    dataset = (artifacts.get("datasets") or {}).get(findings.dataset_artifact_id) or {}
    source_hash = dataset.get("input_hash")
    if not source_hash:
        return None
    plan = artifacts.get("plan")
    window, net = findings.window, findings.net_revenue
    material = ", ".join(f"{d.dimension}={d.value}" for d in findings.material_changes) or "none"
    steps = ", ".join(step.agent for step in getattr(plan, "steps", [])) or "unknown"
    parts = [
        f"Previous successful run compared {window.current_start} to {window.current_end} with "
        f"{window.prior_start} to {window.prior_end} ({window.mode}).",
        f"Net revenue {net.current:,.2f} against {net.prior:,.2f} ({_percent(net.percent_change)}).",
        f"Material Changes: {len(findings.material_changes)} ({material}) at thresholds "
        f"{findings.materiality.relative_change_percent:g}% relative and "
        f"{findings.materiality.absolute_change_percent_of_prior_total:g}% absolute.",
        f"Rows: {findings.included_rows} included, {findings.quarantined_rows} quarantined, "
        f"{findings.duplicate_rows} duplicates.",
        f"Steps run: {steps}.",
    ]
    critic = artifacts.get("critic_outcome")
    if critic:
        parts.append(f"Critic outcome: {critic}.")
    return " ".join(parts), str(source_hash)


def _user_words(state: Mapping[str, Any]) -> str:
    """The user's own question, taken from the first message, never from the Planner's Plan wording."""
    messages = state.get("messages") or []
    content = getattr(messages[0], "content", "") if messages else ""
    return content if isinstance(content, str) else ""


def memory_update_node(state: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
    """Crew-level Memory Update: store this run's Past-run Summary, then infer from the user's words and
    run records only. Not an Agent; never fails the run."""
    artifacts = state.get("artifacts", {})
    configurable = (config or {}).get("configurable") or {}
    user_id = configurable.get("user_id", "default")
    built = build_summary(artifacts)
    out: dict[str, Any] = {"memory_update": {"status": "skipped"}}
    try:
        memory = MemoryStore(get_store())
    except RuntimeError:
        return {"current_step": "memory_update_node", "artifacts": out}

    if built is not None:
        content, source_hash = built
        try:
            if any(r.content == content and r.source_hash == source_hash for r in memory.list(user_id, "summary")):
                out["memory_update"] = {"status": "unchanged"}  # a rerun with identical results adds nothing
            else:
                record = memory.add(user_id, "summary", content, origin="run_record", source_hash=source_hash)
                out["memory_update"] = {"status": "written", "memory_id": record.id}
        except MemoryUnavailable:
            out["memory_update"] = {"status": "unavailable"}

    model = configurable.get("inference_model")
    if model is not None:
        from analyst_crew.memory.infer import infer

        # Run records are the code-built digest of Findings, never the Report, Critic or Planner prose.
        out["inference"] = infer(model, _user_words(state), built[0] if built else "", user_id, memory)
    else:
        out["inference"] = InferenceResult(status="skipped")
    return {"current_step": "memory_update_node", "artifacts": out}
