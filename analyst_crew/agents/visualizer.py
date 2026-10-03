"""Visualizer graph node (ADR-0003), deterministic fallback chart set only.

The Visualizer sits between the Analyst and the Reporter. Here it emits the
fallback chart set (revenue bridge waterfall, top drivers bar, target variance
when targets are available) built purely from the Evidence Catalog, and marks
``chart_mode: fallback``. No model is involved. A failure to produce charts
never fails the run: the Step is recorded as failed, the Report ships without
charts, and the next run retries rather than reusing the failure.
"""
from pathlib import Path
from typing import Any

from analyst_crew.agents.contracts import CREW_AGENTS, Crew, Plan
from analyst_crew.agents.resume import _resume_decision, execution_key
from analyst_crew.reporting.catalog import build_evidence_catalog
from analyst_crew.agents.reporter import findings_hash
from analyst_crew.reporting.charts import (
    ChartArtifact, build_chart_artifacts, build_fallback_chart_specs, chart_set_reference, load_charts,
)
from analyst_crew.state import CrewState
from analyst_crew.tools.analysis import BaselineFindings


def visualizer_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    artifacts = state.get("artifacts", {})
    plan = artifacts.get("plan")
    configurable = (config or {}).get("configurable", {})
    if not isinstance(plan, Plan):
        return {"current_step": "visualizer_node", "artifacts": {"visualizer_error": "No executable Plan"}}
    step = next((candidate for candidate in plan.steps if candidate.agent == "Visualizer"), None)
    if step is None:
        return {"current_step": "visualizer_node"}
    raw_findings = artifacts.get("findings")
    if raw_findings is None:
        return {"current_step": "visualizer_node", "artifacts": {"visualizer_error": "No Findings to chart"}}
    findings = raw_findings if isinstance(raw_findings, BaselineFindings) else BaselineFindings.model_validate(raw_findings)

    current_hash = findings_hash(findings)
    findings_reference = {
        "kind": "Findings", "artifact_id": f"findings-{current_hash[:16]}", "content_hash": current_hash,
        "dataset_artifact_id": findings.dataset_artifact_id,
    }
    # The Visualizer key is its Step contract plus the Findings hash; analysis parameters reach it only
    # through the Findings, so they are deliberately not part of the key.
    current_key = execution_key(step, input_artifact_hashes=[current_hash])
    resume_outcome, resume_reason = _resume_decision(state, step.step_id, current_key)

    charts: list[ChartArtifact] = []
    chart_mode: str | None = None
    handoff = None
    failure: Exception | None = None
    try:
        catalog = build_evidence_catalog(findings)
        analyst_step = next((candidate for candidate in plan.steps if candidate.agent == "Analyst"), None)
        findings_names = (analyst_step.declared_artifacts or [analyst_step.expected_output]) if analyst_step else []
        crew = Crew(
            available_agents=configurable.get("available_agents", CREW_AGENTS),
            data_sources=configurable.get("data_sources", {}),
        )
        handoff = crew.resolve_handoff(
            step, plan,
            {**artifacts.get("step_outputs", {}), **{name: findings_reference for name in findings_names}},
            dict(configurable.get("analysis_parameters", {})),
        )
        if resume_outcome == "reused":
            stored = load_charts(artifacts.get("charts"))
            if artifacts.get("chart_mode") is None or any(not Path(chart.location).is_file() for chart in stored):
                # Defensive: never trust a reuse whose output we cannot actually produce.
                resume_outcome, resume_reason = "recomputed", "invalidated"
            else:
                charts, chart_mode = stored, artifacts["chart_mode"]
        if resume_outcome != "reused":
            charts = build_chart_artifacts(
                build_fallback_chart_specs(catalog), catalog, configurable.get("artifact_dir", ".artifacts"),
            )
            chart_mode = "fallback"
    except Exception as exc:  # noqa: BLE001 — a chart failure degrades visibly and never fails the run
        failure = exc
        charts, chart_mode = [], None

    completed = failure is None
    retry_counts = {"validation_retries": 0, "tool_retries": 0}
    outcome: dict[str, Any] = {
        "status": "completed" if completed else "failed",
        "artifact_ids": step.declared_artifacts or [step.expected_output],
        "input_artifact_ids": [findings_reference["artifact_id"]],
        "retry_counts": retry_counts,
        "execution_key": current_key,
        "resume_outcome": resume_outcome,
        "resume_reason": resume_reason,
        "chart_mode": chart_mode,
    }
    if failure is not None:
        outcome["error"] = f"{type(failure).__name__}: {failure}" if not isinstance(failure, (RuntimeError, ValueError)) else str(failure)
    return {
        "current_step": "visualizer_node",
        "steps_completed": [step.step_id] if completed else [],
        "artifacts": {
            "charts": charts,
            "chart_mode": chart_mode,
            "chart_set": chart_set_reference(charts),
            "handoffs": artifacts.get("handoffs", {}) if handoff is None else {
                **artifacts.get("handoffs", {}), step.step_id: handoff.model_dump(mode="json"),
            },
            "step_outcomes": {**artifacts.get("step_outcomes", {}), step.step_id: outcome},
            "retry_counts": {**artifacts.get("retry_counts", {}), step.step_id: retry_counts},
        },
    }
