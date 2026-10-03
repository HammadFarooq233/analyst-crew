"""Visualizer Agent and graph node (ADR-0003).

The Visualizer sits between the Analyst and the Reporter. Its LLM *selects* what
to plot (a mark and Evidence Entry ids per series); code derives every title
from catalog labels, fills every value from the Evidence Catalog and renders the
PNGs, so the model never supplies data or text that appears on a chart. Output
that breaks the closed vocabulary gets one repair retry with validation
feedback; provider failures get a bounded retry. If the model still fails, the
deterministic fallback chart set (revenue bridge waterfall, top drivers bar,
target variance when targets are available) is built from the catalog,
``chart_mode`` is ``fallback`` and the failure is recorded on the Step outcome,
the ``chart_failure`` artifact and the run metadata. The Step is then recorded
as failed so the next run retries the model instead of reusing the failure. A
failure to produce any charts never fails the run: the Report ships without them.
"""
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from analyst_crew.agents.contracts import CREW_AGENTS, Crew, Handoff, Plan
from analyst_crew.agents.resume import _resume_decision, execution_key
from analyst_crew.reporting.catalog import build_evidence_catalog
from analyst_crew.agents.reporter import _is_transient_provider_error, findings_hash
from analyst_crew.llm import make_chat_model
from analyst_crew.reporting.catalog import EvidenceCatalog
from analyst_crew.reporting.chart_choice import VisualizerContent, chart_specs_from_content
from analyst_crew.reporting.charts import (
    MAX_CHARTS, ChartArtifact, ChartSpec, build_chart_artifacts, build_fallback_chart_specs,
    chart_set_reference, load_charts,
)
from analyst_crew.state import CrewState
from analyst_crew.tools.analysis import BaselineFindings
from analyst_crew.tools.retry import invoke_with_optional_config, retry_with_backoff

VISUALIZER_SYSTEM_PROMPT = f"""You are the Visualizer Agent for Analyst Crew, a multi-agent data analysis copilot.

You choose which charts best show the answer to the user's question. Your only source of facts is the Evidence
Catalog below: numbered Evidence Entries, each with an id, a label, a display value and the Findings field it
came from. You never supply data values, titles or series names: you only pick a mark and Evidence Entry ids.
The system fills in every value and label from the catalog and renders the chart.

Rules:
1. Choose at most {MAX_CHARTS} charts, most informative first. Use only these marks:
   - "bar": one series; one horizontal bar per entry (for example the absolute change of the biggest drivers).
   - "grouped_bar": two or more series with the same number of entries each, compared side by side (for example
     the prior-period and current-period entries of the same items, listed in the same order).
   - "waterfall": one series; every entry but the last is a step and the last is the total (the revenue bridge:
     the effect entries in order, then the total of all effects).
2. Every entry id must exist in the catalog and be numeric. Do not mix money, counts and percentages in one chart.
3. Prefer: the revenue bridge as a waterfall, the largest drivers of the net revenue change as a bar, and target
   variance as a bar when the catalog has it. Skip a chart the catalog cannot support.
4. Respond with a single JSON object and nothing else, in exactly this shape and with no other fields:
   {{"charts": [{{"mark": "waterfall", "series": [{{"entry_ids": ["E1", "E2"]}}]}}]}}"""


def _catalog_prompt(entries: list[dict[str, Any]]) -> str:
    lines = []
    for entry in entries:
        flag = " [MATERIAL CHANGE]" if entry.get("material") else ""
        lines.append(f"{entry['id']} | {entry['label']} | {entry['display']} | {entry['source']}{flag}")
    return "\n".join(lines)


def production_visualizer_model() -> Any:
    """Build the live Visualizer seam: invoked with a plain context ``dict``, returns ``VisualizerContent``.
    A response that does not parse raises ``ValueError`` so the single validation retry can feed it back."""
    from langchain_core.messages import HumanMessage, SystemMessage

    structured = make_chat_model(temperature=0).with_structured_output(
        VisualizerContent, method="json_mode", include_raw=True,
    )

    def _invoke(context: dict[str, Any], config: Any = None) -> VisualizerContent:
        system = VISUALIZER_SYSTEM_PROMPT + "\n\nEvidence Catalog (id | label | display | source):\n" + _catalog_prompt(
            context["evidence_catalog"]
        )
        user = (
            f"User question: {context['user_question']}\n"
            f"Step objective: {context['step_objective']}\n"
            f"Validation feedback from a prior attempt (empty if this is the first): "
            f"{context.get('validation_feedback') or 'none'}\n"
            "Return the JSON object now."
        )
        result = structured.invoke([SystemMessage(content=system), HumanMessage(content=user)], config=config)
        if isinstance(result, dict) and "parsed" in result:
            if result["parsed"] is None:
                raise ValueError(f"response was not valid chart-choice JSON: {result.get('parsing_error')}")
            return result["parsed"]
        return result

    return RunnableLambda(_invoke)


def _summarize_schema_error(exc: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']} (got {error.get('input')!r})"
        for error in exc.errors()
    )


class VisualizerAgent:
    """Validated Visualizer loop, mirroring the Reporter's bounded-retry rule: invalid output gets exactly one
    repair retry with the validation errors fed back; provider failures (timeouts, connection errors, 429/5xx)
    get a bounded retry and any other provider failure is terminal at once."""

    max_provider_attempts = 2
    backoff_seconds = 0.0

    def __init__(self, model: Any):
        self.model = model
        self.validation_retries = 0
        self.tool_retries = 0

    def run(self, handoff: Handoff, catalog: EvidenceCatalog, *, config: Any = None) -> list[ChartSpec]:
        context = {
            "step_objective": handoff.step.objective,
            "user_question": handoff.user_question,
            "evidence_catalog": [
                entry.model_dump() for entry in catalog.entries
                if isinstance(entry.value, (int, float)) and not isinstance(entry.value, bool)
            ],
        }
        feedback = ""
        for attempt in range(2):
            calls = 0

            def call() -> Any:
                nonlocal calls
                calls += 1
                return invoke_with_optional_config(self.model, {**context, "validation_feedback": feedback}, config)

            try:
                try:
                    raw, _ = retry_with_backoff(
                        call, is_transient=_is_transient_provider_error,
                        max_attempts=self.max_provider_attempts, backoff_seconds=self.backoff_seconds,
                    )
                finally:
                    self.tool_retries += calls - 1
                try:
                    content = raw if isinstance(raw, VisualizerContent) else VisualizerContent.model_validate(raw)
                except ValidationError as exc:
                    raise ValueError(
                        f"response does not match the required JSON shape: {_summarize_schema_error(exc)}"
                    ) from exc
                return chart_specs_from_content(content, catalog)
            except ValueError as exc:
                if attempt:
                    raise
                self.validation_retries += 1
                feedback = (
                    f"Your previous response was invalid: {exc}. Return the corrected JSON object only, "
                    "choosing only marks bar, grouped_bar or waterfall, and only numeric ids from the catalog."
                )
        raise RuntimeError("Visualizer request loop exhausted")


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
    model_failure: Exception | None = None
    agent = None
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
            artifact_dir = configurable.get("artifact_dir", ".artifacts")
            try:
                model = configurable.get("visualizer_model") or production_visualizer_model()
                agent = VisualizerAgent(model)
                charts = build_chart_artifacts(agent.run(handoff, catalog, config=config), catalog, artifact_dir)
                chart_mode = "llm"
            except Exception as exc:  # noqa: BLE001 — any Visualizer failure degrades to the fallback chart set
                model_failure = exc
                charts = build_chart_artifacts(build_fallback_chart_specs(catalog), catalog, artifact_dir)
                chart_mode = "fallback"
    except Exception as exc:  # noqa: BLE001 — a chart failure degrades visibly and never fails the run
        failure = exc
        charts, chart_mode = [], None

    failure = failure or model_failure
    completed = failure is None
    retry_counts = {
        "validation_retries": getattr(agent, "validation_retries", 0),
        "tool_retries": getattr(agent, "tool_retries", 0),
    }
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
    chart_failure = None
    if failure is not None:
        outcome["error"] = (
            str(failure) if isinstance(failure, (RuntimeError, ValueError))
            else f"{type(failure).__name__}: {failure}"
        )
        chart_failure = {
            "agent": "Visualizer", "step_id": step.step_id, "error": outcome["error"], "retry_counts": retry_counts,
            "degraded_to": "fallback_charts" if chart_mode == "fallback" else "no_charts",
        }
    return {
        "current_step": "visualizer_node",
        "steps_completed": [step.step_id] if completed else [],
        "artifacts": {
            "charts": charts,
            "chart_mode": chart_mode,
            "chart_failure": chart_failure,
            "chart_set": chart_set_reference(charts),
            "handoffs": artifacts.get("handoffs", {}) if handoff is None else {
                **artifacts.get("handoffs", {}), step.step_id: handoff.model_dump(mode="json"),
            },
            "step_outcomes": {**artifacts.get("step_outcomes", {}), step.step_id: outcome},
            "retry_counts": {**artifacts.get("retry_counts", {}), step.step_id: retry_counts},
        },
    }
