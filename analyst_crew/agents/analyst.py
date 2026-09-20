"""Analyst graph node: resolves the Analyst Handoff and produces validated Findings."""
from collections.abc import Mapping
from typing import Any

from analyst_crew.agents.contracts import CREW_AGENTS, Crew, Plan
from analyst_crew.agents.resume import _resume_decision, execution_key
from analyst_crew.state import CrewState
from analyst_crew.tools.analysis import AnalystAgent, BaselineFindings, production_analyst_model


def analyst_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    """Execute the Analyst Step from the Fetcher's Dataset Artifact Handoff.

    Resolves the optional Postgres target / FRED CPI enrichments, resolves
    the Handoff, then derives this Step's execution key from the Step
    contract, the resolved ``analysis_parameters``, and the input Dataset
    Artifact's content hash (see ``execution_key``). Because that content
    hash comes straight from the Fetcher's just-resolved output, a Fetcher
    recompute (changed CSV content) naturally changes the Analyst's key too
    — cascading invalidation, with no separate downstream-invalidation list.
    A parameter-only change (e.g. a different ``as_of_date``) changes only
    the Analyst's key, so the Fetcher's Dataset Artifact is reused untouched
    while Findings recompute. ``_resume_decision`` compares the key against
    the one recorded the last time this thread completed this Step; a match
    reuses the previously produced Findings untouched.
    """
    plan = state.get("artifacts", {}).get("plan")
    configurable = (config or {}).get("configurable", {})
    if not isinstance(plan, Plan):
        return {"current_step": "analyst_node", "artifacts": {"analyst_error": "No executable Plan"}}
    analyst_step = next((step for step in plan.steps if step.agent == "Analyst"), None)
    if analyst_step is None:
        return {"current_step": "analyst_node"}
    agent = None
    try:
        crew = Crew(
            available_agents=configurable.get("available_agents", CREW_AGENTS),
            data_sources=configurable.get("data_sources", {}),
        )
        analysis_parameters = dict(configurable.get("analysis_parameters", {}))
        catalog = configurable.get("data_sources", {})
        as_of_date = analysis_parameters.get("as_of_date")

        # Reused external fetches: if the last Handoff this thread resolved
        # for this Step used the same as_of_date (the sole driver of the
        # comparable-period window these optional sources are queried for),
        # its already-retrieved target/CPI result is byte-identical to what
        # a fresh retrieval would return, so it is reused directly instead
        # of re-issuing the Postgres query / FRED HTTP call.
        prior_handoff = state.get("artifacts", {}).get("handoffs", {}).get(analyst_step.step_id)
        prior_parameters = prior_handoff.get("analysis_parameters", {}) if isinstance(prior_handoff, Mapping) else {}

        def _reused_optional_source(key: str) -> Mapping[str, Any] | None:
            if not as_of_date or prior_parameters.get("as_of_date") != as_of_date:
                return None
            cached = prior_parameters.get(key)
            return cached if isinstance(cached, Mapping) else None

        target_source = catalog.get("sales_targets_postgres") if isinstance(catalog, Mapping) else None
        if target_source is not None and as_of_date:
            # The optional Postgres target Data Source is only ever
            # retrieved here through the read-only Fetcher Tool, and only
            # for the exact comparable period the core CSV analysis already
            # computes via the shared comparison_window() function — never
            # reimplemented. A failure here never raises; it degrades to a
            # recorded source status the Analyst Tool turns into a Finding.
            cached_target_data = _reused_optional_source("target_data")
            if cached_target_data is not None:
                analysis_parameters["target_data"] = cached_target_data
            else:
                from analyst_crew.tools.analysis import comparison_window
                from analyst_crew.tools.targets import retrieve_targets

                window = comparison_window(as_of_date)
                analysis_parameters["target_data"] = retrieve_targets(window.current_start, window.current_end)

        cpi_source = catalog.get("cpi_fred") if isinstance(catalog, Mapping) else None
        if cpi_source is not None and as_of_date:
            # The optional FRED CPI Data Source is only ever retrieved here
            # through the bounded-retry Fetcher Tool, and only for the exact
            # comparable period the core CSV analysis already computes via
            # the shared comparison_window() function — never reimplemented.
            # A failure here never raises; it degrades to a recorded source
            # status the Analyst Tool turns into a Finding and a limitation.
            cached_cpi_data = _reused_optional_source("cpi_data")
            if cached_cpi_data is not None:
                analysis_parameters["cpi_data"] = cached_cpi_data
            else:
                from analyst_crew.tools.analysis import comparison_window
                from analyst_crew.tools.fred import retrieve_cpi

                cpi_window = comparison_window(as_of_date)
                analysis_parameters["cpi_data"] = retrieve_cpi(cpi_window.prior_start, cpi_window.current_end)
        handoff = crew.resolve_handoff(
            analyst_step,
            plan,
            state.get("artifacts", {}).get("step_outputs", {}),
            analysis_parameters,
        )

        dataset_references = {
            key: value
            for key, value in handoff.input_artifacts.items()
            if isinstance(value, dict) and value.get("kind") == "Dataset Artifact"
        }
        input_hashes = sorted({
            str(value["content_hash"]) for value in dataset_references.values() if value.get("content_hash")
        })
        current_key = execution_key(
            analyst_step,
            analysis_parameters=analysis_parameters,
            input_artifact_hashes=input_hashes,
        )
        resume_outcome, resume_reason = _resume_decision(state, analyst_step.step_id, current_key)

        prior_outcome = None
        findings = None
        if resume_outcome == "reused":
            prior_outcome = state.get("artifacts", {}).get("step_outcomes", {}).get(analyst_step.step_id, {})
            reused_findings = state.get("artifacts", {}).get("findings")
            if reused_findings is None:
                # Defensive fallback: a completed prior outcome claims a key
                # match, but no Findings are actually on state — never trust
                # a reuse we can't actually produce output for.
                resume_outcome, resume_reason = "recomputed", "invalidated"
            else:
                findings = (
                    reused_findings if isinstance(reused_findings, BaselineFindings)
                    else BaselineFindings.model_validate(reused_findings)
                )

        if resume_outcome == "reused":
            retry_counts = prior_outcome.get("retry_counts") or {"validation_retries": 0, "tool_retries": 0}
        else:
            model = configurable.get("analyst_model") or production_analyst_model()
            agent = AnalystAgent(model)
            findings = agent.run(handoff, config=config)
            retry_counts = {"validation_retries": agent.validation_retries, "tool_retries": agent.tool_retries}

        return {
            "current_step": "analyst_node",
            "steps_completed": [analyst_step.step_id],
            "artifacts": {
                "terminal_failure": None,
                "findings": findings,
                "handoffs": {
                    **state.get("artifacts", {}).get("handoffs", {}),
                    analyst_step.step_id: handoff.model_dump(mode="json"),
                },
                "step_outcomes": {
                    **state.get("artifacts", {}).get("step_outcomes", {}),
                    analyst_step.step_id: {
                        "status": "completed",
                        "artifact_ids": analyst_step.declared_artifacts or [analyst_step.expected_output],
                        "input_artifact_ids": [findings.dataset_artifact_id],
                        "retry_counts": retry_counts,
                        "execution_key": current_key,
                        "resume_outcome": resume_outcome,
                        "resume_reason": resume_reason,
                    },
                },
                "retry_counts": {
                    **state.get("artifacts", {}).get("retry_counts", {}),
                    analyst_step.step_id: retry_counts,
                },
            },
        }
    except (OSError, RuntimeError, ValueError) as exc:
        retry_counts = {
            "validation_retries": getattr(agent, "validation_retries", 0),
            "tool_retries": getattr(agent, "tool_retries", 0),
        }
        return {
            "current_step": "analyst_node",
            "artifacts": {
                "analyst_error": str(exc),
                "retry_counts": {
                    **state.get("artifacts", {}).get("retry_counts", {}),
                    analyst_step.step_id: retry_counts,
                },
                "terminal_failure": {
                    "agent": "Analyst",
                    "step_id": analyst_step.step_id,
                    "required": True,
                    "status": "failed",
                    "error": str(exc),
                    "retry_counts": retry_counts,
                },
            },
        }
