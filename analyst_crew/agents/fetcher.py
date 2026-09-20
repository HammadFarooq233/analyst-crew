"""Fetcher graph node: resolves the Fetcher Handoff and produces the Dataset Artifact."""
from typing import Any

from analyst_crew.agents.contracts import PHASE1_AGENTS, Crew, Plan
from analyst_crew.agents.resume import _resume_decision, _source_content_hash, execution_key
from analyst_crew.state import CrewState
from analyst_crew.tools.sales import FetcherAgent, production_fetcher_model


def fetcher_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    """Execute the first Fetcher Step through its resolved Handoff.

    Before doing any expensive work (reading/parsing/validating the CSV and
    writing a Parquet Dataset Artifact), this resolves the Handoff, computes
    the source CSV's cheap content hash, and derives this Step's execution
    key (see ``execution_key``). ``_resume_decision`` compares that key
    against the one recorded the last time this thread completed this Step:
    a match reuses the previously produced Dataset Artifact reference
    untouched (no re-read, no re-parse); a mismatch — or no valid prior
    completion — recomputes it. The resume outcome/reason are always
    recorded on ``step_outcomes`` for CrewState inspection.
    """
    plan = state.get("artifacts", {}).get("plan")
    configurable = (config or {}).get("configurable", {})
    catalog = configurable.get("data_sources", {})
    if not isinstance(plan, Plan):
        return {"current_step": "fetcher_node", "artifacts": {"fetcher_error": "No executable Plan"}}
    fetch_step = next((step for step in plan.steps if step.agent == "Fetcher"), None)
    if fetch_step is None:
        return {"current_step": "fetcher_node"}
    agent = None
    try:
        crew = Crew(available_agents=configurable.get("available_agents", PHASE1_AGENTS), data_sources=catalog)
        handoff = crew.resolve_handoff(fetch_step, plan, state.get("artifacts", {}).get("datasets", {}))

        source_hash = None
        if handoff.source_values:
            source_reference = next(iter(handoff.source_values.values())).reference
            source_hash = _source_content_hash(source_reference)
        current_key = execution_key(
            fetch_step,
            analysis_parameters={},
            input_artifact_hashes=[source_hash] if source_hash else [],
        )
        resume_outcome, resume_reason = _resume_decision(state, fetch_step.step_id, current_key)

        reference = None
        prior_outcome = None
        if resume_outcome == "reused":
            prior_outcome = state.get("artifacts", {}).get("step_outcomes", {}).get(fetch_step.step_id, {})
            prior_artifact_id = next(iter(prior_outcome.get("artifact_ids", [])), None)
            reference = (
                state.get("artifacts", {}).get("datasets", {}).get(prior_artifact_id)
                if prior_artifact_id else None
            )
            if reference is None:
                # Defensive fallback: the prior outcome claims a key match
                # but its Dataset Artifact reference is missing from state —
                # never trust a reuse we can't actually produce output for.
                resume_outcome, resume_reason = "recomputed", "invalidated"

        output_names = fetch_step.declared_artifacts or [fetch_step.expected_output]

        if resume_outcome == "reused":
            retry_counts = prior_outcome.get("retry_counts") or {"validation_retries": 0, "tool_retries": 0}
        else:
            model = configurable.get("fetcher_model") or production_fetcher_model()
            agent = FetcherAgent(model)
            artifact = agent.run(handoff, artifact_dir=configurable.get("artifact_dir", ".artifacts"), config=config)
            reference = artifact.as_reference()
            retry_counts = {"validation_retries": agent.validation_retries, "tool_retries": agent.tool_retries}

        return {
            "current_step": "fetcher_node",
            "steps_completed": [fetch_step.step_id],
            "artifacts": {
                "terminal_failure": None,
                "datasets": {reference["artifact_id"]: reference},
                "step_outputs": {name: reference for name in output_names},
                "dataset_profile": reference.get("profile", {}),
                "handoffs": {
                    **state.get("artifacts", {}).get("handoffs", {}),
                    fetch_step.step_id: handoff.model_dump(mode="json"),
                },
                # merge_dicts is a shallow update of "artifacts" — it would
                # otherwise replace the whole step_outcomes/retry_counts
                # sub-dicts wholesale, wiping the Analyst's outcome recorded
                # by a previous invoke on this thread before analyst_node
                # even runs this turn. Spread the prior entries explicitly,
                # matching analyst_node's own merge below.
                "step_outcomes": {
                    **state.get("artifacts", {}).get("step_outcomes", {}),
                    fetch_step.step_id: {
                        "status": "completed",
                        "artifact_ids": [reference["artifact_id"]],
                        "retry_counts": retry_counts,
                        "execution_key": current_key,
                        "resume_outcome": resume_outcome,
                        "resume_reason": resume_reason,
                    },
                },
                "retry_counts": {
                    **state.get("artifacts", {}).get("retry_counts", {}),
                    fetch_step.step_id: retry_counts,
                },
            },
        }
    except (OSError, RuntimeError, ValueError) as exc:
        retry_counts = {
            "validation_retries": getattr(agent, "validation_retries", 0),
            "tool_retries": getattr(agent, "tool_retries", 0),
        }
        return {
            "current_step": "fetcher_node",
            "artifacts": {
                "fetcher_error": str(exc),
                "retry_counts": {
                    **state.get("artifacts", {}).get("retry_counts", {}),
                    fetch_step.step_id: retry_counts,
                },
                "terminal_failure": {
                    "agent": "Fetcher",
                    "step_id": fetch_step.step_id,
                    "required": True,
                    "status": "failed",
                    "error": str(exc),
                    "retry_counts": retry_counts,
                },
            },
        }
