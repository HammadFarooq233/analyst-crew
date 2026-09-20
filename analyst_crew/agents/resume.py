"""Step execution keys and the resume/invalidation decision (issue #11)."""
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from analyst_crew.agents.contracts import Step
from analyst_crew.state import CrewState


def _step_contract_fingerprint(step: Step) -> dict[str, Any]:
    """The declarative parts of a Step contract that define what work it
    does, independent of any runtime data. Changing any of these means the
    Step no longer means the same thing, so any previously computed output
    for it must be invalidated."""
    return {
        "step_id": step.step_id,
        "agent": step.agent,
        "objective": step.objective,
        "source_ids": sorted(step.source_ids),
        "expected_output": step.expected_output,
        "declared_artifacts": sorted(step.declared_artifacts or [step.expected_output]),
        "depends_on": sorted(step.depends_on),
    }


def execution_key(
    step: Step,
    *,
    analysis_parameters: Mapping[str, Any] | None = None,
    input_artifact_hashes: Sequence[str] = (),
) -> str:
    """Stable content hash identifying one Step's resolved execution (issue #11).

    Composed from exactly three things, matching the acceptance criteria:

    1. The Step's own contract (``_step_contract_fingerprint``) — step_id,
       agent, objective, declared source_ids/artifacts, and dependencies.
       A Planner-authored change to what a Step is supposed to do (a new
       objective, a different declared source, a changed dependency) always
       changes the key.
    2. The resolved ``analysis_parameters`` for this Handoff. The Fetcher
       Step never receives ``analysis_parameters`` in its key (callers pass
       ``{}``), so a parameter-only change (e.g. a different ``as_of_date``)
       leaves the Fetcher's key unchanged — its Dataset Artifact is reused —
       while the Analyst's key (which does receive them) changes and it
       recomputes.
    3. The content hash(es) of the Step's input artifacts. For the Fetcher
       this is the source CSV's own byte hash, computed cheaply before any
       parse/validate work; for the Analyst this is the Fetcher's output
       Dataset Artifact content hash. Because the Analyst's key transitively
       depends on the Fetcher's output hash, a Fetcher recompute (triggered
       by changed CSV content) automatically changes the Analyst's key too —
       cascading invalidation falls out of this composition; it is never a
       separately maintained "invalidate downstream" list.

    Two Steps with identical contract, parameters, and input hashes always
    hash to the same key and are safe to treat as interchangeable on resume;
    a difference in any one of the three changes the key and forces a
    recompute.
    """
    payload = {
        "step": _step_contract_fingerprint(step),
        "analysis_parameters": dict(analysis_parameters or {}),
        "input_artifact_hashes": sorted(input_artifact_hashes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_content_hash(reference: Any) -> str | None:
    """Cheap identity hash for a catalogued CSV Data Source, computed before
    any expensive parse/validate work. This is the same byte hash the
    Fetcher Tool itself records as ``DatasetArtifact.input_hash`` once a
    fetch actually runs (see ``analyst_crew.tools.sales.fetch_csv``), so a
    match here guarantees the Fetcher would produce byte-identical output."""
    if not isinstance(reference, Mapping) or reference.get("kind") != "CSV":
        return None
    location = reference.get("location")
    if not location:
        return None
    path = Path(str(location))
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resume_decision(state: CrewState, step_id: str, current_key: str) -> tuple[str, str]:
    """Decide whether a Step's previously checkpointed output is still valid
    to reuse for this thread (issue #11).

    Returns ``(resume_outcome, resume_reason)``:

    - ``resume_outcome`` is ``"reused"`` (skip recomputation, reuse the
      stored output) or ``"recomputed"`` (run the Step's work again).
    - ``resume_reason`` explains why, as one of five distinct, inspectable
      labels:
        - ``"unchanged"``: a completed prior outcome exists for this Step
          and its execution key still matches this run's -> reused.
        - ``"invalidated"``: a completed prior outcome exists but its
          execution key no longer matches (a changed Step contract,
          analysis parameter, or input artifact content) -> recomputed.
        - ``"failed"``: the prior run recorded a terminal failure for this
          exact Step -> recomputed.
        - ``"interrupted"``: this thread has been invoked before (its
          accumulated message history holds more than the single message
          this invoke is adding) but this Step has neither a completed
          outcome nor a recorded failure — the previous invoke must have
          been interrupted before this Step's node returned, and therefore
          before anything about that attempt could be checkpointed ->
          recomputed from a clean Handoff (LangGraph's checkpointer already
          restarts at the last node boundary that *did* complete; nothing
          about a genuinely interrupted node survives to resume from).
        - ``"never_ran"``: this is the first invoke on this thread ->
          recomputed.
    """
    artifacts = state.get("artifacts", {})
    prior_outcome = artifacts.get("step_outcomes", {}).get(step_id)
    prior_failure = artifacts.get("terminal_failure")
    had_prior_invoke = len(state.get("messages", [])) > 1

    if isinstance(prior_failure, Mapping) and prior_failure.get("step_id") == step_id:
        return "recomputed", "failed"
    if isinstance(prior_outcome, Mapping) and prior_outcome.get("status") == "completed":
        if prior_outcome.get("execution_key") == current_key:
            return "reused", "unchanged"
        return "recomputed", "invalidated"
    if had_prior_invoke:
        return "recomputed", "interrupted"
    return "recomputed", "never_ran"
