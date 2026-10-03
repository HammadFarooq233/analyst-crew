"""Issue #18/#19 — live acceptance for the Critic, using the existing opt-in markers (mirrors test_visualizer_live).

- ``llm`` + ``integration``: the Critic is its own named Langfuse span (``Critic``) with the LLM generation nested
  under it; with a flagged Critique the Reporter revision is its own span (``Reporter (revision)``) with the
  Reporter's LLM generation nested under it.
- ``integration``: a forced Critic failure ships the Report and records the failure and ``critic_outcome`` on the trace.

All of these skip without credentials or a reachable Langfuse, so they are collected but skipped offline.
"""
import os
import time
import uuid

import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from analyst_crew.agents import invoke
from analyst_crew.llm import credentials_available
from analyst_crew.reporting.critique import Critique
from analyst_crew.tools.sales import sales_data_source
from tests.crew_harness import (
    QUESTION, ScriptedAnalystModel, ScriptedFetcherModel, ScriptedPlannerModel, ScriptedReporterModel,
    ScriptedVisualizerModel, base_rows, write_sales,
)
from tests.test_critic import ALL_AGENTS, ScriptedCriticModel, critic_plan, issue
from tests.test_reporter_live import SALES_CSV, _live_config
from tests.test_tracing import _langfuse_available

load_dotenv()


def _fetch_trace(trace_id, *span_names):
    from langfuse import Langfuse

    client = Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"], secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                      host=os.environ["LANGFUSE_HOST"])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            trace = client.api.trace.get(trace_id)
            names = {o.name for o in trace.observations or []}
            if set(span_names) <= names:
                return trace
        except Exception:
            pass
        time.sleep(2)
    pytest.fail(f"no indexed trace with spans {span_names} for {trace_id}")


def _generations_under(trace, span_name):
    by_id = {o.id: o for o in trace.observations}
    span = next(o for o in trace.observations if o.name == span_name)

    def under(observation):
        while observation.parent_observation_id:
            if observation.parent_observation_id == span.id:
                return True
            observation = by_id.get(observation.parent_observation_id)
            if observation is None:
                return False
        return False

    return [o for o in trace.observations if o.type == "GENERATION" and under(o)]


def _scripted_config(tmp_path, *, critic_model, reporter_model=None, thread="critic-live"):
    configurable = {
        "thread_id": f"{thread}-{uuid.uuid4().hex[:8]}",
        "data_sources": sales_data_source(write_sales(tmp_path / "sales.csv", base_rows())),
        "artifact_dir": tmp_path / "artifacts",
        "available_agents": list(ALL_AGENTS),
        "planner_model": ScriptedPlannerModel(critic_plan()),
        "fetcher_model": ScriptedFetcherModel(),
        "analyst_model": ScriptedAnalystModel(),
        "visualizer_model": ScriptedVisualizerModel(),
        "critic_model": critic_model,
        "analysis_parameters": {"as_of_date": "2025-06-30"},
    }
    if reporter_model is not None:
        configurable["reporter_model"] = reporter_model
    return {"configurable": configurable}


@pytest.mark.llm
@pytest.mark.integration
def test_critic_is_its_own_trace_span_with_the_llm_generation_nested_under_it(tmp_path):
    if not credentials_available() or not _langfuse_available():
        pytest.skip("LLM credentials or Langfuse not available")
    if not os.path.exists(SALES_CSV):
        pytest.skip("run scripts/generate_sales_data.py first")

    result = invoke(QUESTION, _live_config(tmp_path, f"live-critic-{uuid.uuid4().hex[:8]}"))
    if result["artifacts"].get("plan_error"):
        pytest.skip(f"Planner API error: {result['artifacts']['plan_error']}")
    if result["artifacts"].get("critic_outcome") is None:
        pytest.skip("the live Plan did not include a Critic Step")

    trace = _fetch_trace(result["artifacts"]["langfuse_trace_id"], "Critic")

    assert _generations_under(trace, "Critic"), "the Critic's LLM call must be nested under its own span"
    assert (trace.metadata or {}).get("critic_outcome") == result["artifacts"]["critic_outcome"]


@pytest.mark.integration
def test_forced_critic_failure_ships_the_report_and_records_the_failure_on_the_trace(tmp_path):
    if not _langfuse_available():
        pytest.skip("Langfuse not available")
    pytest.importorskip("duckdb")
    config = _scripted_config(tmp_path, critic_model=ScriptedCriticModel(RuntimeError("forced Critic failure")),
                              reporter_model=ScriptedReporterModel(), thread="forced-critic-failure")

    result = invoke(QUESTION, config, checkpointer=MemorySaver())

    assert result["artifacts"]["critic_outcome"] == "failed"
    assert result["artifacts"]["report"] is not None and not result["artifacts"]["report"].reviewer_notes
    trace = _fetch_trace(result["artifacts"]["langfuse_trace_id"], "Critic")
    assert trace.metadata["critic_outcome"] == "failed"
    assert "forced Critic failure" in trace.metadata["critic_failure"]["error"]


@pytest.mark.llm
@pytest.mark.integration
def test_reporter_revision_is_its_own_span_with_the_llm_generation_nested_under_it(tmp_path):
    if not credentials_available() or not _langfuse_available():
        pytest.skip("LLM credentials or Langfuse not available")
    pytest.importorskip("duckdb")
    # The Critic is scripted to flag Finding 1 once, so the revision happens deterministically; the Reporter is live.
    config = _scripted_config(tmp_path, critic_model=ScriptedCriticModel(Critique(issues=[issue("claim_1")]), Critique()),
                              thread="live-revision")

    result = invoke(QUESTION, config, checkpointer=MemorySaver())
    if result["artifacts"].get("report_mode") == "fallback":
        pytest.skip("the live Reporter fell back, so there was no draft to revise")

    trace = _fetch_trace(result["artifacts"]["langfuse_trace_id"], "Reporter (revision)", "Critic (re-review)")

    assert _generations_under(trace, "Reporter (revision)"), "the revision's LLM call must be nested under its span"
