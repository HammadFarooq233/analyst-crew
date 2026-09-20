"""Issue #13 — live acceptance for the Reporter, using the existing markers.

- ``llm``: the documented CLI scenario against a real model yields an ``llm``-mode
  Report whose every number matches the Evidence Catalog.
- ``integration``: the Reporter appears as its own span in the Langfuse trace with
  the LLM generation nested under it, and a forced Reporter failure leaves a
  visibly marked Fallback Report with the failure and report mode on the trace.
"""
import os
import time
import uuid

import pytest
from dotenv import load_dotenv

from analyst_crew.agents import invoke
from analyst_crew.llm import credentials_available
from analyst_crew.reporting.report import render_report
from analyst_crew.tools.fred import fred_cpi_data_source
from analyst_crew.tools.sales import sales_data_source
from analyst_crew.tools.targets import sales_targets_data_source
from tests.crew_harness import (
    QUESTION, ScriptedAnalystModel, ScriptedFetcherModel, ScriptedPlannerModel, ScriptedReporterModel, base_rows,
    crew_plan, write_sales,
)
from tests.report_properties import report_property_violations
from tests.test_tracing import _langfuse_available

load_dotenv()
SALES_CSV = "data/sales.csv"


def _live_config(tmp_path, thread_id):
    return {"configurable": {
        "thread_id": thread_id,
        "user_id": "live-test",
        "data_sources": {**sales_data_source(SALES_CSV), **sales_targets_data_source(), **fred_cpi_data_source()},
        "artifact_dir": str(tmp_path / "artifacts"),
        "analysis_parameters": {"as_of_date": "2025-06-30"},
    }}


def _fetch_trace(trace_id):
    from langfuse import Langfuse

    client = Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"], secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                      host=os.environ["LANGFUSE_HOST"])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            trace = client.api.trace.get(trace_id)
            if any(o.name == "reporter_node" for o in trace.observations or []):
                return trace
        except Exception:
            pass
        time.sleep(2)
    pytest.fail(f"no indexed trace with a reporter_node span for {trace_id}")


def _live_result(tmp_path):
    result = invoke(QUESTION, _live_config(tmp_path, f"live-report-{uuid.uuid4().hex[:8]}"))
    if result["artifacts"].get("plan_error"):
        pytest.skip(f"Planner API error: {result['artifacts']['plan_error']}")
    return result


@pytest.mark.llm
def test_documented_scenario_yields_an_llm_mode_report_with_every_number_traceable(tmp_path):
    if not credentials_available():
        pytest.skip("LLM credentials not set")
    if not os.path.exists(SALES_CSV):
        pytest.skip("run scripts/generate_sales_data.py first")

    result = _live_result(tmp_path)

    artifacts = result["artifacts"]
    assert artifacts["report_mode"] == "llm", artifacts.get("terminal_failure")
    report, catalog, findings = artifacts["report"], artifacts["evidence_catalog"], artifacts["findings"]
    assert report_property_violations(report, catalog, findings) == []
    assert "Fallback Report" not in render_report(report, catalog)


@pytest.mark.llm
@pytest.mark.integration
def test_reporter_is_its_own_trace_span_with_the_llm_generation_nested_under_it(tmp_path):
    if not credentials_available() or not _langfuse_available():
        pytest.skip("LLM credentials or Langfuse not available")
    if not os.path.exists(SALES_CSV):
        pytest.skip("run scripts/generate_sales_data.py first")

    result = _live_result(tmp_path)
    trace = _fetch_trace(result["artifacts"]["langfuse_trace_id"])

    by_id = {o.id: o for o in trace.observations}
    span = next(o for o in trace.observations if o.name == "reporter_node")

    def under_reporter(observation):
        while observation.parent_observation_id:
            if observation.parent_observation_id == span.id:
                return True
            observation = by_id.get(observation.parent_observation_id)
            if observation is None:
                return False
        return False

    generations = [o for o in trace.observations if o.type == "GENERATION" and under_reporter(o)]
    assert generations, "the Reporter's LLM call must be nested under its own node span"
    assert (trace.metadata or {}).get("report_mode") == result["artifacts"]["report_mode"]


@pytest.mark.integration
def test_forced_reporter_failure_records_report_mode_and_failure_on_the_trace(tmp_path, monkeypatch):
    if not _langfuse_available():
        pytest.skip("Langfuse not available")
    pytest.importorskip("duckdb")
    source = write_sales(tmp_path / "sales.csv", base_rows())
    config = {"configurable": {
        "thread_id": f"forced-failure-{uuid.uuid4().hex[:8]}",
        "data_sources": sales_data_source(source),
        "artifact_dir": tmp_path / "artifacts",
        "planner_model": ScriptedPlannerModel(crew_plan()),
        "fetcher_model": ScriptedFetcherModel(),
        "analyst_model": ScriptedAnalystModel(),
        "reporter_model": ScriptedReporterModel(RuntimeError("forced Reporter failure")),
        "analysis_parameters": {"as_of_date": "2025-06-30"},
    }}

    from langgraph.checkpoint.memory import MemorySaver

    result = invoke(QUESTION, config, checkpointer=MemorySaver())

    assert result["artifacts"]["report_mode"] == "fallback"
    assert "without the Reporter Agent" in render_report(result["artifacts"]["report"], result["artifacts"]["evidence_catalog"])
    trace = _fetch_trace(result["artifacts"]["langfuse_trace_id"])
    assert trace.metadata["report_mode"] == "fallback"
    assert trace.metadata["terminal_failure"]["agent"] == "Reporter"
    assert "forced Reporter failure" in trace.metadata["terminal_failure"]["error"]
