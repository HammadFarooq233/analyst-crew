"""Issue #16 — live acceptance for the LLM Visualizer, using the existing opt-in markers.

- ``llm``: the documented scenario against a real model yields ``chart_mode: llm`` with rendered PNGs whose
  every plotted value is an Evidence Entry raw value.
- ``integration``: the Visualizer is its own Langfuse span with the LLM generation nested under it, and a forced
  Visualizer failure leaves a marked fallback with ``chart_mode`` and the failure on the trace.
"""
import os
import time
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from analyst_crew.agents import invoke
from analyst_crew.llm import credentials_available
from analyst_crew.tools.sales import sales_data_source
from tests.crew_harness import (
    FULL_AGENTS, QUESTION, ScriptedAnalystModel, ScriptedFetcherModel, ScriptedPlannerModel, ScriptedReporterModel,
    ScriptedVisualizerModel, base_rows, crew_plan, write_sales,
)
from tests.report_properties import chart_property_violations
from tests.test_reporter_live import SALES_CSV, _live_config
from tests.test_tracing import _langfuse_available

load_dotenv()


def _fetch_trace(trace_id):
    from langfuse import Langfuse

    client = Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"], secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                      host=os.environ["LANGFUSE_HOST"])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            trace = client.api.trace.get(trace_id)
            if any(o.name == "visualizer_node" for o in trace.observations or []):
                return trace
        except Exception:
            pass
        time.sleep(2)
    pytest.fail(f"no indexed trace with a visualizer_node span for {trace_id}")


def _live_result(tmp_path):
    result = invoke(QUESTION, _live_config(tmp_path, f"live-visualizer-{uuid.uuid4().hex[:8]}"))
    if result["artifacts"].get("plan_error"):
        pytest.skip(f"Planner API error: {result['artifacts']['plan_error']}")
    return result


@pytest.mark.llm
def test_documented_scenario_yields_llm_charts_rendered_as_pngs(tmp_path):
    if not credentials_available():
        pytest.skip("LLM credentials not set")
    if not os.path.exists(SALES_CSV):
        pytest.skip("run scripts/generate_sales_data.py first")

    artifacts = _live_result(tmp_path)["artifacts"]

    assert artifacts["chart_mode"] == "llm", artifacts.get("chart_failure")
    charts = artifacts["charts"]
    assert 1 <= len(charts) <= 3
    assert all(Path(chart.location).read_bytes().startswith(b"\x89PNG") for chart in charts)
    assert chart_property_violations(charts, artifacts["evidence_catalog"]) == []


@pytest.mark.llm
@pytest.mark.integration
def test_visualizer_is_its_own_trace_span_with_the_llm_generation_nested_under_it(tmp_path):
    if not credentials_available() or not _langfuse_available():
        pytest.skip("LLM credentials or Langfuse not available")
    if not os.path.exists(SALES_CSV):
        pytest.skip("run scripts/generate_sales_data.py first")

    result = _live_result(tmp_path)
    trace = _fetch_trace(result["artifacts"]["langfuse_trace_id"])

    by_id = {o.id: o for o in trace.observations}
    span = next(o for o in trace.observations if o.name == "visualizer_node")

    def under_visualizer(observation):
        while observation.parent_observation_id:
            if observation.parent_observation_id == span.id:
                return True
            observation = by_id.get(observation.parent_observation_id)
            if observation is None:
                return False
        return False

    assert [o for o in trace.observations if o.type == "GENERATION" and under_visualizer(o)], (
        "the Visualizer's LLM call must be nested under its own node span"
    )
    assert (trace.metadata or {}).get("chart_mode") == result["artifacts"]["chart_mode"]


@pytest.mark.integration
def test_forced_visualizer_failure_records_chart_mode_and_failure_on_the_trace(tmp_path):
    if not _langfuse_available():
        pytest.skip("Langfuse not available")
    pytest.importorskip("duckdb")
    source = write_sales(tmp_path / "sales.csv", base_rows())
    config = {"configurable": {
        "thread_id": f"forced-viz-failure-{uuid.uuid4().hex[:8]}",
        "data_sources": sales_data_source(source),
        "artifact_dir": tmp_path / "artifacts",
        "available_agents": list(FULL_AGENTS),
        "planner_model": ScriptedPlannerModel(crew_plan(visualizer=True)),
        "fetcher_model": ScriptedFetcherModel(),
        "analyst_model": ScriptedAnalystModel(),
        "reporter_model": ScriptedReporterModel(),
        "visualizer_model": ScriptedVisualizerModel(RuntimeError("forced Visualizer failure")),
        "analysis_parameters": {"as_of_date": "2025-06-30"},
    }}

    result = invoke(QUESTION, config, checkpointer=MemorySaver())

    assert result["artifacts"]["chart_mode"] == "fallback"
    assert result["artifacts"]["report"].chart_mode == "fallback"
    trace = _fetch_trace(result["artifacts"]["langfuse_trace_id"])
    assert trace.metadata["chart_mode"] == "fallback"
    assert "forced Visualizer failure" in trace.metadata["chart_failure"]["error"]
