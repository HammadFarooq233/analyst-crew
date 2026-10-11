import logging

from langgraph.checkpoint.memory import MemorySaver

from analyst_crew.state.serde import crew_serde, crew_state_types
from analyst_crew.reporting.catalog import EvidenceCatalog
from analyst_crew.reporting.report import Report
from analyst_crew.agents.contracts import Plan
from analyst_crew.tools.analysis import BaselineFindings
from tests.crew_harness import base_rows, run_crew, write_sales


def test_every_state_model_is_allowlisted():
    names = {cls.__name__ for cls in crew_state_types()}
    assert {"Plan", "BaselineFindings", "Report", "EvidenceCatalog"} <= names


def test_full_run_checkpoints_load_without_unregistered_type_warnings(tmp_path, monkeypatch, caplog):
    source = tmp_path / "sales.csv"
    write_sales(source, base_rows())
    saver = MemorySaver(serde=crew_serde())
    with caplog.at_level(logging.WARNING):
        run_crew(source, tmp_path, monkeypatch, thread_id="serde", checkpointer=saver)
        result, _ = run_crew(source, tmp_path, monkeypatch, thread_id="serde", checkpointer=saver)
    artifacts = result["artifacts"]
    assert isinstance(artifacts["plan"], Plan)
    assert isinstance(artifacts["report"], Report)
    assert isinstance(artifacts["evidence_catalog"], EvidenceCatalog)
    assert isinstance(artifacts["findings"], BaselineFindings)
    assert "unregistered type" not in caplog.text


def test_memory_state_models_are_allowlisted():
    names = {cls.__name__ for cls in crew_state_types()}
    assert {"MemoryRecord", "AppliedMemory", "InferredProposal", "InferenceProposals", "InferenceResult"} <= names


def test_chart_state_models_are_allowlisted():
    names = {cls.__name__ for cls in crew_state_types()}
    assert {"ChartSpec", "ChartSeries", "ChartArtifact", "ChartReference", "ChartPoint", "ChartSeriesData"} <= names


def test_full_run_with_charts_checkpoints_round_trip_under_strict_msgpack(tmp_path, monkeypatch, caplog):
    """Strict mode blocks any unregistered type (the permissive default only warns, once per process),
    so a missing allowlist entry would turn a stored chart into a plain dict and fail the isinstance checks."""
    from langgraph.checkpoint.serde import _msgpack

    from analyst_crew.reporting.charts import ChartArtifact, ChartReference
    from tests.crew_harness import FULL_AGENTS, crew_plan

    monkeypatch.setattr(_msgpack, "STRICT_MSGPACK_ENABLED", True)
    source = tmp_path / "sales.csv"
    write_sales(source, base_rows())
    saver = MemorySaver(serde=crew_serde())
    kwargs = {"plan": crew_plan(visualizer=True), "available_agents": FULL_AGENTS, "thread_id": "serde-charts",
              "checkpointer": saver}
    with caplog.at_level(logging.WARNING):
        run_crew(source, tmp_path, monkeypatch, **kwargs)
        result, _ = run_crew(source, tmp_path, monkeypatch, **kwargs)
    artifacts = result["artifacts"]
    assert artifacts["charts"] and all(isinstance(chart, ChartArtifact) for chart in artifacts["charts"])
    assert all(isinstance(ref, ChartReference) for ref in artifacts["report"].charts)
    assert "unregistered type" not in caplog.text


def test_applied_memories_checkpoint_round_trip_under_strict_msgpack(tmp_path, monkeypatch, caplog):
    from langgraph.checkpoint.serde import _msgpack
    from langgraph.store.memory import InMemoryStore

    from analyst_crew.memory import MemoryStore
    from analyst_crew.memory.models import AppliedMemory

    monkeypatch.setattr(_msgpack, "STRICT_MSGPACK_ENABLED", True)
    source = tmp_path / "sales.csv"
    write_sales(source, base_rows())
    store = InMemoryStore()
    MemoryStore(store).add("priya", "preference", "x", key="materiality_relative_threshold_percent", value=8.0)
    kwargs = {"thread_id": "serde-memory", "checkpointer": MemorySaver(serde=crew_serde()), "user_id": "priya",
              "store": store}
    with caplog.at_level(logging.WARNING):
        run_crew(source, tmp_path, monkeypatch, **kwargs)
        result, _ = run_crew(source, tmp_path, monkeypatch, **kwargs)
    applied = result["artifacts"]["applied_memories"]
    assert applied and all(isinstance(item, AppliedMemory) for item in applied)
    assert "unregistered type" not in caplog.text
