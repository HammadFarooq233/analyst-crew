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
