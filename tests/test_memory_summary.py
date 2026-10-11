"""Issue #31 through the primary seam: the whole Crew via ``invoke()`` with an injected per-user
store and a CSV in a temporary directory. A Past-run Summary is written by the Crew's Memory
Update step from run records, tied to the source-data hash, and given to the Planner only while
that hash still matches."""
import hashlib

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.memory import MemoryStore, UnavailableStore
from analyst_crew.reporting.report import render_report
from tests.crew_harness import (
    ScriptedPlannerModel, ScriptedReporterModel, crew_plan, run_crew, sales_row, write_sales,
)
from tests.report_properties import report_property_violations


def _rows(june_north=14):
    return [
        sales_row("N-M", "2025-05-02", quantity=10, unit_price=10, region="North"),
        sales_row("S-M", "2025-05-03", quantity=10, unit_price=10, region="South"),
        sales_row("N-J", "2025-06-02", quantity=8, unit_price=june_north, region="North"),
        sales_row("S-J", "2025-06-03", quantity=10, unit_price=10, region="South"),
    ]


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", _rows())


def _run(source, tmp_path, monkeypatch, store, thread, user="priya", **kwargs):
    kwargs.setdefault("reporter_model", ScriptedReporterModel())
    planner = kwargs.setdefault("planner_model", ScriptedPlannerModel(crew_plan()))
    result, _ = run_crew(source, tmp_path, monkeypatch, thread_id=thread, user_id=user, store=store, **kwargs)
    return result["artifacts"], planner


def _summaries(store, user="priya"):
    return MemoryStore(store).list(user, "summary")


def test_a_successful_run_stores_a_summary_tied_to_the_source_hash(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, "t-1")
    (summary,) = _summaries(store)
    assert summary.origin == "run_record" and summary.user_id == "priya"
    assert summary.source_hash == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "Net revenue" in summary.content and "Material Changes" in summary.content
    assert artifacts["memory_update"]["status"] == "written"


def test_the_summary_is_given_to_the_planner_on_a_later_run_and_listed_under_applied_memories(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _run(source, tmp_path, monkeypatch, store, "t-1")
    (summary,) = _summaries(store)
    artifacts, planner = _run(source, tmp_path, monkeypatch, store, "t-2")
    assert summary.content in planner.prompts[0]
    assert [(m.memory_id, m.kind, m.origin) for m in artifacts["applied_memories"]] == [(summary.id, "summary", "run_record")]
    rendered = render_report(artifacts["report"], artifacts["evidence_catalog"])
    assert "## Applied Memories" in rendered and "Past-run Summary" in rendered
    assert report_property_violations(artifacts["report"], artifacts["evidence_catalog"], artifacts["findings"]) == []
    assert _run_metadata({"artifacts": artifacts})["applied_memory_ids"] == [summary.id]


def test_changed_source_data_makes_the_summary_stale_and_it_is_not_applied(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _run(source, tmp_path, monkeypatch, store, "t-1")
    (old,) = _summaries(store)
    write_sales(source, _rows(june_north=15))
    artifacts, planner = _run(source, tmp_path, monkeypatch, store, "t-2")
    assert old.content not in planner.prompts[0] and "(none found)" in planner.prompts[0]
    assert artifacts["applied_memories"] == []
    assert "Applied Memories" not in render_report(artifacts["report"], artifacts["evidence_catalog"])
    # The run on the new data stores a fresh summary for the new hash; the stale one is replaced.
    (current,) = _summaries(store)
    assert current.source_hash == hashlib.sha256(source.read_bytes()).hexdigest() != old.source_hash


def test_a_summary_is_never_written_from_agent_prose(source, tmp_path, monkeypatch):
    from tests.crew_harness import valid_content

    def prosey(context):
        content = valid_content(context)
        content.summary.text = "ZEBRA-PROSE glorious quarter, {} ".format(content.summary.text)
        return content

    store = InMemoryStore()
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, "t-1", reporter_model=ScriptedReporterModel(prosey))
    assert "ZEBRA" in artifacts["report"].summary.text
    (summary,) = _summaries(store)
    assert "ZEBRA" not in summary.content


def test_a_failed_run_stores_no_summary(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, "t-1",
                        reporter_model=ScriptedReporterModel(RuntimeError("down")))
    assert artifacts.get("terminal_failure")
    assert _summaries(store) == []
    assert artifacts["memory_update"]["status"] == "skipped"


def test_summaries_are_per_user(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _run(source, tmp_path, monkeypatch, store, "t-1", user="priya")
    artifacts, planner = _run(source, tmp_path, monkeypatch, store, "t-2", user="sam")
    assert artifacts["applied_memories"] == [] and "(none found)" in planner.prompts[0]


def test_a_forgotten_summary_is_not_applied(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _run(source, tmp_path, monkeypatch, store, "t-1")
    (summary,) = _summaries(store)
    MemoryStore(store).forget("priya", summary.id)
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, "t-2")
    assert artifacts["applied_memories"] == []


def test_a_resumed_thread_keeps_its_plan_and_stores_no_duplicate_summary(source, tmp_path, monkeypatch):
    store, saver = InMemoryStore(), MemorySaver()
    _run(source, tmp_path, monkeypatch, store, "same", checkpointer=saver)
    artifacts, planner = _run(source, tmp_path, monkeypatch, store, "same", checkpointer=saver)
    assert planner.calls == 0  # the checkpointed Plan is reused, so no summary was given to the Planner
    assert artifacts["applied_memories"] == []
    assert len(MemoryStore(store).history("priya", "summary")) == 1


def test_an_unreachable_store_does_not_fail_the_run_or_write(source, tmp_path, monkeypatch):
    artifacts, _ = _run(source, tmp_path, monkeypatch, UnavailableStore(), "t-1")
    assert artifacts["report"] is not None and not artifacts.get("terminal_failure")
    assert artifacts["memory_update"]["status"] == "unavailable"
