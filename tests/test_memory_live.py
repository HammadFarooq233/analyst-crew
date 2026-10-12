"""Issue #33: live acceptance of Phase 3 -- the falsifiable two-session scenario on the real model, a real
Postgres and a real Langfuse. Opt-in (``-m "llm and integration"``) and self-skipping without credentials,
Postgres or Langfuse.

It never touches the development data: a scratch database is created on the configured Postgres server,
used for the checkpointer and the Memory store, and dropped afterwards; user ids are unique per run.

Session 1 states a non-default threshold (8%, against the 15% default) through ``remember`` with the real
model. Session 2 is a new thread that restates nothing; the pre-computed North change (+12.0%, 6.0% of the
prior total) must be a Material Change and be listed under Applied Memories, on the Report and on the
Langfuse trace. Control one: after forgetting the memory the change is absent. Control two: a different user
does not see it. Only code-owned facts are asserted, never the model's wording.
"""
import os
import time
import uuid

import pytest
from dotenv import load_dotenv

from analyst_crew.agents import invoke
from analyst_crew.llm import credentials_available
from analyst_crew.memory import MemoryStore
from analyst_crew.memory.update import production_memory_model, remember
from analyst_crew.reporting.report import render_report
from analyst_crew.tools.sales import sales_data_source
from tests.crew_harness import sales_row, write_sales
from tests.report_properties import report_property_violations
from tests.test_tracing import _langfuse_available

load_dotenv()

SAID = "for me, unusual means a swing over 8%"
QUESTION = "Compare June 2025 sales to May 2025 and flag anything unusual"


def _rows():
    """Prior (May) net 100 + 100, current (June) 112 + 100. North is +12.0% (6.0% of the prior total of 200):
    material at an 8% threshold, not at the 15% / 5% defaults."""
    return [
        sales_row("N-M", "2025-05-02", quantity=10, unit_price=10, region="North"),
        sales_row("S-M", "2025-05-03", quantity=10, unit_price=10, region="South"),
        sales_row("N-J", "2025-06-02", quantity=8, unit_price=14, region="North"),
        sales_row("S-J", "2025-06-03", quantity=10, unit_price=10, region="South"),
    ]


def _postgres_available() -> bool:
    conn = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn:
        return False
    try:
        import psycopg

        with psycopg.connect(conn, connect_timeout=3):
            return True
    except Exception:
        return False


@pytest.fixture
def scratch_postgres(monkeypatch):
    """A throwaway database on the configured server; POSTGRES_CONNECTION_STRING points at it for the test."""
    if not credentials_available():
        pytest.skip("LLM credentials not set")
    if not _postgres_available():
        pytest.skip("Postgres not available")
    if not _langfuse_available():
        pytest.skip("Langfuse not available")
    import psycopg
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    original = os.environ["POSTGRES_CONNECTION_STRING"]
    name = f"analyst_accept_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(original, autocommit=True) as admin:
        admin.execute(f'CREATE DATABASE "{name}"')
    scratch = make_conninfo(original, dbname=name)
    assert conninfo_to_dict(scratch)["dbname"] == name
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", scratch)
    try:
        yield scratch
    finally:
        with psycopg.connect(original, autocommit=True) as admin:
            admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


def _run(csv, tmp_path, user, thread):
    config = {"configurable": {
        "thread_id": thread,
        "user_id": user,
        "data_sources": sales_data_source(str(csv)),
        "artifact_dir": str(tmp_path / "artifacts"),
        "analysis_parameters": {"as_of_date": "2025-06-30"},
    }}
    result = invoke(QUESTION, config)
    if result["artifacts"].get("plan_error"):
        pytest.skip(f"Planner API error: {result['artifacts']['plan_error']}")
    artifacts = result["artifacts"]
    failure = artifacts.get("terminal_failure")
    assert failure is None, f"{failure.get('agent')} failed: {failure.get('error')}"
    assert "findings" in artifacts and "report" in artifacts
    return artifacts


def _material(artifacts):
    return {(d.dimension, d.value) for d in artifacts["findings"].material_changes}


def _rendered(artifacts):
    return render_report(artifacts["report"], artifacts["evidence_catalog"])


def _trace_metadata(trace_id):
    from langfuse import Langfuse

    client = Langfuse(public_key=os.environ["LANGFUSE_PUBLIC_KEY"], secret_key=os.environ["LANGFUSE_SECRET_KEY"],
                      host=os.environ["LANGFUSE_HOST"])
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            trace = client.api.trace.get(trace_id)
            if trace.metadata and "report_mode" in trace.metadata:
                return trace.metadata
        except Exception:
            pass
        time.sleep(2)
    pytest.fail(f"no indexed trace with run metadata for {trace_id}")


@pytest.mark.llm
@pytest.mark.integration
def test_a_stated_threshold_applies_in_a_new_thread_and_both_controls_hold(tmp_path, scratch_postgres):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")

    csv = write_sales(tmp_path / "sales.csv", _rows())
    priya, sam = f"priya-{uuid.uuid4().hex[:8]}", f"sam-{uuid.uuid4().hex[:8]}"

    # Control zero: without any memory the defaults call nothing material and nothing is applied.
    baseline = _run(csv, tmp_path, priya, f"base-{uuid.uuid4().hex[:8]}")
    assert _material(baseline) == set()
    assert baseline["findings"].materiality.source == "default"
    assert baseline["applied_memories"] == [] and "Applied Memories" not in _rendered(baseline)

    # Session 1: the real model turns the sentence into a stored typed preference (persisted in Postgres).
    from langgraph.store.postgres import PostgresStore

    with PostgresStore.from_conn_string(scratch_postgres) as pg:
        pg.setup()
        memory = MemoryStore(pg)
        outcome = remember(SAID, priya, memory, production_memory_model())
        assert outcome.stored, outcome.message
        assert outcome.record.kind == "preference", outcome.record
        memory_id = outcome.record.id

    # Session 2: a brand-new thread restates nothing; the stored threshold applies.
    applied = _run(csv, tmp_path, priya, f"session-2-{uuid.uuid4().hex[:8]}")
    assert ("region", "north") in _material(applied)
    assert applied["findings"].materiality.source == "memory"
    assert applied["findings"].materiality.relative_change_percent == 8.0
    rendered = _rendered(applied)
    assert "## Applied Memories" in rendered
    # Past-run Summaries from Priya's earlier runs on this data may also be applied; the preference must be.
    assert [item.memory_id for item in applied["applied_memories"] if item.kind == "preference"] == [memory_id]
    assert report_property_violations(applied["report"], applied["evidence_catalog"], applied["findings"]) == []
    assert applied["findings"].revenue_bridge.reconciliation_gap == pytest.approx(0, abs=1e-6)

    # The trace carries the applied memory ids next to the existing run markers.
    metadata = _trace_metadata(applied["langfuse_trace_id"])
    assert memory_id in metadata["applied_memory_ids"]

    # Control two: a different user never sees Priya's memory.
    other = _run(csv, tmp_path, sam, f"other-{uuid.uuid4().hex[:8]}")
    assert _material(other) == set()
    assert "Applied Memories" not in _rendered(other)
    assert "applied_memory_ids" not in _trace_metadata(other["langfuse_trace_id"])

    # Control one: after forgetting, a new thread no longer shows the change.
    with PostgresStore.from_conn_string(scratch_postgres) as pg:
        MemoryStore(pg).forget(priya, memory_id)
    forgotten = _run(csv, tmp_path, priya, f"forgotten-{uuid.uuid4().hex[:8]}")
    assert _material(forgotten) == set()
    assert forgotten["findings"].materiality.source != "memory"
    assert memory_id not in [item.memory_id for item in forgotten["applied_memories"]]
