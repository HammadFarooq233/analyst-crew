"""Issue #27 through the primary seam: the Crew graph via ``invoke()`` with an
injected per-user store, scripted models and a CSV in a temporary directory."""
import pytest
from langgraph.store.memory import InMemoryStore

from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.memory import MemoryStore, UnavailableStore
from analyst_crew.reporting.report import render_report
from analyst_crew.tools.memory import clear_namespace, seed_memory
from tests.crew_harness import ScriptedPlannerModel, base_rows, crew_plan, run_crew, write_sales
from tests.report_properties import report_property_violations

FACT = "regions are organised by sales manager"


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", base_rows())


def _planner_prompt(source, tmp_path, monkeypatch, *, user_id, store):
    planner = ScriptedPlannerModel(crew_plan())
    result, _ = run_crew(source, tmp_path, monkeypatch, planner_model=planner, user_id=user_id, store=store)
    return planner.prompts[0], result


def test_planner_is_given_the_users_memory_even_with_a_scripted_model(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    seed_memory(store, FACT, user_id="priya")
    prompt, result = _planner_prompt(source, tmp_path, monkeypatch, user_id="priya", store=store)
    assert FACT in prompt
    assert result["artifacts"]["memory_status"] == "available"


def test_planner_is_not_given_another_users_memory(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    seed_memory(store, FACT, user_id="priya")
    prompt, _ = _planner_prompt(source, tmp_path, monkeypatch, user_id="sam", store=store)
    assert FACT not in prompt
    assert "(none found)" in prompt


def test_memory_survives_a_new_run_on_the_same_store(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    seed_memory(store, FACT, user_id="priya")
    _planner_prompt(source, tmp_path, monkeypatch, user_id="priya", store=store)
    prompt, _ = _planner_prompt(source, tmp_path, monkeypatch, user_id="priya", store=store)
    assert FACT in prompt


def test_storage_outage_runs_with_defaults_and_a_caveat_names_it(source, tmp_path, monkeypatch):
    prompt, result = _planner_prompt(source, tmp_path, monkeypatch, user_id="priya", store=UnavailableStore())
    artifacts = result["artifacts"]
    assert "plan_error" not in artifacts and artifacts.get("terminal_failure") is None
    assert artifacts["memory_status"] == "unavailable"
    assert "(none found)" in prompt
    report = artifacts["report"]
    assert any("memory" in line.text.lower() and "unavailable" in line.text.lower() for line in report.caveats)
    rendered = render_report(report, artifacts["evidence_catalog"])
    assert "unavailable" in rendered.split("## Caveats")[1].split("##")[0]
    assert report_property_violations(report, artifacts["evidence_catalog"], artifacts["findings"]) == []
    assert _run_metadata(result)["memory_status"] == "unavailable"


def test_a_healthy_run_has_no_memory_caveat(source, tmp_path, monkeypatch):
    _, result = _planner_prompt(source, tmp_path, monkeypatch, user_id="priya", store=InMemoryStore())
    assert not any("memory" in line.text.lower() for line in result["artifacts"]["report"].caveats)


def test_the_seed_helper_and_clear_still_work_and_are_per_user():
    store = InMemoryStore()
    seed_memory(store, "one", user_id="a")
    seed_memory(store, "two")
    memory = MemoryStore(store)
    assert [r.content for r in memory.list("a")] == ["one"]
    assert [r.content for r in memory.list("default")] == ["two"]
    clear_namespace(store)
    assert memory.list("a") == [] and memory.list("default") == []
