"""Issue #30 through the primary seam: the whole Crew graph via ``invoke()``, checking what each
Agent role was given. A Free-text Fact reaches the Planner and the Reporter only, never changes a
displayed value, and is disclosed under Applied Memories with its origin."""
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from analyst_crew.memory import MemoryStore
from analyst_crew.memory.update import MemoryProposal, remember
from analyst_crew.reporting.report import render_report
from tests.crew_harness import (
    ScriptedAnalystModel, ScriptedFetcherModel, ScriptedPlannerModel, ScriptedReporterModel,
    ScriptedVisualizerModel, crew_plan, run_crew, sales_row, write_sales,
)
from tests.test_critic import ALL_AGENTS, ScriptedCriticModel, critic_plan
from tests.test_memory_preferences import ScriptedMemoryModel

FACT = "regions follow our sales managers"


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", [
        sales_row("N-M", "2025-05-02", quantity=10, unit_price=10, region="North"),
        sales_row("N-J", "2025-06-02", quantity=8, unit_price=14, region="North"),
    ])


def _say(store, sentence=FACT, user="priya"):
    return remember(sentence, user, MemoryStore(store), ScriptedMemoryModel(MemoryProposal(kind="fact"))).record


def _run(source, tmp_path, monkeypatch, store, **kwargs):
    kwargs.setdefault("user_id", "priya")
    kwargs.setdefault("reporter_model", ScriptedReporterModel())
    return run_crew(source, tmp_path, monkeypatch, store=store, **kwargs)


def _section(rendered, name):
    return rendered.split(f"## {name}")[1].split("\n## ")[0] if f"## {name}" in rendered else None


class Recording:
    """Wraps a scripted model and records every context it is invoked with."""

    def __init__(self, model):
        self.model, self.seen = model, []

    def invoke(self, context):
        self.seen.append(context)
        return self.model.invoke(context)


def test_a_fact_reaches_the_planner_and_the_reporter_and_no_other_agent(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _say(store)
    planner, reporter = ScriptedPlannerModel(critic_plan()), ScriptedReporterModel()
    visualizer, critic = ScriptedVisualizerModel(), ScriptedCriticModel()
    fetcher, analyst = Recording(ScriptedFetcherModel()), Recording(ScriptedAnalystModel())

    _run(source, tmp_path, monkeypatch, store, planner_model=planner, reporter_model=reporter,
         visualizer_model=visualizer, critic_model=critic, fetcher_model=fetcher, analyst_model=analyst,
         available_agents=ALL_AGENTS)

    assert FACT in planner.prompts[0]
    assert FACT in str(reporter.contexts[0])
    for name, seen in {"Fetcher": fetcher.seen, "Analyst": analyst.seen, "Visualizer": visualizer.contexts,
                       "Critic": critic.contexts}.items():
        assert seen, f"the {name} was never called"
        assert FACT not in str(seen), f"the {name} was given the fact"


def test_a_fact_never_changes_a_displayed_value(source, tmp_path, monkeypatch):
    without, _ = _run(source, tmp_path, monkeypatch, InMemoryStore(), thread_id="a")
    store = InMemoryStore()
    _say(store)
    with_fact, _ = _run(source, tmp_path, monkeypatch, store, thread_id="b")

    assert with_fact["artifacts"]["findings"] == without["artifacts"]["findings"]
    assert with_fact["artifacts"]["evidence_catalog"] == without["artifacts"]["evidence_catalog"]
    assert with_fact["artifacts"]["memory_parameters"] == {}


def test_the_fact_is_listed_under_applied_memories_with_its_origin(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _say(store)
    result, _ = _run(source, tmp_path, monkeypatch, store)
    artifacts = result["artifacts"]
    rendered = render_report(artifacts["report"], artifacts["evidence_catalog"])
    applied = _section(rendered, "Applied Memories")
    assert applied is not None and "fact" in applied.lower() and "stated by you" in applied
    assert [(m.kind, m.origin) for m in artifacts["applied_memories"]] == [("fact", "stated")]


def test_no_fact_means_no_framing_context_for_the_reporter(source, tmp_path, monkeypatch):
    _, reporter = _run(source, tmp_path, monkeypatch, InMemoryStore())
    assert "framing_facts" not in reporter.contexts[0]


def test_a_forgotten_fact_stops_reaching_either_agent(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    record = _say(store)
    MemoryStore(store).forget("priya", record.id)
    planner = ScriptedPlannerModel(crew_plan())
    result, reporter = _run(source, tmp_path, monkeypatch, store, planner_model=planner)
    assert FACT not in planner.prompts[0] and FACT not in str(reporter.contexts[0])
    assert result["artifacts"]["applied_memories"] == []


class TestResume:
    def _twice(self, source, tmp_path, monkeypatch, store, between):
        saver = MemorySaver()
        planner = ScriptedPlannerModel(crew_plan())
        _run(source, tmp_path, monkeypatch, store, thread_id="same", checkpointer=saver, planner_model=planner)
        between()
        reporter = ScriptedReporterModel()
        second, _ = _run(source, tmp_path, monkeypatch, store, thread_id="same", checkpointer=saver,
                         planner_model=planner, reporter_model=reporter)
        return second, planner, reporter

    def test_an_unchanged_fact_reuses_the_plan_and_the_report(self, source, tmp_path, monkeypatch):
        store = InMemoryStore()
        _say(store)
        second, planner, reporter = self._twice(source, tmp_path, monkeypatch, store, lambda: None)
        assert planner.calls == 1 and reporter.calls == 0
        assert {o["resume_reason"] for o in second["artifacts"]["step_outcomes"].values()} == {"unchanged"}

    def test_a_forgotten_fact_replans_and_rewrites_the_report_without_it(self, source, tmp_path, monkeypatch):
        store = InMemoryStore()
        record = _say(store)
        second, planner, reporter = self._twice(
            source, tmp_path, monkeypatch, store, lambda: MemoryStore(store).forget("priya", record.id))
        assert planner.calls == 2 and FACT not in planner.prompts[1]
        assert reporter.calls == 1 and "framing_facts" not in reporter.contexts[0]
        assert second["artifacts"]["report"].applied_memories == []

    def test_a_new_fact_replans_and_rewrites_the_report_with_it(self, source, tmp_path, monkeypatch):
        store = InMemoryStore()
        _say(store)
        second, planner, reporter = self._twice(
            source, tmp_path, monkeypatch, store, lambda: _say(store, "we sell mostly to schools"))
        assert planner.calls == 2 and "we sell mostly to schools" in planner.prompts[1]
        assert reporter.calls == 1 and "we sell mostly to schools" in str(reporter.contexts[0])
