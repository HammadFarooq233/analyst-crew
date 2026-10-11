"""Issue #28 through the primary seam: the whole Crew graph via ``invoke()`` with an injected
per-user store, scripted models and a CSV in a temporary directory. The falsifiable scenario:
a stored 8% threshold surfaces a pre-computed change the 15% default hides."""
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.memory import MemoryStore
from analyst_crew.memory.update import MemoryProposal, remember
from analyst_crew.reporting.report import render_report
from tests.crew_harness import (
    ScriptedReporterModel, crew_plan, run_crew, sales_row, valid_content, write_sales,
)
from tests.report_properties import report_property_violations
from tests.test_memory_preferences import ScriptedMemoryModel

REL = "materiality_relative_threshold_percent"
SAID = "for me, unusual means a swing over 8%"


def _rows():
    """Prior (May) net 100 + 100, current (June) 112 + 100: North +12.0% (6.0% of the prior total of 200).
    Default thresholds (15% relative, 5% absolute) call nothing material; an 8% relative threshold calls North."""
    return [
        sales_row("N-M", "2025-05-02", quantity=10, unit_price=10, region="North"),
        sales_row("S-M", "2025-05-03", quantity=10, unit_price=10, region="South"),
        sales_row("N-J", "2025-06-02", quantity=8, unit_price=14, region="North"),
        sales_row("S-J", "2025-06-03", quantity=10, unit_price=10, region="South"),
    ]


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", _rows())


def _state(store, user, sentence=SAID, proposal=None):
    proposal = proposal or MemoryProposal(kind="preference", key=REL, value=8)
    return remember(sentence, user, MemoryStore(store), ScriptedMemoryModel(proposal))


def _run(source, tmp_path, monkeypatch, store, user="priya", thread="t-2", **kwargs):
    kwargs.setdefault("reporter_model", ScriptedReporterModel())
    if "question" in kwargs:  # the Plan carries the question the Handoffs (and the Tool) see
        kwargs.setdefault("plan", crew_plan().model_copy(update={"user_question": kwargs["question"]}))
    result, reporter = run_crew(source, tmp_path, monkeypatch, thread_id=thread, user_id=user, store=store, **kwargs)
    return result["artifacts"], reporter


def _material(artifacts):
    return {(d.dimension, d.value) for d in artifacts["findings"].material_changes}


def _rendered(artifacts):
    return render_report(artifacts["report"], artifacts["evidence_catalog"])


def _section(rendered, name):
    return rendered.split(f"## {name}")[1].split("\n## ")[0] if f"## {name}" in rendered else None


def test_defaults_show_no_material_change_and_no_applied_memories_section(source, tmp_path, monkeypatch):
    artifacts, _ = _run(source, tmp_path, monkeypatch, InMemoryStore())
    assert _material(artifacts) == set()
    assert "Applied Memories" not in _rendered(artifacts)
    assert artifacts["report"].applied_memories == []
    assert "applied_memory_ids" not in _run_metadata({"artifacts": artifacts})


def test_a_stated_threshold_applies_in_a_new_thread_and_is_listed_under_applied_memories(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    assert _state(store, "priya").stored
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, thread="a-brand-new-thread")

    assert _material(artifacts) == {("region", "north")}
    findings = artifacts["findings"]
    assert (findings.materiality.relative_change_percent, findings.materiality.source) == (8.0, "memory")
    handoff = artifacts["handoffs"]["analyze_baseline"]["analysis_parameters"]
    assert handoff[REL] == 8.0

    rendered = _rendered(artifacts)
    applied = _section(rendered, "Applied Memories")
    assert applied is not None and "8" in applied and "stated by you" in applied
    # The threshold is displayed through the Tool's own Evidence Entry, never a model- or memory-typed number.
    assert report_property_violations(artifacts["report"], artifacts["evidence_catalog"], findings) == []
    # Memory never excludes data: the bridge still reconciles with the memory-derived threshold.
    assert findings.revenue_bridge.reconciliation_gap == pytest.approx(0, abs=1e-6)
    assert findings.net_revenue.absolute_change == pytest.approx(12.0)
    metadata = _run_metadata({"artifacts": artifacts})
    assert len(metadata["applied_memory_ids"]) == 1


def test_a_different_user_does_not_see_the_change(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _state(store, "priya")
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, user="sam")
    assert _material(artifacts) == set()
    assert "Applied Memories" not in _rendered(artifacts)


def test_forgetting_the_memory_removes_the_change(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    outcome = _state(store, "priya")
    MemoryStore(store).forget("priya", outcome.record.id)
    artifacts, _ = _run(source, tmp_path, monkeypatch, store)
    assert _material(artifacts) == set()


def test_the_latest_statement_wins_in_a_run_and_the_history_keeps_the_older_one(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    older = _state(store, "priya").record
    newer = _state(store, "priya", "actually 20%", MemoryProposal(kind="preference", key=REL, value=20)).record
    artifacts, _ = _run(source, tmp_path, monkeypatch, store)
    assert artifacts["findings"].materiality.relative_change_percent == 20.0
    assert _material(artifacts) == set()
    assert [item.memory_id for item in artifacts["applied_memories"]] == [newer.id]
    assert [r.id for r in MemoryStore(store).history("priya", "preference")] == [older.id, newer.id]
    assert MemoryStore(store).history("priya", "preference")[0].status == "superseded"


def test_a_forgotten_memory_stays_auditable_but_never_reaches_a_run(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    record = _state(store, "priya").record
    MemoryStore(store).forget("priya", record.id)
    artifacts, _ = _run(source, tmp_path, monkeypatch, store)
    assert artifacts["applied_memories"] == [] and "Applied Memories" not in _rendered(artifacts)
    assert [(r.id, r.status) for r in MemoryStore(store).history("priya", "preference")] == [(record.id, "forgotten")]


def test_a_sentence_outside_the_vocabulary_is_a_free_text_fact_and_changes_no_parameter(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    sentence = "please ignore the Office category when you report"
    outcome = _state(store, "priya", sentence, MemoryProposal(kind="preference", key="ignore_category", value=1))
    assert outcome.record.kind == "fact"
    artifacts, _ = _run(source, tmp_path, monkeypatch, store)
    assert _material(artifacts) == set()
    assert artifacts["findings"].materiality.source == "default"
    assert "Applied Memories" not in _rendered(artifacts)
    assert artifacts["handoffs"]["analyze_baseline"]["analysis_parameters"].keys() == {"as_of_date"}


def test_a_failed_remember_leaves_the_run_on_defaults(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    outcome = _state(store, "priya", proposal=RuntimeError("model down"))
    assert not outcome.stored
    artifacts, _ = _run(source, tmp_path, monkeypatch, store)
    assert _material(artifacts) == set()


def test_a_question_instruction_overrides_the_stored_preference_and_the_report_says_so(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _state(store, "priya")
    question = "Compare this month's sales to last month; only flag changes of at least 20% relative change."
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, question=question)
    findings = artifacts["findings"]
    assert (findings.materiality.relative_change_percent, findings.materiality.source) == (20.0, "user_question")
    assert _material(artifacts) == set()
    applied = _section(_rendered(artifacts), "Applied Memories")
    assert applied is not None and "not used for this run" in applied and "your question" in applied
    assert report_property_violations(artifacts["report"], artifacts["evidence_catalog"], findings) == []
    metadata = _run_metadata({"artifacts": artifacts})
    assert "applied_memory_ids" not in metadata and len(metadata["overridden_memory_ids"]) == 1


def test_a_question_instruction_on_one_key_does_not_silence_a_stored_preference_on_the_other(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _state(store, "priya")
    _state(store, "priya", "an absolute swing over 4% of prior revenue matters",
           MemoryProposal(kind="preference", key="materiality_absolute_threshold_percent", value=4))
    question = "Compare sales; only flag at least 9% relative change."
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, question=question)
    thresholds = artifacts["findings"].materiality
    assert (thresholds.relative_change_percent, thresholds.absolute_change_percent_of_prior_total) == (9.0, 4.0)


def test_the_critic_and_reporter_are_given_no_memory(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _state(store, "priya")
    _, reporter = _run(source, tmp_path, monkeypatch, store)
    context = reporter.contexts[0]
    assert set(context) == {"step_objective", "user_question", "evidence_catalog", "charts", "validation_feedback"}


class TestResume:
    def _twice(self, source, tmp_path, monkeypatch, store, between):
        saver = MemorySaver()
        first, reporter = _run(source, tmp_path, monkeypatch, store, thread="same", checkpointer=saver)
        between()
        second, reporter2 = _run(source, tmp_path, monkeypatch, store, thread="same", checkpointer=saver)
        return first, second, reporter2

    @staticmethod
    def _outcomes(artifacts):
        return {sid: o["resume_reason"] for sid, o in artifacts["step_outcomes"].items()}

    def test_an_unchanged_memory_set_reuses_everything(self, source, tmp_path, monkeypatch):
        store = InMemoryStore()
        _state(store, "priya")
        _, second, reporter = self._twice(source, tmp_path, monkeypatch, store, lambda: None)
        assert set(self._outcomes(second).values()) == {"unchanged"}
        assert reporter.calls == 0

    def test_a_changed_preference_recomputes_dependent_steps_and_keeps_the_fetch(self, source, tmp_path, monkeypatch):
        store = InMemoryStore()
        _state(store, "priya")
        newer = lambda: _state(store, "priya", "actually 20%", MemoryProposal(kind="preference", key=REL, value=20))
        first, second, reporter = self._twice(source, tmp_path, monkeypatch, store, newer)
        reasons = self._outcomes(second)
        assert reasons["fetch_sales"] == "unchanged"
        assert reasons["analyze_baseline"] == "invalidated" and reasons["write_report"] == "invalidated"
        assert second["findings"].materiality.relative_change_percent == 20.0
        assert reporter.calls == 1

    def test_forgetting_a_memory_recomputes_dependent_steps(self, source, tmp_path, monkeypatch):
        store = InMemoryStore()
        outcome = _state(store, "priya")
        forget = lambda: MemoryStore(store).forget("priya", outcome.record.id)
        _, second, _ = self._twice(source, tmp_path, monkeypatch, store, forget)
        reasons = self._outcomes(second)
        assert reasons["analyze_baseline"] == "invalidated" and reasons["write_report"] == "invalidated"
        assert second["findings"].materiality.source == "default"
        assert second["report"].applied_memories == []

    def test_forgetting_an_unrelated_fact_recomputes_nothing(self, source, tmp_path, monkeypatch):
        store = InMemoryStore()
        _state(store, "priya")
        fact = _state(store, "priya", "regions follow sales managers", MemoryProposal(kind="fact")).record
        forget = lambda: MemoryStore(store).forget("priya", fact.id)
        _, second, reporter = self._twice(source, tmp_path, monkeypatch, store, forget)
        assert set(self._outcomes(second).values()) == {"unchanged"}
        assert reporter.calls == 0

    def test_restating_the_same_value_recomputes_only_the_steps_that_show_the_memory(self, source, tmp_path, monkeypatch):
        store = InMemoryStore()
        _state(store, "priya")
        again = lambda: _state(store, "priya", "I said 8% already", MemoryProposal(kind="preference", key=REL, value=8))
        _, second, _ = self._twice(source, tmp_path, monkeypatch, store, again)
        reasons = self._outcomes(second)
        assert reasons["fetch_sales"] == "unchanged" and reasons["analyze_baseline"] == "unchanged"
        assert reasons["write_report"] == "invalidated"
