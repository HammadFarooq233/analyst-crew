"""Issue #32 through the primary seam: the whole Crew graph via ``invoke()`` with an injected store
and a scripted inference model. An inferred fact applies at once and is marked inferred; an
inferred Typed Preference changes no number until confirmed; nothing is inferred from Agent prose."""
import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.memory import MemoryStore
from analyst_crew.memory.models import InferenceResult
from analyst_crew.reporting.report import render_report
from tests.crew_harness import ScriptedReporterModel, run_crew, valid_content
from tests.report_properties import report_property_violations
from tests.test_memory_inference import REL, ScriptedInferenceModel, fact, pref
from tests.test_memory_preference_crew import _material, _rendered, _rows, _section
from tests.crew_harness import sales_row, write_sales

QUESTION = ("Our regions follow the sales managers, and for me unusual means a swing over 8%. "
            "How did sales change from May to June?")
FACT_ = fact()


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", _rows())


def _run(source, tmp_path, monkeypatch, store, model=None, thread="t-1", user="priya", question=QUESTION, **kwargs):
    kwargs.setdefault("reporter_model", ScriptedReporterModel())
    result, reporter = run_crew(source, tmp_path, monkeypatch, thread_id=thread, user_id=user, store=store,
                                question=question, inference_model=model, **kwargs)
    return result["artifacts"], reporter


def _applied(artifacts):
    """The memories applied other than Past-run Summaries (which earlier runs write on their own)."""
    return [m for m in artifacts["applied_memories"] if m.kind != "summary"]


def test_without_an_inference_model_nothing_is_inferred(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    artifacts, _ = _run(source, tmp_path, monkeypatch, store)
    assert artifacts["inference"].status == "skipped"
    assert MemoryStore(store).history("priya", "fact") == [] and MemoryStore(store).pending("priya") == []


def test_an_inferred_fact_applies_on_the_next_run_marked_inferred_under_applied_memories(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    first, _ = _run(source, tmp_path, monkeypatch, store, ScriptedInferenceModel(FACT_))
    assert isinstance(first["inference"], InferenceResult) and len(first["inference"].facts) == 1
    (record,) = MemoryStore(store).list("priya", "fact")
    assert record.origin == "inferred"

    second, reporter = _run(source, tmp_path, monkeypatch, store, thread="t-2", question="How did sales change?")
    assert [(m.memory_id, m.kind, m.origin) for m in _applied(second)] == [(record.id, "fact", "inferred")]
    applied = _section(_rendered(second), "Applied Memories")
    assert applied is not None and "inferred" in applied and "change no figure" in applied
    assert report_property_violations(second["report"], second["evidence_catalog"], second["findings"]) == []
    assert record.content in str(reporter.contexts[0])
    assert record.id in _run_metadata({"artifacts": second})["applied_memory_ids"]


def test_an_inferred_preference_is_pending_and_changes_no_number_until_confirmed(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    first, _ = _run(source, tmp_path, monkeypatch, store, ScriptedInferenceModel(pref()))
    (pending,) = MemoryStore(store).pending("priya")
    assert first["inference"].pending == [pending.id]

    second, _ = _run(source, tmp_path, monkeypatch, store, thread="t-2", question="How did sales change?")
    assert _material(second) == set()
    assert second["findings"].materiality.relative_change_percent == 15.0
    assert _applied(second) == [] and "inferred" not in _rendered(second)

    MemoryStore(store).confirm("priya", pending.id)
    third, _ = _run(source, tmp_path, monkeypatch, store, thread="t-3", question="How did sales change?")
    assert _material(third) == {("region", "north")}
    assert third["findings"].materiality.relative_change_percent == 8.0
    applied = _section(_rendered(third), "Applied Memories")
    assert applied is not None and "inferred" in applied


def test_a_rejected_preference_never_applies(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _run(source, tmp_path, monkeypatch, store, ScriptedInferenceModel(pref()))
    memory = MemoryStore(store)
    memory.reject("priya", memory.pending("priya")[0].id)
    later, _ = _run(source, tmp_path, monkeypatch, store, thread="t-2", question="How did sales change?")
    assert _material(later) == set() and _applied(later) == []


def test_nothing_is_inferred_from_agent_prose(source, tmp_path, monkeypatch):
    def prosey(context):
        content = valid_content(context)
        content.summary.text = "ZEBRA-PROSE glorious quarter. {} ".format(content.summary.text)
        return content

    model = ScriptedInferenceModel(fact("the quarter was glorious", "ZEBRA-PROSE glorious quarter"))
    store = InMemoryStore()
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, model, reporter_model=ScriptedReporterModel(prosey))
    assert "ZEBRA" in artifacts["report"].summary.text
    assert all("ZEBRA" not in shown for shown in model.shown) and model.shown  # the proposer never saw it
    assert MemoryStore(store).history("priya", "fact") == []  # and a fact quoting it is not grounded
    assert len(artifacts["inference"].rejected) == 1


def test_the_proposer_is_given_the_users_question_and_the_run_records(source, tmp_path, monkeypatch):
    model = ScriptedInferenceModel()
    _run(source, tmp_path, monkeypatch, InMemoryStore(), model)
    (shown,) = model.shown
    assert "unusual means a swing over 8%" in shown and "Previous successful run compared" in shown


def test_inference_is_per_user(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _run(source, tmp_path, monkeypatch, store, ScriptedInferenceModel(FACT_, pref()))
    assert MemoryStore(store).history("sam") == []


def test_a_resumed_thread_does_not_infer_twice(source, tmp_path, monkeypatch):
    store, saver = InMemoryStore(), MemorySaver()
    for _ in range(2):
        _run(source, tmp_path, monkeypatch, store, ScriptedInferenceModel(FACT_, pref()), checkpointer=saver)
    memory = MemoryStore(store)
    assert len(memory.list("priya", "fact")) == 1 and len(memory.pending("priya")) == 1


def test_a_failing_inference_model_does_not_fail_the_run(source, tmp_path, monkeypatch):
    model = ScriptedInferenceModel()
    model.failure = RuntimeError("down")
    artifacts, _ = _run(source, tmp_path, monkeypatch, InMemoryStore(), model)
    assert artifacts["report"] is not None and artifacts["inference"].status == "failed"


def test_the_run_metadata_marks_inferred_writes(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, ScriptedInferenceModel(FACT_, pref()))
    metadata = _run_metadata({"artifacts": artifacts})
    assert metadata["inferred_memory_ids"] == artifacts["inference"].facts
    assert metadata["pending_memory_ids"] == artifacts["inference"].pending


def test_the_run_metadata_records_the_memory_update_outcome_and_written_id(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    artifacts, _ = _run(source, tmp_path, monkeypatch, store, ScriptedInferenceModel())
    metadata = _run_metadata({"artifacts": artifacts})
    written = [r.id for r in MemoryStore(store).list("priya", "summary")]
    assert metadata["memory_update"] == "written"
    assert metadata["memory_update_ids"] == written and len(written) == 1


def test_the_run_metadata_records_a_memory_update_that_wrote_nothing():
    metadata = _run_metadata({"artifacts": {"memory_update": {"status": "unavailable"}}})
    assert metadata["memory_update"] == "unavailable" and "memory_update_ids" not in metadata
