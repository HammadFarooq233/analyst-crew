"""Spec (#23): applying memory never changes any displayed value except through the resolved parameters.
Primary seam: the Crew graph via ``invoke()`` with scripted models, compared run against run."""
from langgraph.store.memory import InMemoryStore

from tests.report_properties import displayed_value_differences, report_property_violations
from tests.test_memory_fact_crew import FACT, _engines, _say, source  # noqa: F401 — fixtures reused
from tests.test_memory_preference_crew import _run, _state


def _violations(artifacts):
    return report_property_violations(artifacts["report"], artifacts["evidence_catalog"], artifacts["findings"])


def test_a_free_text_fact_changes_no_displayed_value(source, tmp_path, monkeypatch):
    without, _ = _run(source, tmp_path, monkeypatch, InMemoryStore(), thread="a")
    store = InMemoryStore()
    _say(store, FACT, user="priya")
    with_fact, reporter = _run(source, tmp_path, monkeypatch, store, thread="b")
    assert with_fact["applied_memories"] and FACT in str(reporter.contexts[0]), "the fact was in play"
    assert displayed_value_differences(without, with_fact) == []
    assert _violations(with_fact) == []


def test_a_threshold_preference_differs_only_through_the_resolved_parameter(source, tmp_path, monkeypatch):
    store = InMemoryStore()
    _state(store, "priya")
    baseline, _ = _run(source, tmp_path, monkeypatch, InMemoryStore(), thread="a")
    remembered, _ = _run(source, tmp_path, monkeypatch, store, thread="b")
    assert displayed_value_differences(baseline, remembered), "the threshold changed what is flagged"

    # The same parameter handed over explicitly, with no memory at all, shows exactly the same values.
    question = "Compare this month's sales to last month; only flag changes of at least 8% relative change."
    explicit, _ = _run(source, tmp_path, monkeypatch, InMemoryStore(), thread="c", question=question)
    assert remembered["findings"].materiality.relative_change_percent == 8.0
    assert explicit["findings"].materiality.relative_change_percent == 8.0
    assert displayed_value_differences(explicit, remembered) == []
    assert _violations(remembered) == []
