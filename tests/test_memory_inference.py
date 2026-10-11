"""Issue #32, pure seams: the Memory Update step's inference from the user's own words and run
records. A model proposes; code grounds and validates. Inferred facts apply at once (marked
inferred); inferred Typed Preferences are stored pending and change no number until confirmed."""
import pytest
from langgraph.store.memory import InMemoryStore

from analyst_crew.memory import MemoryNotFound, MemoryStore, UnavailableStore
from analyst_crew.memory.infer import infer
from analyst_crew.memory.models import InferenceProposals, InferredProposal
from analyst_crew.memory.resolve import resolve_preferences

REL = "materiality_relative_threshold_percent"
WORDS = "Our regions follow the sales managers, and for me unusual means a swing over 8%. How did sales change?"


class ScriptedInferenceModel:
    """Stands in for the proposing LLM; records everything it was shown."""

    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.failure = None
        self.shown = []

    def with_structured_output(self, *_a, **_k):
        return self

    def invoke(self, messages, *_a, **_k):
        self.shown.append("\n".join(str(m.content) for m in messages))
        if self.failure is not None:
            raise self.failure
        return InferenceProposals(proposals=self.proposals)


def fact(content="regions follow the sales managers", evidence="regions follow the sales managers"):
    return InferredProposal(kind="fact", content=content, evidence=evidence)


def pref(value=8, evidence="unusual means a swing over 8%", key=REL):
    return InferredProposal(kind="preference", content="unusual threshold", evidence=evidence, key=key, value=value)


def _infer(proposals, store=None, words=WORDS, records="", user="priya"):
    store = store if store is not None else InMemoryStore()
    model = ScriptedInferenceModel(*proposals)
    return infer(model, words, records, user, MemoryStore(store)), store, model


def test_a_grounded_fact_is_stored_active_and_marked_inferred():
    result, store, _ = _infer([fact()])
    (record,) = MemoryStore(store).list("priya", "fact")
    assert (record.origin, record.status) == ("inferred", "active")
    assert result.facts == [record.id] and result.status == "written"


def test_a_grounded_preference_is_pending_and_resolves_to_no_parameter():
    result, store, _ = _infer([pref()])
    memory = MemoryStore(store)
    assert memory.read_for_role("Analyst", "priya") == [] and memory.list("priya") == []
    (pending,) = memory.pending("priya")
    assert (pending.key, pending.value, pending.origin, pending.status) == (REL, 8.0, "inferred", "pending")
    assert result.pending == [pending.id]
    assert resolve_preferences(memory.read_for_role("Analyst", "priya"), "q").parameters == {}


@pytest.mark.parametrize("proposal", [
    fact(evidence="the invented quarter was glorious"),
    fact(content="north grew 12% every year", evidence="regions follow the sales managers"),
    fact(content="  ", evidence="regions follow"),
    pref(evidence="unusual means a swing over 9%", value=9),
    pref(value=12, evidence="unusual means a swing over 8%"),
    pref(key="exclude_category", value=8),
    pref(value=-3, evidence="swing over 8%"),
])
def test_ungrounded_or_invalid_proposals_store_nothing_and_say_why(proposal):
    result, store, _ = _infer([proposal])
    assert MemoryStore(store).history("priya") == []
    assert result.facts == [] and result.pending == [] and len(result.rejected) == 1


def test_a_fact_may_be_grounded_in_the_run_records_but_a_preference_may_not():
    records = "Previous successful run compared May with June. Rows: 4 included."
    result, store, _ = _infer([fact("the last run covered May and June", "compared May with June"),
                               pref(4, "Rows: 4 included")], words="How did sales change?", records=records)
    assert len(result.facts) == 1 and result.pending == [] and len(result.rejected) == 1


def test_the_proposer_sees_only_the_users_words_and_the_run_records():
    _, _, model = _infer([], records="RUNREC-DIGEST")
    (shown,) = model.shown
    assert "for me unusual means a swing over 8%" in shown and "RUNREC-DIGEST" in shown


def test_over_extraction_is_capped():
    many = [fact(f"regions follow the sales managers {chr(97 + i)}", "regions follow the sales managers")
            for i in range(9)]
    result, store, _ = _infer(many)
    assert len(result.facts) == 3 and len(MemoryStore(store).list("priya", "fact")) == 3


def test_repeating_the_same_words_does_not_duplicate_and_a_rejected_proposal_is_not_reproposed():
    store = InMemoryStore()
    _infer([fact(), pref()], store)
    again, _, _ = _infer([fact(), pref()], store)
    memory = MemoryStore(store)
    assert len(memory.list("priya", "fact")) == 1 and len(memory.pending("priya")) == 1
    assert again.facts == [] and again.pending == []
    memory.reject("priya", memory.pending("priya")[0].id)
    third, _, _ = _infer([pref()], store)
    assert third.pending == [] and memory.pending("priya") == []


def test_an_active_preference_with_the_same_value_is_not_proposed_again():
    store = InMemoryStore()
    MemoryStore(store).add("priya", "preference", "x", key=REL, value=8.0)
    result, _, _ = _infer([pref()], store)
    assert result.pending == []


def test_a_model_failure_stores_nothing_and_is_reported():
    store = InMemoryStore()
    model = ScriptedInferenceModel()
    model.failure = RuntimeError("model down")
    result = infer(model, WORDS, "", "priya", MemoryStore(store))
    assert result.status == "failed" and MemoryStore(store).history("priya") == []


def test_an_unreachable_store_is_reported_not_raised():
    result = infer(ScriptedInferenceModel(fact()), WORDS, "", "priya", MemoryStore(UnavailableStore()))
    assert result.status == "unavailable"


# ---- confirming and rejecting -------------------------------------------------

def test_confirming_activates_the_preference_and_supersedes_the_older_one():
    store = InMemoryStore()
    memory = MemoryStore(store)
    old = memory.add("priya", "preference", "old", key=REL, value=10.0)
    _infer([pref()], store)
    (pending,) = memory.pending("priya")
    confirmed = memory.confirm("priya", pending.id)
    assert confirmed.status == "active" and confirmed.origin == "inferred" and confirmed.supersedes == old.id
    assert memory.pending("priya") == []
    assert [r.id for r in memory.list("priya", "preference")] == [pending.id]
    parameters = resolve_preferences(memory.read_for_role("Analyst", "priya"), "q").parameters
    assert parameters[REL] == 8.0
    assert {r.id: r.status for r in memory.history("priya", "preference")}[old.id] == "superseded"


def test_rejecting_keeps_the_history_and_applies_nothing():
    store = InMemoryStore()
    _infer([pref()], store)
    memory = MemoryStore(store)
    (pending,) = memory.pending("priya")
    assert memory.reject("priya", pending.id).status == "rejected"
    assert memory.pending("priya") == [] and memory.read_for_role("Analyst", "priya") == []
    assert [r.status for r in memory.history("priya")] == ["rejected"]


def test_confirm_and_reject_only_reach_the_users_own_pending_preferences():
    store = InMemoryStore()
    _infer([pref(), fact()], store)
    memory = MemoryStore(store)
    (pending,) = memory.pending("priya")
    (active_fact,) = memory.list("priya", "fact")
    for method in (memory.confirm, memory.reject):
        with pytest.raises(MemoryNotFound):
            method("sam", pending.id)
        with pytest.raises(MemoryNotFound):
            method("priya", active_fact.id)
        with pytest.raises(MemoryNotFound):
            method("priya", "mem-nope")
    assert memory.pending("priya") == [pending]
