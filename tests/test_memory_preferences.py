"""Issue #28, pure seams: the closed Typed Preference vocabulary, resolution into
analysis parameters, and the Memory Update (`remember`) step with a scripted model."""
import pytest
from langgraph.store.memory import InMemoryStore

from analyst_crew.memory import MemoryRecord, MemoryStore, UnavailableStore
from analyst_crew.memory.resolve import resolve_preferences
from analyst_crew.memory.update import MemoryProposal, remember
from analyst_crew.memory.vocabulary import PREFERENCE_KEYS, PreferenceError, validate_preference

REL = "materiality_relative_threshold_percent"
ABS = "materiality_absolute_threshold_percent"


def _pref(key, value, *, id="mem-1", created="2026-01-01T00:00:00+00:00"):
    return MemoryRecord(id=id, user_id="u", kind="preference", content=f"{key}={value}", created_at=created,
                        key=key, value=value)


# ---- vocabulary -------------------------------------------------------------

def test_vocabulary_is_the_two_threshold_keys():
    assert set(PREFERENCE_KEYS) == {REL, ABS}


@pytest.mark.parametrize("raw,expected", [(8, 8.0), (8.5, 8.5), ("8", 8.0), ("8%", 8.0), (100, 100.0)])
def test_valid_values_are_accepted(raw, expected):
    assert validate_preference(REL, raw) == expected


@pytest.mark.parametrize("key,raw", [
    (REL, 0), (REL, -3), (REL, 100000), (REL, float("nan")), (REL, True), (REL, None), (REL, "lots"), (REL, ""),
    (ABS, 101), ("grouping_dimension", "region"), ("ignore_category", 1),
])
def test_invalid_keys_and_values_are_rejected_with_a_reason(key, raw):
    with pytest.raises(PreferenceError) as excinfo:
        validate_preference(key, raw)
    assert str(excinfo.value)


# ---- resolution -------------------------------------------------------------

def test_a_stored_threshold_resolves_into_analysis_parameters_and_is_recorded_as_applied():
    resolution = resolve_preferences([_pref(REL, 8.0)], "How did sales change?")
    assert resolution.parameters == {REL: 8.0, "materiality_source": "memory"}
    assert [(a.memory_id, a.key, a.status) for a in resolution.applied] == [("mem-1", REL, "applied")]


def test_no_preferences_resolve_to_nothing():
    resolution = resolve_preferences([], "How did sales change?")
    assert resolution.parameters == {} and resolution.applied == []


def test_the_latest_statement_for_a_key_wins_and_only_it_is_applied():
    older = _pref(REL, 10.0, id="mem-old", created="2026-01-01T00:00:00+00:00")
    newer = _pref(REL, 20.0, id="mem-new", created="2026-02-01T00:00:00+00:00")
    resolution = resolve_preferences([newer, older], "q")
    assert resolution.parameters[REL] == 20.0
    assert [a.memory_id for a in resolution.applied] == ["mem-new"]


def test_a_question_instruction_overrides_the_stored_preference_and_is_disclosed():
    resolution = resolve_preferences([_pref(REL, 8.0)], "Flag changes with at least 25% relative change.")
    assert resolution.parameters == {}  # nothing memory-derived is applied; the question parses natively
    assert [(a.key, a.status) for a in resolution.applied] == [(REL, "overridden")]


def test_the_question_wins_per_key_while_other_stored_keys_still_apply():
    records = [_pref(REL, 8.0, id="r"), _pref(ABS, 3.0, id="a")]
    resolution = resolve_preferences(records, "Flag changes with at least 25% relative change.")
    assert resolution.parameters[REL] == 25.0
    assert resolution.parameters[ABS] == 3.0
    assert {(a.key, a.status) for a in resolution.applied} == {(REL, "overridden"), (ABS, "applied")}


def test_a_record_that_no_longer_validates_is_never_applied():
    assert resolve_preferences([_pref(REL, -5.0)], "q").parameters == {}
    assert resolve_preferences([_pref("ignore_category", 1.0)], "q").parameters == {}


# ---- remember (Memory Update) ----------------------------------------------

class ScriptedMemoryModel:
    """Stands in for the proposing LLM: returns a canned proposal or raises."""

    def __init__(self, proposal):
        self.proposal = proposal
        self.sentences = []

    def with_structured_output(self, *_a, **_k):
        return self

    def invoke(self, messages, *_a, **_k):
        self.sentences.append(messages[-1].content)
        if isinstance(self.proposal, Exception):
            raise self.proposal
        return self.proposal


def _remember(sentence, proposal, store=None, user="priya"):
    store = store if store is not None else InMemoryStore()
    outcome = remember(sentence, user, MemoryStore(store), ScriptedMemoryModel(proposal))
    return outcome, store


SAID = "for me, unusual means a swing over 8%"


def test_a_valid_preference_is_stored_as_a_typed_preference_and_interpretation_is_reported():
    outcome, store = _remember(SAID, MemoryProposal(kind="preference", key=REL, value=8))
    assert outcome.stored and outcome.record.kind == "preference"
    [record] = MemoryStore(store).list("priya", "preference")
    assert (record.key, record.value, record.origin) == (REL, 8.0, "stated")
    assert "8" in outcome.message and "relative" in outcome.message.lower()


@pytest.mark.parametrize("proposal", [
    MemoryProposal(kind="preference", key="ignore_category", value=1),
    MemoryProposal(kind="preference", key=REL, value=-4),
    MemoryProposal(kind="preference", key=REL, value=900000),
    MemoryProposal(kind="preference", key=REL, value=12),  # 12 never appears in the sentence
    MemoryProposal(kind="preference", key=REL, value=None),
])
def test_a_preference_that_does_not_validate_degrades_to_a_free_text_fact(proposal):
    outcome, store = _remember(SAID, proposal)
    assert outcome.stored and outcome.record.kind == "fact" and outcome.record.value is None
    memory = MemoryStore(store)
    assert memory.list("priya", "preference") == []
    assert [r.content for r in memory.list("priya", "fact")] == [SAID]
    assert "free-text" in outcome.message.lower()


def test_a_fact_proposal_is_stored_as_a_fact_with_the_users_own_words():
    outcome, store = _remember("our regions follow sales managers", MemoryProposal(kind="fact"))
    assert outcome.record.kind == "fact"
    assert [r.content for r in MemoryStore(store).list("priya", "fact")] == ["our regions follow sales managers"]


@pytest.mark.parametrize("proposal", [RuntimeError("model down"), "not a proposal", None, {"kind": "banana"}])
def test_an_unusable_proposal_stores_nothing_and_says_why(proposal):
    outcome, store = _remember(SAID, proposal)
    assert not outcome.stored and outcome.record is None and outcome.message
    assert MemoryStore(store).list("priya") == []


def test_a_blank_sentence_stores_nothing():
    outcome, store = _remember("   ", MemoryProposal(kind="fact"))
    assert not outcome.stored and MemoryStore(store).list("priya") == []


def test_a_storage_outage_stores_nothing_and_says_so():
    outcome = remember(SAID, "priya", MemoryStore(UnavailableStore()),
                       ScriptedMemoryModel(MemoryProposal(kind="preference", key=REL, value=8)))
    assert not outcome.stored and "unavailable" in outcome.message.lower()


def test_a_dict_proposal_is_validated_like_a_model_proposal():
    outcome, _ = _remember(SAID, {"kind": "preference", "key": REL, "value": 8})
    assert outcome.stored and outcome.record.kind == "preference"
