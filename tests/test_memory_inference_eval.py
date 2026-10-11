"""Issue #32: the inference eval. It measures over-extraction (memories proposed where the user said
nothing worth keeping) and hallucinated extraction (memories whose evidence or numbers are not in
the input), by properties of the proposals, never by their wording. It follows the pattern of
``tests/test_critic_eval.py``: offline guards run in the default suite and prove the property
checks can fail; the live tests carry ``@pytest.mark.eval`` and need provider credentials.
"""
import re
from dataclasses import dataclass

import pytest
from langgraph.store.memory import InMemoryStore

from analyst_crew.llm import credentials_available
from analyst_crew.memory import MemoryStore
from analyst_crew.memory.infer import (
    MAX_FACTS_PER_RUN, MAX_PREFERENCES_PER_RUN, infer, production_inference_model, propose,
)
from analyst_crew.memory.models import InferredProposal
from analyst_crew.memory.vocabulary import PREFERENCE_KEYS
from tests.test_memory_inference import REL, ScriptedInferenceModel

RUNS_PER_CASE = 3
RUN_RECORDS = ("Previous successful run compared 2025-06-01 to 2025-06-30 with 2025-05-01 to 2025-05-31 (month). "
               "Net revenue 40.00 against 30.00 (+33.3%). Material Changes: 0 (none) at thresholds 15% relative and "
               "5% absolute. Rows: 4 included, 0 quarantined, 0 duplicates. Steps run: Fetcher, Analyst, Reporter.")


@dataclass(frozen=True)
class Case:
    name: str
    words: str
    expect: str  # "nothing", "fact" or "preference"


CASES = (
    Case("plain question", "How did sales change from May to June?", "nothing"),
    Case("ranking question", "Which region grew fastest last month?", "nothing"),
    Case("one-off request", "Just for this report, keep it short. What was June net revenue?", "nothing"),
    Case("stated organisation", "Our regions follow the sales managers, so group the story that way. "
                                "How did sales change?", "fact"),
    Case("stated threshold", "For me, unusual means a swing over 8%. How did sales change from May to June?",
         "preference"),
)


def _norm(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _numbers(text):
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in re.findall(r"\d+(?:\.\d+)?", text or "")}


def inference_violations(case: Case, proposals: list[InferredProposal]) -> list[str]:
    """Properties a proposal set must satisfy. Nothing here looks at how a proposal is worded."""
    problems: list[str] = []
    words, source = _norm(case.words), _norm(case.words + " " + RUN_RECORDS)
    if case.expect == "nothing" and proposals:
        problems.append(f"over-extraction: {len(proposals)} proposal(s) where nothing durable was said")
    if len([p for p in proposals if p.kind == "fact"]) > MAX_FACTS_PER_RUN:
        problems.append("over-extraction: more facts than one run may store")
    if len([p for p in proposals if p.kind == "preference"]) > MAX_PREFERENCES_PER_RUN:
        problems.append("over-extraction: more preferences than one run may store")
    for proposal in proposals:
        evidence = _norm(proposal.evidence)
        allowed = words if proposal.kind == "preference" else source
        if not evidence or evidence not in allowed:
            problems.append(f"hallucinated {proposal.kind}: evidence is not in the allowed source text")
        if not _numbers(proposal.content) <= _numbers(case.words + " " + RUN_RECORDS):
            problems.append(f"hallucinated {proposal.kind}: content has a number absent from the input")
        if proposal.kind == "preference":
            if proposal.key not in PREFERENCE_KEYS:
                problems.append("hallucinated preference: key is outside the vocabulary")
            elif not _numbers(str(proposal.value)) <= _numbers(case.words):
                problems.append("hallucinated preference: value is not in the user's words")
    if case.expect in ("fact", "preference") and not any(p.kind == case.expect for p in proposals):
        problems.append(f"missed extraction: no {case.expect} proposed for a plainly stated one")
    return problems


def stored_count(case: Case, proposals) -> int:
    """How many memories the whole step (model plus code checks) would store or leave pending."""
    store = InMemoryStore()
    memory = MemoryStore(store)
    model = ScriptedInferenceModel(*proposals)
    result = infer(model, case.words, RUN_RECORDS, "eval", memory)
    return len(result.facts) + len(result.pending)


def _good(case):
    if case.expect == "fact":
        return [InferredProposal(kind="fact", content="regions follow the sales managers",
                                 evidence="Our regions follow the sales managers")]
    if case.expect == "preference":
        return [InferredProposal(kind="preference", content="threshold", key=REL, value=8,
                                 evidence="unusual means a swing over 8%")]
    return []


# ---- offline guards: the property checks can pass and can fail ------------------------

@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_well_behaved_proposals_satisfy_every_property(case):
    assert inference_violations(case, _good(case)) == []


def test_over_extraction_is_detected_for_neutral_input():
    neutral = CASES[0]
    junk = [InferredProposal(kind="fact", content="the user asks about sales", evidence="How did sales change")]
    assert any("over-extraction" in p for p in inference_violations(neutral, junk))


def test_hallucinated_evidence_numbers_and_keys_are_detected():
    case = CASES[4]
    bad = [
        InferredProposal(kind="fact", content="north grew 41% in June", evidence="invented quote"),
        InferredProposal(kind="preference", content="x", key="exclude_category", value=12, evidence="swing over 8%"),
    ]
    problems = " ".join(inference_violations(case, bad))
    assert "evidence is not in" in problems and "number absent" in problems
    assert "outside the vocabulary" in problems or "value is not in the user's words" in problems


def test_a_missed_plainly_stated_preference_is_detected():
    assert any("missed extraction" in p for p in inference_violations(CASES[4], []))


def test_a_proposer_that_invents_things_stores_nothing_for_neutral_input():
    neutral = CASES[0]
    junk = [InferredProposal(kind="fact", content="sales are a big deal", evidence="a quote nobody said"),
            InferredProposal(kind="preference", content="x", key=REL, value=8, evidence="swing over 8%")]
    assert stored_count(neutral, junk) == 0


def test_a_proposer_that_restates_the_question_is_still_capped_to_grounded_facts():
    case = CASES[3]
    flood = [InferredProposal(kind="fact", content=f"regions follow the sales managers {chr(97 + i)}",
                              evidence="Our regions follow the sales managers") for i in range(8)]
    assert stored_count(case, flood) == MAX_FACTS_PER_RUN


# ---- live eval -----------------------------------------------------------------------

def _skip_without_llm():
    if not credentials_available():
        pytest.skip("LLM credentials not set")


@pytest.mark.eval
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
def test_the_real_proposer_neither_over_extracts_nor_hallucinates(case):
    _skip_without_llm()
    model = production_inference_model()
    for _ in range(RUNS_PER_CASE):
        proposals = propose(model, case.words, RUN_RECORDS)
        assert inference_violations(case, proposals) == []
        if case.expect == "nothing":
            assert stored_count(case, proposals) == 0
