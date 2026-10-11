"""Automatic inference in the Crew-level Memory Update step (ADR-0005). Not an Agent.

Source is limited to the user's own words and the Crew's run records (the code-built digest of
Findings); never free-form Agent prose, which is never passed in. A model proposes candidates;
code grounds each one before anything is stored:

- every proposal must quote its evidence, and the quote must appear in the allowed source text
  (a Typed Preference only in the user's own words);
- a number in a fact or preference must appear in that evidence;
- a Typed Preference must pass the closed vocabulary's validation.

Inferred Free-text Facts are stored active and marked inferred. Inferred Typed Preferences are
stored pending and change no number until the user confirms them through the CLI.
"""
from __future__ import annotations

import re
from typing import Any

from analyst_crew.memory.models import InferenceProposals, InferenceResult, InferredProposal
from analyst_crew.memory.store import MemoryStore, MemoryUnavailable
from analyst_crew.memory.update import number_in
from analyst_crew.memory.vocabulary import PREFERENCE_KEYS, PreferenceError, validate_preference

MAX_FACTS_PER_RUN = 3
MAX_PREFERENCES_PER_RUN = 2
MAX_FACT_LENGTH = 300

INFERENCE_PROMPT = """You look for durable, user-stated context worth remembering in what a user said and in the system's run records.
Propose a memory ONLY when the user's own words plainly state it. Most requests state nothing worth remembering: then return no proposals.
The run records are shown only as background. They are already stored separately as past-run summaries, so NEVER copy, restate or summarise them, or any figure in them, as a memory.
Return JSON {{"proposals": [...]}}; each proposal has kind ("fact" or "preference"), content, evidence, and for a preference key and value.
- evidence must be copied word for word from the text you were given. Never paraphrase it and never invent it.
- kind "preference" ONLY when the user states a numeric threshold for one of these keys; value is that number:
{vocabulary}
- kind "fact" for lasting context such as how regions are organised. A question is not a fact, and neither is an instruction about this one report (its length, format or focus, "just for now", "this time"). Do not restate numbers the text does not contain.
At most {facts} facts and {preferences} preferences."""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _numbers(text: str) -> set[str]:
    return {n.rstrip("0").rstrip(".") if "." in n else n for n in re.findall(r"\d+(?:\.\d+)?", text or "")}


_FUNCTION_WORDS = frozenset(
    "the and for with that this these those our your their its are was were been being has have had not but "
    "from into than then also who whom whose which when where while will would can could should may might "
    "all any each every some such per via".split())


def _content_words(text: str) -> set[str]:
    """Lower-cased words of three or more letters, minus function words, with a trailing plural 's' dropped."""
    words = re.findall(r"[a-z][a-z'-]*", (text or "").lower())
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in words if len(w) >= 3 and w not in _FUNCTION_WORDS}


def production_inference_model() -> Any:
    """The live proposer seam. Tests and the eval inject a scripted or chosen model instead."""
    from analyst_crew.llm import make_chat_model

    return make_chat_model(temperature=0)


def propose(model: Any, user_words: str, run_records: str) -> list[InferredProposal]:
    """Ask the model for candidates. It is shown the user's words and the run records, nothing else."""
    from langchain_core.messages import HumanMessage, SystemMessage

    vocabulary = "\n".join(f"- {spec.key}: {spec.label}" for spec in PREFERENCE_KEYS.values())
    system = INFERENCE_PROMPT.format(
        vocabulary=vocabulary, facts=MAX_FACTS_PER_RUN, preferences=MAX_PREFERENCES_PER_RUN)
    human = f"What the user said:\n{user_words}\n\nRun records:\n{run_records or '(none)'}"
    raw = model.with_structured_output(InferenceProposals).invoke(
        [SystemMessage(content=system), HumanMessage(content=human)])
    if isinstance(raw, dict):
        raw = InferenceProposals.model_validate(raw)
    if not isinstance(raw, InferenceProposals):
        raise ValueError(f"the model returned {type(raw).__name__}, not proposals")
    return raw.proposals


def _grounding_problem(proposal: InferredProposal, words: str, records: str) -> str | None:
    evidence = _norm(proposal.evidence)
    if not evidence:
        return "no evidence was quoted"
    sources = [words] if proposal.kind == "preference" else [words, records]
    if not any(evidence in source for source in sources):
        where = "the user's own words" if proposal.kind == "preference" else "the user's words or the run records"
        return f"the quoted evidence does not appear in {where}"
    return None


def infer(model: Any, user_words: str, run_records: str, user_id: str, memory: MemoryStore) -> InferenceResult:
    """Run one inference pass. Never raises: a model failure or an outage stores nothing and says so."""
    try:
        proposals = propose(model, user_words, run_records)
    except Exception as exc:  # noqa: BLE001 — a model failure must never store anything
        return InferenceResult(status="failed", rejected=[f"the model could not propose ({type(exc).__name__})"])

    words, records = _norm(user_words), _norm(run_records)
    result = InferenceResult()
    try:
        history = memory.history(user_id)
        known_facts = {_norm(r.content) for r in history if r.kind == "fact"}
        known_prefs = {(r.key, r.value) for r in history if r.kind == "preference"}
        for proposal in proposals:
            problem = _grounding_problem(proposal, words, records)
            if problem is None and proposal.kind == "fact":
                content = " ".join(proposal.content.split())
                if not content or len(content) > MAX_FACT_LENGTH:
                    problem = "the fact is empty or too long"
                elif not _numbers(content) <= _numbers(proposal.evidence):
                    problem = "the fact states a number its evidence does not contain"
                elif not _content_words(content) <= _content_words(proposal.evidence):
                    problem = "the fact adds words its evidence does not contain"
                elif _norm(content) in known_facts:
                    continue  # already known (or deliberately forgotten): not stored again
                elif len(result.facts) >= MAX_FACTS_PER_RUN:
                    problem = "too many facts proposed for one run"
                else:
                    record = memory.add(user_id, "fact", content, origin="inferred")
                    known_facts.add(_norm(content))
                    result.facts.append(record.id)
                    continue
            elif problem is None:
                try:
                    value = validate_preference(proposal.key or "", proposal.value)
                    if not number_in(proposal.evidence, value):
                        raise PreferenceError(f"{value:g} does not appear in the quoted evidence")
                except PreferenceError as exc:
                    problem = str(exc)
                else:
                    if (proposal.key, value) in known_prefs:
                        continue  # already active, pending, rejected or deliberately dropped
                    if len(result.pending) >= MAX_PREFERENCES_PER_RUN:
                        problem = "too many preferences proposed for one run"
                    else:
                        spec = PREFERENCE_KEYS[proposal.key]
                        record = memory.add_pending(
                            user_id, f"{spec.label}: {value:g}%", key=proposal.key, value=value)
                        known_prefs.add((proposal.key, value))
                        result.pending.append(record.id)
                        continue
            result.rejected.append(f"{proposal.kind} not stored: {problem}")
    except MemoryUnavailable:
        return result.model_copy(update={"status": "unavailable"})
    if result.facts or result.pending:
        result.status = "written"
    return result
