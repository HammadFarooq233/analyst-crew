"""The Crew-level Memory Update step behind `remember` (ADR-0005). Not an Agent.

A model proposes a candidate Typed Preference or Free-text Fact from the user's sentence;
code validates it. A preference that does not validate degrades to a Free-text Fact and
never becomes a parameter. An unusable proposal or a storage failure stores nothing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from analyst_crew.memory.models import MemoryRecord
from analyst_crew.memory.store import MemoryStore, MemoryUnavailable
from analyst_crew.memory.vocabulary import PREFERENCE_KEYS, PreferenceError, validate_preference

PROPOSAL_PROMPT = """You turn one sentence a user said about how they like their sales analysis done into a candidate memory.
Return JSON with fields: kind ("preference" or "fact"), key, value.
Use kind "preference" ONLY when the sentence states a numeric threshold for one of these keys, and set key and value (a number):
{vocabulary}
Everything else (how regions are organised, categories to ignore, general context) is kind "fact"; leave key and value null.
Never invent a number that the sentence does not state."""


class MemoryProposal(BaseModel):
    kind: Literal["preference", "fact"]
    key: str | None = None
    value: float | str | None = None


@dataclass
class RememberOutcome:
    stored: bool
    message: str
    record: MemoryRecord | None = None


def _number_in(sentence: str, value: float) -> bool:
    shown = {f"{value:g}", f"{value:.1f}", f"{value:.2f}"}
    return any(re.search(rf"(?<![\d.]){re.escape(text)}(?![\d.]*\d)", sentence) for text in shown)


def production_memory_model() -> Any:
    """The live proposer seam: a chat model that `remember` asks for a ``MemoryProposal``.
    Tests inject a scripted model instead."""
    from analyst_crew.llm import make_chat_model

    return make_chat_model(temperature=0)


def _propose(model: Any, sentence: str) -> MemoryProposal:
    from langchain_core.messages import HumanMessage, SystemMessage

    vocabulary = "\n".join(f"- {spec.key}: {spec.label}" for spec in PREFERENCE_KEYS.values())
    messages = [SystemMessage(content=PROPOSAL_PROMPT.format(vocabulary=vocabulary)), HumanMessage(content=sentence)]
    raw = model.with_structured_output(MemoryProposal).invoke(messages)
    if isinstance(raw, MemoryProposal):
        return raw
    if isinstance(raw, dict):
        return MemoryProposal.model_validate(raw)
    raise ValueError(f"the model returned {type(raw).__name__}, not a proposal")


def remember(sentence: str, user_id: str, memory: MemoryStore, model: Any) -> RememberOutcome:
    sentence = (sentence or "").strip()
    if not sentence:
        return RememberOutcome(False, "Nothing was stored: the sentence is empty.")
    try:
        proposal = _propose(model, sentence)
    except (ValidationError, ValueError, TypeError) as exc:
        return RememberOutcome(False, f"Nothing was stored: the proposal could not be understood ({exc}).")
    except Exception as exc:  # noqa: BLE001 — a model failure must never store anything
        return RememberOutcome(False, f"Nothing was stored: the model could not interpret the sentence ({exc}).")

    reason = None
    if proposal.kind == "preference":
        try:
            value = validate_preference(proposal.key or "", proposal.value)
            if not _number_in(sentence, value):
                raise PreferenceError(f"{value:g} does not appear in what you said")
        except PreferenceError as exc:
            reason = str(exc)
        else:
            spec = PREFERENCE_KEYS[proposal.key]
            content = f"{spec.label}: {value:g}%"
            try:
                record = memory.add(user_id, "preference", content, key=proposal.key, value=value)
            except MemoryUnavailable as exc:
                return RememberOutcome(False, f"Nothing was stored: Memory storage is unavailable ({exc}).")
            return RememberOutcome(True, f"Stored as a typed preference: {content}. It applies from your next run.", record)

    try:
        record = memory.add(user_id, "fact", sentence)
    except MemoryUnavailable as exc:
        return RememberOutcome(False, f"Nothing was stored: Memory storage is unavailable ({exc}).")
    why = f" (not usable as a parameter: {reason})" if reason else ""
    return RememberOutcome(True, f"Stored as a free-text fact{why}. It changes no analysis parameter.", record)
