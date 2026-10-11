"""Pure resolution of stored Typed Preferences into the Handoff's analysis parameters.

The Crew calls this once per run. The latest statement per key wins. An instruction in
the question beats a stored preference for that run and is disclosed as ``overridden``.
Nothing here can exclude data: the only outputs are the vocabulary's threshold parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from analyst_crew.memory.models import AppliedMemory, MemoryRecord
from analyst_crew.memory.vocabulary import PREFERENCE_KEYS, PreferenceError, validate_preference
from analyst_crew.tools.analysis import _materiality_from_question

RELATIVE_KEY = "materiality_relative_threshold_percent"
ABSOLUTE_KEY = "materiality_absolute_threshold_percent"


@dataclass
class Resolution:
    parameters: dict[str, Any] = field(default_factory=dict)
    applied: list[AppliedMemory] = field(default_factory=list)


def resolve_preferences(records: list[MemoryRecord], question: str) -> Resolution:
    latest: dict[str, tuple[MemoryRecord, float]] = {}
    for record in sorted(records, key=lambda r: (r.created_at, r.id)):
        if record.kind != "preference" or record.key not in PREFERENCE_KEYS:
            continue
        try:
            latest[record.key] = (record, validate_preference(record.key, record.value))
        except PreferenceError:
            continue  # a record that no longer validates never becomes a parameter

    relative, absolute = _materiality_from_question(question)
    stated = {RELATIVE_KEY: relative, ABSOLUTE_KEY: absolute}

    resolution = Resolution()
    memory_applied = False
    for key, (record, value) in latest.items():
        if stated.get(key) is not None:
            status = "overridden"
        else:
            status = "applied"
            resolution.parameters[key] = value
            memory_applied = True
        resolution.applied.append(AppliedMemory(
            memory_id=record.id, kind="preference", key=key, origin=record.origin, status=status))
    if memory_applied:
        # Explicit parameters beat question phrasing downstream, so carry the question's own
        # thresholds across too; otherwise a stored key would silence an instruction on the other key.
        for key, value in stated.items():
            if value is not None:
                resolution.parameters[key] = value
        resolution.parameters["materiality_source"] = "memory"
    return resolution
