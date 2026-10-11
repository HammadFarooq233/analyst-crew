"""State models for stored and applied Memory. Allowlisted in the checkpoint serializer."""
from typing import Literal

from pydantic import BaseModel

MemoryKind = Literal["preference", "fact", "summary"]
MemoryOrigin = Literal["stated", "inferred", "run_record"]

MEMORY_KINDS: tuple[str, ...] = ("preference", "fact", "summary")


class MemoryRecord(BaseModel):
    id: str
    user_id: str
    kind: MemoryKind
    origin: MemoryOrigin = "stated"
    content: str
    created_at: str
    # Typed Preferences only: a key from the closed vocabulary and its validated value.
    key: str | None = None
    value: float | None = None


class AppliedMemory(BaseModel):
    """One Memory that influenced a run, as resolved by the Crew. ``overridden`` means a
    stored Typed Preference was not used because the question stated its own instruction."""

    memory_id: str
    kind: MemoryKind
    key: str | None = None
    origin: MemoryOrigin = "stated"
    status: Literal["applied", "overridden"] = "applied"
