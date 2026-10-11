"""State model for one stored Memory. Allowlisted in the checkpoint serializer."""
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
