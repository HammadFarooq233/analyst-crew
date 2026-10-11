"""Persistent per-user Memory on a LangGraph store, keyed by user and kind.

Isolation is structural: every record lives in the namespace
``("memory", <user>, <kind>)`` and is only ever read through that namespace.
Retrieval is deterministic and role-scoped (ADR-0005); nothing here is a
model-callable Tool. Any backend failure surfaces as ``MemoryUnavailable`` so
callers can degrade to defaults instead of failing the run.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langgraph.store.base import BaseStore

from analyst_crew.memory.models import MEMORY_KINDS, MemoryRecord

ROOT = "memory"
LEGACY_NAMESPACE = ("planner",)

# Which Memory kinds each Agent role may be given. Absent roles get nothing.
ROLE_KINDS: dict[str, tuple[str, ...]] = {
    "Planner": ("fact", "summary"),
    "Analyst": ("preference",),
    "Reporter": ("fact",),
}


class MemoryUnavailable(Exception):
    """The Memory store could not be read or written."""


class UnavailableStore(BaseStore):
    """A store standing in for one that cannot be reached: every operation fails."""

    def batch(self, ops):
        raise ConnectionError("Memory storage is unavailable")

    async def abatch(self, ops):
        raise ConnectionError("Memory storage is unavailable")


def _label(user_id: str) -> str:
    # Store namespace labels may not contain ".".
    return user_id.replace("%", "%25").replace(".", "%2E")


class MemoryStore:
    def __init__(self, store: BaseStore):
        self._store = store

    @staticmethod
    def namespace(user_id: str, kind: str) -> tuple[str, str, str]:
        if not user_id:
            raise ValueError("user_id must not be empty")
        if kind not in MEMORY_KINDS:
            raise ValueError(f"Unknown memory kind: {kind!r}")
        return (ROOT, _label(user_id), kind)

    def add(self, user_id: str, kind: str, content: str, origin: str = "stated") -> MemoryRecord:
        record = MemoryRecord(
            id=f"mem-{uuid.uuid4().hex[:8]}", user_id=user_id, kind=kind, origin=origin, content=content,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            self._store.put(self.namespace(user_id, kind), record.id, record.model_dump())
        except Exception as exc:  # noqa: BLE001 — any backend failure is an outage
            raise MemoryUnavailable(str(exc)) from exc
        return record

    def list(self, user_id: str, kind: str | None = None) -> list[MemoryRecord]:
        kinds = MEMORY_KINDS if kind is None else (kind,)
        records: list[MemoryRecord] = []
        try:
            for each in kinds:
                for item in self._store.search(self.namespace(user_id, each), limit=1000):
                    records.append(MemoryRecord.model_validate(item.value))
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc
        return sorted(records, key=lambda record: (record.created_at, record.id))

    def read_for_role(self, role: str, user_id: str) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for kind in ROLE_KINDS.get(role, ()):
            records.extend(self.list(user_id, kind))
        return sorted(records, key=lambda record: (record.created_at, record.id))

    def delete(self, user_id: str, kind: str, memory_id: str) -> None:
        try:
            self._store.delete(self.namespace(user_id, kind), memory_id)
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc

    def clear(self, user_id: str | None = None) -> None:
        """Delete every memory (one user's when ``user_id`` is given)."""
        prefix = (ROOT,) if user_id is None else (ROOT, _label(user_id))
        try:
            for namespace in self._store.list_namespaces(prefix=prefix, limit=10000):
                for item in self._store.search(namespace, limit=1000):
                    self._store.delete(namespace, item.key)
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc

    def retire_legacy_namespace(self) -> None:
        """Remove the old Planner-only, user-less namespace so no orphans linger."""
        try:
            for item in self._store.search(LEGACY_NAMESPACE, limit=1000):
                self._store.delete(LEGACY_NAMESPACE, item.key)
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc
