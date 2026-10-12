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


class MemoryNotFound(Exception):
    """No active memory with that id exists for the user."""


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

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _put(self, record: MemoryRecord) -> None:
        self._store.put(self.namespace(record.user_id, record.kind), record.id, record.model_dump())

    def add(self, user_id: str, kind: str, content: str, origin: str = "stated", *,
            key: str | None = None, value: float | None = None,
            source_hash: str | None = None) -> MemoryRecord:
        """Store a memory. A newer Typed Preference on the same key supersedes the active one(s):
        the older record stays in the history, marked superseded and linked to its replacement."""
        record = MemoryRecord(
            id=f"mem-{uuid.uuid4().hex[:8]}", user_id=user_id, kind=kind, origin=origin, content=content,
            created_at=self._now(), key=key, value=value,
            source_hash=source_hash,
        )
        try:
            if kind == "preference" and key:
                older = [r for r in self._read(user_id, kind) if r.status == "active" and r.key == key]
            elif kind == "summary":  # only the latest Past-run Summary stays active
                older = [r for r in self._read(user_id, kind) if r.status == "active"]
            else:
                older = []
            if older:
                record = record.model_copy(update={"supersedes": older[-1].id})
            self._put(record)  # the replacement lands first, so a failure never leaves a gap
            for previous in older:
                self._put(previous.model_copy(update={
                    "status": "superseded", "superseded_by": record.id, "ended_at": record.created_at}))
        except Exception as exc:  # noqa: BLE001 — any backend failure is an outage
            raise MemoryUnavailable(str(exc)) from exc
        return record

    def _read(self, user_id: str, kind: str) -> list[MemoryRecord]:
        items = self._store.search(self.namespace(user_id, kind), limit=1000)
        return sorted((MemoryRecord.model_validate(item.value) for item in items),
                      key=lambda record: (record.created_at, record.id))

    def history(self, user_id: str, kind: str | None = None) -> list[MemoryRecord]:
        """Every memory ever stored for the user, including superseded and forgotten ones."""
        kinds = MEMORY_KINDS if kind is None else (kind,)
        try:
            records = [record for each in kinds for record in self._read(user_id, each)]
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc
        return sorted(records, key=lambda record: (record.created_at, record.id))

    def list(self, user_id: str, kind: str | None = None) -> list[MemoryRecord]:
        """The active memories: what retrieval can see."""
        return [record for record in self.history(user_id, kind) if record.status == "active"]

    def forget(self, user_id: str, memory_id: str) -> MemoryRecord:
        """Remove an active memory from retrieval. It stays in the history, marked forgotten."""
        target = next((record for record in self.list(user_id) if record.id == memory_id), None)
        if target is None:
            raise MemoryNotFound(f"No active memory {memory_id!r} for user {user_id!r}.")
        forgotten = target.model_copy(update={"status": "forgotten", "ended_at": self._now()})
        try:
            self._put(forgotten)
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc
        return forgotten

    def pending(self, user_id: str) -> list[MemoryRecord]:
        """Inferred Typed Preferences awaiting confirmation. They are never retrieved for a run."""
        return [r for r in self.history(user_id, "preference") if r.status == "pending"]

    def add_pending(self, user_id: str, content: str, *, key: str, value: float) -> MemoryRecord:
        """Store an inferred Typed Preference as pending: it changes no number until confirmed."""
        record = MemoryRecord(
            id=f"mem-{uuid.uuid4().hex[:8]}", user_id=user_id, kind="preference", origin="inferred",
            content=content, created_at=self._now(), key=key, value=value, status="pending")
        try:
            self._put(record)
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc
        return record

    def _pending(self, user_id: str, memory_id: str) -> MemoryRecord:
        target = next((r for r in self.pending(user_id) if r.id == memory_id), None)
        if target is None:
            raise MemoryNotFound(f"No pending preference {memory_id!r} for user {user_id!r}.")
        return target

    def confirm(self, user_id: str, memory_id: str) -> MemoryRecord:
        """Make a pending preference active now, superseding the active one on the same key."""
        target = self._pending(user_id, memory_id)
        now = self._now()
        try:
            older = [r for r in self._read(user_id, "preference") if r.status == "active" and r.key == target.key]
            confirmed = target.model_copy(update={
                "status": "active", "created_at": now, "supersedes": older[-1].id if older else None})
            self._put(confirmed)
            for previous in older:
                self._put(previous.model_copy(update={
                    "status": "superseded", "superseded_by": confirmed.id, "ended_at": now}))
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc
        return confirmed

    def reject(self, user_id: str, memory_id: str) -> MemoryRecord:
        """Decline a pending preference. It stays in the history so it is not proposed again."""
        target = self._pending(user_id, memory_id)
        rejected = target.model_copy(update={"status": "rejected", "ended_at": self._now()})
        try:
            self._put(rejected)
        except Exception as exc:  # noqa: BLE001
            raise MemoryUnavailable(str(exc)) from exc
        return rejected

    def read_for_role(self, role: str, user_id: str) -> list[MemoryRecord]:
        records: list[MemoryRecord] = []
        for kind in ROLE_KINDS.get(role, ()):
            records.extend(self.list(user_id, kind))
        return sorted(records, key=lambda record: (record.created_at, record.id))

    def delete(self, user_id: str, kind: str, memory_id: str) -> None:
        """Forget a memory (kept in the history). Kept for callers that know the kind."""
        try:
            self.forget(user_id, memory_id)
        except MemoryNotFound:
            pass

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
