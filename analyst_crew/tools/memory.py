"""Test helpers over the per-user Memory store. The old Planner-only
``("planner",)`` namespace and its model-callable search Tool are retired
(ADR-0005); retrieval is deterministic and lives in ``analyst_crew.memory``."""
from langgraph.store.base import BaseStore

from analyst_crew.memory import MemoryStore


def seed_memory(store: BaseStore, fact: str, user_id: str = "default", kind: str = "fact") -> str:
    return MemoryStore(store).add(user_id, kind, fact).id


def clear_namespace(store: BaseStore, user_id: str | None = None) -> None:
    MemoryStore(store).clear(user_id)
