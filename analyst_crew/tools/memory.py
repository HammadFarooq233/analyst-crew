import uuid

from langgraph.store.base import BaseStore
from langmem import create_search_memory_tool


def create_memory_tool(store: BaseStore | None = None):
    return create_search_memory_tool(
        namespace=("planner",),
        store=store,
    )


def seed_memory(store: BaseStore, fact: str):
    key = f"mem-{uuid.uuid4().hex[:8]}"
    store.put(("planner",), key, {"content": fact})
    return key


def clear_namespace(store: BaseStore):
    for item in store.search(("planner",), limit=1000):
        store.delete(("planner",), item.key)