"""Store contract (issue #27): the same assertions hold for the in-memory store
and the Postgres store, so offline tests and live runs behave the same."""
import os

import pytest
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

from analyst_crew.memory import MemoryRecord, MemoryStore, MemoryUnavailable, UnavailableStore


def _postgres_conn():
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn_string:
        pytest.skip("POSTGRES_CONNECTION_STRING not set")
    try:
        import psycopg

        psycopg.connect(conn_string, connect_timeout=3).close()
    except Exception:
        pytest.skip("Postgres not reachable")
    return conn_string


@pytest.fixture(params=["in_memory", pytest.param("postgres", marks=pytest.mark.integration)])
def backing_store(request):
    if request.param == "in_memory":
        yield InMemoryStore()
        return
    from langgraph.store.postgres import PostgresStore

    with PostgresStore.from_conn_string(_postgres_conn()) as store:
        store.setup()
        yield store


@pytest.fixture
def memory(backing_store):
    memory = MemoryStore(backing_store)
    memory.clear()
    yield memory
    memory.clear()


def test_a_written_memory_is_read_back_with_its_user_kind_and_origin(memory):
    written = memory.add("priya", "fact", "regions are organised by sales manager")
    [read] = memory.list("priya")
    assert isinstance(read, MemoryRecord)
    assert read == written
    assert (read.user_id, read.kind, read.origin) == ("priya", "fact", "stated")
    assert read.content == "regions are organised by sales manager"


def test_memory_for_one_user_is_never_listed_for_another(memory):
    memory.add("priya", "fact", "priya's fact")
    memory.add("priya.s@example.com", "fact", "dotted user fact")
    assert memory.list("sam") == []
    assert [r.content for r in memory.list("priya")] == ["priya's fact"]
    assert [r.content for r in memory.list("priya.s@example.com")] == ["dotted user fact"]
    assert memory.list("pri") == []


def test_memories_are_separated_by_kind(memory):
    memory.add("priya", "fact", "a fact")
    memory.add("priya", "summary", "a summary")
    assert [r.content for r in memory.list("priya", "fact")] == ["a fact"]
    assert [r.content for r in memory.list("priya", "summary")] == ["a summary"]
    assert {r.content for r in memory.list("priya")} == {"a fact", "a summary"}


def test_memories_are_listed_oldest_first(memory):
    for text in ("one", "two", "three"):
        memory.add("priya", "fact", text)
    assert [r.content for r in memory.list("priya")] == ["one", "two", "three"]


def test_delete_removes_only_that_memory(memory):
    keep = memory.add("priya", "fact", "keep")
    drop = memory.add("priya", "fact", "drop")
    memory.delete("priya", "fact", drop.id)
    assert memory.list("priya") == [keep]


def test_each_agent_role_reads_only_its_kinds(memory):
    memory.add("priya", "fact", "fact")
    memory.add("priya", "summary", "summary")
    memory.add("priya", "preference", "preference")
    assert {r.kind for r in memory.read_for_role("Planner", "priya")} == {"fact", "summary"}
    assert {r.kind for r in memory.read_for_role("Reporter", "priya")} == {"fact"}
    assert {r.kind for r in memory.read_for_role("Analyst", "priya")} == {"preference"}
    for role in ("Critic", "Fetcher", "Visualizer"):
        assert memory.read_for_role(role, "priya") == []


def test_the_retired_planner_namespace_is_removed(backing_store):
    backing_store.put(("planner",), "mem-old", {"content": "old planner-only fact"})
    memory = MemoryStore(backing_store)
    memory.retire_legacy_namespace()
    assert backing_store.search(("planner",), limit=10) == []


def test_an_unreachable_store_raises_memory_unavailable():
    memory = MemoryStore(UnavailableStore())
    with pytest.raises(MemoryUnavailable):
        memory.list("priya")
    with pytest.raises(MemoryUnavailable):
        memory.add("priya", "fact", "x")


def test_unavailable_store_is_a_store():
    assert isinstance(UnavailableStore(), BaseStore)
