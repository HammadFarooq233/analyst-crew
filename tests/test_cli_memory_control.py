"""Issue #29: the `list` and `forget` subcommands, with supersede through `remember`."""
import contextlib

from langgraph.store.memory import InMemoryStore

import cli
from analyst_crew.memory import MemoryStore, UnavailableStore
from analyst_crew.memory.update import MemoryProposal
from tests.test_memory_preferences import ScriptedMemoryModel

REL = "materiality_relative_threshold_percent"


def _cli(monkeypatch, capsys, store, *argv, proposal=None):
    @contextlib.contextmanager
    def fake_store():
        yield store

    monkeypatch.setattr(cli, "memory_store_context", fake_store)
    if proposal is not None:
        monkeypatch.setattr(cli, "production_memory_model", lambda: ScriptedMemoryModel(proposal))
    monkeypatch.setattr("sys.argv", ["cli", *argv])
    code = 0
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code or 0
    captured = capsys.readouterr()
    return captured.out, captured.err, code


def _state(monkeypatch, capsys, store, sentence, value, user="priya"):
    return _cli(monkeypatch, capsys, store, "remember", sentence, "--user-id", user,
                proposal=MemoryProposal(kind="preference", key=REL, value=value))


def test_list_shows_each_memory_with_id_kind_origin_and_when_it_was_stored(monkeypatch, capsys):
    store = InMemoryStore()
    memory = MemoryStore(store)
    fact = memory.add("priya", "fact", "regions follow sales managers")
    pref = memory.add("priya", "preference", "Material Change relative threshold: 8%", key=REL, value=8.0)
    memory.add("priya", "fact", "inferred one", origin="inferred")
    memory.add("sam", "fact", "sam's private fact")
    out, _, code = _cli(monkeypatch, capsys, store, "list", "--user-id", "priya")
    assert code == 0
    for record in (fact, pref):
        line = next(line for line in out.splitlines() if record.id in line)
        assert record.kind in line and record.origin in line and record.created_at[:19].replace("T", " ") in line
        assert record.content in line
    assert "inferred" in out and "sam's private fact" not in out


def test_list_for_a_user_with_no_memories_says_so(monkeypatch, capsys):
    out, _, code = _cli(monkeypatch, capsys, InMemoryStore(), "list", "--user-id", "priya")
    assert code == 0 and "No memories" in out


def test_a_newer_statement_replaces_the_older_in_the_list_and_the_history_shows_both(monkeypatch, capsys):
    store = InMemoryStore()
    _state(monkeypatch, capsys, store, "unusual means over 10%", 10)
    _state(monkeypatch, capsys, store, "actually unusual means over 20%", 20)
    out, _, _ = _cli(monkeypatch, capsys, store, "list", "--user-id", "priya")
    assert "20%" in out and "10%" not in out
    out, _, _ = _cli(monkeypatch, capsys, store, "list", "--user-id", "priya", "--history")
    assert "10%" in out and "20%" in out and "superseded" in out and "active" in out


def test_forget_removes_the_memory_from_the_list_but_the_history_shows_it_forgotten(monkeypatch, capsys):
    store = InMemoryStore()
    record = MemoryStore(store).add("priya", "fact", "regions follow sales managers")
    out, _, code = _cli(monkeypatch, capsys, store, "forget", record.id, "--user-id", "priya")
    assert code == 0 and record.id in out and "forgotten" in out.lower()
    out, _, _ = _cli(monkeypatch, capsys, store, "list", "--user-id", "priya")
    assert record.id not in out
    out, _, _ = _cli(monkeypatch, capsys, store, "list", "--user-id", "priya", "--history")
    assert record.id in out and "forgotten" in out


def test_forgetting_an_unknown_memory_or_another_users_memory_fails_and_changes_nothing(monkeypatch, capsys):
    store = InMemoryStore()
    mine = MemoryStore(store).add("priya", "fact", "mine")
    out, err, code = _cli(monkeypatch, capsys, store, "forget", "mem-nope", "--user-id", "priya")
    assert code == 1 and "mem-nope" in (out + err)
    _, _, code = _cli(monkeypatch, capsys, store, "forget", mine.id, "--user-id", "sam")
    assert code == 1
    assert [r.id for r in MemoryStore(store).list("priya")] == [mine.id]


def test_list_and_forget_during_a_storage_outage_say_so(monkeypatch, capsys):
    out, err, code = _cli(monkeypatch, capsys, UnavailableStore(), "list")
    assert code == 1 and "unavailable" in (out + err).lower()
    out, err, code = _cli(monkeypatch, capsys, UnavailableStore(), "forget", "mem-x")
    assert code == 1 and "unavailable" in (out + err).lower()


def test_list_without_persistent_storage_refuses_rather_than_pretending(monkeypatch, capsys):
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setattr("sys.argv", ["cli", "list"])
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected exit")
    assert "POSTGRES_CONNECTION_STRING" in capsys.readouterr().err
