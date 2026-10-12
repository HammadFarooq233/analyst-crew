"""Issue #28: the `remember` subcommand, and the bare run invocation still working beside it."""
import contextlib

import pytest
from langgraph.store.memory import InMemoryStore

import cli
from analyst_crew.memory import MemoryStore, UnavailableStore
from analyst_crew.memory.update import MemoryProposal
from tests.test_cli_report import _run_cli  # noqa: F401  (fixtures reused below)
from tests.test_memory_preferences import ScriptedMemoryModel

REL = "materiality_relative_threshold_percent"


def _remember(monkeypatch, capsys, store, proposal, *argv):
    @contextlib.contextmanager
    def fake_store():
        yield store

    monkeypatch.setattr(cli, "memory_store_context", fake_store)
    monkeypatch.setattr(cli, "production_memory_model", lambda: ScriptedMemoryModel(proposal))
    monkeypatch.setattr("sys.argv", ["cli", "remember", *argv])
    code = 0
    try:
        cli.main()
    except SystemExit as exc:
        code = exc.code or 0
    captured = capsys.readouterr()
    return captured.out, captured.err, code


def test_remember_stores_a_typed_preference_and_prints_how_it_was_interpreted(monkeypatch, capsys):
    store = InMemoryStore()
    out, _, code = _remember(monkeypatch, capsys, store, MemoryProposal(kind="preference", key=REL, value=8),
                             "for me, unusual means a swing over 8%", "--user-id", "priya")
    assert code == 0 and "typed preference" in out
    [record] = MemoryStore(store).list("priya", "preference")
    assert record.value == 8.0


def test_remember_defaults_to_the_default_user(monkeypatch, capsys):
    store = InMemoryStore()
    _remember(monkeypatch, capsys, store, MemoryProposal(kind="fact"), "regions follow sales managers")
    assert [r.content for r in MemoryStore(store).list("default")] == ["regions follow sales managers"]


def test_a_failed_remember_exits_non_zero_stores_nothing_and_says_why(monkeypatch, capsys):
    store = InMemoryStore()
    out, err, code = _remember(monkeypatch, capsys, store, RuntimeError("model down"), "8% matters", "--user-id", "priya")
    assert code == 1 and "Nothing was stored" in (out + err)
    assert MemoryStore(store).list("priya") == []


def test_remember_during_a_storage_outage_says_so(monkeypatch, capsys):
    out, err, code = _remember(monkeypatch, capsys, UnavailableStore(),
                               MemoryProposal(kind="preference", key=REL, value=8), "over 8% matters")
    assert code == 1 and "unavailable" in (out + err).lower()


def test_remember_without_persistent_storage_refuses_rather_than_pretending(monkeypatch, capsys):
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setattr("sys.argv", ["cli", "remember", "over 8% matters"])
    with pytest.raises(SystemExit) as excinfo:
        cli.main()
    assert excinfo.value.code == 1
    assert "POSTGRES_CONNECTION_STRING" in capsys.readouterr().err


def test_the_bare_run_invocation_and_the_explicit_run_subcommand_both_still_work(monkeypatch, tmp_path, capsys):
    out, _, code = _run_cli(monkeypatch, tmp_path, capsys)
    assert code == 0 and "## Summary" in out
    out, _, code = _run_cli(monkeypatch, tmp_path, capsys, run_subcommand=True)
    assert code == 0 and "## Summary" in out
