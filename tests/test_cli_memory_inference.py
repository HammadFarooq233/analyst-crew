"""Issue #32: the CLI path for pending inferred preferences (`confirm`, `reject`, `list`) and the
notice a run prints when it left preferences waiting."""
from langgraph.store.memory import InMemoryStore

import cli
from analyst_crew.memory import MemoryStore
from tests.test_cli_memory_control import _cli

REL = "materiality_relative_threshold_percent"


def _pending(store, user="priya", value=8.0):
    return MemoryStore(store).add_pending(user, f"Material Change relative threshold: {value:g}%", key=REL, value=value)


def test_list_shows_pending_preferences_separately_from_active_memories(monkeypatch, capsys):
    store = InMemoryStore()
    pending = _pending(store)
    active = MemoryStore(store).add("priya", "fact", "regions follow sales managers")
    out, _, code = _cli(monkeypatch, capsys, store, "list", "--user-id", "priya")
    assert code == 0 and active.id in out
    block = out.split("Pending", 1)[1]
    assert pending.id in block and "confirm" in block.lower() and active.id not in block


def test_confirm_activates_the_preference(monkeypatch, capsys):
    store = InMemoryStore()
    pending = _pending(store)
    out, _, code = _cli(monkeypatch, capsys, store, "confirm", pending.id, "--user-id", "priya")
    assert code == 0 and pending.id in out and "applies" in out.lower()
    assert [r.id for r in MemoryStore(store).read_for_role("Analyst", "priya")] == [pending.id]


def test_reject_discards_the_preference_but_keeps_the_history(monkeypatch, capsys):
    store = InMemoryStore()
    pending = _pending(store)
    out, _, code = _cli(monkeypatch, capsys, store, "reject", pending.id, "--user-id", "priya")
    assert code == 0 and "rejected" in out.lower()
    assert MemoryStore(store).read_for_role("Analyst", "priya") == []
    listing, _, _ = _cli(monkeypatch, capsys, store, "list", "--user-id", "priya", "--history")
    assert pending.id in listing and "rejected" in listing


def test_confirm_and_reject_fail_clearly_for_unknown_ids_and_other_users(monkeypatch, capsys):
    store = InMemoryStore()
    pending = _pending(store)
    for command in ("confirm", "reject"):
        for memory_id, user in (("mem-nope", "priya"), (pending.id, "sam")):
            _, err, code = _cli(monkeypatch, capsys, store, command, memory_id, "--user-id", user)
            assert code == 1 and "No pending preference" in err
    assert MemoryStore(store).pending("priya") == [pending]


def test_a_storage_outage_confirms_nothing(monkeypatch, capsys):
    from analyst_crew.memory import UnavailableStore

    _, err, code = _cli(monkeypatch, capsys, UnavailableStore(), "confirm", "mem-1")
    assert code == 1 and "unavailable" in err.lower()


def test_the_run_output_says_when_preferences_wait_for_confirmation():
    from analyst_crew.memory.models import InferenceResult

    result = {"artifacts": {"inference": InferenceResult(status="written", pending=["mem-abc12345"])}}
    text = cli.format_pending_notice(result)
    assert "mem-abc12345" in text and "confirm" in text
    assert cli.format_pending_notice({"artifacts": {}}) == ""
