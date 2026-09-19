"""Issue #11 — resume/invalidation behavior through the real Planner ->
Fetcher -> Analyst graph, exercised offline against an in-memory
``MemorySaver`` checkpointer (no live Postgres required; see the
POSTGRES-marked ``TestStatePersistence`` in ``tests/test_checkpointer.py``
for the equivalent behavior against real Postgres).

Covers the acceptance criteria named in the issue:
  - a resumed thread reuses a completed Step's output only when its
    execution key still matches ("unchanged resume");
  - changed CSV content invalidates the Fetcher output and cascades into
    an Analyst recompute, with no separately maintained downstream list
    ("source change");
  - changed analysis_parameters reuse the Fetcher's Dataset Artifact but
    recompute Findings ("parameter change");
  - interrupted and failed Steps restart cleanly from the last safe
    boundary rather than replaying broken state ("interruption recovery");
  - reused external (Postgres/FRED) fetches avoid duplicate calls when
    their source identity (the as_of_date window) is unchanged;
  - CrewState's step_outcomes make reused/recomputed/invalidated/
    interrupted/failed outcomes distinctly inspectable.

Also includes small pure unit tests of ``execution_key``/``_resume_decision``
directly, without any graph execution.
"""
import csv
from datetime import date

import pytest
from langgraph.checkpoint.memory import MemorySaver

import analyst_crew.agents.planner as planner_module
from analyst_crew.agents import Plan, Step
from analyst_crew.agents.planner import _resume_decision, execution_key
from analyst_crew.tools.sales import CANONICAL_FIELDS, sales_data_source


def _write_sales(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(order_id, order_date, quantity=1, unit_price=10):
    return {
        "order_id": order_id, "order_date": order_date, "product_id": "P001",
        "product_name": "Desk Lamp", "category": "Office", "region": "North",
        "sales_channel": "web", "customer_segment": "consumer",
        "quantity": quantity, "unit_price": unit_price,
        "discount_amount": 0, "returned_quantity": 0,
    }


def _base_rows():
    return [
        _row("M1", "2025-05-01"), _row("M2", "2025-05-05", quantity=2),
        _row("J1", "2025-06-01"), _row("J2", "2025-06-05", quantity=3),
    ]


def _plan():
    fetch = Step(
        step_id="fetch_sales", agent="Fetcher", objective="Fetch normalized sales",
        source_ids=["sales_csv"], expected_output="sales_dataset", declared_artifacts=["sales_dataset"],
    )
    analyze = Step(
        step_id="analyze_baseline", agent="Analyst", objective="Compare baseline sales periods",
        expected_output="baseline_findings", declared_artifacts=["baseline_findings"], depends_on=["fetch_sales"],
    )
    return Plan(steps=[fetch, analyze], user_question="Compare baseline sales")


class CountingScriptedPlannerModel:
    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return self.plan


class CountingScriptedFetcherModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, _context):
        self.calls += 1
        return {"tool": "CSV.read", "source_id": "sales_csv"}


class RaisingFetcherModel:
    """Simulates a process interruption mid-Step: an exception type the
    node's ``except (OSError, RuntimeError, ValueError)`` never catches, so
    it propagates all the way out of ``invoke()`` uncaught — nothing about
    this attempt is ever returned to (and therefore checkpointed by) the
    graph."""

    def invoke(self, _context):
        raise KeyboardInterrupt("simulated interruption mid-Fetcher-Step")


class CountingScriptedAnalystModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, request):
        self.calls += 1
        artifact_id = next(iter(request["available_artifacts"].values()))
        return {"tool": "DuckDB.sales_baseline", "dataset_artifact_id": artifact_id}


class AlwaysInvalidFetcherModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, _context):
        self.calls += 1
        return {"tool": "execute_python", "source_id": "sales_csv"}


def _runtime(source, tmp_path, *, thread_id, as_of_date, planner_model, fetcher_model, analyst_model,
             checkpointer, extra_sources=None):
    return {
        "configurable": {
            "thread_id": thread_id,
            "data_sources": {**sales_data_source(source), **(extra_sources or {})},
            "artifact_dir": tmp_path / "artifacts",
            "available_agents": ["Fetcher", "Analyst"],
            "planner_model": planner_model,
            "fetcher_model": fetcher_model,
            "analyst_model": analyst_model,
            "analysis_parameters": {"as_of_date": as_of_date},
        }
    }, {"checkpointer": checkpointer}


def _invoke(source, tmp_path, monkeypatch, *, thread_id, as_of_date, checkpointer,
            planner_model=None, fetcher_model=None, analyst_model=None, extra_sources=None):
    planner_model = planner_model or CountingScriptedPlannerModel(_plan())
    fetcher_model = fetcher_model or CountingScriptedFetcherModel()
    analyst_model = analyst_model or CountingScriptedAnalystModel()
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    runtime, kwargs = _runtime(
        source, tmp_path, thread_id=thread_id, as_of_date=as_of_date,
        planner_model=planner_model, fetcher_model=fetcher_model, analyst_model=analyst_model,
        checkpointer=checkpointer, extra_sources=extra_sources,
    )
    result = planner_module.invoke("Compare baseline sales", runtime, **kwargs)
    return result, planner_model, fetcher_model, analyst_model


@pytest.fixture
def duckdb_and_pyarrow():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


class TestExecutionKeyUnit:
    """Pure unit coverage of the key/decision functions, no graph execution."""

    def _steps(self):
        return _plan().steps

    def test_identical_inputs_produce_identical_key(self):
        fetch_step, _ = self._steps()
        key1 = execution_key(fetch_step, analysis_parameters={}, input_artifact_hashes=["abc"])
        key2 = execution_key(fetch_step, analysis_parameters={}, input_artifact_hashes=["abc"])
        assert key1 == key2

    def test_changed_input_hash_changes_key(self):
        fetch_step, _ = self._steps()
        key1 = execution_key(fetch_step, analysis_parameters={}, input_artifact_hashes=["abc"])
        key2 = execution_key(fetch_step, analysis_parameters={}, input_artifact_hashes=["def"])
        assert key1 != key2

    def test_changed_step_contract_changes_key(self):
        fetch_step, _ = self._steps()
        changed = fetch_step.model_copy(update={"objective": "A different objective"})
        assert execution_key(fetch_step, input_artifact_hashes=["abc"]) != execution_key(changed, input_artifact_hashes=["abc"])

    def test_changed_analysis_parameters_changes_key_only_when_passed(self):
        _, analyst_step = self._steps()
        key1 = execution_key(analyst_step, analysis_parameters={"as_of_date": "2025-06-30"}, input_artifact_hashes=["h"])
        key2 = execution_key(analyst_step, analysis_parameters={"as_of_date": "2025-07-31"}, input_artifact_hashes=["h"])
        assert key1 != key2
        # Fetcher never receives analysis_parameters in its key at all.
        fetch_step, _ = self._steps()
        fetch_key1 = execution_key(fetch_step, analysis_parameters={}, input_artifact_hashes=["h"])
        fetch_key2 = execution_key(fetch_step, analysis_parameters={}, input_artifact_hashes=["h"])
        assert fetch_key1 == fetch_key2

    def test_resume_decision_never_ran(self):
        state = {"messages": [object()], "artifacts": {}}
        outcome, reason = _resume_decision(state, "fetch_sales", "key1")
        assert (outcome, reason) == ("recomputed", "never_ran")

    def test_resume_decision_unchanged_reuses(self):
        state = {
            "messages": [object(), object()],
            "artifacts": {"step_outcomes": {"fetch_sales": {"status": "completed", "execution_key": "key1"}}},
        }
        assert _resume_decision(state, "fetch_sales", "key1") == ("reused", "unchanged")

    def test_resume_decision_invalidated_on_key_mismatch(self):
        state = {
            "messages": [object(), object()],
            "artifacts": {"step_outcomes": {"fetch_sales": {"status": "completed", "execution_key": "key1"}}},
        }
        assert _resume_decision(state, "fetch_sales", "key2") == ("recomputed", "invalidated")

    def test_resume_decision_failed(self):
        state = {
            "messages": [object(), object()],
            "artifacts": {"terminal_failure": {"step_id": "fetch_sales"}},
        }
        assert _resume_decision(state, "fetch_sales", "key1") == ("recomputed", "failed")

    def test_resume_decision_interrupted(self):
        state = {"messages": [object(), object()], "artifacts": {}}
        assert _resume_decision(state, "fetch_sales", "key1") == ("recomputed", "interrupted")


class TestUnchangedResume:
    def test_second_invoke_reuses_fetcher_and_analyst_without_recomputing(self, tmp_path, monkeypatch, duckdb_and_pyarrow):
        source = _write_sales(tmp_path / "sales.csv", _base_rows())
        thread_id = "resume-unchanged-001"
        checkpointer = MemorySaver()

        result1, _, fetcher1, analyst1 = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30", checkpointer=checkpointer,
        )
        assert result1["artifacts"]["step_outcomes"]["fetch_sales"]["resume_outcome"] == "recomputed"
        assert result1["artifacts"]["step_outcomes"]["fetch_sales"]["resume_reason"] == "never_ran"
        assert result1["artifacts"]["step_outcomes"]["analyze_baseline"]["resume_outcome"] == "recomputed"
        assert fetcher1.calls == 1
        assert analyst1.calls == 1

        result2, _, fetcher2, analyst2 = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30", checkpointer=checkpointer,
        )
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_outcome"] == "reused"
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_reason"] == "unchanged"
        assert result2["artifacts"]["step_outcomes"]["analyze_baseline"]["resume_outcome"] == "reused"
        assert result2["artifacts"]["step_outcomes"]["analyze_baseline"]["resume_reason"] == "unchanged"
        # The Fetcher/Analyst Tool-calling models were never invoked a second time.
        assert fetcher2.calls == 0
        assert analyst2.calls == 0
        # The Dataset Artifact and Findings are reused byte-identically.
        assert result2["artifacts"]["datasets"] == result1["artifacts"]["datasets"]
        assert result2["artifacts"]["findings"].net_revenue.current == result1["artifacts"]["findings"].net_revenue.current


class TestSourceChangeInvalidation:
    def test_changed_csv_content_recomputes_fetcher_and_cascades_to_analyst(self, tmp_path, monkeypatch, duckdb_and_pyarrow):
        source = tmp_path / "sales.csv"
        _write_sales(source, _base_rows())
        thread_id = "resume-source-change-001"
        checkpointer = MemorySaver()

        result1, _, _, _ = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30", checkpointer=checkpointer,
        )
        original_content_hash = result1["artifacts"]["findings"].dataset_content_hash

        # Change the CSV content on disk between the two invokes on the same thread.
        _write_sales(source, _base_rows() + [_row("J3", "2025-06-10", quantity=5)])

        result2, _, fetcher2, analyst2 = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30", checkpointer=checkpointer,
        )
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_outcome"] == "recomputed"
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_reason"] == "invalidated"
        assert result2["artifacts"]["step_outcomes"]["analyze_baseline"]["resume_outcome"] == "recomputed"
        assert result2["artifacts"]["step_outcomes"]["analyze_baseline"]["resume_reason"] == "invalidated"
        assert fetcher2.calls == 1
        assert analyst2.calls == 1
        assert result2["artifacts"]["findings"].dataset_content_hash != original_content_hash
        assert result2["artifacts"]["findings"].net_revenue.current != result1["artifacts"]["findings"].net_revenue.current


class TestParameterChangeReusesFetcher:
    def test_changed_as_of_date_reuses_fetcher_but_recomputes_analyst(self, tmp_path, monkeypatch, duckdb_and_pyarrow):
        source = _write_sales(tmp_path / "sales.csv", _base_rows())
        thread_id = "resume-param-change-001"
        checkpointer = MemorySaver()

        result1, _, fetcher1, analyst1 = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30", checkpointer=checkpointer,
        )
        assert fetcher1.calls == 1
        assert analyst1.calls == 1

        result2, _, fetcher2, analyst2 = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-05", checkpointer=checkpointer,
        )
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_outcome"] == "reused"
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_reason"] == "unchanged"
        assert fetcher2.calls == 0  # No re-fetch: the CSV content did not change.
        assert result2["artifacts"]["step_outcomes"]["analyze_baseline"]["resume_outcome"] == "recomputed"
        assert result2["artifacts"]["step_outcomes"]["analyze_baseline"]["resume_reason"] == "invalidated"
        assert analyst2.calls == 1  # Findings did recompute for the new as_of_date.
        assert result2["artifacts"]["findings"].window.current_end.isoformat() == "2025-06-05"
        # Same Dataset Artifact reused for both windows.
        assert result2["artifacts"]["findings"].dataset_content_hash == result1["artifacts"]["findings"].dataset_content_hash


class TestInterruptionAndFailureRecovery:
    def test_interrupted_fetcher_restarts_cleanly_on_resume(self, tmp_path, monkeypatch, duckdb_and_pyarrow):
        source = _write_sales(tmp_path / "sales.csv", _base_rows())
        thread_id = "resume-interrupted-001"
        checkpointer = MemorySaver()

        with pytest.raises(KeyboardInterrupt):
            _invoke(
                source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30",
                checkpointer=checkpointer, fetcher_model=RaisingFetcherModel(),
            )

        # A clean second invoke on the same thread must restart the Fetcher
        # Step (not replay a broken/partial state) and complete successfully.
        result2, _, fetcher2, analyst2 = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30", checkpointer=checkpointer,
        )
        assert result2["artifacts"].get("terminal_failure") is None
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["status"] == "completed"
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_outcome"] == "recomputed"
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_reason"] == "interrupted"
        assert fetcher2.calls == 1
        assert analyst2.calls == 1
        assert result2["artifacts"]["findings"] is not None

    def test_failed_fetcher_restarts_cleanly_on_resume(self, tmp_path, monkeypatch, duckdb_and_pyarrow):
        source = _write_sales(tmp_path / "sales.csv", _base_rows())
        thread_id = "resume-failed-001"
        checkpointer = MemorySaver()

        result1, _, fetcher1, _ = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30",
            checkpointer=checkpointer, fetcher_model=AlwaysInvalidFetcherModel(),
        )
        assert result1["artifacts"]["terminal_failure"]["agent"] == "Fetcher"
        assert result1["artifacts"]["terminal_failure"]["step_id"] == "fetch_sales"
        assert "fetch_sales" not in result1["artifacts"].get("step_outcomes", {})

        result2, _, fetcher2, analyst2 = _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30", checkpointer=checkpointer,
        )
        assert result2["artifacts"].get("terminal_failure") is None
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["status"] == "completed"
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_outcome"] == "recomputed"
        assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_reason"] == "failed"
        assert fetcher2.calls == 1
        assert analyst2.calls == 1
        assert result2["artifacts"]["findings"] is not None


class TestReusedExternalFetchesAvoidDuplicateCalls:
    def test_unchanged_as_of_date_does_not_reissue_target_or_cpi_fetch(self, tmp_path, monkeypatch, duckdb_and_pyarrow):
        source = _write_sales(tmp_path / "sales.csv", _base_rows())
        thread_id = "resume-external-fetch-001"
        checkpointer = MemorySaver()

        target_calls = []
        cpi_calls = []

        def fake_retrieve_targets(start, end):
            target_calls.append((start, end))
            return {"status": "available", "detail": None, "rows": []}

        def fake_retrieve_cpi(start, end):
            cpi_calls.append((start, end))
            return {"status": "available", "detail": None, "observations": [
                {"date": "2025-05-01", "value": 300.0}, {"date": "2025-06-01", "value": 303.0},
            ]}

        monkeypatch.setattr("analyst_crew.tools.targets.retrieve_targets", fake_retrieve_targets)
        monkeypatch.setattr("analyst_crew.tools.fred.retrieve_cpi", fake_retrieve_cpi)

        extra_sources = {
            "sales_targets_postgres": {"kind": "Postgres", "table": "sales_targets", "grain": ["month", "region", "category"]},
            "cpi_fred": {"kind": "FRED", "series_id": "CPIAUCNS"},
        }

        _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30",
            checkpointer=checkpointer, extra_sources=extra_sources,
        )
        assert len(target_calls) == 1
        assert len(cpi_calls) == 1

        _invoke(
            source, tmp_path, monkeypatch, thread_id=thread_id, as_of_date="2025-06-30",
            checkpointer=checkpointer, extra_sources=extra_sources,
        )
        # Same as_of_date -> same comparable-period window -> the cached
        # target/CPI result from the prior Handoff is reused rather than
        # re-querying Postgres / re-calling the FRED API.
        assert len(target_calls) == 1
        assert len(cpi_calls) == 1


class TestPriorPlanReuse:
    """A live Planner words step ids differently on every call, which would
    change every execution key. Resume must reuse the checkpointed Plan."""

    def _plan(self, question="Q"):
        return Plan(
            user_question=question,
            steps=[
                Step(step_id="fetch_sales", agent="Fetcher", objective="fetch", expected_output="ds",
                     source_ids=["sales_csv"], declared_artifacts=["ds"]),
                Step(step_id="analyze", agent="Analyst", objective="analyze", expected_output="findings",
                     declared_artifacts=["findings"], depends_on=["fetch_sales"]),
            ],
        )

    def _agent(self, agents=("Fetcher", "Analyst"), sources=("sales_csv",)):
        class _Agent:
            available_agents = set(agents)
            data_sources = set(sources)
        return _Agent()

    def test_reuses_matching_prior_plan(self):
        plan = self._plan()
        state = {"artifacts": {"plan": plan}}
        assert planner_module._reusable_prior_plan(state, "Q", self._agent()) is plan

    def test_replans_when_question_changes(self):
        state = {"artifacts": {"plan": self._plan("Q")}}
        assert planner_module._reusable_prior_plan(state, "different", self._agent()) is None

    def test_replans_when_capabilities_no_longer_fit(self):
        state = {"artifacts": {"plan": self._plan()}}
        assert planner_module._reusable_prior_plan(state, "Q", self._agent(agents=("Fetcher",))) is None
        assert planner_module._reusable_prior_plan(state, "Q", self._agent(sources=("other",))) is None

    def test_replans_when_no_prior_plan(self):
        assert planner_module._reusable_prior_plan({"artifacts": {}}, "Q", self._agent()) is None
