"""Issue #10 — bounded Agent/Tool failure behavior through the real Planner
-> Fetcher -> Analyst graph.

Covers: exactly-one repair retry for contract-invalid Fetcher/Analyst
output, explicit terminal failure (with retained Plan and diagnostics) once
that retry is also exhausted, retry counts recorded in CrewState, and a
regression proving the Planner is never invoked a second time to produce a
replacement Plan even when Fetcher/Analyst retries occur.
"""
import csv

import pytest

import analyst_crew.agents.planner as planner_module
from analyst_crew.agents import Plan, Step
from analyst_crew.tools.sales import CANONICAL_FIELDS, sales_data_source


class CountingScriptedPlannerModel:
    """Mimics the raw ChatOpenAI seam PlannerAgent binds
    with_structured_output(...) onto, so a full graph.invoke() exercises the
    real planner_node while letting the test count LLM calls."""

    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return self.plan


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


def _runtime(source, tmp_path, *, planner_model, fetcher_model, analyst_model):
    return {
        "configurable": {
            "data_sources": sales_data_source(source),
            "artifact_dir": tmp_path / "artifacts",
            "available_agents": ["Fetcher", "Analyst"],
            "planner_model": planner_model,
            "fetcher_model": fetcher_model,
            "analyst_model": analyst_model,
            "analysis_parameters": {"as_of_date": "2025-06-30"},
        }
    }


class InvalidThenValidFetcherModel:
    """Contract-invalid Fetcher output once, then valid — exactly one
    repair retry with the validation error fed back as feedback."""

    def __init__(self):
        self.calls = []

    def invoke(self, context):
        self.calls.append(context)
        if len(self.calls) == 1:
            return {"tool": "execute_python", "source_id": "sales_csv"}
        assert "Invalid Fetcher request" in context["validation_feedback"]
        return {"tool": "CSV.read", "source_id": "sales_csv"}


class AlwaysInvalidFetcherModel:
    """Contract-invalid Fetcher output on both attempts — the repair retry
    is exhausted too, so the Step terminates explicitly."""

    def __init__(self):
        self.calls = []

    def invoke(self, context):
        self.calls.append(context)
        return {"tool": "execute_python", "source_id": "sales_csv"}


class ScriptedAnalystModel:
    def invoke(self, request):
        artifact_id = next(iter(request["available_artifacts"].values()))
        return {"tool": "DuckDB.sales_baseline", "dataset_artifact_id": artifact_id}


@pytest.mark.usefixtures("monkeypatch")
def test_fetcher_invalid_output_once_then_valid_succeeds_with_one_retry(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    source = _write_sales(tmp_path / "sales.csv", [
        _row("M1", "2025-05-01"), _row("J1", "2025-06-01"),
    ])
    planner_model = CountingScriptedPlannerModel(_plan())
    fetcher_model = InvalidThenValidFetcherModel()
    runtime = _runtime(source, tmp_path, planner_model=planner_model,
                        fetcher_model=fetcher_model, analyst_model=ScriptedAnalystModel())
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")

    result = planner_module.invoke("Compare baseline sales", runtime)

    assert result["artifacts"].get("terminal_failure") is None
    assert result["artifacts"]["step_outcomes"]["fetch_sales"]["status"] == "completed"
    assert result["artifacts"]["step_outcomes"]["fetch_sales"]["retry_counts"] == {
        "validation_retries": 1, "tool_retries": 0,
    }
    assert result["artifacts"]["retry_counts"]["fetch_sales"] == {
        "validation_retries": 1, "tool_retries": 0,
    }
    assert len(fetcher_model.calls) == 2
    # The Plan remains inspectable and unchanged after the retry.
    assert result["artifacts"]["plan"].steps[0].step_id == "fetch_sales"


def test_fetcher_invalid_output_twice_terminates_explicitly_with_diagnostics(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    source = _write_sales(tmp_path / "sales.csv", [
        _row("M1", "2025-05-01"), _row("J1", "2025-06-01"),
    ])
    planner_model = CountingScriptedPlannerModel(_plan())
    fetcher_model = AlwaysInvalidFetcherModel()
    runtime = _runtime(source, tmp_path, planner_model=planner_model,
                        fetcher_model=fetcher_model, analyst_model=ScriptedAnalystModel())
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")

    result = planner_module.invoke("Compare baseline sales", runtime)

    # Exactly one repair retry was attempted (two total calls), then a clean
    # terminal stop for the whole run — no Analyst execution, no second Plan.
    assert len(fetcher_model.calls) == 2
    assert planner_model.calls == 1
    failure = result["artifacts"]["terminal_failure"]
    assert failure["agent"] == "Fetcher"
    assert failure["step_id"] == "fetch_sales"
    assert failure["required"] is True
    assert failure["retry_counts"] == {"validation_retries": 1, "tool_retries": 0}
    assert "disallowed" in failure["error"]
    # No Analyst Step ran — the required-source failure stopped the run.
    assert "analyze_baseline" not in result["artifacts"].get("step_outcomes", {})
    assert result["artifacts"].get("findings") is None
    # The original Plan remains inspectable after the failure.
    assert result["artifacts"]["plan"].steps[0].step_id == "fetch_sales"
    assert result["artifacts"]["plan"].steps[1].step_id == "analyze_baseline"


def test_planner_invoked_exactly_once_even_when_fetcher_and_analyst_retry(tmp_path, monkeypatch):
    """Regression for issue #10: no execution path invokes the Planner a
    second time to produce a replacement Plan, even when a downstream Step
    needs a repair retry."""
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    source = _write_sales(tmp_path / "sales.csv", [
        _row("M1", "2025-05-01"), _row("J1", "2025-06-01"),
    ])
    planner_model = CountingScriptedPlannerModel(_plan())
    fetcher_model = InvalidThenValidFetcherModel()

    class InvalidThenValidAnalystModel:
        def __init__(self):
            self.calls = 0

        def invoke(self, request):
            self.calls += 1
            if self.calls == 1:
                return {"tool": "DuckDB.sql", "dataset_artifact_id": "nope"}
            artifact_id = next(iter(request["available_artifacts"].values()))
            return {"tool": "DuckDB.sales_baseline", "dataset_artifact_id": artifact_id}

    analyst_model = InvalidThenValidAnalystModel()
    runtime = _runtime(source, tmp_path, planner_model=planner_model,
                        fetcher_model=fetcher_model, analyst_model=analyst_model)
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")

    result = planner_module.invoke("Compare baseline sales", runtime)

    assert result["artifacts"].get("terminal_failure") is None
    assert result["artifacts"]["step_outcomes"]["analyze_baseline"]["status"] == "completed"
    assert result["artifacts"]["step_outcomes"]["analyze_baseline"]["retry_counts"]["validation_retries"] == 1
    assert planner_model.calls == 1, "Planner must be invoked exactly once per Crew run"
    # The single Plan the one Planner call produced is still the Plan that ran.
    assert result["artifacts"]["plan"] is planner_model.plan


def test_cli_prints_concise_terminal_failure_and_preserves_diagnostics():
    from cli import format_result

    result = {
        "artifacts": {
            "terminal_failure": {
                "agent": "Fetcher",
                "step_id": "fetch_sales",
                "required": True,
                "status": "failed",
                "error": "disallowed Tool",
                "retry_counts": {"validation_retries": 1, "tool_retries": 0},
            },
            "fetcher_error": "disallowed Tool",
        },
    }
    rendered = format_result(result)
    assert "Error: Fetcher" in rendered
    assert "fetch_sales" in rendered
    assert "disallowed Tool" in rendered
    # Concise: no dump of the full artifacts dict in the printed summary.
    assert "retry_counts" not in rendered
    # But the full diagnostics are still reachable on the returned state.
    assert result["artifacts"]["terminal_failure"]["retry_counts"] == {
        "validation_retries": 1, "tool_retries": 0,
    }
