import csv
import hashlib
from datetime import date

import pytest

import analyst_crew.agents.planner as planner_module
from analyst_crew.agents import Plan, PlannerAgent, Step
from analyst_crew.tools.analysis import (
    AnalystAgent,
    BaselineFindings,
    ComparisonWindow,
    MetricComparison,
    run_sales_baseline,
)
from analyst_crew.tools.sales import CANONICAL_FIELDS, sales_data_source


class ScriptedFetcherModel:
    def invoke(self, request):
        return {"tool": "CSV.read", "source_id": "sales_csv"}


class ScriptedAnalystModel:
    def __init__(self, artifact_id):
        self.artifact_id = artifact_id

    def invoke(self, request):
        artifact_id = next(iter(request["available_artifacts"].values()))
        return {"tool": "DuckDB.sales_baseline", "dataset_artifact_id": artifact_id}


class ScriptedPlannerModel:
    def __init__(self, plan):
        self.plan = plan

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _messages):
        return self.plan


def _write_sales(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(order_id, order_date, quantity, unit_price, discount=0, returned=0):
    return {
        "order_id": order_id,
        "order_date": order_date,
        "product_id": "P001",
        "product_name": "Desk Lamp",
        "category": "Office",
        "region": "North",
        "sales_channel": "web",
        "customer_segment": "consumer",
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_amount": discount,
        "returned_quantity": returned,
    }


def test_complete_month_baseline_tool_and_integrity(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    source = _write_sales(tmp_path / "sales.csv", [
        _row("M1", "2025-05-01", 2, 10, 1),
        _row("M2", "2025-05-15", 1, 20, 0, 1),
        _row("J1", "2025-06-01", 3, 10, 0, 1),
        _row("J2", "2025-06-30", 2, 15, 5),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    assert findings.window.mode == "complete_month"
    assert findings.gross_revenue.current == 60
    assert findings.gross_revenue.prior == 40
    assert findings.net_revenue.current == 45
    assert findings.net_revenue.prior == 19
    assert findings.orders.current == 2
    assert findings.units.current == 5
    assert findings.integrity_verified

    with open(artifact.location, "ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))


def test_incomplete_month_uses_comparable_prior_mtd(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    source = _write_sales(tmp_path / "sales.csv", [
        _row("M1", "2025-05-01", 1, 10),
        _row("M2", "2025-05-20", 10, 10),
        _row("J1", "2025-06-15", 2, 10),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 15))
    assert findings.window.mode == "comparable_mtd"
    assert findings.window.prior_end.isoformat() == "2025-05-15"
    assert findings.gross_revenue.current == 20
    assert findings.gross_revenue.prior == 10


def test_explicit_month_end_coverage_marks_month_complete_without_final_day_order(tmp_path):
    from analyst_crew.tools.sales import fetch_csv

    source = _write_sales(tmp_path / "sales.csv", [
        _row("M1", "2025-05-01", 1, 10),
        _row("M2", "2025-05-31", 1, 10),
        _row("J1", "2025-06-29", 2, 10),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")

    findings = run_sales_baseline(
        artifact.as_reference(),
        as_of_date=date(2025, 6, 30),
    )

    assert findings.window.mode == "complete_month"
    assert findings.window.current_end == date(2025, 6, 30)
    assert findings.window.prior_end == date(2025, 5, 31)


def test_full_crew_records_handoffs_outcomes_and_deterministic_findings(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    source = _write_sales(tmp_path / "sales.csv", [
        _row("M1", "2025-05-01", 2, 10, 1),
        _row("M2", "2025-05-15", 1, 20, 0, 1),
        _row("J1", "2025-06-01", 3, 10, 0, 1),
        _row("J2", "2025-06-30", 2, 15, 5),
    ])
    fetch = Step(
        step_id="fetch_sales",
        agent="Fetcher",
        objective="Fetch normalized sales",
        source_ids=["sales_csv"],
        expected_output="sales_dataset",
        declared_artifacts=["sales_dataset"],
    )
    analyze = Step(
        step_id="analyze_baseline",
        agent="Analyst",
        objective="Compare baseline sales periods",
        expected_output="baseline_findings",
        declared_artifacts=["baseline_findings"],
        depends_on=["fetch_sales"],
    )
    plan = Plan(steps=[fetch, analyze], user_question="Compare baseline sales")
    runtime = {
        "configurable": {
            "data_sources": sales_data_source(source),
            "artifact_dir": tmp_path / "artifacts",
            "available_agents": ["Fetcher", "Analyst"],
            "planner_model": ScriptedPlannerModel(plan),
            "fetcher_model": ScriptedFetcherModel(),
            "analyst_model": ScriptedAnalystModel("unused"),
            "analysis_parameters": {"as_of_date": "2025-06-30"},
        }
    }
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    result = planner_module.invoke("Compare baseline sales", runtime)

    assert result["steps_completed"] == ["fetch_sales", "analyze_baseline"]
    assert set(result["artifacts"]["handoffs"]) == {"fetch_sales", "analyze_baseline"}
    assert result["artifacts"]["step_outcomes"]["fetch_sales"]["status"] == "completed"
    assert result["artifacts"]["step_outcomes"]["analyze_baseline"]["status"] == "completed"
    assert result["artifacts"]["step_outcomes"]["analyze_baseline"]["artifact_ids"] == ["baseline_findings"]
    findings = result["artifacts"]["findings"]
    assert findings.net_revenue.current == 45
    assert findings.net_revenue.prior == 19
    assert findings.gross_revenue.absolute_change == 20
    assert findings.orders.absolute_change == 0
    assert findings.units.absolute_change == 2
    assert findings.dataset_content_hash == hashlib.sha256(
        open(result["artifacts"]["datasets"][findings.dataset_artifact_id]["location"], "rb").read()
    ).hexdigest()


def test_analyst_rejects_model_sql_and_repairs_invalid_request():
    handoff = type("Handoff", (), {
        "step": type("Step", (), {"objective": "baseline"})(),
        "user_question": "q",
        "analysis_parameters": {"as_of_date": "2025-06-30"},
        "input_artifacts": {"sales": {
            "kind": "Dataset Artifact",
            "artifact_id": "dataset-1",
        }},
    })()

    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, request):
            self.calls += 1
            if self.calls == 1:
                return {"tool": "DuckDB.sql", "sql": "DROP TABLE sales"}
            return {"tool": "DuckDB.sales_baseline", "dataset_artifact_id": "dataset-1"}

    model = Model()
    sentinel = object()
    assert AnalystAgent(model, lambda reference, **_kwargs: sentinel).run(handoff) is sentinel
    assert model.calls == 2


def test_planner_enforces_phase_one_fetcher_then_analyst_contract():
    fetch = Step(
        step_id="fetch_sales",
        agent="Fetcher",
        objective="Fetch sales",
        source_ids=["sales_csv"],
        expected_output="sales_dataset",
        declared_artifacts=["sales_dataset"],
    )
    analyst = Step(
        step_id="analyze_sales",
        agent="Analyst",
        objective="Compare periods",
        expected_output="baseline_findings",
        declared_artifacts=["baseline_findings"],
        depends_on=["fetch_sales"],
    )
    valid_plan = Plan(steps=[fetch, analyst], user_question="Compare sales")
    planner = PlannerAgent(
        llm=ScriptedPlannerModel(valid_plan),
        available_agents=["Fetcher", "Analyst"],
        data_sources=["sales_csv"],
    )
    assert planner.generate_plan("Compare sales").steps == [fetch, analyst]

    invalid_plan = Plan(steps=[fetch], user_question="Compare sales")
    invalid_planner = PlannerAgent(
        llm=ScriptedPlannerModel(invalid_plan),
        available_agents=["Fetcher", "Analyst"],
        data_sources=["sales_csv"],
    )
    with pytest.raises(ValueError, match="exactly Fetcher then Analyst"):
        invalid_planner.generate_plan("Compare sales")


def test_cli_renders_validated_findings_as_concise_interpretation():
    from cli import format_result

    findings = BaselineFindings(
        window=ComparisonWindow(
            mode="complete_month",
            current_start=date(2025, 6, 1),
            current_end=date(2025, 6, 30),
            prior_start=date(2025, 5, 1),
            prior_end=date(2025, 5, 31),
        ),
        gross_revenue=MetricComparison(
            current=60,
            prior=40,
            absolute_change=20,
            percent_change=50,
        ),
        net_revenue=MetricComparison(
            current=45,
            prior=19,
            absolute_change=26,
            percent_change=136.8421,
        ),
        orders=MetricComparison(
            current=2,
            prior=2,
            absolute_change=0,
            percent_change=0,
        ),
        units=MetricComparison(
            current=5,
            prior=3,
            absolute_change=2,
            percent_change=66.6667,
        ),
        dataset_artifact_id="dataset-1",
        dataset_content_hash="abc123",
    )

    rendered = format_result({"artifacts": {"findings": findings}})

    assert "Baseline sales Findings (complete_month)" in rendered
    assert "2025-06-01 to 2025-06-30" in rendered
    assert "Net revenue 45.00 vs 19.00 (+26.00, +136.8%)" in rendered
    assert "gross revenue 60.00 vs 40.00 (+20.00, +50.0%)" in rendered
    assert "orders 2 vs 2 (+0, +0.0%)" in rendered
    assert "units 5 vs 3 (+2, +66.7%)" in rendered
