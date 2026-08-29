import csv
from datetime import date

import pytest

import analyst_crew.agents.planner as planner_module
from analyst_crew.agents import Plan, Step
from analyst_crew.tools.analysis import BaselineFindings, run_sales_baseline
from analyst_crew.tools.sales import CANONICAL_FIELDS, sales_data_source
from analyst_crew.tools.targets import sales_targets_data_source


class ScriptedFetcherModel:
    def invoke(self, request):
        return {"tool": "CSV.read", "source_id": "sales_csv"}


class ScriptedAnalystModel:
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


def _dim_row(order_id, order_date, region, category, quantity, unit_price):
    return {
        "order_id": order_id,
        "order_date": order_date,
        "product_id": "P001",
        "product_name": "Widget",
        "category": category,
        "region": region,
        "sales_channel": "web",
        "customer_segment": "consumer",
        "quantity": quantity,
        "unit_price": unit_price,
        "discount_amount": 0,
        "returned_quantity": 0,
    }


def _write_sales(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _region_category_source(tmp_path):
    return _write_sales(tmp_path / "sales.csv", [
        # North/office: actual net revenue 500 (5 * 100), 1 order.
        _dim_row("J1", "2025-06-01", "North", "Office", 5, 100),
        # South/furniture: actual net revenue 2000 (2 * 1000), 1 order.
        _dim_row("J2", "2025-06-05", "South", "Furniture", 2, 1000),
        # Prior period rows so the core analysis has a comparable baseline.
        _dim_row("M1", "2025-05-01", "North", "Office", 5, 100),
        _dim_row("M2", "2025-05-05", "South", "Furniture", 2, 1000),
    ])


def test_target_comparison_computes_actual_target_variance_and_percent_by_region_category(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _region_category_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    target_data = {
        "status": "available",
        "detail": None,
        "rows": [
            {"target_month": "2025-06-01", "region": "north", "category": "office",
             "target_net_revenue": 400.0, "target_orders": 2},
            {"target_month": "2025-06-01", "region": "south", "category": "furniture",
             "target_net_revenue": 2500.0, "target_orders": 1},
        ],
    }
    findings = run_sales_baseline(
        artifact.as_reference(), as_of_date=date(2025, 6, 30), target_data=target_data,
    )

    assert findings.limitations == []
    comparison = findings.target_comparison
    assert comparison is not None
    assert comparison.source_status.status == "available"
    rows = {(row.region, row.category): row for row in comparison.rows}

    north_office = rows[("north", "office")]
    assert north_office.actual_net_revenue == 500.0
    assert north_office.target_net_revenue == 400.0
    assert north_office.net_revenue_variance == pytest.approx(100.0)
    assert north_office.net_revenue_variance_percent == pytest.approx(25.0)
    assert north_office.actual_orders == 1.0
    assert north_office.target_orders == 2.0
    assert north_office.orders_variance == pytest.approx(-1.0)
    assert north_office.orders_variance_percent == pytest.approx(-50.0)

    south_furniture = rows[("south", "furniture")]
    assert south_furniture.actual_net_revenue == 2000.0
    assert south_furniture.target_net_revenue == 2500.0
    assert south_furniture.net_revenue_variance == pytest.approx(-500.0)
    assert south_furniture.net_revenue_variance_percent == pytest.approx(-20.0)


def test_target_comparison_handles_zero_target_without_invalid_percentage(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _region_category_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    target_data = {
        "status": "available",
        "detail": None,
        "rows": [
            {"target_month": "2025-06-01", "region": "north", "category": "office",
             "target_net_revenue": 0.0, "target_orders": 0},
            {"target_month": "2025-06-01", "region": "south", "category": "furniture",
             "target_net_revenue": 2500.0, "target_orders": 1},
        ],
    }
    findings = run_sales_baseline(
        artifact.as_reference(), as_of_date=date(2025, 6, 30), target_data=target_data,
    )

    north_office = next(row for row in findings.target_comparison.rows if (row.region, row.category) == ("north", "office"))
    assert north_office.target_net_revenue == 0.0
    assert north_office.net_revenue_variance == pytest.approx(500.0)
    assert north_office.net_revenue_variance_percent is None
    assert north_office.target_orders == 0.0
    assert north_office.orders_variance_percent is None


def test_target_comparison_includes_region_category_missing_from_targets_as_zero_target(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _region_category_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    # Only one of the two actual region/category cells has a seeded target.
    target_data = {
        "status": "available",
        "detail": None,
        "rows": [
            {"target_month": "2025-06-01", "region": "north", "category": "office",
             "target_net_revenue": 400.0, "target_orders": 2},
        ],
    }
    findings = run_sales_baseline(
        artifact.as_reference(), as_of_date=date(2025, 6, 30), target_data=target_data,
    )
    south_furniture = next(row for row in findings.target_comparison.rows if (row.region, row.category) == ("south", "furniture"))
    assert south_furniture.target_net_revenue == 0.0
    assert south_furniture.net_revenue_variance_percent is None


def test_target_comparison_is_omitted_when_not_configured(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _region_category_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))
    assert findings.target_comparison is None
    assert findings.limitations == []


def test_target_comparison_records_limitation_and_status_when_source_unavailable_but_csv_findings_still_complete(tmp_path):
    """End-to-end degradation: an unavailable optional Postgres target source
    must never affect the required CSV-based Findings, only add a status and
    a limitation."""
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _region_category_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    target_data = {"status": "unavailable", "detail": "Postgres target query failed: connection refused", "rows": []}
    findings = run_sales_baseline(
        artifact.as_reference(), as_of_date=date(2025, 6, 30), target_data=target_data,
    )

    # Core CSV Findings are fully intact.
    assert findings.net_revenue.current == 2500.0
    assert findings.gross_revenue.current == 2500.0
    assert findings.revenue_bridge is not None

    assert findings.target_comparison is not None
    assert findings.target_comparison.rows == []
    assert findings.target_comparison.source_status.status == "unavailable"
    assert len(findings.limitations) == 1
    assert "unavailable" in findings.limitations[0]


def test_target_comparison_records_invalid_status_when_source_query_errors(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _region_category_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    target_data = {"status": "invalid", "detail": "Postgres target rows are malformed: could not convert string to float", "rows": []}
    findings = run_sales_baseline(
        artifact.as_reference(), as_of_date=date(2025, 6, 30), target_data=target_data,
    )
    assert findings.net_revenue.current == 2500.0
    assert findings.target_comparison.source_status.status == "invalid"
    assert len(findings.limitations) == 1


def test_cli_renders_target_comparison_section():
    from cli import format_result
    from analyst_crew.tools.analysis import ComparisonWindow, MetricComparison, TargetComparison, TargetSourceStatus, TargetVariance

    findings = BaselineFindings(
        window=ComparisonWindow(
            mode="complete_month",
            current_start=date(2025, 6, 1),
            current_end=date(2025, 6, 30),
            prior_start=date(2025, 5, 1),
            prior_end=date(2025, 5, 31),
        ),
        gross_revenue=MetricComparison(current=60, prior=40, absolute_change=20, percent_change=50),
        net_revenue=MetricComparison(current=45, prior=19, absolute_change=26, percent_change=136.8421),
        orders=MetricComparison(current=2, prior=2, absolute_change=0, percent_change=0),
        units=MetricComparison(current=5, prior=3, absolute_change=2, percent_change=66.6667),
        dataset_artifact_id="dataset-1",
        dataset_content_hash="abc123",
        target_comparison=TargetComparison(
            rows=[
                TargetVariance(
                    region="north", category="office",
                    actual_net_revenue=500.0, target_net_revenue=400.0,
                    net_revenue_variance=100.0, net_revenue_variance_percent=25.0,
                    actual_orders=1.0, target_orders=2.0,
                    orders_variance=-1.0, orders_variance_percent=-50.0,
                ),
            ],
            source_status=TargetSourceStatus(status="available"),
        ),
    )
    rendered = format_result({"artifacts": {"findings": findings}})
    assert "Sales targets" in rendered
    assert "north/office" in rendered
    assert "500.00 vs target 400.00" in rendered

    unavailable_findings = findings.model_copy(update={
        "target_comparison": TargetComparison(rows=[], source_status=TargetSourceStatus(status="unavailable", detail="connection refused"))
    })
    rendered_unavailable = format_result({"artifacts": {"findings": unavailable_findings}})
    assert "Sales targets: unavailable" in rendered_unavailable
    assert "connection refused" in rendered_unavailable


def test_full_crew_degrades_gracefully_when_postgres_targets_unavailable(tmp_path, monkeypatch):
    """End-to-end offline degradation test through the real Planner -> Fetcher
    -> Analyst graph: the optional Postgres target Data Source is catalogued
    (so the Analyst node attempts retrieval), but no Postgres is reachable in
    this offline test environment. The required CSV Findings must complete
    exactly as they would without the optional source, plus a recorded
    unavailable status and limitation."""
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    source = _write_sales(tmp_path / "sales.csv", [
        _dim_row("M1", "2025-05-01", "North", "Office", 5, 100),
        _dim_row("J1", "2025-06-01", "North", "Office", 5, 100),
    ])
    fetch = Step(
        step_id="fetch_sales", agent="Fetcher", objective="Fetch normalized sales",
        source_ids=["sales_csv"], expected_output="sales_dataset", declared_artifacts=["sales_dataset"],
    )
    analyze = Step(
        step_id="analyze_baseline", agent="Analyst", objective="Compare baseline sales periods",
        expected_output="baseline_findings", declared_artifacts=["baseline_findings"], depends_on=["fetch_sales"],
    )
    plan = Plan(steps=[fetch, analyze], user_question="Compare baseline sales")
    runtime = {
        "configurable": {
            "data_sources": {**sales_data_source(source), **sales_targets_data_source()},
            "artifact_dir": tmp_path / "artifacts",
            "available_agents": ["Fetcher", "Analyst"],
            "planner_model": ScriptedPlannerModel(plan),
            "fetcher_model": ScriptedFetcherModel(),
            "analyst_model": ScriptedAnalystModel(),
            "analysis_parameters": {"as_of_date": "2025-06-30"},
        }
    }
    # No live Postgres in this offline test — retrieve_targets() must degrade
    # to "unavailable" rather than raise, and the graph run must still complete.
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    result = planner_module.invoke("Compare baseline sales", runtime)

    assert result["artifacts"]["step_outcomes"]["analyze_baseline"]["status"] == "completed"
    findings = result["artifacts"]["findings"]
    assert findings.net_revenue.current == 500.0
    assert findings.revenue_bridge is not None

    assert findings.target_comparison is not None
    assert findings.target_comparison.source_status.status == "unavailable"
    assert findings.target_comparison.rows == []
    assert len(findings.limitations) == 1
    assert "unavailable" in findings.limitations[0]
