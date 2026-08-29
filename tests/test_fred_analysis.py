import csv
from datetime import date

import pytest

import analyst_crew.agents.planner as planner_module
from analyst_crew.agents import Plan, Step
from analyst_crew.tools.analysis import BaselineFindings, NON_CAUSALITY_CAVEAT, run_sales_baseline
from analyst_crew.tools.fred import FRED_SERIES_ID, fred_cpi_data_source
from analyst_crew.tools.sales import CANONICAL_FIELDS, sales_data_source


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


def _write_sales(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _revenue_source(tmp_path):
    # Prior period (May 2025): net revenue 1000. Current period (June 2025):
    # net revenue 1200. Nominal change +200.
    return _write_sales(tmp_path / "sales.csv", [
        _row("M1", "2025-05-01", 100, 10),
        _row("J1", "2025-06-01", 100, 12),
    ])


def test_cpi_adjustment_computes_nominal_and_inflation_adjusted_change(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _revenue_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    cpi_data = {
        "status": "available",
        "detail": None,
        "observations": [
            {"date": "2025-05-01", "value": 300.0},
            {"date": "2025-06-01", "value": 306.0},
        ],
    }
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30), cpi_data=cpi_data)

    assert findings.limitations == []
    adjustment = findings.cpi_adjustment
    assert adjustment is not None
    assert adjustment.series_id == FRED_SERIES_ID
    assert adjustment.source_status.status == "available"
    assert adjustment.nominal_change == pytest.approx(200.0)
    assert adjustment.prior_cpi_value == pytest.approx(300.0)
    assert adjustment.current_cpi_value == pytest.approx(306.0)
    assert adjustment.prior_cpi_observation_dates == ["2025-05-01"]
    assert adjustment.current_cpi_observation_dates == ["2025-06-01"]
    # adjusted_current = 1200 * (300 / 306) = 1176.4706 -> adjusted_change = 176.4706
    assert adjustment.inflation_adjusted_change == pytest.approx(1200 * (300 / 306) - 1000, rel=1e-6)
    assert adjustment.non_causality_caveat == NON_CAUSALITY_CAVEAT
    assert "does not establish causation" in adjustment.non_causality_caveat


def test_cpi_adjustment_is_omitted_when_not_configured(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _revenue_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))
    assert findings.cpi_adjustment is None
    assert findings.limitations == []


def test_cpi_adjustment_records_limitation_and_status_when_source_unavailable_but_csv_findings_still_complete(tmp_path):
    """End-to-end degradation: an unavailable optional FRED source must never
    affect the required CSV-based Findings, only add a status and a
    limitation."""
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _revenue_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    cpi_data = {"status": "unavailable", "detail": "FRED request failed: connection refused", "observations": []}
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30), cpi_data=cpi_data)

    # Core CSV Findings are fully intact.
    assert findings.net_revenue.current == 1200.0
    assert findings.net_revenue.absolute_change == 200.0
    assert findings.revenue_bridge is not None

    assert findings.cpi_adjustment is not None
    assert findings.cpi_adjustment.source_status.status == "unavailable"
    assert findings.cpi_adjustment.inflation_adjusted_change is None
    assert findings.cpi_adjustment.nominal_change == pytest.approx(200.0)
    assert len(findings.limitations) == 1
    assert "unavailable" in findings.limitations[0]


def test_cpi_adjustment_records_limitation_when_observations_do_not_cover_window(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _revenue_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    # Observations that fall outside the May/June comparison window.
    cpi_data = {
        "status": "available",
        "detail": None,
        "observations": [{"date": "2025-01-01", "value": 290.0}],
    }
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30), cpi_data=cpi_data)
    assert findings.cpi_adjustment.source_status.status == "invalid"
    assert findings.cpi_adjustment.inflation_adjusted_change is None
    assert len(findings.limitations) == 1


def test_cpi_adjustment_never_appears_without_the_non_causality_caveat(tmp_path):
    """No Finding may claim CPI caused a dimensional result — the caveat
    string must always accompany an available adjustment."""
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _revenue_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    cpi_data = {
        "status": "available",
        "detail": None,
        "observations": [
            {"date": "2025-05-01", "value": 300.0},
            {"date": "2025-06-01", "value": 306.0},
        ],
    }
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30), cpi_data=cpi_data)
    assert findings.cpi_adjustment.non_causality_caveat
    assert findings.cpi_adjustment.methodology
    for causal_word in ("caused", "due to", "because of"):
        assert causal_word not in findings.cpi_adjustment.methodology.lower()
        assert causal_word not in findings.cpi_adjustment.non_causality_caveat.lower()


def test_full_crew_offline_degrades_fred_cpi_while_csv_findings_complete(tmp_path, monkeypatch):
    """Full Planner -> Fetcher -> Analyst graph, offline: with FRED_API_KEY
    unset, the optional CPI enrichment records a not_configured status and a
    limitation while the required CSV-based Findings still complete."""
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")

    source = _revenue_source(tmp_path)
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
    plan = Plan(steps=[fetch, analyze], user_question="Compare baseline sales with inflation context")
    runtime = {
        "configurable": {
            "data_sources": {**sales_data_source(source), **fred_cpi_data_source()},
            "artifact_dir": tmp_path / "artifacts",
            "available_agents": ["Fetcher", "Analyst"],
            "planner_model": ScriptedPlannerModel(plan),
            "fetcher_model": ScriptedFetcherModel(),
            "analyst_model": ScriptedAnalystModel(),
            "analysis_parameters": {"as_of_date": "2025-06-30"},
        }
    }
    result = planner_module.invoke("Compare baseline sales with inflation context", runtime)

    assert result["steps_completed"] == ["fetch_sales", "analyze_baseline"]
    findings = result["artifacts"]["findings"]
    assert isinstance(findings, BaselineFindings)
    # Required CSV Findings are unaffected by the degraded optional source.
    assert findings.net_revenue.current == 1200.0
    assert findings.net_revenue.absolute_change == 200.0
    assert findings.revenue_bridge is not None

    assert findings.cpi_adjustment is not None
    assert findings.cpi_adjustment.source_status.status == "not_configured"
    assert findings.cpi_adjustment.inflation_adjusted_change is None
    assert any("FRED" in limitation for limitation in findings.limitations)
