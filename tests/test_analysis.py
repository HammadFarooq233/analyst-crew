import csv
import hashlib
from datetime import date

import pytest

import analyst_crew.agents.planner as planner_module
from analyst_crew.agents import Plan, PlannerAgent, Step
from analyst_crew.tools.analysis import (
    DEFAULT_MATERIALITY_ABSOLUTE_PERCENT_OF_PRIOR_TOTAL,
    DEFAULT_MATERIALITY_RELATIVE_PERCENT,
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


def _dim_row(order_id, order_date, product_id, product_name, category, region, channel, segment,
             quantity, unit_price, discount=0, returned=0):
    return {
        "order_id": order_id,
        "order_date": order_date,
        "product_id": product_id,
        "product_name": product_name,
        "category": category,
        "region": region,
        "sales_channel": channel,
        "customer_segment": segment,
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

    # The revenue bridge is exposed through the same validated Findings the
    # full Crew invocation produces, not computed separately.
    bridge = findings.revenue_bridge
    assert bridge is not None
    assert bridge.continuing_product_count == 1
    assert bridge.new_product_count == 0
    assert bridge.discontinued_product_count == 0
    assert bridge.volume_effect == pytest.approx(80 / 3, abs=0.001)
    assert bridge.price_effect == pytest.approx(-4.0, abs=0.001)
    assert bridge.interaction_effect == pytest.approx(-8 / 3, abs=0.001)
    assert bridge.discount_effect == pytest.approx(-4.0)
    assert bridge.return_effect == pytest.approx(10.0)
    assert bridge.total_effect == pytest.approx(findings.net_revenue.absolute_change, abs=0.01)
    assert bridge.reconciliation_gap <= 0.01


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


def _dimension_scenario_source(tmp_path):
    return _write_sales(tmp_path / "sales.csv", [
        # P001/Office/North/web/consumer: unchanged 200 -> 200, never material.
        _dim_row("M1", "2025-05-01", "P001", "Desk Lamp", "Office", "North", "web", "consumer", 2, 100),
        _dim_row("J1", "2025-06-01", "P001", "Desk Lamp", "Office", "North", "web", "consumer", 2, 100),
        # P003/Technology/East/marketplace/small_business: 1000 -> 1200, +20% relative.
        _dim_row("M2", "2025-05-05", "P003", "Monitor", "Technology", "East", "marketplace", "small_business", 10, 100),
        _dim_row("J2", "2025-06-05", "P003", "Monitor", "Technology", "East", "marketplace", "small_business", 12, 100),
        # P002/Furniture/South/store/enterprise: new in June, no May baseline at all.
        _dim_row("J3", "2025-06-10", "P002", "Notebook", "Furniture", "South", "store", "enterprise", 10, 50),
    ])


def test_dimension_breakdown_flags_material_changes_by_default(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _dimension_scenario_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    assert findings.net_revenue.current == 1900
    assert findings.net_revenue.prior == 1200
    assert findings.net_revenue.absolute_change == 700

    assert findings.materiality.relative_change_percent == DEFAULT_MATERIALITY_RELATIVE_PERCENT
    assert findings.materiality.absolute_change_percent_of_prior_total == DEFAULT_MATERIALITY_ABSOLUTE_PERCENT_OF_PRIOR_TOTAL
    assert findings.materiality.source == "default"

    assert set(findings.dimensions) == {"product", "category", "region", "sales_channel", "customer_segment"}

    products = {driver.value: driver for driver in findings.dimensions["product"]}
    assert products["Desk Lamp"].current == 200
    assert products["Desk Lamp"].prior == 200
    assert products["Desk Lamp"].absolute_change == 0
    assert products["Desk Lamp"].percent_change == 0
    assert products["Desk Lamp"].is_material is False

    monitor = products["Monitor"]
    assert monitor.current == 1200 and monitor.prior == 1000
    assert monitor.absolute_change == 200
    assert monitor.percent_change == pytest.approx(20.0)
    assert monitor.prior_baseline_missing is False
    assert monitor.contribution_percent == pytest.approx(200 / 700 * 100)
    assert monitor.is_material is True

    notebook = products["Notebook"]
    assert notebook.current == 500 and notebook.prior == 0
    assert notebook.absolute_change == 500
    assert notebook.percent_change is None
    assert notebook.prior_baseline_missing is True
    assert notebook.contribution_percent == pytest.approx(500 / 700 * 100)
    assert notebook.is_material is True

    # Every driver carries a deterministic id and its source Dataset Artifact.
    for driver in findings.dimensions["product"]:
        assert driver.dataset_artifact_id == findings.dataset_artifact_id
        assert len(driver.aggregation_id) == 16

    # Dimensional totals reconcile with the overall net revenue change.
    for dimension, drivers in findings.dimensions.items():
        total = sum(driver.absolute_change for driver in drivers)
        assert total == pytest.approx(findings.net_revenue.absolute_change, abs=0.01), dimension

    material_keys = {(driver.dimension, driver.value) for driver in findings.material_changes}
    assert ("product", "Monitor") in material_keys
    assert ("product", "Notebook") in material_keys
    assert ("product", "Desk Lamp") not in material_keys
    assert ("category", "furniture") in material_keys
    assert ("category", "technology") in material_keys
    assert ("category", "office") not in material_keys


def test_dimension_aggregation_id_is_deterministic_per_dataset_window_and_value(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _dimension_scenario_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    first = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))
    second = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    first_ids = {(d.dimension, d.value): d.aggregation_id for d in first.dimensions["product"]}
    second_ids = {(d.dimension, d.value): d.aggregation_id for d in second.dimensions["product"]}
    assert first_ids == second_ids
    assert len(set(first_ids.values())) == len(first_ids)


def test_analysis_parameters_override_materiality_thresholds(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _dimension_scenario_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(
        artifact.as_reference(),
        as_of_date=date(2025, 6, 30),
        analysis_parameters={"materiality_absolute_threshold_percent": 100},
    )
    assert findings.materiality.source == "analysis_parameters"
    assert findings.materiality.absolute_change_percent_of_prior_total == 100
    assert findings.materiality.relative_change_percent == DEFAULT_MATERIALITY_RELATIVE_PERCENT
    # 100% of the prior period total (1200) is 1200, larger than either driver's change.
    assert findings.material_changes == []


def test_user_question_override_raises_relative_threshold_and_defaults_absolute(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _dimension_scenario_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(
        artifact.as_reference(),
        as_of_date=date(2025, 6, 30),
        user_question="Only flag material changes with at least 25% relative change this month.",
    )
    assert findings.materiality.source == "user_question"
    assert findings.materiality.relative_change_percent == 25.0
    assert findings.materiality.absolute_change_percent_of_prior_total == DEFAULT_MATERIALITY_ABSOLUTE_PERCENT_OF_PRIOR_TOTAL

    material_keys = {(driver.dimension, driver.value) for driver in findings.material_changes}
    # Monitor's 20% relative change no longer clears the raised 25% bar.
    assert ("product", "Monitor") not in material_keys
    # Notebook has no prior baseline, so the relative test is satisfied by
    # convention and it stays material on the absolute-impact test alone.
    assert ("product", "Notebook") in material_keys


def test_analysis_parameters_take_priority_over_user_question_wording():
    from analyst_crew.tools.analysis import _resolve_materiality

    thresholds = _resolve_materiality(
        {"materiality_relative_threshold_percent": 10},
        "Flag changes with at least 40% relative change and 40% absolute impact.",
    )
    assert thresholds.source == "analysis_parameters"
    assert thresholds.relative_change_percent == 10
    assert thresholds.absolute_change_percent_of_prior_total == DEFAULT_MATERIALITY_ABSOLUTE_PERCENT_OF_PRIOR_TOTAL


def test_zero_but_present_prior_baseline_is_not_reported_as_invalid_percentage(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    # A discount fully offsets gross revenue in May, so the product's net
    # revenue is exactly zero while still present in the prior period.
    source = _write_sales(tmp_path / "sales.csv", [
        _dim_row("M1", "2025-05-01", "P004", "Clearance Item", "Office", "North", "web", "consumer", 1, 50, discount=50),
        _dim_row("J1", "2025-06-01", "P004", "Clearance Item", "Office", "North", "web", "consumer", 1, 50),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    driver = findings.dimensions["product"][0]
    assert driver.prior == 0
    assert driver.current == 50
    assert driver.percent_change is None
    assert driver.prior_baseline_missing is True


def test_analyst_agent_forwards_analysis_parameters_and_question_to_tool():
    handoff = type("Handoff", (), {
        "step": type("Step", (), {"objective": "baseline"})(),
        "user_question": "Flag changes with at least 30% relative change",
        "analysis_parameters": {"as_of_date": "2025-06-30", "materiality_absolute_threshold_percent": 12},
        "input_artifacts": {"sales": {
            "kind": "Dataset Artifact",
            "artifact_id": "dataset-1",
        }},
    })()

    class Model:
        def invoke(self, request):
            return {"tool": "DuckDB.sales_baseline", "dataset_artifact_id": "dataset-1"}

    captured = {}

    def tool(reference, **kwargs):
        captured.update(kwargs)
        return "sentinel"

    assert AnalystAgent(Model(), tool).run(handoff) == "sentinel"
    assert captured["as_of_date"] == "2025-06-30"
    assert captured["analysis_parameters"] == {"as_of_date": "2025-06-30", "materiality_absolute_threshold_percent": 12}
    assert captured["user_question"] == "Flag changes with at least 30% relative change"


def test_cli_material_changes_section_highlights_only_material_drivers():
    from cli import format_result
    from analyst_crew.tools.analysis import DimensionDriver, MaterialityThresholds

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
        materiality=MaterialityThresholds(),
        dimensions={
            "product": [
                DimensionDriver(
                    dimension="product", value="Desk Lamp", current=10, prior=10, absolute_change=0,
                    percent_change=0, prior_baseline_missing=False, contribution_percent=0,
                    is_material=False, aggregation_id="a" * 16, dataset_artifact_id="dataset-1",
                ),
                DimensionDriver(
                    dimension="product", value="Monitor", current=35, prior=9, absolute_change=26,
                    percent_change=288.8889, prior_baseline_missing=False, contribution_percent=100.0,
                    is_material=True, aggregation_id="b" * 16, dataset_artifact_id="dataset-1",
                ),
            ],
        },
        material_changes=[
            DimensionDriver(
                dimension="product", value="Monitor", current=35, prior=9, absolute_change=26,
                percent_change=288.8889, prior_baseline_missing=False, contribution_percent=100.0,
                is_material=True, aggregation_id="b" * 16, dataset_artifact_id="dataset-1",
            ),
        ],
    )

    rendered = format_result({"artifacts": {"findings": findings}})
    assert "Material Changes" in rendered
    assert "product=Monitor" in rendered
    assert "product=Desk Lamp" not in rendered


def _bridge_row(order_id, order_date, product_id, quantity, unit_price, discount=0, returned=0, product_name="Product"):
    return _dim_row(
        order_id, order_date, product_id, product_name, "Category", "North", "web", "consumer",
        quantity, unit_price, discount, returned,
    )


def test_revenue_bridge_isolates_pure_price_effect(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    # Same quantity (5) both periods, price rises from 10 to 12: pure price effect.
    source = _write_sales(tmp_path / "sales.csv", [
        _bridge_row("M1", "2025-05-01", "P100", 5, 10),
        _bridge_row("J1", "2025-06-01", "P100", 5, 12),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    bridge = findings.revenue_bridge
    assert bridge.volume_effect == pytest.approx(0.0)
    assert bridge.price_effect == pytest.approx(10.0)
    assert bridge.interaction_effect == pytest.approx(0.0)
    assert bridge.discount_effect == pytest.approx(0.0)
    assert bridge.return_effect == pytest.approx(0.0)
    assert bridge.new_product_effect == pytest.approx(0.0)
    assert bridge.discontinued_product_effect == pytest.approx(0.0)
    assert bridge.continuing_product_count == 1
    assert bridge.new_product_count == 0
    assert bridge.discontinued_product_count == 0


def test_revenue_bridge_isolates_pure_volume_effect(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    # Same price (10) both periods, quantity rises from 5 to 8: pure volume effect.
    source = _write_sales(tmp_path / "sales.csv", [
        _bridge_row("M1", "2025-05-01", "P100", 5, 10),
        _bridge_row("J1", "2025-06-01", "P100", 8, 10),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    bridge = findings.revenue_bridge
    assert bridge.volume_effect == pytest.approx(30.0)
    assert bridge.price_effect == pytest.approx(0.0)
    assert bridge.interaction_effect == pytest.approx(0.0)


def test_revenue_bridge_computes_price_volume_interaction(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    # Quantity 5 -> 8, price 10 -> 12: both volume and price move, so a distinct
    # interaction term is required to make the three effects sum exactly to the
    # gross-revenue change (96 - 50 = 46).
    source = _write_sales(tmp_path / "sales.csv", [
        _bridge_row("M1", "2025-05-01", "P100", 5, 10),
        _bridge_row("J1", "2025-06-01", "P100", 8, 12),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    bridge = findings.revenue_bridge
    assert bridge.volume_effect == pytest.approx(30.0)
    assert bridge.price_effect == pytest.approx(10.0)
    assert bridge.interaction_effect == pytest.approx(6.0)
    assert bridge.volume_effect + bridge.price_effect + bridge.interaction_effect == pytest.approx(46.0)


def test_revenue_bridge_isolates_discount_effect(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    # Same quantity and price both periods; only the discount amount changes.
    source = _write_sales(tmp_path / "sales.csv", [
        _bridge_row("M1", "2025-05-01", "P100", 5, 10, discount=5),
        _bridge_row("J1", "2025-06-01", "P100", 5, 10, discount=15),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    bridge = findings.revenue_bridge
    assert bridge.volume_effect == pytest.approx(0.0)
    assert bridge.price_effect == pytest.approx(0.0)
    assert bridge.interaction_effect == pytest.approx(0.0)
    assert bridge.discount_effect == pytest.approx(-10.0)
    assert bridge.return_effect == pytest.approx(0.0)
    assert bridge.total_effect == pytest.approx(findings.net_revenue.absolute_change)


def test_revenue_bridge_isolates_return_effect(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    # Same quantity and price both periods; only the returned quantity changes.
    source = _write_sales(tmp_path / "sales.csv", [
        _bridge_row("M1", "2025-05-01", "P100", 5, 10, returned=1),
        _bridge_row("J1", "2025-06-01", "P100", 5, 10, returned=2),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    bridge = findings.revenue_bridge
    assert bridge.volume_effect == pytest.approx(0.0)
    assert bridge.price_effect == pytest.approx(0.0)
    assert bridge.interaction_effect == pytest.approx(0.0)
    assert bridge.discount_effect == pytest.approx(0.0)
    # Returned value goes from 1*10=10 to 2*10=20, a 10 increase, so a -10 effect.
    assert bridge.return_effect == pytest.approx(-10.0)


def test_revenue_bridge_routes_launched_product_to_new_product_effect(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    # P100 continues unchanged; P200 only appears in June (no May baseline at all).
    source = _write_sales(tmp_path / "sales.csv", [
        _bridge_row("M1", "2025-05-01", "P100", 5, 10),
        _bridge_row("J1", "2025-06-01", "P100", 5, 10),
        _bridge_row("J2", "2025-06-05", "P200", 4, 25),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    bridge = findings.revenue_bridge
    # P200 must not be forced through the price/volume/interaction formulas —
    # continuing-product effects come from P100 alone (all zero, since unchanged).
    assert bridge.volume_effect == pytest.approx(0.0)
    assert bridge.price_effect == pytest.approx(0.0)
    assert bridge.interaction_effect == pytest.approx(0.0)
    assert bridge.new_product_effect == pytest.approx(100.0)  # 4 * 25
    assert bridge.discontinued_product_effect == pytest.approx(0.0)
    assert bridge.new_product_count == 1
    assert bridge.continuing_product_count == 1
    assert bridge.discontinued_product_count == 0


def test_revenue_bridge_routes_discontinued_product_to_discontinued_effect(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    # P100 continues unchanged; P300 only appears in May (discontinued by June).
    source = _write_sales(tmp_path / "sales.csv", [
        _bridge_row("M1", "2025-05-01", "P100", 5, 10),
        _bridge_row("M2", "2025-05-05", "P300", 3, 20),
        _bridge_row("J1", "2025-06-01", "P100", 5, 10),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    bridge = findings.revenue_bridge
    assert bridge.volume_effect == pytest.approx(0.0)
    assert bridge.price_effect == pytest.approx(0.0)
    assert bridge.interaction_effect == pytest.approx(0.0)
    assert bridge.new_product_effect == pytest.approx(0.0)
    assert bridge.discontinued_product_effect == pytest.approx(-60.0)  # -(3 * 20)
    assert bridge.discontinued_product_count == 1
    assert bridge.continuing_product_count == 1
    assert bridge.new_product_count == 0


def test_revenue_bridge_reconciles_to_net_revenue_change_with_mixed_scenario(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    source = _write_sales(tmp_path / "sales.csv", [
        # Continuing product with price, volume, discount, and return movement.
        _bridge_row("M1", "2025-05-01", "P100", 5, 10, discount=5, returned=1),
        _bridge_row("J1", "2025-06-01", "P100", 8, 12, discount=15, returned=2),
        # Launched in June.
        _bridge_row("J2", "2025-06-05", "P200", 4, 25),
        # Discontinued after May.
        _bridge_row("M2", "2025-05-05", "P300", 3, 20),
    ])
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    findings = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    bridge = findings.revenue_bridge
    total = (
        bridge.volume_effect + bridge.price_effect + bridge.interaction_effect
        + bridge.discount_effect + bridge.return_effect
        + bridge.new_product_effect + bridge.discontinued_product_effect
    )
    assert total == pytest.approx(findings.net_revenue.absolute_change, abs=0.01)
    assert bridge.total_effect == pytest.approx(findings.net_revenue.absolute_change, abs=0.01)
    assert bridge.reconciliation_gap <= 0.01
    assert bridge.continuing_product_count == 1
    assert bridge.new_product_count == 1
    assert bridge.discontinued_product_count == 1
    assert bridge.methodology
    assert bridge.limitations
    assert bridge.source_references
    assert bridge.dataset_artifact_id == findings.dataset_artifact_id


def test_revenue_bridge_aggregation_id_is_deterministic(tmp_path):
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    from analyst_crew.tools.sales import fetch_csv

    artifact = fetch_csv("sales_csv", _dimension_scenario_source(tmp_path), artifact_dir=tmp_path / "artifacts")
    first = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))
    second = run_sales_baseline(artifact.as_reference(), as_of_date=date(2025, 6, 30))

    assert first.revenue_bridge.aggregation_id == second.revenue_bridge.aggregation_id
    assert len(first.revenue_bridge.aggregation_id) == 16


def test_cli_renders_revenue_bridge_concisely():
    from cli import format_result
    from analyst_crew.tools.analysis import RevenueBridge

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
        revenue_bridge=RevenueBridge(
            volume_effect=26.6667,
            price_effect=-4.0,
            interaction_effect=-2.6667,
            discount_effect=-4.0,
            return_effect=10.0,
            new_product_effect=0.0,
            discontinued_product_effect=0.0,
            total_effect=26.0,
            reconciliation_gap=0.0,
            continuing_product_count=1,
            new_product_count=0,
            discontinued_product_count=0,
            source_references=["Dataset Artifact dataset-1"],
            dataset_artifact_id="dataset-1",
            aggregation_id="a" * 16,
        ),
    )

    rendered = format_result({"artifacts": {"findings": findings}})
    assert "Revenue bridge" in rendered
    assert "volume: +26.67" in rendered
    assert "price: -4.00" in rendered
    assert "price-volume interaction: -2.67" in rendered
    assert "discount: -4.00" in rendered
    assert "return: +10.00" in rendered
    assert "total: +26.00" in rendered


def test_cli_material_changes_section_states_when_nothing_is_material():
    from cli import format_result

    findings = BaselineFindings(
        window=ComparisonWindow(
            mode="complete_month",
            current_start=date(2025, 6, 1),
            current_end=date(2025, 6, 30),
            prior_start=date(2025, 5, 1),
            prior_end=date(2025, 5, 31),
        ),
        gross_revenue=MetricComparison(current=60, prior=59, absolute_change=1, percent_change=1.7),
        net_revenue=MetricComparison(current=45, prior=44, absolute_change=1, percent_change=2.3),
        orders=MetricComparison(current=2, prior=2, absolute_change=0, percent_change=0),
        units=MetricComparison(current=5, prior=5, absolute_change=0, percent_change=0),
        dataset_artifact_id="dataset-1",
        dataset_content_hash="abc123",
    )

    rendered = format_result({"artifacts": {"findings": findings}})
    assert "none" in rendered.lower()
