"""Hand-built ``BaselineFindings`` for pure (model-free, DuckDB-free) tests of
the reporting layer. Values are chosen so display strings are easy to state as
literals in assertions."""
from datetime import date

from analyst_crew.tools.analysis import (
    BaselineFindings, ComparisonWindow, CpiAdjustment, CpiSourceStatus, DimensionDriver, MaterialityThresholds,
    MetricComparison, RevenueBridge, TargetComparison, TargetSourceStatus, TargetVariance,
)


def metric(current, prior):
    change = current - prior
    return MetricComparison(
        current=current, prior=prior, absolute_change=change,
        percent_change=None if prior == 0 else change / prior * 100,
    )


def driver(dimension, value, current, prior, *, overall_change=200.0, material=False):
    change = current - prior
    return DimensionDriver(
        dimension=dimension, value=value, current=current, prior=prior, absolute_change=change,
        percent_change=None if prior == 0 else change / prior * 100,
        prior_baseline_missing=prior == 0,
        contribution_percent=None if overall_change == 0 else change / overall_change * 100,
        is_material=material, aggregation_id=f"agg-{dimension}-{value}", dataset_artifact_id="dataset-abc",
    )


def bridge():
    return RevenueBridge(
        volume_effect=150.0, price_effect=30.0, interaction_effect=5.0, discount_effect=-10.0,
        return_effect=-5.0, new_product_effect=40.0, discontinued_product_effect=-10.0,
        total_effect=200.0, reconciliation_gap=0.0, continuing_product_count=4, new_product_count=1,
        discontinued_product_count=1, source_references=["Dataset Artifact dataset-abc"],
        dataset_artifact_id="dataset-abc", aggregation_id="bridge-agg",
    )


def target_comparison(status="available", detail=None, rows=True):
    return TargetComparison(
        rows=[TargetVariance(
            region="north", category="office", actual_net_revenue=900.0, target_net_revenue=1000.0,
            net_revenue_variance=-100.0, net_revenue_variance_percent=-10.0,
            actual_orders=45.0, target_orders=50.0, orders_variance=-5.0, orders_variance_percent=-10.0,
        )] if rows and status == "available" else [],
        source_status=TargetSourceStatus(status=status, detail=detail),
    )


def cpi(status="available", detail=None):
    if status != "available":
        return CpiAdjustment(
            series_id="CPIAUCNS", nominal_change=200.0, inflation_adjusted_change=None,
            prior_cpi_value=None, current_cpi_value=None,
            source_status=CpiSourceStatus(status=status, detail=detail),
        )
    return CpiAdjustment(
        series_id="CPIAUCNS", nominal_change=200.0, inflation_adjusted_change=150.0,
        prior_cpi_value=300.0, current_cpi_value=306.0,
        prior_cpi_observation_dates=["2025-05-01"], current_cpi_observation_dates=["2025-06-01"],
        source_status=CpiSourceStatus(status="available"),
    )


def make_findings(*, targets=None, cpi_adjustment=None, limitations=None, dimensions=None,
                  material_changes=None, quarantined=3, duplicates=1) -> BaselineFindings:
    """Net revenue 1,200.00 vs 1,000.00 (+200.00, +20.0%), complete-month window."""
    if dimensions is None:
        region_north = driver("region", "North", 700.0, 500.0, material=True)
        region_south = driver("region", "South", 500.0, 500.0)
        product_lamp = driver("product", "Desk Lamp 2000", 300.0, 0.0, material=True)
        dimensions = {"region": [region_north, region_south], "product": [product_lamp]}
        if material_changes is None:
            material_changes = [region_north, product_lamp]
    return BaselineFindings(
        window=ComparisonWindow(
            mode="complete_month", current_start=date(2025, 6, 1), current_end=date(2025, 6, 30),
            prior_start=date(2025, 5, 1), prior_end=date(2025, 5, 31),
        ),
        gross_revenue=metric(1500.0, 1300.0), net_revenue=metric(1200.0, 1000.0),
        orders=metric(60.0, 50.0), units=metric(120.0, 100.0),
        dataset_artifact_id="dataset-abc", dataset_content_hash="f" * 64,
        source_rows=1000, included_rows=996, quarantined_rows=quarantined, duplicate_rows=duplicates,
        excluded_revenue_impact=1234.5, quarantine_reasons={"malformed_row": 2, "duplicate_order_id": 1},
        materiality=MaterialityThresholds(),
        dimensions=dimensions, material_changes=material_changes or [],
        revenue_bridge=bridge(), target_comparison=targets, cpi_adjustment=cpi_adjustment,
        limitations=limitations or [],
    )
