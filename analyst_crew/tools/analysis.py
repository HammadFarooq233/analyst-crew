"""Deterministic, application-owned DuckDB Tools for baseline sales Findings."""
from __future__ import annotations

import calendar
import hashlib
import os
import re
from datetime import date
from pathlib import Path
from typing import Any, Literal

import config
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from analyst_crew.tools.fred import FRED_SERIES_ID
from analyst_crew.tools.retry import retry_with_backoff

# Every dimension breakdown partitions the same set of rows used for the
# overall net revenue comparison, so the sum of a dimension's per-value
# absolute_change values must equal the overall net_revenue.absolute_change
# exactly, up to floating-point summation order. This tolerance (one hundredth
# of a currency unit) is intentionally looser than the double-precision error
# actually observed, so it only trips on a genuine reconciliation bug.
RECONCILIATION_TOLERANCE = 0.01

DEFAULT_MATERIALITY_RELATIVE_PERCENT = 15.0
DEFAULT_MATERIALITY_ABSOLUTE_PERCENT_OF_PRIOR_TOTAL = 5.0

DIMENSIONS = ("product", "category", "region", "sales_channel", "customer_segment")


class MetricComparison(BaseModel):
    current: float
    prior: float
    absolute_change: float
    percent_change: float | None


class ComparisonWindow(BaseModel):
    mode: Literal["complete_month", "comparable_mtd"]
    current_start: date
    current_end: date
    prior_start: date
    prior_end: date


class MaterialityThresholds(BaseModel):
    relative_change_percent: float = DEFAULT_MATERIALITY_RELATIVE_PERCENT
    absolute_change_percent_of_prior_total: float = DEFAULT_MATERIALITY_ABSOLUTE_PERCENT_OF_PRIOR_TOTAL
    source: Literal["default", "analysis_parameters", "user_question"] = "default"


class DimensionDriver(BaseModel):
    """One dimension value's contribution to the period's net revenue change.

    ``prior_baseline_missing`` is True whenever the prior-period baseline is
    zero for this dimension value — either because the value did not appear
    in the prior window at all, or because its net revenue happened to net to
    exactly zero. In both cases a relative percentage is mathematically
    undefined, so ``percent_change`` is left as None rather than reported as
    NaN/Inf or a misleading 0%/∞%. The Material Change relative-change test is
    then treated as satisfied by convention (introducing revenue from a zero
    baseline is an unbounded relative change) and materiality falls back to
    the absolute-impact test alone.
    """

    dimension: str
    value: str
    current: float
    prior: float
    absolute_change: float
    percent_change: float | None
    prior_baseline_missing: bool
    contribution_percent: float | None
    is_material: bool
    aggregation_id: str
    dataset_artifact_id: str


BRIDGE_METHODOLOGY = (
    "For products present in both the current and prior period ('continuing' "
    "products), gross-revenue change decomposes into three exact components using "
    "each product's period unit price p and quantity q: volume effect "
    "(q_current - q_prior) * p_prior; price effect (p_current - p_prior) * "
    "q_prior; and price-volume interaction (p_current - p_prior) * (q_current - "
    "q_prior). These three always sum exactly to the continuing-product gross-"
    "revenue change. Discount effect is the negative of the change in total "
    "discount amount, and return effect is the negative of the change in total "
    "return value, so an increase in discounting or returns reduces net revenue. "
    "New-product effect is the full current-period net revenue of products with "
    "no valid prior-period baseline; discontinued-product effect is the negative "
    "of the full prior-period net revenue of products with no current-period "
    "presence. All seven effects sum to the observed net-revenue change within a "
    "documented rounding tolerance. This is a descriptive decomposition of "
    "observed revenue, not a causal explanation of why volume or price moved."
)

BRIDGE_LIMITATIONS = (
    "Unit price is a product-period weighted average (period gross revenue "
    "divided by period quantity), so it does not capture within-period price "
    "variation across individual orders of the same product.",
    "Product identity is determined at the product_id grain only; a product "
    "renamed or re-identified between periods appears as a discontinuation plus "
    "a launch rather than as a continuing product.",
    "The bridge is a descriptive decomposition, not a causal analysis of why "
    "volume, price, discounting, or returns changed.",
)


class RevenueBridge(BaseModel):
    """Auditable decomposition of the net-revenue change into distinct,
    non-causal effects that reconcile exactly (within ``RECONCILIATION_TOLERANCE``)
    to ``BaselineFindings.net_revenue.absolute_change``."""

    volume_effect: float
    price_effect: float
    interaction_effect: float
    discount_effect: float
    return_effect: float
    new_product_effect: float
    discontinued_product_effect: float
    total_effect: float
    reconciliation_gap: float
    continuing_product_count: int
    new_product_count: int
    discontinued_product_count: int
    methodology: str = BRIDGE_METHODOLOGY
    limitations: list[str] = list(BRIDGE_LIMITATIONS)
    source_references: list[str]
    dataset_artifact_id: str
    aggregation_id: str


class TargetSourceStatus(BaseModel):
    """Status of the optional Postgres sales-targets enrichment. ``available``
    means the target comparison below is trustworthy; any other status means
    the required CSV-based Findings still completed, but the optional
    target comparison did not (see ``BaselineFindings.limitations``)."""

    status: Literal["not_configured", "available", "unavailable", "invalid"]
    detail: str | None = None


class TargetVariance(BaseModel):
    """Actual vs. target net revenue and order count for one region/category
    cell of the same comparable period used by the core sales analysis."""

    region: str
    category: str
    actual_net_revenue: float
    target_net_revenue: float
    net_revenue_variance: float
    net_revenue_variance_percent: float | None
    actual_orders: float
    target_orders: float
    orders_variance: float
    orders_variance_percent: float | None


class TargetComparison(BaseModel):
    rows: list[TargetVariance] = []
    source_status: TargetSourceStatus


NON_CAUSALITY_CAVEAT = (
    "National CPI is contextual only and does not establish causation for any "
    "product, category, region, channel, or segment result."
)

CPI_METHODOLOGY = (
    "Nominal net-revenue change is the observed current-period net revenue "
    "minus the observed prior-period net revenue, with no adjustment. The "
    "approximate inflation-adjusted change deflates current-period net "
    "revenue by the ratio of the prior-period to current-period CPI-U (All "
    "Items, Not Seasonally Adjusted, FRED series CPIAUCNS) before comparing "
    "against prior-period net revenue: adjusted_current_net_revenue = "
    "current_net_revenue * (prior_cpi / current_cpi); "
    "inflation_adjusted_change = adjusted_current_net_revenue - "
    "prior_net_revenue. This is a national, all-items price-level "
    "approximation — not a category-, region-, or product-specific deflator "
    "— computed by fixed application arithmetic, never an Agent. "
    + NON_CAUSALITY_CAVEAT
)


class CpiSourceStatus(BaseModel):
    """Status of the optional FRED CPI enrichment. ``available`` means the
    inflation-adjusted change below is trustworthy; any other status means
    the required CSV-based Findings still completed, but the optional
    inflation adjustment did not (see ``BaselineFindings.limitations``)."""

    status: Literal["not_configured", "available", "unavailable", "invalid"]
    detail: str | None = None


class CpiAdjustment(BaseModel):
    """Nominal vs. approximate inflation-adjusted net-revenue change for the
    same comparable period used by the core sales analysis, sourced from a
    catalogued FRED CPI series. Always carries a non-causality caveat: this
    is national, contextual price-level information, never a causal
    explanation for a dimensional sales result."""

    series_id: str
    source: str = "FRED (Federal Reserve Bank of St. Louis)"
    nominal_change: float
    inflation_adjusted_change: float | None
    prior_cpi_value: float | None
    current_cpi_value: float | None
    prior_cpi_observation_dates: list[str] = []
    current_cpi_observation_dates: list[str] = []
    methodology: str = CPI_METHODOLOGY
    non_causality_caveat: str = NON_CAUSALITY_CAVEAT
    source_status: CpiSourceStatus


class BaselineFindings(BaseModel):
    window: ComparisonWindow
    gross_revenue: MetricComparison
    net_revenue: MetricComparison
    orders: MetricComparison
    units: MetricComparison
    dataset_artifact_id: str
    dataset_content_hash: str
    integrity_verified: bool = True
    source_rows: int = 0
    included_rows: int = 0
    quarantined_rows: int = 0
    duplicate_rows: int = 0
    excluded_revenue_impact: float = 0.0
    quarantine_reasons: dict[str, int] = {}
    materiality: MaterialityThresholds = MaterialityThresholds()
    dimensions: dict[str, list[DimensionDriver]] = {}
    material_changes: list[DimensionDriver] = []
    revenue_bridge: RevenueBridge | None = None
    target_comparison: TargetComparison | None = None
    cpi_adjustment: CpiAdjustment | None = None
    limitations: list[str] = []


class AnalystToolRequest(BaseModel):
    tool: Literal["DuckDB.sales_baseline"]
    dataset_artifact_id: str


class TransientAnalystError(Exception):
    """Retryable Analyst Tool failure."""


def production_analyst_model() -> Any:
    """Build the live Analyst Tool-calling seam.

    ``AnalystAgent.run`` invokes this with a plain context ``dict`` (step
    objective, available Dataset Artifacts, allowed Tools, validation
    feedback) — a ``ChatOpenAI`` structured-output runnable only accepts a
    PromptValue, str, or list of ``BaseMessage``, so this wraps that dict
    into a single formatted prompt before delegating to the structured call.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    model = ChatOpenAI(
        model=config.get("llm.model", "google/gemma-4-26b-a4b-it:free"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
        max_tokens=int(config.get("llm.max_tokens", 4096)),
    )
    structured = model.with_structured_output(AnalystToolRequest)

    def _invoke(context: dict[str, Any]) -> AnalystToolRequest:
        prompt = (
            "You are the Analyst Tool-calling seam for Analyst Crew. "
            "Choose exactly one allowed Tool and one available Dataset Artifact — never invent either.\n\n"
            f"Step objective: {context.get('step_objective')}\n"
            f"User question: {context.get('user_question')}\n"
            f"Available Dataset Artifacts: {context.get('available_artifacts')}\n"
            f"Allowed tools: {context.get('allowed_tools')}\n"
            f"Validation feedback from a prior attempt (empty if this is the first): "
            f"{context.get('validation_feedback') or 'none'}\n"
        )
        return structured.invoke(prompt)

    return RunnableLambda(_invoke)


_DATE_RANGE_SQL = """
SELECT min(order_date)::DATE AS min_date, max(order_date)::DATE AS max_date
FROM read_parquet(?)
"""

_BASELINE_SQL = """
SELECT
    count(DISTINCT order_id)::DOUBLE AS orders,
    coalesce(sum(quantity), 0)::DOUBLE AS units,
    coalesce(sum(quantity * unit_price), 0)::DOUBLE AS gross_revenue,
    coalesce(sum(
        quantity * unit_price
        - discount_amount
        - returned_quantity * unit_price
    ), 0)::DOUBLE AS net_revenue
FROM read_parquet(?)
WHERE order_date BETWEEN ? AND ?
"""

# Application-owned, fixed column names only — never Agent- or user-supplied —
# so building the SQL string with them carries no injection risk.
_DIMENSION_COLUMNS = {
    "product": "product_id",
    "category": "category",
    "region": "region",
    "sales_channel": "sales_channel",
    "customer_segment": "customer_segment",
}

_DIMENSION_LABEL_EXPR = {
    "product": "any_value(product_name)",
}

_DIMENSION_SQL_TEMPLATE = """
SELECT
    {column} AS dimension_key,
    {label_expr} AS dimension_label,
    coalesce(sum(
        quantity * unit_price
        - discount_amount
        - returned_quantity * unit_price
    ), 0)::DOUBLE AS net_revenue
FROM read_parquet(?)
WHERE order_date BETWEEN ? AND ?
GROUP BY {column}
"""


_REGION_CATEGORY_SQL = """
SELECT
    region AS region,
    category AS category,
    count(DISTINCT order_id)::DOUBLE AS orders,
    coalesce(sum(
        quantity * unit_price
        - discount_amount
        - returned_quantity * unit_price
    ), 0)::DOUBLE AS net_revenue
FROM read_parquet(?)
WHERE order_date BETWEEN ? AND ?
GROUP BY region, category
"""


_PRODUCT_PERIOD_SQL = """
SELECT
    product_id AS product_key,
    coalesce(sum(quantity), 0)::DOUBLE AS quantity,
    coalesce(sum(quantity * unit_price), 0)::DOUBLE AS gross_revenue,
    coalesce(sum(discount_amount), 0)::DOUBLE AS discount_amount,
    coalesce(sum(returned_quantity * unit_price), 0)::DOUBLE AS return_value,
    coalesce(sum(
        quantity * unit_price
        - discount_amount
        - returned_quantity * unit_price
    ), 0)::DOUBLE AS net_revenue
FROM read_parquet(?)
WHERE order_date BETWEEN ? AND ?
GROUP BY product_id
"""


def _materiality_from_question(question: str) -> tuple[float | None, float | None]:
    """Deterministically parse explicit override percentages from prose.

    Only simple, unambiguous "<number>% relative/change" and
    "<number>% absolute/impact" phrasings (in either number-first or
    keyword-first order) are recognised. This is a fixed regex match, not a
    model inference, so it cannot invent a threshold the question did not
    state.
    """
    if not question:
        return None, None

    def find(keyword: str) -> float | None:
        number = r"(\d+(?:\.\d+)?)\s*%"
        patterns = (
            rf"{number}[^.]{{0,40}}?{keyword}",
            rf"{keyword}[^.]{{0,40}}?{number}",
        )
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                return float(match.group(1))
        return None

    return find(r"relative"), find(r"absolute|impact")


def _resolve_materiality(
    analysis_parameters: dict[str, Any] | None,
    user_question: str,
) -> MaterialityThresholds:
    """Explicit `analysis_parameters` overrides win over question phrasing,
    which wins over the ADR-0001 defaults (15% relative, 5% absolute)."""
    parameters = analysis_parameters or {}
    relative = parameters.get("materiality_relative_threshold_percent")
    absolute = parameters.get("materiality_absolute_threshold_percent")
    source: Literal["default", "analysis_parameters", "user_question"] = "default"
    if relative is not None or absolute is not None:
        source = "analysis_parameters"
    else:
        relative, absolute = _materiality_from_question(user_question)
        if relative is not None or absolute is not None:
            source = "user_question"
    return MaterialityThresholds(
        relative_change_percent=float(relative) if relative is not None else DEFAULT_MATERIALITY_RELATIVE_PERCENT,
        absolute_change_percent_of_prior_total=(
            float(absolute) if absolute is not None else DEFAULT_MATERIALITY_ABSOLUTE_PERCENT_OF_PRIOR_TOTAL
        ),
        source=source,
    )


def _previous_month(day: date) -> tuple[int, int]:
    return (day.year - 1, 12) if day.month == 1 else (day.year, day.month - 1)


def _comparison_window(latest: date) -> ComparisonWindow:
    current_last = calendar.monthrange(latest.year, latest.month)[1]
    prior_year, prior_month = _previous_month(latest)
    prior_last = calendar.monthrange(prior_year, prior_month)[1]
    if latest.day == current_last:
        return ComparisonWindow(
            mode="complete_month",
            current_start=date(latest.year, latest.month, 1),
            current_end=latest,
            prior_start=date(prior_year, prior_month, 1),
            prior_end=date(prior_year, prior_month, prior_last),
        )
    comparable_day = min(latest.day, prior_last)
    return ComparisonWindow(
        mode="comparable_mtd",
        current_start=date(latest.year, latest.month, 1),
        current_end=latest,
        prior_start=date(prior_year, prior_month, 1),
        prior_end=date(prior_year, prior_month, comparable_day),
    )


def comparison_window(as_of_date: date | str) -> ComparisonWindow:
    """Public alias for the comparable-period window logic, so callers outside
    this module (e.g. the Analyst node retrieving Postgres targets) reuse the
    exact same current/prior period selection rather than reimplementing it."""
    latest = date.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date
    return _comparison_window(latest)


def _comparison(current: float, prior: float) -> MetricComparison:
    change = current - prior
    percent = None if prior == 0 else change / prior * 100
    return MetricComparison(
        current=current,
        prior=prior,
        absolute_change=change,
        percent_change=percent,
    )


def _dimension_aggregation_id(
    dataset_content_hash: str, window: ComparisonWindow, dimension: str, key: str
) -> str:
    payload = "|".join((
        dataset_content_hash, window.mode,
        window.current_start.isoformat(), window.current_end.isoformat(),
        window.prior_start.isoformat(), window.prior_end.isoformat(),
        dimension, key,
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _dimension_totals(connection: Any, location: Path, start: date, end: date, dimension: str) -> dict[str, tuple[str, float]]:
    column = _DIMENSION_COLUMNS[dimension]
    label_expr = _DIMENSION_LABEL_EXPR.get(dimension, column)
    sql = _DIMENSION_SQL_TEMPLATE.format(column=column, label_expr=label_expr)
    rows = connection.execute(sql, [str(location), start, end]).fetchall()
    return {str(key): (str(label), float(net_revenue)) for key, label, net_revenue in rows}


def _dimension_breakdown(
    connection: Any,
    location: Path,
    window: ComparisonWindow,
    *,
    dataset_artifact_id: str,
    dataset_content_hash: str,
    overall_net_change: float,
    thresholds: MaterialityThresholds,
    prior_total_net_revenue: float,
) -> dict[str, list[DimensionDriver]]:
    absolute_threshold_amount = abs(prior_total_net_revenue) * thresholds.absolute_change_percent_of_prior_total / 100

    breakdown: dict[str, list[DimensionDriver]] = {}
    for dimension in DIMENSIONS:
        current_totals = _dimension_totals(connection, location, window.current_start, window.current_end, dimension)
        prior_totals = _dimension_totals(connection, location, window.prior_start, window.prior_end, dimension)
        keys = set(current_totals) | set(prior_totals)

        drivers: list[DimensionDriver] = []
        for key in keys:
            current_label, current_value = current_totals.get(key, ("", 0.0))
            prior_label, prior_value = prior_totals.get(key, ("", 0.0))
            label = current_label or prior_label or key
            absolute_change = current_value - prior_value
            prior_baseline_missing = prior_value == 0
            percent_change = None if prior_baseline_missing else absolute_change / prior_value * 100
            contribution_percent = None if overall_net_change == 0 else absolute_change / overall_net_change * 100
            relative_ok = True if percent_change is None else abs(percent_change) >= thresholds.relative_change_percent
            absolute_ok = abs(absolute_change) >= absolute_threshold_amount
            drivers.append(DimensionDriver(
                dimension=dimension,
                value=label,
                current=current_value,
                prior=prior_value,
                absolute_change=absolute_change,
                percent_change=percent_change,
                prior_baseline_missing=prior_baseline_missing,
                contribution_percent=contribution_percent,
                is_material=relative_ok and absolute_ok,
                aggregation_id=_dimension_aggregation_id(dataset_content_hash, window, dimension, key),
                dataset_artifact_id=dataset_artifact_id,
            ))

        drivers.sort(key=lambda driver: (-abs(driver.absolute_change), driver.value))

        reconciliation_gap = abs(sum(driver.absolute_change for driver in drivers) - overall_net_change)
        if reconciliation_gap > RECONCILIATION_TOLERANCE:
            raise ValueError(
                f"Dimension '{dimension}' net revenue change does not reconcile with the overall "
                f"net revenue change (gap {reconciliation_gap:.4f} exceeds tolerance {RECONCILIATION_TOLERANCE})"
            )

        breakdown[dimension] = drivers
    return breakdown


def _bridge_aggregation_id(dataset_content_hash: str, window: ComparisonWindow) -> str:
    payload = "|".join((
        dataset_content_hash, window.mode,
        window.current_start.isoformat(), window.current_end.isoformat(),
        window.prior_start.isoformat(), window.prior_end.isoformat(),
        "revenue_bridge",
    ))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _product_period_totals(connection: Any, location: Path, start: date, end: date) -> dict[str, dict[str, float]]:
    rows = connection.execute(_PRODUCT_PERIOD_SQL, [str(location), start, end]).fetchall()
    columns = ("quantity", "gross_revenue", "discount_amount", "return_value", "net_revenue")
    return {str(row[0]): dict(zip(columns, row[1:], strict=True)) for row in rows}


def _revenue_bridge(
    connection: Any,
    location: Path,
    window: ComparisonWindow,
    *,
    dataset_artifact_id: str,
    dataset_content_hash: str,
    net_revenue_change: float,
) -> RevenueBridge:
    """Decompose the net-revenue change using only application-owned DuckDB
    aggregation; the effect formulas themselves are fixed Python arithmetic
    over those verified totals, never model-authored SQL or attribution."""
    current = _product_period_totals(connection, location, window.current_start, window.current_end)
    prior = _product_period_totals(connection, location, window.prior_start, window.prior_end)

    # A product missing from a period's aggregation never had a valid quantity
    # there (quarantine already rejects non-positive quantities), so presence
    # in the dict is exactly "has a valid baseline in that period" — continuing
    # products are never forced through the price/volume formulas otherwise.
    continuing_products = set(current) & set(prior)
    new_products = set(current) - set(prior)
    discontinued_products = set(prior) - set(current)

    volume_effect = 0.0
    price_effect = 0.0
    interaction_effect = 0.0
    discount_effect = 0.0
    return_effect = 0.0
    for product_id in continuing_products:
        current_totals, prior_totals = current[product_id], prior[product_id]
        quantity_current, quantity_prior = current_totals["quantity"], prior_totals["quantity"]
        price_current = current_totals["gross_revenue"] / quantity_current if quantity_current else 0.0
        price_prior = prior_totals["gross_revenue"] / quantity_prior if quantity_prior else 0.0

        volume_effect += (quantity_current - quantity_prior) * price_prior
        price_effect += (price_current - price_prior) * quantity_prior
        interaction_effect += (price_current - price_prior) * (quantity_current - quantity_prior)
        discount_effect += -(current_totals["discount_amount"] - prior_totals["discount_amount"])
        return_effect += -(current_totals["return_value"] - prior_totals["return_value"])

    new_product_effect = sum(current[product_id]["net_revenue"] for product_id in new_products)
    discontinued_product_effect = -sum(prior[product_id]["net_revenue"] for product_id in discontinued_products)

    total_effect = (
        volume_effect + price_effect + interaction_effect
        + discount_effect + return_effect
        + new_product_effect + discontinued_product_effect
    )
    reconciliation_gap = abs(total_effect - net_revenue_change)
    if reconciliation_gap > RECONCILIATION_TOLERANCE:
        raise ValueError(
            "Revenue bridge effects do not reconcile with the overall net revenue "
            f"change (gap {reconciliation_gap:.4f} exceeds tolerance {RECONCILIATION_TOLERANCE})"
        )

    return RevenueBridge(
        volume_effect=volume_effect,
        price_effect=price_effect,
        interaction_effect=interaction_effect,
        discount_effect=discount_effect,
        return_effect=return_effect,
        new_product_effect=new_product_effect,
        discontinued_product_effect=discontinued_product_effect,
        total_effect=total_effect,
        reconciliation_gap=reconciliation_gap,
        continuing_product_count=len(continuing_products),
        new_product_count=len(new_products),
        discontinued_product_count=len(discontinued_products),
        source_references=[
            f"Dataset Artifact {dataset_artifact_id} (content hash {dataset_content_hash[:12]})",
            "DuckDB.sales_baseline product-period aggregation (application-owned SQL)",
        ],
        dataset_artifact_id=dataset_artifact_id,
        aggregation_id=_bridge_aggregation_id(dataset_content_hash, window),
    )


def _region_category_actuals(
    connection: Any, location: Path, start: date, end: date
) -> dict[tuple[str, str], tuple[float, float]]:
    """(region, category) -> (orders, net_revenue) for one period, from the
    same application-owned DuckDB aggregation used for the dimension
    breakdown — never a model-authored query."""
    rows = connection.execute(_REGION_CATEGORY_SQL, [str(location), start, end]).fetchall()
    return {
        (str(region).casefold(), str(category).casefold()): (float(orders), float(net_revenue))
        for region, category, orders, net_revenue in rows
    }


def _percent_variance(delta: float, base: float) -> float | None:
    """None when the target baseline is zero — mirrors how a zero prior-period
    baseline is handled for dimension drivers (issue #6): a relative percent
    against a zero target is mathematically undefined, not 0% or infinite."""
    return None if base == 0 else delta / base * 100


def _build_target_comparison(
    connection: Any,
    location: Path,
    window: ComparisonWindow,
    target_data: dict[str, Any] | None,
) -> tuple[TargetComparison | None, list[str]]:
    """Compare actual net revenue and order count against Postgres sales
    targets for the same comparable current period as the core CSV analysis,
    by region and category. ``target_data`` is the already-retrieved result
    of the read-only Fetcher Tool (``analyst_crew.tools.targets.retrieve_targets``);
    this function performs no Postgres I/O itself and never raises for an
    unavailable/invalid optional source — it records status and a limitation
    instead, so the required CSV Findings are unaffected."""
    if target_data is None:
        return None, []

    status = target_data.get("status", "not_configured")
    detail = target_data.get("detail")
    if status != "available":
        limitation = f"Postgres sales-target comparison {status}: {detail or 'no further detail available'}"
        return TargetComparison(rows=[], source_status=TargetSourceStatus(status=status, detail=detail)), [limitation]

    actuals = _region_category_actuals(connection, location, window.current_start, window.current_end)
    targets_by_key: dict[tuple[str, str], dict[str, Any]] = {
        (str(row["region"]).casefold(), str(row["category"]).casefold()): row
        for row in target_data.get("rows", [])
    }

    if not targets_by_key:
        limitation = "Postgres returned zero target rows for the comparable period; no target comparison could be computed."
        return TargetComparison(rows=[], source_status=TargetSourceStatus(status="invalid", detail=limitation)), [limitation]

    rows: list[TargetVariance] = []
    for region, category in sorted(set(actuals) | set(targets_by_key)):
        actual_orders, actual_net_revenue = actuals.get((region, category), (0.0, 0.0))
        target_row = targets_by_key.get((region, category))
        target_net_revenue = float(target_row["target_net_revenue"]) if target_row else 0.0
        target_orders = float(target_row["target_orders"]) if target_row else 0.0
        net_revenue_variance = actual_net_revenue - target_net_revenue
        orders_variance = actual_orders - target_orders
        rows.append(TargetVariance(
            region=region,
            category=category,
            actual_net_revenue=actual_net_revenue,
            target_net_revenue=target_net_revenue,
            net_revenue_variance=net_revenue_variance,
            net_revenue_variance_percent=_percent_variance(net_revenue_variance, target_net_revenue),
            actual_orders=actual_orders,
            target_orders=target_orders,
            orders_variance=orders_variance,
            orders_variance_percent=_percent_variance(orders_variance, target_orders),
        ))

    return TargetComparison(rows=rows, source_status=TargetSourceStatus(status="available")), []


def _select_period_cpi(
    observations: list[dict[str, Any]], start: date, end: date
) -> tuple[float, list[str]] | None:
    """Average the CPI observation(s) whose date falls within one comparison
    period. FRED's monthly CPI observations are dated to the first of the
    month, so a period covering that date yields exactly one observation in
    the ordinary case; averaging only guards the rare case of more than one
    observation falling in the window. Returns None when no observation
    covers the period at all."""
    candidates = [obs for obs in observations if start <= date.fromisoformat(obs["date"]) <= end]
    if not candidates:
        return None
    value = sum(obs["value"] for obs in candidates) / len(candidates)
    return value, sorted(obs["date"] for obs in candidates)


def _build_cpi_adjustment(
    window: ComparisonWindow,
    net_revenue: MetricComparison,
    cpi_data: dict[str, Any] | None,
) -> tuple[CpiAdjustment | None, list[str]]:
    """Approximate inflation-adjusted net-revenue change for the same
    comparable period as the core sales analysis, from an already-retrieved
    FRED CPI Fetcher Tool result (``analyst_crew.tools.fred.retrieve_cpi``).
    This function performs no HTTP I/O itself, does pure deterministic
    application arithmetic (never an Agent computation), and never raises
    for an unavailable/invalid optional source — it records status and a
    limitation instead, so the required CSV Findings are unaffected."""
    if cpi_data is None:
        return None, []

    status = cpi_data.get("status", "not_configured")
    detail = cpi_data.get("detail")
    if status != "available":
        limitation = f"FRED CPI inflation adjustment {status}: {detail or 'no further detail available'}"
        return CpiAdjustment(
            series_id=FRED_SERIES_ID,
            nominal_change=net_revenue.absolute_change,
            inflation_adjusted_change=None,
            prior_cpi_value=None,
            current_cpi_value=None,
            source_status=CpiSourceStatus(status=status, detail=detail),
        ), [limitation]

    observations = cpi_data.get("observations", [])
    prior_selection = _select_period_cpi(observations, window.prior_start, window.prior_end)
    current_selection = _select_period_cpi(observations, window.current_start, window.current_end)
    if prior_selection is None or current_selection is None or current_selection[0] == 0:
        limitation = (
            "FRED CPI observations do not cover the comparison window; "
            "inflation-adjusted change unavailable."
        )
        return CpiAdjustment(
            series_id=FRED_SERIES_ID,
            nominal_change=net_revenue.absolute_change,
            inflation_adjusted_change=None,
            prior_cpi_value=None if prior_selection is None else prior_selection[0],
            current_cpi_value=None if current_selection is None else current_selection[0],
            prior_cpi_observation_dates=[] if prior_selection is None else prior_selection[1],
            current_cpi_observation_dates=[] if current_selection is None else current_selection[1],
            source_status=CpiSourceStatus(status="invalid", detail=limitation),
        ), [limitation]

    prior_cpi, prior_dates = prior_selection
    current_cpi, current_dates = current_selection
    adjusted_current_net_revenue = net_revenue.current * (prior_cpi / current_cpi)
    inflation_adjusted_change = adjusted_current_net_revenue - net_revenue.prior

    return CpiAdjustment(
        series_id=FRED_SERIES_ID,
        nominal_change=net_revenue.absolute_change,
        inflation_adjusted_change=inflation_adjusted_change,
        prior_cpi_value=prior_cpi,
        current_cpi_value=current_cpi,
        prior_cpi_observation_dates=prior_dates,
        current_cpi_observation_dates=current_dates,
        source_status=CpiSourceStatus(status="available"),
    ), []


def run_sales_baseline(
    dataset_reference: dict[str, Any],
    *,
    as_of_date: date | str,
    analysis_parameters: dict[str, Any] | None = None,
    user_question: str = "",
    target_data: dict[str, Any] | None = None,
    cpi_data: dict[str, Any] | None = None,
) -> BaselineFindings:
    """Execute only application-owned SQL templates against a verified artifact."""
    if dataset_reference.get("kind") != "Dataset Artifact":
        raise ValueError("Analyst requires a Dataset Artifact reference")
    location = Path(str(dataset_reference.get("location", "")))
    expected_hash = str(dataset_reference.get("content_hash", ""))
    if not location.is_file():
        raise ValueError("Dataset Artifact is unreadable")
    actual_hash = hashlib.sha256(location.read_bytes()).hexdigest()
    if not expected_hash or actual_hash != expected_hash:
        raise ValueError("Dataset Artifact integrity verification failed")

    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError("Analyst requires DuckDB") from exc

    connection = duckdb.connect(database=":memory:")
    try:
        _, latest_transaction = connection.execute(_DATE_RANGE_SQL, [str(location)]).fetchone()
        latest = date.fromisoformat(as_of_date) if isinstance(as_of_date, str) else as_of_date
        if latest is None:
            raise ValueError("Dataset Artifact contains no dated sales rows")
        if latest_transaction is not None and latest < latest_transaction:
            raise ValueError("Analysis as-of date precedes Dataset Artifact rows")
        window = _comparison_window(latest)

        def metrics(start: date, end: date) -> dict[str, float]:
            row = connection.execute(
                _BASELINE_SQL,
                [str(location), start, end],
            ).fetchone()
            return dict(zip(("orders", "units", "gross_revenue", "net_revenue"), row, strict=True))

        current = metrics(window.current_start, window.current_end)
        prior = metrics(window.prior_start, window.prior_end)

        net_revenue = _comparison(current["net_revenue"], prior["net_revenue"])
        thresholds = _resolve_materiality(analysis_parameters, user_question)
        artifact_id = str(dataset_reference["artifact_id"])
        dimensions = _dimension_breakdown(
            connection,
            location,
            window,
            dataset_artifact_id=artifact_id,
            dataset_content_hash=actual_hash,
            overall_net_change=net_revenue.absolute_change,
            thresholds=thresholds,
            prior_total_net_revenue=prior["net_revenue"],
        )
        revenue_bridge = _revenue_bridge(
            connection,
            location,
            window,
            dataset_artifact_id=artifact_id,
            dataset_content_hash=actual_hash,
            net_revenue_change=net_revenue.absolute_change,
        )
        target_comparison, target_limitations = _build_target_comparison(connection, location, window, target_data)
        cpi_adjustment, cpi_limitations = _build_cpi_adjustment(window, net_revenue, cpi_data)
    finally:
        connection.close()

    material_changes = sorted(
        (driver for values in dimensions.values() for driver in values if driver.is_material),
        key=lambda driver: (-abs(driver.absolute_change), driver.dimension, driver.value),
    )

    profile = dataset_reference.get("profile", {})
    return BaselineFindings(
        window=window,
        gross_revenue=_comparison(current["gross_revenue"], prior["gross_revenue"]),
        net_revenue=net_revenue,
        orders=_comparison(current["orders"], prior["orders"]),
        units=_comparison(current["units"], prior["units"]),
        dataset_artifact_id=artifact_id,
        dataset_content_hash=actual_hash,
        source_rows=int(profile.get("source_rows", dataset_reference.get("row_count", 0))),
        included_rows=int(profile.get("included_rows", dataset_reference.get("row_count", 0))),
        quarantined_rows=int(profile.get("quarantined_rows", 0)),
        duplicate_rows=int(profile.get("duplicate_rows", 0)),
        excluded_revenue_impact=float(profile.get("excluded_revenue_impact", 0.0)),
        quarantine_reasons=dict(profile.get("quarantine_reasons", {})),
        materiality=thresholds,
        dimensions=dimensions,
        material_changes=material_changes,
        revenue_bridge=revenue_bridge,
        target_comparison=target_comparison,
        cpi_adjustment=cpi_adjustment,
        limitations=target_limitations + cpi_limitations,
    )


class AnalystAgent:
    """Validated Agent loop selecting narrow deterministic DuckDB Tools.

    Contract-invalid Agent output (a disallowed Tool or an undeclared
    Dataset Artifact) receives at most one repair retry, with the validation
    error fed back into the next model call as feedback — never a second
    call to the Planner. Transient Tool failures (network timeouts,
    connection errors) receive a bounded retry via
    ``analyst_crew.tools.retry.retry_with_backoff``; any other Tool failure
    is never retried. ``validation_retries``/``tool_retries`` record how
    many retries were actually used, for CrewState/tracing visibility.
    """

    allowed_tools = frozenset({"DuckDB.sales_baseline"})
    max_tool_attempts = 2

    def __init__(self, model: Any, tool: Any = run_sales_baseline):
        self.model = model
        self.tool = tool
        self.validation_retries = 0
        self.tool_retries = 0
        self.last_validation_error: str | None = None

    def run(self, handoff: Any) -> BaselineFindings:
        dataset_references = {
            key: value
            for key, value in handoff.input_artifacts.items()
            if isinstance(value, dict) and value.get("kind") == "Dataset Artifact"
        }
        if not dataset_references:
            raise ValueError("Analyst Handoff has no Dataset Artifact")
        feedback = ""
        for attempt in range(2):
            try:
                request = self.model.invoke({
                    "step_objective": handoff.step.objective,
                    "user_question": handoff.user_question,
                    "available_artifacts": {
                        key: value["artifact_id"] for key, value in dataset_references.items()
                    },
                    "allowed_tools": sorted(self.allowed_tools),
                    "validation_feedback": feedback,
                })
                if isinstance(request, AnalystToolRequest):
                    request = request.model_dump()
                if not isinstance(request, dict) or request.get("tool") not in self.allowed_tools:
                    raise ValueError("disallowed Tool")
                selected = next(
                    (value for value in dataset_references.values()
                     if value["artifact_id"] == request.get("dataset_artifact_id")),
                    None,
                )
                if selected is None:
                    raise ValueError("undeclared Dataset Artifact")
            except ValueError as exc:
                self.last_validation_error = str(exc)
                if attempt:
                    raise
                self.validation_retries += 1
                feedback = f"Invalid Analyst request: {exc}. Select only an allowed Tool and artifact."
                continue

            def _call_tool(selected=selected) -> BaselineFindings:
                analysis_as_of = handoff.analysis_parameters.get("as_of_date")
                if not analysis_as_of:
                    raise ValueError("Analyst requires an explicit as_of_date coverage boundary")
                return self.tool(
                    selected,
                    as_of_date=analysis_as_of,
                    analysis_parameters=handoff.analysis_parameters,
                    user_question=handoff.user_question,
                    target_data=handoff.analysis_parameters.get("target_data"),
                    cpi_data=handoff.analysis_parameters.get("cpi_data"),
                )

            result, outcome = retry_with_backoff(
                _call_tool,
                is_transient=lambda exc: isinstance(exc, (TimeoutError, ConnectionError, TransientAnalystError)),
                max_attempts=self.max_tool_attempts,
                backoff_seconds=0.0,
            )
            self.tool_retries = outcome.retries
            return result
        raise RuntimeError("Analyst request loop exhausted")
