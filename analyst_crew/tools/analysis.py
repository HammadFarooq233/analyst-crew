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
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

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


class AnalystToolRequest(BaseModel):
    tool: Literal["DuckDB.sales_baseline"]
    dataset_artifact_id: str


class TransientAnalystError(Exception):
    """Retryable Analyst Tool failure."""


def production_analyst_model() -> Any:
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
    return model.with_structured_output(AnalystToolRequest)


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


def run_sales_baseline(
    dataset_reference: dict[str, Any],
    *,
    as_of_date: date | str,
    analysis_parameters: dict[str, Any] | None = None,
    user_question: str = "",
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
    )


class AnalystAgent:
    """Validated Agent loop selecting narrow deterministic DuckDB Tools."""

    allowed_tools = frozenset({"DuckDB.sales_baseline"})

    def __init__(self, model: Any, tool: Any = run_sales_baseline):
        self.model = model
        self.tool = tool

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
                if attempt:
                    raise
                feedback = f"Invalid Analyst request: {exc}. Select only an allowed Tool and artifact."
                continue
            for tool_attempt in range(2):
                try:
                    analysis_as_of = handoff.analysis_parameters.get("as_of_date")
                    if not analysis_as_of:
                        raise ValueError("Analyst requires an explicit as_of_date coverage boundary")
                    return self.tool(
                        selected,
                        as_of_date=analysis_as_of,
                        analysis_parameters=handoff.analysis_parameters,
                        user_question=handoff.user_question,
                    )
                except (TimeoutError, ConnectionError, TransientAnalystError):
                    if tool_attempt == 1:
                        raise
        raise RuntimeError("Analyst request loop exhausted")
