"""Deterministic, application-owned DuckDB Tools for baseline sales Findings."""
from __future__ import annotations

import calendar
import hashlib
import os
from datetime import date
from pathlib import Path
from typing import Any, Literal

import config
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


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


def run_sales_baseline(
    dataset_reference: dict[str, Any],
    *,
    as_of_date: date | str,
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
    finally:
        connection.close()

    profile = dataset_reference.get("profile", {})
    return BaselineFindings(
        window=window,
        gross_revenue=_comparison(current["gross_revenue"], prior["gross_revenue"]),
        net_revenue=_comparison(current["net_revenue"], prior["net_revenue"]),
        orders=_comparison(current["orders"], prior["orders"]),
        units=_comparison(current["units"], prior["units"]),
        dataset_artifact_id=str(dataset_reference["artifact_id"]),
        dataset_content_hash=actual_hash,
        source_rows=int(profile.get("source_rows", dataset_reference.get("row_count", 0))),
        included_rows=int(profile.get("included_rows", dataset_reference.get("row_count", 0))),
        quarantined_rows=int(profile.get("quarantined_rows", 0)),
        duplicate_rows=int(profile.get("duplicate_rows", 0)),
        excluded_revenue_impact=float(profile.get("excluded_revenue_impact", 0.0)),
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
                    return self.tool(selected, as_of_date=analysis_as_of)
                except (TimeoutError, ConnectionError, TransientAnalystError):
                    if tool_attempt == 1:
                        raise
        raise RuntimeError("Analyst request loop exhausted")
