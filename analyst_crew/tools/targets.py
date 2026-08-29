"""Deterministic Postgres sales-targets seed and read-only Fetcher Tool.

The targets table holds six months of internal sales targets at month,
region, and category grain, scaled to plausibly match the synthetic
retailer's approximate revenue mix (see ``analyst_crew.tools.sales``). The
Fetcher retrieval path (``retrieve_targets``) executes exactly one fixed,
parameterized, read-only ``SELECT`` — it can never run arbitrary or
mutating SQL, by construction rather than by convention.
"""
from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

DEFAULT_TARGET_SEED = 20260301  # same deterministic-seed family as the synthetic sales generator

# Six months aligned with the synthetic retailer's Jan-Jun 2025 order date range.
TARGET_MONTHS = tuple((2025, month) for month in range(1, 7))
TARGET_REGIONS = ("north", "south", "east", "west")
TARGET_CATEGORIES = ("office", "furniture", "technology")

# Approximate region and category mix derived from the synthetic generator's
# product catalog (analyst_crew.tools.sales.generate_sales_csv): Office
# (Desk Lamp/Notebook, low average order value), Furniture (Ergonomic
# Chair/Standing Desk, high average order value), Technology (Monitor/
# Keyboard, mid average order value). These constants exist only to give the
# seeded targets a plausible order of magnitude; they are not derived from
# any live CSV so the seed stays fully deterministic and offline.
_REGION_WEIGHTS = {"north": 1.15, "south": 1.05, "east": 0.95, "west": 0.85}
_CATEGORY_MONTHLY_BASE_NET_REVENUE = {
    "office": 115_000.0,
    "furniture": 2_800_000.0,
    "technology": 1_000_000.0,
}
_CATEGORY_AVG_ORDER_VALUE = {"office": 68.4, "furniture": 1_710.0, "technology": 587.8}

TARGETS_TABLE = "sales_targets"

# Fixed, application-owned DDL/DML used only by the offline seed script — the
# runtime Fetcher retrieval path below never executes any of this.
TARGETS_SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TARGETS_TABLE} (
    target_month DATE NOT NULL,
    region TEXT NOT NULL,
    category TEXT NOT NULL,
    target_net_revenue DOUBLE PRECISION NOT NULL,
    target_orders INTEGER NOT NULL,
    PRIMARY KEY (target_month, region, category)
)
"""
_DELETE_TARGETS_SQL = f"DELETE FROM {TARGETS_TABLE}"
_INSERT_TARGET_SQL = (
    f"INSERT INTO {TARGETS_TABLE} "
    "(target_month, region, category, target_net_revenue, target_orders) "
    "VALUES (%s, %s, %s, %s, %s)"
)

# The one and only query the runtime Fetcher Tool may execute: a fixed,
# read-only, parameterized SELECT. It is asserted read-only in
# retrieve_targets() rather than trusted by convention.
_SELECT_TARGETS_SQL = (
    "SELECT target_month, region, category, target_net_revenue, target_orders "
    f"FROM {TARGETS_TABLE} WHERE target_month = %s"
)


class TransientTargetsError(Exception):
    """Retryable Postgres target Fetcher Tool failure."""


@dataclass(frozen=True)
class TargetRow:
    target_month: date
    region: str
    category: str
    target_net_revenue: float
    target_orders: int


def generate_target_rows(*, seed: int = DEFAULT_TARGET_SEED) -> list[TargetRow]:
    """Deterministically derive six months of month x region x category sales
    targets. A small seeded jitter (+/-5%) avoids perfectly uniform,
    unrealistic targets while remaining fully reproducible for a fixed seed."""
    rng = random.Random(seed)
    total_region_weight = sum(_REGION_WEIGHTS.values())
    rows: list[TargetRow] = []
    for month_index, (year, month) in enumerate(TARGET_MONTHS):
        seasonal = 1.0 + 0.03 * month_index
        for category in TARGET_CATEGORIES:
            month_category_net_revenue = _CATEGORY_MONTHLY_BASE_NET_REVENUE[category] * seasonal
            for region in TARGET_REGIONS:
                jitter = 1.0 + rng.uniform(-0.05, 0.05)
                region_share = _REGION_WEIGHTS[region] / total_region_weight
                target_net_revenue = round(month_category_net_revenue * region_share * jitter, 2)
                target_orders = max(1, round(target_net_revenue / _CATEGORY_AVG_ORDER_VALUE[category]))
                rows.append(TargetRow(
                    target_month=date(year, month, 1),
                    region=region,
                    category=category,
                    target_net_revenue=target_net_revenue,
                    target_orders=target_orders,
                ))
    return rows


def sales_targets_data_source() -> dict[str, dict[str, Any]]:
    """Allowed, optional, read-only Postgres target Data Source catalog
    entry. Deliberately carries only descriptive metadata — no host, port,
    user, or password — so it can appear safely in a Plan or Handoff. The
    Fetcher Tool resolves the live connection from ``POSTGRES_CONNECTION_STRING``
    (the same credential source already used for the LangGraph checkpointer),
    never from the catalog."""
    return {
        "sales_targets_postgres": {
            "kind": "Postgres",
            "table": TARGETS_TABLE,
            "grain": ["month", "region", "category"],
        }
    }


def seed_targets(conn_string: str, *, seed: int = DEFAULT_TARGET_SEED) -> int:
    """Create the targets table if needed and idempotently (re)seed it with
    the deterministic six-month target set. Returns the row count written.

    This is an offline administrative seed utility, not the runtime Fetcher
    Tool — it may create/replace rows. The read-only restriction below
    applies only to ``retrieve_targets``, the path the Analyst investigation
    actually exercises at runtime.
    """
    import psycopg

    rows = generate_target_rows(seed=seed)
    with psycopg.connect(conn_string) as conn:
        with conn.cursor() as cur:
            cur.execute(TARGETS_SCHEMA_SQL)
            cur.execute(_DELETE_TARGETS_SQL)
            cur.executemany(
                _INSERT_TARGET_SQL,
                [
                    (row.target_month, row.region, row.category, row.target_net_revenue, row.target_orders)
                    for row in rows
                ],
            )
        conn.commit()
    return len(rows)


_CREDENTIAL_URI_PATTERN = re.compile(r"//[^\s:@/]+:[^\s@/]+@")


def _redact(text: str) -> str:
    """Strip a `user:password@` credential shape from any error text before
    it can flow into analysis parameters, a Handoff, or Findings."""
    return _CREDENTIAL_URI_PATTERN.sub("//***:***@", text)


def retrieve_targets(
    window_current_start: date,
    window_current_end: date,
    *,
    conn_string: str | None = None,
) -> dict[str, Any]:
    """Read-only Postgres Fetcher Tool.

    Executes exactly one fixed, parameterized ``SELECT`` against the targets
    table for the calendar month covering the comparable current period —
    the same window the core CSV sales analysis already computed. Never
    accepts, builds, or executes arbitrary or mutating SQL.

    Returns a plain dict (not raising) so optional-source failures degrade
    gracefully: ``{"status": "available" | "unavailable" | "invalid",
    "detail": str | None, "rows": [...]}``.
    """
    if _SELECT_TARGETS_SQL.strip().upper().split(None, 1)[0] != "SELECT":
        # Defense in depth: the query text above is a fixed module constant,
        # so this can never actually trip, but the Tool boundary enforces
        # read-only in code rather than only by convention.
        raise RuntimeError("Postgres target Tool must execute a read-only SELECT")

    resolved_conn_string = conn_string if conn_string is not None else os.getenv("POSTGRES_CONNECTION_STRING")
    if not resolved_conn_string:
        return {"status": "unavailable", "detail": "POSTGRES_CONNECTION_STRING is not set", "rows": []}

    try:
        import psycopg
    except ImportError as exc:
        return {"status": "unavailable", "detail": _redact(f"psycopg is not installed: {exc}"), "rows": []}

    target_month = date(window_current_start.year, window_current_start.month, 1)
    try:
        with psycopg.connect(resolved_conn_string, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute(_SELECT_TARGETS_SQL, (target_month,))
                fetched = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 — any connection/query failure degrades this optional source
        return {"status": "unavailable", "detail": _redact(f"Postgres target query failed: {exc}"), "rows": []}

    try:
        rows = [
            {
                "target_month": row[0].isoformat() if hasattr(row[0], "isoformat") else str(row[0]),
                "region": str(row[1]).strip().casefold(),
                "category": str(row[2]).strip().casefold(),
                "target_net_revenue": float(row[3]),
                "target_orders": int(row[4]),
            }
            for row in fetched
        ]
    except (TypeError, ValueError, IndexError) as exc:
        return {"status": "invalid", "detail": _redact(f"Postgres target rows are malformed: {exc}"), "rows": []}

    return {"status": "available", "detail": None, "rows": rows}
