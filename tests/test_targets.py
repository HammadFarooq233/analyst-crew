import os
from datetime import date

import pytest

from analyst_crew.tools.targets import (
    DEFAULT_TARGET_SEED,
    TARGET_CATEGORIES,
    TARGET_MONTHS,
    TARGET_REGIONS,
    TARGETS_TABLE,
    _SELECT_TARGETS_SQL,
    _redact,
    generate_target_rows,
    retrieve_targets,
    sales_targets_data_source,
    seed_targets,
)


def _postgres_available() -> bool:
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn_string:
        return False
    try:
        import psycopg

        conn = psycopg.connect(conn_string, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


def test_generate_target_rows_is_deterministic_and_covers_month_region_category_grain():
    first = generate_target_rows()
    second = generate_target_rows()
    assert first == second

    assert len(first) == len(TARGET_MONTHS) * len(TARGET_REGIONS) * len(TARGET_CATEGORIES)
    months = {row.target_month for row in first}
    assert months == {date(year, month, 1) for year, month in TARGET_MONTHS}
    assert {row.region for row in first} == set(TARGET_REGIONS)
    assert {row.category for row in first} == set(TARGET_CATEGORIES)

    # A different seed produces a different (still deterministic) target set.
    other = generate_target_rows(seed=DEFAULT_TARGET_SEED + 1)
    assert other != first
    assert other == generate_target_rows(seed=DEFAULT_TARGET_SEED + 1)


def test_generate_target_rows_values_are_positive_and_plausible_scale():
    rows = generate_target_rows()
    for row in rows:
        assert row.target_net_revenue > 0
        assert row.target_orders > 0
        # Aligned with the synthetic retailer's per-category order-of-magnitude:
        # furniture is the highest-ticket category, office the lowest.
        assert row.target_net_revenue < 5_000_000


def test_sales_targets_data_source_exposes_no_credentials():
    catalog = sales_targets_data_source()
    entry = catalog["sales_targets_postgres"]
    assert entry["kind"] == "Postgres"
    assert entry["table"] == TARGETS_TABLE
    serialized = str(entry).lower()
    for secret_token in ("password", "host", "port", "user=", "@localhost", "analyst:analyst"):
        assert secret_token not in serialized


def test_retrieve_targets_is_unavailable_when_connection_string_missing():
    result = retrieve_targets(date(2025, 6, 1), date(2025, 6, 30), conn_string="")
    assert result["status"] == "unavailable"
    assert result["rows"] == []
    assert "POSTGRES_CONNECTION_STRING" in result["detail"]


def test_retrieve_targets_falls_back_to_env_var(monkeypatch):
    monkeypatch.delenv("POSTGRES_CONNECTION_STRING", raising=False)
    result = retrieve_targets(date(2025, 6, 1), date(2025, 6, 30))
    assert result["status"] == "unavailable"


def test_retrieve_targets_query_is_read_only_select_and_parameterized():
    normalized = _SELECT_TARGETS_SQL.strip().upper()
    assert normalized.startswith("SELECT")
    for forbidden in ("INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", ";"):
        assert forbidden not in normalized
    assert _SELECT_TARGETS_SQL.count("%s") == 1


def test_retrieve_targets_unavailable_on_connection_failure(monkeypatch):
    class BoomConnect:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("connection to server at localhost failed")

    import analyst_crew.tools.targets as targets_module
    import types

    fake_psycopg = types.SimpleNamespace(connect=BoomConnect())
    monkeypatch.setitem(__import__("sys").modules, "psycopg", fake_psycopg)

    result = retrieve_targets(date(2025, 6, 1), date(2025, 6, 30), conn_string="postgresql://analyst:analyst@localhost:5434/analyst_crew")
    assert result["status"] == "unavailable"
    assert result["rows"] == []
    assert "analyst:analyst" not in result["detail"]


def test_retrieve_targets_returns_available_rows_from_a_fake_connection(monkeypatch):
    import types

    captured_sql = {}

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def execute(self, sql, params):
            captured_sql["sql"] = sql
            captured_sql["params"] = params

        def fetchall(self):
            return [
                (date(2025, 6, 1), "North", "Office", 10000.0, 150),
                (date(2025, 6, 1), "south", "furniture", "250000", "80"),
            ]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def cursor(self):
            return FakeCursor()

    fake_psycopg = types.SimpleNamespace(connect=lambda *args, **kwargs: FakeConnection())
    monkeypatch.setitem(__import__("sys").modules, "psycopg", fake_psycopg)

    result = retrieve_targets(date(2025, 6, 15), date(2025, 6, 30), conn_string="postgresql://analyst:analyst@localhost:5434/analyst_crew")
    assert result["status"] == "available"
    assert captured_sql["params"] == (date(2025, 6, 1),)
    assert result["rows"] == [
        {"target_month": "2025-06-01", "region": "north", "category": "office", "target_net_revenue": 10000.0, "target_orders": 150},
        {"target_month": "2025-06-01", "region": "south", "category": "furniture", "target_net_revenue": 250000.0, "target_orders": 80},
    ]


def test_retrieve_targets_marks_invalid_on_malformed_rows(monkeypatch):
    import types

    class FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def execute(self, sql, params):
            pass

        def fetchall(self):
            return [(date(2025, 6, 1), "North", "Office", "not-a-number", 150)]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def cursor(self):
            return FakeCursor()

    fake_psycopg = types.SimpleNamespace(connect=lambda *args, **kwargs: FakeConnection())
    monkeypatch.setitem(__import__("sys").modules, "psycopg", fake_psycopg)

    result = retrieve_targets(date(2025, 6, 1), date(2025, 6, 30), conn_string="postgresql://analyst:analyst@localhost:5434/analyst_crew")
    assert result["status"] == "invalid"
    assert result["rows"] == []


def test_redact_strips_credentials_from_error_text():
    text = "connection to postgresql://analyst:analyst@localhost:5434/analyst_crew failed"
    redacted = _redact(text)
    assert "analyst:analyst" not in redacted
    assert "***:***@localhost" in redacted


@pytest.mark.integration
def test_seed_and_retrieve_targets_against_live_postgres():
    if not _postgres_available():
        pytest.skip("Postgres not available")
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")

    row_count = seed_targets(conn_string)
    assert row_count == len(TARGET_MONTHS) * len(TARGET_REGIONS) * len(TARGET_CATEGORIES)

    # Re-seeding is idempotent: same deterministic row count, no duplicate-key errors.
    assert seed_targets(conn_string) == row_count

    result = retrieve_targets(date(2025, 6, 1), date(2025, 6, 30), conn_string=conn_string)
    assert result["status"] == "available"
    assert len(result["rows"]) == len(TARGET_REGIONS) * len(TARGET_CATEGORIES)
    for row in result["rows"]:
        assert row["region"] in TARGET_REGIONS
        assert row["category"] in TARGET_CATEGORIES
        assert row["target_net_revenue"] > 0
        assert row["target_orders"] > 0
