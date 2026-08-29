"""Optional monthly CPI enrichment from the FRED (Federal Reserve Bank of St.
Louis) public API — a read-only, bounded-retry Fetcher Tool.

The CPI series used for this enrichment is a fixed, application-owned
configuration constant (``FRED_SERIES_ID``), never something an Agent
selects or invents: CPI-U, All Items, Not Seasonally Adjusted
(``CPIAUCNS``). This mirrors ``analyst_crew.tools.targets``'s optional-source
pattern exactly — the Fetcher retrieval path (``retrieve_cpi``) never raises;
it returns a plain ``{status, detail, observations}`` result so an
unavailable or invalid FRED response degrades gracefully while the required
CSV-based sales analysis continues unaffected.
"""
from __future__ import annotations

import os
import re
import time
from datetime import date
from typing import Any

# Catalogued, fixed CPI series — CPI-U, All Items, Not Seasonally Adjusted.
# This constant is the only series id the Fetcher Tool may ever request; it
# is never derived from Agent output.
FRED_SERIES_ID = "CPIAUCNS"

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

# A small bounded retry budget for transient failures only (timeouts,
# connection errors, 5xx). Authentication (401/403) and validation (400,
# malformed payload) failures never retry — retrying those would just repeat
# a failure that will never succeed.
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5


class TransientFredError(Exception):
    """Retryable FRED Fetcher Tool failure."""


def fred_cpi_data_source() -> dict[str, dict[str, Any]]:
    """Allowed, optional, read-only FRED CPI Data Source catalog entry.
    Deliberately carries only descriptive metadata — no API key — so it can
    appear safely in a Plan or Handoff. The Fetcher Tool resolves the live
    credential from ``FRED_API_KEY``, never from the catalog."""
    return {
        "cpi_fred": {
            "kind": "FRED",
            "series_id": FRED_SERIES_ID,
            "series_title": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
            "frequency": "monthly",
            "seasonally_adjusted": False,
        }
    }


_API_KEY_QUERY_PATTERN = re.compile(r"(api_key=)[^&\s]+", re.IGNORECASE)


def _redact(text: str) -> str:
    """Strip a `api_key=...` credential shape from any error text before it
    can flow into analysis parameters, a Handoff, or Findings."""
    return _API_KEY_QUERY_PATTERN.sub(r"\1***", text)


def _parse_observations(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_observations = payload["observations"]
    observations: list[dict[str, Any]] = []
    for raw in raw_observations:
        value = raw.get("value")
        if value in (None, ".", ""):
            # FRED uses "." for a missing observation value; skip rather
            # than fail the whole response over one missing month.
            continue
        observations.append({"date": str(raw["date"]), "value": float(value)})
    return observations


def retrieve_cpi(
    observation_start: date,
    observation_end: date,
    *,
    api_key: str | None = None,
    session: Any = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> dict[str, Any]:
    """Read-only FRED Fetcher Tool.

    Requests exactly one fixed, catalogued series (``FRED_SERIES_ID``) for
    the observation window needed by the Analyst's comparison. Never accepts
    or forwards an Agent-supplied series id.

    Returns a plain dict (not raising) so this optional source degrades
    gracefully: ``{"status": "not_configured" | "available" | "unavailable"
    | "invalid", "detail": str | None, "observations": [{"date": str,
    "value": float}, ...]}``.

    Transient failures (timeouts, connection errors, HTTP 5xx) receive up to
    ``max_attempts`` tries with a short exponential backoff. Authentication
    failures (HTTP 401/403) and validation failures (HTTP 400, a malformed
    JSON payload) fail immediately without retrying.
    """
    resolved_api_key = api_key if api_key is not None else os.getenv("FRED_API_KEY")
    if not resolved_api_key:
        return {"status": "not_configured", "detail": "FRED_API_KEY is not set", "observations": []}

    try:
        import requests
    except ImportError as exc:
        return {"status": "unavailable", "detail": f"requests is not installed: {exc}", "observations": []}

    http = session if session is not None else requests
    params = {
        "series_id": FRED_SERIES_ID,
        "observation_start": observation_start.isoformat(),
        "observation_end": observation_end.isoformat(),
        "api_key": resolved_api_key,
        "file_type": "json",
    }

    last_detail = "unknown error"
    for attempt in range(max_attempts):
        try:
            response = http.get(FRED_OBSERVATIONS_URL, params=params, timeout=10)
        except Exception as exc:  # noqa: BLE001 — network/timeout errors are transient and retried below
            last_detail = _redact(f"FRED request failed: {exc}")
            if attempt == max_attempts - 1:
                return {"status": "unavailable", "detail": last_detail, "observations": []}
            time.sleep(backoff_seconds * (2 ** attempt))
            continue

        status_code = response.status_code
        if status_code in (401, 403):
            return {
                "status": "invalid",
                "detail": _redact(f"FRED authentication failed: HTTP {status_code}"),
                "observations": [],
            }
        if status_code == 400:
            return {
                "status": "invalid",
                "detail": _redact(f"FRED request was invalid: HTTP 400 {response.text[:200]}"),
                "observations": [],
            }
        if 500 <= status_code < 600:
            last_detail = f"FRED server error: HTTP {status_code}"
            if attempt == max_attempts - 1:
                return {"status": "unavailable", "detail": last_detail, "observations": []}
            time.sleep(backoff_seconds * (2 ** attempt))
            continue
        if status_code != 200:
            # Any other non-2xx status is not a documented transient FRED
            # failure mode; treat it as unavailable without retrying.
            return {
                "status": "unavailable",
                "detail": _redact(f"FRED request failed: HTTP {status_code}"),
                "observations": [],
            }

        try:
            payload = response.json()
            observations = _parse_observations(payload)
        except (ValueError, KeyError, TypeError) as exc:
            return {
                "status": "invalid",
                "detail": _redact(f"FRED response is malformed: {exc}"),
                "observations": [],
            }

        if not observations:
            return {
                "status": "invalid",
                "detail": "FRED returned zero usable observations for the requested window",
                "observations": [],
            }

        return {"status": "available", "detail": None, "observations": observations}

    return {"status": "unavailable", "detail": last_detail, "observations": []}
