from datetime import date

import pytest

from analyst_crew.tools.fred import (
    FRED_OBSERVATIONS_URL,
    FRED_SERIES_ID,
    _redact,
    fred_cpi_data_source,
    retrieve_cpi,
)


class FakeResponse:
    def __init__(self, status_code=200, json_payload=None, text=""):
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text

    def json(self):
        if self._json_payload is None:
            raise ValueError("no json payload")
        return self._json_payload


class FakeSession:
    """Deterministic stand-in for `requests` — returns a scripted sequence of
    responses/exceptions and records every call for assertion."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _valid_payload(observations):
    return {"observations": observations}


def test_fred_cpi_data_source_exposes_no_credentials():
    catalog = fred_cpi_data_source()
    entry = catalog["cpi_fred"]
    assert entry["kind"] == "FRED"
    assert entry["series_id"] == FRED_SERIES_ID == "CPIAUCNS"
    serialized = str(entry).lower()
    for secret_token in ("api_key", "password", "token"):
        assert secret_token not in serialized


def test_retrieve_cpi_is_not_configured_when_api_key_missing():
    result = retrieve_cpi(date(2025, 5, 1), date(2025, 6, 30), api_key="")
    assert result["status"] == "not_configured"
    assert result["observations"] == []
    assert "FRED_API_KEY" in result["detail"]


def test_retrieve_cpi_falls_back_to_env_var(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    result = retrieve_cpi(date(2025, 5, 1), date(2025, 6, 30))
    assert result["status"] == "not_configured"


def test_retrieve_cpi_success_parses_observations_and_uses_fixed_series():
    session = FakeSession([
        FakeResponse(200, _valid_payload([
            {"date": "2025-05-01", "value": "320.5"},
            {"date": "2025-06-01", "value": "321.0"},
        ])),
    ])
    result = retrieve_cpi(date(2025, 5, 1), date(2025, 6, 30), api_key="secret-key", session=session)
    assert result["status"] == "available"
    assert result["detail"] is None
    assert result["observations"] == [
        {"date": "2025-05-01", "value": 320.5},
        {"date": "2025-06-01", "value": 321.0},
    ]
    assert len(session.calls) == 1
    assert session.calls[0]["url"] == FRED_OBSERVATIONS_URL
    assert session.calls[0]["params"]["series_id"] == FRED_SERIES_ID
    assert session.calls[0]["params"]["api_key"] == "secret-key"
    assert session.calls[0]["params"]["file_type"] == "json"


def test_retrieve_cpi_skips_missing_dot_values():
    session = FakeSession([
        FakeResponse(200, _valid_payload([
            {"date": "2025-05-01", "value": "."},
            {"date": "2025-06-01", "value": "321.0"},
        ])),
    ])
    result = retrieve_cpi(date(2025, 5, 1), date(2025, 6, 30), api_key="secret-key", session=session)
    assert result["status"] == "available"
    assert result["observations"] == [{"date": "2025-06-01", "value": 321.0}]


def test_retrieve_cpi_malformed_response_is_invalid_and_not_retried():
    session = FakeSession([FakeResponse(200, json_payload={"unexpected": "shape"})])
    result = retrieve_cpi(date(2025, 5, 1), date(2025, 6, 30), api_key="secret-key", session=session)
    assert result["status"] == "invalid"
    assert result["observations"] == []
    assert len(session.calls) == 1


def test_retrieve_cpi_zero_observations_is_invalid():
    session = FakeSession([FakeResponse(200, _valid_payload([]))])
    result = retrieve_cpi(date(2025, 5, 1), date(2025, 6, 30), api_key="secret-key", session=session)
    assert result["status"] == "invalid"
    assert result["observations"] == []


@pytest.mark.parametrize("status_code", [401, 403])
def test_retrieve_cpi_auth_failure_is_invalid_and_not_retried(status_code):
    session = FakeSession([FakeResponse(status_code, text="unauthorized")])
    result = retrieve_cpi(date(2025, 5, 1), date(2025, 6, 30), api_key="bad-key", session=session)
    assert result["status"] == "invalid"
    assert result["observations"] == []
    assert len(session.calls) == 1


def test_retrieve_cpi_validation_failure_400_is_invalid_and_not_retried():
    session = FakeSession([FakeResponse(400, text="Bad Request. Variable series_id is not a known series.")])
    result = retrieve_cpi(date(2025, 5, 1), date(2025, 6, 30), api_key="secret-key", session=session)
    assert result["status"] == "invalid"
    assert result["observations"] == []
    assert len(session.calls) == 1


def test_retrieve_cpi_transient_failure_then_success_retries():
    session = FakeSession([
        FakeResponse(503, text="service unavailable"),
        FakeResponse(200, _valid_payload([{"date": "2025-06-01", "value": "321.0"}])),
    ])
    result = retrieve_cpi(
        date(2025, 5, 1), date(2025, 6, 30), api_key="secret-key", session=session, backoff_seconds=0
    )
    assert result["status"] == "available"
    assert len(session.calls) == 2


def test_retrieve_cpi_transient_failure_exhausted_is_unavailable():
    session = FakeSession([
        ConnectionError("connection reset"),
        ConnectionError("connection reset"),
        ConnectionError("connection reset"),
    ])
    result = retrieve_cpi(
        date(2025, 5, 1), date(2025, 6, 30), api_key="secret-key", session=session,
        max_attempts=3, backoff_seconds=0,
    )
    assert result["status"] == "unavailable"
    assert result["observations"] == []
    assert len(session.calls) == 3


def test_retrieve_cpi_server_error_retries_up_to_max_attempts():
    session = FakeSession([
        FakeResponse(500, text="internal error"),
        FakeResponse(502, text="bad gateway"),
        FakeResponse(200, _valid_payload([{"date": "2025-06-01", "value": "321.0"}])),
    ])
    result = retrieve_cpi(
        date(2025, 5, 1), date(2025, 6, 30), api_key="secret-key", session=session,
        max_attempts=3, backoff_seconds=0,
    )
    assert result["status"] == "available"
    assert len(session.calls) == 3


def test_redact_strips_api_key_from_error_text():
    text = f"FRED request failed: GET {FRED_OBSERVATIONS_URL}?series_id=CPIAUCNS&api_key=super-secret-value&file_type=json"
    redacted = _redact(text)
    assert "super-secret-value" not in redacted
    assert "api_key=***" in redacted


@pytest.mark.integration
def test_retrieve_cpi_against_live_fred_api():
    import os

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        pytest.skip("FRED_API_KEY not available")

    result = retrieve_cpi(date(2025, 1, 1), date(2025, 6, 30), api_key=api_key)
    assert result["status"] == "available"
    assert result["observations"]
    for observation in result["observations"]:
        assert "date" in observation
        assert isinstance(observation["value"], float)
        assert observation["value"] > 0
