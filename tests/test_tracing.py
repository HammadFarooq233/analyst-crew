import os
from urllib.request import urlopen

import pytest
from dotenv import load_dotenv

load_dotenv()


def _langfuse_available() -> bool:
    keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    if not all(os.getenv(k) for k in keys):
        return False
    try:
        host = os.getenv("LANGFUSE_HOST", "")
        with urlopen(f"{host.rstrip('/')}/api/public/health", timeout=3) as response:
            return response.status == 200
    except Exception:
        return False


def _openrouter_available() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


@pytest.mark.integration
class TestTracing:
    def test_trace_exists(self):
        if not _langfuse_available():
            pytest.skip("Langfuse not available")
        if not _openrouter_available():
            pytest.skip("OPENROUTER_API_KEY not set")

        import uuid

        from langfuse import Langfuse

        from analyst_crew.agents import invoke
        from analyst_crew.tools.sales import sales_data_source

        thread_id = f"test-trace-{uuid.uuid4().hex[:8]}"

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": "test-user",
                "data_sources": sales_data_source("data/sales.csv"),
            }
        }

        result = invoke("Compare revenue trends across Q1 and Q2", config)

        assert "current_step" in result
        plan_error = result.get("artifacts", {}).get("plan_error")
        if plan_error is not None:
            pytest.skip(f"Skipping trace check — API error: {plan_error}")

        trace_id = result.get("artifacts", {}).get("langfuse_trace_id")
        assert trace_id is not None, "invoke() did not record a langfuse_trace_id"

        host = os.getenv("LANGFUSE_HOST", "")
        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=host,
        )

        import time

        # Fetch the trace directly by id — a fast, consistently-indexed
        # lookup — rather than searching by tag, which goes through Langfuse
        # v2's async ClickHouse indexing and can lag well behind the direct
        # id lookup, making tag-based polling flaky under load.
        trace = None
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and trace is None:
            try:
                trace = client.api.trace.get(trace_id)
            except Exception:
                trace = None
            if trace is None:
                time.sleep(1)

        assert trace is not None, f"No trace found for trace_id={trace_id}"
        assert trace.session_id == thread_id
        assert "agent:Planner" in (trace.tags or [])
