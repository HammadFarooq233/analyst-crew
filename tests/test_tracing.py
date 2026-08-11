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

        thread_id = f"test-trace-{uuid.uuid4().hex[:8]}"

        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": "test-user",
            }
        }

        result = invoke("Compare revenue trends across Q1 and Q2", config)

        assert "current_step" in result
        plan_error = result.get("artifacts", {}).get("plan_error")
        if plan_error is not None:
            pytest.skip(f"Skipping trace check — API error: {plan_error}")

        host = os.getenv("LANGFUSE_HOST", "")
        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=host,
        )

        import time

        trace_found = False
        # Self-hosted Langfuse v2 may take several seconds to index ingestion.
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and not trace_found:
            traces = client.api.trace.list(tags=["agent:Planner"])
            trace_found = any(
                hasattr(t, "session_id") and t.session_id == thread_id
                for t in traces
            )
            if not trace_found:
                time.sleep(1)

        assert trace_found, (
            f"No trace found for thread_id={thread_id} with tag agent:Planner"
        )
