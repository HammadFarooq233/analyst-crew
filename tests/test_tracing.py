import os

import pytest


def _langfuse_available() -> bool:
    keys = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
    if not all(os.getenv(k) for k in keys):
        return False
    try:
        from langfuse import Langfuse

        host = os.getenv("LANGFUSE_HOST", "")
        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=host,
        )
        client.auth_check()
        return True
    except Exception:
        return False


def _gemini_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


@pytest.mark.integration
class TestTracing:
    def test_trace_exists(self):
        if not _langfuse_available():
            pytest.skip("Langfuse not available")
        if not _gemini_available():
            pytest.skip("GEMINI_API_KEY not set")

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

        import time

        time.sleep(2)

        host = os.getenv("LANGFUSE_HOST", "")
        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=host,
        )

        traces = client.api.trace.list(tags=["agent:Planner"])
        trace_found = any(
            hasattr(t, "session_id") and t.session_id == thread_id
            for t in traces
        )
        assert trace_found, (
            f"No trace found for thread_id={thread_id} with tag agent:Planner"
        )