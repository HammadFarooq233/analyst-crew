import os

import pytest

from analyst_crew.agents import invoke


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


def _gemini_available() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


@pytest.mark.integration
class TestStatePersistence:
    def test_state_persistence(self):
        if not _postgres_available():
            pytest.skip("Postgres not available")
        if not _gemini_available():
            pytest.skip("GEMINI_API_KEY not set")

        thread_id = "test-state-persistence-thread-001"
        config = {
            "configurable": {
                "thread_id": thread_id,
                "user_id": "test-user",
            }
        }

        result1 = invoke("Compare this month's sales to last month", config)

        assert "current_step" in result1
        assert "steps_completed" in result1
        assert "artifacts" in result1
        assert "messages" in result1

        result2 = invoke("Now break it down by region", config)

        assert "current_step" in result2
        assert "steps_completed" in result2
        assert "artifacts" in result2
        assert "messages" in result2

        assert len(result2["messages"]) > len(result1["messages"])
        assert len(result2["steps_completed"]) >= len(result1["steps_completed"])