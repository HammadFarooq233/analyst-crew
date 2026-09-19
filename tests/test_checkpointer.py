from analyst_crew.llm import credentials_available as llm_credentials_available
import csv
import os
from uuid import uuid4

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


def _openrouter_available() -> bool:
    return llm_credentials_available()


@pytest.mark.integration
class TestStatePersistence:
    def test_state_persistence(self):
        if not _postgres_available():
            pytest.skip("Postgres not available")
        if not _openrouter_available():
            pytest.skip("LLM credentials not set")

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


@pytest.mark.integration
class TestResumeAgainstPostgres:
    """Live-Postgres counterpart to the offline MemorySaver resume coverage
    in ``tests/test_resume.py`` (issue #11) — proves execution-key reuse
    also holds against the real ``PostgresSaver`` checkpointer the Crew
    uses in production. Scripted Fetcher/Analyst/Planner models are used so
    this only requires Postgres, not a live LLM."""

    def test_unchanged_resume_reuses_via_postgres_checkpointer(self, tmp_path):
        if not _postgres_available():
            pytest.skip("Postgres not available")
        pytest.importorskip("duckdb")
        pytest.importorskip("pyarrow")

        from langgraph.checkpoint.postgres import PostgresSaver

        from analyst_crew.agents import Plan, Step
        from analyst_crew.tools.sales import CANONICAL_FIELDS, sales_data_source

        source = tmp_path / "sales.csv"
        with source.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
            writer.writeheader()
            for order_id, order_date in (("M1", "2025-05-01"), ("J1", "2025-06-01")):
                writer.writerow({
                    "order_id": order_id, "order_date": order_date, "product_id": "P001",
                    "product_name": "Desk Lamp", "category": "Office", "region": "North",
                    "sales_channel": "web", "customer_segment": "consumer",
                    "quantity": 1, "unit_price": 10, "discount_amount": 0, "returned_quantity": 0,
                })

        class ScriptedPlannerModel:
            def __init__(self, plan):
                self.plan = plan

            def with_structured_output(self, *_args, **_kwargs):
                return self

            def invoke(self, _messages):
                return self.plan

        class CountingScriptedFetcherModel:
            def __init__(self):
                self.calls = 0

            def invoke(self, _context):
                self.calls += 1
                return {"tool": "CSV.read", "source_id": "sales_csv"}

        class CountingScriptedAnalystModel:
            def __init__(self):
                self.calls = 0

            def invoke(self, request):
                self.calls += 1
                artifact_id = next(iter(request["available_artifacts"].values()))
                return {"tool": "DuckDB.sales_baseline", "dataset_artifact_id": artifact_id}

        fetch = Step(
            step_id="fetch_sales", agent="Fetcher", objective="Fetch normalized sales",
            source_ids=["sales_csv"], expected_output="sales_dataset", declared_artifacts=["sales_dataset"],
        )
        analyze = Step(
            step_id="analyze_baseline", agent="Analyst", objective="Compare baseline sales periods",
            expected_output="baseline_findings", declared_artifacts=["baseline_findings"], depends_on=["fetch_sales"],
        )
        plan = Plan(steps=[fetch, analyze], user_question="Compare baseline sales")

        thread_id = f"test-resume-postgres-thread-{uuid4()}"
        conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
        with PostgresSaver.from_conn_string(conn_string) as checkpointer:
            checkpointer.setup()

            def _invoke(fetcher_model, analyst_model):
                runtime = {
                    "configurable": {
                        "thread_id": thread_id,
                        "data_sources": sales_data_source(source),
                        "artifact_dir": tmp_path / "artifacts",
                        "available_agents": ["Fetcher", "Analyst"],
                        "planner_model": ScriptedPlannerModel(plan),
                        "fetcher_model": fetcher_model,
                        "analyst_model": analyst_model,
                        "analysis_parameters": {"as_of_date": "2025-06-30"},
                    }
                }
                return invoke("Compare baseline sales", runtime, checkpointer=checkpointer)

            fetcher1, analyst1 = CountingScriptedFetcherModel(), CountingScriptedAnalystModel()
            result1 = _invoke(fetcher1, analyst1)
            assert result1["artifacts"]["step_outcomes"]["fetch_sales"]["resume_outcome"] == "recomputed"
            assert fetcher1.calls == 1
            assert analyst1.calls == 1

            fetcher2, analyst2 = CountingScriptedFetcherModel(), CountingScriptedAnalystModel()
            result2 = _invoke(fetcher2, analyst2)
            assert result2["artifacts"]["step_outcomes"]["fetch_sales"]["resume_outcome"] == "reused"
            assert result2["artifacts"]["step_outcomes"]["analyze_baseline"]["resume_outcome"] == "reused"
            assert fetcher2.calls == 0
            assert analyst2.calls == 0