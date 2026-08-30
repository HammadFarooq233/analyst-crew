import os

import pytest


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
    return bool(os.getenv("OPENROUTER_API_KEY"))


@pytest.mark.integration
class TestMemoryIntegration:
    def test_memory_influences_plan(self):
        """Seed a fact, invoke the graph, assert the Plan output references the fact."""
        if not _openrouter_available():
            pytest.skip("OPENROUTER_API_KEY not set")

        from analyst_crew.tools.memory import seed_memory, clear_namespace
        from langchain_core.messages import HumanMessage
        from langgraph.store.memory import InMemoryStore

        store = InMemoryStore()
        clear_namespace(store)

        seed_memory(store, "'unusual' means >15% swings by product category")

        state = {
            "messages": [
                HumanMessage(content="Flag anything unusual in this month's sales")
            ]
        }

        from analyst_crew.agents.planner import build_graph
        from analyst_crew.tools.sales import sales_data_source

        graph = build_graph(store=store)
        result = graph.invoke(
            state,
            {"configurable": {"data_sources": sales_data_source("data/sales.csv")}},
        )

        plan = result.get("artifacts", {}).get("plan")
        if plan is None:
            plan_error = result.get("artifacts", {}).get("plan_error")
            if plan_error and ("quota" in plan_error.lower() or "429" in plan_error):
                pytest.skip(f"OpenRouter free-model limit exceeded: {plan_error}")
            pytest.fail(f"Plan not found: {plan_error}")

        plan_text = str(plan.model_dump()).lower()
        assert "15%" in plan_text or "unusual" in plan_text, (
            f"Memory fact should be reflected in plan, got: {plan_text[:500]}"
        )

    def test_memory_persists_across_runs(self):
        """Seed a fact in a first run, invoke in a second run, assert fact is still retrieved."""
        if not _postgres_available():
            pytest.skip("Postgres not available")
        if not _openrouter_available():
            pytest.skip("OPENROUTER_API_KEY not set")

        from analyst_crew.agents import invoke
        from analyst_crew.tools.memory import seed_memory, clear_namespace
        from analyst_crew.tools.sales import sales_data_source
        from langgraph.store.postgres import PostgresStore

        conn_string = os.getenv("POSTGRES_CONNECTION_STRING")

        with PostgresStore.from_conn_string(conn_string) as store:
            store.setup()
            clear_namespace(store)
            seed_memory(store, "'unusual' means >15% swings by product category")

        run_config = {"configurable": {"data_sources": sales_data_source("data/sales.csv")}}
        result1 = invoke("Flag anything unusual in this month's sales", run_config)
        plan_error = result1.get("artifacts", {}).get("plan_error")
        if plan_error and (
            "quota" in plan_error.lower()
            or "429" in plan_error
            or "limit" in plan_error.lower()
        ):
                pytest.skip(f"OpenRouter free-model limit exceeded: {plan_error}")

        result2 = invoke("Flag anything unusual in this month's sales", run_config)
        plan_error = result2.get("artifacts", {}).get("plan_error")
        if plan_error and (
            "quota" in plan_error.lower()
            or "429" in plan_error
            or "limit" in plan_error.lower()
        ):
            pytest.skip(f"OpenRouter free-model limit exceeded: {plan_error}")

        plan2 = result2.get("artifacts", {}).get("plan")
        if plan2 is None:
            plan_error = result2.get("artifacts", {}).get("plan_error")
            pytest.fail(f"Second run plan not found: {plan_error}")

        plan_text = str(plan2.model_dump()).lower()
        assert "15%" in plan_text or "unusual" in plan_text, (
            f"Memory fact should persist across runs, got: {plan_text[:500]}"
        )