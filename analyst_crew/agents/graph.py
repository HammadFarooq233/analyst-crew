"""Crew graph wiring and the ``invoke()`` entry point."""
import os
import uuid
from contextlib import ExitStack
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from langgraph.store.postgres import PostgresStore

from analyst_crew.agents import planner as planner_module
from analyst_crew.agents.analyst import analyst_node
from analyst_crew.agents.contracts import Plan
from analyst_crew.agents.fetcher import fetcher_node
from analyst_crew.agents.critic import critic_node
from analyst_crew.agents.reporter import reporter_node
from analyst_crew.agents.visualizer import visualizer_node
from analyst_crew.agents.tracing import _finish_langfuse_trace, _run_metadata, get_langfuse_handler
from analyst_crew.memory.node import memory_node
from analyst_crew.memory.summary import memory_update_node
from analyst_crew.memory import MemoryStore, MemoryUnavailable, UnavailableStore
from analyst_crew.state import CrewState
from analyst_crew.state.serde import crew_serde


def build_graph(checkpointer=None, store=None, runtime_config: dict | None = None):
    def merged_config(config: dict | None) -> dict:
        base = runtime_config or {}
        incoming = config or {}
        return {
            **base,
            **incoming,
            "configurable": {
                **base.get("configurable", {}),
                **incoming.get("configurable", {}),
            },
        }

    graph = StateGraph(CrewState)
    graph.add_node("planner_node", lambda state, config=None: planner_module.planner_node(state, merged_config(config)))
    graph.add_node("fetcher_node", lambda state, config=None: fetcher_node(state, merged_config(config)))
    graph.add_node("analyst_node", lambda state, config=None: analyst_node(state, merged_config(config)))
    graph.add_node("memory_node", lambda state, config=None: memory_node(state, merged_config(config)))
    graph.add_edge(START, "planner_node")
    def next_node(state: CrewState) -> str:
        plan = state.get("artifacts", {}).get("plan")
        return "memory_node" if isinstance(plan, Plan) and any(step.agent == "Fetcher" for step in plan.steps) else END
    graph.add_conditional_edges("planner_node", next_node, {"memory_node": "memory_node", END: END})
    # The Crew resolves stored Typed Preferences into analysis parameters before any Step that uses them.
    graph.add_edge("memory_node", "fetcher_node")
    def after_fetcher(state: CrewState) -> str:
        if state.get("artifacts", {}).get("terminal_failure"):
            return END
        plan = state.get("artifacts", {}).get("plan")
        return "analyst_node" if isinstance(plan, Plan) and any(step.agent == "Analyst" for step in plan.steps) else END
    graph.add_conditional_edges("fetcher_node", after_fetcher, {"analyst_node": "analyst_node", END: "memory_update_node"})
    graph.add_node("reporter_node", lambda state, config=None: reporter_node(state, merged_config(config)))
    graph.add_node("visualizer_node", lambda state, config=None: visualizer_node(state, merged_config(config)))

    def has_step(state: CrewState, agent: str) -> bool:
        plan = state.get("artifacts", {}).get("plan")
        return isinstance(plan, Plan) and any(step.agent == agent for step in plan.steps)

    def after_analyst(state: CrewState) -> str:
        if state.get("artifacts", {}).get("terminal_failure"):
            return END
        if has_step(state, "Visualizer"):
            return "visualizer_node"
        return "reporter_node" if has_step(state, "Reporter") else END
    graph.add_conditional_edges("analyst_node", after_analyst, {
        "visualizer_node": "visualizer_node", "reporter_node": "reporter_node", END: "memory_update_node",
    })

    def after_visualizer(state: CrewState) -> str:
        # A chart failure is recorded on the Step and never stops the run; the Report ships without charts.
        return "reporter_node" if has_step(state, "Reporter") else END
    graph.add_conditional_edges("visualizer_node", after_visualizer, {"reporter_node": "reporter_node", END: "memory_update_node"})
    graph.add_node("critic_node", lambda state, config=None: critic_node(state, merged_config(config)))

    def after_reporter(state: CrewState) -> str:
        # A Reporter failure still ships its Fallback Report; the Critic reviews whichever Report exists.
        failure = state.get("artifacts", {}).get("terminal_failure")
        if failure and failure.get("required", True):
            return END
        return "critic_node" if has_step(state, "Critic") and state.get("artifacts", {}).get("report") is not None else END
    graph.add_conditional_edges("reporter_node", after_reporter, {"critic_node": "critic_node", END: "memory_update_node"})
    graph.add_edge("critic_node", "memory_update_node")
    # The Crew's Memory Update step runs last, from run records only (ADR-0005). It is not an Agent.
    graph.add_node("memory_update_node", lambda state, config=None: memory_update_node(state, merged_config(config)))
    graph.add_edge("memory_update_node", END)
    return graph.compile(checkpointer=checkpointer, store=store)


def _retired(store):
    """Drop the retired Planner-only namespace the first time a store is used; never fails the run."""
    try:
        MemoryStore(store).retire_legacy_namespace()
    except MemoryUnavailable:
        pass
    return store


def invoke(
    user_question: str,
    config: dict | None = None,
    *,
    checkpointer: Any = None,
    store: Any = None,
) -> CrewState:
    """Run one Crew turn.

    ``checkpointer``/``store`` let a caller supply their own checkpointer
    (e.g. an in-memory ``langgraph.checkpoint.memory.MemorySaver`` for
    offline resume tests) instead of the default Postgres-or-nothing
    selection below. Without a live ``POSTGRES_CONNECTION_STRING`` and no
    explicit ``checkpointer``, the graph runs uncheckpointed exactly as
    before — resume/reuse behavior (issue #11) requires a checkpointer that
    actually persists CrewState across calls with the same ``thread_id``.
    """
    if config is None:
        config = {}
    config.setdefault("configurable", {})
    thread_id = config["configurable"].setdefault("thread_id", str(uuid.uuid4()))
    user_id = config["configurable"].setdefault("user_id", "default")

    state = {"messages": [HumanMessage(content=user_question)]}

    langfuse_client, langfuse_trace, handler = get_langfuse_handler(
        session_id=thread_id,
        tags=["agent:Planner"],
        user_id=user_id,
    )
    if handler is not None:
        config.setdefault("callbacks", [])
        config["callbacks"].append(handler)

    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")

    def _do_invoke(_state, _config, _checkpointer, _store):
        graph = build_graph(
            checkpointer=_checkpointer,
            store=_store,
            runtime_config=_config,
        )
        return graph.invoke(_state, _config)

    def _run():
        if checkpointer is not None:
            return _do_invoke(state, config, checkpointer, _retired(store if store is not None else InMemoryStore()))
        if conn_string:
            with PostgresSaver.from_conn_string(conn_string) as pg_checkpointer, ExitStack() as stack:
                pg_checkpointer.serde = crew_serde()
                pg_checkpointer.setup()
                # Memory persists across runs in Postgres. If only the Memory store is down the run
                # continues with defaults and states that in the Report; it never fails the run.
                try:
                    pg_store = stack.enter_context(PostgresStore.from_conn_string(conn_string))
                    pg_store.setup()
                except Exception:  # noqa: BLE001
                    pg_store = UnavailableStore()
                return _do_invoke(state, config, pg_checkpointer, _retired(pg_store))
        return _do_invoke(state, config, None, _retired(store if store is not None else InMemoryStore()))

    result = _run()
    _finish_langfuse_trace(langfuse_client, langfuse_trace, metadata=_run_metadata(result))
    if langfuse_trace is not None:
        result.setdefault("artifacts", {})["langfuse_trace_id"] = langfuse_trace.id

    return result
