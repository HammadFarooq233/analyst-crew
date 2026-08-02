import os
import uuid
from typing import Any, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, model_validator

import config
from analyst_crew.state import CrewState


class Step(BaseModel):
    step_id: str
    agent: Literal["Fetcher", "Analyst", "Visualizer", "Critic", "Reporter"]
    produces: str
    depends_on: list[str] = []

    @model_validator(mode="after")
    def validate_depends_on(self) -> "Step":
        if self.step_id in self.depends_on:
            raise ValueError(
                f"Step '{self.step_id}' references itself in depends_on"
            )
        return self


class Plan(BaseModel):
    steps: list[Step]
    user_question: str

    @model_validator(mode="after")
    def validate_step_ids_unique(self) -> "Plan":
        ids = [s.step_id for s in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Step step_ids must be unique")
        return self

    @model_validator(mode="after")
    def validate_dependency_order(self) -> "Plan":
        step_ids = {s.step_id for s in self.steps}
        produced: set[str] = set()
        for step in self.steps:
            for dep in step.depends_on:
                if dep not in step_ids:
                    raise ValueError(
                        f"Step '{step.step_id}' depends on unknown step '{dep}'"
                    )
                if dep not in produced:
                    raise ValueError(
                        f"Step '{step.step_id}' depends on step '{dep}' "
                        "which appears after it in the plan"
                    )
            produced.add(step.step_id)
        return self


SYSTEM_PROMPT = """You are the Planner Agent for Analyst Crew, a multi-agent data analysis copilot.

Your role is to decompose a user's analytical question into an ordered Plan of Steps.
Each Step names a target Agent, describes what it produces, and lists which prior Steps it depends on.

Available Agents:
- Fetcher: Retrieves data from sources (CSV, Postgres, FRED API). Returns raw data — no analysis, no interpretation.
- Analyst: Performs computation on data — statistics, joins, comparisons, trend detection. Consumes data, produces findings.
- Visualizer: Generates charts and tables from findings.
- Critic: Reviews a draft report for uncited claims, missing caveats, and logical gaps.
- Reporter: Writes the narrative report from findings and charts. Synthesizes results into prose, cites specific numbers.

Before drafting the Plan, call search_memory to retrieve any relevant facts, preferences, or conventions the user has established in prior sessions. Incorporate these into your Plan decisions.

Rules:
1. Produce between 2 and 5 Steps.
2. Each Step must have a unique step_id (snake_case string).
3. Each Step must name exactly one Agent from the list above.
4. depends_on must reference only step_ids from earlier Steps in the plan (never itself, never a later Step).
5. The Steps must form a valid dependency order — no cycles, no forward references.
6. Think about the data flow: Fetcher provides data, Analyst computes on it, Visualizer creates charts, Reporter writes prose, Critic reviews.
7. Every question should produce a Reporter Step as the final Step.
8. Include the exact user_question text in the Plan."""


class PlannerAgent:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        store=None,
    ):
        self.model = model or str(config.get("llm.model", "gemini-2.0-flash-lite"))
        self.temperature = (
            temperature
            if temperature is not None
            else float(config.get("llm.temperature", 0.0))
        )
        self.max_tokens = max_tokens or int(config.get("llm.max_tokens", 4096))

        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file or environment."
            )

        self._llm = ChatGoogleGenerativeAI(
            model=self.model,
            google_api_key=self._api_key,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )

        if store is not None:
            from analyst_crew.tools.memory import create_memory_tool

            memory_tool = create_memory_tool(store)
            self._llm = self._llm.bind_tools([memory_tool])

        self._chain = self._llm.with_structured_output(Plan)

    def generate_plan(self, question: str) -> Plan:
        if not question.strip():
            raise ValueError("Question must not be empty.")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]

        try:
            result = self._chain.invoke(messages)
        except Exception as e:
            msg = str(e)
            if "quota" in msg.lower() or "429" in msg:
                raise RuntimeError(
                    "Gemini API quota exceeded. Check your usage limits or billing."
                ) from e
            raise RuntimeError(f"Planner LLM call failed: {msg}") from e

        if not isinstance(result, Plan):
            raise RuntimeError(
                f"LLM returned unexpected type {type(result).__name__}, expected Plan."
            )

        if not result.steps:
            raise RuntimeError("LLM returned an empty Plan (no Steps).")

        return result


def _extract_question(state: CrewState) -> str:
    messages = state.get("messages", [])
    if messages:
        return messages[0].content
    return ""


def planner_node(state: CrewState) -> dict[str, Any]:
    question = _extract_question(state)

    try:
        store = get_store()
        agent = PlannerAgent(store=store)
        plan = agent.generate_plan(question)
    except RuntimeError as e:
        return {
            "current_step": "planner_node",
            "artifacts": {"plan_error": str(e)},
        }

    return {
        "current_step": "planner_node",
        "artifacts": {"plan": plan},
    }


def get_langfuse_handler(
    session_id: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
) -> CallbackHandler | None:
    if not all(
        [
            os.getenv("LANGFUSE_PUBLIC_KEY"),
            os.getenv("LANGFUSE_SECRET_KEY"),
            os.getenv("LANGFUSE_HOST"),
        ]
    ):
        return None
    try:
        kwargs = {}
        if session_id is not None:
            kwargs["session_id"] = session_id
        if tags is not None:
            kwargs["tags"] = tags
        if user_id is not None:
            kwargs["user_id"] = user_id
        return CallbackHandler(**kwargs)
    except Exception:
        return None


def build_graph(checkpointer=None, store=None):
    graph = StateGraph(CrewState)
    graph.add_node("planner_node", planner_node)
    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", END)
    return graph.compile(checkpointer=checkpointer, store=store)


def invoke(user_question: str, config: dict | None = None) -> CrewState:
    if config is None:
        config = {}
    config.setdefault("configurable", {})
    thread_id = config["configurable"].setdefault("thread_id", str(uuid.uuid4()))
    user_id = config["configurable"].setdefault("user_id", "default")

    state = {"messages": [HumanMessage(content=user_question)]}

    handler = get_langfuse_handler(
        session_id=thread_id,
        tags=["agent:Planner"],
        user_id=user_id,
    )
    if handler is not None:
        config.setdefault("callbacks", [])
        config["callbacks"].append(handler)

    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")

    def _do_invoke(_state, _config, _checkpointer, _store):
        graph = build_graph(checkpointer=_checkpointer, store=_store)
        return graph.invoke(_state, _config)

    if conn_string:
        with (
            PostgresSaver.from_conn_string(conn_string) as checkpointer,
            PostgresStore.from_conn_string(conn_string) as store,
        ):
            checkpointer.setup()
            store.setup()
            return _do_invoke(state, config, checkpointer, store)

    memory_store = InMemoryStore()
    return _do_invoke(state, config, None, memory_store)