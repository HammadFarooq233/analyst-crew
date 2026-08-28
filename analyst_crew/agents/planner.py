import os
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.memory import InMemoryStore
from langgraph.config import get_store
from langfuse import Langfuse
from pydantic import AliasChoices, BaseModel, Field, model_validator

import config
from analyst_crew.state import CrewState


AGENTS = ("Fetcher", "Analyst", "Visualizer", "Critic", "Reporter")
PHASE1_AGENTS = ("Fetcher", "Analyst")


class SourceReference(BaseModel):
    source_id: str
    reference: Any = Field(description="Catalog-resolved source descriptor")

    @model_validator(mode="after")
    def validate_reference(self) -> "SourceReference":
        if _contains_secret({"source_id": self.source_id}) or _contains_secret(self.reference) or _is_raw_location(self.reference):
            raise ValueError("SourceReference cannot contain an unresolved filesystem location")
        return self


class ArtifactBinding(BaseModel):
    artifact_id: str
    reference: Any = None
    status: Literal["resolved", "declared"] = "declared"

    @model_validator(mode="after")
    def validate_reference(self) -> "ArtifactBinding":
        if _contains_secret({"artifact_id": self.artifact_id}) or _contains_secret(self.reference) or _is_raw_location(self.reference):
            raise ValueError("ArtifactBinding cannot contain an unresolved filesystem location")
        return self


def _is_raw_location(value: Any) -> bool:
    if isinstance(value, str):
        return value.startswith(("/", "~", "file://"))
    if isinstance(value, Mapping):
        return any(_is_raw_location(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_is_raw_location(item) for item in value)
    return False


def _contains_secret(value: Any, *, field_name: str = "") -> bool:
    if isinstance(value, Mapping):
        return any(_contains_secret(v, field_name=str(k)) for k, v in value.items())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_secret(v, field_name=field_name) for v in value)
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    credential_field = any(token in field_name.lower() for token in ("password", "secret", "api_key", "apikey", "token", "credential"))
    # Contract IDs, prose, and artifact values may be arbitrary strings. Only
    # credential-shaped content is rejected here; source locations are checked
    # at the catalog boundary, where their field meaning is known.
    location_field = any(token in field_name.lower() for token in ("source_location", "source_path", "source_uri", "location", "path", "uri"))
    explicit_shape = any(token in lowered for token in ("password=", "secret=", "api_key=", "apikey=", "token="))
    explicit_location = any(token in lowered for token in (
        "location=", "path=", "uri=", "source_location=", "source_path=", "source_uri="
    ))
    return credential_field or explicit_shape or explicit_location or (location_field and value.startswith(("/", "~", "file://")))


class Step(BaseModel):
    step_id: str
    agent: Literal["Fetcher", "Analyst", "Visualizer", "Critic", "Reporter"]
    objective: str = Field(validation_alias=AliasChoices("objective", "produces", "description"))
    source_ids: list[str] = Field(default_factory=list, validation_alias=AliasChoices("source_ids", "data_source_ids"))
    expected_output: str = Field(validation_alias=AliasChoices("expected_output", "produces", "description"))
    declared_artifacts: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)

    @property
    def produces(self) -> str:
        """Phase 0 compatibility name for the expected output."""
        return self.expected_output

    @property
    def data_source_ids(self) -> list[str]:
        return self.source_ids

    @model_validator(mode="after")
    def validate_depends_on(self) -> "Step":
        if not self.step_id.strip():
            raise ValueError("Step step_id must not be empty")
        if not self.objective.strip() or not self.expected_output.strip():
            raise ValueError("Step objective and expected_output must not be empty")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError(f"Step '{self.step_id}' has duplicate source_ids")
        if _contains_secret({"objective": self.objective, "expected_output": self.expected_output,
                             "source_ids": self.source_ids, "declared_artifacts": self.declared_artifacts}):
            raise ValueError(f"Step '{self.step_id}' contains a secret or unresolved location")
        if self.step_id in self.depends_on:
            raise ValueError(
                f"Step '{self.step_id}' references itself in depends_on"
            )
        return self


class Plan(BaseModel):
    steps: list[Step] = Field(validation_alias=AliasChoices("steps", "plan"))
    user_question: str

    @model_validator(mode="after")
    def validate_safe_question(self) -> "Plan":
        if _contains_secret({"user_question": self.user_question}):
            raise ValueError("Plan contains a secret or unresolved location")
        return self

    @property
    def agents(self) -> set[str]:
        return {step.agent for step in self.steps}

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
                    raise ValueError(f"Step '{step.step_id}' depends on unknown step '{dep}'")
                if dep not in produced:
                    raise ValueError(f"Step '{step.step_id}' depends on step '{dep}' which appears after it in the plan")
            produced.add(step.step_id)
        return self


class Handoff(BaseModel):
    step: Step
    source_ids: list[str] = Field(default_factory=list)
    source_values: dict[str, SourceReference] = Field(default_factory=dict)
    declared_artifacts: dict[str, ArtifactBinding] = Field(default_factory=dict)
    input_artifacts: dict[str, Any] = Field(default_factory=dict, validation_alias=AliasChoices("input_artifacts", "artifacts"))
    user_question: str
    analysis_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_safe_values(self) -> "Handoff":
        if _contains_secret({"source_ids": self.source_ids,
                             "user_question": self.user_question,
                             "analysis_parameters": self.analysis_parameters,
                             "input_artifacts": self.input_artifacts}):
            raise ValueError("Handoff contains a secret or unresolved location")
        if (_is_raw_location(self.input_artifacts)
                or _is_raw_location(self.analysis_parameters)
                or _is_raw_location(self.source_values)):
            raise ValueError("Handoff contains an unresolved filesystem location")
        return self


def resolve_handoff(step: Step, plan: Plan, artifacts: Mapping[str, Any],
                    data_sources: Mapping[str, Any] | Sequence[str] = (),
                    analysis_parameters: Mapping[str, Any] | None = None) -> Handoff:
    plan_step = next((candidate for candidate in plan.steps if candidate.step_id == step.step_id), None)
    if plan_step is None or plan_step != step:
        raise ValueError(f"Step '{step.step_id}' does not match the Step contract in the Plan")
    catalog = set(data_sources.keys()) if isinstance(data_sources, Mapping) else set(data_sources)
    unknown = set(step.source_ids) - catalog
    if unknown:
        raise ValueError(f"Step '{step.step_id}' references unknown Data Sources: {sorted(unknown)}")
    # Dependencies refer to prior Steps, but their runtime inputs are keyed by
    # the artifact IDs those Steps declare—not by implementation step IDs.
    required_artifacts: list[str] = []
    for dependency_id in step.depends_on:
        dependency = next(candidate for candidate in plan.steps if candidate.step_id == dependency_id)
        required_artifacts.extend(dependency.declared_artifacts or [dependency.expected_output])
    required_artifacts = list(dict.fromkeys(required_artifacts))
    missing = set(required_artifacts) - set(artifacts)
    if missing:
        raise ValueError(f"Step '{step.step_id}' is missing input artifacts: {sorted(missing)}")
    declared_names = list(step.declared_artifacts or [step.expected_output])
    declared = {
        name: ArtifactBinding(
            artifact_id=name,
            reference=artifacts.get(name, {
                "kind": "planned_output",
                "producer_step_id": step.step_id,
                "artifact_id": name,
            }),
            status="resolved" if name in artifacts else "declared",
        )
        for name in declared_names
    }
    inputs = {artifact: artifacts[artifact] for artifact in required_artifacts}
    inputs.update({artifact: artifacts[artifact] for artifact in declared if artifact in artifacts})
    source_values = ({source: SourceReference(source_id=source, reference=data_sources[source]) for source in step.source_ids}
                     if isinstance(data_sources, Mapping)
                   else {source: SourceReference(source_id=source, reference={
                       "kind": "catalog_reference", "catalog": "data_sources", "source_id": source,
                   }) for source in step.source_ids})
    return Handoff(step=step, source_ids=list(step.source_ids), source_values=source_values,
                   declared_artifacts=declared, input_artifacts=inputs,
                   user_question=plan.user_question,
                   analysis_parameters=dict(analysis_parameters or {}))


class Crew:
    """Capability boundary used by callers to validate and resolve Plans."""

    def __init__(self, available_agents: Sequence[str] = PHASE1_AGENTS,
                 data_sources: Mapping[str, Any] | Sequence[str] = ()):
        self.available_agents = set(available_agents)
        unknown = self.available_agents - set(AGENTS)
        if unknown:
            raise ValueError(f"Unknown available Agents: {sorted(unknown)}")
        self.data_sources = data_sources

    def validate_plan(self, plan: Plan) -> Plan:
        unavailable = plan.agents - self.available_agents
        if unavailable:
            raise ValueError(f"Plan references unavailable Agents: {sorted(unavailable)}")
        unknown = {source for step in plan.steps for source in step.source_ids}
        catalog = set(self.data_sources.keys()) if isinstance(self.data_sources, Mapping) else set(self.data_sources)
        if unknown - catalog:
            raise ValueError(f"Plan references unknown Data Sources: {sorted(unknown - catalog)}")
        return plan

    def resolve_handoff(self, step: Step, plan: Plan, artifacts: Mapping[str, Any],
                        analysis_parameters: Mapping[str, Any] | None = None) -> Handoff:
        self.validate_plan(plan)
        if step.agent not in self.available_agents:
            raise ValueError(f"Step references unavailable Agent: {step.agent}")
        return resolve_handoff(step, plan, artifacts, self.data_sources, analysis_parameters)

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
8. Include the exact user_question text in the Plan.
9. Use exactly these output keys: top-level `steps`, and for each Step `produces`.
    Do not use `plan` or `description` as substitutes."""


class PlannerAgent:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        store=None,
        available_agents: Sequence[str] | None = None,
        data_sources: Mapping[str, Any] | Sequence[str] = (),
        llm=None,
    ):
        self.available_agents = set(PHASE1_AGENTS if available_agents is None else available_agents)
        invalid = self.available_agents - set(AGENTS)
        if invalid:
            raise ValueError(f"Unknown available Agents: {sorted(invalid)}")
        self.data_sources = set(data_sources.keys()) if isinstance(data_sources, Mapping) else set(data_sources)
        self.model = model or str(config.get("llm.model", "google/gemma-4-26b-a4b-it:free"))
        self.temperature = (
            temperature
            if temperature is not None
            else float(config.get("llm.temperature", 0.0))
        )
        self.max_tokens = max_tokens or int(config.get("llm.max_tokens", 4096))

        self._api_key = os.getenv("OPENROUTER_API_KEY")
        if llm is None and not self._api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file or environment."
            )

        self._llm = llm or ChatOpenAI(
            model=self.model,
            api_key=self._api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            default_headers={
                "HTTP-Referer": "https://github.com/analyst-crew",
                "X-Title": "Analyst Crew",
            },
        )

        if store is not None:
            from analyst_crew.tools.memory import create_memory_tool

            memory_tool = create_memory_tool(store)
            self._llm = self._llm.bind_tools([memory_tool])

        self._chain = self._llm.with_structured_output(
            Plan,
            method="json_mode",
        )

    def generate_plan(self, question: str, available_agents: Sequence[str] | None = None,
                      data_sources: Mapping[str, Any] | Sequence[str] | None = None) -> Plan:
        if not question.strip():
            raise ValueError("Question must not be empty.")

        allowed_agents = set(available_agents) if available_agents is not None else self.available_agents
        invalid = allowed_agents - set(AGENTS)
        if invalid:
            raise ValueError(f"Unknown available Agents: {sorted(invalid)}")
        if not allowed_agents:
            raise ValueError("available_agents must include at least one Agent")
        source_catalog = (set(data_sources.keys()) if isinstance(data_sources, Mapping)
                          else set(data_sources)) if data_sources is not None else self.data_sources
        capability_prompt = (
            SYSTEM_PROMPT
            + "\n\nValidated capabilities for this run (you MUST use only these Agents):\n"
            + "Agents: " + ", ".join(sorted(allowed_agents))
            + "\nData Sources: " + ", ".join(sorted(source_catalog))
        )
        messages = [
            SystemMessage(content=capability_prompt),
            HumanMessage(content=question),
        ]

        try:
            result = self._chain.invoke(messages)
        except Exception as e:
            msg = str(e)
            if "quota" in msg.lower() or "429" in msg:
                raise RuntimeError(
                    "OpenRouter free-model quota or rate limit exceeded. "
                    "Try again later or select another free model in config/settings.toml."
                ) from e
            raise RuntimeError(f"Planner LLM call failed: {msg}") from e

        if not isinstance(result, Plan):
            raise RuntimeError(
                f"LLM returned unexpected type {type(result).__name__}, expected Plan."
            )

        if not result.steps:
            raise RuntimeError("LLM returned an empty Plan (no Steps).")

        unavailable = result.agents - allowed_agents
        if unavailable:
            raise ValueError(f"Plan references unavailable Agents: {sorted(unavailable)}")
        catalog = (set(data_sources.keys()) if isinstance(data_sources, Mapping) else set(data_sources)) if data_sources is not None else self.data_sources
        unknown_sources = {source for step in result.steps for source in step.source_ids} - catalog
        if unknown_sources:
            raise ValueError(f"Plan references unknown Data Sources: {sorted(unknown_sources)}")

        return result


def _extract_question(state: CrewState) -> str:
    messages = state.get("messages", [])
    if messages:
        return messages[0].content
    return ""


def planner_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    question = _extract_question(state)

    try:
        store = get_store()
        configurable = (config or {}).get("configurable", {})
        agent = PlannerAgent(
            store=store,
            available_agents=configurable.get("available_agents"),
            data_sources=configurable.get("data_sources", ()),
        )
        plan = agent.generate_plan(question)
    except (RuntimeError, ValueError) as e:
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
) -> None:
    # The bundled self-hosted deployment is Langfuse v2. Its ingestion API is
    # compatible with the v2 SDK, but not with the OTLP exporter introduced by
    # newer SDKs. The trace is recorded after the graph run by invoke().
    return None


def _create_langfuse_trace(session_id: str, user_id: str) -> None:
    """Create a trace using the Langfuse v2 SDK matching the server deployment."""
    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not all((host, public_key, secret_key)):
        return

    try:
        client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        client.trace(
            name="planner",
            session_id=session_id,
            user_id=user_id,
            tags=["agent:Planner"],
        )
        client.flush()
        # Langfuse v2 acknowledges ingestion before the trace is queryable.
        # Wait briefly for indexing so callers can safely query after invoke().
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            traces = client.api.trace.list(tags=["agent:Planner"])
            if any(trace.session_id == session_id for trace in traces.data):
                return
            time.sleep(0.5)
    except Exception:
        # Observability must never make a successful planner request fail.
        return


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

    def _run():
        if conn_string:
            with (
                PostgresSaver.from_conn_string(conn_string) as checkpointer,
                PostgresStore.from_conn_string(conn_string) as store,
            ):
                checkpointer.setup()
                store.setup()
                return _do_invoke(state, config, checkpointer, store)
        return _do_invoke(state, config, None, InMemoryStore())

    result = _run()
    _create_langfuse_trace(thread_id, user_id)

    return result
