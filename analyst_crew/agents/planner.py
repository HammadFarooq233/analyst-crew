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
from analyst_crew.tools.analysis import AnalystAgent, production_analyst_model
from analyst_crew.tools.sales import FetcherAgent, production_fetcher_model


AGENTS = ("Fetcher", "Analyst", "Visualizer", "Critic", "Reporter")
PHASE1_AGENTS = ("Fetcher", "Analyst")


class SourceReference(BaseModel):
    source_id: str
    reference: Any = Field(description="Catalog-resolved source descriptor")

    @model_validator(mode="after")
    def validate_reference(self) -> "SourceReference":
        catalogued_csv = isinstance(self.reference, Mapping) and self.reference.get("kind") == "CSV" and isinstance(self.reference.get("location"), str)
        descriptor = {key: value for key, value in self.reference.items() if key != "location"} if catalogued_csv else self.reference
        if _contains_secret({"source_id": self.source_id}) or _contains_secret(descriptor) or (_is_raw_location(self.reference) and not catalogued_csv):
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
        # A Dataset Artifact location is resolved, integrity-bound runtime data,
        # not an Agent-invented source location. Validate the rest of each
        # reference while permitting that single application-owned field.
        def without_resolved_locations(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {
                    key: without_resolved_locations(item)
                    for key, item in value.items()
                    if not any(token in str(key).lower() for token in ("path", "location", "sidecar"))
                }
            if isinstance(value, list):
                return [without_resolved_locations(item) for item in value]
            return value

        safe_input_artifacts = {
            key: (without_resolved_locations(artifact)
                  if isinstance(artifact, Mapping) and artifact.get("kind") == "Dataset Artifact"
                  else artifact)
            for key, artifact in self.input_artifacts.items()
        }
        if _contains_secret({"source_ids": self.source_ids,
                             "user_question": self.user_question,
                             "analysis_parameters": self.analysis_parameters,
                             "input_artifacts": safe_input_artifacts}):
            raise ValueError("Handoff contains a secret or unresolved location")
        resolved_sources = {key: value.reference for key, value in self.source_values.items() if isinstance(value.reference, Mapping) and value.reference.get("kind") == "CSV"}
        source_values = {key: value for key, value in self.source_values.items() if key not in resolved_sources}
        if (_is_raw_location(safe_input_artifacts)
                or _is_raw_location(self.analysis_parameters)
                or _is_raw_location(source_values)):
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
1. Produce only Steps that can execute with the validated capabilities for this run.
2. Each Step must have a unique step_id (snake_case string).
3. Each Step must name exactly one Agent from the list above.
4. depends_on must reference only step_ids from earlier Steps in the plan (never itself, never a later Step).
5. The Steps must form a valid dependency order — no cycles, no forward references.
6. For the Phase 1 sales comparison, produce exactly a Fetcher Step followed by a dependent Analyst Step.
7. The Fetcher must use a supplied Data Source and declare one Dataset Artifact output. The Analyst must depend on that Fetcher output and declare Findings.
8. Include the exact user_question text in the Plan.
9. Use these Step keys: `step_id`, `agent`, `objective`, `source_ids`, `expected_output`, `declared_artifacts`, and `depends_on`.
10. Never invent a Data Source ID, filesystem location, executable SQL, or arithmetic result."""


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

        if allowed_agents == set(PHASE1_AGENTS) and "sales_csv" in catalog:
            if len(result.steps) != 2 or [step.agent for step in result.steps] != ["Fetcher", "Analyst"]:
                raise ValueError("Phase 1 sales Plan must contain exactly Fetcher then Analyst Steps")
            fetch_step, analyst_step = result.steps
            if fetch_step.source_ids != ["sales_csv"]:
                raise ValueError("Phase 1 Fetcher must use the catalogued sales_csv Data Source")
            if analyst_step.depends_on != [fetch_step.step_id]:
                raise ValueError("Phase 1 Analyst must depend on the Fetcher Step")
            if not (fetch_step.declared_artifacts or [fetch_step.expected_output]):
                raise ValueError("Phase 1 Fetcher must declare a Dataset Artifact output")
            if not (analyst_step.declared_artifacts or [analyst_step.expected_output]):
                raise ValueError("Phase 1 Analyst must declare a Findings output")

        return result


def _extract_question(state: CrewState) -> str:
    messages = state.get("messages", [])
    if messages:
        return messages[0].content
    return ""


def planner_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    question = _extract_question(state)

    try:
        configurable = (config or {}).get("configurable", {})
        planner_model = configurable.get("planner_model")
        try:
            store = get_store()
        except RuntimeError:
            store = None
        agent = PlannerAgent(
            store=None if planner_model is not None else store,
            available_agents=configurable.get("available_agents"),
            data_sources=configurable.get("data_sources", ()),
            llm=planner_model,
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


def fetcher_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    """Execute the first Fetcher Step through its resolved Handoff."""
    plan = state.get("artifacts", {}).get("plan")
    configurable = (config or {}).get("configurable", {})
    catalog = configurable.get("data_sources", {})
    if not isinstance(plan, Plan):
        return {"current_step": "fetcher_node", "artifacts": {"fetcher_error": "No executable Plan"}}
    fetch_step = next((step for step in plan.steps if step.agent == "Fetcher"), None)
    if fetch_step is None:
        return {"current_step": "fetcher_node"}
    try:
        crew = Crew(available_agents=configurable.get("available_agents", PHASE1_AGENTS), data_sources=catalog)
        handoff = crew.resolve_handoff(fetch_step, plan, state.get("artifacts", {}).get("datasets", {}))
        model = configurable.get("fetcher_model") or production_fetcher_model()
        artifact = FetcherAgent(model).run(handoff, artifact_dir=configurable.get("artifact_dir", ".artifacts"))
        reference = artifact.as_reference()
        output_names = fetch_step.declared_artifacts or [fetch_step.expected_output]
        return {
            "current_step": "fetcher_node",
            "steps_completed": [fetch_step.step_id],
            "artifacts": {
                "datasets": {artifact.artifact_id: reference},
                "step_outputs": {name: reference for name in output_names},
                "dataset_profile": artifact.profile,
                "handoffs": {fetch_step.step_id: handoff.model_dump(mode="json")},
                "step_outcomes": {fetch_step.step_id: {
                    "status": "completed",
                    "artifact_ids": [artifact.artifact_id],
                }},
            },
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {"current_step": "fetcher_node", "artifacts": {"fetcher_error": str(exc), "terminal_failure": {"agent": "Fetcher", "required": True, "status": "failed"}}}


def analyst_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    """Execute the Analyst Step from the Fetcher's Dataset Artifact Handoff."""
    plan = state.get("artifacts", {}).get("plan")
    configurable = (config or {}).get("configurable", {})
    if not isinstance(plan, Plan):
        return {"current_step": "analyst_node", "artifacts": {"analyst_error": "No executable Plan"}}
    analyst_step = next((step for step in plan.steps if step.agent == "Analyst"), None)
    if analyst_step is None:
        return {"current_step": "analyst_node"}
    try:
        crew = Crew(
            available_agents=configurable.get("available_agents", PHASE1_AGENTS),
            data_sources=configurable.get("data_sources", {}),
        )
        analysis_parameters = dict(configurable.get("analysis_parameters", {}))
        catalog = configurable.get("data_sources", {})
        target_source = catalog.get("sales_targets_postgres") if isinstance(catalog, Mapping) else None
        as_of_date = analysis_parameters.get("as_of_date")
        if target_source is not None and as_of_date:
            # The optional Postgres target Data Source is only ever
            # retrieved here through the read-only Fetcher Tool, and only
            # for the exact comparable period the core CSV analysis already
            # computes via the shared comparison_window() function — never
            # reimplemented. A failure here never raises; it degrades to a
            # recorded source status the Analyst Tool turns into a Finding.
            from analyst_crew.tools.analysis import comparison_window
            from analyst_crew.tools.targets import retrieve_targets

            window = comparison_window(as_of_date)
            analysis_parameters["target_data"] = retrieve_targets(window.current_start, window.current_end)

        cpi_source = catalog.get("cpi_fred") if isinstance(catalog, Mapping) else None
        if cpi_source is not None and as_of_date:
            # The optional FRED CPI Data Source is only ever retrieved here
            # through the bounded-retry Fetcher Tool, and only for the exact
            # comparable period the core CSV analysis already computes via
            # the shared comparison_window() function — never reimplemented.
            # A failure here never raises; it degrades to a recorded source
            # status the Analyst Tool turns into a Finding and a limitation.
            from analyst_crew.tools.analysis import comparison_window
            from analyst_crew.tools.fred import retrieve_cpi

            cpi_window = comparison_window(as_of_date)
            analysis_parameters["cpi_data"] = retrieve_cpi(cpi_window.prior_start, cpi_window.current_end)
        handoff = crew.resolve_handoff(
            analyst_step,
            plan,
            state.get("artifacts", {}).get("step_outputs", {}),
            analysis_parameters,
        )
        model = configurable.get("analyst_model") or production_analyst_model()
        findings = AnalystAgent(model).run(handoff)
        return {
            "current_step": "analyst_node",
            "steps_completed": [analyst_step.step_id],
            "artifacts": {
                "findings": findings,
                "handoffs": {
                    **state.get("artifacts", {}).get("handoffs", {}),
                    analyst_step.step_id: handoff.model_dump(mode="json"),
                },
                "step_outcomes": {
                    **state.get("artifacts", {}).get("step_outcomes", {}),
                    analyst_step.step_id: {
                        "status": "completed",
                        "artifact_ids": analyst_step.declared_artifacts or [analyst_step.expected_output],
                        "input_artifact_ids": [findings.dataset_artifact_id],
                    },
                },
            },
        }
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "current_step": "analyst_node",
            "artifacts": {
                "analyst_error": str(exc),
                "terminal_failure": {"agent": "Analyst", "required": True, "status": "failed"},
            },
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
    graph.add_node("planner_node", lambda state, config=None: planner_node(state, merged_config(config)))
    graph.add_node("fetcher_node", lambda state, config=None: fetcher_node(state, merged_config(config)))
    graph.add_node("analyst_node", lambda state, config=None: analyst_node(state, merged_config(config)))
    graph.add_edge(START, "planner_node")
    def next_node(state: CrewState) -> str:
        plan = state.get("artifacts", {}).get("plan")
        return "fetcher_node" if isinstance(plan, Plan) and any(step.agent == "Fetcher" for step in plan.steps) else END
    graph.add_conditional_edges("planner_node", next_node, {"fetcher_node": "fetcher_node", END: END})
    def after_fetcher(state: CrewState) -> str:
        if state.get("artifacts", {}).get("terminal_failure"):
            return END
        plan = state.get("artifacts", {}).get("plan")
        return "analyst_node" if isinstance(plan, Plan) and any(step.agent == "Analyst" for step in plan.steps) else END
    graph.add_conditional_edges("fetcher_node", after_fetcher, {"analyst_node": "analyst_node", END: END})
    graph.add_edge("analyst_node", END)
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
        graph = build_graph(
            checkpointer=_checkpointer,
            store=_store,
            runtime_config=_config,
        )
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
