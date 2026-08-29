import hashlib
import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
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
from analyst_crew.tools.analysis import AnalystAgent, BaselineFindings, production_analyst_model
from analyst_crew.tools.retry import invoke_with_optional_config
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


def _step_contract_fingerprint(step: Step) -> dict[str, Any]:
    """The declarative parts of a Step contract that define what work it
    does, independent of any runtime data. Changing any of these means the
    Step no longer means the same thing, so any previously computed output
    for it must be invalidated."""
    return {
        "step_id": step.step_id,
        "agent": step.agent,
        "objective": step.objective,
        "source_ids": sorted(step.source_ids),
        "expected_output": step.expected_output,
        "declared_artifacts": sorted(step.declared_artifacts or [step.expected_output]),
        "depends_on": sorted(step.depends_on),
    }


def execution_key(
    step: Step,
    *,
    analysis_parameters: Mapping[str, Any] | None = None,
    input_artifact_hashes: Sequence[str] = (),
) -> str:
    """Stable content hash identifying one Step's resolved execution (issue #11).

    Composed from exactly three things, matching the acceptance criteria:

    1. The Step's own contract (``_step_contract_fingerprint``) — step_id,
       agent, objective, declared source_ids/artifacts, and dependencies.
       A Planner-authored change to what a Step is supposed to do (a new
       objective, a different declared source, a changed dependency) always
       changes the key.
    2. The resolved ``analysis_parameters`` for this Handoff. The Fetcher
       Step never receives ``analysis_parameters`` in its key (callers pass
       ``{}``), so a parameter-only change (e.g. a different ``as_of_date``)
       leaves the Fetcher's key unchanged — its Dataset Artifact is reused —
       while the Analyst's key (which does receive them) changes and it
       recomputes.
    3. The content hash(es) of the Step's input artifacts. For the Fetcher
       this is the source CSV's own byte hash, computed cheaply before any
       parse/validate work; for the Analyst this is the Fetcher's output
       Dataset Artifact content hash. Because the Analyst's key transitively
       depends on the Fetcher's output hash, a Fetcher recompute (triggered
       by changed CSV content) automatically changes the Analyst's key too —
       cascading invalidation falls out of this composition; it is never a
       separately maintained "invalidate downstream" list.

    Two Steps with identical contract, parameters, and input hashes always
    hash to the same key and are safe to treat as interchangeable on resume;
    a difference in any one of the three changes the key and forces a
    recompute.
    """
    payload = {
        "step": _step_contract_fingerprint(step),
        "analysis_parameters": dict(analysis_parameters or {}),
        "input_artifact_hashes": sorted(input_artifact_hashes),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_content_hash(reference: Any) -> str | None:
    """Cheap identity hash for a catalogued CSV Data Source, computed before
    any expensive parse/validate work. This is the same byte hash the
    Fetcher Tool itself records as ``DatasetArtifact.input_hash`` once a
    fetch actually runs (see ``analyst_crew.tools.sales.fetch_csv``), so a
    match here guarantees the Fetcher would produce byte-identical output."""
    if not isinstance(reference, Mapping) or reference.get("kind") != "CSV":
        return None
    location = reference.get("location")
    if not location:
        return None
    path = Path(str(location))
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resume_decision(state: CrewState, step_id: str, current_key: str) -> tuple[str, str]:
    """Decide whether a Step's previously checkpointed output is still valid
    to reuse for this thread (issue #11).

    Returns ``(resume_outcome, resume_reason)``:

    - ``resume_outcome`` is ``"reused"`` (skip recomputation, reuse the
      stored output) or ``"recomputed"`` (run the Step's work again).
    - ``resume_reason`` explains why, as one of five distinct, inspectable
      labels:
        - ``"unchanged"``: a completed prior outcome exists for this Step
          and its execution key still matches this run's -> reused.
        - ``"invalidated"``: a completed prior outcome exists but its
          execution key no longer matches (a changed Step contract,
          analysis parameter, or input artifact content) -> recomputed.
        - ``"failed"``: the prior run recorded a terminal failure for this
          exact Step -> recomputed.
        - ``"interrupted"``: this thread has been invoked before (its
          accumulated message history holds more than the single message
          this invoke is adding) but this Step has neither a completed
          outcome nor a recorded failure — the previous invoke must have
          been interrupted before this Step's node returned, and therefore
          before anything about that attempt could be checkpointed ->
          recomputed from a clean Handoff (LangGraph's checkpointer already
          restarts at the last node boundary that *did* complete; nothing
          about a genuinely interrupted node survives to resume from).
        - ``"never_ran"``: this is the first invoke on this thread ->
          recomputed.
    """
    artifacts = state.get("artifacts", {})
    prior_outcome = artifacts.get("step_outcomes", {}).get(step_id)
    prior_failure = artifacts.get("terminal_failure")
    had_prior_invoke = len(state.get("messages", [])) > 1

    if isinstance(prior_failure, Mapping) and prior_failure.get("step_id") == step_id:
        return "recomputed", "failed"
    if isinstance(prior_outcome, Mapping) and prior_outcome.get("status") == "completed":
        if prior_outcome.get("execution_key") == current_key:
            return "reused", "unchanged"
        return "recomputed", "invalidated"
    if had_prior_invoke:
        return "recomputed", "interrupted"
    return "recomputed", "never_ran"


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
                      data_sources: Mapping[str, Any] | Sequence[str] | None = None,
                      config: dict | None = None) -> Plan:
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
            result = invoke_with_optional_config(self._chain, messages, config)
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
        plan = agent.generate_plan(question, config=config)
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
    """Execute the first Fetcher Step through its resolved Handoff.

    Before doing any expensive work (reading/parsing/validating the CSV and
    writing a Parquet Dataset Artifact), this resolves the Handoff, computes
    the source CSV's cheap content hash, and derives this Step's execution
    key (see ``execution_key``). ``_resume_decision`` compares that key
    against the one recorded the last time this thread completed this Step:
    a match reuses the previously produced Dataset Artifact reference
    untouched (no re-read, no re-parse); a mismatch — or no valid prior
    completion — recomputes it. The resume outcome/reason are always
    recorded on ``step_outcomes`` for CrewState inspection.
    """
    plan = state.get("artifacts", {}).get("plan")
    configurable = (config or {}).get("configurable", {})
    catalog = configurable.get("data_sources", {})
    if not isinstance(plan, Plan):
        return {"current_step": "fetcher_node", "artifacts": {"fetcher_error": "No executable Plan"}}
    fetch_step = next((step for step in plan.steps if step.agent == "Fetcher"), None)
    if fetch_step is None:
        return {"current_step": "fetcher_node"}
    agent = None
    try:
        crew = Crew(available_agents=configurable.get("available_agents", PHASE1_AGENTS), data_sources=catalog)
        handoff = crew.resolve_handoff(fetch_step, plan, state.get("artifacts", {}).get("datasets", {}))

        source_hash = None
        if handoff.source_values:
            source_reference = next(iter(handoff.source_values.values())).reference
            source_hash = _source_content_hash(source_reference)
        current_key = execution_key(
            fetch_step,
            analysis_parameters={},
            input_artifact_hashes=[source_hash] if source_hash else [],
        )
        resume_outcome, resume_reason = _resume_decision(state, fetch_step.step_id, current_key)

        reference = None
        prior_outcome = None
        if resume_outcome == "reused":
            prior_outcome = state.get("artifacts", {}).get("step_outcomes", {}).get(fetch_step.step_id, {})
            prior_artifact_id = next(iter(prior_outcome.get("artifact_ids", [])), None)
            reference = (
                state.get("artifacts", {}).get("datasets", {}).get(prior_artifact_id)
                if prior_artifact_id else None
            )
            if reference is None:
                # Defensive fallback: the prior outcome claims a key match
                # but its Dataset Artifact reference is missing from state —
                # never trust a reuse we can't actually produce output for.
                resume_outcome, resume_reason = "recomputed", "invalidated"

        output_names = fetch_step.declared_artifacts or [fetch_step.expected_output]

        if resume_outcome == "reused":
            retry_counts = prior_outcome.get("retry_counts") or {"validation_retries": 0, "tool_retries": 0}
        else:
            model = configurable.get("fetcher_model") or production_fetcher_model()
            agent = FetcherAgent(model)
            artifact = agent.run(handoff, artifact_dir=configurable.get("artifact_dir", ".artifacts"), config=config)
            reference = artifact.as_reference()
            retry_counts = {"validation_retries": agent.validation_retries, "tool_retries": agent.tool_retries}

        return {
            "current_step": "fetcher_node",
            "steps_completed": [fetch_step.step_id],
            "artifacts": {
                "terminal_failure": None,
                "datasets": {reference["artifact_id"]: reference},
                "step_outputs": {name: reference for name in output_names},
                "dataset_profile": reference.get("profile", {}),
                "handoffs": {
                    **state.get("artifacts", {}).get("handoffs", {}),
                    fetch_step.step_id: handoff.model_dump(mode="json"),
                },
                # merge_dicts is a shallow update of "artifacts" — it would
                # otherwise replace the whole step_outcomes/retry_counts
                # sub-dicts wholesale, wiping the Analyst's outcome recorded
                # by a previous invoke on this thread before analyst_node
                # even runs this turn. Spread the prior entries explicitly,
                # matching analyst_node's own merge below.
                "step_outcomes": {
                    **state.get("artifacts", {}).get("step_outcomes", {}),
                    fetch_step.step_id: {
                        "status": "completed",
                        "artifact_ids": [reference["artifact_id"]],
                        "retry_counts": retry_counts,
                        "execution_key": current_key,
                        "resume_outcome": resume_outcome,
                        "resume_reason": resume_reason,
                    },
                },
                "retry_counts": {
                    **state.get("artifacts", {}).get("retry_counts", {}),
                    fetch_step.step_id: retry_counts,
                },
            },
        }
    except (OSError, RuntimeError, ValueError) as exc:
        retry_counts = {
            "validation_retries": getattr(agent, "validation_retries", 0),
            "tool_retries": getattr(agent, "tool_retries", 0),
        }
        return {
            "current_step": "fetcher_node",
            "artifacts": {
                "fetcher_error": str(exc),
                "retry_counts": {
                    **state.get("artifacts", {}).get("retry_counts", {}),
                    fetch_step.step_id: retry_counts,
                },
                "terminal_failure": {
                    "agent": "Fetcher",
                    "step_id": fetch_step.step_id,
                    "required": True,
                    "status": "failed",
                    "error": str(exc),
                    "retry_counts": retry_counts,
                },
            },
        }


def analyst_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    """Execute the Analyst Step from the Fetcher's Dataset Artifact Handoff.

    Resolves the optional Postgres target / FRED CPI enrichments, resolves
    the Handoff, then derives this Step's execution key from the Step
    contract, the resolved ``analysis_parameters``, and the input Dataset
    Artifact's content hash (see ``execution_key``). Because that content
    hash comes straight from the Fetcher's just-resolved output, a Fetcher
    recompute (changed CSV content) naturally changes the Analyst's key too
    — cascading invalidation, with no separate downstream-invalidation list.
    A parameter-only change (e.g. a different ``as_of_date``) changes only
    the Analyst's key, so the Fetcher's Dataset Artifact is reused untouched
    while Findings recompute. ``_resume_decision`` compares the key against
    the one recorded the last time this thread completed this Step; a match
    reuses the previously produced Findings untouched.
    """
    plan = state.get("artifacts", {}).get("plan")
    configurable = (config or {}).get("configurable", {})
    if not isinstance(plan, Plan):
        return {"current_step": "analyst_node", "artifacts": {"analyst_error": "No executable Plan"}}
    analyst_step = next((step for step in plan.steps if step.agent == "Analyst"), None)
    if analyst_step is None:
        return {"current_step": "analyst_node"}
    agent = None
    try:
        crew = Crew(
            available_agents=configurable.get("available_agents", PHASE1_AGENTS),
            data_sources=configurable.get("data_sources", {}),
        )
        analysis_parameters = dict(configurable.get("analysis_parameters", {}))
        catalog = configurable.get("data_sources", {})
        as_of_date = analysis_parameters.get("as_of_date")

        # Reused external fetches: if the last Handoff this thread resolved
        # for this Step used the same as_of_date (the sole driver of the
        # comparable-period window these optional sources are queried for),
        # its already-retrieved target/CPI result is byte-identical to what
        # a fresh retrieval would return, so it is reused directly instead
        # of re-issuing the Postgres query / FRED HTTP call.
        prior_handoff = state.get("artifacts", {}).get("handoffs", {}).get(analyst_step.step_id)
        prior_parameters = prior_handoff.get("analysis_parameters", {}) if isinstance(prior_handoff, Mapping) else {}

        def _reused_optional_source(key: str) -> Mapping[str, Any] | None:
            if not as_of_date or prior_parameters.get("as_of_date") != as_of_date:
                return None
            cached = prior_parameters.get(key)
            return cached if isinstance(cached, Mapping) else None

        target_source = catalog.get("sales_targets_postgres") if isinstance(catalog, Mapping) else None
        if target_source is not None and as_of_date:
            # The optional Postgres target Data Source is only ever
            # retrieved here through the read-only Fetcher Tool, and only
            # for the exact comparable period the core CSV analysis already
            # computes via the shared comparison_window() function — never
            # reimplemented. A failure here never raises; it degrades to a
            # recorded source status the Analyst Tool turns into a Finding.
            cached_target_data = _reused_optional_source("target_data")
            if cached_target_data is not None:
                analysis_parameters["target_data"] = cached_target_data
            else:
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
            cached_cpi_data = _reused_optional_source("cpi_data")
            if cached_cpi_data is not None:
                analysis_parameters["cpi_data"] = cached_cpi_data
            else:
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

        dataset_references = {
            key: value
            for key, value in handoff.input_artifacts.items()
            if isinstance(value, dict) and value.get("kind") == "Dataset Artifact"
        }
        input_hashes = sorted({
            str(value["content_hash"]) for value in dataset_references.values() if value.get("content_hash")
        })
        current_key = execution_key(
            analyst_step,
            analysis_parameters=analysis_parameters,
            input_artifact_hashes=input_hashes,
        )
        resume_outcome, resume_reason = _resume_decision(state, analyst_step.step_id, current_key)

        prior_outcome = None
        findings = None
        if resume_outcome == "reused":
            prior_outcome = state.get("artifacts", {}).get("step_outcomes", {}).get(analyst_step.step_id, {})
            reused_findings = state.get("artifacts", {}).get("findings")
            if reused_findings is None:
                # Defensive fallback: a completed prior outcome claims a key
                # match, but no Findings are actually on state — never trust
                # a reuse we can't actually produce output for.
                resume_outcome, resume_reason = "recomputed", "invalidated"
            else:
                findings = (
                    reused_findings if isinstance(reused_findings, BaselineFindings)
                    else BaselineFindings.model_validate(reused_findings)
                )

        if resume_outcome == "reused":
            retry_counts = prior_outcome.get("retry_counts") or {"validation_retries": 0, "tool_retries": 0}
        else:
            model = configurable.get("analyst_model") or production_analyst_model()
            agent = AnalystAgent(model)
            findings = agent.run(handoff, config=config)
            retry_counts = {"validation_retries": agent.validation_retries, "tool_retries": agent.tool_retries}

        return {
            "current_step": "analyst_node",
            "steps_completed": [analyst_step.step_id],
            "artifacts": {
                "terminal_failure": None,
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
                        "retry_counts": retry_counts,
                        "execution_key": current_key,
                        "resume_outcome": resume_outcome,
                        "resume_reason": resume_reason,
                    },
                },
                "retry_counts": {
                    **state.get("artifacts", {}).get("retry_counts", {}),
                    analyst_step.step_id: retry_counts,
                },
            },
        }
    except (OSError, RuntimeError, ValueError) as exc:
        retry_counts = {
            "validation_retries": getattr(agent, "validation_retries", 0),
            "tool_retries": getattr(agent, "tool_retries", 0),
        }
        return {
            "current_step": "analyst_node",
            "artifacts": {
                "analyst_error": str(exc),
                "retry_counts": {
                    **state.get("artifacts", {}).get("retry_counts", {}),
                    analyst_step.step_id: retry_counts,
                },
                "terminal_failure": {
                    "agent": "Analyst",
                    "step_id": analyst_step.step_id,
                    "required": True,
                    "status": "failed",
                    "error": str(exc),
                    "retry_counts": retry_counts,
                },
            },
        }


def _ensure_langchain_callback_compat() -> None:
    """Make the pinned Langfuse v2 SDK's LangChain integration importable
    against this project's langchain 1.x.

    ``langfuse.callback`` (the v2 SDK's ``CallbackHandler``) imports
    ``langchain.callbacks.base.BaseCallbackHandler`` and
    ``langchain.schema.{agent,document}`` — legacy import paths that
    langchain>=1.0 no longer exposes (those symbols now live only in
    ``langchain_core``, which this project already depends on directly).
    This registers thin ``sys.modules`` shims for the missing legacy paths,
    pointing them at their ``langchain_core`` equivalents, before
    ``langfuse.callback`` is imported. It only patches paths that are
    actually missing, so it is a no-op against any langchain version that
    still exposes them.
    """
    import sys
    import types

    try:
        import langchain.callbacks.base  # noqa: F401
    except ImportError:
        from langchain_core.callbacks.base import BaseCallbackHandler

        callbacks_pkg = types.ModuleType("langchain.callbacks")
        callbacks_base = types.ModuleType("langchain.callbacks.base")
        callbacks_base.BaseCallbackHandler = BaseCallbackHandler
        callbacks_pkg.base = callbacks_base
        sys.modules["langchain.callbacks"] = callbacks_pkg
        sys.modules["langchain.callbacks.base"] = callbacks_base

    try:
        import langchain.schema.agent  # noqa: F401
        have_schema_agent = True
    except ImportError:
        have_schema_agent = False
    try:
        import langchain.schema.document  # noqa: F401
        have_schema_document = True
    except ImportError:
        have_schema_document = False

    if not (have_schema_agent and have_schema_document):
        schema_pkg = sys.modules.get("langchain.schema") or types.ModuleType("langchain.schema")
        if not have_schema_agent:
            from langchain_core.agents import AgentAction, AgentFinish

            schema_agent = types.ModuleType("langchain.schema.agent")
            schema_agent.AgentAction = AgentAction
            schema_agent.AgentFinish = AgentFinish
            schema_pkg.agent = schema_agent
            sys.modules["langchain.schema.agent"] = schema_agent
        if not have_schema_document:
            from langchain_core.documents import Document

            schema_document = types.ModuleType("langchain.schema.document")
            schema_document.Document = Document
            schema_pkg.document = schema_document
            sys.modules["langchain.schema.document"] = schema_document
        sys.modules["langchain.schema"] = schema_pkg


def get_langfuse_handler(
    session_id: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
) -> tuple[Any, Any, Any]:
    """Build one Langfuse v2 trace and a ``CallbackHandler`` bound to it for
    this Crew run.

    The handler is shared across the whole ``graph.invoke()`` call (threaded
    through every node's ``config``), so every Planner/Fetcher/Analyst LLM
    call and Tool invocation becomes a nested observation on this same
    trace, instead of each ``invoke()`` producing its own flat, unlinked
    trace record.

    Returns ``(client, trace, handler)`` — all ``None`` when Langfuse isn't
    configured or the SDK/compat shim fails; observability must never make a
    successful Crew run fail.
    """
    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not all((host, public_key, secret_key)):
        return None, None, None

    try:
        _ensure_langchain_callback_compat()
        from langfuse.callback import CallbackHandler

        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        trace = client.trace(
            name="crew_run",
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
        )
        handler = CallbackHandler(stateful_client=trace)
        return client, trace, handler
    except Exception:
        # Observability must never make a successful Crew run fail.
        return None, None, None


def _finish_langfuse_trace(
    client: Any,
    trace: Any,
    metadata: dict[str, Any] | None = None,
    *,
    wait_for_index: bool = True,
) -> None:
    """Attach this run's retry counts, terminal-failure outcome, and source
    status to the trace ``get_langfuse_handler`` already created and
    threaded through the graph run (see ``_run_metadata``), then flush and —
    for callers that immediately query the API afterward — wait briefly for
    Langfuse v2's async ingestion to index it.
    """
    if client is None or trace is None:
        return
    try:
        trace.update(metadata=metadata or {})
        client.flush()
        if not wait_for_index:
            return
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if client.api.trace.get(trace.id) is not None:
                    return
            except Exception:
                pass
            time.sleep(0.5)
    except Exception:
        # Observability must never make a successful Crew run fail.
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


def _run_metadata(result: CrewState) -> dict[str, Any]:
    """Retry counts, terminal-outcome, Step resume outcomes, and optional-
    source status summary for tracing metadata — populated from the same
    ``artifacts`` the CLI and any other caller can already inspect after
    ``invoke()`` returns, and attached to the one Langfuse trace this run's
    ``CallbackHandler`` already linked Planner/Fetcher/Analyst spans to."""
    artifacts = result.get("artifacts", {})
    metadata: dict[str, Any] = {}
    retry_counts = artifacts.get("retry_counts")
    if retry_counts:
        metadata["retry_counts"] = retry_counts
    terminal_failure = artifacts.get("terminal_failure")
    if terminal_failure:
        metadata["terminal_failure"] = terminal_failure
    step_outcomes = artifacts.get("step_outcomes")
    if step_outcomes:
        metadata["step_outcomes"] = {
            step_id: {
                "status": outcome.get("status"),
                "resume_outcome": outcome.get("resume_outcome"),
                "resume_reason": outcome.get("resume_reason"),
            }
            for step_id, outcome in step_outcomes.items()
        }
    findings = artifacts.get("findings")
    if findings is not None:
        findings = findings if isinstance(findings, BaselineFindings) else BaselineFindings.model_validate(findings)
        source_status: dict[str, Any] = {}
        if findings.target_comparison is not None:
            source_status["sales_targets_postgres"] = findings.target_comparison.source_status.model_dump()
        if findings.cpi_adjustment is not None:
            source_status["cpi_fred"] = findings.cpi_adjustment.source_status.model_dump()
        if source_status:
            metadata["source_status"] = source_status
    return metadata


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
            return _do_invoke(state, config, checkpointer, store if store is not None else InMemoryStore())
        if conn_string:
            with (
                PostgresSaver.from_conn_string(conn_string) as pg_checkpointer,
                PostgresStore.from_conn_string(conn_string) as pg_store,
            ):
                pg_checkpointer.setup()
                pg_store.setup()
                return _do_invoke(state, config, pg_checkpointer, pg_store)
        return _do_invoke(state, config, None, store if store is not None else InMemoryStore())

    result = _run()
    _finish_langfuse_trace(langfuse_client, langfuse_trace, metadata=_run_metadata(result))

    return result
