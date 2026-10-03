"""Plan, Step, Handoff and Crew capability contracts (ADR-0001)."""
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator


AGENTS = ("Fetcher", "Analyst", "Visualizer", "Critic", "Reporter")
PHASE1_AGENTS = ("Fetcher", "Analyst")
PHASE2_AGENTS = ("Fetcher", "Analyst", "Visualizer", "Reporter", "Critic")
# The Agents the Crew can currently execute; the Planner may only assign Steps to these.
CREW_AGENTS = PHASE2_AGENTS


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

    def __init__(self, available_agents: Sequence[str] = CREW_AGENTS,
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
