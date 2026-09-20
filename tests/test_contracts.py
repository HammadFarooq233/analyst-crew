import pytest
from pydantic import ValidationError

from analyst_crew.agents import ArtifactBinding, Crew, Handoff, Plan, PlannerAgent, SourceReference, Step, resolve_handoff


class ScriptedModel:
    def __init__(self, plan):
        self.plan = plan
        self.messages = None

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _messages):
        self.messages = _messages
        return self.plan


def test_handoff_resolves_only_catalogued_sources_and_prior_artifacts():
    fetch = Step(step_id="fetch", agent="Fetcher", objective="fetch sales", source_ids=["sales"], expected_output="sales artifact")
    analyze = Step(step_id="analyze", agent="Analyst", objective="compare sales", expected_output="findings", declared_artifacts=["findings", "chart"], depends_on=["fetch"])
    plan = Plan(steps=[fetch, analyze], user_question="Compare sales")

    handoff = resolve_handoff(analyze, plan, {"sales artifact": "artifact-ref", "chart": "chart-ref"}, {"sales": {"kind": "csv"}}, {"period": "monthly"})
    assert handoff.input_artifacts == {"sales artifact": "artifact-ref", "chart": "chart-ref"}
    assert handoff.source_ids == []
    assert handoff.declared_artifacts["findings"].status == "declared"
    assert handoff.declared_artifacts["chart"].reference == "chart-ref"
    assert handoff.input_artifacts["chart"] == "chart-ref"

    fetch_handoff = resolve_handoff(fetch, plan, {}, {"sales": {"kind": "csv"}})
    assert fetch_handoff.source_ids == ["sales"]
    assert fetch_handoff.source_values["sales"].source_id == "sales"
    assert fetch_handoff.source_values["sales"].reference == {"kind": "csv"}
    assert fetch_handoff.declared_artifacts["sales artifact"].status == "declared"
    assert fetch_handoff.declared_artifacts["sales artifact"].reference == {
        "kind": "planned_output", "producer_step_id": "fetch", "artifact_id": "sales artifact"
    }
    sequence_handoff = resolve_handoff(fetch, plan, {}, ["sales"])
    assert sequence_handoff.source_values["sales"].reference == {
        "kind": "catalog_reference", "catalog": "data_sources", "source_id": "sales"
    }
    assert handoff.user_question == "Compare sales"
    assert handoff.analysis_parameters == {"period": "monthly"}

    with pytest.raises(ValueError, match="unknown Data Sources"):
        resolve_handoff(fetch, plan, {}, {})

    equivalent = Step.model_validate(analyze.model_dump())
    equivalent_handoff = resolve_handoff(equivalent, plan, {"sales artifact": "artifact-ref"})
    assert equivalent_handoff.step == analyze


def test_contracts_reject_forward_dependencies_and_secrets():
    later = Step(step_id="later", agent="Analyst", objective="analyze", expected_output="findings")
    with pytest.raises(ValidationError, match="appears after"):
        Plan(steps=[Step(step_id="first", agent="Fetcher", objective="fetch", expected_output="x", depends_on=["later"]), later], user_question="q")

    with pytest.raises(ValidationError, match="secret"):
        Step(step_id="safe_id", agent="Fetcher", objective="read password=secret", expected_output="x")
    with pytest.raises(ValidationError, match="secret"):
        Plan(steps=[later], user_question="token=hidden")
    with pytest.raises(ValidationError, match="secret"):
        Plan(steps=[later], user_question="source_location=/tmp/data.csv")

    with pytest.raises(ValidationError, match="location"):
        Handoff(step=later, input_artifacts={"note": "/tmp/reference"}, user_question="q")
    with pytest.raises(ValidationError, match="location"):
        Handoff(step=later, input_artifacts={"source_location": "/tmp/data.csv"}, user_question="q")
    unsafe_handoffs = [
        {"user_question": "token=hidden"},
        {"source_ids": ["api_key=hidden"], "user_question": "q"},
        {"declared_artifacts": {"secret=hidden": "ref"}, "user_question": "q"},
    ]
    for values in unsafe_handoffs:
        with pytest.raises(ValidationError, match="secret"):
            Handoff(step=later, **values)

    # Slash-prefixed and file-URI values are allowed unless they are explicitly
    # marked as source locations.
    assert Handoff(step=later, user_question="Use file:///tmp/example").user_question.startswith("Use")
    with pytest.raises(ValidationError, match="location"):
        Handoff(step=later, user_question="location=/tmp/data.csv")
    with pytest.raises(ValidationError, match="location"):
        Handoff(step=later, analysis_parameters={"path": "/tmp/data.csv"}, user_question="q")


def test_scripted_planner_respects_capabilities_and_crew_api():
    plan = Plan(steps=[Step(step_id="fetch", agent="Fetcher", objective="fetch", expected_output="artifact")], user_question="q")
    model = ScriptedModel(plan)
    planner = PlannerAgent(llm=model, available_agents=["Fetcher"])
    assert planner.generate_plan("q").steps[0].agent == "Fetcher"
    assert "MUST use only these Agents" in model.messages[0].content
    assert "Agents: Fetcher" in model.messages[0].content
    default_planner = PlannerAgent(llm=ScriptedModel(plan))
    assert default_planner.available_agents == {"Fetcher", "Analyst", "Reporter"}

    with pytest.raises(ValueError, match="unavailable Agents"):
        PlannerAgent(llm=ScriptedModel(plan), available_agents=["Analyst"]).generate_plan("q")

    with pytest.raises(ValueError, match="available_agents"):
        PlannerAgent(llm=ScriptedModel(plan), available_agents=[]).generate_plan("q")
    with pytest.raises(ValueError, match="Unknown available Agents"):
        PlannerAgent(llm=ScriptedModel(plan), available_agents=["NotAnAgent"]).generate_plan("q")

    crew = Crew(available_agents=["Fetcher"], data_sources=[])
    assert crew.resolve_handoff(plan.steps[0], plan, {}).user_question == "q"


def test_contract_values_reject_secret_and_raw_locations_but_prose_is_allowed():
    step = Step(step_id="analyze", agent="Analyst", objective="analyze", expected_output="findings")
    with pytest.raises(ValidationError):
        ArtifactBinding(artifact_id="secret=hidden", reference="ref")
    with pytest.raises(ValidationError):
        ArtifactBinding(artifact_id="findings", reference={"token": "token=hidden"})
    with pytest.raises(ValidationError):
        SourceReference(source_id="sales", reference={"location": "/tmp/sales.csv"})
    with pytest.raises(ValidationError):
        Handoff(step=step, analysis_parameters={"nested": {"uri": "file:///tmp/input"}}, user_question="q")
    assert Plan(steps=[step], user_question="Explain the file:///tmp/example reference")
