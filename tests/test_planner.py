import os

import pytest
from langchain_core.messages import HumanMessage

from analyst_crew.agents import Plan, invoke
from analyst_crew.state import CrewState


class TestStateShape:
    def test_state_shape(self):
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        result = invoke("Compare this month's sales to last month")

        assert "current_step" in result
        assert "steps_completed" in result
        assert "artifacts" in result
        assert "messages" in result

        assert isinstance(result["current_step"], str)
        assert result["current_step"] == "planner_node"
        assert isinstance(result["steps_completed"], list)
        assert isinstance(result["artifacts"], dict)
        assert isinstance(result["messages"], list)
        assert len(result["messages"]) >= 1
        assert isinstance(result["messages"][0], HumanMessage)

        plan = result["artifacts"].get("plan")
        assert plan is not None
        validated = Plan.model_validate(plan.model_dump())
        assert len(validated.steps) >= 2

        step_ids = {s.step_id for s in validated.steps}
        assert len(step_ids) == len(validated.steps)
        for step in validated.steps:
            assert step.step_id not in step.depends_on
            assert step.depends_on is None or set(step.depends_on).issubset(step_ids)


@pytest.mark.llm
class TestLLMIntegration:
    def test_llm_plan_valid(self):
        if not os.getenv("GEMINI_API_KEY"):
            pytest.skip("GEMINI_API_KEY not set")

        result = invoke("Compare revenue trends across Q1 and Q2")

        plan_error = result.get("artifacts", {}).get("plan_error")
        if plan_error is not None:
            pytest.skip(f"Skipping LLM test — API error: {plan_error}")

        plan = result["artifacts"]["plan"]

        validated = Plan.model_validate(plan.model_dump())
        assert len(validated.steps) >= 1
        assert validated.user_question == "Compare revenue trends across Q1 and Q2"

        valid_agents = {"Fetcher", "Analyst", "Visualizer", "Critic", "Reporter"}
        step_ids = set()
        for step in validated.steps:
            assert step.step_id, "step_id must not be empty"
            assert step.step_id not in step_ids, f"Duplicate step_id: {step.step_id}"
            step_ids.add(step.step_id)

            assert step.agent in valid_agents, f"Invalid agent: {step.agent}"
            assert step.produces, "produces must not be empty"
            assert step.step_id not in step.depends_on, "Step depends on itself"
            for dep in step.depends_on:
                assert dep in step_ids, f"Step depends on future/unknown step: {dep}"