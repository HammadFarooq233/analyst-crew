from langchain_core.messages import HumanMessage

from analyst_crew.agents import Plan, invoke
from analyst_crew.state import CrewState


class TestStateShape:
    def test_state_shape(self):
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