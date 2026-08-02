from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, model_validator

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


def _extract_question(state: CrewState) -> str:
    messages = state.get("messages", [])
    if messages:
        return messages[0].content
    return ""


def planner_node(state: CrewState) -> dict[str, Any]:
    question = _extract_question(state)

    plan = Plan(
        steps=[
            Step(
                step_id="fetch_data",
                agent="Fetcher",
                produces="Raw sales data for this month and last month",
            ),
            Step(
                step_id="analyze_trends",
                agent="Analyst",
                produces="Comparative statistics between this month and last month sales",
                depends_on=["fetch_data"],
            ),
            Step(
                step_id="write_report",
                agent="Reporter",
                produces="Narrative report summarizing the sales comparison",
                depends_on=["analyze_trends"],
            ),
        ],
        user_question=question,
    )

    return {
        "current_step": "planner_node",
        "artifacts": {"plan": plan},
    }


def build_graph() -> StateGraph:
    graph = StateGraph(CrewState)
    graph.add_node("planner_node", planner_node)
    graph.add_edge(START, "planner_node")
    graph.add_edge("planner_node", END)
    return graph


def invoke(user_question: str) -> CrewState:
    from langchain_core.messages import HumanMessage

    graph = build_graph().compile()
    result = graph.invoke({"messages": [HumanMessage(content=user_question)]})
    return result