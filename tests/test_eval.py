from analyst_crew.llm import credentials_available as llm_credentials_available
import os
import json

import pytest
from deepeval import assert_test
from deepeval.metrics import BaseMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams, SingleTurnParams
from langchain_core.messages import HumanMessage
from analyst_crew.llm import make_chat_model
from langgraph.store.memory import InMemoryStore

from analyst_crew.agents.planner import PlannerAgent
from analyst_crew.tools.memory import clear_namespace, seed_memory
from analyst_crew.tools.sales import sales_data_source


class OpenRouterEvaluator(DeepEvalBaseLLM):
    def __init__(self, model_name: str = "configured-llm"):
        self._model_name = model_name
        self._model = make_chat_model(temperature=0.0)

    def generate(self, prompt: str) -> str:
        response = self._model.invoke([HumanMessage(content=prompt)])
        return response.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return self._model_name

    def load_model(self):
        return self._model


@pytest.mark.eval
def test_golden_plan_structure():
    """
    DeepEval golden test case for the Planner.

    This test validates the end-to-end Planner by submitting a fixed
    golden question and asserting *structural* properties of the resulting Plan.

    Why structural assertions (not exact text matching):
    LLM outputs are inherently non-deterministic — the same question can
    produce valid Plans with different step_id names, different ordering, or
    different produces descriptions. Exact text matching would be fragile and
    cause false negatives. Structural assertions check that the Plan follows
    the defined schema and contains the required Agent types, which is the
    real contract the Planner must fulfill.

    What this test checks via GEval:
    1. The Plan contains at least 2 Steps.
    2. Every Step's agent field is one of the five known Agent names.
    3. No Step's depends_on references its own step_id.
    4. Steps are ordered so dependencies precede dependents.
    5. At least one Step references Fetcher and at least one references Analyst.
    """
    if not llm_credentials_available():
        pytest.skip("LLM credentials not set")

    store = InMemoryStore()
    clear_namespace(store)
    seed_memory(store, "'unusual' means >15% swings by product category")

    agent = PlannerAgent(store=store, data_sources=sales_data_source("data/sales.csv"))
    question = "Compare this month's sales to last month and flag anything unusual"
    plan = agent.generate_plan(question)

    actual_output = json.dumps(plan.model_dump(), indent=2)

    evaluator = OpenRouterEvaluator()

    metric = GEval(
        name="Plan Structure",
        criteria="\n".join([
            "The plan contains at least 2 steps — check that the 'steps' array has 2 or more items.",
            "Every step's 'agent' field is one of: Fetcher, Analyst, Visualizer, Critic, Reporter — check each step's agent value.",
            "No step's 'depends_on' array references its own 'step_id' — for each step, verify step_id is not in depends_on.",
            "Steps are ordered so dependencies come before dependents — for each step, all step_ids in depends_on must appear earlier in the steps array.",
            "At least one step uses Fetcher agent and at least one uses Analyst agent — verify presence of both agent types in the agent fields.",
        ]),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=evaluator,
        threshold=0.8,
    )

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_output,
    )

    assert_test(test_case, [metric])

class ReportPropertiesMetric(BaseMetric):
    """Deterministic DeepEval metric over a Report's properties (issue #13):
    no stray digits, every limitation and the data-quality disclosure present,
    every Material Change covered, every rendered number in the Evidence
    Catalog. Score is the share of checks that hold; no wording is compared."""

    _required_params = [SingleTurnParams.ACTUAL_OUTPUT]

    def __init__(self, report, catalog, findings, threshold: float = 1.0):
        self.report, self.catalog, self.findings = report, catalog, findings
        self.threshold = threshold
        self.async_mode = False
        self.include_reason = True

    @property
    def __name__(self):
        return "Report Properties"

    def measure(self, test_case, *args, **kwargs) -> float:
        from tests.report_properties import report_property_violations

        violations = report_property_violations(self.report, self.catalog, self.findings)
        checks = 4 + len(self.report.findings) + len(violations)  # coarse denominator; any violation fails the threshold
        self.score = 1.0 if not violations else max(0.0, 1 - len(violations) / checks)
        self.reason = "all Report properties hold" if not violations else "; ".join(violations)
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


@pytest.mark.eval
def test_golden_report_properties(tmp_path):
    """The live Reporter's Report satisfies the ADR-0002 properties for the documented scenario."""
    if not llm_credentials_available():
        pytest.skip("LLM credentials not set")
    if not os.path.exists("data/sales.csv"):
        pytest.skip("run scripts/generate_sales_data.py first")

    from analyst_crew.agents import invoke
    from analyst_crew.reporting.report import render_report
    from analyst_crew.tools.fred import fred_cpi_data_source
    from analyst_crew.tools.targets import sales_targets_data_source

    question = "Compare this month's sales to last month and flag anything unusual"
    result = invoke(question, {"configurable": {
        "data_sources": {**sales_data_source("data/sales.csv"), **sales_targets_data_source(), **fred_cpi_data_source()},
        "artifact_dir": str(tmp_path / "artifacts"),
        "analysis_parameters": {"as_of_date": "2025-06-30"},
    }})
    artifacts = result["artifacts"]
    if artifacts.get("plan_error"):
        pytest.skip(f"Planner API error: {artifacts['plan_error']}")
    assert artifacts["report_mode"] == "llm", artifacts.get("terminal_failure")

    rendered = render_report(artifacts["report"], artifacts["evidence_catalog"])
    metric = ReportPropertiesMetric(artifacts["report"], artifacts["evidence_catalog"], artifacts["findings"])
    assert_test(LLMTestCase(input=question, actual_output=rendered), [metric])
