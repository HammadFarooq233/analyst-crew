"""Issue #18 — the Critic Step (flag only). Primary seam: the Crew graph via ``invoke()`` with an in-memory
checkpointer, a CSV in a temporary directory and scripted per-Agent models. Second seam: pure Critique
validation. Asserts outcomes, the stored Critique, Reviewer Notes and resume labels; never model wording."""
import logging

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from analyst_crew.agents import Step
from analyst_crew.agents.planner import PlannerAgent
from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.reporting.critique import Critique, CritiqueIssue, validate_critique
from analyst_crew.reporting.report import Report, render_report
from analyst_crew.state.serde import crew_serde, crew_state_types
from analyst_crew.tools.sales import sales_data_source
from tests.crew_harness import (
    FULL_AGENTS, QUESTION, ScriptedReporterModel, base_rows, crew_plan, run_crew, sales_row, valid_content, write_sales,
)

ALL_AGENTS = (*FULL_AGENTS, "Critic")


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", base_rows())


def critic_plan(*, critic_depends_on=("write_report",)):
    plan = crew_plan(visualizer=True)
    plan.steps.append(Step(
        step_id="review_report", agent="Critic", objective="Review the draft Report for judgment errors",
        expected_output="report_critique", declared_artifacts=["report_critique"], depends_on=list(critic_depends_on),
    ))
    return plan


class ScriptedCriticModel:
    """One response per call (the last repeats): a ``Critique``, a dict, an Exception to raise, or a
    callable taking the Critic's context."""

    def __init__(self, *responses):
        self.responses = list(responses) or [Critique(issues=[])]
        self.contexts = []

    @property
    def calls(self):
        return len(self.contexts)

    def invoke(self, context):
        self.contexts.append(context)
        response = self.responses[min(len(self.contexts), len(self.responses)) - 1]
        if callable(response):
            response = response(context)
        if isinstance(response, BaseException):
            raise response
        return response


def issue(target="claim_1", category="causal_overreach", rationale="The wording implies a cause the evidence does not show."):
    return CritiqueIssue(category=category, target=target, rationale=rationale)


def run_with_critic(source, tmp_path, monkeypatch, critic=None, **kwargs):
    critic = critic if critic is not None else ScriptedCriticModel()
    kwargs.setdefault("plan", critic_plan())
    result, _ = run_crew(source, tmp_path, monkeypatch, available_agents=ALL_AGENTS, critic_model=critic, **kwargs)
    return result, critic


def outcome(result, step_id="review_report"):
    return result["artifacts"]["step_outcomes"][step_id]


class TestCritiqueValidation:
    def test_a_category_outside_the_closed_set_is_invalid(self):
        with pytest.raises(ValidationError):
            Critique.model_validate({"issues": [{"category": "bad_vibes", "target": "summary", "rationale": "x"}]})

    def test_all_four_categories_are_valid(self):
        for category in ("causal_overreach", "misread_evidence", "overstated_driver", "unsupported_emphasis"):
            Critique(issues=[issue(category=category)])

    def test_a_target_that_is_not_the_summary_or_an_existing_claim_is_invalid(self, source, tmp_path, monkeypatch):
        result, _ = run_with_critic(source, tmp_path, monkeypatch)
        report, catalog = result["artifacts"]["report"], result["artifacts"]["evidence_catalog"]
        count = len(report.findings)
        validate_critique(Critique(issues=[issue("summary"), issue(f"claim_{count}")]), report, catalog)
        for bad in (f"claim_{count + 1}", "claim_0", "claim_x", "Claim 1", "evidence", ""):
            with pytest.raises(ValueError, match="target"):
                validate_critique(Critique(issues=[issue(bad)]), report, catalog)

    def test_a_rationale_may_not_carry_a_bare_digit_or_an_unknown_evidence_id(self, source, tmp_path, monkeypatch):
        result, _ = run_with_critic(source, tmp_path, monkeypatch)
        report, catalog = result["artifacts"]["report"], result["artifacts"]["evidence_catalog"]
        with pytest.raises(ValueError, match="digit"):
            validate_critique(Critique(issues=[issue(rationale="It grew 33 percent")]), report, catalog)
        with pytest.raises(ValueError, match="E9999"):
            validate_critique(Critique(issues=[issue(rationale="See {E9999}")]), report, catalog)
        known = next(iter(catalog.ids))
        validate_critique(Critique(issues=[issue(rationale="Compare with {" + known + "}")]), report, catalog)

    def test_an_empty_critique_is_valid(self, source, tmp_path, monkeypatch):
        result, _ = run_with_critic(source, tmp_path, monkeypatch)
        validate_critique(Critique(issues=[]), result["artifacts"]["report"], result["artifacts"]["evidence_catalog"])


class TestPlanShape:
    class _Model:
        def __init__(self, plan):
            self.plan = plan

        def with_structured_output(self, *_a, **_k):
            return self

        def invoke(self, _messages):
            return self.plan

    def _planner(self, plan, agents=ALL_AGENTS):
        return PlannerAgent(llm=self._Model(plan), available_agents=agents, data_sources=sales_data_source("sales.csv"))

    def test_sales_plan_ends_with_the_critic_depending_on_the_reporter(self):
        plan = self._planner(critic_plan()).generate_plan(QUESTION)
        assert [step.agent for step in plan.steps] == ["Fetcher", "Analyst", "Visualizer", "Reporter", "Critic"]
        assert plan.steps[4].depends_on == ["write_report"]

    def test_the_critic_is_mandatory_once_it_is_available(self):
        with pytest.raises(ValueError, match="Critic"):
            self._planner(crew_plan(visualizer=True)).generate_plan(QUESTION)

    def test_the_critic_must_depend_only_on_the_reporter(self):
        with pytest.raises(ValueError, match="Critic must depend only on the Reporter"):
            self._planner(critic_plan(critic_depends_on=("write_report", "analyze_baseline"))).generate_plan(QUESTION)
        with pytest.raises(ValueError, match="Critic must depend only on the Reporter"):
            self._planner(critic_plan(critic_depends_on=("analyze_baseline",))).generate_plan(QUESTION)

    def test_a_plan_naming_the_critic_is_invalid_when_it_is_unavailable(self):
        with pytest.raises(ValueError, match="unavailable Agents"):
            self._planner(critic_plan(), agents=FULL_AGENTS).generate_plan(QUESTION)

    def test_the_critic_step_must_not_name_data_sources(self):
        plan = critic_plan()
        plan.steps[4] = plan.steps[4].model_copy(update={"source_ids": ["sales_csv"]})
        with pytest.raises(ValueError, match="Critic Step must not name Data Sources"):
            self._planner(plan).generate_plan(QUESTION)

    def test_a_crew_without_the_critic_still_runs(self, source, tmp_path, monkeypatch):
        result, _ = run_crew(source, tmp_path, monkeypatch, plan=crew_plan(visualizer=True), available_agents=FULL_AGENTS)
        artifacts = result["artifacts"]
        assert isinstance(artifacts["report"], Report)
        assert artifacts.get("critic_outcome") is None
        assert "Reviewer Notes" not in render_report(artifacts["report"], artifacts["evidence_catalog"])


class TestCriticOutcomes:
    def test_a_clean_critique_leaves_the_report_without_reviewer_notes(self, source, tmp_path, monkeypatch):
        result, critic = run_with_critic(source, tmp_path, monkeypatch)
        artifacts = result["artifacts"]

        assert critic.calls == 1
        assert artifacts["critic_outcome"] == "clean"
        assert artifacts["critique"] == Critique(issues=[])
        assert artifacts["report"].reviewer_notes == []
        assert "Reviewer Notes" not in render_report(artifacts["report"], artifacts["evidence_catalog"])
        assert outcome(result)["status"] == "completed"
        assert _run_metadata(result)["critic_outcome"] == "clean"
        assert artifacts.get("terminal_failure") is None

    def test_the_critic_reads_the_draft_report_and_catalog_and_has_no_tools(self, source, tmp_path, monkeypatch):
        result, critic = run_with_critic(source, tmp_path, monkeypatch)
        context = critic.contexts[0]
        count = len(result["artifacts"]["report"].findings)
        assert context["evidence_catalog"]
        assert set(context["report"]["targets"]) == {"summary", *[f"claim_{n}" for n in range(1, count + 1)]}
        assert not any("tool" in key for key in context)

    def test_flagged_issues_become_reviewer_notes_and_the_claims_are_untouched(self, source, tmp_path, monkeypatch):
        flagged = Critique(issues=[issue("claim_1"), issue("summary", "misread_evidence")])
        result, _ = run_with_critic(source, tmp_path, monkeypatch, ScriptedCriticModel(flagged))
        artifacts = result["artifacts"]
        report = artifacts["report"]
        draft = artifacts["draft_report"]

        assert artifacts["critic_outcome"] == "unresolved"
        assert artifacts["critique"] == flagged
        assert [(n.category, n.target) for n in report.reviewer_notes] == [
            ("causal_overreach", "claim_1"), ("misread_evidence", "summary")]
        assert draft.reviewer_notes == []
        # The Critic cannot add or change Claims: only reviewer_notes differ from the draft.
        assert report.model_dump(exclude={"reviewer_notes"}) == draft.model_dump(exclude={"reviewer_notes"})
        rendered = render_report(report, artifacts["evidence_catalog"])
        assert "## Reviewer Notes" in rendered
        assert rendered.index("## Methodology") < rendered.index("## Reviewer Notes")
        assert _run_metadata(result)["critic_outcome"] == "unresolved"

    def test_an_invalid_target_gets_one_retry_with_feedback(self, source, tmp_path, monkeypatch):
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_99")]), Critique(issues=[issue("claim_1")]))
        result, _ = run_with_critic(source, tmp_path, monkeypatch, critic)

        assert critic.calls == 3, "the invalid response, its one retry, then the review of the revised Report (#19)"
        assert "claim_99" in critic.contexts[1]["validation_feedback"]
        assert not critic.contexts[0]["validation_feedback"]
        assert result["artifacts"]["critic_outcome"] == "unresolved"
        assert outcome(result)["retry_counts"]["validation_retries"] == 1
        assert outcome(result)["status"] == "completed"

    def test_unparseable_output_takes_the_same_retry_path(self, source, tmp_path, monkeypatch):
        critic = ScriptedCriticModel({"issues": [{"category": "nope", "target": "summary", "rationale": "x"}]}, Critique())
        result, _ = run_with_critic(source, tmp_path, monkeypatch, critic)
        assert critic.calls == 2
        assert result["artifacts"]["critic_outcome"] == "clean"

    def test_two_invalid_responses_count_as_a_critic_failure_that_ships_the_report(self, source, tmp_path, monkeypatch):
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_99")]))
        result, _ = run_with_critic(source, tmp_path, monkeypatch, critic)
        artifacts = result["artifacts"]

        assert critic.calls == 2
        assert artifacts["critic_outcome"] == "failed"
        assert artifacts["critique"] is None
        assert isinstance(artifacts["report"], Report) and artifacts["report"].reviewer_notes == []
        assert artifacts["report"].report_mode == "llm"
        assert outcome(result)["status"] == "failed" and "claim_99" in outcome(result)["error"]
        assert artifacts.get("terminal_failure") is None, "a Critic failure never fails the run"
        assert _run_metadata(result)["critic_outcome"] == "failed"

    def test_a_provider_failure_ships_the_unreviewed_report(self, source, tmp_path, monkeypatch):
        critic = ScriptedCriticModel(RuntimeError("provider exploded"))
        result, _ = run_with_critic(source, tmp_path, monkeypatch, critic)
        assert result["artifacts"]["critic_outcome"] == "failed"
        assert "provider exploded" in outcome(result)["error"]
        assert result["artifacts"].get("terminal_failure") is None
        assert isinstance(result["artifacts"]["report"], Report)


class TestCriticResume:
    def _rerun(self, source, tmp_path, monkeypatch, saver, critic, **kwargs):
        return run_with_critic(source, tmp_path, monkeypatch, critic, thread_id="resume", checkpointer=saver, **kwargs)

    def test_an_unchanged_rerun_reuses_the_critique_without_a_model_call(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        flagged = Critique(issues=[issue("claim_1")])
        first, critic = self._rerun(source, tmp_path, monkeypatch, saver, ScriptedCriticModel(flagged))
        second_critic = ScriptedCriticModel(Critique())
        second, _ = self._rerun(source, tmp_path, monkeypatch, saver, second_critic)

        assert critic.calls == 2 and second_critic.calls == 0  # review + re-review of the revision (#19)
        assert (outcome(second)["resume_outcome"], outcome(second)["resume_reason"]) == ("reused", "unchanged")
        assert outcome(second)["execution_key"] == outcome(first)["execution_key"]
        assert second["artifacts"]["critic_outcome"] == "unresolved"
        assert second["artifacts"]["critique"] == flagged
        # The Reporter reused its draft, and the Report still carries the Reviewer Notes exactly once.
        assert second["artifacts"]["report"] == first["artifacts"]["report"]
        assert second["artifacts"]["draft_report"].reviewer_notes == []

    def test_a_recorded_critic_failure_is_retried_on_resume_not_reused(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        self._rerun(source, tmp_path, monkeypatch, saver, ScriptedCriticModel(RuntimeError("down")))
        healed = ScriptedCriticModel(Critique())
        second, _ = self._rerun(source, tmp_path, monkeypatch, saver, healed)

        assert healed.calls == 1
        assert (outcome(second)["resume_outcome"], outcome(second)["resume_reason"]) == ("recomputed", "failed")
        assert second["artifacts"]["critic_outcome"] == "clean"
        assert outcome(second)["status"] == "completed"

    def test_a_changed_draft_report_recomputes_the_critique(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        self._rerun(source, tmp_path, monkeypatch, saver, ScriptedCriticModel())

        def reworded(context):
            content = valid_content(context)
            return content.model_copy(update={"claims": content.claims[:-1]})

        # Changed source data changes the Findings, so the Reporter recomputes and its new draft hashes differently.
        write_sales(source, [*base_rows(), sales_row("J3", "2025-06-10", quantity=5)])
        second_critic = ScriptedCriticModel()
        second, _ = self._rerun(source, tmp_path, monkeypatch, saver, second_critic,
                                reporter_model=ScriptedReporterModel(reworded))
        assert second_critic.calls == 1
        assert (outcome(second)["resume_outcome"], outcome(second)["resume_reason"]) == ("recomputed", "invalidated")


class TestSerializer:
    def test_critique_models_are_allowlisted(self):
        assert {"Critique", "CritiqueIssue"} <= {cls.__name__ for cls in crew_state_types()}

    def test_a_full_run_round_trips_under_strict_msgpack(self, source, tmp_path, monkeypatch, caplog):
        from langgraph.checkpoint.serde import _msgpack

        monkeypatch.setattr(_msgpack, "STRICT_MSGPACK_ENABLED", True)
        saver = MemorySaver(serde=crew_serde())
        flagged = Critique(issues=[issue("summary")])
        with caplog.at_level(logging.WARNING):
            run_with_critic(source, tmp_path, monkeypatch, ScriptedCriticModel(flagged), thread_id="s", checkpointer=saver)
            result, critic = run_with_critic(source, tmp_path, monkeypatch, ScriptedCriticModel(flagged),
                                             thread_id="s", checkpointer=saver)
        artifacts = result["artifacts"]
        assert isinstance(artifacts["critique"], Critique) and critic.calls == 0
        assert isinstance(artifacts["report"], Report) and artifacts["report"].reviewer_notes
        assert "unregistered type" not in caplog.text


class TestTerminalOutput:
    def test_reviewer_notes_show_in_the_terminal_output_only_when_issues_remain(self, source, tmp_path, monkeypatch):
        from cli import format_result

        flagged, _ = run_with_critic(source, tmp_path, monkeypatch, ScriptedCriticModel(Critique(issues=[issue("summary")])))
        clean, _ = run_with_critic(source, tmp_path, monkeypatch, thread_id="clean")
        assert "## Reviewer Notes" in format_result(flagged)
        assert "Reviewer Notes" not in format_result(clean)

    def test_a_critic_failure_is_stated_beneath_the_unreviewed_report(self, source, tmp_path, monkeypatch):
        from cli import format_result

        result, _ = run_with_critic(source, tmp_path, monkeypatch, ScriptedCriticModel(RuntimeError("down")))
        output = format_result(result)
        assert "# Analyst Crew Report" in output
        assert "Critic" in output.split("## Methodology")[-1] and "not reviewed" in output
