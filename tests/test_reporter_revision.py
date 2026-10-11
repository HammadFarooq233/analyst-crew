"""Issue #19 — the Reporter revision pass. Primary seam: the Crew graph via ``invoke()`` with an in-memory
checkpointer, a CSV in a temporary directory and scripted per-Agent models (never a real LLM). Asserts outcomes,
stored artifacts, validation guarantees, trace span labels and resume labels; never model wording."""
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.checkpoint.memory import MemorySaver

from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.reporting.critique import Critique
from analyst_crew.reporting.report import Claim, Report, render_report
from tests.crew_harness import (
    ScriptedReporterModel, base_rows, entry_id, invalid_stray_digit, invalid_unknown_id, valid_content, write_sales,
)
from tests.test_critic import ScriptedCriticModel, issue, outcome, run_with_critic


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", base_rows())


def shorter(context):
    """A revision a real model could give: one Claim fewer."""
    content = valid_content(context)
    return content.model_copy(update={"claims": content.claims[:-1]})


def bad_chart(context):
    content = valid_content(context)
    return content.model_copy(update={"summary": Claim(
        text="See {chart:nope} for net revenue {" + entry_id(context, "net_revenue.absolute_change") + "}.",
        evidence_ids=[entry_id(context, "net_revenue.absolute_change")])})


def run(source, tmp_path, monkeypatch, critic, reporter, **kwargs):
    return run_with_critic(source, tmp_path, monkeypatch, critic, reporter_model=reporter, **kwargs)


class TestRevisionOutcomes:
    def test_a_clean_critique_triggers_no_revision(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(valid_content)
        result, _ = run(source, tmp_path, monkeypatch, ScriptedCriticModel(), reporter)
        assert reporter.calls == 1
        assert result["artifacts"]["critic_outcome"] == "clean"

    def test_a_non_empty_critique_triggers_exactly_one_revision_even_if_issues_persist(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(valid_content, shorter, shorter, shorter)
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_1")]))  # every review flags the same issue
        result, _ = run(source, tmp_path, monkeypatch, critic, reporter)
        assert reporter.calls == 2, "one draft plus exactly one revision"
        assert result["artifacts"]["critic_outcome"] == "unresolved"

    def test_a_resolved_critique_ships_the_revised_report_without_notes(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(valid_content, shorter)
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_1")]), Critique())
        result, _ = run(source, tmp_path, monkeypatch, critic, reporter)
        artifacts = result["artifacts"]
        draft, final = artifacts["draft_report"], artifacts["report"]

        assert artifacts["critic_outcome"] == "revised"
        assert len(final.findings) == len(draft.findings) - 1
        assert final.reviewer_notes == [] and draft.reviewer_notes == []
        assert artifacts["critique"] == Critique(issues=[issue("claim_1")])
        assert final.report_mode == "llm"
        assert "Reviewer Notes" not in render_report(final, artifacts["evidence_catalog"])
        assert _run_metadata(result)["critic_outcome"] == "revised"
        assert outcome(result)["status"] == "completed"
        assert artifacts.get("terminal_failure") is None

    def test_the_revision_receives_the_critique_and_the_draft(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(valid_content, shorter)
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_1", "misread_evidence")]), Critique())
        run(source, tmp_path, monkeypatch, critic, reporter)
        assert "revision" not in reporter.contexts[0]
        revision = reporter.contexts[1]["revision"]
        assert [(i["target"], i["category"]) for i in revision["issues"]] == [("claim_1", "misread_evidence")]
        assert revision["draft"][0]["target"] == "summary"
        assert reporter.contexts[1]["evidence_catalog"] and reporter.contexts[1]["charts"]

    def test_caveats_methodology_and_charts_are_untouched_by_revision(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(valid_content, shorter)
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_1")]), Critique())
        artifacts = run(source, tmp_path, monkeypatch, critic, reporter)[0]["artifacts"]
        draft, final = artifacts["draft_report"], artifacts["report"]
        assert final.caveats == draft.caveats and final.methodology == draft.methodology
        assert final.charts == draft.charts and final.chart_mode == draft.chart_mode

    def test_unresolved_issues_after_revision_become_reviewer_notes_on_the_revised_report(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(valid_content, shorter)
        still = Critique(issues=[issue("summary", "overstated_driver")])
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_1")]), still)
        result, _ = run(source, tmp_path, monkeypatch, critic, reporter)
        artifacts = result["artifacts"]
        final = artifacts["report"]

        assert artifacts["critic_outcome"] == "unresolved"
        assert len(final.findings) == len(artifacts["draft_report"].findings) - 1, "the revised Report ships"
        assert [(n.category, n.target) for n in final.reviewer_notes] == [("overstated_driver", "summary")]
        assert artifacts["draft_report"].reviewer_notes == []
        assert "## Reviewer Notes" in render_report(final, artifacts["evidence_catalog"])
        assert outcome(result)["status"] == "completed"


class TestFailedRevision:
    @pytest.mark.parametrize("bad", ["unknown_id", "stray_digit", "bad_chart", "raises"])
    def test_a_revision_that_fails_validation_keeps_the_draft_and_lists_its_issues(self, source, tmp_path, monkeypatch, bad):
        revision = {"unknown_id": invalid_unknown_id, "stray_digit": invalid_stray_digit, "bad_chart": bad_chart,
                    "raises": RuntimeError("provider down")}[bad]
        reporter = ScriptedReporterModel(valid_content, revision)
        flagged = Critique(issues=[issue("claim_1"), issue("summary", "misread_evidence")])
        critic = ScriptedCriticModel(flagged)
        result, _ = run(source, tmp_path, monkeypatch, critic, reporter)
        artifacts = result["artifacts"]

        assert reporter.calls == 2, "a failed revision is not retried"
        assert critic.calls == 1, "nothing to re-review when the revision failed"
        assert artifacts["critic_outcome"] == "unresolved"
        final, draft = artifacts["report"], artifacts["draft_report"]
        assert final.model_dump(exclude={"reviewer_notes"}) == draft.model_dump(exclude={"reviewer_notes"})
        assert [(n.category, n.target) for n in final.reviewer_notes] == [
            ("causal_overreach", "claim_1"), ("misread_evidence", "summary")]
        assert artifacts["revision_failure"]
        assert outcome(result)["status"] == "completed"
        assert artifacts.get("terminal_failure") is None
        assert artifacts["critique"] == flagged

    def test_a_fallback_draft_is_not_revised_and_keeps_its_issues_as_notes(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(RuntimeError("reporter down"))
        critic = ScriptedCriticModel(Critique(issues=[issue("summary")]))
        result, _ = run(source, tmp_path, monkeypatch, critic, reporter)
        assert reporter.calls == 1
        assert not any("revision" in context for context in reporter.contexts)
        assert result["artifacts"]["critic_outcome"] == "unresolved"
        assert result["artifacts"]["report"].report_mode == "fallback"

    def test_a_failed_re_review_still_ships_the_validated_revision_with_the_original_issues(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(valid_content, shorter)
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_1")]), RuntimeError("critic down"))
        result, _ = run(source, tmp_path, monkeypatch, critic, reporter)
        artifacts = result["artifacts"]
        assert artifacts["critic_outcome"] == "unresolved"
        assert len(artifacts["report"].findings) == len(artifacts["draft_report"].findings) - 1
        assert [n.target for n in artifacts["report"].reviewer_notes] == ["claim_1"]


class _Spans(BaseCallbackHandler):
    def __init__(self):
        self.names = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.names.append(kwargs.get("name") or (serialized or {}).get("name"))


class TestTrace:
    def test_the_revision_is_a_second_labelled_reporter_span(self, source, tmp_path, monkeypatch):
        spans = _Spans()
        reporter = ScriptedReporterModel(valid_content, shorter)
        critic = ScriptedCriticModel(Critique(issues=[issue("claim_1")]), Critique())
        run(source, tmp_path, monkeypatch, critic, reporter, callbacks=[spans])
        assert spans.names.count("reporter_node") == 1
        assert spans.names.count("Reporter (revision)") == 1

    def test_no_revision_span_without_a_revision(self, source, tmp_path, monkeypatch):
        spans = _Spans()
        run_with_critic(source, tmp_path, monkeypatch, ScriptedCriticModel(), callbacks=[spans])
        assert "Reporter (revision)" not in spans.names


class TestResume:
    def _rerun(self, source, tmp_path, monkeypatch, saver, critic, reporter):
        return run(source, tmp_path, monkeypatch, critic, reporter, thread_id="resume", checkpointer=saver)

    def test_an_unchanged_rerun_reuses_the_final_report_and_critique_with_no_model_calls(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        flagged = Critique(issues=[issue("claim_1")])
        first, _ = self._rerun(source, tmp_path, monkeypatch, saver,
                               ScriptedCriticModel(flagged, Critique()), ScriptedReporterModel(valid_content, shorter))
        critic, reporter = ScriptedCriticModel(flagged), ScriptedReporterModel(valid_content)
        second, _ = self._rerun(source, tmp_path, monkeypatch, saver, critic, reporter)

        assert critic.calls == 0 and reporter.calls == 0
        assert (outcome(second)["resume_outcome"], outcome(second)["resume_reason"]) == ("reused", "unchanged")
        assert second["artifacts"]["critic_outcome"] == "revised"
        assert second["artifacts"]["critique"] == flagged
        assert second["artifacts"]["report"] == first["artifacts"]["report"]
        assert second["artifacts"]["draft_report"] == first["artifacts"]["draft_report"]
        assert len(second["artifacts"]["report"].findings) < len(second["artifacts"]["draft_report"].findings)

    def test_an_unchanged_rerun_reuses_unresolved_notes_after_a_failed_revision(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        flagged = Critique(issues=[issue("summary")])
        first, _ = self._rerun(source, tmp_path, monkeypatch, saver,
                               ScriptedCriticModel(flagged), ScriptedReporterModel(valid_content, RuntimeError("x")))
        critic, reporter = ScriptedCriticModel(), ScriptedReporterModel(valid_content)
        second, _ = self._rerun(source, tmp_path, monkeypatch, saver, critic, reporter)
        assert critic.calls == 0 and reporter.calls == 0
        assert second["artifacts"]["report"] == first["artifacts"]["report"]
        assert second["artifacts"]["critic_outcome"] == "unresolved"

    def test_the_revised_report_round_trips_under_strict_msgpack(self, source, tmp_path, monkeypatch):
        from langgraph.checkpoint.serde import _msgpack

        from analyst_crew.state.serde import crew_serde

        monkeypatch.setattr(_msgpack, "STRICT_MSGPACK_ENABLED", True)
        saver = MemorySaver(serde=crew_serde())
        flagged = Critique(issues=[issue("claim_1")])
        self._rerun(source, tmp_path, monkeypatch, saver,
                    ScriptedCriticModel(flagged, flagged), ScriptedReporterModel(valid_content, shorter))
        critic = ScriptedCriticModel()
        second, _ = self._rerun(source, tmp_path, monkeypatch, saver, critic, ScriptedReporterModel())
        assert critic.calls == 0 and isinstance(second["artifacts"]["report"], Report)
        assert second["artifacts"]["report"].reviewer_notes
