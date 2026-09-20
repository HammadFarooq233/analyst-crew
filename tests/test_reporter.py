"""Issue #13 — Reporter Agent through the primary seam: the Crew graph via
``invoke()``, an in-memory checkpointer, a CSV in a temporary directory and
scripted models injected through the per-Agent overrides.

Asserts external behavior only: the returned Crew state, the Report artifact,
``report_mode``, step outcomes and terminal failure, and the rendered output.
Properties over exact wording: every number in a rendered Report matches a
catalog entry, every limitation appears, the fallback is marked.
"""
import pytest

from analyst_crew.agents import Plan
from analyst_crew.agents.planner import PlannerAgent
from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.reporting.catalog import EvidenceCatalog
from analyst_crew.reporting.report import Report, render_report
from analyst_crew.tools.fred import fred_cpi_data_source
from analyst_crew.tools.sales import sales_data_source
from analyst_crew.tools.targets import sales_targets_data_source
from tests.report_properties import report_property_violations
from tests.crew_harness import (
    QUESTION, ScriptedReporterModel, base_rows, body_numbers, crew_plan, displays, invalid_stray_digit,
    invalid_unknown_id, run_crew, valid_content, write_sales,
)


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", base_rows())


def _report_and_catalog(result):
    artifacts = result["artifacts"]
    return artifacts["report"], artifacts["evidence_catalog"], render_report(artifacts["report"], artifacts["evidence_catalog"])


class TestValidReporter:
    def test_yields_an_llm_mode_report_whose_rendered_numbers_all_come_from_the_catalog(self, source, tmp_path, monkeypatch):
        result, reporter = run_crew(source, tmp_path, monkeypatch)

        report, catalog, rendered = _report_and_catalog(result)
        assert isinstance(report, Report) and isinstance(catalog, EvidenceCatalog)
        assert result["artifacts"]["report_mode"] == report.report_mode == "llm"
        assert report.fallback_notice is None
        assert result["artifacts"].get("terminal_failure") is None
        assert reporter.calls == 1
        assert "Fallback" not in rendered

        catalog_text = " ".join(displays(catalog))
        numbers = body_numbers(rendered)
        assert numbers, "a Report states numbers"
        assert all(number in catalog_text for number in numbers)
        # Net revenue 30 -> 40: the catalog's own display strings are what Priya reads.
        assert "+10.00" in rendered and "2025-06-01 to 2025-06-30" in rendered

    def test_claims_keep_their_evidence_ids_and_every_one_exists_in_the_catalog(self, source, tmp_path, monkeypatch):
        result, _ = run_crew(source, tmp_path, monkeypatch)

        report, catalog, _ = _report_and_catalog(result)
        assert report.summary.evidence_ids
        for claim in [report.summary, *report.findings]:
            assert claim.evidence_ids and set(claim.evidence_ids) <= catalog.ids

    def test_reporter_step_records_a_completed_outcome_and_a_handoff_without_secrets(self, source, tmp_path, monkeypatch):
        result, _ = run_crew(source, tmp_path, monkeypatch)

        artifacts = result["artifacts"]
        outcome = artifacts["step_outcomes"]["write_report"]
        assert outcome["status"] == "completed" and outcome["report_mode"] == "llm"
        assert outcome["resume_outcome"] == "recomputed" and outcome["resume_reason"] == "never_ran"
        assert outcome["retry_counts"] == {"validation_retries": 0, "tool_retries": 0}
        assert "write_report" in result["steps_completed"]

        handoff = artifacts["handoffs"]["write_report"]
        findings_reference = handoff["input_artifacts"]["baseline_findings"]
        assert findings_reference["kind"] == "Findings" and findings_reference["content_hash"]
        assert handoff["user_question"] == QUESTION
        assert "sales.csv" not in str(handoff)

    def test_charts_section_is_explicitly_reserved_and_empty(self, source, tmp_path, monkeypatch):
        result, _ = run_crew(source, tmp_path, monkeypatch)

        report, _, rendered = _report_and_catalog(result)
        assert report.charts == []
        assert "## Charts" in rendered and "Reserved" in rendered

    def test_the_model_sees_only_the_catalog_the_question_and_the_objective(self, source, tmp_path, monkeypatch):
        _, reporter = run_crew(source, tmp_path, monkeypatch)

        context = reporter.contexts[0]
        assert set(context) == {"step_objective", "user_question", "evidence_catalog", "validation_feedback"}
        assert context["user_question"] == QUESTION
        assert all({"id", "label", "display", "source"} <= set(entry) for entry in context["evidence_catalog"])
        assert "sales.csv" not in str(context)


class TestValidationAndRetry:
    @pytest.mark.parametrize("bad, expected_feedback", [
        (invalid_unknown_id, "E9999"),
        (invalid_stray_digit, "digit"),
        ({"summary": {"text": "no shape"}}, "JSON shape"),
    ], ids=["unknown-evidence-id", "stray-digit", "schema-violation"])
    def test_invalid_output_is_retried_once_with_feedback_then_succeeds(self, bad, expected_feedback, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(bad, valid_content)

        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=reporter)

        assert reporter.calls == 2
        assert expected_feedback in reporter.contexts[1]["validation_feedback"]
        assert reporter.contexts[0]["validation_feedback"] == ""
        assert result["artifacts"]["report_mode"] == "llm"
        assert result["artifacts"].get("terminal_failure") is None
        assert result["artifacts"]["step_outcomes"]["write_report"]["retry_counts"] == {
            "validation_retries": 1, "tool_retries": 0,
        }

    def test_a_placeholder_naming_an_unknown_entry_is_rejected(self, source, tmp_path, monkeypatch):
        def unknown_placeholder(context):
            content = valid_content(context)
            return content.model_copy(update={"summary": content.summary.model_copy(update={"text": "It moved {E9999}."})})

        reporter = ScriptedReporterModel(unknown_placeholder, valid_content)

        run_crew(source, tmp_path, monkeypatch, reporter_model=reporter)

        assert "E9999" in reporter.contexts[1]["validation_feedback"]

    def test_transient_provider_failure_is_retried_within_bounds_and_recovers(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(TimeoutError("provider timed out"), valid_content)

        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=reporter)

        assert reporter.calls == 2
        assert result["artifacts"]["report_mode"] == "llm"
        assert result["artifacts"]["step_outcomes"]["write_report"]["retry_counts"] == {
            "validation_retries": 0, "tool_retries": 1,
        }


class TestFallbackReport:
    def _assert_visibly_marked_fallback(self, result):
        report, catalog, rendered = _report_and_catalog(result)
        assert result["artifacts"]["report_mode"] == report.report_mode == "fallback"
        assert "Fallback Report" in rendered and "without the Reporter Agent" in rendered
        # The deterministic template still answers with cited, catalog-sourced numbers.
        assert "+10.00" in rendered and "+33.3%" in rendered
        catalog_text = " ".join(displays(catalog))
        assert all(number in catalog_text for number in body_numbers(rendered))
        assert report.summary.evidence_ids and all(claim.evidence_ids for claim in report.findings)
        return report

    def test_second_invalid_response_yields_a_marked_fallback_and_a_recorded_terminal_failure(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(invalid_stray_digit, invalid_unknown_id)

        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=reporter)

        assert reporter.calls == 2
        self._assert_visibly_marked_fallback(result)
        failure = result["artifacts"]["terminal_failure"]
        assert failure["agent"] == "Reporter" and failure["step_id"] == "write_report"
        assert failure["required"] is False and "E9999" in failure["error"]
        assert failure["retry_counts"]["validation_retries"] == 1
        outcome = result["artifacts"]["step_outcomes"]["write_report"]
        assert outcome["status"] == "failed" and outcome["report_mode"] == "fallback"
        # The analysis that already succeeded is intact.
        assert result["artifacts"]["findings"].net_revenue.absolute_change == pytest.approx(10.0)
        assert result["artifacts"]["step_outcomes"]["analyze_baseline"]["status"] == "completed"

    def test_non_transient_provider_failure_is_terminal_immediately_and_falls_back(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(PermissionError("401 invalid credentials"), valid_content)

        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=reporter)

        assert reporter.calls == 1
        self._assert_visibly_marked_fallback(result)
        assert "401" in result["artifacts"]["terminal_failure"]["error"]

    def test_persistent_transient_provider_failure_falls_back_after_bounded_retries(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(TimeoutError("timed out"))

        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=reporter)

        assert reporter.calls == 2
        self._assert_visibly_marked_fallback(result)
        assert result["artifacts"]["terminal_failure"]["retry_counts"]["tool_retries"] == 1

    def test_reporter_that_cannot_be_constructed_still_yields_a_fallback(self, source, tmp_path, monkeypatch):
        import analyst_crew.agents.reporter as reporter_module

        def unavailable():
            raise RuntimeError("LLM credentials not set")

        monkeypatch.setattr(reporter_module, "production_reporter_model", unavailable)
        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=None)

        self._assert_visibly_marked_fallback(result)
        assert "credentials" in result["artifacts"]["terminal_failure"]["error"]

    def test_failure_and_report_mode_are_attached_to_the_trace_metadata(self, source, tmp_path, monkeypatch):
        failing, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=ScriptedReporterModel(RuntimeError("boom")))
        healthy, _ = run_crew(source, tmp_path, monkeypatch, thread_id="thread-2")

        failed_metadata, healthy_metadata = _run_metadata(failing), _run_metadata(healthy)
        assert failed_metadata["report_mode"] == "fallback"
        assert failed_metadata["terminal_failure"]["agent"] == "Reporter"
        assert healthy_metadata["report_mode"] == "llm" and "terminal_failure" not in healthy_metadata
        assert healthy_metadata["step_outcomes"]["write_report"]["status"] == "completed"


class TestMandatoryCaveatsAndMethodology:
    def _rendered_sections(self, result):
        _, _, rendered = _report_and_catalog(result)
        caveats = rendered.split("## Caveats")[1].split("## Methodology")[0]
        methodology = rendered.split("## Methodology")[1].split("## Evidence")[0]
        return caveats, methodology

    @pytest.mark.parametrize("make_reporter", [ScriptedReporterModel, lambda: ScriptedReporterModel(RuntimeError("down"))],
                             ids=["llm", "fallback"])
    def test_every_findings_limitation_and_data_quality_disclosure_appears_in_every_mode(self, make_reporter, source, tmp_path, monkeypatch):
        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=make_reporter())

        caveats, methodology = self._rendered_sections(result)
        findings = result["artifacts"]["findings"]
        for limitation in findings.revenue_bridge.limitations:
            assert limitation in caveats
        assert "4 source rows" in caveats and "0 were quarantined" in caveats and "including 0 excluded as duplicate" in caveats
        assert "2025-06-01 to 2025-06-30" in methodology
        assert "15.0%" in methodology and "5.0%" in methodology
        assert findings.dataset_content_hash in methodology and findings.dataset_artifact_id in methodology
        assert findings.revenue_bridge.methodology in methodology

    def test_absent_optional_sources_are_stated_as_not_part_of_the_run(self, source, tmp_path, monkeypatch):
        result, _ = run_crew(source, tmp_path, monkeypatch)

        caveats, _ = self._rendered_sections(result)
        assert "Sales-target comparison was not part of this run" in caveats
        assert "CPI" in caveats and "not part of this run" in caveats

    def test_unavailable_and_not_configured_optional_sources_appear_as_caveats(self, source, tmp_path, monkeypatch):
        extra = {**sales_targets_data_source(), **fred_cpi_data_source()}
        result, _ = run_crew(source, tmp_path, monkeypatch, extra_sources=extra)

        caveats, methodology = self._rendered_sections(result)
        findings = result["artifacts"]["findings"]
        assert findings.target_comparison.source_status.status != "available"
        for limitation in findings.limitations:
            assert limitation in caveats
        assert "Sales-target comparison is not included in this Report" in caveats
        assert "Inflation-adjusted (CPI) comparison is not included in this Report" in caveats
        assert "Optional source, Postgres sales targets" in methodology and "Optional source, FRED CPI" in methodology

    def test_an_available_cpi_source_brings_the_non_causality_caveat(self, source, tmp_path, monkeypatch):
        import analyst_crew.tools.fred as fred_module

        observations = [{"date": "2025-05-01", "value": 300.0}, {"date": "2025-06-01", "value": 306.0}]
        monkeypatch.setattr(fred_module, "retrieve_cpi", lambda *_a, **_k: {"status": "available", "observations": observations})
        result, _ = run_crew(source, tmp_path, monkeypatch, extra_sources=fred_cpi_data_source())

        caveats, methodology = self._rendered_sections(result)
        findings = result["artifacts"]["findings"]
        assert findings.cpi_adjustment.source_status.status == "available"
        assert findings.cpi_adjustment.non_causality_caveat in caveats
        assert findings.cpi_adjustment.methodology in methodology

    def test_caveats_cannot_be_dropped_by_what_the_model_writes(self, source, tmp_path, monkeypatch):
        def silent_model(context):
            content = valid_content(context)
            return content.model_copy(update={"claims": content.claims[:1]})

        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=ScriptedReporterModel(silent_model))

        caveats, _ = self._rendered_sections(result)
        for limitation in result["artifacts"]["findings"].revenue_bridge.limitations:
            assert limitation in caveats


class TestFallbackCoversMaterialChanges:
    def test_every_material_change_is_a_templated_claim_with_size_and_share(self, source, tmp_path, monkeypatch):
        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=ScriptedReporterModel(RuntimeError("down")))

        report, catalog, rendered = _report_and_catalog(result)
        material = result["artifacts"]["findings"].material_changes
        assert len(material) == 5  # one value in each of the five dimensions, all above both thresholds
        assert len(report.findings) == len(material)
        assert "+100.0%" in rendered  # each is the whole of the +10.00 change


class TestResume:
    def test_unchanged_rerun_reuses_the_report_with_no_reporter_model_call(self, source, tmp_path, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        saver = MemorySaver()
        first, reporter = run_crew(source, tmp_path, monkeypatch, checkpointer=saver)
        second, _ = run_crew(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=reporter)

        assert reporter.calls == 1
        outcome = second["artifacts"]["step_outcomes"]["write_report"]
        assert (outcome["resume_outcome"], outcome["resume_reason"]) == ("reused", "unchanged")
        assert second["artifacts"]["report"] == first["artifacts"]["report"]
        assert second["artifacts"]["report_mode"] == "llm"

    def test_changed_findings_invalidate_the_report_and_regenerate_it(self, source, tmp_path, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        saver = MemorySaver()
        first, reporter = run_crew(source, tmp_path, monkeypatch, checkpointer=saver)
        write_sales(source, base_rows() + [_extra_june_order()])
        second, _ = run_crew(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=reporter)

        assert reporter.calls == 2
        outcome = second["artifacts"]["step_outcomes"]["write_report"]
        assert (outcome["resume_outcome"], outcome["resume_reason"]) == ("recomputed", "invalidated")
        assert second["artifacts"]["evidence_catalog"] != first["artifacts"]["evidence_catalog"]
        rendered = render_report(second["artifacts"]["report"], second["artifacts"]["evidence_catalog"])
        assert "+50.00" in rendered and "+10.00" not in rendered

    def test_a_failed_reporter_is_retried_on_the_next_run_not_reused(self, source, tmp_path, monkeypatch):
        from langgraph.checkpoint.memory import MemorySaver

        saver = MemorySaver()
        failing = ScriptedReporterModel(RuntimeError("down"))
        first, _ = run_crew(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=failing)
        recovered = ScriptedReporterModel()
        second, _ = run_crew(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=recovered)

        assert first["artifacts"]["report_mode"] == "fallback"
        assert recovered.calls == 1
        assert second["artifacts"]["report_mode"] == "llm"
        assert second["artifacts"]["step_outcomes"]["write_report"]["resume_reason"] == "failed"
        assert second["artifacts"].get("terminal_failure") is None


def _extra_june_order():
    from tests.crew_harness import sales_row

    return sales_row("J3", "2025-06-10", quantity=4)


class TestPlannerCapabilities:
    class _Model:
        def __init__(self, plan):
            self.plan = plan

        def with_structured_output(self, *_a, **_k):
            return self

        def invoke(self, _messages):
            return self.plan

    def _planner(self, plan, agents=("Fetcher", "Analyst", "Reporter")):
        return PlannerAgent(llm=self._Model(plan), available_agents=agents, data_sources=sales_data_source("sales.csv"))

    def test_plan_with_a_reporter_step_is_valid_when_the_reporter_is_available(self):
        plan = self._planner(crew_plan()).generate_plan(QUESTION)

        assert [step.agent for step in plan.steps] == ["Fetcher", "Analyst", "Reporter"]

    def test_plan_referencing_the_reporter_is_invalid_when_it_is_unavailable(self):
        with pytest.raises(ValueError, match="unavailable Agents"):
            self._planner(crew_plan(), agents=("Fetcher", "Analyst")).generate_plan(QUESTION)

    def test_plan_without_a_reporter_step_is_invalid_once_the_reporter_is_available(self):
        with pytest.raises(ValueError, match="Reporter"):
            self._planner(crew_plan(reporter=False)).generate_plan(QUESTION)

    def test_reporter_must_depend_on_the_analyst_step(self):
        plan = crew_plan()
        plan.steps[2] = plan.steps[2].model_copy(update={"depends_on": ["fetch_sales"]})
        with pytest.raises(ValueError, match="Reporter must depend on the Analyst"):
            self._planner(plan).generate_plan(QUESTION)

    def test_a_plan_without_a_reporter_step_produces_no_report(self, source, tmp_path, monkeypatch):
        result, reporter = run_crew(source, tmp_path, monkeypatch, plan=crew_plan(reporter=False),
                                    available_agents=("Fetcher", "Analyst"))

        assert reporter.calls == 0
        assert "report" not in result["artifacts"]
        assert isinstance(result["artifacts"]["plan"], Plan)


class TestReportPropertyCheck:
    """Guards the shared property check itself, so the live and eval uses cannot pass vacuously."""

    def test_a_healthy_report_has_no_violations_and_regressions_are_flagged(self, source, tmp_path, monkeypatch):
        result, _ = run_crew(source, tmp_path, monkeypatch)
        report, catalog = result["artifacts"]["report"], result["artifacts"]["evidence_catalog"]
        findings = result["artifacts"]["findings"]

        assert report_property_violations(report, catalog, findings) == []

        broken = report.model_copy(update={
            "caveats": [], "findings": report.findings[:1],
            "summary": report.summary.model_copy(update={"text": "Sales grew 33 percent."}),
        })
        violations = " | ".join(report_property_violations(broken, catalog, findings))
        assert "digit outside a placeholder" in violations
        assert "limitation missing from Caveats" in violations
        assert "Material Change not covered" in violations
