"""Issue #17 — the Reporter can reference charts by placeholder. Driven through the Crew graph via ``invoke()``
(shared harness, scripted models, in-memory checkpointer); validation is also checked as a pure unit.
Asserts properties (ids, outcomes, retry counts, rendered links), never model wording."""
import pytest
from langgraph.checkpoint.memory import MemorySaver

from analyst_crew.reporting.catalog import build_evidence_catalog
from analyst_crew.reporting.report import (
    Claim, ReporterContent, ReportValidationError, render_report, validate_reporter_content,
)
from tests.crew_harness import (
    FULL_AGENTS, ScriptedReporterModel, base_rows, crew_plan, entry_id, run_crew, sales_row, valid_content, write_sales,
)
from tests.findings_factory import make_findings
from tests.report_properties import report_property_violations


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", base_rows())


def run_full(source, tmp_path, monkeypatch, *reporter_responses, **kwargs):
    kwargs.setdefault("plan", crew_plan(visualizer=True))
    if "reporter_model" not in kwargs:
        kwargs["reporter_model"] = ScriptedReporterModel(*reporter_responses)
    return run_crew(source, tmp_path, monkeypatch, available_agents=FULL_AGENTS, **kwargs)


def with_chart_claim(chart_id):
    """A valid Reporter response whose extra Claim mentions a chart by placeholder."""
    def respond(context):
        content = valid_content(context)
        claim = Claim(text="The revenue bridge is drawn in {chart:%s}." % chart_id,
                      evidence_ids=[entry_id(context, "revenue_bridge.volume_effect")])
        return content.model_copy(update={"claims": [*content.claims, claim]})
    return respond


def _catalog():
    return build_evidence_catalog(make_findings())


class TestHandoffCarriesCharts:
    def test_the_model_context_lists_chart_ids_and_titles_resolved_from_the_visualizer_output(self, source, tmp_path, monkeypatch):
        result, reporter = run_full(source, tmp_path, monkeypatch)

        listed = reporter.contexts[0]["charts"]
        charts = result["artifacts"]["charts"]
        assert listed == [{"chart_id": chart.chart_id, "title": chart.spec.title} for chart in charts]
        assert [item["chart_id"] for item in listed] == ["revenue_bridge", "top_drivers"]

    def test_a_crew_without_a_visualizer_gives_the_model_no_charts(self, source, tmp_path, monkeypatch):
        _, reporter = run_crew(source, tmp_path, monkeypatch)

        assert reporter.contexts[0]["charts"] == []


class TestValidChartReference:
    def test_a_claim_may_reference_a_chart_by_placeholder(self, source, tmp_path, monkeypatch):
        result, reporter = run_full(source, tmp_path, monkeypatch, with_chart_claim("revenue_bridge"))

        artifacts = result["artifacts"]
        report, catalog = artifacts["report"], artifacts["evidence_catalog"]
        assert reporter.calls == 1 and report.report_mode == "llm"
        assert report.findings[-1].chart_ids == ["revenue_bridge"]
        assert all(claim.chart_ids == [] for claim in report.findings[:-1])
        rendered = render_report(report, catalog)
        title = artifacts["charts"][0].spec.title
        findings = rendered.split("## Findings")[1].split("## Charts")[0]
        assert "{chart:" not in rendered and title in findings
        # Charts stay a code-assembled section of image links, no prose.
        charts_section = rendered.split("## Charts")[1].split("## Caveats")[0]
        assert charts_section.count("![") == 2
        assert report_property_violations(report, catalog, artifacts["findings"]) == []

    def test_a_chart_reference_does_not_count_as_evidence_and_is_not_a_stray_digit(self):
        catalog = _catalog()
        first = catalog.entries[0].id
        content = ReporterContent(
            summary=Claim(text="Net revenue changed by {%s}." % catalog.by_source("net_revenue.absolute_change").id),
            claims=[Claim(text="See {chart:chart_1} for the drivers.", evidence_ids=[first])],
        )

        validated = validate_reporter_content(content, catalog, chart_ids={"chart_1"})

        assert validated.claims[0].chart_ids == ["chart_1"]
        assert validated.claims[0].evidence_ids == [first]

    def test_a_claim_citing_only_a_chart_still_needs_evidence(self):
        catalog = _catalog()
        content = ReporterContent(
            summary=Claim(text="Net revenue changed by {%s}." % catalog.by_source("net_revenue.absolute_change").id),
            claims=[Claim(text="See {chart:chart_1}.")],
        )

        with pytest.raises(ReportValidationError, match="cites no Evidence Entry"):
            validate_reporter_content(content, catalog, chart_ids={"chart_1"})


class TestInvalidChartReference:
    def test_a_reference_to_a_nonexistent_chart_is_rejected_then_retried_once_with_feedback(self, source, tmp_path, monkeypatch):
        result, reporter = run_full(source, tmp_path, monkeypatch, with_chart_claim("no_such_chart"), with_chart_claim("top_drivers"))

        artifacts = result["artifacts"]
        assert reporter.calls == 2
        assert "no_such_chart" in reporter.contexts[1]["validation_feedback"]
        assert artifacts["report"].report_mode == "llm" and artifacts["report"].findings[-1].chart_ids == ["top_drivers"]
        assert artifacts["step_outcomes"]["write_report"]["retry_counts"]["validation_retries"] == 1
        assert artifacts["terminal_failure"] is None

    def test_a_reference_when_the_crew_has_no_charts_is_invalid(self, source, tmp_path, monkeypatch):
        reporter = ScriptedReporterModel(with_chart_claim("revenue_bridge"))

        result, _ = run_crew(source, tmp_path, monkeypatch, reporter_model=reporter)

        assert reporter.calls == 2 and result["artifacts"]["report"].report_mode == "fallback"

    def test_validation_names_every_unknown_chart_id(self):
        catalog = _catalog()
        first = catalog.entries[0].id
        content = ReporterContent(
            summary=Claim(text="See {chart:ghost}.", evidence_ids=[first]),
            claims=[Claim(text="Also {chart:phantom}.", evidence_ids=[first])],
        )

        with pytest.raises(ReportValidationError) as raised:
            validate_reporter_content(content, catalog, chart_ids={"chart_1"})

        assert "ghost" in str(raised.value) and "phantom" in str(raised.value)


class TestFallbackWithCharts:
    def test_two_bad_chart_references_fall_back_to_a_report_that_still_renders_the_charts(self, source, tmp_path, monkeypatch):
        result, reporter = run_full(source, tmp_path, monkeypatch, with_chart_claim("ghost"))

        artifacts = result["artifacts"]
        report, catalog = artifacts["report"], artifacts["evidence_catalog"]
        assert reporter.calls == 2 and report.report_mode == "fallback"
        assert "ghost" in artifacts["terminal_failure"]["error"]
        assert [ref.chart_id for ref in report.charts] == ["revenue_bridge", "top_drivers"]
        rendered = render_report(report, catalog)
        assert "Fallback Report" in rendered and "{chart:" not in rendered
        assert rendered.split("## Charts")[1].split("## Caveats")[0].count("![") == 2
        assert report_property_violations(report, catalog, artifacts["findings"]) == []


class TestResumeWithChartReferences:
    def test_unchanged_rerun_reuses_the_report_with_chart_references_and_no_model_call(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        _, reporter = run_full(source, tmp_path, monkeypatch, with_chart_claim("revenue_bridge"), checkpointer=saver)

        second, _ = run_full(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=reporter)

        outcome = second["artifacts"]["step_outcomes"]["write_report"]
        assert outcome["resume_outcome"] == "reused" and reporter.calls == 1
        assert second["artifacts"]["report"].findings[-1].chart_ids == ["revenue_bridge"]

    def test_changed_source_data_recomputes_the_charts_and_the_reporter_sees_the_new_chart_set(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        first, reporter = run_full(source, tmp_path, monkeypatch, with_chart_claim("revenue_bridge"), checkpointer=saver)
        write_sales(source, base_rows() + [sales_row("J3", "2025-06-10", quantity=4)])

        second, _ = run_full(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=reporter)

        outcome = second["artifacts"]["step_outcomes"]["write_report"]
        assert (outcome["resume_outcome"], outcome["resume_reason"]) == ("recomputed", "invalidated")
        assert reporter.calls == 2
        old_hashes = [chart.content_hash for chart in first["artifacts"]["charts"]]
        new_hashes = [chart.content_hash for chart in second["artifacts"]["charts"]]
        assert not set(old_hashes) & set(new_hashes)
        assert [ref.content_hash for ref in second["artifacts"]["report"].charts] == new_hashes
        assert second["artifacts"]["report"].findings[-1].chart_ids == ["revenue_bridge"]
