"""Issue #15 — the deterministic Visualizer Step through the primary seam: the Crew graph via ``invoke()``,
an in-memory checkpointer, a CSV in a temporary directory and scripted models. Asserts plan shape, charts in
the Report, resume reuse and cascade, and graceful degradation; never pixels."""
import pytest
from langgraph.checkpoint.memory import MemorySaver

from analyst_crew.agents import Plan
from analyst_crew.agents.planner import PlannerAgent
from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.reporting.charts import ChartArtifact
from analyst_crew.reporting.report import render_report
from analyst_crew.tools.sales import sales_data_source
from tests.crew_harness import (
    FULL_AGENTS, QUESTION, ScriptedReporterModel, ScriptedVisualizerModel, base_rows, crew_plan, entry_id, run_crew,
    sales_row, write_sales,
)
from tests.report_properties import chart_property_violations, report_property_violations


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", base_rows())


def drivers_choice(context):
    """A valid Visualizer response: the revenue bridge waterfall and the net revenue driver changes."""
    bridge = [f"revenue_bridge.{name}" for name in (
        "volume_effect", "price_effect", "interaction_effect", "discount_effect", "return_effect",
        "new_product_effect", "discontinued_product_effect", "total_effect")]
    drivers = [e["id"] for e in context["evidence_catalog"]
               if e["source"].startswith("dimensions.") and e["source"].endswith(".absolute_change")]
    return {"charts": [
        {"mark": "waterfall", "series": [{"entry_ids": [entry_id(context, s) for s in bridge]}]},
        {"mark": "bar", "series": [{"entry_ids": drivers}]},
    ]}


def run_full(source, tmp_path, monkeypatch, **kwargs):
    kwargs.setdefault("plan", crew_plan(visualizer=True))
    return run_crew(source, tmp_path, monkeypatch, available_agents=FULL_AGENTS, **kwargs)


class TestPlanShape:
    class _Model:
        def __init__(self, plan):
            self.plan = plan

        def with_structured_output(self, *_a, **_k):
            return self

        def invoke(self, _messages):
            return self.plan

    def _planner(self, plan, agents=FULL_AGENTS):
        return PlannerAgent(llm=self._Model(plan), available_agents=agents, data_sources=sales_data_source("sales.csv"))

    def test_sales_plan_is_fetcher_analyst_visualizer_reporter_when_all_are_available(self):
        plan = self._planner(crew_plan(visualizer=True)).generate_plan(QUESTION)

        assert [step.agent for step in plan.steps] == ["Fetcher", "Analyst", "Visualizer", "Reporter"]
        assert plan.steps[3].depends_on == ["analyze_baseline", "visualize_findings"]

    def test_a_plan_without_the_visualizer_is_invalid_once_the_visualizer_is_available(self):
        with pytest.raises(ValueError, match="Visualizer"):
            self._planner(crew_plan()).generate_plan(QUESTION)

    def test_a_plan_naming_the_visualizer_is_invalid_when_it_is_unavailable(self):
        with pytest.raises(ValueError, match="unavailable Agents"):
            self._planner(crew_plan(visualizer=True), agents=("Fetcher", "Analyst", "Reporter")).generate_plan(QUESTION)

    def test_the_visualizer_must_depend_on_the_analyst_step(self):
        plan = crew_plan(visualizer=True)
        plan.steps[2] = plan.steps[2].model_copy(update={"depends_on": ["fetch_sales"]})
        with pytest.raises(ValueError, match="Visualizer must depend on the Analyst"):
            self._planner(plan).generate_plan(QUESTION)

    def test_the_reporter_must_depend_on_the_analyst_and_the_visualizer(self):
        plan = crew_plan(visualizer=True)
        plan.steps[3] = plan.steps[3].model_copy(update={"depends_on": ["analyze_baseline"]})
        with pytest.raises(ValueError, match="Reporter must depend on the Analyst and Visualizer"):
            self._planner(plan).generate_plan(QUESTION)

    def test_a_crew_without_the_visualizer_still_gets_and_runs_the_shorter_plan(self, source, tmp_path, monkeypatch):
        plan = self._planner(crew_plan(), agents=("Fetcher", "Analyst", "Reporter")).generate_plan(QUESTION)
        assert [step.agent for step in plan.steps] == ["Fetcher", "Analyst", "Reporter"]

        result, _ = run_crew(source, tmp_path, monkeypatch)

        assert "charts" not in result["artifacts"]
        assert result["artifacts"]["report"].charts == []
        assert result["artifacts"]["report_mode"] == "llm"

    def test_the_planner_prompt_describes_the_visualizer_step_and_dependencies(self):
        from analyst_crew.agents.planner import SYSTEM_PROMPT

        assert "Visualizer Step" in SYSTEM_PROMPT and "Analyst and Visualizer" in SYSTEM_PROMPT


class TestChartsEndToEnd:
    def test_the_report_charts_section_holds_chart_references_rendered_as_image_links(self, source, tmp_path, monkeypatch):
        result, _ = run_full(source, tmp_path, monkeypatch)
        artifacts = result["artifacts"]

        charts = artifacts["charts"]
        assert [chart.chart_id for chart in charts] == ["revenue_bridge", "top_drivers"]
        assert artifacts["chart_mode"] == "fallback"
        assert [ref.artifact_id for ref in artifacts["report"].charts] == [chart.artifact_id for chart in charts]
        rendered = render_report(artifacts["report"], artifacts["evidence_catalog"])
        charts_section = rendered.split("## Charts")[1].split("## Caveats")[0]
        for chart in charts:
            assert f"![{chart.spec.title}]({chart.filename})" in charts_section
            assert (tmp_path / "artifacts" / "charts" / chart.filename).is_file()
        assert "Reserved" not in rendered

    def test_plotted_values_are_evidence_entry_raw_values(self, source, tmp_path, monkeypatch):
        result, _ = run_full(source, tmp_path, monkeypatch)
        artifacts = result["artifacts"]

        assert chart_property_violations(artifacts["charts"], artifacts["evidence_catalog"]) == []
        assert report_property_violations(artifacts["report"], artifacts["evidence_catalog"], artifacts["findings"]) == []
        bridge = artifacts["charts"][0]
        assert [point.value for point in bridge.series[0].points][-1] == artifacts["findings"].revenue_bridge.total_effect

    def test_the_property_check_flags_a_tampered_value_and_a_missing_png(self, source, tmp_path, monkeypatch):
        result, _ = run_full(source, tmp_path, monkeypatch)
        catalog = result["artifacts"]["evidence_catalog"]
        chart = result["artifacts"]["charts"][1]
        point = chart.series[0].points[0].model_copy(update={"value": 123456.0})
        tampered = chart.model_copy(update={
            "series": [chart.series[0].model_copy(update={"points": [point]})], "location": str(tmp_path / "gone.png"),
        })

        violations = " | ".join(chart_property_violations([tampered], catalog))
        assert "is not entry" in violations and "missing" in violations

    def test_the_visualizer_step_records_a_completed_outcome_and_a_handoff_without_secrets(self, source, tmp_path, monkeypatch):
        result, _ = run_full(source, tmp_path, monkeypatch, visualizer_model=ScriptedVisualizerModel(drivers_choice))
        artifacts = result["artifacts"]

        outcome = artifacts["step_outcomes"]["visualize_findings"]
        assert outcome["status"] == "completed" and outcome["chart_mode"] == "llm"
        assert (outcome["resume_outcome"], outcome["resume_reason"]) == ("recomputed", "never_ran")
        assert "visualize_findings" in result["steps_completed"]
        handoff = artifacts["handoffs"]["visualize_findings"]
        assert handoff["input_artifacts"]["baseline_findings"]["kind"] == "Findings"
        assert "sales.csv" not in str(handoff) and str(tmp_path) not in str(handoff)
        reporter_handoff = artifacts["handoffs"]["write_report"]
        assert reporter_handoff["input_artifacts"]["sales_charts"]["artifact_ids"] == [
            chart.artifact_id for chart in artifacts["charts"]
        ]

    def test_the_run_is_deterministic_chart_hashes_repeat_across_threads_and_directories(self, source, tmp_path, monkeypatch):
        first, _ = run_full(source, tmp_path / "a", monkeypatch, thread_id="one")
        second, _ = run_full(source, tmp_path / "b", monkeypatch, thread_id="two")

        assert [c.content_hash for c in first["artifacts"]["charts"]] == [c.content_hash for c in second["artifacts"]["charts"]]

    def test_chart_mode_is_attached_to_the_trace_metadata(self, source, tmp_path, monkeypatch):
        result, _ = run_full(source, tmp_path, monkeypatch)

        assert _run_metadata(result)["chart_mode"] == "fallback"



class TestResume:
    def test_unchanged_rerun_reuses_charts_and_the_report_with_no_model_call_or_render(self, source, tmp_path, monkeypatch):
        saver, model = MemorySaver(), ScriptedVisualizerModel(drivers_choice)
        first, reporter = run_full(source, tmp_path, monkeypatch, checkpointer=saver, visualizer_model=model)
        renders = []
        import analyst_crew.reporting.charts as charts_module

        real = charts_module.render_chart_png
        monkeypatch.setattr(charts_module, "render_chart_png", lambda *a: renders.append(1) or real(*a))
        second, _ = run_full(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=reporter,
                             visualizer_model=model)

        assert renders == [] and reporter.calls == 1 and model.calls == 1
        outcomes = second["artifacts"]["step_outcomes"]
        assert (outcomes["visualize_findings"]["resume_outcome"], outcomes["visualize_findings"]["resume_reason"]) == ("reused", "unchanged")
        assert (outcomes["write_report"]["resume_outcome"], outcomes["write_report"]["resume_reason"]) == ("reused", "unchanged")
        assert second["artifacts"]["charts"] == first["artifacts"]["charts"]
        assert second["artifacts"]["chart_mode"] == "llm"

    def test_changed_source_data_recomputes_the_charts_and_cascades_to_the_report(self, source, tmp_path, monkeypatch):
        saver, model = MemorySaver(), ScriptedVisualizerModel(drivers_choice)
        first, reporter = run_full(source, tmp_path, monkeypatch, checkpointer=saver, visualizer_model=model)
        write_sales(source, base_rows() + [sales_row("J3", "2025-06-10", quantity=4)])
        second, _ = run_full(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=reporter,
                             visualizer_model=model)

        outcomes = second["artifacts"]["step_outcomes"]
        assert (outcomes["visualize_findings"]["resume_outcome"], outcomes["visualize_findings"]["resume_reason"]) == ("recomputed", "invalidated")
        assert (outcomes["write_report"]["resume_outcome"], outcomes["write_report"]["resume_reason"]) == ("recomputed", "invalidated")
        assert reporter.calls == 2
        old_hashes = {c.content_hash for c in first["artifacts"]["charts"]}
        assert not old_hashes & {c.content_hash for c in second["artifacts"]["charts"]}
        assert chart_property_violations(second["artifacts"]["charts"], second["artifacts"]["evidence_catalog"]) == []

    def test_the_visualizer_key_is_its_step_contract_plus_the_findings_hash(self, source, tmp_path, monkeypatch):
        from analyst_crew.agents.reporter import findings_hash
        from analyst_crew.agents.resume import execution_key

        first, _ = run_full(source, tmp_path, monkeypatch)

        step = first["artifacts"]["plan"].steps[2]
        expected = execution_key(step, input_artifact_hashes=[findings_hash(first["artifacts"]["findings"])])
        assert first["artifacts"]["step_outcomes"]["visualize_findings"]["execution_key"] == expected

    def test_the_reporter_key_additionally_includes_the_chart_hashes(self, source, tmp_path, monkeypatch):
        from analyst_crew.agents.reporter import findings_hash
        from analyst_crew.agents.resume import execution_key

        first, _ = run_full(source, tmp_path, monkeypatch)

        artifacts = first["artifacts"]
        step, findings = artifacts["plan"].steps[3], artifacts["findings"]
        params = {"as_of_date": "2025-06-30"}
        chart_hashes = [chart.content_hash for chart in artifacts["charts"]]
        expected = execution_key(step, analysis_parameters=params,
                                 input_artifact_hashes=[findings_hash(findings), *chart_hashes])
        without_charts = execution_key(step, analysis_parameters=params, input_artifact_hashes=[findings_hash(findings)])
        assert artifacts["step_outcomes"]["write_report"]["execution_key"] == expected != without_charts

    def test_a_deleted_chart_file_is_regenerated_not_reused(self, source, tmp_path, monkeypatch):
        saver, model = MemorySaver(), ScriptedVisualizerModel(drivers_choice)
        first, reporter = run_full(source, tmp_path, monkeypatch, checkpointer=saver, visualizer_model=model)
        for chart in first["artifacts"]["charts"]:
            (tmp_path / "artifacts" / "charts" / chart.filename).unlink()

        second, _ = run_full(source, tmp_path, monkeypatch, checkpointer=saver, reporter_model=reporter,
                             visualizer_model=model)

        outcome = second["artifacts"]["step_outcomes"]["visualize_findings"]
        assert (outcome["resume_outcome"], outcome["resume_reason"]) == ("recomputed", "invalidated")
        assert chart_property_violations(second["artifacts"]["charts"], second["artifacts"]["evidence_catalog"]) == []
        assert [c.content_hash for c in second["artifacts"]["charts"]] == [c.content_hash for c in first["artifacts"]["charts"]]


class TestDegradation:
    def _break_rendering(self, monkeypatch):
        import analyst_crew.reporting.charts as charts_module

        def boom(*_args):
            raise RuntimeError("renderer exploded")

        monkeypatch.setattr(charts_module, "render_chart_png", boom)

    def test_a_chart_failure_never_destroys_the_report_and_is_recorded(self, source, tmp_path, monkeypatch):
        self._break_rendering(monkeypatch)

        result, reporter = run_full(source, tmp_path, monkeypatch)

        artifacts = result["artifacts"]
        assert artifacts["report"].report_mode == "llm" and artifacts["report"].charts == []
        assert reporter.calls == 1
        outcome = artifacts["step_outcomes"]["visualize_findings"]
        assert outcome["status"] == "failed" and "renderer exploded" in outcome["error"]
        assert artifacts["chart_mode"] is None and artifacts["charts"] == []
        assert artifacts.get("terminal_failure") is None

    def test_a_failed_visualizer_is_retried_on_the_next_run_not_reused(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        with monkeypatch.context() as broken:
            self._break_rendering(broken)
            run_full(source, tmp_path, monkeypatch, checkpointer=saver, visualizer_model=ScriptedVisualizerModel(drivers_choice))

        second, _ = run_full(source, tmp_path, monkeypatch, checkpointer=saver,
                             visualizer_model=ScriptedVisualizerModel(drivers_choice))

        outcome = second["artifacts"]["step_outcomes"]["visualize_findings"]
        assert (outcome["status"], outcome["resume_reason"]) == ("completed", "failed")
        assert len(second["artifacts"]["report"].charts) == 2
        assert second["artifacts"]["step_outcomes"]["write_report"]["resume_outcome"] == "recomputed"
