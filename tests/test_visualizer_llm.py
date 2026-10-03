"""Issue #16 — the LLM Visualizer through the primary seam (Crew ``invoke()``, in-memory checkpointer, scripted
models), plus pure unit tests of Chart Spec validation for model output. Asserts external behavior: chart_mode,
plotted values equal Evidence Entry raw values, labels come from the catalog, retry/fallback outcomes, trace
metadata. Never pixels or exact model wording."""
import pytest
from langgraph.checkpoint.memory import MemorySaver

from analyst_crew.agents.tracing import _run_metadata
from analyst_crew.reporting.catalog import build_evidence_catalog
from analyst_crew.reporting.chart_choice import VisualizerContent, chart_specs_from_content
from analyst_crew.reporting.charts import ChartSpecError
from tests.crew_harness import (
    FULL_AGENTS, ScriptedVisualizerModel, base_rows, crew_plan, entry_id, run_crew, write_sales,
)
from tests.findings_factory import make_findings
from tests.report_properties import chart_property_violations


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


@pytest.fixture
def source(tmp_path):
    return write_sales(tmp_path / "sales.csv", base_rows())


def run_full(source, tmp_path, monkeypatch, model, **kwargs):
    return run_crew(source, tmp_path, monkeypatch, plan=crew_plan(visualizer=True), available_agents=FULL_AGENTS,
                    visualizer_model=model, **kwargs)


def bridge_choice(context):
    sources = [f"revenue_bridge.{name}" for name in (
        "volume_effect", "price_effect", "interaction_effect", "discount_effect", "return_effect",
        "new_product_effect", "discontinued_product_effect", "total_effect")]
    return {"charts": [{"mark": "waterfall", "series": [{"entry_ids": [entry_id(context, s) for s in sources]}]}]}


def two_charts(context):
    choice = bridge_choice(context)
    metrics = ("gross_revenue", "net_revenue")
    choice["charts"].append({"mark": "grouped_bar", "series": [
        {"entry_ids": [entry_id(context, f"{m}.{period}") for m in metrics]} for period in ("prior", "current")]})
    return choice


def invalid_mark(context):
    choice = bridge_choice(context)
    choice["charts"][0]["mark"] = "line"
    return choice


def unknown_reference(context):
    return {"charts": [{"mark": "bar", "series": [{"entry_ids": ["E9999"]}]}]}


class TestLlmCharts:
    def test_valid_output_yields_llm_charts_whose_values_come_from_the_catalog(self, source, tmp_path, monkeypatch):
        model = ScriptedVisualizerModel(two_charts)
        result, _ = run_full(source, tmp_path, monkeypatch, model)
        artifacts = result["artifacts"]

        assert artifacts["chart_mode"] == "llm"
        assert model.calls == 1
        assert [chart.spec.mark for chart in artifacts["charts"]] == ["waterfall", "grouped_bar"]
        assert chart_property_violations(artifacts["charts"], artifacts["evidence_catalog"]) == []
        assert [ref.artifact_id for ref in artifacts["report"].charts] == [c.artifact_id for c in artifacts["charts"]]
        assert artifacts["report"].chart_mode == "llm"
        assert _run_metadata(result)["chart_mode"] == "llm"
        outcome = artifacts["step_outcomes"]["visualize_findings"]
        assert outcome["status"] == "completed" and outcome["chart_mode"] == "llm"
        assert "chart_failure" not in _run_metadata(result)

    def test_titles_and_series_names_come_from_code_never_the_model(self, source, tmp_path, monkeypatch):
        def with_invented_text(context):
            choice = two_charts(context)
            for chart in choice["charts"]:
                chart["title"] = "INVENTED TITLE"
                for series in chart["series"]:
                    series["name"] = "INVENTED SERIES"
            return choice

        model = ScriptedVisualizerModel(with_invented_text, two_charts)
        result, _ = run_full(source, tmp_path, monkeypatch, model)
        artifacts = result["artifacts"]
        catalog_labels = {e.label for e in artifacts["evidence_catalog"].entries}

        assert artifacts["chart_mode"] == "llm"
        for chart in artifacts["charts"]:
            assert chart.spec.title.strip() and "INVENTED" not in chart.spec.title
            assert all("INVENTED" not in s.name for s in chart.spec.series)
            assert all(point.label in catalog_labels for s in chart.series for point in s.points)

    def test_the_model_cannot_supply_data_values(self, source, tmp_path, monkeypatch):
        def with_data(context):
            choice = bridge_choice(context)
            choice["charts"][0]["series"][0]["values"] = [111.5, 222.5, 333.5]
            return choice

        model = ScriptedVisualizerModel(with_data, bridge_choice)
        result, _ = run_full(source, tmp_path, monkeypatch, model)

        assert result["artifacts"]["chart_mode"] == "llm"
        assert model.calls == 2
        assert "values" in model.contexts[1]["validation_feedback"]
        point_values = [p.value for c in result["artifacts"]["charts"] for s in c.series for p in s.points]
        assert not {111.5, 222.5, 333.5} & set(point_values)

    def test_the_model_sees_only_catalog_entries_and_the_question(self, source, tmp_path, monkeypatch):
        model = ScriptedVisualizerModel(bridge_choice)
        run_full(source, tmp_path, monkeypatch, model)

        context = model.contexts[0]
        assert context["user_question"] and context["evidence_catalog"]
        assert {"id", "label", "display", "source"} <= set(context["evidence_catalog"][0])


class TestRetryAndFallback:
    def test_invalid_output_gets_one_retry_with_feedback_then_is_used(self, source, tmp_path, monkeypatch):
        model = ScriptedVisualizerModel(invalid_mark, bridge_choice)
        result, _ = run_full(source, tmp_path, monkeypatch, model)
        artifacts = result["artifacts"]

        assert model.calls == 2
        assert not model.contexts[0]["validation_feedback"]
        assert "line" in model.contexts[1]["validation_feedback"]
        assert artifacts["chart_mode"] == "llm"
        assert artifacts["step_outcomes"]["visualize_findings"]["retry_counts"]["validation_retries"] == 1
        assert artifacts.get("chart_failure") is None

    @pytest.mark.parametrize("bad", [invalid_mark, unknown_reference,
                                     lambda c: {"charts": bridge_choice(c)["charts"] * 4}])
    def test_output_that_stays_invalid_falls_back_and_records_the_failure(self, source, tmp_path, monkeypatch, bad):
        model = ScriptedVisualizerModel(bad)
        result, _ = run_full(source, tmp_path, monkeypatch, model)
        artifacts = result["artifacts"]

        assert model.calls == 2
        assert artifacts["chart_mode"] == "fallback"
        assert [chart.chart_id for chart in artifacts["charts"]] == ["revenue_bridge", "top_drivers"]
        assert artifacts["report"].chart_mode == "fallback"
        outcome = artifacts["step_outcomes"]["visualize_findings"]
        assert outcome["status"] == "failed" and outcome["error"]
        assert outcome["retry_counts"]["validation_retries"] == 1
        assert artifacts["chart_failure"]["step_id"] == "visualize_findings"
        metadata = _run_metadata(result)
        assert metadata["chart_mode"] == "fallback" and metadata["chart_failure"]["error"]
        assert not artifacts.get("terminal_failure")
        assert artifacts["report_mode"] == "llm"

    def test_a_provider_failure_retries_boundedly_then_falls_back(self, source, tmp_path, monkeypatch):
        model = ScriptedVisualizerModel(ConnectionError("provider down"))
        result, _ = run_full(source, tmp_path, monkeypatch, model)
        artifacts = result["artifacts"]

        assert model.calls == 2
        assert artifacts["chart_mode"] == "fallback"
        outcome = artifacts["step_outcomes"]["visualize_findings"]
        assert outcome["status"] == "failed" and "provider down" in outcome["error"]
        assert outcome["retry_counts"]["tool_retries"] == 1
        assert "provider down" in artifacts["chart_failure"]["error"]

    def test_a_non_transient_provider_failure_is_terminal_at_once(self, source, tmp_path, monkeypatch):
        model = ScriptedVisualizerModel(PermissionError("bad key"))
        result, _ = run_full(source, tmp_path, monkeypatch, model)

        assert model.calls == 1
        assert result["artifacts"]["chart_mode"] == "fallback"
        assert result["artifacts"]["report"].charts

    def test_a_malformed_response_takes_the_validation_retry_path(self, source, tmp_path, monkeypatch):
        model = ScriptedVisualizerModel({"charts": "not a list"}, bridge_choice)
        result, _ = run_full(source, tmp_path, monkeypatch, model)

        assert model.calls == 2 and result["artifacts"]["chart_mode"] == "llm"


class TestResume:
    def test_an_unchanged_rerun_reuses_llm_charts_without_a_model_call(self, source, tmp_path, monkeypatch):
        saver, model = MemorySaver(), ScriptedVisualizerModel(bridge_choice)
        run_full(source, tmp_path, monkeypatch, model, checkpointer=saver)
        result, _ = run_full(source, tmp_path, monkeypatch, model, checkpointer=saver)

        assert model.calls == 1
        assert result["artifacts"]["chart_mode"] == "llm"
        assert result["artifacts"]["step_outcomes"]["visualize_findings"]["resume_outcome"] == "reused"

    def test_a_fallback_after_failure_is_retried_on_the_next_run(self, source, tmp_path, monkeypatch):
        saver = MemorySaver()
        run_full(source, tmp_path, monkeypatch, ScriptedVisualizerModel(ConnectionError("down")), checkpointer=saver)
        healed = ScriptedVisualizerModel(bridge_choice)
        result, _ = run_full(source, tmp_path, monkeypatch, healed, checkpointer=saver)

        assert healed.calls == 1
        assert result["artifacts"]["chart_mode"] == "llm"
        outcome = result["artifacts"]["step_outcomes"]["visualize_findings"]
        assert outcome["resume_outcome"] == "recomputed" and outcome["resume_reason"] == "failed"
        assert result["artifacts"].get("chart_failure") is None


class TestChartSpecValidationForModelOutput:
    @pytest.fixture
    def catalog(self):
        return build_evidence_catalog(make_findings())

    def content(self, catalog, **chart):
        ids = [e.id for e in catalog.entries if e.source.startswith("revenue_bridge.") and e.source.endswith("_effect")]
        return VisualizerContent.model_validate({"charts": [{"mark": "bar", "series": [{"entry_ids": ids}], **chart}]})

    def test_a_valid_choice_becomes_specs_with_code_derived_titles(self, catalog):
        specs = chart_specs_from_content(self.content(catalog), catalog)

        assert len(specs) == 1 and specs[0].mark == "bar" and specs[0].title.strip()
        assert specs[0].chart_id and specs[0].series[0].name.strip()

    def test_unknown_mark_unknown_reference_and_too_many_charts_are_rejected(self, catalog):
        with pytest.raises(ValueError, match="mark"):
            VisualizerContent.model_validate({"charts": [{"mark": "line", "series": [{"entry_ids": ["E1"]}]}]})
        with pytest.raises(ChartSpecError, match="E9999"):
            chart_specs_from_content(self.content(catalog, series=[{"entry_ids": ["E9999"]}]), catalog)
        too_many = VisualizerContent(charts=self.content(catalog).charts * 4)
        with pytest.raises(ChartSpecError, match="at most 3"):
            chart_specs_from_content(too_many, catalog)

    def test_a_value_field_is_rejected(self):
        with pytest.raises(ValueError):
            VisualizerContent.model_validate(
                {"charts": [{"mark": "bar", "series": [{"entry_ids": ["E1"], "values": [1]}]}]})

    def test_a_non_numeric_entry_is_rejected(self, catalog):
        mode = next(e.id for e in catalog.entries if e.source == "window.mode")
        with pytest.raises(ChartSpecError, match="numeric"):
            chart_specs_from_content(self.content(catalog, series=[{"entry_ids": [mode]}]), catalog)
