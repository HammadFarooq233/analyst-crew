"""Issue #15 — Chart Spec validation, the deterministic data fill, the fallback
chart set and PNG rendering. Pure code: no model, no Crew graph."""
import pytest
from pydantic import ValidationError

from analyst_crew.reporting.catalog import build_evidence_catalog
from analyst_crew.reporting.charts import (
    MAX_CHARTS, ChartSeries, ChartSpec, ChartSpecError, build_chart_artifacts, build_fallback_chart_specs,
    fill_chart_data, validate_chart_specs,
)
from tests.findings_factory import driver, make_findings, target_comparison

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def catalog_for(**kwargs):
    return build_evidence_catalog(make_findings(**kwargs))


def ids(catalog, *sources):
    return [catalog.by_source(source).id for source in sources]


def bar(catalog, *sources, chart_id="c1"):
    return ChartSpec(chart_id=chart_id, mark="bar", title="A chart",
                     series=[ChartSeries(name="s", entry_ids=ids(catalog, *sources))])


class TestFallbackChartSet:
    def test_is_the_revenue_bridge_waterfall_then_top_drivers_when_no_targets(self):
        catalog = catalog_for()

        specs = build_fallback_chart_specs(catalog)

        assert [spec.chart_id for spec in specs] == ["revenue_bridge", "top_drivers"]
        assert [spec.mark for spec in specs] == ["waterfall", "bar"]

    def test_target_variance_is_added_only_when_targets_are_available(self):
        with_targets = build_fallback_chart_specs(catalog_for(targets=target_comparison()))
        unavailable = build_fallback_chart_specs(catalog_for(targets=target_comparison(status="unavailable", detail="down")))

        assert [spec.chart_id for spec in with_targets] == ["revenue_bridge", "top_drivers", "target_variance"]
        assert [spec.chart_id for spec in unavailable] == ["revenue_bridge", "top_drivers"]

    def test_waterfall_plots_every_bridge_effect_then_the_total(self):
        catalog = catalog_for()

        waterfall = build_fallback_chart_specs(catalog)[0]

        assert waterfall.series[0].entry_ids == ids(
            catalog, "revenue_bridge.volume_effect", "revenue_bridge.price_effect",
            "revenue_bridge.interaction_effect", "revenue_bridge.discount_effect", "revenue_bridge.return_effect",
            "revenue_bridge.new_product_effect", "revenue_bridge.discontinued_product_effect",
            "revenue_bridge.total_effect",
        )

    def test_top_drivers_are_the_material_drivers_largest_absolute_change_first(self):
        catalog = catalog_for()

        top_drivers = build_fallback_chart_specs(catalog)[1]

        # Desk Lamp 2000 changed by +300.00 and North by +200.00; South (no change, not material) is left out.
        assert top_drivers.series[0].entry_ids == ids(
            catalog, "dimensions.product.Desk Lamp 2000.absolute_change", "dimensions.region.North.absolute_change",
        )

    def test_top_drivers_fall_back_to_all_drivers_when_none_is_material(self):
        quiet = {"region": [driver("region", "North", 510.0, 500.0), driver("region", "South", 500.0, 530.0)]}
        catalog = catalog_for(dimensions=quiet)

        top_drivers = build_fallback_chart_specs(catalog)[1]

        assert top_drivers.series[0].entry_ids == ids(
            catalog, "dimensions.region.South.absolute_change", "dimensions.region.North.absolute_change",
        )

    def test_fallback_specs_are_valid_chart_specs(self):
        catalog = catalog_for(targets=target_comparison())

        specs = build_fallback_chart_specs(catalog)

        assert validate_chart_specs(specs, catalog) == specs
        assert len(specs) <= MAX_CHARTS


class TestDataFill:
    def test_every_value_is_the_evidence_entry_raw_value_and_every_label_the_catalog_label(self):
        catalog = catalog_for()
        waterfall = build_fallback_chart_specs(catalog)[0]

        [series] = fill_chart_data(waterfall, catalog)

        assert [point.value for point in series.points] == [150.0, 30.0, 5.0, -10.0, -5.0, 40.0, -10.0, 200.0]
        assert series.points[0].label == "Revenue bridge: volume effect"
        assert series.points[0].display == "+150.00"
        assert series.points[0].entry_id == catalog.by_source("revenue_bridge.volume_effect").id


class TestChartSpecValidation:
    def test_unknown_mark_is_rejected(self):
        with pytest.raises(ValidationError):
            ChartSpec(chart_id="c", mark="line", title="t", series=[ChartSeries(name="s", entry_ids=["E1"])])

    def test_series_reference_that_is_not_in_the_catalog_is_rejected(self):
        catalog = catalog_for()
        spec = ChartSpec(chart_id="c", mark="bar", title="t", series=[ChartSeries(name="s", entry_ids=["E99999"])])

        with pytest.raises(ChartSpecError, match="E99999"):
            validate_chart_specs([spec], catalog)

    def test_more_than_three_charts_is_rejected(self):
        catalog = catalog_for()
        specs = [bar(catalog, "net_revenue.current", chart_id=f"c{n}") for n in range(MAX_CHARTS + 1)]

        with pytest.raises(ChartSpecError, match="at most"):
            validate_chart_specs(specs, catalog)

    def test_exactly_three_charts_is_accepted(self):
        catalog = catalog_for()
        specs = [bar(catalog, "net_revenue.current", chart_id=f"c{n}") for n in range(MAX_CHARTS)]

        assert validate_chart_specs(specs, catalog) == specs

    def test_entry_without_a_numeric_value_cannot_be_plotted(self):
        catalog = catalog_for()

        with pytest.raises(ChartSpecError, match="numeric"):
            validate_chart_specs([bar(catalog, "window.mode")], catalog)

    def test_duplicate_chart_ids_and_empty_series_are_rejected(self):
        catalog = catalog_for()
        empty = ChartSpec(chart_id="c2", mark="bar", title="t", series=[ChartSeries(name="s", entry_ids=[])])

        with pytest.raises(ChartSpecError, match="duplicate"):
            validate_chart_specs([bar(catalog, "net_revenue.current"), bar(catalog, "net_revenue.prior")], catalog)
        with pytest.raises(ChartSpecError, match="no Evidence Entries"):
            validate_chart_specs([empty], catalog)

    def test_grouped_bar_needs_series_of_equal_length(self):
        catalog = catalog_for()
        spec = ChartSpec(chart_id="g", mark="grouped_bar", title="t", series=[
            ChartSeries(name="a", entry_ids=ids(catalog, "net_revenue.current", "net_revenue.prior")),
            ChartSeries(name="b", entry_ids=ids(catalog, "net_revenue.current")),
        ])

        with pytest.raises(ChartSpecError, match="same number"):
            validate_chart_specs([spec], catalog)

    def test_unequal_grouped_bar_feedback_names_the_lengths_and_the_series(self):
        catalog = catalog_for()
        spec = ChartSpec(chart_id="chart_3", mark="grouped_bar", title="t", series=[
            ChartSeries(name="a", entry_ids=ids(catalog, "net_revenue.current", "net_revenue.prior")),
            ChartSeries(name="b", entry_ids=ids(catalog, "net_revenue.current")),
        ])

        with pytest.raises(ChartSpecError) as caught:
            validate_chart_specs([spec], catalog)

        message = str(caught.value)
        assert "chart_3" in message and "series 1 has 2" in message and "series 2 has 1" in message
        assert all(entry_id in message for entry_id in spec.series[0].entry_ids)
        assert "pair" in message


class TestRendering:
    def test_same_findings_give_identical_chart_artifact_hashes(self, tmp_path):
        catalog = catalog_for(targets=target_comparison())
        specs = build_fallback_chart_specs(catalog)

        first = build_chart_artifacts(specs, catalog, tmp_path / "one")
        second = build_chart_artifacts(specs, catalog_for(targets=target_comparison()), tmp_path / "two")

        assert [a.content_hash for a in first] == [a.content_hash for a in second]
        assert len({a.content_hash for a in first}) == 3

    def test_artifacts_are_content_addressed_png_files_carrying_spec_and_data(self, tmp_path):
        catalog = catalog_for()
        specs = build_fallback_chart_specs(catalog)

        artifacts = build_chart_artifacts(specs, catalog, tmp_path)

        for artifact, spec in zip(artifacts, specs):
            png = (tmp_path / "charts" / artifact.filename).read_bytes()
            assert png.startswith(PNG_SIGNATURE)
            assert artifact.artifact_id == f"chart-{artifact.content_hash[:16]}"
            assert artifact.filename == f"{artifact.artifact_id}.png"
            assert artifact.spec == spec
            assert artifact.series == fill_chart_data(spec, catalog)

    def test_changed_findings_change_the_hash(self, tmp_path):
        base = catalog_for()
        changed = build_evidence_catalog(make_findings(dimensions={
            "region": [driver("region", "North", 900.0, 500.0, material=True)],
        }))

        before = build_chart_artifacts(build_fallback_chart_specs(base), base, tmp_path)
        after = build_chart_artifacts(build_fallback_chart_specs(changed), changed, tmp_path)

        assert before[1].content_hash != after[1].content_hash

    def test_invalid_specs_are_never_rendered(self, tmp_path):
        catalog = catalog_for()
        bad = ChartSpec(chart_id="c", mark="bar", title="t", series=[ChartSeries(name="s", entry_ids=["E99999"])])

        with pytest.raises(ChartSpecError):
            build_chart_artifacts([bad], catalog, tmp_path)
        assert not (tmp_path / "charts").exists()
