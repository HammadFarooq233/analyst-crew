"""Secondary seam (issue #13): the pure Findings -> Evidence Catalog function.

Tested directly with hand-built Findings because the catalog has to cover a
combinatorial shape (optional CPI and targets, empty dimensions, missing prior
baselines) that would be slow and indirect to reach through the graph.
Behavior only: stable ids, every reportable field citable, absent optional
sections yield no entries.
"""
from analyst_crew.reporting.catalog import build_evidence_catalog
from tests.findings_factory import cpi, driver, make_findings, target_comparison


def _display(catalog, source):
    return catalog.by_source(source).display


def test_same_findings_always_produce_the_same_entry_ids_and_content():
    first = build_evidence_catalog(make_findings(targets=target_comparison(), cpi_adjustment=cpi()))
    second = build_evidence_catalog(make_findings(targets=target_comparison(), cpi_adjustment=cpi()))

    assert first.model_dump() == second.model_dump()
    assert [entry.id for entry in first.entries] == [f"E{n}" for n in range(1, len(first.entries) + 1)]
    assert len({entry.source for entry in first.entries}) == len(first.entries)


def test_every_overall_metric_is_citable_with_current_prior_change_and_percent():
    catalog = build_evidence_catalog(make_findings())

    assert _display(catalog, "net_revenue.current") == "1,200.00"
    assert _display(catalog, "net_revenue.prior") == "1,000.00"
    assert _display(catalog, "net_revenue.absolute_change") == "+200.00"
    assert _display(catalog, "net_revenue.percent_change") == "+20.0%"
    assert _display(catalog, "orders.current") == "60"
    assert _display(catalog, "units.absolute_change") == "+20"
    assert _display(catalog, "gross_revenue.percent_change") == "+15.4%"
    assert catalog.by_source("net_revenue.current").value == 1200.0


def test_comparison_window_and_mode_are_citable_labels():
    catalog = build_evidence_catalog(make_findings())

    assert _display(catalog, "window.current_period") == "2025-06-01 to 2025-06-30"
    assert _display(catalog, "window.prior_period") == "2025-05-01 to 2025-05-31"
    assert "complete" in _display(catalog, "window.mode").lower()


def test_names_containing_digits_are_catalog_entries_so_they_can_be_cited():
    catalog = build_evidence_catalog(make_findings())

    names = [entry.display for entry in catalog.entries if entry.source.endswith(".name")]
    assert "Desk Lamp 2000" in names
    assert "North" in names


def test_every_dimension_driver_is_citable_and_material_ones_are_flagged():
    catalog = build_evidence_catalog(make_findings())

    north = [entry for entry in catalog.entries if entry.group == "region:North"]
    south = [entry for entry in catalog.entries if entry.group == "region:South"]
    assert {entry.display for entry in north} >= {"North", "700.00", "500.00", "+200.00", "+40.0%", "+100.0%"}
    assert south, "non-material drivers are still reportable"
    assert all(entry.material for entry in north)
    assert not any(entry.material for entry in south)
    assert _display(catalog, "material_changes.count") == "2"


def test_driver_without_prior_baseline_has_no_percent_entry_but_a_status_entry():
    catalog = build_evidence_catalog(make_findings())

    lamp = [entry for entry in catalog.entries if entry.group == "product:Desk Lamp 2000"]
    assert not any(entry.source.endswith(".percent_change") for entry in lamp)
    assert any("baseline" in entry.display.lower() for entry in lamp)


def test_revenue_bridge_effects_and_reconciliation_are_citable():
    catalog = build_evidence_catalog(make_findings())

    assert _display(catalog, "revenue_bridge.volume_effect") == "+150.00"
    assert _display(catalog, "revenue_bridge.discount_effect") == "-10.00"
    assert _display(catalog, "revenue_bridge.total_effect") == "+200.00"
    assert _display(catalog, "revenue_bridge.reconciliation_gap") == "0.00"
    assert _display(catalog, "revenue_bridge.new_product_count") == "1"


def test_absent_optional_sections_yield_no_entries():
    catalog = build_evidence_catalog(make_findings(targets=None, cpi_adjustment=None))

    assert not [entry for entry in catalog.entries if entry.source.startswith(("target_comparison", "cpi_adjustment"))]


def test_available_targets_and_cpi_are_citable_with_variance_and_non_causality_text():
    catalog = build_evidence_catalog(make_findings(targets=target_comparison(), cpi_adjustment=cpi()))

    row = [entry for entry in catalog.entries if entry.group == "target:north/office"]
    assert {entry.display for entry in row} >= {"north", "office", "900.00", "1,000.00", "-100.00", "-10.0%"}
    assert _display(catalog, "cpi_adjustment.inflation_adjusted_change") == "+150.00"
    assert _display(catalog, "cpi_adjustment.prior_cpi_value") == "300.000"
    assert "does not establish causation" in _display(catalog, "cpi_adjustment.non_causality_caveat")


def test_unavailable_optional_sources_produce_a_status_entry_and_no_figures():
    catalog = build_evidence_catalog(make_findings(
        targets=target_comparison("unavailable", "connection refused"),
        cpi_adjustment=cpi("not_configured", "FRED_API_KEY is not set"),
    ))

    assert "connection refused" in _display(catalog, "target_comparison.status")
    assert "unavailable" in _display(catalog, "target_comparison.status")
    assert "not_configured" in _display(catalog, "cpi_adjustment.status") or "not configured" in _display(catalog, "cpi_adjustment.status")
    assert not [entry for entry in catalog.entries if entry.source.startswith("target_comparison.rows")]
    assert not any(entry.source == "cpi_adjustment.inflation_adjusted_change" for entry in catalog.entries)


def test_data_quality_counts_and_reasons_are_citable():
    catalog = build_evidence_catalog(make_findings())

    assert _display(catalog, "data_quality.source_rows") == "1,000"
    assert _display(catalog, "data_quality.included_rows") == "996"
    assert _display(catalog, "data_quality.quarantined_rows") == "3"
    assert _display(catalog, "data_quality.duplicate_rows") == "1"
    assert _display(catalog, "data_quality.excluded_revenue_impact") == "1,234.50"
    assert any("malformed_row" in entry.label for entry in catalog.entries if entry.source.startswith("data_quality.quarantine_reasons"))


def test_findings_limitations_and_methodology_text_are_entries():
    catalog = build_evidence_catalog(make_findings(limitations=["Postgres sales-target comparison unavailable: down"]))

    displays = {entry.display for entry in catalog.entries}
    assert "Postgres sales-target comparison unavailable: down" in displays
    assert any("volume effect" in entry.display for entry in catalog.entries if entry.source == "revenue_bridge.methodology")
    assert any(entry.source.startswith("revenue_bridge.limitations") for entry in catalog.entries)
    assert _display(catalog, "methodology.materiality.relative_change_percent") == "15.0%"
    assert _display(catalog, "methodology.materiality.source") == "default"
    assert _display(catalog, "methodology.dataset_content_hash") == "f" * 64
    assert _display(catalog, "methodology.dataset_artifact_id") == "dataset-abc"


def test_findings_with_empty_dimensions_still_build():
    catalog = build_evidence_catalog(make_findings(dimensions={}, material_changes=[]))

    assert _display(catalog, "material_changes.count") == "0"
    assert not any(entry.source.startswith("dimensions") for entry in catalog.entries)


def test_two_drivers_sharing_a_name_still_get_unique_sources():
    twin_a = driver("product", "Widget", 100.0, 50.0)
    twin_b = driver("product", "Widget", 80.0, 40.0)
    catalog = build_evidence_catalog(make_findings(dimensions={"product": [twin_a, twin_b]}, material_changes=[]))

    sources = [entry.source for entry in catalog.entries]
    assert len(sources) == len(set(sources))
    assert sum(entry.display == "Widget" for entry in catalog.entries) == 2
