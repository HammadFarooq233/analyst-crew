"""Evidence Catalog: a deterministic flattening of Findings into citable entries (ADR-0002).

Pure code, no model. The Reporter selects from the catalog; it never computes,
copies or formats a value. Ids are sequential over a fixed traversal of the
Findings, so the same Findings always produce the same ids.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from analyst_crew.tools.analysis import BaselineFindings, DimensionDriver, MetricComparison

WINDOW_MODE_DISPLAY = {
    "complete_month": "the latest two complete calendar months",
    "comparable_mtd": "month-to-date against the equivalent number of days in the preceding month",
}

_METRICS = (
    ("gross_revenue", "Gross revenue", "money"),
    ("net_revenue", "Net revenue", "money"),
    ("orders", "Orders", "count"),
    ("units", "Units", "count"),
)

_BRIDGE_EFFECTS = (
    ("volume_effect", "volume effect"),
    ("price_effect", "unit-price effect"),
    ("interaction_effect", "price-volume interaction effect"),
    ("discount_effect", "discount effect"),
    ("return_effect", "return effect"),
    ("new_product_effect", "new-product effect"),
    ("discontinued_product_effect", "discontinued-product effect"),
    ("total_effect", "total of all effects"),
)

_BRIDGE_COUNTS = (
    ("continuing_product_count", "Continuing products"),
    ("new_product_count", "New products"),
    ("discontinued_product_count", "Discontinued products"),
)


def money(value: float) -> str:
    return f"{value:,.2f}"


def signed_money(value: float) -> str:
    return f"{value:+,.2f}"


def count(value: float) -> str:
    return f"{value:,.0f}"


def signed_count(value: float) -> str:
    return f"{value:+,.0f}"


def signed_percent(value: float) -> str:
    return f"{value:+.1f}%"


def percent(value: float) -> str:
    return f"{value:.1f}%"


class EvidenceEntry(BaseModel):
    """One citable fact: a stable id, its raw value, its display rendering and
    the Findings field it came from. ``group`` ties together the entries that
    describe one driver or target row; ``material`` marks Material Change
    drivers."""

    id: str
    label: str
    value: float | int | str | bool | None
    display: str
    source: str
    group: str | None = None
    material: bool = False

    @property
    def is_numeric(self) -> bool:
        """True when ``value`` is a plottable number (a bool is not)."""
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)


class EvidenceCatalog(BaseModel):
    entries: list[EvidenceEntry]

    @property
    def ids(self) -> set[str]:
        return {entry.id for entry in self.entries}

    def get(self, entry_id: str) -> EvidenceEntry | None:
        return next((entry for entry in self.entries if entry.id == entry_id), None)

    def by_source(self, source: str) -> EvidenceEntry:
        for entry in self.entries:
            if entry.source == source:
                return entry
        raise KeyError(source)

    def find_source(self, source: str) -> EvidenceEntry | None:
        return next((entry for entry in self.entries if entry.source == source), None)

    def group(self, group: str) -> list[EvidenceEntry]:
        return [entry for entry in self.entries if entry.group == group]


class _Builder:
    def __init__(self) -> None:
        self.entries: list[EvidenceEntry] = []
        self._sources: set[str] = set()

    def unique(self, key: str, taken: set[str]) -> str:
        candidate, suffix = key, 1
        while candidate in taken:
            suffix += 1
            candidate = f"{key}~{suffix}"
        taken.add(candidate)
        return candidate

    def add(self, label: str, value: Any, display: str, source: str, *, group: str | None = None,
            material: bool = False) -> None:
        if source in self._sources:
            raise ValueError(f"Duplicate Evidence Entry source: {source}")
        self._sources.add(source)
        self.entries.append(EvidenceEntry(
            id=f"E{len(self.entries) + 1}", label=label, value=value, display=display,
            source=source, group=group, material=material,
        ))

    def metric(self, source: str, label: str, comparison: MetricComparison, kind: str) -> None:
        plain, signed = (money, signed_money) if kind == "money" else (count, signed_count)
        self.add(f"{label}, current period", comparison.current, plain(comparison.current), f"{source}.current")
        self.add(f"{label}, prior period", comparison.prior, plain(comparison.prior), f"{source}.prior")
        self.add(f"{label}, absolute change", comparison.absolute_change,
                 signed(comparison.absolute_change), f"{source}.absolute_change")
        if comparison.percent_change is not None:
            self.add(f"{label}, percent change", comparison.percent_change,
                     signed_percent(comparison.percent_change), f"{source}.percent_change")

    def driver(self, key: str, driver: DimensionDriver) -> None:
        dimension = driver.dimension.replace("_", " ")
        group, prefix = f"{driver.dimension}:{key}", f"dimensions.{driver.dimension}.{key}"
        material = driver.is_material
        label = f"{dimension.capitalize()} '{driver.value}'"

        def add(suffix: str, what: str, value: Any, display: str) -> None:
            self.add(f"{label}: {what}", value, display, f"{prefix}.{suffix}", group=group, material=material)

        add("name", f"{dimension} name", driver.value, driver.value)
        add("current", "net revenue, current period", driver.current, money(driver.current))
        add("prior", "net revenue, prior period", driver.prior, money(driver.prior))
        add("absolute_change", "net revenue absolute change", driver.absolute_change,
            signed_money(driver.absolute_change))
        if driver.percent_change is not None:
            add("percent_change", "net revenue percent change", driver.percent_change,
                signed_percent(driver.percent_change))
        if driver.prior_baseline_missing:
            add("baseline_status", "prior-period baseline", "missing",
                "no prior-period baseline (percent change undefined)")
        if driver.contribution_percent is not None:
            add("contribution_percent", "share of the overall net revenue change", driver.contribution_percent,
                signed_percent(driver.contribution_percent))


def build_evidence_catalog(findings: BaselineFindings) -> EvidenceCatalog:
    builder = _Builder()
    add = builder.add

    window = findings.window
    add("Comparison mode", window.mode, WINDOW_MODE_DISPLAY[window.mode], "window.mode")
    add("Current period", f"{window.current_start.isoformat()}/{window.current_end.isoformat()}",
        f"{window.current_start.isoformat()} to {window.current_end.isoformat()}", "window.current_period")
    add("Prior period", f"{window.prior_start.isoformat()}/{window.prior_end.isoformat()}",
        f"{window.prior_start.isoformat()} to {window.prior_end.isoformat()}", "window.prior_period")

    for attribute, label, kind in _METRICS:
        builder.metric(attribute, label, getattr(findings, attribute), kind)

    add("Number of Material Changes", len(findings.material_changes), str(len(findings.material_changes)),
        "material_changes.count")

    for dimension, drivers in findings.dimensions.items():
        taken: set[str] = set()
        for driver in drivers:
            builder.driver(builder.unique(driver.value, taken), driver)

    bridge = findings.revenue_bridge
    if bridge is not None:
        for attribute, label in _BRIDGE_EFFECTS:
            value = getattr(bridge, attribute)
            add(f"Revenue bridge: {label}", value, signed_money(value), f"revenue_bridge.{attribute}")
        add("Revenue bridge: reconciliation gap to the observed net revenue change", bridge.reconciliation_gap,
            money(bridge.reconciliation_gap), "revenue_bridge.reconciliation_gap")
        for attribute, label in _BRIDGE_COUNTS:
            value = getattr(bridge, attribute)
            add(f"Revenue bridge: {label.lower()} count", value, count(value), f"revenue_bridge.{attribute}")

    targets = findings.target_comparison
    if targets is not None:
        status = targets.source_status
        add("Sales-target comparison source status", status.status,
            status.status + (f": {status.detail}" if status.detail else ""), "target_comparison.status")
        taken = set()
        for row in targets.rows:
            key = builder.unique(f"{row.region}/{row.category}", taken)
            group, prefix = f"target:{key}", f"target_comparison.rows.{key}"
            label = f"Target row {row.region}/{row.category}"

            def add_row(suffix: str, what: str, value: Any, display: str,
                        group=group, prefix=prefix, label=label) -> None:
                add(f"{label}: {what}", value, display, f"{prefix}.{suffix}", group=group)

            add_row("region", "region", row.region, row.region)
            add_row("category", "category", row.category, row.category)
            add_row("actual_net_revenue", "actual net revenue", row.actual_net_revenue, money(row.actual_net_revenue))
            add_row("target_net_revenue", "target net revenue", row.target_net_revenue, money(row.target_net_revenue))
            add_row("net_revenue_variance", "net revenue variance to target", row.net_revenue_variance,
                    signed_money(row.net_revenue_variance))
            if row.net_revenue_variance_percent is not None:
                add_row("net_revenue_variance_percent", "net revenue variance to target, percent",
                        row.net_revenue_variance_percent, signed_percent(row.net_revenue_variance_percent))
            add_row("actual_orders", "actual orders", row.actual_orders, count(row.actual_orders))
            add_row("target_orders", "target orders", row.target_orders, count(row.target_orders))
            add_row("orders_variance", "orders variance to target", row.orders_variance,
                    signed_count(row.orders_variance))
            if row.orders_variance_percent is not None:
                add_row("orders_variance_percent", "orders variance to target, percent",
                        row.orders_variance_percent, signed_percent(row.orders_variance_percent))

    cpi = findings.cpi_adjustment
    if cpi is not None:
        status = cpi.source_status
        add("CPI inflation-adjustment source status", status.status,
            status.status + (f": {status.detail}" if status.detail else ""), "cpi_adjustment.status")
        if status.status == "available":
            add("CPI series", cpi.series_id, cpi.series_id, "cpi_adjustment.series_id")
            add("Nominal net revenue change", cpi.nominal_change, signed_money(cpi.nominal_change),
                "cpi_adjustment.nominal_change")
            if cpi.inflation_adjusted_change is not None:
                add("Approximate inflation-adjusted net revenue change", cpi.inflation_adjusted_change,
                    signed_money(cpi.inflation_adjusted_change), "cpi_adjustment.inflation_adjusted_change")
            if cpi.prior_cpi_value is not None:
                add("CPI, prior period", cpi.prior_cpi_value, f"{cpi.prior_cpi_value:.3f}",
                    "cpi_adjustment.prior_cpi_value")
            if cpi.current_cpi_value is not None:
                add("CPI, current period", cpi.current_cpi_value, f"{cpi.current_cpi_value:.3f}",
                    "cpi_adjustment.current_cpi_value")
            if cpi.prior_cpi_observation_dates:
                dates = ", ".join(cpi.prior_cpi_observation_dates)
                add("CPI observation dates, prior period", dates, dates, "cpi_adjustment.prior_cpi_observation_dates")
            if cpi.current_cpi_observation_dates:
                dates = ", ".join(cpi.current_cpi_observation_dates)
                add("CPI observation dates, current period", dates, dates,
                    "cpi_adjustment.current_cpi_observation_dates")
            add("CPI adjustment methodology", cpi.methodology, cpi.methodology, "cpi_adjustment.methodology")
            add("CPI non-causality caveat", cpi.non_causality_caveat, cpi.non_causality_caveat,
                "cpi_adjustment.non_causality_caveat")

    add("Source rows", findings.source_rows, count(findings.source_rows), "data_quality.source_rows")
    add("Included rows", findings.included_rows, count(findings.included_rows), "data_quality.included_rows")
    add("Quarantined rows", findings.quarantined_rows, count(findings.quarantined_rows),
        "data_quality.quarantined_rows")
    add("Duplicate rows excluded", findings.duplicate_rows, count(findings.duplicate_rows),
        "data_quality.duplicate_rows")
    add("Calculable gross revenue impact of excluded rows", findings.excluded_revenue_impact,
        money(findings.excluded_revenue_impact), "data_quality.excluded_revenue_impact")
    for reason, reason_count in sorted(findings.quarantine_reasons.items()):
        add(f"Quarantined rows: reason '{reason}'", reason_count, count(reason_count),
            f"data_quality.quarantine_reasons.{reason}")

    for index, limitation in enumerate(findings.limitations):
        add("Analysis limitation", limitation, limitation, f"limitations.{index}")
    if bridge is not None:
        add("Revenue bridge methodology", bridge.methodology, bridge.methodology, "revenue_bridge.methodology")
        for index, limitation in enumerate(bridge.limitations):
            add("Revenue bridge limitation", limitation, limitation, f"revenue_bridge.limitations.{index}")

    thresholds = findings.materiality
    add("Material Change relative-change threshold", thresholds.relative_change_percent,
        percent(thresholds.relative_change_percent), "methodology.materiality.relative_change_percent")
    add("Material Change absolute-impact threshold, as a share of prior-period net revenue",
        thresholds.absolute_change_percent_of_prior_total,
        percent(thresholds.absolute_change_percent_of_prior_total),
        "methodology.materiality.absolute_change_percent_of_prior_total")
    add("Material Change threshold source", thresholds.source, thresholds.source, "methodology.materiality.source")
    add("Dataset Artifact id", findings.dataset_artifact_id, findings.dataset_artifact_id,
        "methodology.dataset_artifact_id")
    add("Dataset Artifact content hash", findings.dataset_content_hash, findings.dataset_content_hash,
        "methodology.dataset_content_hash")
    add("Dataset Artifact integrity check", findings.integrity_verified,
        "verified" if findings.integrity_verified else "not verified", "methodology.integrity_verified")

    return EvidenceCatalog(entries=builder.entries)
