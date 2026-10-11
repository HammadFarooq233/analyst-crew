"""What the Visualizer's LLM may return, and how it becomes Chart Specs (ADR-0003).

The model *selects*: a mark and, per series, which Evidence Entry ids to plot. It supplies no values (unknown
fields are rejected, so a ``values`` or ``data`` field fails validation), no titles and no series names. Code
derives every title and series name from catalog labels and fills every value from the catalog, so nothing a
chart shows originates from model output. Validation reuses ``validate_chart_specs`` (closed marks, at most
three charts, every reference resolves to a numeric entry).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from analyst_crew.reporting.catalog import EvidenceCatalog, EvidenceEntry
from analyst_crew.reporting.charts import ChartSeries, ChartSpec, ChartSpecError, validate_chart_specs


class SeriesChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entry_ids: list[str]


class ChartChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mark: Literal["bar", "grouped_bar", "waterfall"]
    series: list[SeriesChoice]


class VisualizerContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    charts: list[ChartChoice]


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _common(values: list[str]) -> str | None:
    return values[0] if values and all(value == values[0] for value in values) else None


def series_name(entries: list[EvidenceEntry]) -> str:
    """A series name built only from catalog labels (and fixed text for the revenue bridge)."""
    if entries and all(entry.source.startswith("revenue_bridge.") for entry in entries):
        return "Revenue bridge effects"
    labels = [entry.label for entry in entries]
    tail = _common([label.split("': ", 1)[1] for label in labels if "': " in label]) if all("': " in l for l in labels) else None
    if tail:
        return _capitalize(tail)
    if all(", " in label for label in labels):
        tail = _common([label.rsplit(", ", 1)[1] for label in labels])
        heads = list(dict.fromkeys(label.rsplit(", ", 1)[0] for label in labels))
        if tail and len(heads) <= 3:
            return f"{' / '.join(heads)}, {tail}"
        if tail:
            return _capitalize(tail)
        head = _common([label.rsplit(", ", 1)[0] for label in labels])
        if head:
            return _capitalize(head)
    return labels[0] if labels else ""


def chart_title(series_entries: list[list[EvidenceEntry]]) -> str:
    title = " vs ".join(dict.fromkeys(series_name(entries) for entries in series_entries))
    dimensions = {
        entry.source.split(".")[1] for entries in series_entries for entry in entries
        if entry.source.startswith("dimensions.") and entry.source.count(".") >= 3
    }
    flat = [entry for entries in series_entries for entry in entries]
    if len(dimensions) == 1 and all(entry.source.startswith("dimensions.") for entry in flat):
        title += f" by {next(iter(dimensions))}"
    return title


_ITEM_FIELDS = (("prior", "prior period"), ("current", "current period"), ("absolute_change", "absolute change"))
_HEADLINE_MONEY = ("gross_revenue", "net_revenue")


def build_chart_menu(entries: list[Mapping[str, Any]]) -> str:
    """Ready-made id lists a model can pick from, built only from catalog ids and sources.

    Entries of one kind (every driver's prior-period net revenue, say) are listed in catalog order, so two lists
    for the same dimension pair up item for item and a grouped_bar built from them cannot have mismatched
    series lengths. The menu carries ids only; values and labels still come from the catalog at render time."""
    lines: list[str] = []

    bridge = [e["id"] for e in entries if e["source"].startswith("revenue_bridge.") and e["source"].endswith("_effect")]
    if bridge:
        lines.append(f"- Revenue bridge waterfall (one series: the effects in order, then the total): {', '.join(bridge)}")

    fields = dict(_ITEM_FIELDS)
    dimensions = dict.fromkeys(e["source"].split(".")[1] for e in entries if e["source"].startswith("dimensions."))
    for dimension in dimensions:
        rows: dict[str, dict[str, str]] = {}
        for entry in entries:
            parts = entry["source"].split(".")
            if parts[0] == "dimensions" and parts[1] == dimension and len(parts) >= 4 and parts[-1] in fields:
                rows.setdefault(".".join(parts[2:-1]), {})[parts[-1]] = entry["id"]
        rows = {key: row for key, row in rows.items() if len(row) == len(fields)}
        if not rows:
            continue
        lines.append(f"- Net revenue by {dimension.replace('_', ' ')}: {len(rows)} items in this order: "
                     f"{', '.join(rows)}. Each list below has exactly {len(rows)} entries, one per item in that order:")
        for field, text in _ITEM_FIELDS:
            lines.append(f"    {text}: {', '.join(row[field] for row in rows.values())}")
        lines.append("    prior vs current as a grouped_bar = two series, the prior-period list then the current-period "
                     "list; changes as a bar = the absolute change list (one series, or a leading slice of it)")

    headline = {
        period: [e["id"] for e in entries if e["source"] in {f"{m}.{period}" for m in _HEADLINE_MONEY}]
        for period in ("prior", "current")
    }
    if headline["prior"] and len(headline["prior"]) == len(headline["current"]):
        lines.append("- Gross vs net revenue (money only; do not add orders or units): prior period "
                     f"{', '.join(headline['prior'])}; current period {', '.join(headline['current'])}")

    actual = [e["id"] for e in entries if e["source"].endswith(".actual_net_revenue")]
    target = [e["id"] for e in entries if e["source"].endswith(".target_net_revenue")]
    variance = [e["id"] for e in entries if e["source"].endswith(".net_revenue_variance")]
    if actual and len(actual) == len(target):
        lines.append(f"- Sales targets ({len(actual)} rows, same order in each list): actual net revenue "
                     f"{', '.join(actual)}; target net revenue {', '.join(target)}; variance to target "
                     f"{', '.join(variance)}")
    return "\n".join(lines)


def chart_specs_from_content(content: VisualizerContent, catalog: EvidenceCatalog) -> list[ChartSpec]:
    """Validated Chart Specs for a model's choices; raises ``ChartSpecError`` listing every problem."""
    if not content.charts:
        raise ChartSpecError(["choose at least one chart"])
    specs = []
    for index, choice in enumerate(content.charts, 1):
        series_entries = [
            [entry for entry in (catalog.get(entry_id) for entry_id in series.entry_ids) if entry is not None]
            for series in choice.series
        ]
        specs.append(ChartSpec(
            chart_id=f"chart_{index}", mark=choice.mark,
            title=chart_title([e for e in series_entries if e]) or f"Chart {index}",
            series=[
                ChartSeries(name=series_name(entries) or f"Series {number}", entry_ids=choice_series.entry_ids)
                for number, (entries, choice_series) in enumerate(zip(series_entries, choice.series), 1)
            ],
        ))
    return validate_chart_specs(specs, catalog)
