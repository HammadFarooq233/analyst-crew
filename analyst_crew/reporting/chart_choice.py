"""What the Visualizer's LLM may return, and how it becomes Chart Specs (ADR-0003).

The model *selects*: a mark and, per series, which Evidence Entry ids to plot. It supplies no values (unknown
fields are rejected, so a ``values`` or ``data`` field fails validation), no titles and no series names. Code
derives every title and series name from catalog labels and fills every value from the catalog, so nothing a
chart shows originates from model output. Validation reuses ``validate_chart_specs`` (closed marks, at most
three charts, every reference resolves to a numeric entry).
"""
from __future__ import annotations

from typing import Literal

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
