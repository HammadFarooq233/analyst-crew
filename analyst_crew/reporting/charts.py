"""Chart Spec, Chart Artifact, the deterministic fallback chart set and PNG rendering (ADR-0003).

A Chart Spec only *selects*: a mark, a title and which Evidence Entries to plot.
Code fills every value from the Evidence Catalog (``fill_chart_data``) and renders
it, so a chart can never contain a number that is not a Tool value. Marks are a
closed set and a Report has at most ``MAX_CHARTS`` charts. Everything here is
deterministic: the same Evidence Catalog produces byte-identical PNGs, and a
Chart Artifact is content-addressed by those bytes like the Dataset Artifact.
"""
from __future__ import annotations

import hashlib
import io
import textwrap
from pathlib import Path
from typing import Any, Literal

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from analyst_crew.reporting.catalog import EvidenceCatalog, EvidenceEntry

MAX_CHARTS = 3
MAX_BARS = 8

BLUE, ORANGE, GREY = "#1f77b4", "#d95f02", "#4d4d4d"


class ChartSeries(BaseModel):
    """One plotted series: the Evidence Entries whose raw values are drawn, in order."""

    name: str
    entry_ids: list[str]


class ChartSpec(BaseModel):
    """What to plot. Never carries a value: code resolves ``entry_ids`` against the catalog.

    Marks: ``bar`` (one series), ``grouped_bar`` (two or more equally long series, categories
    labelled from the first series) and ``waterfall`` (one series; every entry but the last is a
    step, the last is the total drawn from zero)."""

    chart_id: str
    mark: Literal["bar", "grouped_bar", "waterfall"]
    title: str
    series: list[ChartSeries]


class ChartPoint(BaseModel):
    entry_id: str
    label: str
    value: float
    display: str


class ChartSeriesData(BaseModel):
    name: str
    points: list[ChartPoint]


class ChartReference(BaseModel):
    """What a Report's Charts section holds: a pointer to a Chart Artifact, never the image."""

    chart_id: str
    artifact_id: str
    content_hash: str
    title: str
    filename: str


class ChartArtifact(BaseModel):
    """A durable, content-addressed PNG of one Chart Spec, with the spec and the data it drew."""

    chart_id: str
    artifact_id: str
    content_hash: str
    filename: str
    location: str
    spec: ChartSpec
    series: list[ChartSeriesData]

    def reference(self) -> ChartReference:
        return ChartReference(chart_id=self.chart_id, artifact_id=self.artifact_id,
                              content_hash=self.content_hash, title=self.spec.title, filename=self.filename)


def load_charts(raw: Any) -> list[ChartArtifact]:
    """Chart Artifacts from state, which may hold models or their dict form."""
    return [chart if isinstance(chart, ChartArtifact) else ChartArtifact.model_validate(chart) for chart in raw or []]


def chart_set_reference(charts: list[ChartArtifact]) -> dict[str, Any]:
    """The Handoff form of a chart set: ids, hashes and titles only, never an image or a location."""
    return {
        "kind": "Chart Artifacts",
        "artifact_ids": [chart.artifact_id for chart in charts],
        "content_hashes": [chart.content_hash for chart in charts],
        "charts": [{"chart_id": chart.chart_id, "title": chart.spec.title} for chart in charts],
    }


class ChartSpecError(ValueError):
    """Chart Specs that break the closed vocabulary or the catalog references; carries every
    problem found so a retry can be given complete feedback."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def validate_chart_specs(specs: list[ChartSpec], catalog: EvidenceCatalog) -> list[ChartSpec]:
    errors: list[str] = []
    if len(specs) > MAX_CHARTS:
        errors.append(f"a Report has at most {MAX_CHARTS} charts, got {len(specs)}")
    seen: set[str] = set()
    for spec in specs:
        name = f"Chart '{spec.chart_id}'"
        if spec.chart_id in seen:
            errors.append(f"{name} is a duplicate chart id")
        seen.add(spec.chart_id)
        if not spec.title.strip():
            errors.append(f"{name} has no title")
        if not spec.series:
            errors.append(f"{name} has no series")
        for series in spec.series:
            if not series.entry_ids:
                errors.append(f"{name} series '{series.name}' has no Evidence Entries")
            for entry_id in series.entry_ids:
                entry = catalog.get(entry_id)
                if entry is None:
                    errors.append(f"{name} references '{entry_id}', which is not an Evidence Entry id")
                elif not entry.is_numeric:
                    errors.append(f"{name} references '{entry_id}', which has no numeric value to plot")
        if spec.mark in ("bar", "waterfall") and len(spec.series) > 1:
            errors.append(f"{name} mark '{spec.mark}' takes exactly one series")
        if spec.mark == "waterfall" and spec.series and len(spec.series[0].entry_ids) < 2:
            errors.append(f"{name} waterfall needs at least two entries (steps then a total)")
        if spec.mark == "grouped_bar":
            if len(spec.series) < 2:
                errors.append(f"{name} grouped_bar needs at least two series")
            if len({len(series.entry_ids) for series in spec.series}) > 1:
                described = "; ".join(
                    f"series {number} has {len(series.entry_ids)} ({', '.join(series.entry_ids)})"
                    for number, series in enumerate(spec.series, 1)
                )
                errors.append(
                    f"{name} grouped_bar series must have the same number of entries: {described}. Give every "
                    f"series exactly the same number of entries, pairing them item for item (for example item 1's "
                    f"prior-period entry in one series and item 1's current-period entry in the other, in the same "
                    f"order); drop the unpaired entries from the longer series, or use a bar chart instead"
                )
    if errors:
        raise ChartSpecError(errors)
    return specs


def fill_chart_data(spec: ChartSpec, catalog: EvidenceCatalog) -> list[ChartSeriesData]:
    """Every plotted value is an Evidence Entry raw value; every label is its catalog label."""
    data = []
    for series in spec.series:
        points = []
        for entry_id in series.entry_ids:
            entry = catalog.get(entry_id)
            if entry is None or not entry.is_numeric:
                raise ChartSpecError([f"'{entry_id}' is not a plottable Evidence Entry"])
            points.append(ChartPoint(entry_id=entry.id, label=entry.label, value=float(entry.value),
                                     display=entry.display))
        data.append(ChartSeriesData(name=series.name, points=points))
    return data


# --- deterministic fallback chart set --------------------------------------------------------

_BRIDGE_SOURCES = tuple(f"revenue_bridge.{name}" for name in (
    "volume_effect", "price_effect", "interaction_effect", "discount_effect", "return_effect",
    "new_product_effect", "discontinued_product_effect", "total_effect",
))


def _largest_by_magnitude(entries: list[EvidenceEntry]) -> list[EvidenceEntry]:
    return sorted(entries, key=lambda entry: -abs(float(entry.value)))[:MAX_BARS]


def build_fallback_chart_specs(catalog: EvidenceCatalog) -> list[ChartSpec]:
    """The revenue bridge waterfall and top drivers bar, plus target variance when targets are
    available. Built purely from the catalog, with fixed titles; no model judgment."""
    specs: list[ChartSpec] = []

    bridge = [catalog.find_source(source) for source in _BRIDGE_SOURCES]
    if all(entry is not None for entry in bridge):
        specs.append(ChartSpec(
            chart_id="revenue_bridge", mark="waterfall", title="Net revenue bridge: prior to current period",
            series=[ChartSeries(name="Revenue bridge effects", entry_ids=[entry.id for entry in bridge])],
        ))

    changes = [entry for entry in catalog.entries
               if entry.source.startswith("dimensions.") and entry.source.endswith(".absolute_change")]
    drivers = [entry for entry in changes if entry.material] or changes
    if drivers:
        specs.append(ChartSpec(
            chart_id="top_drivers", mark="bar", title="Top drivers of the net revenue change",
            series=[ChartSeries(name="Net revenue absolute change",
                                entry_ids=[entry.id for entry in _largest_by_magnitude(drivers)])],
        ))

    status = catalog.find_source("target_comparison.status")
    variances = [entry for entry in catalog.entries
                 if entry.source.startswith("target_comparison.rows.") and entry.source.endswith(".net_revenue_variance")]
    if status is not None and status.value == "available" and variances:
        specs.append(ChartSpec(
            chart_id="target_variance", mark="bar", title="Net revenue variance to target",
            series=[ChartSeries(name="Net revenue variance to target",
                                entry_ids=[entry.id for entry in _largest_by_magnitude(variances)])],
        ))
    return specs


# --- rendering --------------------------------------------------------------------------------

def _wrap(label: str, width: int = 28) -> str:
    return "\n".join(textwrap.wrap(label, width)) or label


def _draw_bar(ax, data: list[ChartSeriesData]) -> None:
    points = data[0].points
    positions = list(range(len(points)))
    ax.barh(positions, [p.value for p in points], color=[BLUE if p.value >= 0 else ORANGE for p in points])
    ax.set_yticks(positions, [_wrap(p.label) for p in points])
    ax.invert_yaxis()
    ax.axvline(0, color=GREY, linewidth=0.8)
    for position, point in zip(positions, points):
        ax.annotate(point.display, (point.value, position), xytext=(4 if point.value >= 0 else -4, 0),
                    textcoords="offset points", va="center", ha="left" if point.value >= 0 else "right", fontsize=8)
    ax.margins(x=0.25)


def _draw_grouped_bar(ax, data: list[ChartSeriesData]) -> None:
    count = len(data)
    height = 0.8 / count
    palette = [BLUE, ORANGE, GREY, "#7570b3"]
    for index, series in enumerate(data):
        offsets = [position + index * height for position in range(len(series.points))]
        ax.barh(offsets, [p.value for p in series.points], height=height, color=palette[index % len(palette)],
                label=series.name)
        for offset, point in zip(offsets, series.points):
            ax.annotate(point.display, (point.value, offset), xytext=(4 if point.value >= 0 else -4, 0),
                        textcoords="offset points", va="center", ha="left" if point.value >= 0 else "right",
                        fontsize=7)
    ax.set_yticks([position + 0.4 - height / 2 for position in range(len(data[0].points))],
                  [_wrap(p.label) for p in data[0].points])
    ax.invert_yaxis()
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.legend(fontsize=8)
    ax.margins(x=0.25)


def _draw_waterfall(ax, data: list[ChartSeriesData]) -> None:
    points = data[0].points
    steps, total = points[:-1], points[-1]
    running, bottoms = 0.0, []
    for step in steps:
        bottoms.append(running)
        running += step.value
    positions = list(range(len(points)))
    ax.bar(positions[:-1], [s.value for s in steps], bottom=bottoms,
           color=[BLUE if s.value >= 0 else ORANGE for s in steps])
    ax.bar(positions[-1], total.value, color=GREY)
    for position, bottom, point in zip(positions, [*bottoms, 0.0], points):
        top = max(bottom, bottom + point.value)
        ax.annotate(point.display, (position, top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(positions, [_wrap(p.label, 14) for p in points], fontsize=7)
    ax.axhline(0, color=GREY, linewidth=0.8)
    levels = [0.0, *bottoms, *(b + p.value for b, p in zip([*bottoms, 0.0], points))]
    low, high = min(levels), max(levels)
    pad = (high - low) * 0.15 or 1.0
    ax.set_ylim(low - (pad if low < 0 else 0.0), high + pad)


_DRAW = {"bar": _draw_bar, "grouped_bar": _draw_grouped_bar, "waterfall": _draw_waterfall}


def render_chart_png(spec: ChartSpec, data: list[ChartSeriesData]) -> bytes:
    """Deterministic PNG: fixed size, dpi, fonts and no embedded software or time metadata."""
    figure = Figure(figsize=(8, 4.5), dpi=100)
    FigureCanvasAgg(figure)
    ax = figure.subplots()
    _DRAW[spec.mark](ax, data)
    ax.set_title(spec.title, fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", metadata={"Software": None})
    return buffer.getvalue()


def build_chart_artifacts(specs: list[ChartSpec], catalog: EvidenceCatalog,
                          artifact_dir: str | Path) -> list[ChartArtifact]:
    """Validate, fill, render and store each chart as ``<artifact_dir>/charts/chart-<hash16>.png``.
    Nothing is rendered or written unless every spec is valid."""
    validate_chart_specs(specs, catalog)
    target_dir = Path(artifact_dir) / "charts"
    artifacts = []
    for spec in specs:
        data = fill_chart_data(spec, catalog)
        png = render_chart_png(spec, data)
        digest = hashlib.sha256(png).hexdigest()
        artifact_id = f"chart-{digest[:16]}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{artifact_id}.png"
        if not target.exists():
            target.write_bytes(png)
        artifacts.append(ChartArtifact(
            chart_id=spec.chart_id, artifact_id=artifact_id, content_hash=digest, filename=target.name,
            location=str(target), spec=spec, series=data,
        ))
    return artifacts
