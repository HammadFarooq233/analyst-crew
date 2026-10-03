"""Report model, Claim validation, code-assembled sections, Fallback Report and Markdown rendering (ADR-0002).

Everything here is deterministic code. The Reporter Agent supplies only the
Summary and the ordered Findings Claims; Methodology and Caveats are assembled
from the Evidence Catalog and can neither be omitted nor replaced by the model.
Rendering is a pure function of the stored Report and the catalog.
"""
from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from analyst_crew.reporting.catalog import EvidenceCatalog, EvidenceEntry
from analyst_crew.reporting.charts import ChartReference
from analyst_crew.reporting.critique import CritiqueIssue, target_label

PLACEHOLDER = re.compile(r"\{([^{}]*)\}")

FALLBACK_NOTICE = (
    "This Report was generated without the Reporter Agent. It is a deterministic template built directly "
    "from the computed Findings, not a written analysis."
)
FALLBACK_CHARTS_NOTE = (
    "Fallback chart set: chosen by code from the Evidence Catalog, with no Visualizer judgment. "
    "Every plotted value is an Evidence Entry value."
)
CHARTS_NOTICE ="Reserved. Charts are not generated yet; no Visualizer Agent is part of the Crew."


class CitedText(BaseModel):
    """Text whose numbers are placeholders (``{E3}``) that the renderer replaces
    with an Evidence Entry's display string, plus the ids it cites."""

    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Claim(CitedText):
    """One item of the Summary or the Findings section, written by the Reporter
    (or, in a Fallback Report, by a fixed template). ``chart_ids`` lists the charts its
    text mentions by ``{chart:<id>}`` placeholder; they are references, never evidence."""

    chart_ids: list[str] = Field(default_factory=list)


class ReportLine(CitedText):
    """One code-assembled line of the Caveats or Methodology section; never model-written."""


class ReporterContent(BaseModel):
    """What the Reporter Agent returns: the headline Summary and the ordered
    Findings Claims. Nothing else in a Report is model-written."""

    summary: Claim
    claims: list[Claim]


class Report(BaseModel):
    report_mode: Literal["llm", "fallback"]
    fallback_notice: str | None = None
    question: str
    summary: Claim
    findings: list[Claim]
    charts: list[ChartReference] = Field(default_factory=list)
    chart_mode: Literal["llm", "fallback"] | None = None
    caveats: list[ReportLine]
    methodology: list[ReportLine]
    # Critique issues that remain; the section is absent when empty (set only by the Critic Step).
    reviewer_notes: list[CritiqueIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def fallback_is_marked(self) -> "Report":
        if (self.report_mode == "fallback") != bool(self.fallback_notice):
            raise ValueError("A Report has a fallback_notice if and only if report_mode is 'fallback'")
        return self


class ReportValidationError(ValueError):
    """Reporter output that breaks the citation rules; carries every problem
    found so one retry can be given complete feedback."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def placeholders_in(text: str) -> list[str]:
    return PLACEHOLDER.findall(text)


def _cited_ids(text: str) -> list[str]:
    return list(dict.fromkeys(placeholders_in(text)))


def template_claim(text: str) -> Claim:
    """A code-written Claim citing exactly the entries its placeholders name."""
    return Claim(text=text, evidence_ids=_cited_ids(text))


def template_line(text: str) -> ReportLine:
    """A code-assembled Caveats or Methodology line citing the entries its placeholders name."""
    return ReportLine(text=text, evidence_ids=_cited_ids(text))


CHART_PREFIX = "chart:"


def validate_reporter_content(content: ReporterContent, catalog: EvidenceCatalog,
                              chart_ids: Collection[str] = ()) -> ReporterContent:
    """Enforce the ADR-0002 rules and return the content with each Claim's
    ``evidence_ids`` completed with every id its text places.

    Rules: every cited id and every placeholder exists in the catalog; no digit
    remains once placeholders are removed (no number parsing, no tolerance); no
    stray braces; every Claim cites evidence; there is at least one Claim. A
    ``{chart:<id>}`` placeholder must name one of ``chart_ids`` (the charts the
    Handoff carries); it is a reference, so it is neither evidence nor a number.
    """
    known = catalog.ids
    errors: list[str] = []

    def check(name: str, claim: Claim) -> Claim:
        if not claim.text.strip():
            errors.append(f"{name} has no text")
        all_placed = placeholders_in(claim.text)
        placed_charts = [p[len(CHART_PREFIX):] for p in all_placed if p.startswith(CHART_PREFIX)]
        placed = [p for p in all_placed if not p.startswith(CHART_PREFIX)]
        for chart_id in placed_charts:
            if chart_id not in chart_ids:
                errors.append(f"{name} references chart '{chart_id}', which is not a chart of this Report "
                              f"(available chart ids: {', '.join(sorted(chart_ids)) or 'none'})")
        for entry_id in placed:
            if entry_id not in known:
                errors.append(f"{name} uses placeholder '{{{entry_id}}}', which is not an Evidence Entry id")
        remainder = PLACEHOLDER.sub("", claim.text)
        if "{" in remainder or "}" in remainder:
            errors.append(f"{name} has an unbalanced or malformed placeholder brace")
        digits = re.findall(r"\d+", remainder)
        if digits:
            errors.append(
                f"{name} contains the digit(s) {', '.join(sorted(set(digits)))} outside a placeholder; "
                "write every number, date and name that contains a digit as its {E<id>} placeholder"
            )
        for entry_id in claim.evidence_ids:
            if entry_id not in known:
                errors.append(f"{name} cites unknown Evidence Entry id '{entry_id}'")
        if not claim.evidence_ids and not placed:
            errors.append(f"{name} cites no Evidence Entry")
        return Claim(text=claim.text, evidence_ids=list(dict.fromkeys([*claim.evidence_ids, *placed])),
                     chart_ids=list(dict.fromkeys(placed_charts)))

    summary = check("Summary", content.summary)
    claims = [check(f"Claim {index}", claim) for index, claim in enumerate(content.claims, start=1)]
    if not content.claims:
        errors.append("Findings must contain at least one Claim")
    if errors:
        raise ReportValidationError(errors)
    return ReporterContent(summary=summary, claims=claims)


def _entry_placeholder(entry: EvidenceEntry) -> str:
    return "{" + entry.id + "}"


def assemble_caveats(catalog: EvidenceCatalog) -> list[ReportLine]:
    """Every Findings limitation, every non-available optional source status,
    the CPI non-causality caveat when CPI is present, and the quarantine and
    duplicate disclosure. Mandatory: not selectable by the Reporter."""
    caveats: list[ReportLine] = []
    for entry in catalog.entries:
        if entry.source.startswith(("limitations.", "revenue_bridge.limitations.")):
            caveats.append(template_line("{" + entry.id + "}"))

    optional_sources = (
        ("target_comparison.status", "Sales-target comparison", "no sales-targets Data Source was configured"),
        ("cpi_adjustment.status", "Inflation-adjusted (CPI) comparison", "no CPI Data Source was configured"),
    )
    for source, name, absent_reason in optional_sources:
        status = catalog.find_source(source)
        if status is None:
            caveats.append(template_line(f"{name} was not part of this run ({absent_reason})."))
        elif status.value != "available":
            caveats.append(template_line(f"{name} is not included in this Report. Source status: {{{status.id}}}."))

    non_causality = catalog.find_source("cpi_adjustment.non_causality_caveat")
    if non_causality is not None:
        caveats.append(template_line("{" + non_causality.id + "}"))

    def ref(source: str) -> str:
        return _entry_placeholder(catalog.by_source(source))

    caveats.append(template_line(
        f"Data quality: {ref('data_quality.source_rows')} source rows were read, "
        f"{ref('data_quality.included_rows')} were included and {ref('data_quality.quarantined_rows')} were "
        f"quarantined, including {ref('data_quality.duplicate_rows')} excluded as duplicate order ids. "
        f"The calculable gross revenue impact of the excluded rows is {ref('data_quality.excluded_revenue_impact')}."
    ))
    for entry in catalog.entries:
        if entry.source.startswith("data_quality.quarantine_reasons."):
            caveats.append(template_line(f"{entry.label}: {{{entry.id}}}."))
    return caveats


def assemble_methodology(catalog: EvidenceCatalog) -> list[ReportLine]:
    def ref(source: str) -> str:
        return _entry_placeholder(catalog.by_source(source))

    lines = [
        f"Comparison window: {ref('window.current_period')} compared with {ref('window.prior_period')}, "
        f"chosen as {ref('window.mode')}.",
        f"Material Change thresholds: a change of at least {ref('methodology.materiality.relative_change_percent')} "
        f"and an absolute impact of at least "
        f"{ref('methodology.materiality.absolute_change_percent_of_prior_total')} of prior-period net revenue "
        f"(source: {ref('methodology.materiality.source')}).",
    ]
    if catalog.find_source("revenue_bridge.methodology") is not None:
        lines.append(f"Revenue bridge method: {ref('revenue_bridge.methodology')}")
    lines.append(
        f"Sales data source: Dataset Artifact {ref('methodology.dataset_artifact_id')} (content hash "
        f"{ref('methodology.dataset_content_hash')}; integrity check {ref('methodology.integrity_verified')})."
    )
    if catalog.find_source("target_comparison.status") is not None:
        lines.append(f"Optional source, Postgres sales targets: {ref('target_comparison.status')}.")
    if catalog.find_source("cpi_adjustment.status") is not None:
        lines.append(f"Optional source, FRED CPI: {ref('cpi_adjustment.status')}.")
        if catalog.find_source("cpi_adjustment.methodology") is not None:
            lines.append(f"CPI adjustment method: {ref('cpi_adjustment.methodology')}")
    lines.append(
        f"Row counts: {ref('data_quality.source_rows')} source, {ref('data_quality.included_rows')} included, "
        f"{ref('data_quality.quarantined_rows')} quarantined, {ref('data_quality.duplicate_rows')} duplicates excluded."
    )
    return [template_line(line) for line in lines]


def build_report(content: ReporterContent, catalog: EvidenceCatalog, question: str,
                 charts: list[ChartReference] | None = None, chart_mode: str | None = None) -> Report:
    """An ``llm``-mode Report: model-written Summary and Claims, code-assembled rest.
    ``charts`` are the Chart Artifact references for the code-assembled Charts section."""
    return Report(
        report_mode="llm", question=question, summary=content.summary, findings=content.claims,
        charts=charts or [], chart_mode=chart_mode if charts else None,
        caveats=assemble_caveats(catalog), methodology=assemble_methodology(catalog),
    )


def build_fallback_report(catalog: EvidenceCatalog, question: str,
                          charts: list[ChartReference] | None = None, chart_mode: str | None = None) -> Report:
    """Deterministic template Report: fixed Summary (window and net revenue
    change), Material Changes as templated Claims, plus the same code-assembled
    Caveats and Methodology. Visibly marked as generated without the Reporter."""

    def ref(source: str) -> str:
        return _entry_placeholder(catalog.by_source(source))

    change = f"a change of {ref('net_revenue.absolute_change')}"
    if catalog.find_source("net_revenue.percent_change") is not None:
        change += f" ({ref('net_revenue.percent_change')})"
    summary = template_claim(
        f"Net revenue for {ref('window.current_period')} was {ref('net_revenue.current')}, compared with "
        f"{ref('net_revenue.prior')} for {ref('window.prior_period')}: {change}."
    )

    material_groups: dict[str, list[EvidenceEntry]] = {}
    for entry in catalog.entries:
        if entry.material and entry.group:
            material_groups.setdefault(entry.group, []).append(entry)

    def by_suffix(entries: list[EvidenceEntry], suffix: str) -> EvidenceEntry | None:
        return next((entry for entry in entries if entry.source.endswith(f".{suffix}")), None)

    def importance(group: str) -> tuple[float, str]:
        change_entry = by_suffix(material_groups[group], "absolute_change")
        return (-abs(float(change_entry.value)) if change_entry else 0.0, group)

    claims: list[Claim] = []
    for group in sorted(material_groups, key=importance):
        entries = material_groups[group]
        dimension = group.split(":", 1)[0].replace("_", " ")
        parts = [
            f"Material Change in {dimension} {_entry_placeholder(by_suffix(entries, 'name'))}: net revenue was "
            f"{_entry_placeholder(by_suffix(entries, 'current'))} against "
            f"{_entry_placeholder(by_suffix(entries, 'prior'))}, a change of "
            f"{_entry_placeholder(by_suffix(entries, 'absolute_change'))}"
        ]
        if by_suffix(entries, "percent_change") is not None:
            parts.append(f" ({_entry_placeholder(by_suffix(entries, 'percent_change'))})")
        elif by_suffix(entries, "baseline_status") is not None:
            parts.append(f" ({_entry_placeholder(by_suffix(entries, 'baseline_status'))})")
        parts.append(".")
        if by_suffix(entries, "contribution_percent") is not None:
            parts.append(
                f" That is {_entry_placeholder(by_suffix(entries, 'contribution_percent'))} of the overall "
                "net revenue change."
            )
        claims.append(template_claim("".join(parts)))
    if not claims:
        claims.append(template_claim(
            "No dimension driver met both Material Change thresholds "
            f"({ref('methodology.materiality.relative_change_percent')} relative change and "
            f"{ref('methodology.materiality.absolute_change_percent_of_prior_total')} of prior-period net revenue) "
            f"in this period; {ref('material_changes.count')} Material Changes were found."
        ))

    return Report(
        report_mode="fallback", fallback_notice=FALLBACK_NOTICE, question=question, summary=summary,
        findings=claims, charts=charts or [], chart_mode=chart_mode if charts else None,
        caveats=assemble_caveats(catalog), methodology=assemble_methodology(catalog),
    )


def _substitute(text: str, catalog: EvidenceCatalog, charts: Mapping[str, str] | None = None) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.startswith(CHART_PREFIX):
            title = (charts or {}).get(token[len(CHART_PREFIX):])
            return match.group(0) if title is None else f"\u201c{title}\u201d"
        entry = catalog.get(token)
        return match.group(0) if entry is None else entry.display

    return PLACEHOLDER.sub(replace, text)


def _sentence(text: str, catalog: EvidenceCatalog, charts: Mapping[str, str] | None = None) -> str:
    """Substituted text with its first letter capitalised, because a placeholder
    (a lower-cased dimension value, say) may open the sentence."""
    text = _substitute(text, catalog, charts)
    return text[:1].upper() + text[1:]


def _truncate(text: str, limit: int = 110) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _citation(claim: Claim) -> str:
    return f" `[{', '.join(claim.evidence_ids)}]`" if claim.evidence_ids else ""


def render_report(report: Report, catalog: EvidenceCatalog) -> str:
    """Markdown for a stored Report. A pure function of the Report and the
    catalog; no model is ever involved."""
    titles = {chart.chart_id: chart.title for chart in report.charts}
    lines: list[str] = ["# Analyst Crew Report", ""]
    if report.fallback_notice:
        lines += [f"> **Fallback Report.** {report.fallback_notice}", ""]
    lines += [f"**Question:** {report.question}", "", "## Summary", "",
              _sentence(report.summary.text, catalog, titles) + _citation(report.summary), "", "## Findings", ""]
    lines += [f"{number}. {_sentence(claim.text, catalog, titles)}{_citation(claim)}"
              for number, claim in enumerate(report.findings, start=1)]
    lines += ["", "## Charts", ""]
    if report.charts:
        if report.chart_mode == "fallback":
            lines += [f"_{FALLBACK_CHARTS_NOTE}_", ""]
        for chart in report.charts:
            lines += [f"![{chart.title}]({chart.filename})", ""]
        lines.pop()
    else:
        lines.append(f"_{CHARTS_NOTICE}_")
    lines += ["", "## Caveats", ""] + [f"- {_substitute(claim.text, catalog)}" for claim in report.caveats]
    lines += ["", "## Methodology", ""] + [f"- {_substitute(claim.text, catalog)}" for claim in report.methodology]

    if report.reviewer_notes:
        lines += ["", "## Reviewer Notes", ""]
        lines += [
            f"- **{target_label(note.target)}** ({note.category.replace('_', ' ')}): {_sentence(note.rationale, catalog)}"
            for note in report.reviewer_notes
        ]

    cited = {entry_id for claim in [report.summary, *report.findings] for entry_id in claim.evidence_ids}
    evidence = [entry for entry in catalog.entries if entry.id in cited]
    if evidence:
        lines += ["", "## Evidence", ""]
        lines += [f"- `{entry.id}` {entry.label}: {_truncate(entry.display)} (`{entry.source}`)" for entry in evidence]
    return "\n".join(lines) + "\n"
