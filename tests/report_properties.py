"""Property checks over a Report, shared by the live acceptance test and the
DeepEval check. They assert properties that must hold for any wording, never
exact phrasing."""
import re

from analyst_crew.reporting.report import PLACEHOLDER, render_report
from tests.crew_harness import body_numbers


def report_property_violations(report, catalog, findings) -> list[str]:
    violations: list[str] = []
    known = catalog.ids

    for name, claim in [("Summary", report.summary), *[(f"Claim {n}", c) for n, c in enumerate(report.findings, 1)]]:
        if re.search(r"\d", PLACEHOLDER.sub("", claim.text)):
            violations.append(f"{name} has a digit outside a placeholder")
        if not claim.evidence_ids or not set(claim.evidence_ids) <= known:
            violations.append(f"{name} cites missing or unknown evidence")

    rendered = render_report(report, catalog)
    caveats = rendered.split("## Caveats")[1].split("## Methodology")[0]
    limitations = list(findings.limitations)
    if findings.revenue_bridge is not None:
        limitations += findings.revenue_bridge.limitations
    violations += [f"limitation missing from Caveats: {text[:60]}" for text in limitations if text not in caveats]
    if str(findings.quarantined_rows) not in caveats or str(findings.duplicate_rows) not in caveats:
        violations.append("quarantine and duplicate counts missing from Caveats")

    cited = {entry_id for claim in report.findings for entry_id in claim.evidence_ids}
    material_groups = {entry.group for entry in catalog.entries if entry.material}
    for group in sorted(material_groups):
        if not cited & {entry.id for entry in catalog.group(group)}:
            violations.append(f"Material Change not covered by any Claim: {group}")

    catalog_text = " ".join(entry.display for entry in catalog.entries)
    violations += [f"number in the rendered Report not in the catalog: {number}"
                   for number in body_numbers(rendered) if number not in catalog_text]
    return violations


def chart_property_violations(charts, catalog) -> list[str]:
    """Properties of a Chart Artifact set: at most three charts, every plotted value equal to an Evidence
    Entry raw value, every label the catalog label, and the stored PNG matching its content hash."""
    import hashlib
    from pathlib import Path

    violations: list[str] = []
    if len(charts) > 3:
        violations.append(f"{len(charts)} charts, at most 3 allowed")
    for chart in charts:
        for series in chart.series:
            for point in series.points:
                entry = catalog.get(point.entry_id)
                if entry is None:
                    violations.append(f"{chart.chart_id}: plotted point cites unknown entry {point.entry_id}")
                    continue
                if point.value != entry.value:
                    violations.append(f"{chart.chart_id}: plotted {point.value} is not entry {entry.id} value {entry.value}")
                if point.label != entry.label:
                    violations.append(f"{chart.chart_id}: label {point.label!r} is not the catalog label")
        path = Path(chart.location)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != chart.content_hash:
            violations.append(f"{chart.chart_id}: stored PNG missing or not matching its content hash")
    return violations
