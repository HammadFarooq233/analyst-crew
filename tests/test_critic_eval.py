"""Issue #20 — seeded-flaw eval for the Critic (ADR-0003), opt-in via the ``eval`` marker.

A hand-built Report over a valid Evidence Catalog is run through the real Critic. The Report
with a planted causal-overreach Claim must draw a ``causal_overreach`` issue that targets that
Claim; a control Report with the same numbers and no planted flaw must draw no
``causal_overreach`` issue on its Claims. Assertions are on category and target only, never on
wording (the vision's eval risk mitigation).

The Report and Catalog fixtures are pure and run offline; only the live Critic tests are gated
by ``eval`` and skip without LLM credentials.
"""
import pytest

from analyst_crew.agents.contracts import Handoff, Step
from analyst_crew.agents.critic import CriticAgent, production_critic_model
from analyst_crew.llm import credentials_available
from analyst_crew.reporting.catalog import EvidenceCatalog, build_evidence_catalog
from analyst_crew.reporting.report import Claim, Report, ReporterContent, build_report, validate_reporter_content
from tests.findings_factory import make_findings

QUESTION = "Compare this month's sales to last month and flag anything unusual"
PLANTED = "claim_1"


def _entry_id(catalog: EvidenceCatalog, source: str) -> str:
    return catalog.by_source(source).id


def _claim(catalog: EvidenceCatalog, template: str, *sources: str) -> Claim:
    ids = {source: _entry_id(catalog, source) for source in sources}
    return Claim(text=template.format(**{f"s{i}": "{" + ids[s] + "}" for i, s in enumerate(sources)}),
                 evidence_ids=list(ids.values()))


def build_reports() -> tuple[Report, Report, EvidenceCatalog]:
    """(planted Report, control Report, catalog). The two differ only in the first Finding."""
    catalog = build_evidence_catalog(make_findings())
    north = "dimensions.region.North"
    summary = _claim(catalog, "Net revenue changed by {s0} between {s1} and {s2}.",
                     "net_revenue.absolute_change", "window.prior_period", "window.current_period")
    planted = _claim(
        catalog,
        "{s0} grew by {s1}, because the new regional promotion made customers buy more, "
        "so the promotion clearly caused the revenue increase.",
        f"{north}.name", f"{north}.absolute_change",
    )
    control = _claim(
        catalog, "{s0} net revenue rose by {s1}, which was {s2} of the overall change.",
        f"{north}.name", f"{north}.absolute_change", f"{north}.contribution_percent",
    )
    rest = [
        _claim(catalog, "Volume contributed {s0} to the revenue bridge.", "revenue_bridge.volume_effect"),
        _claim(catalog, "Price contributed {s0} to the revenue bridge.", "revenue_bridge.price_effect"),
    ]

    def report(first: Claim) -> Report:
        content = validate_reporter_content(ReporterContent(summary=summary, claims=[first, *rest]), catalog)
        return build_report(content, catalog, QUESTION)

    return report(planted), report(control), catalog


def _real_critique(report: Report, catalog: EvidenceCatalog):
    step = Step(step_id="review_report", agent="Critic", objective="Review the draft Report for judgment errors",
                expected_output="report_critique", declared_artifacts=["report_critique"], depends_on=[])
    handoff = Handoff(step=step, user_question=QUESTION)
    return CriticAgent(production_critic_model()).run(handoff, report, catalog)


def _skip_without_llm():
    if not credentials_available():
        pytest.skip("LLM credentials not set")


def test_eval_reports_are_valid_and_differ_only_in_the_first_finding():
    planted, control, catalog = build_reports()

    assert len(planted.findings) == len(control.findings) == 3
    assert planted.findings[0] != control.findings[0]
    assert planted.findings[1:] == control.findings[1:] and planted.summary == control.summary
    for report in (planted, control):
        for claim in [report.summary, *report.findings]:
            assert claim.evidence_ids and set(claim.evidence_ids) <= catalog.ids


@pytest.mark.eval
def test_critic_flags_a_planted_causal_overreach_claim():
    _skip_without_llm()
    planted, _, catalog = build_reports()

    critique = _real_critique(planted, catalog)

    flagged = [(i.category, i.target) for i in critique.issues]
    assert ("causal_overreach", PLANTED) in flagged, f"Critic flagged {flagged}"


@pytest.mark.eval
def test_critic_does_not_flag_causal_overreach_on_a_control_report():
    _skip_without_llm()
    _, control, catalog = build_reports()

    critique = _real_critique(control, catalog)

    claim_overreach = [i.target for i in critique.issues
                       if i.category == "causal_overreach" and i.target.startswith("claim_")]
    assert claim_overreach == [], f"Critic flagged causal overreach on {claim_overreach}"
