"""Critique models and validation (ADR-0003).

The Critic flags; it never edits. A Critique is a list of issues, each with a
category from a closed set, a target that is the Summary or an existing Claim
(``summary`` or ``claim_<n>``, numbered as the Findings are) and a short
rationale. A rationale is a reviewer's note, not Report body text: digits are
allowed in it (the Report's no-digit rule protects numbers in Claims, which a
Reviewer Note never states as a finding), an over-long one is truncated rather
than rejected so a sound Critique is not lost to a style limit, and any
``{E<id>}`` placeholder in it must be a known Evidence Entry id.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

CritiqueCategory = Literal["causal_overreach", "misread_evidence", "overstated_driver", "unsupported_emphasis"]
MAX_RATIONALE_CHARS = 400
_CLAIM_TARGET = re.compile(r"claim_([1-9]\d*)")
_PLACEHOLDER = re.compile(r"\{([^{}]*)\}")


class CritiqueIssue(BaseModel):
    category: CritiqueCategory
    target: str
    rationale: str


class Critique(BaseModel):
    issues: list[CritiqueIssue] = Field(default_factory=list)


def report_items(report) -> list[tuple[str, object]]:
    """The one place the target list is built: ``("summary", Summary)`` then ``("claim_<n>", Claim)`` numbered
    as the Findings are. Used by the Critic's context, the revision prompt and Critique validation."""
    return [("summary", report.summary), *[(f"claim_{n}", claim) for n, claim in enumerate(report.findings, start=1)]]


def report_targets(report) -> list[str]:
    """Every target a Critique may name for this Report: the Summary and each Findings Claim."""
    return [target for target, _ in report_items(report)]


def target_label(target: str) -> str:
    return "Summary" if target == "summary" else f"Finding {target.removeprefix('claim_')}"


def _truncate_rationale(rationale: str) -> str:
    """Cap a rationale at ``MAX_RATIONALE_CHARS`` without cutting a ``{E<id>}`` placeholder in half."""
    if len(rationale) <= MAX_RATIONALE_CHARS:
        return rationale
    cut = rationale[: MAX_RATIONALE_CHARS - 1]
    if cut.rfind("{") > cut.rfind("}"):
        cut = cut[: cut.rfind("{")]
    return cut.rstrip() + "…"


def validate_critique(critique: Critique, report, catalog) -> Critique:
    """Raise ``ValueError`` listing every problem, or return the Critique with over-long rationales truncated."""
    targets = set(report_targets(report))
    errors: list[str] = []
    for number, found in enumerate(critique.issues, start=1):
        if found.target not in targets or (found.target != "summary" and not _CLAIM_TARGET.fullmatch(found.target)):
            errors.append(
                f"issue {number} has target '{found.target}', which is not an existing Claim or the Summary; "
                f"valid targets are {', '.join(report_targets(report))}"
            )
        rationale = _truncate_rationale(found.rationale)
        if not rationale.strip():
            errors.append(f"issue {number} has no rationale")
        for entry_id in _PLACEHOLDER.findall(rationale):
            if entry_id not in catalog.ids:
                errors.append(f"issue {number} rationale uses placeholder '{{{entry_id}}}', which is not an Evidence Entry id")
        if "{" in _PLACEHOLDER.sub("", rationale) or "}" in _PLACEHOLDER.sub("", rationale):
            errors.append(f"issue {number} rationale has an unbalanced or malformed placeholder brace")
    if errors:
        raise ValueError("; ".join(errors))
    return Critique(issues=[
        found.model_copy(update={"rationale": _truncate_rationale(found.rationale)}) for found in critique.issues
    ])
