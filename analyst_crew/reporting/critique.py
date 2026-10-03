"""Critique models and validation (ADR-0003).

The Critic flags; it never edits. A Critique is a list of issues, each with a
category from a closed set, a target that is the Summary or an existing Claim
(``summary`` or ``claim_<n>``, numbered as the Findings are) and a short
rationale. A rationale follows the Report's number rule: no digit outside an
``{E<id>}`` placeholder and only known Evidence Entry ids.
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


def report_targets(report) -> list[str]:
    """Every target a Critique may name for this Report: the Summary and each Findings Claim."""
    return ["summary", *[f"claim_{number}" for number in range(1, len(report.findings) + 1)]]


def target_label(target: str) -> str:
    return "Summary" if target == "summary" else f"Finding {target.removeprefix('claim_')}"


def validate_critique(critique: Critique, report, catalog) -> Critique:
    """Raise ``ValueError`` listing every problem, or return the Critique unchanged."""
    targets = set(report_targets(report))
    errors: list[str] = []
    for number, found in enumerate(critique.issues, start=1):
        if found.target not in targets or (found.target != "summary" and not _CLAIM_TARGET.fullmatch(found.target)):
            errors.append(
                f"issue {number} has target '{found.target}', which is not an existing Claim or the Summary; "
                f"valid targets are {', '.join(report_targets(report))}"
            )
        rationale = found.rationale
        if not rationale.strip():
            errors.append(f"issue {number} has no rationale")
        if len(rationale) > MAX_RATIONALE_CHARS:
            errors.append(f"issue {number} rationale is longer than {MAX_RATIONALE_CHARS} characters")
        for entry_id in _PLACEHOLDER.findall(rationale):
            if entry_id not in catalog.ids:
                errors.append(f"issue {number} rationale uses placeholder '{{{entry_id}}}', which is not an Evidence Entry id")
        remainder = _PLACEHOLDER.sub("", rationale)
        if "{" in remainder or "}" in remainder:
            errors.append(f"issue {number} rationale has an unbalanced or malformed placeholder brace")
        if re.search(r"\d", remainder):
            errors.append(f"issue {number} rationale contains a digit outside a {{E<id>}} placeholder")
    if errors:
        raise ValueError("; ".join(errors))
    return critique
