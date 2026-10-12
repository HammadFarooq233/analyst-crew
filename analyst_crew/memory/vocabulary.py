"""The closed Typed Preference vocabulary, validated in code (ADR-0005).

Each key maps onto an existing analysis parameter. Extending the vocabulary is a
deliberate code change here. The grouping-dimension preference named in the spec is
not included: no existing analysis parameter selects a grouping dimension (every
dimension is always computed), so it would not map onto anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


class PreferenceError(ValueError):
    """A proposed Typed Preference is not in the vocabulary or its value is out of range."""


@dataclass(frozen=True)
class PreferenceSpec:
    key: str
    label: str
    low: float
    high: float  # inclusive


PREFERENCE_KEYS: dict[str, PreferenceSpec] = {
    spec.key: spec for spec in (
        PreferenceSpec("materiality_relative_threshold_percent", "Material Change relative threshold", 0.1, 1000.0),
        PreferenceSpec("materiality_absolute_threshold_percent",
                       "Material Change absolute-impact threshold (percent of prior-period net revenue)", 0.1, 100.0),
    )
}


def validate_preference(key: str, raw: object) -> float:
    """The validated value for ``key``, or ``PreferenceError`` saying why not."""
    spec = PREFERENCE_KEYS.get(key)
    if spec is None:
        raise PreferenceError(f"{key!r} is not a known preference; known: {', '.join(PREFERENCE_KEYS)}")
    value = raw
    if isinstance(value, str):
        text = value.strip().rstrip("%").strip()
        try:
            value = float(text)
        except ValueError:
            raise PreferenceError(f"{raw!r} is not a number") from None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreferenceError(f"{raw!r} is not a number")
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        raise PreferenceError(f"{raw!r} is not a finite number")
    if not spec.low <= value <= spec.high:
        raise PreferenceError(f"{value:g} is outside the allowed range {spec.low:g} to {spec.high:g} for {spec.label}")
    return value
