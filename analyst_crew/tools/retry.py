"""Shared bounded-retry policy for Fetcher/Analyst Tools (issue #10).

A Tool failure falls into exactly one of three classes:

- transient (network timeouts, connection errors, HTTP 5xx): bounded retry
  with exponential backoff, up to ``max_attempts``.
- authentication/policy (401/403, permission denied): never retried —
  retrying a credential or permission failure just repeats a failure that
  will never succeed on its own. Fails immediately/terminally.
- validation (malformed response, schema violation): never retried by a
  Tool either. Retrying validation failures is Agent *output*'s job — one
  repair attempt with the validation error fed back as feedback — never a
  Tool's, and never a second Plan from the Planner.

``retry_with_backoff`` is the one bounded-retry loop primitive shared by
every Fetcher Tool that needs it (``analyst_crew.tools.targets``,
``analyst_crew.tools.sales`` and ``analyst_crew.tools.analysis``'s
Agent-side tool-call retries). It knows nothing about what "transient"
means for a given Tool — the caller supplies an ``is_transient`` predicate
— so each Tool keeps its own domain-specific classification (HTTP status
codes, exception types, error-message shapes) while sharing the same
attempt-count/backoff mechanics and the same ``RetryOutcome`` shape for
CrewState and tracing visibility.

``analyst_crew.tools.fred.retrieve_cpi`` already implements this exact
three-way policy end-to-end (transient network errors and 5xx retry with
backoff; 401/403/400 fail immediately) against well-established tests; it
is intentionally left as its own implementation rather than force-fit onto
this generic loop, since its retry condition is an HTTP status code
inspected *after* a successful call, not just an exception. New retry
call sites reuse this helper instead of writing a fourth ad hoc loop.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5


@dataclass(frozen=True)
class RetryOutcome:
    """How many attempts a bounded retry loop actually used, for CrewState
    and tracing visibility. ``attempts`` counts every call made (1 means no
    retry was needed); ``retries`` is the convenience ``attempts - 1``."""

    attempts: int
    retries: int


def retry_with_backoff(
    call: Callable[[], T],
    *,
    is_transient: Callable[[BaseException], bool],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, RetryOutcome]:
    """Call ``call()`` up to ``max_attempts`` times.

    A non-transient exception (per ``is_transient``) propagates immediately
    and is never retried — this is how authentication/policy and validation
    failures stay terminal on the first attempt. A transient exception
    retries with exponential backoff until the last attempt, at which point
    it propagates too: the loop is bounded, never infinite.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            result = call()
        except BaseException as exc:
            if not is_transient(exc) or attempt >= max_attempts:
                raise
            sleep(backoff_seconds * (2 ** (attempt - 1)))
            continue
        return result, RetryOutcome(attempts=attempt, retries=attempt - 1)
