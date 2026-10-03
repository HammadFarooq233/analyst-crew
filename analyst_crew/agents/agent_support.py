"""Helpers shared by the Reporter, Visualizer and Critic Agents (ADR-0002, ADR-0003).

All three are tool-less LLM Agents with the same loop: invalid output gets exactly one repair retry with the
problem fed back, and a transient provider failure gets a bounded retry (``run_validated``).
"""
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from analyst_crew.tools.retry import invoke_with_optional_config, retry_with_backoff


class TransientReporterError(Exception):
    """Retryable provider failure."""


def is_transient_provider_error(exc: BaseException) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, TransientReporterError)):
        return True
    if any(cls.__name__ in {"APITimeoutError", "APIConnectionError"} for cls in type(exc).__mro__):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status in (408, 409, 429) or status >= 500)


def catalog_prompt(entries: list[Mapping[str, Any]], *, with_source: bool = False) -> str:
    """Evidence Catalog lines for a prompt: ``id | label | display`` (plus ``| source`` when asked)."""
    lines = []
    for entry in entries:
        flag = " [MATERIAL CHANGE]" if entry.get("material") else ""
        source = f" | {entry['source']}" if with_source else ""
        lines.append(f"{entry['id']} | {entry['label']} | {entry['display']}{source}{flag}")
    return "\n".join(lines)


def summarize_schema_error(exc: ValidationError, *, include_input: bool = False) -> str:
    def describe(error: Mapping[str, Any]) -> str:
        text = f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
        return f"{text} (got {error.get('input')!r})" if include_input else text

    return "; ".join(describe(error) for error in exc.errors())


def format_failure(exc: BaseException) -> str:
    """Failure text for artifacts: our own RuntimeError/ValueError messages as-is, anything else with its type."""
    return str(exc) if isinstance(exc, (RuntimeError, ValueError)) else f"{type(exc).__name__}: {exc}"


def named_span(model: Any, run_name: str, tags: list[str]) -> Any:
    """Wrap ``model`` so each call is its own named run (a span in the trace) with the generation nested under it."""
    return RunnableLambda(
        lambda ctx, config=None: invoke_with_optional_config(model, ctx, config),
    ).with_config(run_name=run_name, tags=tags)


def run_validated(agent: Any, context: dict[str, Any], parse: Callable[[Any], Any], *, retry_hint: str,
                  label: str, model: Any = None, config: Any = None) -> Any:
    """The validated request loop shared by the tool-less Agents.

    ``agent`` carries ``model``, ``max_provider_attempts``, ``backoff_seconds`` and the counters
    ``validation_retries``, ``tool_retries`` and ``last_validation_error``. ``parse(raw)`` turns the model
    output into the validated result and raises ``ValueError`` on a problem. A ``ValueError`` out of the model
    call (unparseable JSON) takes the same repair path; other provider failures are retried only when transient.
    """
    model = agent.model if model is None else model
    feedback = ""
    for attempt in range(2):
        calls = 0

        def call() -> Any:
            nonlocal calls
            calls += 1
            return invoke_with_optional_config(model, {**context, "validation_feedback": feedback}, config)

        try:
            try:
                raw, _ = retry_with_backoff(
                    call, is_transient=is_transient_provider_error,
                    max_attempts=agent.max_provider_attempts, backoff_seconds=agent.backoff_seconds,
                )
            finally:
                # Counted even when the bounded retries are exhausted and the call raises.
                agent.tool_retries += calls - 1
            return parse(raw)
        except ValueError as exc:
            agent.last_validation_error = str(exc)
            if attempt:
                raise
            agent.validation_retries += 1
            feedback = f"Your previous response was invalid: {exc}. {retry_hint}"
    raise RuntimeError(f"{label} request loop exhausted")
