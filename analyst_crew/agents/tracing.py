"""Langfuse tracing helpers and the per-run trace metadata."""
import os
import time
from typing import Any

from langfuse import Langfuse

from analyst_crew.state import CrewState
from analyst_crew.tools.analysis import BaselineFindings


def _ensure_langchain_callback_compat() -> None:
    """Make the pinned Langfuse v2 SDK's LangChain integration importable
    against this project's langchain 1.x.

    ``langfuse.callback`` (the v2 SDK's ``CallbackHandler``) imports
    ``langchain.callbacks.base.BaseCallbackHandler`` and
    ``langchain.schema.{agent,document}`` — legacy import paths that
    langchain>=1.0 no longer exposes (those symbols now live only in
    ``langchain_core``, which this project already depends on directly).
    This registers thin ``sys.modules`` shims for the missing legacy paths,
    pointing them at their ``langchain_core`` equivalents, before
    ``langfuse.callback`` is imported. It only patches paths that are
    actually missing, so it is a no-op against any langchain version that
    still exposes them.
    """
    import sys
    import types

    try:
        import langchain.callbacks.base  # noqa: F401
    except ImportError:
        from langchain_core.callbacks.base import BaseCallbackHandler

        callbacks_pkg = types.ModuleType("langchain.callbacks")
        callbacks_base = types.ModuleType("langchain.callbacks.base")
        callbacks_base.BaseCallbackHandler = BaseCallbackHandler
        callbacks_pkg.base = callbacks_base
        sys.modules["langchain.callbacks"] = callbacks_pkg
        sys.modules["langchain.callbacks.base"] = callbacks_base

    try:
        import langchain.schema.agent  # noqa: F401
        have_schema_agent = True
    except ImportError:
        have_schema_agent = False
    try:
        import langchain.schema.document  # noqa: F401
        have_schema_document = True
    except ImportError:
        have_schema_document = False

    if not (have_schema_agent and have_schema_document):
        schema_pkg = sys.modules.get("langchain.schema") or types.ModuleType("langchain.schema")
        if not have_schema_agent:
            from langchain_core.agents import AgentAction, AgentFinish

            schema_agent = types.ModuleType("langchain.schema.agent")
            schema_agent.AgentAction = AgentAction
            schema_agent.AgentFinish = AgentFinish
            schema_pkg.agent = schema_agent
            sys.modules["langchain.schema.agent"] = schema_agent
        if not have_schema_document:
            from langchain_core.documents import Document

            schema_document = types.ModuleType("langchain.schema.document")
            schema_document.Document = Document
            schema_pkg.document = schema_document
            sys.modules["langchain.schema.document"] = schema_document
        sys.modules["langchain.schema"] = schema_pkg


def get_langfuse_handler(
    session_id: str | None = None,
    tags: list[str] | None = None,
    user_id: str | None = None,
) -> tuple[Any, Any, Any]:
    """Build one Langfuse v2 trace and a ``CallbackHandler`` bound to it for
    this Crew run.

    The handler is shared across the whole ``graph.invoke()`` call (threaded
    through every node's ``config``), so every Planner/Fetcher/Analyst LLM
    call and Tool invocation becomes a nested observation on this same
    trace, instead of each ``invoke()`` producing its own flat, unlinked
    trace record.

    Returns ``(client, trace, handler)`` — all ``None`` when Langfuse isn't
    configured or the SDK/compat shim fails; observability must never make a
    successful Crew run fail.
    """
    host = os.getenv("LANGFUSE_HOST")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if not all((host, public_key, secret_key)):
        return None, None, None

    try:
        _ensure_langchain_callback_compat()
        from langfuse.callback import CallbackHandler

        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        trace = client.trace(
            name="crew_run",
            session_id=session_id,
            user_id=user_id,
            tags=tags or [],
        )
        handler = CallbackHandler(stateful_client=trace)
        return client, trace, handler
    except Exception:
        # Observability must never make a successful Crew run fail.
        return None, None, None


def _finish_langfuse_trace(
    client: Any,
    trace: Any,
    metadata: dict[str, Any] | None = None,
    *,
    wait_for_index: bool = True,
) -> None:
    """Attach this run's retry counts, terminal-failure outcome, and source
    status to the trace ``get_langfuse_handler`` already created and
    threaded through the graph run (see ``_run_metadata``), then flush and —
    for callers that immediately query the API afterward — wait briefly for
    Langfuse v2's async ingestion to index it.
    """
    if client is None or trace is None:
        return
    try:
        trace.update(metadata=metadata or {})
        client.flush()
        if not wait_for_index:
            return
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                if client.api.trace.get(trace.id) is not None:
                    return
            except Exception:
                pass
            time.sleep(0.5)
    except Exception:
        # Observability must never make a successful Crew run fail.
        return


def _run_metadata(result: CrewState) -> dict[str, Any]:
    """Retry counts, terminal-outcome, Step resume outcomes, and optional-
    source status summary for tracing metadata — populated from the same
    ``artifacts`` the CLI and any other caller can already inspect after
    ``invoke()`` returns, and attached to the one Langfuse trace this run's
    ``CallbackHandler`` already linked Planner/Fetcher/Analyst spans to."""
    artifacts = result.get("artifacts", {})
    metadata: dict[str, Any] = {}
    retry_counts = artifacts.get("retry_counts")
    if retry_counts:
        metadata["retry_counts"] = retry_counts
    terminal_failure = artifacts.get("terminal_failure")
    if terminal_failure:
        metadata["terminal_failure"] = terminal_failure
    report_mode = artifacts.get("report_mode")
    if report_mode:
        metadata["report_mode"] = report_mode
    chart_mode = artifacts.get("chart_mode")
    if chart_mode:
        metadata["chart_mode"] = chart_mode
    chart_failure = artifacts.get("chart_failure")
    if chart_failure:
        metadata["chart_failure"] = chart_failure
    critic_outcome = artifacts.get("critic_outcome")
    if critic_outcome:
        metadata["critic_outcome"] = critic_outcome
    critic_failure = artifacts.get("critic_failure")
    if critic_failure:
        metadata["critic_failure"] = critic_failure
    step_outcomes = artifacts.get("step_outcomes")
    if step_outcomes:
        metadata["step_outcomes"] = {
            step_id: {
                "status": outcome.get("status"),
                "resume_outcome": outcome.get("resume_outcome"),
                "resume_reason": outcome.get("resume_reason"),
            }
            for step_id, outcome in step_outcomes.items()
        }
    findings = artifacts.get("findings")
    if findings is not None:
        findings = findings if isinstance(findings, BaselineFindings) else BaselineFindings.model_validate(findings)
        source_status: dict[str, Any] = {}
        if findings.target_comparison is not None:
            source_status["sales_targets_postgres"] = findings.target_comparison.source_status.model_dump()
        if findings.cpi_adjustment is not None:
            source_status["cpi_fred"] = findings.cpi_adjustment.source_status.model_dump()
        if source_status:
            metadata["source_status"] = source_status
    return metadata
