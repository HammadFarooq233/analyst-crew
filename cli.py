import argparse
import os
import shutil
import sys
import uuid
from pathlib import Path

import config  # noqa: F401 — triggers load_dotenv()
from analyst_crew.agents import invoke
from analyst_crew.reporting.catalog import EvidenceCatalog, build_evidence_catalog
from analyst_crew.reporting.charts import ChartArtifact, load_charts
from analyst_crew.reporting.report import Report, render_report
from analyst_crew.tools.analysis import BaselineFindings


def format_plan(result) -> str:
    plan_error = result.get("artifacts", {}).get("plan_error")
    if plan_error is not None:
        return f"Error: {plan_error}"

    plan = result.get("artifacts", {}).get("plan")
    if plan is None:
        return "No plan found in result."

    lines = [f"Question: {plan.user_question}", ""]
    for step in plan.steps:
        deps = ", ".join(step.depends_on) if step.depends_on else "none"
        lines.append(f"  [{step.step_id}] {step.agent}")
        lines.append(f"    Produces: {step.produces}")
        lines.append(f"    Depends on: {deps}")
        lines.append("")
    return "\n".join(lines)


def format_terminal_failure(result) -> str:
    """Concise, human-readable summary of a terminal Step failure. Full
    diagnostics (retry counts, the original error, which Step/Agent failed,
    and the original Plan) remain on the returned CrewState/``artifacts``
    dict for programmatic inspection — this is deliberately just the
    one-line summary printed to the user."""
    failure = result.get("artifacts", {}).get("terminal_failure")
    if failure is None:
        return ""
    agent = failure.get("agent", "unknown Agent")
    step_id = failure.get("step_id")
    step_note = f" (Step '{step_id}')" if step_id else ""
    retry_counts = failure.get("retry_counts") or {}
    retries_note = (
        f" after {retry_counts.get('validation_retries', 0)} validation "
        f"retr{'y' if retry_counts.get('validation_retries', 0) == 1 else 'ies'} and "
        f"{retry_counts.get('tool_retries', 0)} tool retr{'y' if retry_counts.get('tool_retries', 0) == 1 else 'ies'}"
        if retry_counts else ""
    )
    error = failure.get("error", "unknown error")
    return (
        f"Error: {agent}{step_note} failed terminally{retries_note}: {error}\n"
        "The original Plan and full diagnostics remain available in the returned state."
    )


def rendered_report(result) -> str | None:
    """The Markdown Report from a Crew result, or None when the run produced none."""
    artifacts = result.get("artifacts", {})
    report = artifacts.get("report")
    if report is None:
        return None
    if not isinstance(report, Report):
        report = Report.model_validate(report)
    catalog = artifacts.get("evidence_catalog")
    if catalog is None:
        catalog = build_evidence_catalog(_findings(artifacts))
    elif not isinstance(catalog, EvidenceCatalog):
        catalog = EvidenceCatalog.model_validate(catalog)
    return render_report(report, catalog)


def stored_charts(result) -> list[ChartArtifact]:
    """The Chart Artifacts referenced by the Report in this result (none for a Crew without a Visualizer)."""
    artifacts = result.get("artifacts", {})
    report = artifacts.get("report")
    if report is None:
        return []
    referenced = {ref.artifact_id for ref in (report if isinstance(report, Report) else Report.model_validate(report)).charts}
    return [chart for chart in load_charts(artifacts.get("charts")) if chart.artifact_id in referenced]


def format_charts(result) -> str:
    """Chart titles and the paths of the stored PNGs, listed under the rendered Report."""
    charts = stored_charts(result)
    if not charts:
        return ""
    return "\nCharts generated:\n" + "\n".join(f"  - {chart.spec.title}: {chart.location}" for chart in charts)


def save_charts(result, directory: Path) -> list[Path]:
    """Copy the Report's chart PNGs beside the saved Markdown so its relative image links resolve."""
    saved = []
    for chart in stored_charts(result):
        target = directory / chart.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chart.location, target)
        saved.append(target)
    return saved


def format_reporter_failure(failure) -> str:
    retry_counts = failure.get("retry_counts") or {}
    validation, provider = retry_counts.get("validation_retries", 0), retry_counts.get("tool_retries", 0)
    return (
        f"The Reporter Agent failed after {validation} validation "
        f"retr{'y' if validation == 1 else 'ies'} and {provider} provider retr{'y' if provider == 1 else 'ies'}: "
        f"{failure.get('error', 'unknown error')}. The Fallback Report above was generated without the "
        "Reporter Agent; the failure is recorded in the returned state and on the trace."
    )


def _findings(artifacts) -> BaselineFindings:
    findings = artifacts["findings"]
    return findings if isinstance(findings, BaselineFindings) else BaselineFindings.model_validate(findings)


def format_result(result) -> str:
    artifacts = result.get("artifacts", {})
    terminal_failure = artifacts.get("terminal_failure")
    # A required-Step failure wins over any Report still on a resumed thread's
    # state; only a Reporter failure (required False) ships with its Fallback Report.
    if terminal_failure is not None and terminal_failure.get("required", True):
        return format_terminal_failure(result)

    report = rendered_report(result)
    if report is not None:
        report += format_charts(result)
        if terminal_failure is not None:
            return f"{report}\n{format_reporter_failure(terminal_failure)}"
        return report

    findings = artifacts.get("findings")
    if findings is None:
        return format_plan(result)
    return format_findings(_findings(artifacts))


def format_findings(findings: BaselineFindings) -> str:
    """Findings summary, printed only when a run produced no Report (its Plan
    had no Reporter Step)."""
    window = findings.window
    def metric(label, comparison, *, decimals=2):
        percent = "n/a" if comparison.percent_change is None else f"{comparison.percent_change:+.1f}%"
        return (
            f"{label} {comparison.current:,.{decimals}f} vs {comparison.prior:,.{decimals}f} "
            f"({comparison.absolute_change:+,.{decimals}f}, {percent})"
        )

    return (
        f"Baseline sales Findings ({window.mode}): "
        f"{window.current_start.isoformat()} to {window.current_end.isoformat()} vs "
        f"{window.prior_start.isoformat()} to {window.prior_end.isoformat()}. "
        f"{metric('Net revenue', findings.net_revenue)}; "
        f"{metric('gross revenue', findings.gross_revenue)}; "
        f"{metric('orders', findings.orders, decimals=0)}; "
        f"{metric('units', findings.units, decimals=0)}. "
        f"Evidence: {findings.source_rows} source rows, {findings.included_rows} included, "
        f"{findings.quarantined_rows} quarantined, {findings.duplicate_rows} duplicates excluded; "
        f"excluded gross revenue impact {findings.excluded_revenue_impact:,.2f}; "
        f"reasons {findings.quarantine_reasons}.\n"
        f"{format_material_changes(findings)}\n"
        f"{format_revenue_bridge(findings)}\n"
        f"{format_target_comparison(findings)}\n"
        f"{format_cpi_adjustment(findings)}"
    )


def format_cpi_adjustment(findings: BaselineFindings) -> str:
    """Render the optional FRED CPI inflation-adjustment context. When the
    optional Data Source was not part of this run, this is silent — when it
    was attempted but unavailable/invalid, the recorded status and the
    matching limitation are both surfaced so the degradation is visible.
    Nominal and inflation-adjusted changes are always labeled separately,
    and the non-causality caveat is always shown alongside an available
    adjustment."""
    cpi = findings.cpi_adjustment
    if cpi is None:
        return "CPI inflation context: not part of this run."
    status = cpi.source_status
    if status.status != "available":
        detail = f" ({status.detail})" if status.detail else ""
        return f"CPI inflation context: {status.status}{detail}. Nominal net revenue change {cpi.nominal_change:+,.2f}."
    return (
        f"CPI inflation context (FRED series {cpi.series_id}): "
        f"nominal net revenue change {cpi.nominal_change:+,.2f}; "
        f"approximate inflation-adjusted change {cpi.inflation_adjusted_change:+,.2f} "
        f"(prior CPI {cpi.prior_cpi_value:.3f} on {', '.join(cpi.prior_cpi_observation_dates)}, "
        f"current CPI {cpi.current_cpi_value:.3f} on {', '.join(cpi.current_cpi_observation_dates)}). "
        f"{cpi.non_causality_caveat}"
    )


def format_target_comparison(findings: BaselineFindings) -> str:
    """Render the optional Postgres target-vs-actual comparison. When the
    optional Data Source was not part of this run, this is silent — when it
    was attempted but unavailable/invalid, the recorded status and the
    matching limitation are both surfaced so the degradation is visible."""
    comparison = findings.target_comparison
    if comparison is None:
        return "Sales targets: not part of this run."
    status = comparison.source_status
    if status.status != "available":
        detail = f" ({status.detail})" if status.detail else ""
        return f"Sales targets: {status.status}{detail}."
    if not comparison.rows:
        return "Sales targets: available, no region/category rows to compare."
    lines = ["Sales targets (actual vs. target, by region and category):"]
    for row in comparison.rows:
        revenue_percent = "n/a" if row.net_revenue_variance_percent is None else f"{row.net_revenue_variance_percent:+.1f}%"
        orders_percent = "n/a" if row.orders_variance_percent is None else f"{row.orders_variance_percent:+.1f}%"
        lines.append(
            f"  - {row.region}/{row.category}: net revenue {row.actual_net_revenue:,.2f} vs "
            f"target {row.target_net_revenue:,.2f} ({row.net_revenue_variance:+,.2f}, {revenue_percent}); "
            f"orders {row.actual_orders:,.0f} vs target {row.target_orders:,.0f} "
            f"({row.orders_variance:+,.0f}, {orders_percent})"
        )
    return "\n".join(lines)


def format_revenue_bridge(findings: BaselineFindings) -> str:
    """Render the net-revenue bridge as a concise, auditable line per effect —
    Priya sees how the components reconcile rather than a single number."""
    bridge = findings.revenue_bridge
    if bridge is None:
        return "Revenue bridge: unavailable."
    lines = [
        "Revenue bridge (reconciles to net revenue change "
        f"{findings.net_revenue.absolute_change:+,.2f}, gap {bridge.reconciliation_gap:.4f}):",
        f"  - volume: {bridge.volume_effect:+,.2f}",
        f"  - price: {bridge.price_effect:+,.2f}",
        f"  - price-volume interaction: {bridge.interaction_effect:+,.2f}",
        f"  - discount: {bridge.discount_effect:+,.2f}",
        f"  - return: {bridge.return_effect:+,.2f}",
        f"  - new products ({bridge.new_product_count}): {bridge.new_product_effect:+,.2f}",
        f"  - discontinued products ({bridge.discontinued_product_count}): {bridge.discontinued_product_effect:+,.2f}",
        f"  - total: {bridge.total_effect:+,.2f} (continuing products: {bridge.continuing_product_count})",
    ]
    return "\n".join(lines)


def format_material_changes(findings: BaselineFindings) -> str:
    """Highlight only the drivers that pass the resolved Material Change
    thresholds — the many non-material dimension rows are deliberately left
    out of the CLI summary so real noise doesn't drown out real signal."""
    materiality = findings.materiality
    header = (
        f"Material Changes (>= {materiality.relative_change_percent:g}% relative and "
        f">= {materiality.absolute_change_percent_of_prior_total:g}% of prior-period net revenue, "
        f"thresholds: {materiality.source}):"
    )
    if not findings.material_changes:
        return f"{header} none — no dimension driver met both thresholds this period."
    lines = [header]
    for driver in findings.material_changes:
        percent = "n/a (no prior baseline)" if driver.percent_change is None else f"{driver.percent_change:+.1f}%"
        contribution = "n/a" if driver.contribution_percent is None else f"{driver.contribution_percent:+.1f}%"
        lines.append(
            f"  - {driver.dimension}={driver.value}: {driver.current:,.2f} vs {driver.prior:,.2f} "
            f"({driver.absolute_change:+,.2f}, {percent}, contribution {contribution} of net change)"
        )
    return "\n".join(lines)


def seed_memory_before_invoke(fact: str):
    from analyst_crew.tools.memory import seed_memory

    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if conn_string:
        from langgraph.store.postgres import PostgresStore

        with PostgresStore.from_conn_string(conn_string) as store:
            store.setup()
            seed_memory(store, fact)
    else:
        print(
            "Warning: POSTGRES_CONNECTION_STRING not set. "
            "Seeded memory requires Postgres to persist across sessions. "
            "Continuing without the seeded memory.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyst Crew CLI")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--user-id", type=str, default="default")
    parser.add_argument("--memory-seed", type=str, default=None)
    parser.add_argument("--sales-csv", type=str, default=None)
    parser.add_argument("--artifact-dir", type=str, default=".artifacts")
    parser.add_argument("--as-of-date", type=str, required=True)
    parser.add_argument("--report-file", type=str, default=None,
                        help="also write the rendered Report to this Markdown file (nothing is written otherwise)")
    args = parser.parse_args()

    if not args.sales_csv:
        parser.error("--sales-csv is required for Phase 1 execution")

    if args.thread_id is None:
        args.thread_id = str(uuid.uuid4())

    if args.question:
        question = args.question
    elif not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    else:
        print("Error: --question is required or pipe input via stdin", file=sys.stderr)
        sys.exit(1)

    if args.memory_seed is not None:
        seed_memory_before_invoke(args.memory_seed)

    invoke_config = {
        "configurable": {
            "thread_id": args.thread_id,
            "user_id": args.user_id,
        }
    }
    if args.sales_csv:
        from analyst_crew.tools.sales import sales_data_source
        from analyst_crew.tools.targets import sales_targets_data_source
        from analyst_crew.tools.fred import fred_cpi_data_source
        invoke_config["configurable"]["data_sources"] = {
            **sales_data_source(args.sales_csv),
            **sales_targets_data_source(),
            **fred_cpi_data_source(),
        }
        invoke_config["configurable"]["artifact_dir"] = args.artifact_dir
        invoke_config["configurable"]["analysis_parameters"] = {
            "as_of_date": args.as_of_date,
        }

    try:
        result = invoke(question, invoke_config)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_result(result))

    artifacts = result.get("artifacts", {})
    if args.report_file:
        report = rendered_report(result)
        if report is None:
            print("No Report was produced, so no file was written.", file=sys.stderr)
        else:
            path = Path(args.report_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(report, encoding="utf-8")
            print(f"Report saved to {path}", file=sys.stderr)
            for chart_path in save_charts(result, path.parent):
                print(f"Chart saved to {chart_path}", file=sys.stderr)

    terminal_failure = artifacts.get("terminal_failure")
    required_failure = terminal_failure is not None and terminal_failure.get("required", True)
    if required_failure or artifacts.get("plan_error") is not None:
        sys.exit(1)


if __name__ == "__main__":
    main()
