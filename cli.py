import argparse
import os
import sys
import uuid

import config  # noqa: F401 — triggers load_dotenv()
from analyst_crew.agents import invoke
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


def format_result(result) -> str:
    findings = result.get("artifacts", {}).get("findings")
    if findings is None:
        return format_plan(result)
    if not isinstance(findings, BaselineFindings):
        findings = BaselineFindings.model_validate(findings)
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
        f"{format_material_changes(findings)}"
    )


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
        invoke_config["configurable"]["data_sources"] = sales_data_source(args.sales_csv)
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


if __name__ == "__main__":
    main()
