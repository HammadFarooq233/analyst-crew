"""Shared harness for tests that drive the whole Crew graph through ``invoke()``
with an in-memory checkpointer, a CSV built in a temporary directory, and
scripted models injected through the per-Agent model overrides."""
import csv
import re

from langgraph.checkpoint.memory import MemorySaver

import analyst_crew.agents.planner as planner_module
from analyst_crew.agents import Plan, Step
from analyst_crew.reporting.report import Claim, ReporterContent
from analyst_crew.tools.sales import CANONICAL_FIELDS, sales_data_source

QUESTION = "Compare this month's sales to last month and flag anything unusual"


def write_sales(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def sales_row(order_id, order_date, quantity=1, unit_price=10):
    return {
        "order_id": order_id, "order_date": order_date, "product_id": "P001",
        "product_name": "Desk Lamp", "category": "Office", "region": "North",
        "sales_channel": "web", "customer_segment": "consumer",
        "quantity": quantity, "unit_price": unit_price,
        "discount_amount": 0, "returned_quantity": 0,
    }


def base_rows():
    """May net revenue 30, June 40: a +10.00 (+33.3%) change, every dimension a single value."""
    return [
        sales_row("M1", "2025-05-01"), sales_row("M2", "2025-05-05", quantity=2),
        sales_row("J1", "2025-06-01"), sales_row("J2", "2025-06-05", quantity=3),
    ]


def crew_plan(*, reporter=True):
    steps = [
        Step(step_id="fetch_sales", agent="Fetcher", objective="Fetch normalized sales",
             source_ids=["sales_csv"], expected_output="sales_dataset", declared_artifacts=["sales_dataset"]),
        Step(step_id="analyze_baseline", agent="Analyst", objective="Compare baseline sales periods",
             expected_output="baseline_findings", declared_artifacts=["baseline_findings"], depends_on=["fetch_sales"]),
    ]
    if reporter:
        steps.append(Step(
            step_id="write_report", agent="Reporter", objective="Write the cited Report",
            expected_output="sales_report", declared_artifacts=["sales_report"], depends_on=["analyze_baseline"],
        ))
    return Plan(steps=steps, user_question=QUESTION)


class ScriptedPlannerModel:
    def __init__(self, plan):
        self.plan = plan
        self.calls = 0

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _messages):
        self.calls += 1
        return self.plan


class ScriptedFetcherModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, _context):
        self.calls += 1
        return {"tool": "CSV.read", "source_id": "sales_csv"}


class ScriptedAnalystModel:
    def __init__(self):
        self.calls = 0

    def invoke(self, request):
        self.calls += 1
        artifact_id = next(iter(request["available_artifacts"].values()))
        return {"tool": "DuckDB.sales_baseline", "dataset_artifact_id": artifact_id}


class ScriptedReporterModel:
    """Plays back one response per call (the last repeats). A response may be a
    ``ReporterContent``, a dict, an Exception to raise, or a callable taking
    the Reporter's context and returning one of those."""

    def __init__(self, *responses):
        self.responses = list(responses) or [valid_content]
        self.contexts = []

    @property
    def calls(self):
        return len(self.contexts)

    def invoke(self, context):
        self.contexts.append(context)
        response = self.responses[min(len(self.contexts), len(self.responses)) - 1]
        if callable(response):
            response = response(context)
        if isinstance(response, BaseException):
            raise response
        return response


def entry_id(context, source):
    return next(entry["id"] for entry in context["evidence_catalog"] if entry["source"] == source)


def valid_content(context):
    """A Reporter response a real model could give: every number a placeholder,
    every Claim citing ids that exist in the catalog it was shown."""
    def ph(source):
        return "{" + entry_id(context, source) + "}"

    claims = []
    for group in dict.fromkeys(e["group"] for e in context["evidence_catalog"] if e["material"]):
        members = {e["source"].rsplit(".", 1)[1]: e["id"] for e in context["evidence_catalog"] if e["group"] == group}
        claims.append(Claim(
            text=f"{{{members['name']}}} moved net revenue by {{{members['absolute_change']}}}, "
                 f"{{{members['contribution_percent']}}} of the overall change.",
            evidence_ids=[members["name"], members["absolute_change"], members["contribution_percent"]],
        ))
    claims.append(Claim(
        text=f"Volume contributed {ph('revenue_bridge.volume_effect')} to the bridge.",
        evidence_ids=[entry_id(context, "revenue_bridge.volume_effect")],
    ))
    return ReporterContent(
        summary=Claim(
            text=f"Net revenue changed by {ph('net_revenue.absolute_change')} in {ph('window.current_period')} "
                 f"compared with {ph('window.prior_period')}.",
            evidence_ids=[entry_id(context, "net_revenue.absolute_change")],
        ),
        claims=claims,
    )


def invalid_unknown_id(context):
    content = valid_content(context)
    return content.model_copy(update={"claims": [Claim(text="Sales moved by {E9999}.", evidence_ids=["E9999"])]})


def invalid_stray_digit(context):
    content = valid_content(context)
    return content.model_copy(update={"summary": Claim(
        text="Net revenue grew 33 percent.", evidence_ids=[entry_id(context, "net_revenue.percent_change")],
    )})


USE_DEFAULT = object()


def run_crew(source, tmp_path, monkeypatch, *, thread_id="thread-1", as_of_date="2025-06-30", checkpointer=None,
             plan=None, reporter_model=USE_DEFAULT, planner_model=None, fetcher_model=None, analyst_model=None,
             extra_sources=None, available_agents=("Fetcher", "Analyst", "Reporter")):
    planner_model = planner_model or ScriptedPlannerModel(plan or crew_plan())
    fetcher_model = fetcher_model or ScriptedFetcherModel()
    analyst_model = analyst_model or ScriptedAnalystModel()
    if reporter_model is USE_DEFAULT:
        reporter_model = ScriptedReporterModel()
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    runtime = {"configurable": {
        "thread_id": thread_id,
        "data_sources": {**sales_data_source(source), **(extra_sources or {})},
        "artifact_dir": tmp_path / "artifacts",
        "available_agents": list(available_agents),
        "planner_model": planner_model,
        "fetcher_model": fetcher_model,
        "analyst_model": analyst_model,
        "analysis_parameters": {"as_of_date": as_of_date},
    }}
    if reporter_model is not None:
        runtime["configurable"]["reporter_model"] = reporter_model
    result = planner_module.invoke(QUESTION, runtime, checkpointer=checkpointer or MemorySaver())
    return result, reporter_model


def displays(catalog):
    return [entry.display for entry in catalog.entries]


def body_numbers(rendered):
    """Every number a reader sees in the Report body: outside the Evidence
    appendix and the `[E3]` citation markers."""
    body = rendered.split("## Evidence")[0]
    body = re.sub(r"`\[[^\]]*\]`", "", body)
    return re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", body)
