"""Deterministic synthetic sales source and CSV Fetcher Tool."""
from __future__ import annotations
import csv, hashlib, json, os, random
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import config
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from analyst_crew.tools.retry import invoke_with_optional_config, retry_with_backoff

CANONICAL_FIELDS = ("order_id", "order_date", "product_id", "product_name", "category", "region", "sales_channel", "customer_segment", "quantity", "unit_price", "discount_amount", "returned_quantity")
DEFAULT_SEED = 20260301

class TransientFetcherError(Exception):
    """Retryable Fetcher Tool failure."""

class FetcherToolRequest(BaseModel):
    tool: str
    source_id: str

def production_fetcher_model() -> Any:
    """Build the live Fetcher Tool-calling seam.

    ``FetcherAgent.run`` invokes this with a plain context ``dict`` (step
    objective, declared sources, allowed Tools, validation feedback) — a
    ``ChatOpenAI`` structured-output runnable only accepts a PromptValue,
    str, or list of ``BaseMessage``, so this wraps that dict into a single
    formatted prompt before delegating to the structured call.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    model = ChatOpenAI(model=config.get("llm.model", "google/gemma-4-26b-a4b-it:free"), api_key=api_key, base_url="https://openrouter.ai/api/v1", temperature=0, max_tokens=int(config.get("llm.max_tokens", 4096)))
    structured = model.with_structured_output(FetcherToolRequest)

    def _invoke(context: dict[str, Any], config: Any = None) -> FetcherToolRequest:
        # ``config`` (accepted because RunnableLambda only forwards it when a
        # parameter literally named ``config`` is present) carries this
        # run's Langfuse CallbackHandler through from FetcherAgent.run, so
        # this call becomes a nested observation on the same trace as the
        # Planner/Analyst calls rather than an untracked side call.
        prompt = (
            "You are the Fetcher Tool-calling seam for Analyst Crew. "
            "Choose exactly one allowed Tool and one declared source_id — never invent either.\n\n"
            f"Step objective: {context.get('step_objective')}\n"
            f"User question: {context.get('user_question')}\n"
            f"Declared source_ids: {context.get('source_ids')}\n"
            f"Source descriptors: {context.get('source_descriptors')}\n"
            f"Allowed tools: {context.get('allowed_tools')}\n"
            f"Validation feedback from a prior attempt (empty if this is the first): "
            f"{context.get('validation_feedback') or 'none'}\n"
        )
        return structured.invoke(prompt, config=config)

    return RunnableLambda(_invoke)

@dataclass(frozen=True)
class DatasetArtifact:
    artifact_id: str; source_id: str; location: str; content_hash: str; schema: dict[str, str]; profile: dict[str, Any]; row_count: int; date_range: dict[str, str]; quality: dict[str, Any]; created_at: str; input_hash: str = ""; source_metadata: dict[str, Any] | None = None; generator: dict[str, Any] | None = None; transform: dict[str, Any] | None = None
    def as_reference(self) -> dict[str, Any]: return {"kind": "Dataset Artifact", **self.__dict__}

def generate_sales_csv(path: str | Path, *, seed: int = DEFAULT_SEED, rows: int = 30_000) -> Path:
    if not 25_000 <= rows <= 50_000: raise ValueError("rows must be between 25,000 and 50,000")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); rng = random.Random(seed); start = date(2025, 1, 1)
    products = [("P001", "Desk Lamp", "Office", 24), ("P002", "Notebook", "Office", 8), ("P003", "Ergonomic Chair", "Furniture", 280), ("P004", "Standing Desk", "Furniture", 520), ("P005", "Monitor", "Technology", 210), ("P006", "Keyboard", "Technology", 65)]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_FIELDS); writer.writeheader()
        for index in range(rows):
            day = date(2025, 1, 1) + timedelta(days=rng.randrange(181)); pid, name, category, base = rng.choice(products); quantity = rng.randint(1, 8); price = base * (1.04 if day.month >= 5 and category == "Technology" else 1); discount = round(price * quantity * rng.choice((0, 0, .05, .1)), 2); returned = 1 if rng.random() < .025 else 0
            writer.writerow({"order_id": f"ORD-{index + 1:06d}", "order_date": day.isoformat(), "product_id": pid, "product_name": name, "category": category, "region": rng.choice(("North", "South", "East", "West")), "sales_channel": rng.choice(("web", "marketplace", "store")), "customer_segment": rng.choice(("consumer", "small_business", "enterprise")), "quantity": quantity, "unit_price": f"{price:.2f}", "discount_amount": f"{discount:.2f}", "returned_quantity": returned})
        # Stable defects exercise quarantine and duplicate handling without reducing
        # the requested valid-order count.
        writer.writerow({"order_id": "ORD-000001", "order_date": start.isoformat(), "product_id": "P001", "product_name": "Desk Lamp", "category": "Office", "region": "North", "sales_channel": "web", "customer_segment": "consumer", "quantity": 1, "unit_price": "24", "discount_amount": "0", "returned_quantity": 0})
        writer.writerow({"order_id": "MALFORMED", "order_date": "not-a-date", "product_id": "P001", "product_name": "Desk Lamp", "category": "Office", "region": "North", "sales_channel": "web", "customer_segment": "consumer", "quantity": "-1", "unit_price": "24", "discount_amount": "0", "returned_quantity": 0})
    return path

def fetch_csv(source_id: str, csv_path: str | Path, *, artifact_dir: str | Path = ".artifacts", generator_metadata: dict[str, Any] | None = None) -> DatasetArtifact:
    source = Path(csv_path)
    if not source.is_file(): raise FileNotFoundError(f"CSV Data Source is unreadable: {source}")
    input_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANONICAL_FIELDS: raise ValueError(f"CSV Data Source must have canonical fields: {list(CANONICAL_FIELDS)}")
        valid, reasons, quarantine_records, seen = [], [], [], set()
        source_rows = 0
        for row in reader:
            source_rows += 1
            try:
                values = {key: (value.strip() if isinstance(value, str) else value) for key, value in dict(row).items()}
                for field in ("category", "region", "sales_channel", "customer_segment"): values[field] = values[field].casefold()
                values["quantity"] = int(values["quantity"]); values["unit_price"] = float(values["unit_price"]); values["discount_amount"] = float(values["discount_amount"]); values["returned_quantity"] = int(values["returned_quantity"]); datetime.strptime(values["order_date"], "%Y-%m-%d")
                if not values["order_id"]: reasons.append("missing_order_id"); quarantine_records.append({"row": values, "reason": "missing_order_id"}); continue
                if values["order_id"] in seen: reasons.append("duplicate_order_id"); quarantine_records.append({"row": values, "reason": "duplicate_order_id"}); continue
                if values["quantity"] <= 0: reasons.append("non_positive_quantity"); quarantine_records.append({"row": values, "reason": "non_positive_quantity"}); continue
                if values["unit_price"] < 0: reasons.append("negative_unit_price"); quarantine_records.append({"row": values, "reason": "negative_unit_price"}); continue
                if values["returned_quantity"] > values["quantity"]: reasons.append("return_exceeds_quantity"); quarantine_records.append({"row": values, "reason": "return_exceeds_quantity"}); continue
                if values["discount_amount"] > values["quantity"] * values["unit_price"]: reasons.append("discount_exceeds_gross"); quarantine_records.append({"row": values, "reason": "discount_exceeds_gross"}); continue
                seen.add(values["order_id"])
                valid.append(values)
            except (KeyError, TypeError, ValueError): reasons.append("malformed_row"); quarantine_records.append({"row": dict(row), "reason": "malformed_row"})
    if not valid: raise ValueError("CSV Data Source contains zero valid rows")
    if len(reasons) / (len(valid) + len(reasons)) > .01: raise ValueError(f"CSV Data Source has too many invalid rows: {len(reasons)}")
    transform = {"name": "canonical_csv_to_parquet", "version": 2, "normalization": "trim_and_casefold_dimensions"}; generator = generator_metadata or {}; source_metadata = {"source_id": source_id, "path": str(source), "size_bytes": source.stat().st_size}; identity = {"rows": valid, "schema": list(CANONICAL_FIELDS), "input_hash": input_hash, "source_metadata": source_metadata, "generator": generator, "transform": transform}; payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(); digest = hashlib.sha256(payload).hexdigest(); artifact_dir = Path(artifact_dir); artifact_dir.mkdir(parents=True, exist_ok=True); target = artifact_dir / f"{digest}.parquet"
    try:
        import pyarrow as pa; import pyarrow.parquet as pq
    except ImportError as exc: raise RuntimeError("CSV Fetcher requires pyarrow to write Parquet") from exc
    for row in valid: row["order_date"] = date.fromisoformat(row["order_date"])
    schema = pa.schema([pa.field("order_id", pa.string()), pa.field("order_date", pa.date32()), pa.field("product_id", pa.string()), pa.field("product_name", pa.string()), pa.field("category", pa.string()), pa.field("region", pa.string()), pa.field("sales_channel", pa.string()), pa.field("customer_segment", pa.string()), pa.field("quantity", pa.int64()), pa.field("unit_price", pa.float64()), pa.field("discount_amount", pa.float64()), pa.field("returned_quantity", pa.int64())]); table = pa.Table.from_pylist(valid, schema=schema)
    if not target.exists(): pq.write_table(table, target)
    byte_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    checksum = target.with_suffix(".sha256")
    if checksum.exists() and checksum.read_text(encoding="utf-8").strip() != byte_hash:
        pq.write_table(table, target); byte_hash = hashlib.sha256(target.read_bytes()).hexdigest()
    checksum.write_text(byte_hash, encoding="utf-8")
    dates = sorted(row["order_date"] for row in valid)
    quarantine_path = target.with_suffix(".quarantine.json")
    quarantine_path.write_text(json.dumps(quarantine_records, sort_keys=True), encoding="utf-8")
    def calculable_gross(record: dict[str, Any]) -> float:
        try:
            return float(record.get("quantity", 0)) * float(record.get("unit_price", 0))
        except (TypeError, ValueError):
            return 0.0
    profile = {"source_rows": source_rows, "included_rows": len(valid), "quarantined_rows": len(reasons), "duplicate_rows": reasons.count("duplicate_order_id"), "quarantine_reasons": {reason: reasons.count(reason) for reason in sorted(set(reasons))}, "excluded_revenue_impact": sum(calculable_gross(r["row"]) for r in quarantine_records), "quarantine_sidecar": str(quarantine_path), "null_counts": {field: sum(not row[field] for row in valid) for field in CANONICAL_FIELDS}}
    provenance = {"input_hash": input_hash, "source_metadata": source_metadata, "generator": generator, "transform": transform}
    return DatasetArtifact(f"dataset-{digest[:16]}", source_id, str(target), byte_hash, {field.name: str(field.type) for field in table.schema}, profile, len(valid), {"min": dates[0], "max": dates[-1]}, {"quarantined_rows": len(reasons), "duplicate_rows": profile["duplicate_rows"], "reasons": profile["quarantine_reasons"], "quarantine_sidecar": str(quarantine_path)}, datetime.now(timezone.utc).isoformat(), **provenance)

def sales_data_source(csv_path: str | Path, *, generator_metadata: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    source = {"kind": "CSV", "location": str(csv_path), "schema": list(CANONICAL_FIELDS)}
    if generator_metadata: source["generator"] = generator_metadata
    return {"sales_csv": source}

def fetcher_tool(handoff: Any, *, artifact_dir: str | Path = ".artifacts") -> DatasetArtifact:
    """Fetcher execution seam consuming only a resolved catalog Handoff."""
    if getattr(handoff.step, "agent", None) != "Fetcher": raise ValueError("Fetcher Tool requires a Fetcher Handoff")
    if len(handoff.source_values) != 1: raise ValueError("Fetcher requires exactly one resolved Data Source")
    source_id, source = next(iter(handoff.source_values.items())); reference = source.reference
    if not isinstance(reference, dict) or reference.get("kind") != "CSV": raise ValueError("Fetcher requires a catalogued CSV source")
    return fetch_csv(source_id, reference["location"], artifact_dir=artifact_dir, generator_metadata=reference.get("generator"))

class FetcherAgent:
    """Small validated tool-calling seam; production models can implement invoke().

    Contract-invalid Agent output (a disallowed Tool or an undeclared Data
    Source) receives at most one repair retry, with the validation error fed
    back into the next model call as feedback — never a second call to the
    Planner. Transient Tool failures (network timeouts, connection errors)
    receive a bounded retry via ``analyst_crew.tools.retry.retry_with_backoff``;
    any other Tool failure (e.g. a permission or schema error) is never
    retried. ``validation_retries``/``tool_retries`` record how many retries
    were actually used, for CrewState/tracing visibility.
    """
    allowed_tools = frozenset({"CSV.read"})
    max_tool_attempts = 2

    def __init__(self, model: Any, tool: Any = fetcher_tool):
        self.model = model
        self.tool = tool
        self.validation_retries = 0
        self.tool_retries = 0
        self.last_validation_error: str | None = None

    def run(self, handoff: Any, *, artifact_dir: str | Path = ".artifacts", config: Any = None) -> DatasetArtifact:
        context = {"step_objective": handoff.step.objective, "user_question": handoff.user_question, "source_ids": handoff.source_ids, "source_descriptors": {key: value.reference for key, value in handoff.source_values.items()}, "allowed_tools": sorted(self.allowed_tools)}
        feedback = ""
        for attempt in range(2):
            try:
                request = invoke_with_optional_config(self.model, {**context, "validation_feedback": feedback}, config)
                if isinstance(request, FetcherToolRequest): request = request.model_dump()
                if not isinstance(request, dict) or request.get("tool") not in self.allowed_tools: raise ValueError("disallowed Tool")
                if request.get("source_id") not in handoff.source_ids: raise ValueError("undeclared Data Source")
            except ValueError as exc:
                self.last_validation_error = str(exc)
                if attempt: raise
                self.validation_retries += 1
                feedback = f"Invalid Fetcher request: {exc}. Return only CSV.read and a declared source_id."
                continue
            result, outcome = retry_with_backoff(
                lambda: self.tool(handoff, artifact_dir=artifact_dir),
                is_transient=lambda exc: isinstance(exc, (TimeoutError, ConnectionError, TransientFetcherError)),
                max_attempts=self.max_tool_attempts,
                backoff_seconds=0.0,
            )
            self.tool_retries = outcome.retries
            return result
        raise RuntimeError("Fetcher request loop exhausted")
