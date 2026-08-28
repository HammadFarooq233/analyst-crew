from analyst_crew.tools.sales import CANONICAL_FIELDS, fetch_csv, fetcher_tool, generate_sales_csv, sales_data_source
from analyst_crew.agents import Crew, Plan, Step, resolve_handoff
import analyst_crew.agents.planner as planner_module
import hashlib
import json

class ScriptedFetcherModel:
    def invoke(self, request):
        return {"tool": "CSV.read", "source_id": "sales_csv"}
import pytest

def test_generator_is_reproducible(tmp_path):
    a = generate_sales_csv(tmp_path / "a.csv", rows=25_000); b = generate_sales_csv(tmp_path / "b.csv", rows=25_000)
    assert a.read_bytes() == b.read_bytes(); assert a.read_text().splitlines()[0].split(",") == list(CANONICAL_FIELDS)

def test_csv_catalog_and_artifact_are_content_addressed(tmp_path):
    pytest.importorskip("pyarrow")
    source = generate_sales_csv(tmp_path / "sales.csv", rows=25_000); catalog = sales_data_source(source); first = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts"); second = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    assert catalog["sales_csv"]["kind"] == "CSV"; assert first.artifact_id == second.artifact_id; assert first.content_hash == second.content_hash; assert first.row_count == 25_000; assert first.location.endswith(".parquet")
    assert first.input_hash and first.source_metadata["size_bytes"] > 0 and first.transform["version"] == 2

def test_empty_valid_input_fails_clearly(tmp_path):
    path = tmp_path / "empty.csv"; path.write_text(",".join(CANONICAL_FIELDS) + "\n" + ",".join(["bad"] * len(CANONICAL_FIELDS)) + "\n")
    with pytest.raises(ValueError, match="zero valid rows"):
        fetch_csv("sales_csv", path, artifact_dir=tmp_path / "artifacts")

def test_fetcher_consumes_resolved_handoff_and_returns_state_reference(tmp_path):
    pytest.importorskip("pyarrow")
    source = generate_sales_csv(tmp_path / "sales.csv", rows=25_000)
    step = Step(step_id="fetch", agent="Fetcher", objective="fetch", source_ids=["sales_csv"], expected_output="sales")
    plan = Plan(steps=[step], user_question="q")
    handoff = Crew(["Fetcher"], sales_data_source(source)).resolve_handoff(step, plan, {})
    artifact = fetcher_tool(handoff, artifact_dir=tmp_path / "artifacts")
    reference = artifact.as_reference()
    assert reference["kind"] == "Dataset Artifact" and "location" in reference
    assert "rows" not in reference

def test_graph_planner_fetcher_stores_reference_not_rows(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    source = generate_sales_csv(tmp_path / "sales.csv", rows=25_000)
    step = Step(step_id="fetch", agent="Fetcher", objective="fetch", source_ids=["sales_csv"], expected_output="sales")
    plan = Plan(steps=[step], user_question="fetch sales")
    monkeypatch.setattr(planner_module, "planner_node", lambda state, config=None: {"current_step": "planner_node", "artifacts": {"plan": plan}})
    result = planner_module.build_graph(runtime_config={"configurable": {"data_sources": sales_data_source(source), "artifact_dir": tmp_path / "artifacts", "available_agents": ["Fetcher"], "fetcher_model": ScriptedFetcherModel()}}).invoke({"messages": []}, {"configurable": {"data_sources": sales_data_source(source), "artifact_dir": tmp_path / "artifacts", "available_agents": ["Fetcher"], "fetcher_model": ScriptedFetcherModel()}})
    assert result["artifacts"]["datasets"]
    assert result["artifacts"]["dataset_profile"]["included_rows"] == 25_000
    assert "rows" not in result["artifacts"]["datasets"]

def test_fetcher_model_rejects_disallowed_and_undeclared_requests(tmp_path):
    pytest.importorskip("pyarrow")
    source = generate_sales_csv(tmp_path / "sales.csv", rows=25_000)
    step = Step(step_id="fetch", agent="Fetcher", objective="fetch", source_ids=["sales_csv"], expected_output="sales")
    plan = Plan(steps=[step], user_question="q")
    handoff = Crew(["Fetcher"], sales_data_source(source)).resolve_handoff(step, plan, {})
    class BadModel:
        def __init__(self, response): self.response = response
        def invoke(self, request): return self.response
    from analyst_crew.tools.sales import FetcherAgent
    with pytest.raises(ValueError, match="disallowed"):
        FetcherAgent(BadModel({"tool": "execute_python", "source_id": "sales_csv"})).run(handoff)
    with pytest.raises(ValueError, match="undeclared"):
        FetcherAgent(BadModel({"tool": "CSV.read", "source_id": "other"})).run(handoff)

def _fetch_handoff(tmp_path):
    source = tmp_path / "sales.csv"
    source.write_text(",".join(CANONICAL_FIELDS) + "\n" + ",".join(["x"] * len(CANONICAL_FIELDS)))
    step = Step(step_id="fetch", agent="Fetcher", objective="fetch", source_ids=["sales_csv"], expected_output="sales")
    plan = Plan(steps=[step], user_question="q")
    return Crew(["Fetcher"], sales_data_source(source)).resolve_handoff(step, plan, {})

def test_fetcher_repair_and_tool_retry_policies(tmp_path):
    from analyst_crew.tools.sales import FetcherAgent, TransientFetcherError
    handoff = _fetch_handoff(tmp_path)
    class Model:
        def __init__(self, responses): self.responses = iter(responses); self.calls = 0
        def invoke(self, context): self.calls += 1; return next(self.responses)
    valid = {"tool": "CSV.read", "source_id": "sales_csv"}
    model = Model([{"tool": "bad"}, valid])
    calls = []
    def transient_then_success(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1: raise TransientFetcherError("temporary")
        return object()
    assert FetcherAgent(model, transient_then_success).run(handoff) is not None
    assert model.calls == 2 and len(calls) == 2
    repeated = Model([{"tool": "bad"}, {"tool": "also-bad"}, valid])
    with pytest.raises(ValueError): FetcherAgent(repeated, transient_then_success).run(handoff)
    assert repeated.calls == 2
    exhausted_calls = []
    def always_transient(*args, **kwargs):
        exhausted_calls.append(1); raise TimeoutError("temporary")
    with pytest.raises(TimeoutError): FetcherAgent(Model([valid]), always_transient).run(handoff)
    assert len(exhausted_calls) == 2
    validation_calls = []
    def permanent_failure(*args, **kwargs):
        validation_calls.append(1); raise ValueError("schema failure")
    with pytest.raises(ValueError): FetcherAgent(Model([valid]), permanent_failure).run(handoff)
    assert len(validation_calls) == 1

def test_fetcher_node_constructs_production_model_when_not_injected(monkeypatch):
    from analyst_crew.tools.sales import FetcherAgent
    sentinel = object()
    monkeypatch.setattr(planner_module, "production_fetcher_model", lambda: sentinel)
    monkeypatch.setattr(planner_module, "FetcherAgent", lambda model: type("Agent", (), {"run": lambda self, handoff, artifact_dir: (_ for _ in ()).throw(ValueError("expected test stop"))})())
    step = Step(step_id="fetch", agent="Fetcher", objective="fetch", source_ids=["sales_csv"], expected_output="sales")
    result = planner_module.fetcher_node({"artifacts": {"plan": Plan(steps=[step], user_question="q")}}, {"configurable": {"data_sources": {"sales_csv": {"kind": "CSV", "location": "/tmp/sales.csv"}}, "fetcher_model": None}})
    assert result["artifacts"]["terminal_failure"]["agent"] == "Fetcher"

def test_artifact_hash_is_parquet_bytes_and_corruption_is_detectable(tmp_path):
    pytest.importorskip("pyarrow")
    source = generate_sales_csv(tmp_path / "sales.csv", rows=25_000)
    artifact = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    assert hashlib.sha256(open(artifact.location, "rb").read()).hexdigest() == artifact.content_hash
    with open(artifact.location, "ab") as handle: handle.write(b"corrupt")
    assert hashlib.sha256(open(artifact.location, "rb").read()).hexdigest() != artifact.content_hash
    repaired = fetch_csv("sales_csv", source, artifact_dir=tmp_path / "artifacts")
    assert hashlib.sha256(open(repaired.location, "rb").read()).hexdigest() == repaired.content_hash
    assert repaired.artifact_id == artifact.artifact_id
    assert "quarantine_records" not in artifact.profile and "quarantine_records" not in artifact.as_reference()

def test_custom_generator_provenance_and_cli_requires_source(monkeypatch, tmp_path):
    import cli
    with pytest.raises(SystemExit):
        monkeypatch.setattr("sys.argv", ["cli"]); cli.main()
    captured = {}
    monkeypatch.setattr(cli, "invoke", lambda question, config: captured.update(config) or {"artifacts": {}})
    monkeypatch.setattr("sys.argv", ["cli", "--question", "q", "--sales-csv", str(tmp_path / "sales.csv"), "--artifact-dir", str(tmp_path / "artifacts")])
    cli.main()
    assert captured["configurable"]["data_sources"]["sales_csv"]["kind"] == "CSV"
    assert captured["configurable"]["artifact_dir"] == str(tmp_path / "artifacts")
