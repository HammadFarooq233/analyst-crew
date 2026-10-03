"""Issue #13 — the CLI prints the rendered Report by default, writes a Markdown
file only when asked, and shows a Fallback Report with a concise failure line
when the Reporter fails. Driven through ``cli.main()`` against the real graph
with scripted models injected into the config ``cli`` builds."""
import re

import pytest
from langgraph.checkpoint.memory import MemorySaver

import analyst_crew.agents as agents
import cli
from tests.crew_harness import (
    FULL_AGENTS, QUESTION, ScriptedAnalystModel, ScriptedFetcherModel, ScriptedPlannerModel, ScriptedReporterModel, base_rows,
    crew_plan, write_sales,
)


@pytest.fixture(autouse=True)
def _engines():
    pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")


def _run_cli(monkeypatch, tmp_path, capsys, *extra_args, reporter=None, plan=None, available_agents=None):
    source = write_sales(tmp_path / "sales.csv", base_rows())
    reporter = reporter or ScriptedReporterModel()
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    def scripted_invoke(question, config):
        config["configurable"].update(
            planner_model=ScriptedPlannerModel(plan or crew_plan()), fetcher_model=ScriptedFetcherModel(),
            analyst_model=ScriptedAnalystModel(), reporter_model=reporter,
        )
        # A Crew without the Visualizer unless a test asks for one: the default Plan has no Visualizer Step.
        config["configurable"]["available_agents"] = available_agents or ["Fetcher", "Analyst", "Reporter"]
        return agents.invoke(question, config, checkpointer=MemorySaver())

    monkeypatch.setattr(cli, "invoke", scripted_invoke)
    monkeypatch.setattr("sys.argv", [
        "cli", "--question", QUESTION, "--sales-csv", str(source), "--artifact-dir", str(tmp_path / "artifacts"),
        "--as-of-date", "2025-06-30", *extra_args,
    ])
    exit_code = 0
    try:
        cli.main()
    except SystemExit as exc:
        exit_code = exc.code or 0
    captured = capsys.readouterr()
    return captured.out, captured.err, exit_code


def test_prints_the_rendered_report_by_default_instead_of_the_findings_dump(monkeypatch, tmp_path, capsys):
    out, _, exit_code = _run_cli(monkeypatch, tmp_path, capsys)

    assert exit_code == 0
    for heading in ("## Summary", "## Findings", "## Charts", "## Caveats", "## Methodology"):
        assert heading in out
    assert "+10.00" in out
    assert "Baseline sales Findings" not in out and "Fallback Report" not in out
    assert not list(tmp_path.glob("*.md")), "no file is written unless requested"


def test_writes_the_rendered_report_to_a_markdown_file_only_when_asked(monkeypatch, tmp_path, capsys):
    target = tmp_path / "out" / "report.md"

    out, err, exit_code = _run_cli(monkeypatch, tmp_path, capsys, "--report-file", str(target))

    assert exit_code == 0
    assert target.read_text(encoding="utf-8").strip() == out.strip()
    assert target.read_text(encoding="utf-8").startswith("# Analyst Crew Report")
    assert str(target) in err


def test_reporter_failure_prints_a_marked_fallback_report_and_a_concise_failure_line(monkeypatch, tmp_path, capsys):
    out, _, exit_code = _run_cli(monkeypatch, tmp_path, capsys, reporter=ScriptedReporterModel(RuntimeError("model down")))

    assert "Fallback Report" in out and "without the Reporter Agent" in out
    assert "## Summary" in out and "+10.00" in out and "+33.3%" in out
    failure_lines = [line for line in out.splitlines() if "Reporter" in line and "model down" in line]
    assert len(failure_lines) == 1 and "failed" in failure_lines[0]
    assert exit_code == 0, "a delivered Fallback Report is a degraded success, not a failed run"


def test_fallback_report_is_saved_to_the_requested_file_too(monkeypatch, tmp_path, capsys):
    target = tmp_path / "fallback.md"

    _run_cli(monkeypatch, tmp_path, capsys, "--report-file", str(target), reporter=ScriptedReporterModel(RuntimeError("x")))

    assert "without the Reporter Agent" in target.read_text(encoding="utf-8")


def test_required_source_failures_still_print_only_the_terminal_failure_and_exit_nonzero(monkeypatch, tmp_path, capsys):
    class BrokenFetcher:
        def invoke(self, _context):
            return {"tool": "execute_python", "source_id": "sales_csv"}

    source = write_sales(tmp_path / "sales.csv", base_rows())
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "")
    monkeypatch.setenv("LANGFUSE_HOST", "")
    monkeypatch.setattr(cli, "invoke", lambda question, config: agents.invoke(question, {"configurable": {
        **config["configurable"], "available_agents": ["Fetcher", "Analyst", "Reporter"],
        "planner_model": ScriptedPlannerModel(crew_plan()), "fetcher_model": BrokenFetcher(),
        "analyst_model": ScriptedAnalystModel(), "reporter_model": ScriptedReporterModel(),
    }}, checkpointer=MemorySaver()))
    monkeypatch.setattr("sys.argv", ["cli", "--question", QUESTION, "--sales-csv", str(source),
                                     "--artifact-dir", str(tmp_path / "a"), "--as-of-date", "2025-06-30"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    out = capsys.readouterr().out
    assert exit_info.value.code == 1
    assert "Error: Fetcher" in out and "## Summary" not in out


def test_a_stale_report_never_hides_a_later_required_failure():
    stale = {"artifacts": {
        "report": object(), "terminal_failure": {"agent": "Fetcher", "step_id": "f", "required": True, "error": "boom"},
    }}

    assert cli.format_result(stale).startswith("Error: Fetcher")


def test_without_a_reporter_step_the_findings_summary_is_still_printed(monkeypatch, tmp_path, capsys):
    out, _, exit_code = _run_cli(monkeypatch, tmp_path, capsys, plan=crew_plan(reporter=False),
                                 available_agents=["Fetcher", "Analyst"])

    assert exit_code == 0
    assert "Baseline sales Findings" in out and "## Summary" not in out


def _full_cli(monkeypatch, tmp_path, capsys, *extra_args):
    return _run_cli(monkeypatch, tmp_path, capsys, *extra_args, plan=crew_plan(visualizer=True),
                    available_agents=list(FULL_AGENTS))


def test_default_output_lists_chart_titles_and_paths(monkeypatch, tmp_path, capsys):
    out, _, exit_code = _full_cli(monkeypatch, tmp_path, capsys)

    assert exit_code == 0
    assert "## Charts" in out and "Reserved" not in out
    listing = out.split("Charts generated:")[1].strip().splitlines()
    assert len(listing) == 2
    assert "Net revenue bridge: prior to current period" in listing[0]
    assert "Top drivers of the net revenue change" in listing[1]
    for line in listing:
        path = line.rsplit(": ", 1)[1]
        assert path.endswith(".png") and (tmp_path / "artifacts" / "charts" / path.rsplit("/", 1)[1]).is_file()


def test_saving_the_report_writes_the_pngs_beside_the_markdown_so_image_links_resolve(monkeypatch, tmp_path, capsys):
    target = tmp_path / "out" / "report.md"

    _full_cli(monkeypatch, tmp_path, capsys, "--report-file", str(target))

    links = re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", target.read_text(encoding="utf-8"))
    assert len(links) == 2
    for link in links:
        assert (target.parent / link).read_bytes().startswith(b"\x89PNG")


def test_a_report_without_charts_saves_no_pngs(monkeypatch, tmp_path, capsys):
    target = tmp_path / "out" / "report.md"

    out, _, _ = _run_cli(monkeypatch, tmp_path, capsys, "--report-file", str(target))

    assert not list(target.parent.glob("*.png")) and "Charts generated:" not in out
