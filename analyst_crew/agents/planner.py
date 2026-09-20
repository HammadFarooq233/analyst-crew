from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.config import get_store

import config
from analyst_crew.agents.contracts import (  # noqa: F401 — re-exported for existing importers
    AGENTS, CREW_AGENTS, PHASE1_AGENTS, PHASE2_AGENTS, ArtifactBinding, Crew, Handoff, Plan, SourceReference, Step, resolve_handoff,
)
from analyst_crew.agents.resume import _resume_decision, _source_content_hash, execution_key  # noqa: F401
from analyst_crew.llm import make_chat_model
from analyst_crew.state import CrewState
from analyst_crew.tools.retry import invoke_with_optional_config


SYSTEM_PROMPT = """You are the Planner Agent for Analyst Crew, a multi-agent data analysis copilot.

Your role is to decompose a user's analytical question into an ordered Plan of Steps.
Each Step names a target Agent, describes what it produces, and lists which prior Steps it depends on.

Available Agents:
- Fetcher: Retrieves data from sources (CSV, Postgres, FRED API). Returns raw data — no analysis, no interpretation.
- Analyst: Performs computation on data — statistics, joins, comparisons, trend detection. Consumes data, produces findings.
- Visualizer: Generates charts and tables from findings.
- Critic: Reviews a draft report for uncited claims, missing caveats, and logical gaps.
- Reporter: Writes the narrative report from findings and charts. Synthesizes results into prose, cites specific numbers.

Any relevant facts, preferences, or conventions the user has established in prior sessions are supplied below under "Relevant memory," already retrieved for you — incorporate them into your Plan decisions. If none are supplied, there is no prior memory to draw on.

Rules:
1. Produce only Steps that can execute with the validated capabilities for this run.
2. Each Step must have a unique step_id (snake_case string).
3. Each Step must name exactly one Agent from the list above.
4. depends_on must reference only step_ids from earlier Steps in the plan (never itself, never a later Step).
5. The Steps must form a valid dependency order — no cycles, no forward references.
6. For the sales comparison, produce exactly a Fetcher Step followed by a dependent Analyst Step, and, when Reporter is among the validated capabilities for this run, a final Reporter Step that depends on the Analyst Step.
7. The Fetcher must use a supplied Data Source and declare one Dataset Artifact output. The Analyst must depend on that Fetcher output and declare Findings. The Reporter, when present, must depend only on the Analyst Step, declare one Report output, and have empty source_ids.
8. Include the exact user_question text in the Plan.
9. Use these Step keys: `step_id`, `agent`, `objective`, `source_ids`, `expected_output`, `declared_artifacts`, and `depends_on`.
10. Never invent a Data Source ID, filesystem location, executable SQL, or arithmetic result.
11. `source_ids` lists only IDs from the "Data Sources" list above — never a `declared_artifacts` name from this or any other Step. A Step that consumes another Step's output declares that through `depends_on`, not `source_ids`. Only a Fetcher Step has a non-empty `source_ids`; every other Step's `source_ids` must be empty.
12. Respond with a single JSON object (the Plan) and nothing else — no prose, no markdown fences."""


class PlannerAgent:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        store=None,
        available_agents: Sequence[str] | None = None,
        data_sources: Mapping[str, Any] | Sequence[str] = (),
        llm=None,
    ):
        self.available_agents = set(CREW_AGENTS if available_agents is None else available_agents)
        invalid = self.available_agents - set(AGENTS)
        if invalid:
            raise ValueError(f"Unknown available Agents: {sorted(invalid)}")
        self.data_sources = set(data_sources.keys()) if isinstance(data_sources, Mapping) else set(data_sources)
        self.model = model or str(config.get("llm.model", "liquid/lfm-2.5-2.6b:free"))
        self.temperature = (
            temperature
            if temperature is not None
            else float(config.get("llm.temperature", 0.0))
        )
        self.max_tokens = max_tokens or int(config.get("llm.max_tokens", 4096))

        self._llm = llm or make_chat_model(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        self.validation_retries = 0
        self.last_validation_error: str | None = None

        self._memory_tool = None
        if store is not None:
            from analyst_crew.tools.memory import create_memory_tool

            self._memory_tool = create_memory_tool(store)

        self._chain = self._llm.with_structured_output(
            Plan,
            method="json_mode",
            include_raw=True,
        )

    def _search_memory(self, question: str) -> str:
        """Execute the search_memory Tool deterministically, application-side.

        Structured-output ``json_mode`` forces the model into single-shot
        JSON generation, which is incompatible with letting the model choose
        to make a real tool call mid-response — free-tier models either
        fabricate a fake tool-call JSON blob in place of a Plan, or emit a
        genuine (but then silently dropped) tool_calls response and no
        content at all. Retrieving memory here, before the Plan call, avoids
        depending on model-driven tool selection for this step.
        """
        if self._memory_tool is None:
            return ""
        try:
            return str(self._memory_tool.invoke({"query": question}))
        except Exception:
            return ""

    def generate_plan(self, question: str, available_agents: Sequence[str] | None = None,
                      data_sources: Mapping[str, Any] | Sequence[str] | None = None,
                      config: dict | None = None) -> Plan:
        if not question.strip():
            raise ValueError("Question must not be empty.")

        allowed_agents = set(available_agents) if available_agents is not None else self.available_agents
        invalid = allowed_agents - set(AGENTS)
        if invalid:
            raise ValueError(f"Unknown available Agents: {sorted(invalid)}")
        if not allowed_agents:
            raise ValueError("available_agents must include at least one Agent")
        source_catalog = (set(data_sources.keys()) if isinstance(data_sources, Mapping)
                          else set(data_sources)) if data_sources is not None else self.data_sources
        capability_prompt = (
            SYSTEM_PROMPT
            + "\n\nValidated capabilities for this run (you MUST use only these Agents):\n"
            + "Agents: " + ", ".join(sorted(allowed_agents))
            + "\nData Sources: " + ", ".join(sorted(source_catalog))
            + "\n\nRelevant memory:\n" + (self._search_memory(question) or "(none found)")
        )
        messages = [
            SystemMessage(content=capability_prompt),
            HumanMessage(content=question),
        ]

        catalog = (set(data_sources.keys()) if isinstance(data_sources, Mapping) else set(data_sources)) if data_sources is not None else self.data_sources

        def _validate(plan: Plan) -> None:
            if not isinstance(plan, Plan):
                raise ValueError(f"LLM returned unexpected type {type(plan).__name__}, expected Plan.")
            if not plan.steps:
                raise ValueError("LLM returned an empty Plan (no Steps).")
            unavailable = plan.agents - allowed_agents
            if unavailable:
                raise ValueError(f"Plan references unavailable Agents: {sorted(unavailable)}")
            unknown_sources = {source for step in plan.steps for source in step.source_ids} - catalog
            if unknown_sources:
                raise ValueError(f"Plan references unknown Data Sources: {sorted(unknown_sources)}")
            if allowed_agents in (set(PHASE1_AGENTS), set(PHASE2_AGENTS)) and "sales_csv" in catalog:
                expected_agents = ["Fetcher", "Analyst", "Reporter"] if "Reporter" in allowed_agents else ["Fetcher", "Analyst"]
                if [step.agent for step in plan.steps] != expected_agents:
                    raise ValueError(
                        "Sales Plan must contain exactly " + ", ".join(expected_agents[:-1])
                        + f" then {expected_agents[-1]} Steps"
                    )
                fetch_step, analyst_step = plan.steps[0], plan.steps[1]
                if fetch_step.source_ids != ["sales_csv"]:
                    raise ValueError("Phase 1 Fetcher must use the catalogued sales_csv Data Source")
                if analyst_step.depends_on != [fetch_step.step_id]:
                    raise ValueError("Phase 1 Analyst must depend on the Fetcher Step")
                if not (fetch_step.declared_artifacts or [fetch_step.expected_output]):
                    raise ValueError("Phase 1 Fetcher must declare a Dataset Artifact output")
                if not (analyst_step.declared_artifacts or [analyst_step.expected_output]):
                    raise ValueError("Phase 1 Analyst must declare a Findings output")
                if len(plan.steps) == 3:
                    reporter_step = plan.steps[2]
                    if reporter_step.depends_on != [analyst_step.step_id]:
                        raise ValueError("Reporter must depend on the Analyst Step")
                    if reporter_step.source_ids:
                        raise ValueError("Reporter Step must not name Data Sources; it consumes Findings")

        # Contract-invalid Plan output — malformed JSON or a structurally
        # valid Plan that still violates a Planner rule (unavailable Agent,
        # unknown Data Source, wrong Phase 1 shape) — receives at most one
        # repair retry, with the validation error fed back into the next
        # model call as feedback. Same policy already used for Fetcher/
        # Analyst Agent output (see FetcherAgent.run, AnalystAgent.run).
        # Free-tier models are more prone to this than larger ones, so this
        # matters more here even though the policy itself is pre-existing.
        result = None
        feedback = ""
        for attempt in range(2):
            call_messages = messages if not feedback else messages + [
                HumanMessage(
                    content=f"Your previous response was invalid: {feedback} "
                    "Return the corrected Plan as a single JSON object with no extra text."
                )
            ]
            try:
                raw_result = invoke_with_optional_config(self._chain, call_messages, config)
            except Exception as e:
                msg = str(e)
                if "quota" in msg.lower() or "429" in msg:
                    raise RuntimeError(
                        "LLM provider quota or rate limit exceeded. "
                        "Try again later, or switch model/provider (config/settings.toml, LLM_PROVIDER)."
                    ) from e
                raise RuntimeError(f"Planner LLM call failed: {msg}") from e

            if not (isinstance(raw_result, dict) and "parsed" in raw_result):
                # Scripted test doubles (with_structured_output(...) -> self)
                # ignore include_raw and return a Plan directly — no repair
                # loop applies to those, they're not real LLM output.
                result = raw_result
                break

            candidate = raw_result.get("parsed")
            if candidate is None:
                self.last_validation_error = str(raw_result.get("parsing_error"))
            else:
                try:
                    _validate(candidate)
                    result = candidate
                    break
                except ValueError as exc:
                    self.last_validation_error = str(exc)

            if attempt:
                raise RuntimeError(f"Planner LLM call failed: {self.last_validation_error}")
            self.validation_retries += 1
            feedback = self.last_validation_error

        _validate(result)

        return result


def _extract_question(state: CrewState) -> str:
    messages = state.get("messages", [])
    if messages:
        return messages[0].content
    return ""


def _reusable_prior_plan(state: CrewState, question: str, agent: "PlannerAgent") -> Plan | None:
    """Return the Plan already in the checkpointed state when it is still valid.

    Resuming a thread must not re-plan: a live Planner words step ids and
    objectives differently on every call, and those fields are part of each
    Step's execution key, so re-planning would defeat Step reuse (issue #11).
    The prior Plan is reused only if it answers the same question and still
    fits this run's Agents and Data Sources; otherwise the Planner runs again.
    """
    prior = state.get("artifacts", {}).get("plan")
    if not isinstance(prior, Plan) or prior.user_question != question:
        return None
    if prior.agents - agent.available_agents:
        return None
    if {source for step in prior.steps for source in step.source_ids} - agent.data_sources:
        return None
    return prior


def planner_node(state: CrewState, config: dict | None = None) -> dict[str, Any]:
    question = _extract_question(state)

    try:
        configurable = (config or {}).get("configurable", {})
        planner_model = configurable.get("planner_model")
        try:
            store = get_store()
        except RuntimeError:
            store = None
        agent = PlannerAgent(
            store=None if planner_model is not None else store,
            available_agents=configurable.get("available_agents"),
            data_sources=configurable.get("data_sources", ()),
            llm=planner_model,
        )
        plan = _reusable_prior_plan(state, question, agent)
        if plan is None:
            plan = agent.generate_plan(question, config=config)
    except (RuntimeError, ValueError) as e:
        return {
            "current_step": "planner_node",
            "artifacts": {"plan_error": str(e)},
        }

    return {
        "current_step": "planner_node",
        "artifacts": {"plan": plan},
    }


_GRAPH_LEVEL_NAMES = {
    "invoke": "analyst_crew.agents.graph",
    "build_graph": "analyst_crew.agents.graph",
    "fetcher_node": "analyst_crew.agents.fetcher",
    "analyst_node": "analyst_crew.agents.analyst",
    "get_langfuse_handler": "analyst_crew.agents.tracing",
}


def __getattr__(name: str) -> Any:
    """Backward-compatible access to the graph-level names extracted out of
    this module. Resolved lazily because those modules import this one."""
    module_name = _GRAPH_LEVEL_NAMES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
