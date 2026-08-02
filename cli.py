import argparse
import os
import sys
import uuid

import config  # noqa: F401 — triggers load_dotenv()
from analyst_crew.agents import invoke


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
    args = parser.parse_args()

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

    try:
        result = invoke(question, invoke_config)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(format_plan(result))


if __name__ == "__main__":
    main()