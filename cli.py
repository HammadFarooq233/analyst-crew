import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyst Crew CLI")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--thread-id", type=str, default=None)
    parser.add_argument("--user-id", type=str, default="default")
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