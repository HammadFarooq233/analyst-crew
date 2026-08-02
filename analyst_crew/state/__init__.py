import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    merged = dict(left)
    merged.update(right)
    return merged


class CrewState(TypedDict):
    current_step: str
    steps_completed: Annotated[list[str], operator.add]
    artifacts: Annotated[dict[str, Any], merge_dicts]
    messages: Annotated[list, add_messages]