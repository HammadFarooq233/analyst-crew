"""Checkpoint serializer that allowlists the Crew's own state types.

LangGraph's msgpack serializer warns for every unregistered type it
deserializes from a checkpoint and rejects them under
``LANGGRAPH_STRICT_MSGPACK=true``. Every Pydantic model that can sit in
CrewState is registered here, discovered from the modules that define them.
"""
import inspect

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel



def crew_state_types() -> list[type[BaseModel]]:
    # Imported here because analyst_crew.agents imports this module.
    from analyst_crew.agents import contracts
    from analyst_crew.reporting import catalog, charts, critique, report
    from analyst_crew.tools import analysis

    return [
        cls
        for module in (contracts, analysis, catalog, charts, critique, report)
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, BaseModel) and cls.__module__ == module.__name__
    ]


def crew_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer().with_msgpack_allowlist(crew_state_types())
