"""Single place that decides which chat-model provider the Crew talks to.

``LLM_PROVIDER`` selects ``azure`` (Azure OpenAI / Microsoft Foundry) or
``openrouter`` (the default, preserving the original behavior).
"""

import os
from typing import Any

from langchain_openai import AzureChatOpenAI, ChatOpenAI

import config

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_AZURE_VARS = (
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
)


def provider() -> str:
    return os.getenv("LLM_PROVIDER", "openrouter").strip().lower()


def missing_credentials() -> list[str]:
    """Names of the env vars the active provider still needs (empty if ready)."""
    if provider() == "azure":
        return [name for name in _AZURE_VARS if not os.getenv(name)]
    return [] if os.getenv("OPENROUTER_API_KEY") else ["OPENROUTER_API_KEY"]


def credentials_available() -> bool:
    return not missing_credentials()


def make_chat_model(
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Any:
    """Build the configured chat model, or raise a clear error if it can't be."""
    missing = missing_credentials()
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} not set for LLM_PROVIDER={provider()}. "
            "Add it to your .env file or environment."
        )
    temperature = temperature if temperature is not None else float(config.get("llm.temperature", 0.0))
    max_tokens = max_tokens or int(config.get("llm.max_tokens", 4096))

    if provider() == "azure":
        return AzureChatOpenAI(
            azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return ChatOpenAI(
        model=model or str(config.get("llm.model", "google/gemma-4-26b-a4b-it:free")),
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
        temperature=temperature,
        max_tokens=max_tokens,
        default_headers={
            "HTTP-Referer": "https://github.com/analyst-crew",
            "X-Title": "Analyst Crew",
        },
    )
