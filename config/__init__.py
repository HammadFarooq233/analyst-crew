import os
import tomllib
from pathlib import Path
from dotenv import load_dotenv

_HERE = Path(__file__).parent

load_dotenv()


def _find_config_dir() -> Path:
    return _HERE


def _load_toml(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


_settings = _load_toml(_find_config_dir() / "settings.toml")


def get(key: str, default: object = None) -> object:
    keys = key.split(".")
    node = _settings
    for k in keys:
        if isinstance(node, dict):
            node = node.get(k)
            if node is None:
                return default
        else:
            return default
    return node


def reload() -> None:
    global _settings
    _settings = _load_toml(_find_config_dir() / "settings.toml")