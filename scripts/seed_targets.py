"""Seed the deterministic six-month Postgres sales-targets table.

Usage:
    python scripts/seed_targets.py

Reads the connection string from the same POSTGRES_CONNECTION_STRING
environment variable already used for the LangGraph checkpointer (see
.env.example), so no new credential scheme is introduced.
"""
from __future__ import annotations

import os
import sys

import config  # noqa: F401 — triggers load_dotenv()
from analyst_crew.tools.targets import seed_targets


def main() -> None:
    conn_string = os.getenv("POSTGRES_CONNECTION_STRING")
    if not conn_string:
        print("Error: POSTGRES_CONNECTION_STRING is not set", file=sys.stderr)
        sys.exit(1)
    row_count = seed_targets(conn_string)
    print(f"Seeded {row_count} sales_targets rows.")


if __name__ == "__main__":
    main()
