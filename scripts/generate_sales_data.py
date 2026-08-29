"""Generate the deterministic synthetic six-month retail sales CSV.

Usage:
    python scripts/generate_sales_data.py [output_path]

``output_path`` defaults to ``data/sales.csv``. The generator is seeded
(``analyst_crew.tools.sales.DEFAULT_SEED``) so re-running this script
reproduces byte-identical output — this is the "one documented command" that
produces the required CSV Data Source consumed by the CLI's ``--sales-csv``
flag (see README.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

from analyst_crew.tools.sales import DEFAULT_SEED, generate_sales_csv


def main() -> None:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sales.csv")
    path = generate_sales_csv(output_path, seed=DEFAULT_SEED)
    print(f"Generated deterministic sales CSV at {path} (seed={DEFAULT_SEED}).")


if __name__ == "__main__":
    main()
