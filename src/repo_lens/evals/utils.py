import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tabulate import tabulate


def append_jsonl(results: list[dict[str, Any]], path: Path) -> None:
    """Append records to a JSONL file with timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.open("a") as f:
        for r in results:
            record = {**r, "timestamp": timestamp}
            f.write(json.dumps(record) + "\n")


def print_table(
    results: list[dict[str, Any]], columns: list[str] | None = None
) -> None:
    """Print a list of dicts as a formatted table."""
    if columns:
        results = [{k: r[k] for k in columns} for r in results]
    print(tabulate(results, headers="keys", tablefmt="simple", floatfmt=".2f"))
