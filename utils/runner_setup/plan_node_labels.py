#!/usr/bin/env python3
"""Plan static GitHub runner labels for weighted Slurm admission slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def plan_node_labels(runners: list[str]) -> dict[str, list[str]]:
    """Assign ``nodes:N`` to the first ``floor(capacity / N)`` runners.

    The source order is intentional: ``configs/runners.yaml`` already records
    the fleet's preferred runner order. Prefix assignment keeps the labels
    nested, so larger allocations are available on progressively earlier
    runner slots while ``nodes:1`` remains available everywhere.
    """
    if not runners:
        raise ValueError("runner pool cannot be empty")
    if len(runners) != len(set(runners)):
        raise ValueError("runner pool cannot contain duplicate names")

    capacity = len(runners)
    return {
        runner: [
            f"nodes:{node_count}"
            for node_count in range(1, capacity + 1)
            if index < capacity // node_count
        ]
        for index, runner in enumerate(runners)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pool", help="Runner label to treat as one Slurm pool")
    parser.add_argument(
        "--runner-config",
        type=Path,
        default=Path("configs/runners.yaml"),
    )
    args = parser.parse_args()

    runner_config = yaml.safe_load(args.runner_config.read_text())
    try:
        runners = runner_config["labels"][args.pool]
    except KeyError as error:
        raise SystemExit(f"Unknown runner pool: {args.pool}") from error

    print(json.dumps(plan_node_labels(runners), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
