#!/usr/bin/env python3
"""
ingest_json.py — append a benchmark run row to InferenceX_GLM-5.2_interactivity.csv

Usage examples
--------------
# From a local agg_bmk.json:
  python ingest_json.py --json /path/to/agg_bmk.json --run-id 33724174688 --dataset-ok

# From a GH Actions artifact (requires `gh` CLI authenticated):
  python ingest_json.py --gh-run 33724174688 --artifact agg_bmk --dataset-ok

# Write to a specific CSV (default: InferenceX_GLM-5.2_interactivity.csv):
  python ingest_json.py --json agg_bmk.json --run-id 33724174688 --csv my.csv

# Dry-run (print row, do not write):
  python ingest_json.py --json agg_bmk.json --run-id 33724174688 --dry-run

Notes
-----
- agg_bmk.json is produced by benchmarks/single_node/agentic/*.sh and then
  collected by the GitHub Actions `collect-evals` step.  The top-level keys
  used here are:
    request_metrics.{mean,median,p90,p99}_itl_token_latency_s   (ITL in seconds)
    request_metrics.p90_interactivity_tok_s_user                  (P90 interactivity)
    output_throughput_per_chip                                     (tok/s / GPU)
    concurrency                                                    (int)
    tp                                                             (TP group size)
    ep                                                             (EP value)
    date                                                           (ISO date string)
    infmax_model_prefix / dataset.loader                           (for dataset_ok check)

- dataset_ok heuristic (if --dataset-ok / --no-dataset-ok not given explicitly):
    True  if infmax_model_prefix is set (non-empty) in the JSON
    False otherwise (falls back to 256k-capped dataset)

- The script appends a SINGLE row per invocation.  For a multi-CONC sweep,
  run once per CONC point (the JSON contains one CONC at a time).

Column mapping
--------------
Columns 1-43 match the upstream CSV exactly; columns 44-45 are the two new ones:
  44  ITL P90 (s)
  45  dataset_ok
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CSV_PATH = Path(__file__).parent / "InferenceX_GLM-5.2_interactivity.csv"

# Upstream CSV has 43 columns + 2 new = 45 total.
# We reconstruct the row with empty strings for columns we don't fill.
N_COLS = 45  # header column count (0-indexed: 0..44)

# Column indices (0-based) for fields we fill:
COL_MODEL       = 0   # "GLM-5.2"
COL_HARDWARE    = 3   # "mi355x"
COL_HW_KEY      = 4   # e.g. "mi355x_sglang_mia1"
COL_FRAMEWORK   = 5   # "sglang"
COL_PRECISION   = 6   # "fp4"
COL_TP          = 7
COL_CONCURRENCY = 8
COL_DATE        = 9
COL_P90_INTVTY  = 11  # P90 Interactivity (tok/s/user)
COL_OUT_TPUT    = 13  # Output Throughput/Chip (tok/s)
COL_MEAN_ITL    = 27  # Mean ITL (s)
COL_MED_ITL     = 28  # Median ITL (s)
COL_P99_ITL     = 29  # P99 ITL (s)
COL_STD_ITL     = 30  # Std ITL (s)
COL_DISAGG      = 34  # "false"
COL_SPEC_DEC    = 37  # "mtp"
COL_EP          = 38  # EP
COL_DP_ATTN     = 39  # "false"
COL_MULTINODE   = 40  # "false"
COL_RUN_URL     = 41  # Run URL
COL_ITL_P90     = 42  # ITL P90 (s)   ← new
COL_DATASET_OK  = 43  # dataset_ok    ← new


GH_BASE = "https://github.com/giovanniguastiamd/InferenceX/actions/runs"


def _gh_url(run_id: str) -> str:
    return f"{GH_BASE}/{run_id}"


def download_artifact(run_id: str, artifact_name: str) -> Path:
    """Download a GH Actions artifact and return path to extracted directory."""
    with tempfile.TemporaryDirectory(prefix="ingest_", delete=False) as tmpdir:
        cmd = [
            "gh", "run", "download", run_id,
            "--name", artifact_name,
            "--dir", tmpdir,
            "--repo", "giovanniguastiamd/InferenceX",
        ]
        print(f"[ingest] Downloading artifact '{artifact_name}' from run {run_id}…")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[ingest] gh error:\n{result.stderr}", file=sys.stderr)
            sys.exit(1)
        # Find the JSON file inside the extracted dir
        matches = list(Path(tmpdir).rglob("*.json"))
        if not matches:
            print(f"[ingest] No JSON found in artifact '{artifact_name}'", file=sys.stderr)
            sys.exit(1)
        if len(matches) > 1:
            print(f"[ingest] Multiple JSON files found; using first: {matches[0]}")
        return matches[0]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def detect_dataset_ok(data: dict) -> bool:
    """
    Return True if the JSON indicates the full unfiltered dataset was used.
    Heuristic: infmax_model_prefix is a non-empty string in the JSON.
    """
    prefix = data.get("infmax_model_prefix", "") or ""
    return bool(prefix.strip())


def extract_row(data: dict, run_id: str, dataset_ok: bool) -> list:
    """
    Build a 45-element list (one per CSV column) from the JSON data.
    Empty strings for columns we don't fill.
    """
    row = [""] * N_COLS

    req = data.get("request_metrics", {})
    conc = data.get("concurrency", "")
    tp   = data.get("tp", "")
    ep   = data.get("ep", "")
    date = (data.get("date") or data.get("run_date") or "")[:10]   # ISO date, trim time

    p90_intvty = req.get("p90_interactivity_tok_s_user", "")
    out_tput   = data.get("output_throughput_per_chip", "")

    mean_itl = req.get("mean_itl_token_latency_s", "")
    med_itl  = req.get("median_itl_token_latency_s", "")
    p90_itl  = req.get("p90_itl_token_latency_s", "")
    p99_itl  = req.get("p99_itl_token_latency_s", "")
    std_itl  = req.get("std_itl_token_latency_s", "")

    hw_key = "mi355x_sglang_mia1"
    if tp and int(tp) == 4:
        hw_key = "mi355x_sglang_mia1_tp4"

    row[COL_MODEL]       = "GLM-5.2"
    row[COL_HARDWARE]    = "mi355x"
    row[COL_HW_KEY]      = hw_key
    row[COL_FRAMEWORK]   = "sglang"
    row[COL_PRECISION]   = "fp4"
    row[COL_TP]          = str(tp)
    row[COL_CONCURRENCY] = str(conc)
    row[COL_DATE]        = date
    row[COL_P90_INTVTY]  = "" if p90_intvty == "" else str(p90_intvty)
    row[COL_OUT_TPUT]    = "" if out_tput == "" else str(out_tput)
    row[COL_MEAN_ITL]    = "" if mean_itl == "" else str(mean_itl)
    row[COL_MED_ITL]     = "" if med_itl  == "" else str(med_itl)
    row[COL_P99_ITL]     = "" if p99_itl  == "" else str(p99_itl)
    row[COL_STD_ITL]     = "" if std_itl  == "" else str(std_itl)
    row[COL_DISAGG]      = "false"
    row[COL_SPEC_DEC]    = "mtp"
    row[COL_EP]          = str(ep)
    row[COL_DP_ATTN]     = "false"
    row[COL_MULTINODE]   = "false"
    row[COL_RUN_URL]     = _gh_url(run_id)
    row[COL_ITL_P90]     = "" if p90_itl == "" else str(p90_itl)
    row[COL_DATASET_OK]  = "True" if dataset_ok else "False"

    return row


def read_header(csv_path: Path) -> list[str]:
    """Return the header row (skipping # comment lines)."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        for line in f:
            if not line.startswith("#"):
                return next(csv.reader([line.rstrip("\n")]))
    raise ValueError(f"No header found in {csv_path}")


def append_row(csv_path: Path, row: list, dry_run: bool = False) -> None:
    header = read_header(csv_path)
    if len(row) != len(header):
        print(
            f"[ingest] WARNING: row has {len(row)} cols, header has {len(header)} cols",
            file=sys.stderr,
        )
    line = ",".join(str(v) for v in row)
    if dry_run:
        print("[dry-run] Would append:")
        print(line)
        return
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        f.write(line + "\n")
    print(f"[ingest] Appended row to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", metavar="FILE",
                     help="Path to agg_bmk.json artifact")
    src.add_argument("--gh-run", metavar="RUN_ID",
                     help="GH Actions run ID; artifact downloaded via `gh` CLI")

    parser.add_argument("--artifact", metavar="NAME", default="agg_bmk",
                        help="Artifact name to download (default: agg_bmk)")
    parser.add_argument("--run-id", metavar="RUN_ID",
                        help="GH Actions run ID (required when using --json)")
    parser.add_argument("--dataset-ok", dest="dataset_ok", action="store_true",
                        default=None,
                        help="Mark row as correct dataset (overrides JSON heuristic)")
    parser.add_argument("--no-dataset-ok", dest="dataset_ok", action="store_false",
                        help="Mark row as 256k-capped dataset (overrides JSON heuristic)")
    parser.add_argument("--csv", metavar="FILE", default=str(CSV_PATH),
                        help=f"Target CSV file (default: {CSV_PATH.name})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the row but do not write to CSV")
    args = parser.parse_args()

    # Resolve JSON path
    if args.gh_run:
        run_id = args.gh_run
        json_path = download_artifact(run_id, args.artifact)
    else:
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"[ingest] File not found: {json_path}", file=sys.stderr)
            sys.exit(1)
        run_id = args.run_id
        if not run_id:
            # Try to infer from parent directory name if it looks like a run ID
            parent = json_path.parent.name
            if re.fullmatch(r"\d{8,}", parent):
                run_id = parent
                print(f"[ingest] Inferred run_id={run_id} from directory name")
            else:
                print("[ingest] --run-id is required when using --json", file=sys.stderr)
                sys.exit(1)

    data = load_json(json_path)

    # dataset_ok: explicit flag > JSON heuristic
    if args.dataset_ok is None:
        dataset_ok = detect_dataset_ok(data)
        src_str = "JSON heuristic"
    else:
        dataset_ok = args.dataset_ok
        src_str = "CLI flag"
    print(f"[ingest] dataset_ok={dataset_ok}  (source: {src_str})")

    row = extract_row(data, run_id, dataset_ok)

    csv_path = Path(args.csv)
    append_row(csv_path, row, dry_run=args.dry_run)

    # Summary
    conc = row[COL_CONCURRENCY]
    tp   = row[COL_TP]
    intvty = row[COL_P90_INTVTY]
    tput   = row[COL_OUT_TPUT]
    itl90  = row[COL_ITL_P90]
    print(
        f"[ingest] run={run_id}  CONC={conc}  TP={tp}  "
        f"P90_intvty={intvty}  out_tput/chip={tput}  ITL_p90={itl90}  "
        f"dataset_ok={dataset_ok}"
    )


if __name__ == "__main__":
    main()
