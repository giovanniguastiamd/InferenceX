#!/usr/bin/env python3
"""
ingest_json.py — append benchmark JSON artifacts to progress.csv.

Reads one or more bmk_agentic JSON files produced by the InferenceX pipeline
and appends the extracted metrics as new rows to progress.csv.

Usage
-----
    # Single file:
    python ingest_json.py path/to/conc8.json --run-id 33724174688 --runner mi355x-amds_03

    # Directory of JSON files (one per CONC point):
    python ingest_json.py C:\\tmp\\run_data --run-id 33724174688 --runner mi355x-amds_03 --image lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728

    # Dry-run (print rows, don't write):
    python ingest_json.py path/to/dir --run-id 33724174688 --dry-run

Options
-------
    --run-id        GitHub Actions run ID (e.g. 33724174688).
    --branch        Branch name. Defaults to current git branch.
    --runner        Runner label (e.g. mi355x-amds_03).
    --image         Docker image tag.
    --notes         Free-text notes appended to each row.
    --dataset-ok    Override dataset_ok: true / false / auto (default: auto).
                    auto = True if loader does NOT contain '_256k'.
    --csv           Target CSV file (default: progress.csv).
    --dry-run       Print rows without writing.

JSON field mapping
------------------
    conc                                    → conc, max_running_requests
    tp, ep                                  → tp, ep
    kv_offloading                           → kv_offload
    spec_decoding                           → mtp
    framework                               → framework
    num_requests_successful                 → requests_ok
    request_metrics.throughput.duration_seconds       → duration_s
    request_metrics.throughput.per_gpu.total_tput_tps → throughput_per_gpu_tps
    request_metrics.throughput.output.tokens_per_second → output_tps
    request_metrics.latency.ttft.{mean,p50,p90}        → ttft_{mean,p50,p90}_s
    request_metrics.latency.itl.{mean,p50,p90} × 1000  → itl_{mean,p50,p90}_ms
    request_metrics.latency.intvty.{mean,p50,p90}       → intvty_{mean,p50,p90}
    server_metrics.cache.gpu_cache_hit_rate              → gpu_cache_hit_rate
    server_metrics.cache.cpu_cache_hit_rate              → cpu_cache_hit_rate
    server_metrics.kv_cache.gpu_usage_pct                → kv_gpu_usage_pct
    server_metrics.kv_cache.cpu_usage_pct                → kv_cpu_usage_pct
    dataset.loader (no '_256k') → dataset_ok=True
"""

import argparse
import csv
import json
import pathlib
import subprocess
import sys
from datetime import date

# ---------------------------------------------------------------------------
# CSV schema — must match gen_progress_csv.py HEADER exactly
# ---------------------------------------------------------------------------
HEADER = [
    "date", "run_id", "job_url", "branch", "runner", "image", "framework",
    "tp", "ep", "conc", "max_running_requests", "chunked_prefill_size",
    "kv_offload", "hicache_ratio", "mtp", "spec_steps", "spec_draft_tokens",
    "duration_s", "requests_ok", "throughput_per_gpu_tps", "output_tps",
    "ttft_mean_s", "ttft_p50_s", "ttft_p90_s",
    "itl_mean_ms", "itl_p50_ms", "itl_p90_ms",
    "intvty_mean", "intvty_p50", "intvty_p90",
    "gpu_cache_hit_rate", "cpu_cache_hit_rate", "kv_gpu_usage_pct", "kv_cpu_usage_pct",
    "dataset_ok", "notes",
]
GH_BASE = "https://github.com/giovanniguastiamd/InferenceX/actions/runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(d, *keys, default=None):
    """Safe nested dict access."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def _r(v, n=5):
    """Round to n decimal places, return empty string on None."""
    if v is None:
        return ""
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return ""


def current_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        return ""


def detect_dataset_ok(artifact: dict) -> bool:
    loader = (_g(artifact, "dataset", "loader") or "")
    return "_256k" not in loader


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------

def extract_row(artifact: dict, *, run_id: str, branch: str, runner: str,
                image: str, notes: str, dataset_ok_override) -> dict:
    rm    = artifact.get("request_metrics", {})
    sm    = artifact.get("server_metrics", {})
    lat   = rm.get("latency", {})
    tput  = rm.get("throughput", {})
    ttft  = lat.get("ttft", {})
    itl   = lat.get("itl", {})
    intvty = lat.get("intvty", {})
    per_gpu = _g(tput, "per_gpu") or {}
    cache   = sm.get("cache", {})
    kv_c    = sm.get("kv_cache", {})

    # dataset_ok
    if dataset_ok_override is None:
        dok = detect_dataset_ok(artifact)
    else:
        dok = dataset_ok_override

    img = image or (artifact.get("image") or "")
    spec = artifact.get("spec_decoding") or ""
    mtp_val = spec if spec else ""

    r = {k: "" for k in HEADER}
    r.update({
        "date":                  str(date.today()),
        "run_id":                run_id or "",
        "job_url":               f"{GH_BASE}/{run_id}" if run_id else "",
        "branch":                branch or "",
        "runner":                runner or "",
        "image":                 img,
        "framework":             artifact.get("framework") or "sglang",
        "tp":                    artifact.get("tp", ""),
        "ep":                    artifact.get("ep", ""),
        "conc":                  artifact.get("conc", ""),
        "max_running_requests":  artifact.get("conc", ""),
        "kv_offload":            artifact.get("kv_offloading") or "none",
        "mtp":                   mtp_val,
        "duration_s":            _r(_g(tput, "duration_seconds"), 2),
        "requests_ok":           artifact.get("num_requests_successful", ""),
        "throughput_per_gpu_tps": _r(_g(per_gpu, "total_tput_tps"), 2),
        "output_tps":            _r(_g(tput, "output", "tokens_per_second"), 2),
        "ttft_mean_s":           _r(ttft.get("mean"), 5),
        "ttft_p50_s":            _r(ttft.get("p50"), 5),
        "ttft_p90_s":            _r(ttft.get("p90"), 5),
        "itl_mean_ms":           _r((itl.get("mean") or 0) * 1000, 3),
        "itl_p50_ms":            _r((itl.get("p50") or 0) * 1000, 3),
        "itl_p90_ms":            _r((itl.get("p90") or 0) * 1000, 3),
        "intvty_mean":           _r(intvty.get("mean"), 3),
        "intvty_p50":            _r(intvty.get("p50"), 3),
        "intvty_p90":            _r(intvty.get("p90"), 3),
        "gpu_cache_hit_rate":    _r(cache.get("gpu_cache_hit_rate"), 4),
        "cpu_cache_hit_rate":    _r(cache.get("cpu_cache_hit_rate"), 4),
        "kv_gpu_usage_pct":      _r(kv_c.get("gpu_usage_pct"), 4),
        "kv_cpu_usage_pct":      _r(kv_c.get("cpu_usage_pct"), 4),
        "dataset_ok":            str(dok),
        "notes":                 notes or "",
    })
    return r


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", type=pathlib.Path,
                        help="JSON file or directory of JSON files.")
    parser.add_argument("--run-id",  default="",  help="GitHub Actions run ID.")
    parser.add_argument("--branch",  default="",  help="Branch name (default: current branch).")
    parser.add_argument("--runner",  default="",  help="Runner label (e.g. mi355x-amds_03).")
    parser.add_argument("--image",   default="",  help="Docker image tag.")
    parser.add_argument("--notes",   default="",  help="Free-text notes.")
    parser.add_argument("--dataset-ok", default="auto", choices=["auto", "true", "false"],
                        help="Override dataset_ok. auto=detect from loader name (default).")
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("progress.csv"),
                        help="Target CSV file (default: progress.csv).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rows without writing.")
    args = parser.parse_args()

    branch = args.branch or current_branch()
    dok_override = None if args.dataset_ok == "auto" else (args.dataset_ok == "true")

    # Collect JSON files
    p = args.path
    json_files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
    if not json_files:
        print(f"No JSON files found at: {p}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for f in json_files:
        try:
            artifact = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)
            continue

        row = extract_row(
            artifact,
            run_id=args.run_id,
            branch=branch,
            runner=args.runner,
            image=args.image,
            notes=args.notes,
            dataset_ok_override=dok_override,
        )
        rows.append(row)
        print(
            f"  conc={row['conc']:>3}  tp={row['tp']}  ep={row['ep']}"
            f"  ITL_p90={row['itl_p90_ms']:>6}ms"
            f"  intvty_p90={row['intvty_p90']:>7}"
            f"  output_tps={row['output_tps']:>8}"
            f"  dataset_ok={row['dataset_ok']}"
        )

    if not rows:
        print("No rows extracted.")
        return

    if args.dry_run:
        print(f"\n[dry-run] Would append {len(rows)} row(s) to {args.csv}")
        return

    csv_exists = args.csv.exists()
    with args.csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not csv_exists:
            writer.writerow(HEADER)
        for r in rows:
            writer.writerow([r[k] for k in HEADER])

    print(f"\nAppended {len(rows)} row(s) to {args.csv}")


if __name__ == "__main__":
    main()
