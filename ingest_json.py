#!/usr/bin/env python3
"""
ingest_json.py — append benchmark JSON artifacts to progress.csv.

Two modes
---------
A) From a GitHub Actions run (downloads automatically):

    python ingest_json.py --gh-run 33724174688

B) From a local directory or single file:

    python ingest_json.py C:\\path\\to\\dir

Auto-detected fields (from JSON or GitHub API, no flags needed)
---------------------------------------------------------------
    runner      Extracted from artifact directory name  (e.g. mi355x-amds_03)
    image       Extracted from GH job name              (e.g. lmsysorg/sglang-rocm:...)
    branch      Current git branch
    dataset_ok  True if dataset.loader does not contain '_256k'

Optional overrides
------------------
    --runner LABEL      Override auto-detected runner.
    --image  TAG        Override auto-detected image.
    --branch NAME       Override branch (default: current git branch).
    --notes  TEXT       Free-text notes appended to each row.
    --dataset-ok        Force dataset_ok=True.
    --no-dataset-ok     Force dataset_ok=False.
    --csv    FILE       Target CSV (default: progress.csv).
    --dry-run           Print rows without writing.
    --repo   OWNER/REPO GitHub repo (default: giovanniguastiamd/InferenceX).
"""

import argparse
import csv
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
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

GH_BASE = "https://github.com/{repo}/actions/runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k)
        if d is None:
            return default
    return d


def _r(v, n=5):
    if v is None:
        return ""
    try:
        return round(float(v), n)
    except (TypeError, ValueError):
        return ""


def read_json(f: pathlib.Path) -> dict:
    """Read JSON, handling Windows 260-char path limit via \\?\\ prefix."""
    path_str = str(f.resolve())
    if sys.platform == "win32" and len(path_str) > 240 and not path_str.startswith("\\\\?\\"):
        path_str = "\\\\?\\" + path_str
    with open(path_str, encoding="utf-8") as fh:
        return json.load(fh)


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
# GitHub API helpers
# ---------------------------------------------------------------------------

def gh(*args, repo: str) -> dict | list | str:
    """Run `gh` CLI with GITHUB_TOKEN unset (uses keyring auth)."""
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    cmd = ["gh"] + list(args) + ["--repo", repo]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"gh error: {result.stderr.strip()}", file=sys.stderr)
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return result.stdout.strip()


def get_image_from_run(run_id: str, repo: str) -> str:
    """
    Extract docker image from the first agentic job name.
    Job name format: 'agentic / (IMAGE, MODEL, ...)  / ...'
    """
    data = gh("run", "view", run_id, "--json", "jobs", repo=repo)
    jobs = data.get("jobs", []) if isinstance(data, dict) else []
    for job in jobs:
        name = job.get("name", "")
        # Extract first token inside parentheses
        m = re.search(r'\(([^,)]+)', name)
        if m:
            candidate = m.group(1).strip()
            # Looks like an image tag: contains '/' and ':'
            if "/" in candidate and ":" in candidate:
                return candidate
    return ""


def extract_runner_from_dirname(dirname: str) -> str:
    """
    Artifact dir name ends with the runner label, e.g.:
      bmk_agentic_..._spec-mtp_conc6_mi355x-amds_03
    Extract the last '_'-separated segment that matches a runner pattern.
    """
    # Try to find a segment like 'mi355x-amds_03'
    m = re.search(r'(mi\d+x[-\w]+_\d+)$', dirname)
    if m:
        return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Download artifacts from GitHub
# ---------------------------------------------------------------------------

def download_artifacts(run_id: str, repo: str) -> pathlib.Path:
    """
    Download all bmk_agentic_* artifacts for a run into a temp directory.
    Returns the temp directory path.
    """
    # Use a short base path to stay under Windows 260-char limit
    tmpdir = pathlib.Path(tempfile.gettempdir()) / "ij"
    run_dir = tmpdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    cmd = [
        "gh", "run", "download", run_id,
        "--repo", repo,
        "--pattern", "bmk_agentic_*",
        "--dir", str(run_dir),
    ]
    print(f"Downloading bmk_agentic_* artifacts from run {run_id}…")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"gh error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    return run_dir


# ---------------------------------------------------------------------------
# Row extraction
# ---------------------------------------------------------------------------

def extract_row(artifact: dict, *, run_id: str, branch: str, runner: str,
                image: str, notes: str, dataset_ok_override, repo: str) -> dict:
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

    dok = dataset_ok_override if dataset_ok_override is not None else detect_dataset_ok(artifact)
    img = image or (artifact.get("image") or "")
    spec = artifact.get("spec_decoding") or ""

    base_url = GH_BASE.format(repo=repo)
    r = {k: "" for k in HEADER}
    r.update({
        "date":                  str(date.today()),
        "run_id":                run_id or "",
        "job_url":               f"{base_url}/{run_id}" if run_id else "",
        "branch":                branch or "",
        "runner":                runner or "",
        "image":                 img,
        "framework":             artifact.get("framework") or "sglang",
        "tp":                    artifact.get("tp", ""),
        "ep":                    artifact.get("ep", ""),
        "conc":                  artifact.get("conc", ""),
        "max_running_requests":  artifact.get("conc", ""),
        "kv_offload":            artifact.get("kv_offloading") or "none",
        "mtp":                   spec,
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
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--gh-run", metavar="RUN_ID",
                     help="GitHub Actions run ID — downloads artifacts automatically.")
    src.add_argument("path", nargs="?", type=pathlib.Path,
                     help="Local JSON file or directory of JSON files.")

    parser.add_argument("--repo", default="giovanniguastiamd/InferenceX",
                        help="GitHub repo (default: giovanniguastiamd/InferenceX).")
    parser.add_argument("--runner",  default="",
                        help="Runner label override (auto-detected from artifact dir name).")
    parser.add_argument("--image",   default="",
                        help="Docker image tag override (auto-detected from GH job name).")
    parser.add_argument("--branch",  default="",
                        help="Branch name override (default: current git branch).")
    parser.add_argument("--notes",   default="", help="Free-text notes.")
    dok_grp = parser.add_mutually_exclusive_group()
    dok_grp.add_argument("--dataset-ok",    dest="dataset_ok", action="store_true",
                         default=None, help="Force dataset_ok=True.")
    dok_grp.add_argument("--no-dataset-ok", dest="dataset_ok", action="store_false",
                         help="Force dataset_ok=False.")
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("progress.csv"),
                        help="Target CSV file (default: progress.csv).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print rows without writing.")
    args = parser.parse_args()

    branch = args.branch or current_branch()
    run_id = args.gh_run or ""

    # --- Resolve artifact directory ---
    if args.gh_run:
        artifact_dir = download_artifacts(args.gh_run, args.repo)
    else:
        artifact_dir = args.path

    # --- Collect JSON files (handles long Windows paths) ---
    if artifact_dir.is_dir():
        json_files = sorted(artifact_dir.rglob("*.json"))
    else:
        json_files = [artifact_dir]

    if not json_files:
        print(f"No JSON files found at: {artifact_dir}", file=sys.stderr)
        sys.exit(1)

    # --- Auto-detect image from GitHub API (only for --gh-run or if run_id given) ---
    image = args.image
    if not image and run_id:
        print(f"Auto-detecting image from run {run_id}…")
        image = get_image_from_run(run_id, args.repo)
        if image:
            print(f"  image: {image}")
        else:
            print("  (could not detect image — use --image to set manually)")

    # --- Process files ---
    rows = []
    for f in json_files:
        try:
            artifact = read_json(f)
        except Exception as e:
            print(f"Warning: skipping {f.name}: {e}", file=sys.stderr)
            continue

        # Auto-detect runner from artifact parent dir name
        runner = args.runner
        if not runner:
            runner = extract_runner_from_dirname(f.parent.name)

        row = extract_row(
            artifact,
            run_id=run_id,
            branch=branch,
            runner=runner,
            image=image,
            notes=args.notes,
            dataset_ok_override=args.dataset_ok,
            repo=args.repo,
        )
        rows.append(row)
        print(
            f"  conc={row['conc']:>3}  tp={row['tp']}  ep={row['ep']}"
            f"  runner={row['runner']}"
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
