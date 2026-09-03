#!/usr/bin/env python3
"""
pareto_plot.py — GLM-5.2 MXFP4 / MI355X — SGLang tuning campaign (Aug–Sep 2026)

Usage:
    python pareto_plot.py              # interactive window
    python pareto_plot.py --png        # save pareto_plot.png
    python pareto_plot.py --svg        # save pareto_plot.svg  (URLs clickable in browsers)
    python pareto_plot.py -o out.pdf   # custom output file
    python pareto_plot.py --csv path/to/other.csv

Data source: progress.csv  (one row per benchmark point)

progress.csv columns used:
    run_id, job_url, image, tp, ep, conc,
    kv_offload, hicache_ratio, notes,
    intvty_p90, throughput_per_gpu_tps, itl_p90_ms,
    dataset_ok   (optional; default True; derive from notes if absent)

Marker convention:
    o  = correct dataset (dataset_ok=True)
    x  = invalidated dataset (dataset_ok=False)  ⚠️
    s  = TP=4 arm (tp column == 4)

Color convention:  one color per run_id (encounter order in CSV).

Layout:
    Top panel    — P90 interactivity (tok/s/user) vs CONC
    Bottom panel — Output tok/s/GPU vs CONC
    Right panel  — run legend with clickable GH URLs
"""

import argparse
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

import matplotlib
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CSV_DEFAULT = Path(__file__).parent / "progress.csv"

# Color palette — assigned in encounter order (reproducible)
PALETTE = [
    "tab:blue", "tab:orange", "tab:green", "tab:purple",
    "tab:red", "tab:brown", "tab:pink", "tab:cyan",
]

ATOM_RUN_IDS: set[str] = set()  # populated at runtime from framework=="atom" rows


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_csv(csv_path: Path) -> pd.DataFrame:
    """Load progress.csv, skip comment/blank lines, return typed DataFrame."""
    rows = []
    header = None
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if header is None:
                header = stripped
                continue
            rows.append(stripped)

    if not rows:
        return pd.DataFrame()

    content = header + "\n" + "\n".join(rows)
    df = pd.read_csv(StringIO(content), dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # Numeric
    for col in ("tp", "ep", "conc", "intvty_mean", "intvty_p50", "intvty_p90",
                "throughput_per_gpu_tps", "output_tps",
                "itl_mean_ms", "itl_p50_ms", "itl_p90_ms",
                "ttft_mean_s", "ttft_p50_s", "ttft_p90_s",
                "hicache_ratio", "kv_gpu_usage_pct", "kv_cpu_usage_pct",
                "gpu_cache_hit_rate", "duration_s", "requests_ok"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # dataset_ok: explicit column > derive from notes
    if "dataset_ok" in df.columns:
        df["dataset_ok"] = df["dataset_ok"].map(
            {"True": True, "False": False, "true": True, "false": False,
             "1": True, "0": False}
        ).fillna(True)
    else:
        # Heuristic: if notes contains "256k" or "INVALIDATED" → False
        bad_kw = ("256k", "invalidated", "INVALIDATED")
        notes_col = df["notes"].fillna("") if "notes" in df.columns else pd.Series([""] * len(df))
        df["dataset_ok"] = ~notes_col.str.contains("|".join(bad_kw), case=False, na=False)

    # Compute output per GPU for bottom panel
    if "output_tps" in df.columns and "tp" in df.columns:
        df["output_per_gpu_tps"] = df["output_tps"] / df["tp"].replace(0, float("nan"))

    # Normalize run_id
    if "run_id" not in df.columns and "job_url" in df.columns:
        import re
        df["run_id"] = df["job_url"].str.extract(r"/runs/(\d+)")[0].fillna(df["job_url"])
    else:
        df["run_id"] = df["run_id"].fillna("unknown")

    return df


# ---------------------------------------------------------------------------
# Auto-label from config columns
# ---------------------------------------------------------------------------

def _auto_label(subset: pd.DataFrame) -> str:
    r = subset.iloc[0]
    lines = []

    # Notes first (if short enough)
    note = str(r.get("notes", "") or "").strip()
    if note and note.lower() not in ("nan", ""):
        lines.append(note)

    # TP / EP
    tp_vals = sorted(subset["tp"].dropna().astype(int).unique()) if "tp" in subset.columns else []
    ep_vals = sorted(subset["ep"].dropna().astype(int).unique()) if "ep" in subset.columns else []
    tp_str = "+".join(f"TP={t}" for t in tp_vals) if tp_vals else ""
    ep_str = "+".join(f"EP={e}" for e in ep_vals) if ep_vals else ""
    if tp_str or ep_str:
        lines.append("/".join(filter(None, [tp_str, ep_str])))

    # KV / HiCache
    kv = str(r.get("kv_offload", "") or "").strip()
    hr = r.get("hicache_ratio", None)
    if kv and kv not in ("nan", "none", ""):
        hr_str = f"={hr}" if pd.notna(hr) else ""
        lines.append(f"HiCache{hr_str}")
    else:
        lines.append("no-HiCache")

    # Image / framework tag
    fw  = str(r.get("framework", "") or "").strip()
    img = str(r.get("image", "") or "").strip()
    if fw and fw.lower() not in ("nan", "sglang", ""):
        lines.append(fw)
    elif img and img.lower() != "nan":
        lines.append(img)

    return "\n".join(lines)


def _config_key(subset: pd.DataFrame) -> str:
    r = subset.iloc[0]
    branch = str(r.get("branch", "") or "").strip()
    return branch if branch and branch.lower() != "nan" else ""


def _gh_url(subset: pd.DataFrame) -> str:
    """Best URL for this run: job_url from first row, or construct from run_id."""
    if "job_url" in subset.columns:
        url = str(subset.iloc[0]["job_url"]).strip()
        if url and url.lower() not in ("nan", ""):
            # Strip /job/... suffix → keep the run-level URL
            import re
            m = re.match(r"(https://[^/]+/.+/runs/\d+)", url)
            return m.group(1) if m else url
    run_id = subset.iloc[0]["run_id"]
    return f"https://github.com/giovanniguastiamd/InferenceX/actions/runs/{run_id}"


# ---------------------------------------------------------------------------
# Run grouping
# ---------------------------------------------------------------------------

def build_run_groups(df: pd.DataFrame) -> list[dict]:
    # Detect ATOM run IDs from framework column
    if "framework" in df.columns:
        atom_ids = df.loc[df["framework"].str.lower() == "atom", "run_id"].unique()
        ATOM_RUN_IDS.update(atom_ids)
    seen = []
    for rid in df["run_id"]:
        if rid not in seen:
            seen.append(rid)

    color_iter = iter(PALETTE)
    color_map: dict[str, str] = {}
    groups = []

    for rid in seen:
        subset = df[df["run_id"] == rid]

        is_atom = rid in ATOM_RUN_IDS
        if is_atom:
            color, ls = "black", "--"
        else:
            if rid not in color_map:
                try:
                    color_map[rid] = next(color_iter)
                except StopIteration:
                    color_map[rid] = "gray"
            color, ls = color_map[rid], "-"

        dataset_ok = bool(subset["dataset_ok"].all())

        # TP=4 sub-arm only when the run also has non-TP=4 rows (mixed-TP run).
        # If all rows are TP=4, keep them in main_rows so they get proper line + marker.
        tp_values = subset["tp"].dropna().unique() if "tp" in subset.columns else []
        has_higher_tp = any(t != 4 for t in tp_values)
        if has_higher_tp:
            tp4_mask = subset["tp"] == 4
        else:
            tp4_mask = pd.Series([False] * len(subset), index=subset.index)
        main_rows = subset[~tp4_mask]
        tp4_rows  = subset[tp4_mask]

        def _to_points(rows):
            pts = []
            for _, r in rows.iterrows():
                c    = r.get("conc", None)
                p    = r.get("intvty_p90", None)
                # Bottom panel: output tok/s per GPU (precomputed in load_csv)
                t    = r.get("output_per_gpu_tps", None)
                itl  = r.get("itl_p90_ms", None)
                note = f"c{int(c)}" if pd.notna(c) else ""
                pts.append((
                    float(c)   if pd.notna(c)   else None,
                    float(itl) if pd.notna(itl) else None,
                    float(p)   if pd.notna(p)   else None,
                    float(t)   if pd.notna(t)   else None,
                    note,
                ))
            return sorted(pts, key=lambda x: x[0] or 0)

        groups.append({
            "run_id":     rid,
            "label":      _auto_label(subset),
            "config_key": _config_key(subset),
            "url":        _gh_url(subset),
            "color":      color,
            "linestyle":  ls,
            "dataset_ok": dataset_ok,
            "points":     _to_points(main_rows),
            "points_tp4": _to_points(tp4_rows),
        })

    return groups


# ---------------------------------------------------------------------------
# Pareto upper envelope
# ---------------------------------------------------------------------------

def pareto_upper_envelope(groups: list[dict]):
    best = defaultdict(lambda: -1)
    for g in groups:
        if not g["dataset_ok"]:
            continue
        for pts_key in ("points", "points_tp4"):
            for c, _i, p, _t, _n in g[pts_key]:
                if c is not None and p is not None:
                    best[c] = max(best[c], p)
    xs = sorted(best.keys())
    ys = [best[x] for x in xs]
    return xs, ys


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def build_figure(groups: list[dict]):
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[3.2, 1],
        height_ratios=[1, 1],
        hspace=0.38,
        wspace=0.06,
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0])
    ax_leg = fig.add_subplot(gs[:, 1])
    ax_leg.axis("off")

    legend_entries = []

    for run in groups:
        color  = run["color"]
        ls     = run["linestyle"]
        ok     = run["dataset_ok"]
        marker = "o" if ok else "x"
        alpha  = 1.0 if ok else 0.55
        ms     = 8   if ok else 9
        zorder = 4   if ok else 2

        # --- Interactivity panel ---
        pts = run["points"]
        iv = [(c, p) for c, _i, p, _t, _n in pts if c is not None and p is not None]
        iv.sort()
        if iv:
            cx, cy = zip(*iv)
            line, = ax_top.plot(cx, cy, color=color, ls=ls, marker=marker,
                                markersize=ms, lw=2.0, alpha=alpha, zorder=zorder)
            line.set_url(run["url"])
            for idx, (c, p) in enumerate(iv):
                dy = 5 if idx % 2 == 0 else -12
                ax_top.annotate(f"c{int(c)}", (c, p),
                                xytext=(5, dy), textcoords="offset points",
                                fontsize=7, color=color, alpha=alpha, zorder=zorder + 1)

        for sub in run.get("points_tp4", []):
            c, _i, p, _t, note = sub
            if c is not None and p is not None:
                ax_top.plot(c, p, color=color, marker="s", markersize=10,
                            ls="none", alpha=alpha, zorder=zorder)
                ax_top.annotate(note, (c, p), xytext=(5, 8),
                                textcoords="offset points",
                                fontsize=7, color=color, alpha=alpha)

        # --- Throughput panel ---
        tv = [(c, t) for c, _i, _p, t, _n in pts if c is not None and t is not None]
        tv.sort()
        if tv:
            cx, cy = zip(*tv)
            line2, = ax_bot.plot(cx, cy, color=color, ls=ls, marker=marker,
                                 markersize=ms, lw=2.0, alpha=alpha, zorder=zorder)
            line2.set_url(run["url"])
            for c, t in tv:
                ax_bot.annotate(f"c{int(c)}", (c, t),
                                xytext=(5, 3), textcoords="offset points",
                                fontsize=7, color=color, alpha=alpha, zorder=zorder + 1)

        for sub in run.get("points_tp4", []):
            c, _i, _p, t, note = sub
            if c is not None and t is not None:
                ax_bot.plot(c, t, color=color, marker="s", markersize=10,
                            ls="none", alpha=alpha, zorder=zorder)
                ax_bot.annotate(note, (c, t), xytext=(5, 3),
                                textcoords="offset points",
                                fontsize=7, color=color, alpha=alpha)

        legend_entries.append((run,))

    # --- Pareto upper envelope ---
    px, py = pareto_upper_envelope(groups)
    if px:
        ax_top.step(px, py, where="post", color="gold", lw=2.8, ls=":", alpha=0.8, zorder=1)
        px_step = np.array(px + [px[-1]])
        py_step = np.array(py + [py[-1]])
        ax_top.fill_between(px_step, 0, py_step, step="post",
                            color="gold", alpha=0.06, zorder=0)

    # ATOM reference line (if any ATOM group present)
    atom_groups = [g for g in groups if g["run_id"] in ATOM_RUN_IDS and g["points"]]
    if atom_groups:
        best_atom = max(
            p for g in atom_groups for _, _, p, _, _ in g["points"] if p is not None
        )
        ax_top.axhline(best_atom, color="black", ls=":", lw=1.2, alpha=0.35, zorder=1)
        ax_top.text(25.5, best_atom + 1, f"ATOM\n={best_atom:.0f}", fontsize=7.5,
                    color="gray", ha="right", va="bottom", alpha=0.7)

    # --- Axes — interactivity ---
    ax_top.set_ylabel("P90 interactivity (tok/s/user)", fontsize=11)
    ax_top.set_title(
        "GLM-5.2 MXFP4 / 8×MI355X — SGLang tuning campaign\n"
        "P90 interactivity vs concurrency",
        fontsize=12, pad=8,
    )
    ax_top.set_xlim(0, 26)
    ax_top.set_ylim(0, 175)
    ax_top.set_xticks([1, 2, 4, 6, 8, 10, 12, 16, 20, 24])
    ax_top.grid(True, alpha=0.25)
    ax_top.set_xlabel("Concurrency (CONC)", fontsize=10)

    ax_top.axvspan(0,   6.5, alpha=0.04, color="green",  label="_nolegend_")
    ax_top.axvspan(6.5, 13,  alpha=0.04, color="orange", label="_nolegend_")
    ax_top.axvspan(13,  26,  alpha=0.04, color="red",    label="_nolegend_")
    ax_top.text(1,   168, "interactivity", fontsize=8, color="green",  alpha=0.7)
    ax_top.text(7.5, 168, "crossover",     fontsize=8, color="orange", alpha=0.7)
    ax_top.text(15,  168, "throughput",    fontsize=8, color="red",    alpha=0.7)

    _h_ok  = mlines.Line2D([], [], color="gray", marker="o", ls="none",
                            markersize=8, label="○  correct dataset")
    _h_bad = mlines.Line2D([], [], color="gray", marker="x", ls="none",
                            markersize=8, alpha=0.55, label="✗  invalidated dataset  ⚠")
    _h_sq  = mlines.Line2D([], [], color="gray", marker="s", ls="none",
                            markersize=9, label="□  TP=4 arm")
    _h_env = mlines.Line2D([], [], color="gold", ls=":", lw=2.5,
                            label="⋯  Pareto upper envelope")
    ax_top.legend(handles=[_h_ok, _h_bad, _h_sq, _h_env],
                  loc="upper right", fontsize=8, framealpha=0.9, handlelength=2)

    # --- Axes — throughput ---
    ax_bot.set_ylabel("Output tok/s / GPU", fontsize=11)
    ax_bot.set_title("Throughput vs concurrency", fontsize=11, pad=6)
    ax_bot.set_xlim(0, 26)
    ax_bot.set_ylim(0, 130)
    ax_bot.set_xticks([1, 2, 4, 6, 8, 10, 12, 16, 20, 24])
    ax_bot.grid(True, alpha=0.25)
    ax_bot.set_xlabel("Concurrency (CONC)", fontsize=10)

    # --- Right panel — run legend ---
    y = 0.98
    dy_h = 0.052
    dy_l = 0.030
    dy_g = 0.018

    ax_leg.text(0.0, y, "Runs", transform=ax_leg.transAxes,
                fontsize=11, fontweight="bold", va="top")
    y -= dy_h

    for (run,) in legend_entries:
        color = run["color"]
        ok    = run["dataset_ok"]

        ax_leg.add_patch(mpatches.FancyBboxPatch(
            (0.0, y - 0.018), 0.06, 0.024,
            boxstyle="round,pad=0.002",
            facecolor=color, edgecolor="none",
            transform=ax_leg.transAxes, clip_on=False, alpha=0.85,
        ))

        label_lines = run["label"].count("\n") + 1
        ax_leg.text(0.09, y, run["label"], transform=ax_leg.transAxes,
                    fontsize=8.5, fontweight="bold", color=color, va="top")
        y -= dy_l * label_lines

        ax_leg.text(0.09, y, f"run {run['run_id']}", transform=ax_leg.transAxes,
                    fontsize=7.5, color="dimgray", va="top", family="monospace")
        y -= dy_l

        if run["config_key"]:
            ax_leg.text(0.09, y, run["config_key"], transform=ax_leg.transAxes,
                        fontsize=6.8, color="dimgray", va="top", family="monospace")
            y -= dy_l

        t = ax_leg.text(0.09, y, run["url"], transform=ax_leg.transAxes,
                        fontsize=6.5, color="#1155cc", va="top", family="monospace")
        t.set_url(run["url"])
        y -= dy_l + dy_g

    # Markers section
    y -= 0.01
    if y > 0.08:
        ax_leg.text(0.0, y, "Markers", transform=ax_leg.transAxes,
                    fontsize=9.5, fontweight="bold", va="top")
        y -= dy_l + 0.01
        for mk, desc in [
            ("●", "correct dataset"),
            ("✗", "invalidated dataset  ⚠"),
            ("■", "TP=4 arm"),
        ]:
            if y < 0.02:
                break
            ax_leg.text(0.0, y, f"{mk}  {desc}", transform=ax_leg.transAxes,
                        fontsize=7.5, va="top", color="dimgray")
            y -= dy_l * (desc.count("\n") + 1) + 0.005

    # Footer
    run_ids = [g["run_id"] for g in groups]
    fig.text(0.5, 0.002,
             "Runs: " + "  |  ".join(f"#{r}" for r in run_ids),
             ha="center", fontsize=6.0, color="gray", family="monospace")

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", metavar="FILE", default=str(CSV_DEFAULT),
                        help=f"Input CSV (default: {CSV_DEFAULT.name})")
    parser.add_argument("--png", action="store_true", help="Save as pareto_plot.png")
    parser.add_argument("--svg", action="store_true", help="Save as pareto_plot.svg")
    parser.add_argument("-o", "--output", metavar="FILE", help="Save to FILE")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = load_csv(csv_path)
    if df.empty:
        print("No data rows found in CSV.", file=sys.stderr)
        sys.exit(1)

    groups = build_run_groups(df)

    if not args.output and not args.png and not args.svg:
        try:
            fig = build_figure(groups)
            plt.show()
        except Exception:
            matplotlib.use("Agg")
            fig = build_figure(groups)
            plt.show()
        return

    matplotlib.use("Agg")
    fig = build_figure(groups)

    if args.output:
        fig.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")
    elif args.svg:
        fig.savefig("pareto_plot.svg", bbox_inches="tight")
        print("Saved: pareto_plot.svg")
    elif args.png:
        fig.savefig("pareto_plot.png", dpi=150, bbox_inches="tight")
        print("Saved: pareto_plot.png")

    print("\nGitHub Actions run URLs:")
    for g in groups:
        print(f"  {g['run_id']:>14s}  {g['url']}")


if __name__ == "__main__":
    main()
