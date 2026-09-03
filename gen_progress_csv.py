"""Generate progress.csv with guaranteed correct column counts."""
import csv, io

HEADER = [
    "date","run_id","job_url","branch","runner","image","framework",
    "tp","ep","conc","max_running_requests","chunked_prefill_size",
    "kv_offload","hicache_ratio","mtp","spec_steps","spec_draft_tokens",
    "duration_s","requests_ok","throughput_per_gpu_tps","output_tps",
    "ttft_mean_s","ttft_p50_s","ttft_p90_s",
    "itl_mean_ms","itl_p50_ms","itl_p90_ms",
    "intvty_mean","intvty_p50","intvty_p90",
    "gpu_cache_hit_rate","cpu_cache_hit_rate","kv_gpu_usage_pct","kv_cpu_usage_pct",
    "dataset_ok","notes",
]
N = len(HEADER)

def row(**kw):
    r = {k: "" for k in HEADER}
    r.update({k: v for k, v in kw.items() if v != ""})
    return [r[k] for k in HEADER]

ATOM_URL = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31765309673"
SGL_UP_URL = "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31006984179"
F = "https://github.com/giovanniguastiamd/InferenceX/actions/runs"

ROWS = []

# ── ATOM TP=8/EP=1 GPU-resident KV ──────────────────────────────────────────
for (conc, ot, itlm, itlp50, im, ip50, ip90) in [
    (1, 110.8, 5.54, 5.60, 180.51, 178.57, 157.98),
    (2, 136.3, 5.83, 5.69, 171.53, 175.75, 138.31),
    (4, 182.4, 6.21, 5.86, 161.03, 170.65, 123.30),
]:
    ROWS.append(row(
        date="2026-08-25", run_id="31765309673", job_url=ATOM_URL,
        branch="upstream/main", runner="upstream", image="atom-engine", framework="atom",
        tp=8, ep=1, conc=conc, max_running_requests=conc,
        kv_offload="none", mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        output_tps=ot, itl_mean_ms=itlm, itl_p50_ms=itlp50,
        intvty_mean=im, intvty_p50=ip50, intvty_p90=ip90,
        dataset_ok="True", notes="ATOM TP=8/EP=1 GPU-KV reference",
    ))

# ── ATOM TP=4/EP=1 LMCache ──────────────────────────────────────────────────
for (conc, ot, itlm, itlp50, im, ip50, ip90) in [
    (2,  132.4,  6.15,  5.92, 162.60, 168.92, 128.21),
    (4,  171.3,  7.07,  6.82, 141.44, 146.63, 104.93),
    (8,  302.6,  9.97,  9.17, 100.50, 109.17,  69.93),
    (10, 359.8, 11.47, 10.53,  87.18,  94.97,  64.85),
]:
    ROWS.append(row(
        date="2026-08-25", run_id="31765309673", job_url=ATOM_URL,
        branch="upstream/main", runner="upstream", image="atom-engine", framework="atom",
        tp=4, ep=1, conc=conc, max_running_requests=conc,
        kv_offload="lmcache", mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        output_tps=ot, itl_mean_ms=itlm, itl_p50_ms=itlp50,
        intvty_mean=im, intvty_p50=ip50, intvty_p90=ip90,
        dataset_ok="True", notes="ATOM TP=4/EP=1 LMCache reference",
    ))

# ── SGLang upstream PR#2570 ─────────────────────────────────────────────────
for (conc, ot, ttftm, ttftp50, itlm, itlp50, im, ip50, ip90) in [
    (1,   97.8, 3.028, 0.933,  7.06,  6.77, 141.64, 147.71, 110.13),
    (2,  122.2, 1.409, 0.703,  7.91,  7.75, 126.42, 129.03,  98.04),
    (4,  163.6, 1.175, 0.593,  9.98,  8.62, 100.20, 116.01,  80.58),
    (8,  302.5, 1.082, 0.591, 12.41, 11.31,  80.58,  88.42,  57.54),
    (10, 358.2, 1.436, 0.680, 14.90, 12.95,  67.11,  77.22,  47.96),
]:
    ROWS.append(row(
        date="2026-08-05", run_id="31006984179", job_url=SGL_UP_URL,
        branch="upstream/main", runner="upstream", image="sglang-upstream", framework="sglang",
        tp=4, ep=4, conc=conc, max_running_requests=conc,
        kv_offload="hicache", hicache_ratio=1.5,
        mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        output_tps=ot, ttft_mean_s=ttftm, ttft_p50_s=ttftp50,
        itl_mean_ms=itlm, itl_p50_ms=itlp50,
        intvty_mean=im, intvty_p50=ip50, intvty_p90=ip90,
        dataset_ok="True", notes="SGLang PR#2570 upstream cluster",
    ))

# ── mia1 TP=4/EP=4/HiCache c10 — run 32867048910 ───────────────────────────
ROWS.append(row(
    date="2026-08-25", run_id="32867048910", job_url=f"{F}/32867048910",
    branch="testgg", runner="mi355x-amds_03",
    image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
    tp=4, ep=4, conc=10, max_running_requests=10, chunked_prefill_size=32768,
    kv_offload="hicache", hicache_ratio=1.5, mtp="mtp", spec_steps=5, spec_draft_tokens=6,
    duration_s=3600.96, requests_ok=1405,
    throughput_per_gpu_tps=9584.16, output_tps=366.53,
    ttft_mean_s=1.341, ttft_p50_s=0.577, ttft_p90_s=3.500,
    itl_mean_ms=14.82, itl_p50_ms=11.86, itl_p90_ms=18.42,
    intvty_mean=67.48, intvty_p50=84.33, intvty_p90=54.30,
    gpu_cache_hit_rate=0.889, cpu_cache_hit_rate=0.004,
    kv_gpu_usage_pct=61, kv_cpu_usage_pct=100,
    dataset_ok="True", notes="mia1 TP=4/EP=4 HiCache=1.5 first run",
))

# ── mia1 baseline TP=8/EP=8 no-KV — run 32947505370 ────────────────────────
for (conc, ot, itlp50, itlp90, ip90) in [
    (4,  188.8,  7.30,  9.60, 105.0),
    (6,  244.8,  8.30, 11.80,  85.0),
    (8,  325.6,  8.80, 12.20,  82.0),
    (10, 384.8,  9.50, 14.90,  67.0),
]:
    ROWS.append(row(
        date="2026-08-26", run_id="32947505370", job_url=f"{F}/32947505370",
        branch="testgg-maxreq2x", runner="mi355x-amds_03",
        image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
        tp=8, ep=8, conc=conc, max_running_requests=conc, chunked_prefill_size=32768,
        kv_offload="none", mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        output_tps=ot, itl_p50_ms=itlp50, itl_p90_ms=itlp90, intvty_p90=ip90,
        dataset_ok="True", notes="mia1 baseline TP=8/EP=8 no-KV",
    ))
# TP=4 arm (same run)
ROWS.append(row(
    date="2026-08-26", run_id="32947505370", job_url=f"{F}/32947505370",
    branch="testgg-maxreq2x", runner="mi355x-amds_03",
    image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
    tp=4, ep=4, conc=10, max_running_requests=10, chunked_prefill_size=32768,
    kv_offload="hicache", hicache_ratio=1.5, mtp="mtp", spec_steps=5, spec_draft_tokens=6,
    output_tps=364.0, ttft_p50_s=0.509,
    itl_p50_ms=11.90, itl_p90_ms=19.40, intvty_p90=52.0,
    gpu_cache_hit_rate=0.959, kv_gpu_usage_pct=61, kv_cpu_usage_pct=100,
    dataset_ok="True", notes="mia1 baseline TP=4/EP=4 HiCache arm (same run)",
))

# ── I-8: TP=8/EP=1 no-KV — run 32986446019 ─────────────────────────────────
for (conc, ot, itlp50, itlp90, ip90) in [
    (4,  190.0,  6.95,  9.05, 110.5),
    (6,  251.2,  7.87, 12.15,  82.3),
    (8,  326.4,  8.38, 12.35,  81.1),
    (10, 384.8,  9.27, 14.25,  70.2),
]:
    ROWS.append(row(
        date="2026-08-27", run_id="32986446019", job_url=f"{F}/32986446019",
        branch="testgg-maxreq2x", runner="mi355x-amds_03",
        image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
        tp=8, ep=1, conc=conc, max_running_requests=conc, chunked_prefill_size=32768,
        kv_offload="none", mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        output_tps=ot, itl_p50_ms=itlp50, itl_p90_ms=itlp90, intvty_p90=ip90,
        dataset_ok="True", notes="I-8 TP=8/EP=1 no-KV",
    ))

# ── Bundle I-1+I-3+I-7 re-run — run 33065762186 ─────────────────────────────
for (conc, ot, itlp50, itlp90, ip90) in [
    (4,  191.2, 6.96,  9.18, 109.0),
    (6,     "", 8.03, 12.05,  83.0),
    (10,    "", 9.34, 14.73,  67.9),
]:
    ROWS.append(row(
        date="2026-08-28", run_id="33065762186", job_url=f"{F}/33065762186",
        branch="testgg-maxreq2x", runner="mi355x-amds_03",
        image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
        tp=8, ep=1, conc=conc, max_running_requests=conc, chunked_prefill_size=16384,
        kv_offload="none", mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        output_tps=ot, itl_p50_ms=itlp50, itl_p90_ms=itlp90, intvty_p90=ip90,
        dataset_ok="True", notes="Bundle I-1+I-3+I-7 re-run env-fix — neutral",
    ))

# ── PR #2777 upstream 40xMI355X SLURM — run 33373633743 ─────────────────────
# TP=8/EP=1 interactivity arm
for (conc, ip90) in [(1, 132.1), (2, 123.3), (4, 99.2), (10, 69.1)]:
    ROWS.append(row(
        date="2026-08-31", run_id="33373633743", job_url=f"{F}/33373633743",
        branch="testgg-maxreq2x", runner="mi355x-amds",
        image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
        tp=8, ep=1, conc=conc, max_running_requests=conc, chunked_prefill_size=32768,
        kv_offload="hicache", hicache_ratio=1.5, mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        intvty_p90=ip90, dataset_ok="True",
        notes="PR#2777 40xMI355X SLURM TP=8/EP=1 HiCache=1.5",
    ))
# TP=4/EP=4 arm
for (conc, ip90, ot) in [(4, 80.4, ""), (8, 56.9, ""), (10, "", 353.2), (12, "", 339.6)]:
    ROWS.append(row(
        date="2026-08-31", run_id="33373633743", job_url=f"{F}/33373633743",
        branch="testgg-maxreq2x", runner="mi355x-amds",
        image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
        tp=4, ep=4, conc=conc, max_running_requests=conc, chunked_prefill_size=32768,
        kv_offload="hicache", hicache_ratio=1.5, mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        output_tps=ot, intvty_p90=ip90, dataset_ok="True",
        notes="PR#2777 40xMI355X SLURM TP=4/EP=4 HiCache=1.5",
    ))

# ── HiCache c12 comparison (run 33647171278, 256k INVALIDATED) ──────────────
ROWS.append(row(
    date="2026-09-02", run_id="33647171278", job_url=f"{F}/33647171278",
    branch="testgg-qr-int4", runner="mi355x-amds_03",
    image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
    tp=4, ep=4, conc=12, max_running_requests=12, chunked_prefill_size=32768,
    kv_offload="hicache", hicache_ratio=1.5, mtp="mtp", spec_steps=5, spec_draft_tokens=6,
    itl_p90_ms=21.0, intvty_p90=47.7, kv_gpu_usage_pct=62,
    dataset_ok="False", notes="HiCache c12 256k INVALIDATED — qualitative HiCache=no-HiCache",
))

# ── no-HiCache Pareto sweep (run 33661191728, 256k INVALIDATED) ─────────────
for (conc, ot, itlp50, itlp90, ip90, kv, note) in [
    (2,  140.0,   7.40,   9.50, 105.6,  18, "no-HiCache Pareto 256k INVALIDATED"),
    (4,  184.0,   8.10,  10.90,  92.0,  26, "no-HiCache Pareto 256k INVALIDATED"),
    (8,  316.0,  10.30,  15.00,  66.7,  50, "no-HiCache Pareto 256k INVALIDATED"),
    (10, 368.0,  11.60,  18.50,  54.1,  63, "no-HiCache Pareto 256k INVALIDATED"),
    (12, 424.0,  11.80,  20.40,  48.9,  60, "no-HiCache Pareto 256k INVALIDATED"),
    (20, 316.0,  25.20, 150.40,   6.6, 100, "no-HiCache Pareto 256k INVALIDATED — OOM onset"),
    (24, 152.0, 115.50, 257.10,   3.9, 100, "no-HiCache Pareto 256k INVALIDATED — OOM collapse"),
]:
    ROWS.append(row(
        date="2026-09-02", run_id="33661191728", job_url=f"{F}/33661191728",
        branch="testgg-qr-int4", runner="mi355x-amds_03",
        image="v0.5.16-rocm720-mi35x-20260728", framework="sglang",
        tp=4, ep=4, conc=conc, max_running_requests=conc, chunked_prefill_size=32768,
        kv_offload="none", mtp="mtp", spec_steps=5, spec_draft_tokens=6,
        output_tps=ot, itl_p50_ms=itlp50, itl_p90_ms=itlp90, intvty_p90=ip90,
        kv_gpu_usage_pct=kv, dataset_ok="False", notes=note,
    ))

# validate
for i, r in enumerate(ROWS):
    assert len(r) == N, f"Row {i} has {len(r)} fields (expected {N})"

COMMENTS = (
    "# GLM-5.2 MXFP4 / MI355X — tuning campaign progress\n"
    "# throughput_per_gpu_tps = (input+output) tok/s per GPU\n"
    "# output_tps = output-only tok/s TOTAL across all GPUs (= output_per_chip * tp)\n"
    "# itl_* values in ms. intvty_p90 = 1/ITL_p90 (tok/s/user floor, higher=better)\n"
    "# dataset_ok: False = 256k-capped ISL (p90~170k); True = unfiltered (p90~283k)\n"
)

buf = io.StringIO()
w = csv.writer(buf, lineterminator="\n")
w.writerow(HEADER)
w.writerows(ROWS)

with open("progress.csv", "w", newline="", encoding="utf-8") as f:
    f.write(COMMENTS)
    f.write(buf.getvalue())

print(f"Written {len(ROWS)} rows x {N} columns — OK")
