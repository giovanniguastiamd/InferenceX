# InferenceX SGLang Tuning Methodology

**Scope:** single-node SGLang serving on AMD MI300X/MI355X · agentic-coding scenario
**Audience:** engineers onboarding a new model+hardware combination to InferenceX

> **Campaign data (GLM-5.2 FP4 / MI355X / Aug–Sep 2026):** [campaign_glm52_mi355x_2026.md](campaign_glm52_mi355x_2026.md)
> **SGLang feature gaps vs ATOM (GLM-5.2 specific):** [SGL_missing_features_01_09_2026.md](SGL_missing_features_01_09_2026.md)

---

# 1. Optimization objectives (Pareto)

The evaluation space is divided into **three distinct regimes** by CONC:

| Regime | Indicative CONC | Primary metric | Secondary metric |
|--------|-----------------|----------------|------------------|
| **Interactivity** | 1 – 6 | P90 tok/s/user (maximize) | P90 TTFT (minimize) |
| **Crossover** | 6 – 12 | Crossover CONC C\* (maximize) | tok/s/GPU @ C\* (maximize) |
| **Throughput** | 12 – 24 | tok/s/GPU (maximize) | P90 ITL (minimize) |

**Operational definition of C\*:** the minimum CONC at which `tok/s/user` drops below 80% of the value measured at CONC=1. Configurations with a higher C\* maintain interactivity for more simultaneous users.

Simultaneously minimize/maximize (Pareto tri-objective):

| Objective | Regime | Direction |
|-----------|--------|-----------|
| P90 interactivity @ CONC=4 (tok/s/user) | Interactivity | maximize |
| Crossover CONC C\* | Crossover | maximize |
| Throughput/chip @ CONC=16 (tok/s/GPU) | Throughput | maximize |

Configuration `A` dominates `B` if it improves at least one of the three objectives without degrading the others.

> **Implication for the sweep:** each server lifetime must sample CONC densely in the crossover zone (e.g. [4, 6, 8, 10, 12]) to locate C\* precisely, not just the extreme points.

---

# 2. SGLang architectural constraint

**SGLang does not support runtime mutation** of scheduler parameters. All of the following knobs are fixed at server startup:
- `--chunked-prefill-size`
- `--max-running-requests`
- `--cuda-graph-max-bs`
- `--mem-fraction-static`
- `--speculative-*`
- `--hicache-*`

**Only client-side variable (no restart):** CONC (concurrency).

**Implication for the tuner:** one server lifetime → sweep of multiple CONC values. For each different structural configuration: mandatory restart.

```
For each config C (chunked-prefill, max-running-requests, ...):
  → start server (1 restart, ~15-20 min warmup)
  → measure CONC = [4, 6, 8, 10, 12] sequentially
  → stop server
  = savings: 1 restart × N_conc instead of N_conc restarts
```

---

# 3. Parameter space

Parameter groups ordered by restart cost. Default values are examples from the GLM-5.2 FP4 / MI355X campaign; adapt ranges for new models.

## Group A — Client-only (no restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **CONC** | 1, 2, 4, 8, **10**, 12, 16, 24 | Higher GPU utilization, higher throughput | ITL degrades, TTFT increases, OOM at high CONC | `max-running-requests ≥ CONC`; `HICACHE_RATIO` needs recalibration at high CONC |
| **DURATION** | 120s (smoke), 600s (tuning), 3600s (full) | — | P90/P99 unreliable below ~100 requests | — |

## Group B — Scheduler / Prefill (restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **`--chunked-prefill-size`** | 8192, 16384, **32768**, 65536, 131072 | Low values: decode interleaved more often → ITL and interactivity improve. High values: efficient prefill, input throughput | 131072 → OOM observed. Low values: scheduling overhead | `mem-fraction-static`: 131k chunk requires ~7 GiB/rank headroom vs ~1.7 GiB/rank at 32k |
| **`--max-running-requests`** | `0.5×C`, **`1×C`**, `1.5×C`, `2×C`, `3×C` | More in-flight requests → higher throughput if server is not the bottleneck | Too high: KV cache exhausted, OOM. ITL degrades | Must be ≥ CONC; `cuda-graph-max-bs` ≥ `max-running-requests` |
| **`--cuda-graph-max-bs`** | `1×MRR`, `1.5×MRR`, `2×MRR` | Reduces "graph capture miss" with EAGLE draft+verify | More HBM for graphs, slower startup | With EAGLE num-steps=5 the draft batch can temporarily exceed `max-running-requests`. Note: SGLang already auto-interpolates a list of sizes up to max_bs — explicit per-size override has marginal benefit. |
| **`--mem-fraction-static`** | 0.80, 0.83, **0.85**, 0.87 | More HBM to KV → higher hit rate | Less headroom for activations. OOM with high chunked-prefill | Interacts with `chunked-prefill-size` and `HICACHE_RATIO` |

## Group C — Speculative Decoding EAGLE / MTP (restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **MTP on/off** | **EAGLE** / off | ON: acceptance length ~3.6 → 2-3× decode throughput | Draft overhead increases ITL for short requests | `cuda-graph-max-bs` must cover draft batch |
| **`--speculative-num-steps`** | 3, 4, **5** | More steps → more tokens accepted → higher throughput | Overhead grows linearly; at high CONC draft competes for HBM | Golden AL observed: 3.61 (simulated) |
| **`--speculative-num-draft-tokens`** | 3, 4, **6** | More drafts = more potential accepts | If acceptance rate drops, cost without benefit | `num-steps × num-draft-tokens` = max draft batch size |
| **`--speculative-eagle-topk`** | **1**, 2 | topk>1: more diverse sampling | Quadratic overhead, rarely beneficial | Explore only after stabilizing other parameters |

## Group D — KV Cache / HiCache (restart)

> **Default rule (§5.1):** always start with `kv-offloading: none`. Enable HiCache only if a probe run shows `gpu_kv_usage > 85–95%`. For MLA models (e.g. GLM-5.2) the KV footprint is 10–20× smaller than dense MHA → often fits in HBM without offloading.

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **KV offload backend** | **none** ← default, hicache (fallback if gpu_kv_usage > 85%), mooncake | none: maximum speed, no CPU-GPU bottleneck. HiCache: prevents OOM at high CONC when KV exceeds HBM | HiCache: CPU-GPU bandwidth bottleneck, degrades ITL by 3–4× when KV fits in HBM. Mooncake: additional infrastructure | — |
| **`HICACHE_RATIO`** | 0.5, 1.0, **1.5**, 2.0, 2.5 | More host DRAM → higher hit rate at high CONC | Host OOM at too-high ratios; cluster DRAM budget may limit | Device pool per rank depends on TP. On TP=4 MI355X: ~182.7 GB/rank |
| **`HICACHE_WRITE_POLICY`** | **write_through**, write_through_selective | write_through_selective (PR #2679): writes DRAM only for high-reusability prefill → less wasted bandwidth | More CPU-GPU bandwidth during prefill with write_through | — |
| **`HICACHE_IO_BACKEND`** | **direct**, asyncio | direct: low latency | asyncio: potentially better under heavy I/O | — |
| **`--kv-cache-dtype`** | **fp8_e4m3**, bf16 | fp8: 2× less HBM → more tokens cached → may enable GPU-resident at higher CONC | bf16: better KV quality | Structural: invalidates cache |
| **Mooncake L3** (`L3_PER_RANK_GB`) | 20, **40**, 60, 80 | More L3 → higher hit rate for very long contexts | Requires mooncake infrastructure | Only with `KV_OFFLOAD_BACKEND=mooncake` |

## Group E — Parallelism / Topology (restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **TP** | **4**, 8 | TP8: less HBM/rank → more KV cached; fewer weights/GPU → faster decode when `t_weights` dominates | More RCCL overhead, collective latency | `HICACHE_RATIO` needs recalibration per rank |
| **EP** | 1, 2, **4**, 8 | EP=1: eliminates MoE all-to-all → ITL ↓ at low CONC. High EP: less HBM/rank for expert weights | EP=1: more weights/GPU → less HBM for KV. High EP: communication overhead for expert routing | Must divide the number of experts; check HBM budget before switching |
| **DP Attention** | **OFF**, ON | ON: reduced KV/rank, scalable at high CONC | **BROKEN in SGLang ROCm** (collective hang on long-context prefill). Re-enable only after confirmed upstream fix | `SGLANG_DP_USE_GATHERV=1`, `SGLANG_DP_USE_REDUCE_SCATTER=1` |
| **DCP** | **OFF**, ON | Separates prefill/decode → decode not blocked | Complex infrastructure. Validated for Kimi-K3, not for GLM-5.2 | Second phase |

## Group F — Docker Platform (restart, high cost)

| Parameter | Values | Notes |
|-----------|--------|-------|
| **Docker image** | e.g. v0.5.16-rocm720 (baseline), v0.5.18-rocm720, v0.5.18-rocm724 | Treat as a platform variable. Controlled A/B test. ROCm version changes HIP graph path which affects chunked-prefill and CUDA graph coverage |

---

# 4. Multi-fidelity

| Tier | Duration | Use |
|------|----------|-----|
| Screening | 120–300s | Eliminate clearly worse configurations |
| Tuning | 600–1200s | Reliable ranking of candidates |
| Validation | 3600s (full) | Only Pareto-candidate configurations |

Minimum requirements: ≥ 100 completed requests for reliable P90.

---

# 5. Lessons learned (recipe development methodology)

This section captures generalizable principles derived from the GLM-5.2 / MI355X tuning campaign (Aug–Sep 2026). They apply to most new model+hardware combinations in InferenceX.

---

## 5.1 KV offloading: measure before you enable

**The mistake:** HiCache (CPU KV offloading) is often enabled by default in new recipes — preemptively, without verifying whether the KV cache actually fits in HBM at the target ISL and concurrency. This is a silent performance regression: if the KV fits in HBM, HiCache adds a CPU↔GPU DRAM bottleneck that can degrade ITL by **3–4×** with zero benefit.

**Rule: always start without KV offloading. Enable it only if the data says you need it.**

### Procedure for any new recipe:

```
Step 1 — Probe run (no-HiCache, target CONC, full-duration trace):
  kv-offloading: none
  Monitor: gpu_kv_usage from server metrics (SGLang: /metrics endpoint)

Step 2 — Decision:
  gpu_kv_usage < 85%  →  KV fits comfortably  →  keep no-HiCache
  gpu_kv_usage 85–95% →  marginal  →  test both; prefer no-HiCache if stable
  gpu_kv_usage > 95%  →  OOM risk  →  enable HiCache, tune HICACHE_RATIO

Step 3 — If HiCache is needed:
  Start with HICACHE_RATIO=1.5 and monitor cpu_kv_usage.
  If cpu_kv_usage saturates → increase ratio or reduce CONC.
  Document the headroom in the recipe changelog.
```

### Why MLA / latent-attention models benefit especially:

MLA (Multi-head Latent Attention, used in GLM-5.2 and similar models) compresses the KV cache to a ~512-dimensional latent vector per token instead of storing full K+V per head. This makes the KV footprint roughly **10–20× smaller** than a dense MHA model of equivalent size. For MI355X (80 GB HBM per GPU), KV often fits in HBM at moderate concurrency even with very long ISL. For dense-KV models (e.g. Llama-3-70B) the same ISL and concurrency would likely require offloading.

**Always check the model's KV architecture before assuming HiCache is needed.**

---

## 5.2 Concurrency sweep: cover the full Pareto front before tuning parameters

**The mistake:** tuning structural parameters (chunked-prefill, MRR, etc.) at a single fixed CONC. The optimal parameter may differ across regimes (interactivity vs crossover vs throughput — see §1).

**Rule:** for any new candidate config, run at least CONC ∈ {4, 8, 12} before concluding. A config that wins at c4 may lose at c12 and vice versa.

---

## 5.3 Machine differences invalidate cross-machine comparisons

**The mistake:** comparing absolute metrics (ITL, throughput) across machines with different GPU count, NFS speed, or available HBM.

**Rule:** always report which runner produced a result. Use relative comparisons (A vs B on the same runner) rather than absolute cross-runner claims. When porting a recipe to a new machine, re-establish a local baseline before interpreting deltas.

---

## 5.4 Warmup and trace burn-in

**The mistake:** measuring before the KV radix cache is warmed up. Early timeslices show artificially high TTFT and low cache hit rates.

**Rule:** discard the first 10–15 minutes of any 60-minute run (warmup period). Verify `prefix_cache_hit` reaches a stable plateau (typically >90% for long-context agentic traces) before reading P90 metrics. The benchmark harness marks warmup requests automatically — check the `131 warmup` line in the run summary.

---

## 5.5 INT4 all-reduce: check tensor size before expecting gains

Before enabling `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` (or INT8), verify that the all-reduce tensor will exceed the activation threshold (`_QR_MIN_SIZE`) at the target TP degree:

- At TP=4: threshold ≈ 16 MB. A decode step with 4 tokens produces ~57 KB — **always below threshold**.
- INT4 can only fire on the **prefill path** when the prefill batch accumulates >~1200 tokens at TP=4.
- If HiCache is active and cpu_kv_usage is high, the DRAM I/O bottleneck masks any INT4 gain anyway.

**Rule:** INT4 all-reduce benefits only if (a) tensors are large enough to exceed the threshold AND (b) the bottleneck is communication, not DRAM. Verify both conditions before attributing gains to INT4.

---

# 6. Docker launcher requirements and validation checklist

This section documents the requirements that `runners/launch_mi355x-amds.sh` (Docker fallback) must satisfy to produce results comparable with the upstream SLURM/enroot path. Failures here cause silent measurement errors that are very hard to detect.

## Why this exists

The upstream cluster uses `srun --export=ALL` (SLURM + enroot), which automatically propagates **all** workflow environment variables into the container. The Docker fallback used on mia1-class machines uses an explicit `-e VAR` list and must be kept in sync with the workflow manually.

**Bug discovered 2026-09-03:** `MODEL_PREFIX` was absent from the `-e` list. Result: `infmax_model_prefix` was empty in all JSON artifacts, causing `benchmark_lib.sh` to fall through to the 256k-capped dataset (`cc-traces-weka-062126-256k`) instead of the unfiltered corpus (`cc-traces-weka-062126`). The upstream PR 2777 sweep (SLURM) ran with ISL p90 ≈ 283k tokens; mia1 runs had ISL p90 ≈ 170k — a ~40% underestimate of sequence length that invalidates direct comparisons.

Fix: added `MODEL_PREFIX` (and other missing vars) to the `-e` list in commit `541e137` on all active branches.

## Required env vars: Docker `-e` list vs workflow

The Docker `-e` list in `launch_mi355x-amds.sh` must include every workflow env var that is consumed by `benchmarks/benchmark_lib.sh` or any recipe script and is **not** internally set by the launcher itself.

### Vars that MUST be in the Docker `-e` list

| Env var | Set by workflow | Used in benchmarks | Critical if missing |
|---------|----------------|-------------------|---------------------|
| `MODEL` | ✅ | model path to serve | ✅ fatal |
| `MODEL_PREFIX` | ✅ | dataset loader selection (`semianalysis_cc_traces_weka_062126` vs 256k variant) | ✅ **silent wrong dataset** |
| `MODEL_NAME` | launcher-computed | model short name | ✅ |
| `PRECISION` | ✅ | quantization flags | ✅ |
| `FRAMEWORK` | ✅ | launch path selection | ✅ |
| `TP` | ✅ | tensor parallel degree | ✅ |
| `EP_SIZE` | ✅ | expert parallel degree | ✅ |
| `PP_SIZE` | ✅ | pipeline parallel | medium |
| `DCP_SIZE` | ✅ | decode context parallel | medium |
| `PCP_SIZE` | ✅ | prefill context parallel | medium |
| `DP_ATTENTION` | ✅ | data-parallel attention flag | medium |
| `CONC` | ✅ | concurrency target | ✅ |
| `KV_OFFLOADING` | ✅ | HiCache / none / cpu | ✅ |
| `KV_OFFLOAD_BACKEND` | ✅ | HiCache backend config | ✅ |
| `KV_OFFLOAD_BACKEND_METADATA` | ✅ | backend metadata | medium |
| `MAX_MODEL_LEN` | ✅ | context window limit | ✅ |
| `SPEC_DECODING` | ✅ | MTP / EAGLE / none | ✅ |
| `DISAGG` | ✅ | disaggregation mode | medium |
| `SCENARIO_TYPE` | ✅ | agentic-coding / fixed-seq-len | ✅ |
| `SCENARIO_SUBDIR` | ✅ | subdirectory routing | ✅ |
| `IS_AGENTIC` | ✅ | agentic flag | ✅ |
| `TOTAL_CPU_DRAM_GB` | ✅ | HiCache DRAM budget | medium |
| `DURATION` | ✅ | benchmark duration (s) | ✅ |
| `RUN_EVAL` | ✅ | trigger GSM8K after bmk | medium |
| `EVAL_ONLY` | ✅ | skip bmk, run eval only | medium |
| `EVAL_LIMIT` | ✅ | limit eval samples | low |
| `EXP_NAME` | ✅ | experiment name in JSON output | low |
| `RECIPE_FINGERPRINT` | ✅ | recipe hash in JSON output | low |
| `HF_TOKEN` | ✅ | gated model access | medium |
| `REQUIRE_POWER` | ✅ | power metric gating | low |
| `RESULT_DIR` | workflow-fixed | output directory | ✅ |
| `PORT` | launcher-computed | server port | ✅ |
| `RUNNER_NAME` | runner env | runner identifier | medium |
| `RESULT_FILENAME` | launcher-computed | output filename | ✅ |
| `HF_HUB_CACHE` | workflow-fixed | HF cache path | medium |
| `HF_HOME` | launcher-computed | HF home inside container | medium |
| `MODEL_PATH` | runner `.env` | model weights path | ✅ |
| `AIPERF_DATASET_MMAP_CACHE_DIR` | launcher-computed | aiperf cache | medium |

### Vars set only via runner `.env` (not workflow, not `-e` list)

These are machine-specific and must be set in `~/actions-runner/.env` on each runner:

| Env var | Purpose | Example value |
|---------|---------|---------------|
| `FORCE_DOCKER=1` | bypass SLURM, use Docker fallback | `1` |
| `MODEL_PATH` | local model weights directory | `/it-share/models/GLM-5.2-MXFP4` |
| `HF_HUB_CACHE_HOST` | NFS HF cache mount point | `/mnt/hf_hub_cache` |
| `WEKA_LOADER_OVERRIDE` | force specific dataset loader | unset (use MODEL_PREFIX default) |

### Vars that are optional / conditional (`${VAR:+-e VAR}` pattern)

These are only passed if set in runner `.env` (recipe-specific overrides):

- `CHUNKED_PREFILL_SIZE_OVERRIDE` — override chunked prefill size
- `ROCM_QUICK_REDUCE_QUANTIZATION` — INT4/INT8 all-reduce
- `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` — INT4 cast companion
- `SGLANG_USE_AITER_UNIFIED_ATTN` — AITER unified attention
- `CUDA_GRAPH_BS_LIST_OVERRIDE` — custom CUDA graph batch sizes
- `HICACHE_RATIO` — HiCache DRAM ratio override
- `HICACHE_WRITE_POLICY` — HiCache write policy override

## Validation checklist: before running a new campaign

Run through this checklist whenever starting benchmarks on a new machine or after a significant launcher update.

### 1. Dataset validation (most critical)

After the first run, **immediately check** the JSON artifact:

```bash
python -c "
import json, glob, sys
f = glob.glob('/tmp/results/*.json')[0]
d = json.load(open(f))
prefix = d.get('infmax_model_prefix', '')
loader = d['dataset']['loader']
isl_p90 = d['request_metrics']['tokens']['input']['p90']
print(f'model_prefix : {repr(prefix)}')
print(f'dataset loader: {loader}')
print(f'ISL p90       : {isl_p90:.0f} tokens')
print()
if prefix == '' or '256k' in loader:
    print('WARNING: MODEL_PREFIX empty or 256k dataset — check Docker -e list!')
else:
    print('OK: unfiltered dataset, MODEL_PREFIX propagated correctly')
"
```

Expected values for GLM-5.2 on mia1:
- `infmax_model_prefix`: `'glm5.2'` (not empty)
- `loader`: `semianalysis_cc_traces_weka_062126` (not `_256k`)
- ISL p90: ~280k tokens (not ~130-170k which indicates 256k dataset)

### 2. Comparability with upstream

Before claiming a result matches or beats upstream (ATOM or PR sweep results):

- [ ] Same dataset loader? Check `dataset.loader` in both JSONs.
- [ ] Same ISL distribution? Compare `tokens.input.p50` / `p90`.
- [ ] Same image? Compare `image` field.
- [ ] Same concurrency and TP/EP? Compare `conc`, `tp`, `ep`.
- [ ] Same machine class? Note differences in GPU count, HBM, NFS speed.

### 3. Launcher sync check

When the upstream adds new env vars to `benchmark-tmpl.yml` (workflow), check that `launch_mi355x-amds.sh` Docker `-e` list is updated:

```bash
# Run after merging upstream changes
grep "^  [A-Z_]*:" .github/workflows/benchmark-tmpl.yml | awk '{print $1}' | tr -d ':' > /tmp/workflow_vars.txt
grep "\-e [A-Z_]*" runners/launch_mi355x-amds.sh | grep -oP '\-e \K[A-Z_]+' > /tmp/docker_vars.txt
comm -23 <(sort /tmp/workflow_vars.txt) <(sort /tmp/docker_vars.txt)
# Any output = vars in workflow but not in Docker -e list
```
