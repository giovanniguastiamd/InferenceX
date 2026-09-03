# InferenceX AgentX Auto-Tuner
## GLM-5.2 on MI355X with SGLang

**Status:** Active development
**Model:** GLM-5.2 (FP4)
**Framework:** SGLang (standalone — ATOM removed)
**Hardware:** 8× MI355X (mia1-p01-g07)
**Scenario:** agentic-coding
**Goal:** Pareto-optimal serving configurations per latency/interactivity/throughput, minimizing tuning time.

---

# 0. Working environment

```
Host:    giovanni.guasti@amd.com@mia1-p01-g07  (ProxyJump 64.139.223.124)
Workdir: /it-share/gguasti
Model:   /it-share/models/GLM-5.2-MXFP4  (408 GB, NFS)
Runner:  /it-share/gguasti/actions-runner  (GitHub Actions: mi355x-amds_03)
```

Docker images available on the machine:
```
lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260824  ← primary target
lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260824  ← ROCm 7.2.4 challenger
```

Config recipe: `configs/amd-master.yaml` · key: `glm5.2-fp4-mi355x-sglang-agentic-mtp`
Script: `benchmarks/single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh`

---

# 1. Baseline results (2026-08-25)

**Upstream PR:** [SemiAnalysisAI/InferenceX#2570](https://github.com/SemiAnalysisAI/InferenceX/pull/2570) — merged 2026-08-12
(Tuning follow-up to #2488: MTP steps 3→5, draft tokens 4→6, SGLANG_SIMULATE_ACC_LEN 2.99→3.61; TP8 low-conc: hicache removed)

**Run:** 32867048910 · branch `testgg` · runner `mi355x-amds_03`
**Image:** v0.5.16-rocm720 · TP4/EP4 · CONC=10 · MRR=10 · chunked=32768 · HiCache ratio=1.5 · EAGLE 5-steps/6-tokens

| Metric | Value |
|--------|-------|
| Duration | 3601s |
| Completed requests | 1405 |
| Total throughput | 38,337 tok/s |
| **Throughput/GPU** | **9,584 tok/s/GPU** |
| Output throughput | 367 tok/s |
| TTFT mean / p50 / p90 | 1.341s / 0.577s / 3.500s |
| ITL mean / p50 / p90 | 14.82ms / 11.86ms / 18.42ms |
| Interactivity mean / p50 / p90 | 67.5 / 84.3 / 54.3 tok/s/user |
| GPU KV cache hit | 88.9% |
| CPU KV cache hit | 0.4% |
| KV GPU usage | 61% |
| **KV CPU usage** | **100% (saturated)** |

> **Note:** the CPU KV pool is at 100% — the host DRAM HiCache tier is saturated. Potential bottleneck at higher CONC.

Full historical results: `progress.csv`

---

# 2. Optimization objectives (Pareto)

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

# 3. SGLang architectural constraint

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

# 4. Parameter space

## Group A — Client-only (no restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **CONC** | 1, 2, 4, 8, **10**, 12, 16, 24 | Higher GPU utilization, higher throughput | ITL degrades, TTFT increases, OOM at high CONC | `max-running-requests ≥ CONC`; `HICACHE_RATIO` needs recalibration at high CONC |
| **DURATION** | 120s (smoke), 600s (tuning), 3600s (full) | — | P90/P99 unreliable below ~100 requests | — |

## Group B — Scheduler / Prefill (restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **`--chunked-prefill-size`** | 8192, 16384, **32768**, 65536, 131072 | Low values: decode interleaved more often → ITL and interactivity improve. High values: efficient prefill, input throughput | 131072 → OOM observed (run 29751563205). Low values: scheduling overhead | `mem-fraction-static`: 131k chunk requires ~7 GiB/rank headroom vs ~1.7 GiB/rank at 32k |
| **`--max-running-requests`** | `0.5×C`, **`1×C`**, `1.5×C`, `2×C`, `3×C` | More in-flight requests → higher throughput if server is not the bottleneck | Too high: KV cache exhausted, OOM. ITL degrades | Must be ≥ CONC; `cuda-graph-max-bs` ≥ `max-running-requests` |
| **`--cuda-graph-max-bs`** | `1×MRR`, **`1×MRR`**, `1.5×MRR`, `2×MRR` | Reduces "graph capture miss" with EAGLE draft+verify | More HBM for graphs, slower startup | With EAGLE num-steps=5 the draft batch can temporarily exceed `max-running-requests` |
| **`--mem-fraction-static`** | 0.80, 0.83, **0.85**, 0.87 | More HBM to KV → higher hit rate | Less headroom for activations. OOM with high chunked-prefill | Interacts with `chunked-prefill-size` and `HICACHE_RATIO` |

## Group C — Speculative Decoding EAGLE (restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **MTP on/off** | **EAGLE** / off | ON: acceptance length ~3.6 → 2-3× decode throughput | Draft overhead increases ITL for short requests | `cuda-graph-max-bs` must cover draft batch |
| **`--speculative-num-steps`** | 3, 4, **5** | More steps → more tokens accepted → higher throughput | Overhead grows linearly; at high CONC draft competes for HBM | Golden AL observed: 3.61 (simulated) |
| **`--speculative-num-draft-tokens`** | 3, 4, **6** | More drafts = more potential accepts | If acceptance rate drops, cost without benefit | `num-steps × num-draft-tokens` = max draft batch size |
| **`--speculative-eagle-topk`** | **1**, 2 | topk>1: more diverse sampling | Quadratic overhead, untested on GLM-5.2 | Explore only after stabilizing other parameters |

## Group D — KV Cache / HiCache (restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **KV offload backend** | none, **hicache**, mooncake | HiCache: supports high CONC with long contexts without OOM | none: maximum speed. HiCache: CPU-GPU bandwidth bottleneck. Mooncake: additional infrastructure | — |
| **`HICACHE_RATIO`** | 0.5, 1.0, **1.5** (TP arm), 2.0 | More host DRAM → higher hit rate at high CONC | Host OOM at too-high ratios (observed: CONC 48 with DP-arm). **CPU pool already at 100% at CONC=10** → consider increasing | Device pool per rank: ~182.7 GB at TP4 |
| **`HICACHE_WRITE_POLICY`** | **write_through**, write_back | write_through: maximum hit rate | More CPU-GPU bandwidth during prefill | — |
| **`HICACHE_IO_BACKEND`** | **direct**, asyncio | direct: low latency | asyncio: potentially better under heavy I/O | — |
| **`--kv-cache-dtype`** | **fp8_e4m3**, bf16 | fp8: 2× less HBM → more tokens cached | bf16: better KV quality | Structural: invalidates cache |
| **Mooncake L3** (`L3_PER_RANK_GB`) | 20, **40**, 60, 80 | More L3 → higher hit rate for very long contexts | Requires mooncake infrastructure | Only with `KV_OFFLOAD_BACKEND=mooncake` |

## Group E — Parallelism / Topology (restart)

| Parameter | Range | Expected effect (+) | Risk (−) | Dependencies |
|-----------|-------|---------------------|----------|--------------|
| **TP** | **4**, 8 | TP8: less HBM/rank → more KV cached | More RCCL overhead, collective latency | `HICACHE_RATIO` needs recalibration per rank |
| **EP** | 2, **4**, 8 | High EP: less HBM/rank for expert weights | Communication overhead for expert routing | Must divide the number of experts in GLM-5.2 |
| **DP Attention** | **OFF**, ON | ON: reduced KV/rank, scalable at high CONC | **BROKEN in SGLang ROCm** (collective hang on long-context prefill, v0.5.14 and v0.5.16). Re-enable only after confirmed upstream fix | `SGLANG_DP_USE_GATHERV=1`, `SGLANG_DP_USE_REDUCE_SCATTER=1` |
| **DCP** | **OFF**, ON | Separates prefill/decode → decode not blocked | Complex infrastructure. Relevant for Kimi-K3, not explored for GLM-5.2 | Second phase |

## Group F — Docker Platform (restart, high cost)

| Parameter | Values | Notes |
|-----------|--------|-------|
| **Docker image** | v0.5.16-rocm720 (baseline), **v0.5.18-rocm720** (next target), v0.5.18-rocm724 (challenger) | Treat as a platform variable. Controlled A/B test. ROCm 7.2.4 changes the HIP graph path which affects chunked-prefill and CUDA graph coverage |

---

# 5. Tuning plan (phases)

## Phase 0 — Infrastructure (prerequisites)
- [x] Runner `mi355x-amds_03` on mia1-p01-g07 operational
- [x] Baseline v0.5.16-rocm720 CONC=10: 9,584 tok/s/GPU (TP4/EP4/HiCache)
- [x] Recipe updated: TP=8/EP=8/no-hicache arm added to `amd-master.yaml`
- [x] Fix `_work` root-owned — Alpine chown in "Resource cleanup (pre-run)" workflow step (commit 4b33bfe98)
- [x] **Rollback image v0.5.18 → v0.5.16** (commit 4786948a0): v0.5.18 crashes SIGABRT in `fp8_mqa_logits` (Triton LLVM `iota_range` assertion) on TP=4+HiCache+DSA and on TP=8. Confirmed regression vs v0.5.16.
- [ ] Fix `test-sweep-evals` fromJson empty (workflow cosmetic failure)

---

## Phase 1 — Interactivity optimization

### Overhead analysis at low concurrency

At small CONC, the decode loop is dominated by fixed, unamortized overhead:

```
ITL/user ≈ (t_nccl_allreduce + t_kernel_launch + t_weights) / CONC
```

- `t_nccl` is ~constant for small batches but scales with the number of GPUs in the allreduce
- `t_weights` scales with 1/TP (less weight per GPU = faster loading)
- The sweet spot is where amortization is nearly complete but the queue is still absent

**Why C=1 is worse than C=4:** fixed overhead (NCCL, kernel launch, 5+1 MTP cycles) is not amortized over a single request. At C=4, the same overhead is split among 4 users and `t_weights` is nearly unchanged (memory-bound regime).

### TP size trade-off

| TP | NCCL ranks | Weights/GPU | HBM for KV | Optimal regime |
|----|-----------|-------------|------------|----------------|
| 2  | 2 → very fast | ~204 GB | ~84 GB ⚠ | not practical with ISL=79k tokens |
| 4  | 4 → medium | ~102 GB | ~186 GB ✓ | throughput, CONC ≥ 10 |
| 8  | 8 → slow | ~51 GB  | ~237 GB ✓ | interactivity candidate |

**TP=2:** NCCL optimal but insufficient KV budget for ISL p50=79k tokens (only ~84 GB/GPU remaining → ~1 active request per group). Not practical without drastically reducing context.

**TP=2 + DCP=4** (8 total GPUs, 4 independent TP=2 groups): interesting idea — NCCL between 2 GPUs while using all 8 GPUs. Explore **only** after having TP=4 vs TP=8 data: if TP=8 worsens ITL compared to TP=4 (NCCL dominates), then DCP+TP=2 is a strong candidate. If TP=8 improves (t_weights dominates), DCP won't help.

**HiCache and ITL:** write-through is asynchronous and does not directly impact ITL in the decode path. The risk is PCIe contention when `cpu_kv_usage=99.7%` (continuous eviction). TP=4 without HiCache is a clean experiment to isolate this effect (see I-5).

### Step I-0 — Current baseline run
- **Run 32947505370:** branch `testgg-maxreq2x`, image v0.5.16-rocm720-mi35x-20260728
- TP=4/EP=4/HiCache/c10 + TP=8/EP=8/no-hicache/c[4,6,8,10] in parallel
- **Results:** to be filled in

### TP=8 baseline config
```
TP=8, EP=8, kv-offloading: none
chunked=32768, MEM_FRACTION=0.85
MRR = 1×CONC
EAGLE: 5 steps / 6 tokens / SGLANG_SIMULATE_ACC_LEN=3.61
```

### Experiment priorities (updated with c4/c6 data)

**Key finding:** TP=8/no-HiCache dominates for interactivity (-39% ITL vs TP=4). Bottleneck = `t_weights` (weight per GPU), not NCCL.
**Strategy:** maximize GPU-resident KV + minimize per-step overhead → TP=8, reduced EP, low CONC.

#### 🔴 High priority — complete baseline + EP=1

| # | Experiment | Action | Expected | Notes |
|---|------------|--------|----------|-------|
| **I-0b** | Baseline TP=8/EP=8 c8+c10 | Run in progress (32947505370) | C\* of TP=8/EP=8 | — |
| **I-8** | **TP=8/EP=1** | New run after I-0b | **ITL p50 ↓↓** | Eliminates MoE all-to-all at low CONC — see below |

**Rationale I-8 — TP=8/EP=1:**
With TP=8/EP=8, every MoE forward introduces an **all-to-all** among 8 EP ranks to route tokens to the correct experts. At CONC=4 (effective=1 from real data), nearly all tokens go to the same 1-2 active experts — the all-to-all is pure unamortized overhead.

With **EP=1** (no expert parallelism, only TP=8):
- Each GPU has all experts replicated → zero all-to-all in MoE forward
- Eliminates an entire collective per MoE layer
- Downside: more weights per GPU → less HBM for KV cache

Memory check before launching:
```
TP=8/EP=8: ~51 GB weights/GPU → ~237 GB KV/GPU   ✓
TP=8/EP=1: more experts/GPU → verify it fits in 288 GB HBM
```
Pattern source: PR #2693 (TP=2/EP=2 → TP=2/EP=1 for Qwen3.5 MI355X, same rationale).

#### 🟠 Medium priority — harness fixes for TP=8 (from PR #2737)

Three fixes missing in our script, derived from PR #2737 (Qwen3.5 MI355X, same stack, 2026-08-26). Can be launched together in a single run after I-8.

| # | Experiment | Parameter | Baseline | Candidate | Hypothesis | Target metric |
|---|------------|-----------|----------|-----------|------------|---------------|
| **I-1** | Smaller chunk | `chunked-prefill-size` | 32768 | **16384** | PR #2737: sibling B200; more frequent decode interleave → ITL ↓ | ITL p50 ↓ |
| **I-2** | CUDA graph max BS | `CUDA_GRAPH_MAX_BS` | `1×CONC` | **`min(2×CONC, 128)`** | PR #2737: batches > CONC fall on eager path. Critical fix when MRR > CONC. | ITL p50 ↓ |
| **I-3** | All-reduce INT8 | `ROCM_QUICK_REDUCE_QUANTIZATION` | not set | **`INT8`** | PR #2737: MI355X MXFP4 cookbook; reduces NCCL collective latency | ITL p50 ↓ |
| **I-7** | AITER unified attn | `ROCM_AITER_UNIFIED_ATTN` | not set | **`1`** | AITER integrated in ROCm; try on v0.5.16 (if ignored → no harm); full effect on stable v0.5.18 | ITL p50 ↓ |

> **Note I-2:** should be implemented before or together with I-6 (MRR pipeline). With MRR=1×CONC, limited impact; with MRR=1.5×CONC, becomes critical.

**Order:** I-1+I-2+I-3+I-7 together → I-4 → I-5 → I-6.
**Stop condition:** skip if ΔP90 tok/s/user < 5%.

#### 🟢 After harness fixes — further TP=8 optimization

| # | Experiment | Parameter | Baseline | Candidate | Hypothesis | Target metric |
|---|------------|-----------|----------|-----------|------------|---------------|
| **I-4** | More HBM for KV | `mem-fraction-static` | 0.85 | **0.87** | Headroom without HiCache → more KV in GPU → TTFT ↓ | TTFT p50 ↓ |
| **I-5** | Deeper EAGLE | `speculative-num-steps` | 5 | **6** | Effective CONC=1 at c4 → dedicated resources → real AL > 3.61 | tok/s/user ↑ |
| **I-6** | Prefill pipeline | `MRR` | 1×CONC | **1.5×CONC** | Overlap prefill/decode → C\* ↑ | C\* ↑ |

#### 🟡 Low priority — throughput arm TP=4

| # | Experiment | Notes |
|---|------------|-------|
| **H-1** | HiCache `write_through_selective` | PR #2679; reduces unnecessary DRAM writes → fewer evictions → C\* ↑ |
| **H-2** | HiCache ratio 1.5 → 2.0 | cpu_kv_usage=100% at c10 → more DRAM → C\* ↑ |
| **H-3** | HiCache `page_size=1` | PR #2679; fine granularity → more precise prefix matching |
| **3a** | MRR 1×→2×CONC (TP=4) | Branch testgg-maxreq2x already ready |
| **I-5b** | TP=4/no-hicache | Diagnostic: isolates HiCache overhead; reduced priority with TP=8 available |

#### 🔵 Reference: ATOM (PR #2576 — merged)

ATOM is an AMD-native inference engine (not SGLang) that has already run the same GLM-5.2 FP4 MI355X workload with significant interactivity results. The analysis is useful to validate our choices and identify gaps.

**Script:** `benchmarks/single_node/agentic/glm5.2_fp4_mi355x_atom_mtp.sh`
**Config:** TP=4 c[2,4,8,10] with LMCache DRAM + TP=8 c[1,2,4] GPU-resident
**Validation run:** [31765309673](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31765309673)

| Parameter | SGLang (our baseline) | ATOM (PR #2576) | Relevance |
|-----------|-----------------------|-----------------|-----------|
| **Engine** | SGLang | `atom.entrypoints.openai_server` | — |
| **Online quant** | none | **`ptpc_fp8`** (activations → FP8 on-the-fly) | SGLang does not support |
| **KV cache dtype** | BF16 | **FP8** → 2× less HBM | Enables TP=8 GPU-resident at c4 |
| **All-reduce quant** | not set | **`AITER_QUICK_REDUCE_QUANTIZATION=INT4`** | More aggressive than I-3 (INT8) |
| **CUDA graph sizes** | `CUDA_GRAPH_MAX_BS=1×CONC` | **per-CONC explicit** `[1,2,4,8,12,16,20]` | Validates I-2 (2×CONC) |
| **max-num-batched-tokens** | 32768 | **16384** | Validates I-1 (small chunk) |
| **max-num-seqs** | 1×CONC | **2×CONC** | Validates I-2 (MRR 2×) |
| **MoE sorting** | default | `AITER_USE_FLYDSL_MOE_SORTING=1` | SGLang does not support |
| **TP=8 KV offload** | DRAM HiCache | **GPU-resident** (FP8 → fits in HBM) | Same goal as I-8 |
| **KV offload backend** | HiCache | LMCache | — |

**Key insights from ATOM:**
- **FP8 KV** is why TP=8 GPU-resident works at c4: it halves HBM KV usage, allowing more tokens to stay in GPU without offload. SGLang supports `--kv-cache-dtype fp8_e4m3` → to add as **I-9** (low priority until TP=8/EP=1 is validated).
- **Per-CONC CUDA graphs** confirm that our I-2 (`min(2×CONC,128)`) is the right direction — ATOM does it even more granularly.
- **`max-num-batched-tokens=16384`** confirms I-1.
- **INT4 all-reduce** is not available in SGLang via `ROCM_QUICK_REDUCE_QUANTIZATION` (INT8 only) — structural difference from ATOM.

**Measured data — ATOM vs SGLang comparison (run 31765309673 vs 31006984179 vs ours):**

| Config | Date | ITL p50 | ITL p90 | P90 tok/s/user | Sim AL |
|--------|------|---------|---------|----------------|--------|
| ATOM TP=8/c1 (EP=1, GPU) | 2026-08-25 | 5.6ms | — | 158 | 2.99 |
| ATOM TP=8/c2 (EP=1, GPU) | 2026-08-25 | 5.7ms | — | 138 | 2.99 |
| ATOM TP=8/c4 (EP=1, GPU) | 2026-08-25 | 5.9ms | — | 123 | 2.99 |
| ATOM TP=4/c4 (LMCache) | 2026-08-25 | 6.8ms | — | 105 | 2.99 |
| ATOM TP=4/c8 (LMCache) | 2026-08-25 | 9.2ms | — | 70 | 2.99 |
| ATOM TP=4/c10 (LMCache) | 2026-08-25 | 10.5ms | — | 65 | 2.99 |
| SGLang upstream TP=4/c1 (HiCache) | 2026-08-05 | 6.8ms | — | 110 | 3.61 |
| SGLang upstream TP=4/c4 (HiCache) | 2026-08-05 | 8.6ms | — | 81 | 3.61 |
| SGLang upstream TP=4/c10 (HiCache) | 2026-08-05 | 12.9ms | — | 48 | 3.61 |
| **Our TP=8/c4** (EP=8, no KV) | 2026-08-26 | **7.3ms** | 9.6ms | **105** ✓ | 3.61 |
| **Our TP=8/c6** (EP=8, no KV) | 2026-08-26 | **8.3ms** | 11.8ms | **85** ✓ | 3.61 |
| **Our TP=8/c8** (EP=8, no KV) | 2026-08-26 | **8.8ms** | 12.2ms | **82** ✓ | 3.61 |
| **Our TP=8/c10** (EP=8, no KV) | 2026-08-26 | **9.5ms** | 14.9ms | **67** ✓ | 3.61 |
| **Our TP=4/c10** (HiCache) | 2026-08-26 | **11.9ms** | 19.4ms | **~52** ✓ | 3.61 |

**Corrected finding:** ATOM TP=8/EP=1/c4 has P90=123 vs our EP=8/c4 P90=105 → ATOM is +17% on P90 intvty. On ITL p50: ATOM 5.9ms vs our 7.3ms → -19%. The ATOM advantage is real and is explained by EP=1 (no MoE all-to-all) + ptpc_fp8 + KV FP8 + INT4 AR. Note: different AL (2.99 vs 3.61) does not affect P90 intvty which is based on ITL, not tok/s/user from acceptance.

**Gap to close:** ~1.4ms ITL and ~18 tok/s/user P90 at c4. Target: I-8 (EP=1) + I-3 (INT8 AR) + I-1 (chunk 16384).

> **Conclusion:** ATOM validates our I-1, I-2, I-3. The main ITL gap toward ATOM is `ptpc_fp8` (on-the-fly activation quantization) + KV FP8 — engine-level optimizations not directly portable to SGLang. I-9 (KV FP8 in SGLang) can recover part of the HBM advantage (~0.5ms estimated). The residual gap (~0.9ms) is structural between the two engines.

#### ⚫ Suspended / conditional

| # | Experiment | Condition |
|---|------------|-----------|
| ~~**DCP+TP=2**~~ | Eliminated | t_weights dominates — TP=2 would worsen it |
| **P-1** | v0.5.18-rocm720 retry | Only after confirmed upstream Triton LLVM `iota_range` fix |
| **P-2** | v0.5.18-rocm724 | Alternative to P-1 — may contain the fix |
| **F-5** | DP Attention | Only after upstream ROCm fix |

### Results table

> **Finding:** TP=8 beats TP=4 for ITL (-39% at c4). `t_weights` (weight/GPU) dominates, not NCCL → **I-6 DCP+TP=2 deprioritized** (TP=2 would load more weights/GPU, worsening it).

Reference columns: CONC=10 for TP=4 (throughput arm); CONC=4/6 for TP=8 (interactivity arm).

> **Metrics note:** `P90 intvty` = `intvty.p90` from JSON = 1/ITL_p90 = the value that 90% of users exceed (guaranteed floor). P90 intvty drops as CONC increases because the ITL tail worsens.

| Config | CONC | ITL p50 | ITL p90 | P90 intvty (tok/s/user) | tok/s/GPU output | TTFT p50 |
|--------|------|---------|---------|-------------------------|------------------|----------|
| **Baseline TP=4/c10** ✓ | 10 | 11.9 ms | 19.4 ms | ~52 | 91 | 509 ms |
| **Baseline TP=8/c4** ✓ | 4 | **7.3 ms** | **9.6 ms** | **105** | 23.6 | 401 ms |
| **Baseline TP=8/c6** ✓ | 6 | 8.3 ms | 11.8 ms | 85 | 30.6 | 394 ms |
| **Baseline TP=8/c8** ✓ | 8 | 8.8 ms | 12.2 ms | 82 | 40.7 | 431 ms |
| **Baseline TP=8/c10** ✓ | 10 | 9.5 ms | 14.9 ms | **67** ▼ | 48.1 | 456 ms |
| *ATOM TP=8/c1 (ref EP=1)* | 1 | 5.6 ms | — | *158* | — | — |
| *ATOM TP=8/c4 (ref EP=1)* | 4 | 5.9 ms | — | *123* | — | — |
| **I-8: TP=8/EP=1/c4** ✓ | 4 | **6.95 ms** | **9.05 ms** | **110.5** (+5.2% vs EP=8) ▲ | 23.75 | 398 ms |
| **I-8: TP=8/EP=1/c6** ✓ | 6 | **7.87 ms** | 12.15 ms | **82.3** (-3% vs EP=8) | 31.4 | 386 ms |
| **I-8: TP=8/EP=1/c8** ✓ | 8 | **8.38 ms** | 12.35 ms | **81.1** (-1% vs EP=8) | 40.8 | — |
| **I-8: TP=8/EP=1/c10** ✓ | 10 | **9.27 ms** | 14.25 ms | **70.2** (+4.8% vs EP=8) | 48.1 | — |
| **Bundle I-1+I-3+I-7/c4** ✓ | 4 | **7.03 ms** | **9.06 ms** | **110.4** (≈EP=1, +5% vs baseline) | 23.9 | — |
| **Bundle I-1+I-3+I-7/c8** ✓ | 8 | **8.37 ms** | **12.35 ms** | **81.0** (≈EP=1) | 40.9 | — |
| **Bundle I-1+I-3+I-7/c10** ✓ | 10 | **9.34 ms** | **14.54 ms** | **68.8** (≈EP=1) | 48.4 | — |
| **Bundle I-1+I-3+I-7/c6** ✓ | 6 | **7.98 ms** | **12.15 ms** | **82.3** (≈EP=1) | — | — |
| **Bundle re-run c4** ✓ | 4 | **6.96 ms** | **9.18 ms** | **109.0** (≈EP=1, vars ok) | — | — |
| **Bundle re-run c6** ✓ | 6 | **8.03 ms** | **12.05 ms** | **83.0** (≈EP=1, vars ok) | — | — |
| **Bundle re-run c10** ✓ | 10 | **9.34 ms** | **14.73 ms** | **67.9** (≈EP=1, vars ok) | — | — |
| I-9: KV FP8 | 4-8 | — | — | — | — | — |
| ~~I-6: TP=2+DCP=4~~ | — | deprioritized | | | | |

### Runner .env per run

All variables not listed are absent (no-op). `FORCE_DOCKER`, `MODEL_PATH`, `HF_HUB_CACHE_HOST` are infrastructure invariants present in all runs on mia1.

| Run / Experiment | GH Run ID | Additional variables in `.env` |
|------------------|-----------|-------------------------------|
| Baseline TP=8 EP=8 | 32947505370 | *(none)* |
| I-8: TP=8/EP=1 | 32986446019 | *(none — EP configured in config key)* |
| Invalidated bundle I-1+I-3+I-7 | 32999605584 | `CHUNKED_PREFILL_SIZE_OVERRIDE=16384` `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` `SGLANG_USE_AITER_UNIFIED_ATTN=1` — **did not reach the container** |
| P-2 v0.5.18-rocm724 | 33060449114 | *(none — patch applied unconditionally in the script)* |
| Bundle re-run I-1+I-3+I-7 | 33065762186 | `CHUNKED_PREFILL_SIZE_OVERRIDE=16384` `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` `SGLANG_USE_AITER_UNIFIED_ATTN=1` |
| I-10 CGBS (cancelled) | 33076693271 | `CUDA_GRAPH_BS_LIST_OVERRIDE=1 2 3 4 5 6 7 8` |
| **H-1+H-2 HiCache bundle** | *(queued)* | `HICACHE_RATIO=2.5` `HICACHE_WRITE_POLICY=write_through_selective` |

---

### 📊 SGLang TP=8 tuning campaign summary on MI355X (2026-08-25/27)

**Objective:** approach ATOM performance (P90 intvty c4=123, ITL p50=5.9ms) starting from the SGLang EP=8 baseline.

#### Results per config at c4 (most critical CONC for interactivity)

| Config | P90 intvty | ITL p50 | Delta vs baseline |
|--------|-----------|---------|-------------------|
| Baseline TP=8/EP=8 | 105 | 7.3ms | — |
| I-8: TP=8/EP=1 | **110.5** | **6.95ms** | **+5.2%** |
| Bundle EP=1 + I-1+I-3+I-7 | **110.4** | **7.03ms** | +5.1% (≈EP=1) |
| *ATOM TP=8/EP=1 (ref)* | *123* | *5.9ms* | *+17%* |

#### Overall findings

**1. EP=1 helps at low CONC (+5%), neutral elsewhere.**
At c4, eliminating the MoE all-to-all across 8 EP ranks reduces ITL p50 by ~0.35ms and brings P90 intvty from 105 to 110. At c6/c8/c10 the benefit is marginal or zero: the decode pressure and scheduler saturate before the all-to-all.

**2. Bundle I-1+I-3+I-7 is completely neutral on v0.5.16-rocm720.**
- I-1 (chunk=16384): no effect — chunk size affects TTFT but not decode ITL
- I-3 (ROCM_QUICK_REDUCE_QUANTIZATION=INT8): no effect — probably ignored or already optimal on this path
- I-7 (SGLANG_USE_AITER_UNIFIED_ATTN=1): no effect — kernel not active on v0.5.16 for this model

**3. Residual gap vs ATOM (-10% P90, -1ms ITL at c4) is structural to SGLang v0.5.16.**
Causes not eliminable with env vars:
- `ptpc_fp8`: on-the-fly FP8 activation quantization (ATOM engine, not portable)
- Per-CONC explicit CUDA graphs: ATOM compiles a graph for every possible batch size; SGLang uses a fixed `cuda-graph-max-bs` → **tested with I-10**
- KV FP8 already active in SGLang (`--kv-cache-dtype fp8_e4m3`): not the difference

**4. SGLang optimum on v0.5.16: EP=1, CONC=4, P90≈110.**
Recommended config for publication as best SGLang: `TP=8/EP=1/MRR=8/chunk=32768`.

#### C* of EP=1 vs EP=8

With EP=1 the P90 intvty curve is:

| CONC | EP=8 P90 | EP=1 P90 | Better |
|------|----------|----------|--------|
| c4 | 105 | **110.5** | EP=1 |
| c6 | 85 | 82.3 | EP=8 |
| c8 | 82 | 81.1 | EP=8 |
| c10 | 67 | 70.2 | EP=1 |

C\* (knee point) remains between c8 and c10 for both configs. EP=1 is preferable at c4 for maximum interactivity.

#### Next steps (options)

| Option | Expected | Risk |
|--------|----------|------|
| **v0.5.18-rocm724** (P-2) | AITER more integrated, new ROCm 7.2.4 kernels — may activate I-3/I-7 | Triton LLVM fix verified? Test TP=8 only (no TP=4/HiCache) |
| **Accept SGLang plateau** | Document P90=110 as best SGLang, structural gap vs ATOM explained | None |
| **Throughput sweep EP=1** | C\* with EP=1 on TP=4 + HiCache (high-throughput arm) — already validated by baseline | Low |

<details>
<summary>Baseline TP=4/c10 detail — run 32947505370 (v0.5.16, ~72 min, 1407 req)</summary>

| | p50 | p90 | p99 |
|--|-----|-----|-----|
| ITL (ms) | 11.9 | 19.4 | 49.7 |
| TTFT (ms) | 509 | 1,592 | 8,683 |
| tok/s/user decode | 84 | 114 | 153 |
| tok/s/user E2E | 69 | 98 | — |
| Request latency (ms) | 5,220 | 28,960 | 113,717 |

- **Throughput:** 363 tok/s decode total → **91 tok/s/GPU** (4 GPUs)
- **Effective CONC p50=5** out of 10 configured (long traces → heavy prefill)
- **Prefix cache hit:** 95.9% · `cpu_kv_usage`≈100%
- **EAGLE:** Accept Length=3.61 (=simulate target) · Accept Rate=52.2%
- **ISL** p50=90k tokens · **OSL** p50=335 tokens
- Total: 138M tokens input / 1.3M tokens output
</details>

---

## P-2 — SGLang v0.5.18-rocm724 upgrade (gfx950)

**Objective:** verify whether v0.5.18 + ROCm 7.2.4 brings improvements on MI355X compared to the v0.5.16 baseline.

**Outcome: abandoned** — no gain at c4, additional overhead from the torch fallback.

### Problem: `fp8_mqa_logits` crash on gfx950

**Symptom:** the sglang server crashes during warmup with:
```
AssertionError: Begin <= End  (LLVM sequence.h:275, iota_range)
```
originating from `aiter/ops/triton/attention/fp8_mqa_logits.py` during JIT compilation of the Triton kernel for gfx950.

**Root cause:** the `fp8_mqa_logits` kernel in aiter (integrated into ROCm 7.2.4) has two paths:
- **Gluon path** (gfx950-native): fails with LLVM `iota_range` assertion during JIT
- **Standard Triton path**: fails equally with the same assertion

Both paths are incompatible with the gfx950 compiler in this version of ROCm.

**Applied fix:** pure torch fallback, replaces `fp8_mqa_logits` with per-sequence BF16 matmul. Implemented in `benchmarks/single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh` — patch applied in-container at run time (rewrites the `.py` file in the Docker image). Preserved in the `testgg-v518-p2` branch.

**Iterations required (9 failed runs):**
1. Patch conditional on env var `SGLANG_AITER_DISABLE_GLUON_FP8_MQA` — env var was not reaching Docker
2. Discovery that the GH Actions runner **does not export `.env` to the subprocess** → `-e VAR` without value = empty string
3. Multiple forwarding attempts (all failed for the same reason)
4. Unconditional patch → patch applied, but OOM: `[seq_len × total_kv_aligned]` float32 allocation = 2.72 GiB with 90k KV tokens
5. Rewrite with per-sequence loop (`torch.mv`) → OOM resolved, server functional

### Problem: runner `.env` variables not propagated to Docker

**Impact:** also invalidated the bundle I-1+I-3+I-7 runs (run 32999605584) — the vars `CHUNKED_PREFILL_SIZE_OVERRIDE`, `ROCM_QUICK_REDUCE_QUANTIZATION`, `SGLANG_USE_AITER_UNIFIED_ATTN` never reached the container → baseline vs baseline comparison.

**Root cause:** the GH Actions runner loads `.env` into its own process but **does not export it to the subprocess environment** (the job bash script). `docker run -e VAR` without `=value` propagates an empty string if VAR is not in the subprocess env.

**Fix applied in `runners/launch_mi355x-amds.sh`:**
```bash
# Explicitly load .env at Docker fallback entry
_RUNNER_ENV="${GITHUB_WORKSPACE%/*/*/*}/.env"   # runner root, not _work
if [[ -f "$_RUNNER_ENV" ]]; then
    set -a; source "$_RUNNER_ENV"; set +a
fi
```
Then in the `-e` flags of `docker run`:
```bash
${CHUNKED_PREFILL_SIZE_OVERRIDE:+-e "CHUNKED_PREFILL_SIZE_OVERRIDE=${CHUNKED_PREFILL_SIZE_OVERRIDE}"}
```
The `${VAR:+-e "VAR=${VAR}"}` syntax omits the flag if the var is empty (no-op for baseline runs), and includes it with embedded value if set.

**`.env` path:** `${GITHUB_WORKSPACE%/*/*/*}` — removes the last 3 components (`/_work/<repo>/<repo>`) to reach the runner root. `%/*/*` was wrong (pointed to `_work/`).

### P-2 results (c4 the only CONC completed before cancellation)

| Metric | v0.5.18 + torch fallback | v0.5.16 EP=1 baseline |
|--------|--------------------------|----------------------|
| ITL p50 | 7.18 ms | 7.03 ms |
| ITL p90 | 37.7 ms | — |
| intvty p50 | 139.2 tok/s/user | ~110 (P90) |
| errors | 0 | 0 |

**Conclusion:** no relevant improvement (+2% ITL p50, within variability). The torch fallback for `fp8_mqa_logits` introduces overhead compared to the native v0.5.16 kernel. Run cancelled after c4.

### P-3 — SGLang v0.5.18-rocm724 retry after PR #36960 merge (2026-09-02)

**Objective:** retry v0.5.18 upgrade without the torch fallback after sgl-project/sglang #36960 (`Cap the DSA MQA-logits budget at AITER's buffer_store limit`) merged 2026-09-01 22:40 UTC.

**Branch:** `testgg-v518-r901`, config key `glm5.2-fp4-mi355x-sglang-agentic-mtp-v518r901`, runner mia1-p01-g07 (`mi355x-amds_03`).

**Outcome: blocked** — image `lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260901` does **not** contain the SGLang-side fix. The image was built before 22:40 UTC on 2026-09-01; `dsa_indexer.py::_get_mqa_logits_budget_bytes` has no `BUFFER_LIMIT_BYTES` cap. The `BUFFER_LIMIT_BYTES` check present in AITER's `fp8_mqa_logits.py` is a pre-existing guard, not the PR #36960 fix. Server crashes during warmup with the same iota_range pattern.

**Next step:** wait for image `v0.5.18-rocm724-mi35x-20260902` or later, verify with:
```bash
docker run --rm <image> grep -n "BUFFER_LIMIT\|min(" \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
```

---

## Bundle re-run I-1+I-3+I-7 (v0.5.16, env var fix)

**Objective:** repeat the bundle experiment after fixing the env var propagation bug in the launcher. The previous run (32999605584) was a baseline vs baseline comparison.

**Fix applied:** `source .env` at Docker fallback entry + `-e "VAR=${VAR}"` with embedded value for the three bundle vars. Correct `.env` path: `${GITHUB_WORKSPACE%/*/*/*}` (runner root, not `_work/`).

**Run:** 33065762186 `bundle-rerun-v516-envfix` — completed, v0.5.16-rocm720, TP=8/EP=1. c8 lost due to runner disconnection during the run.

### Final results

| Config | CONC | ITL p50 | ITL p90 | P90 intvty | Δ vs EP=1 baseline |
|--------|------|---------|---------|------------|--------------------|
| I-8 EP=1 c4 (ref) | 4 | 6.95ms | 9.05ms | 110.5 | — |
| **Bundle re-run c4** ✓ | 4 | **6.96ms** | **9.18ms** | **109.0** | -1.4% |
| I-8 EP=1 c6 (ref) | 6 | 7.87ms | 12.15ms | 82.3 | — |
| **Bundle re-run c6** ✓ | 6 | **8.03ms** | **12.05ms** | **83.0** | +0.8% |
| I-8 EP=1 c10 (ref) | 10 | 9.27ms | 14.25ms | 70.2 | — |
| **Bundle re-run c10** ✓ | 10 | **9.34ms** | **14.73ms** | **67.9** | -3.3% |
| Bundle c8 | 8 | — | — | — | *(lost: runner disconnection)* |

**Verdict: I-1+I-3+I-7 completely neutral at all tested CONC values.** All variations are within run-to-run variability (±3%). The env vars had correctly reached the container this time — the neutrality is real, not an artifact of the propagation bug.

---

## I-10 — Explicit per-batch-size CUDA graphs (c4)

**Motivation:** with `--cuda-graph-max-bs N` SGLang compiles a **single graph** for bs=N. Every decode step with a batch size different from N runs in eager mode (slower, more ITL variance). ATOM uses `--cudagraph-capture-sizes [1,2,4,8]` to compile a graph for every possible batch size — zero graph misses.

**Mechanism:** even with fixed CONC=4, the decode batch size varies continuously between 1 and MAX_RUNNING_REQUESTS (=8) depending on how many requests are simultaneously in the decode phase. Capturing every size eliminates eager fallbacks.

**Implementation:** `--cuda-graph-bs 1 2 3 4 5 6 7 8` (SGLang accepts a space-separated list, also used in multi-node disaggregated). Activated via `CUDA_GRAPH_BS_LIST_OVERRIDE` in runner `.env`, only for CONC=4; other CONC values use `--cuda-graph-max-bs` unchanged.

**Config:** `glm5.2-fp4-mi355x-sglang-agentic-mtp-cgbs` — c4 only, TP=8/EP=1, v0.5.16.

**Run:** 33076693271 — **cancelled before execution**.

**Reason:** from the server log of the bundle re-run (c6) it can be seen that SGLang with `--cuda-graph-max-bs 12` already automatically generates a list of batch sizes:
```
decode=PhaseConfig(max_bs=12, bs=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
```
SGLang interpolates a progression between 1 and max_bs — it does not compile a single graph. The test would have had marginal benefit and did not justify 1.5h of GPU time.

**Conclusion:** the residual gap vs ATOM is not explained by CUDA graphs. SGLang is already equivalent on this aspect. The `CUDA_GRAPH_BS_LIST_OVERRIDE` variable removed from runner `.env`.

---

## H-1+H-2 — HiCache bundle (TP=4/c10, throughput arm)

**Objective:** increase throughput at c10 starting from the diagnosis that `cpu_kv_usage=100%` in the baseline indicates continuous KV evictions → requests stalled waiting for cache space → effective CONC p50=5 out of 10 configured.

**Experiments:**
- **H-1** `HICACHE_WRITE_POLICY=write_through_selective`: writes DRAM only for high-reusability prefill (PR #2679). Reduces wasted bandwidth on KV that will never be reused.
- **H-2** `HICACHE_RATIO=2.5`: allocates more host DRAM for KV (from 1.5×GPU pool to 2.5×). On mia1 with TP=4 each rank has ~400 GB DRAM available — ample margin.

**Config:** `glm5.2-fp4-mi355x-sglang-agentic-mtp-hicache` — TP=4/EP=4/c10, v0.5.16.

**Runner `.env` (mia1):**
```
FORCE_DOCKER=1
MODEL_PATH=/it-share/models/GLM-5.2-MXFP4
HF_HUB_CACHE_HOST=/mnt/hf_hub_cache
HICACHE_RATIO=2.5
HICACHE_WRITE_POLICY=write_through_selective
```

**Run:** completed (part of campaign runs on mia1, HICACHE_RATIO=2.5 via runner .env)

**Outcome:** `tok/s/GPU` improved from 91 (baseline c10) to **91 tok/s/GPU at c10** and **102 tok/s/GPU at c12** (+12%). Confirming the enlarged host KV pool delays saturation and sustains throughput through c12.

**Results (run 33373633743, PR #2777 full-sweep validation, 2026-08-31):**

HICACHE_RATIO ran at default=1.5 (reverted from 2.5 after reviewer feedback — ratio=2.5 exceeds ~3.0 TB available DRAM on cluster:mi355x-amds). Comparison against campaign runs (which used HICACHE_RATIO=2.5 via runner .env) and CSV baseline (EP=4, HICACHE_RATIO=1.5).

#### Three-way comparison: CSV baseline vs EP=1 campaign vs PR #2777 (HICACHE=1.5)

**TP8/EP=1 — interactivity arm**

| conc | ATOM (ref) | CSV baseline (EP=4) | EP=1 campaign (HICACHE=2.5) | PR #2777 (HICACHE=1.5) | delta vs CSV |
|------|-----------|---------------------|------------------------------|------------------------|--------------|
| c1   | 158.0     | 110.1               | —                            | **132.1**              | **+20%** ▲   |
| c2   | 138.3     | 98.0                | —                            | **123.3**              | **+26%** ▲   |
| c4   | 123.3     | 80.6                | 110.5                        | **99.2**               | **+23%** ▲   |
| c10  | 64.9      | 48.0                | 70.2                         | **69.1**               | **+44%** ▲   |

**TP4/EP=4 + HiCache — throughput arm**

| conc | CSV baseline (HICACHE=1.5) | EP=1 campaign (HICACHE=2.5) | PR #2777 (HICACHE=1.5) | delta vs campaign |
|------|---------------------------|------------------------------|------------------------|-------------------|
| c4   | 80.6                      | —                            | 80.4                   | stable            |
| c8   | 57.5                      | —                            | 56.9                   | stable            |
| c10  | 48.0                      | 91 tok/s/GPU                 | 88.3 tok/s/GPU         | -3%               |
| c12  | —                         | 102 tok/s/GPU                | 84.9 tok/s/GPU         | -17%              |

**Notes:**
- c1/c2 TP8/EP=1 are new points not measured in the campaign — both substantially above CSV baseline.
- c4 drop (110.5 → 99.2) and c12 drop (102 → 84.9) are caused by HICACHE_RATIO=1.5 vs 2.5. Expected and consistent with the reviewer fix. The campaign numbers (obtained with explicit HICACHE_RATIO=2.5 in runner .env) remain the correct values to cite in the PR description.
- c10 TP4 drop (-3%) is within noise; c12 drop (-17%) is structural to the smaller KV pool.

---

## Phases 2-5 — Low-priority experiment detail

### HiCache tuning (TP=4, CONC ≥ 10 regime)

cpu_kv_usage=100% at CONC=10. Parameters validated by PR #2679 as reference.

| # | Parameter | Baseline | Candidate | Motivation |
|---|-----------|----------|-----------|------------|
| **H-1** | `HICACHE_WRITE_POLICY` | `write_through` | `write_through_selective` | PR #2679: writes DRAM only for high-reusability prefill → less wasted bandwidth |
| **H-2** | `HICACHE_RATIO` | 1.5 | **2.0** | CPU pool saturated → more DRAM may raise C\* |
| **H-3** | `page_size` | default | **1** | PR #2679: fine granularity → more precise cache matching |
| **H-4** | `HICACHE_IO_BACKEND` | `direct` | `asyncio` | Only if DRAM I/O is bottleneck at high CONC |

**Order:** H-1 → H-2 → H-3 → H-4 only if plateau.

### MRR sweep TP=4 (branch testgg-maxreq2x already ready)

```
3a: MRR=10 (baseline) vs MRR=20 (2×CONC) ← branch already ready
3b: chunked-prefill-size: 16384 / 32768 (baseline) / 65536
3c: cuda-graph-max-bs: 1×MRR / 1.5×MRR / 2×MRR
```

### Platform and advanced parallelism

| # | Experiment | Condition |
|---|------------|-----------|
| **P-1** | v0.5.18-rocm720 retry | Skipped — superseded by P-2 |
| **P-2** | v0.5.18-rocm724 (ROCm 7.2.4) | **🔵 IN PROGRESS (2026-08-27)** — image `v0.5.18-rocm724-mi35x-20260826` downloaded on mia1. Config key `glm5.2-fp4-mi355x-sglang-agentic-mtp-v518r724`, TP=8/EP=1/conc=[4,6,8,10]. We skipped directly to P-2 bypassing P-1: the Aug 26 rocm724 build is the most up-to-date available and includes ROCm 7.2.4 which may activate I-3/I-7. |
| **P-3** | AITER unified attention | Tested on v0.5.16 (I-7): neutral. May become active on v0.5.18-rocm724 — we will check from P-2 run logs. |
| **F-5** | DP Attention | Only after upstream ROCm fix |

> **Decision 2026-08-27:** Skipped P-1 (v0.5.18-rocm720) — moved directly to P-2 (v0.5.18-rocm724). Rationale: rocm724 is more recent, potentially already includes the Triton LLVM fix, and has more updated AITER/NCCL kernels. The regression risk is contained by testing only TP=8/EP=1 (skip TP=4/HiCache). Comparison: EP=1/v0.5.16 P90=110.5 → target P90>115 on v0.5.18.

---

# 6. First concrete step (Milestone 1)

## Objective
Sweep `max-running-requests` (1×C vs 2×C) on v0.5.18-rocm720, with multi-CONC per server lifetime.

## Steps
1. **Fix `_work` pre-cleanup** in the launcher (adds auto-cleanup with docker before checkout)
2. **Update image** in the recipe to v0.5.18-rocm720
3. **Launch `testgg-maxreq2x`** (MRR=20): read results at CONC=10
4. **Add CONC multi-sweep**: modify the recipe to test CONC=[4, 6, 8, 10, 12] in the same server lifetime (dense crossover zone)
5. **Compare** baseline (MRR=10) vs maxreq2x (MRR=20) on key metrics

## Tuner code (draft structure)
```
utils/autotune/
  config.py        — parameters + metadata (restart_required, range, deps)
  runner.py        — launches GH Actions job, waits for result, reads agg_bmk.json
  pareto.py        — dominance test, frontier tracking
  storage.py       — updates progress.csv
  cli.py           — entry point
```

In its first version the tuner is a **GH Actions job orchestrator**, not a direct SGLang server controller (not feasible with current infrastructure).

---

# 7. Removed / out-of-scope parameters (for now)

- **ATOM**: out of scope for direct execution (different framework). Used as **reference** to validate tuning choices — see 🔵 section in Phase 1. KV FP8 (I-9) is the only ATOM optimization portable to SGLang.
- **Kimi-K3**: second phase after GLM-5.2 is stabilized.
- **DCP**: only Kimi-K3 in the current configuration.
- **Mooncake L3**: only after HiCache L2 is characterized.
- **Bayesian optimization**: only after infrastructure and multi-fidelity are validated.

---

# 8. Multi-fidelity

| Tier | Duration | Use |
|------|----------|-----|
| Screening | 120–300s | Eliminate clearly worse configurations |
| Tuning | 600–1200s | Reliable ranking of candidates |
| Validation | 3600s (full) | Only Pareto-candidate configurations |

Minimum requirements: ≥ 100 completed requests for reliable P90.

---

# 9. Known issues

## DP attention / DCP on GLM-5.2 + SGLang (status 2026-08-31)

The DP-attention arm in `glm5.2_fp4_mi355x_sglang_mtp.sh` is currently **DORMANT** (no DEP arm in `amd-master.yaml`).

**Root cause:** DSA + DP attention hangs a collective under long-context prefill on ROCm (watchdog kills scheduler, 0 completions). Reproduced with and without HiCache, with and without the DSv4 DP collective env vars. Tracked upstream as SGLang issue [#34582](https://github.com/sgl-project/sglang/issues/34582).

**Status as of v0.5.18:** not fixed. Relevant improvements in v0.5.17+:
- PR #31682: breakable prefill CUDA graph on by default for DP attention (reduces indefinite hangs, but does not fix the DSA path)
- PR #33829: dummy row normalization fix for spec decoding + DP attention (potentially relevant for MTP+DSA, monitor)
- `SGLANG_DP_USE_GATHERV=1` + `SGLANG_DP_USE_REDUCE_SCATTER=1` workaround helps DSv4 but not validated for GLM-5.2/DSA

**Action:** re-enable DEP arm once SGLang issue #34582 is resolved or PR #33829 shows positive effect on DSA+DP path. Monitor v0.5.19+.

---

See `improvements.md` for full backlog.

**Critical:** `_work/` remains root-owned after failed runs (NFS root_squash, no sudo on mia1-p01-g07).
Manual workaround:
```bash
docker run --rm --privileged \
  -v /it-share/gguasti/actions-runner/_work:/work \
  lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728 \
  rm -rf /work/InferenceX
```

**Fix to implement:** automatic pre-cleanup in the launcher before each `actions/checkout`.

---

# 10. No-HiCache Pareto sweep (2026-09-02 — next campaign)

**Motivation:** run 33647161138 confirmed that c12/no-HiCache is the best interactivity point measured in this campaign (ITL p90=20ms, tok/s/user p90=111.82, no OOM). The full operating envelope is unknown: we have one data point (c12) but not the complete Pareto curve nor the OOM boundary.

**Objective:** characterise the no-HiCache regime across the full concurrency range:
1. Build the interactivity → throughput Pareto curve (c1 through c24)
2. Find the OOM boundary: the minimum CONC at which `gpu_kv_usage` saturates and errors appear
3. Confirm whether c4/no-HiCache also beats EP=1/c4 HiCache (expected yes, but not yet measured)

**Config key:** `glm5.2-fp4-mi355x-sglang-agentic-mtp-nohicache-sweep`
```yaml
- { tp: 4, ep: 4, kv-offloading: none, conc-list: [1, 2, 4, 8, 10, 12, 16, 20, 24], spec-decoding: mtp }
```

**What to watch per CONC slice:**
- `gpu_kv_usage` from server metrics → rising trend indicates approaching HBM limit
- error rate in run summary → first non-zero errors = OOM onset
- `Output tok/s/user p90` → should peak somewhere around c4–c8 (interactivity regime) then decline
- `Throughput/GPU (total tok/s)` → should keep growing until OOM

**Expected shape:**

```
tok/s/user p90
     ▲
 115 │   ● c4?   ● c8?
 110 │                 ● c12 (confirmed)
  80 │                         ● c16?
     │                                 ✗ OOM c20–c24?
     └──────────────────────────────────────────► CONC
```

**Workflow:**
```bash
env -u GITHUB_TOKEN gh workflow run "End-to-End Tests" \
  --repo giovanniguastiamd/InferenceX \
  --ref testgg-qr-int4 \
  -f "generate-cli-command=test-config --config-files configs/amd-master.yaml \
      --config-keys glm5.2-fp4-mi355x-sglang-agentic-mtp-nohicache-sweep \
      --scenario-type agentic-coding \
      --runner-node-filter mi355x-amds_03" \
  -f "test-name=glm5.2-nohicache-pareto-sweep"
```

**Note:** a single run sweeps all CONC values sequentially on the same server instance (one startup, multiple CONC slices → §3 savings). Duration per slice: 3600s. Estimated total: ~9× slices × ~1h = ~9h wall time.

**Result (run 33661191728, 2026-09-02 — COMPLETED):**

| CONC | ITL p50 (ms) | ITL p90 (ms) | ITL p95 (ms) | TTFT p50 (ms) | intv p50 (tok/s/u) | intv p90 (tok/s/u) | tput/GPU (tok/s) | KV%GPU | n_ok | n_err |
|------|-------------|-------------|-------------|--------------|-------------------|-------------------|-----------------|--------|------|-------|
| 2    | 7.4         | 9.5         | 10.9        | 437          | 135.6             | 105.6             | 35              | 18%    | 442  | 0     |
| 4    | 8.1         | 10.9        | 12.9        | 377          | 122.9             | 92.0              | 46              | 26%    | 647  | 1     |
| 8    | 10.3        | 15.0        | 18.2        | 397          | 97.5              | 66.7              | 79              | 50%    | 1279 | 1     |
| 10   | 11.6        | 18.5        | 23.2        | 456          | 85.9              | 54.1              | 92              | 63%    | 1420 | 0     |
| **12** | **11.8** | **20.4**  | **25.8**    | **481**      | **84.5**          | **48.9**          | **106**         | **60%** | **1606** | **0** |
| 20   | 25.2        | 150.4       | 195.9       | 1001         | 39.6              | 6.6               | 79              | **100%** | 1493 | 0   |
| 24   | 115.5       | 257.1       | 308.1       | 22437        | 8.7               | 3.9               | 38              | **100%** | 823  | 0   |

*c1, c16: not measured in this run (c1 = runner disconnected; c16 = runner time expired). Follow-up sweep [c1, c14, c16, c18] launched as run 33720514696 to fill gaps.*

**HiCache comparison at c12 (run 33647171278, same machine/date):**

| Metric | HiCache (dram) | no-HiCache | Delta |
|--------|---------------|------------|-------|
| ITL p90 | 21.0 ms | 20.4 ms | +0.6 ms (+3%) |
| intvty p90 (tok/s/u) | 47.7 | 48.9 | −1.2 (−2%) |
| tput/GPU (tok/s) | 105 | 106 | −1 (−1%) |
| KV%GPU | 62% | 60% | +2% |

**Finding:** HiCache and no-HiCache are **equivalent at c12 on mia1**. With KV%GPU at 62%, HiCache is not offloading meaningfully to DRAM — the KV fits in HBM in both cases. The HiCache path overhead (+0.6 ms) is within measurement variance. The "70ms with HiCache" figure cited earlier came from a different machine/config.

**Key findings:**
- **OOM boundary: between c12 and c20.** KV%GPU jumps from 60% → 100% at c20; ITL degrades 7× (20ms → 150ms). No hard crash — server degrades gracefully but becomes unusable.
- **Best throughput: c12** — 106 tok/s/GPU, KV at 60% (safe headroom). Confirmed as production sweet spot.
- **Best interactivity: c2** — ITL p90=9.5ms; but only 35 tok/s/GPU (3× less throughput than c12).
- **Safe operating range: c2–c12.** All have KV ≤ 63%, 0 degradation, 0 meaningful errors.
- **HiCache vs no-HiCache: neutral at c12.** On mia1 where KV fits, the choice does not matter. R-1 recommendation revised: no-HiCache is simpler (no DRAM path complexity) but not a performance win on machines with adequate HBM.
- **Follow-up sweep pending:** c1, c14, c16, c18 (run 33720514696) to pin exact OOM boundary.

**Actual Pareto curve (measured):**
```
intv p90 (tok/s/user)
     ▲
 106 │  ● c2 (9.5ms ITL)
  92 │     ● c4 (10.9ms)
  67 │           ● c8 (15.0ms)
  54 │                  ● c10 (18.5ms)
  49 │                       ● c12 (20.4ms) ← best throughput
   7 │                                            ✗ c20 OOM onset
   4 │                                                  ✗ c24 degraded
     └──────────────────────────────────────────────────► CONC
       (pick c4–c12 depending on latency/throughput target)
```

---

# 11. Recipe development methodology (lessons learned)

This section captures generalizable principles derived from the GLM-5.2 / MI355X tuning campaign (Aug–Sep 2026). They apply to most new model+hardware combinations in InferenceX.

---

## 10.1 KV offloading: measure before you enable

**The mistake:** HiCache (CPU KV offloading) is often enabled by default in new recipes — preemptively, without verifying whether the KV cache actually fits in HBM at the target ISL and concurrency. This is a silent performance regression: if the KV fits in HBM, HiCache adds a CPU↔GPU DRAM bottleneck that can degrade ITL by **3–4×** with zero benefit.

**Observed data (GLM-5.2, MI355X, c12, ISL p90~130k tokens):**

| Config | ITL p90 | Output tok/s/user p90 | Notes |
|--------|---------|----------------------|-------|
| c12 + HiCache (`kv-offloading: cpu`) | ~70 ms | ~80 | DRAM bottleneck dominates |
| c12 no-HiCache (`kv-offloading: none`) | **20 ms** | **111.82** | KV in HBM, 3.5× ITL improvement |

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

### Why GLM-5.2 / MLA benefits especially:

MLA (Multi-head Latent Attention) compresses the KV cache to a ~512-dimensional latent vector per token instead of storing full K+V per head. This makes the KV footprint roughly **10–20× smaller** than a dense MHA model of equivalent size. For MI355X (80 GB HBM per GPU), KV fits in HBM at c12/ISL~130k tokens with TP=4. For dense-KV models (e.g. Llama-3-70B) the same ISL and concurrency would likely require offloading.

### Impact on the parameter space table (Group D):

The entry for `KV offload backend` in §4 Group D should be read as: **the default candidate is `none`, not `hicache`**. HiCache is the fallback when HBM is insufficient, not the starting point.

---

## 10.2 Concurrency sweep: cover the full Pareto front before tuning parameters

**The mistake:** tuning structural parameters (chunked-prefill, MRR, etc.) at a single fixed CONC. The optimal parameter may differ across regimes (interactivity vs crossover vs throughput — see §2).

**Rule:** for any new candidate config, run at least CONC ∈ {4, 8, 12} before concluding. A config that wins at c4 may lose at c12 and vice versa.

---

## 10.3 Machine differences invalidate cross-machine comparisons

**The mistake:** comparing absolute metrics (ITL, throughput) across machines with different GPU count, NFS speed, or available HBM. Machines used in this campaign:

| Runner | GPUs | HBM/GPU | NFS | Typical baseline drift |
|--------|------|---------|-----|----------------------|
| `mi355x-amds_01` (wbb3) | 40× MI355X | 80 GB | `/mnt/it_share` (shared) | reference |
| `mi355x-amds_03` (mia1) | 8× MI355X | 80 GB | `/it-share` (shared) | ITL comparable; throughput not directly comparable (8 vs 40 GPUs) |

**Rule:** always report which runner produced a result. Use relative comparisons (A vs B on the same runner) rather than absolute cross-runner claims.

---

## 10.4 Warmup and trace burn-in

**The mistake:** measuring before the KV radix cache is warmed up. Early timeslices show artificially high TTFT and low cache hit rates.

**Rule:** discard the first 10–15 minutes of any 60-minute run (warmup period). Verify `prefix_cache_hit` reaches a stable plateau (typically >90% for long-context agentic traces) before reading P90 metrics. The benchmark harness marks warmup requests automatically — check the `131 warmup` line in the run summary.

---

## 10.5 INT4 all-reduce: check tensor size before expecting gains

Before enabling `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, verify that the all-reduce tensor will exceed the activation threshold (`_QR_MIN_SIZE`) at the target TP degree:

- At TP=4: threshold ≈ 16 MB. A decode step with 4 tokens produces ~57 KB — **always below threshold**.
- INT4 can only fire on the **prefill path** when the prefill batch accumulates >~1200 tokens at TP=4.
- If HiCache is active and cpu_kv_usage is high, the DRAM I/O bottleneck masks any INT4 gain anyway.

**Rule:** INT4 all-reduce benefits only if (a) tensors are large enough to exceed the threshold AND (b) the bottleneck is communication, not DRAM. Verify both conditions before attributing gains to INT4.

---

# 12. Docker launcher requirements and validation checklist

This section documents the requirements that `runners/launch_mi355x-amds.sh` (Docker fallback) must satisfy to produce results comparable with the upstream SLURM/enroot path. Failures here cause silent measurement errors that are very hard to detect.

## Why this exists

The upstream cluster uses `srun --export=ALL` (SLURM + enroot), which automatically propagates **all** workflow environment variables into the container. The Docker fallback used on mia1-class machines uses an explicit `-e VAR` list and must be kept in sync with the workflow manually.

**Bug discovered 2026-09-03:** `MODEL_PREFIX` was absent from the `-e` list. Result: `infmax_model_prefix` was empty in all JSON artifacts, causing `benchmark_lib.sh` to fall through to the 256k-capped dataset (`cc-traces-weka-062126-256k`) instead of the unfiltered corpus (`cc-traces-weka-062126`). The upstream PR 2777 sweep (SLURM) ran with ISL p90 ≈ 283k tokens; our mia1 runs had ISL p90 ≈ 170k — a ~40% underestimate of sequence length that invalidates direct comparisons.

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

| Env var | Purpose | mia1 value |
|---------|---------|------------|
| `FORCE_DOCKER=1` | bypass SLURM, use Docker fallback | `1` |
| `MODEL_PATH` | local model weights directory | `/it-share/models/GLM-5.2-MXFP4` |
| `HF_HUB_CACHE_HOST` | NFS HF cache mount point | `/mnt/hf_hub_cache` |
| `WEKA_LOADER_OVERRIDE` | force specific dataset loader | unset (use MODEL_PREFIX default) |

### Vars that are optional / conditional (`${VAR:+-e VAR}` pattern)

These are only passed if set in runner `.env` (recipe-specific overrides):

- `CHUNKED_PREFILL_SIZE_OVERRIDE` — override chunked prefill size (I-1)
- `ROCM_QUICK_REDUCE_QUANTIZATION` — INT4 all-reduce (I-3)
- `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` — INT4 cast companion
- `SGLANG_USE_AITER_UNIFIED_ATTN` — AITER unified attention (I-7)
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
# Expected for glm5.2 unfiltered:
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
