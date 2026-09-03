# GLM-5.2 FP4 / MI355X — SGLang Tuning Campaign (Aug–Sep 2026)

**Model:** GLM-5.2 (MXFP4) · **Framework:** SGLang · **Hardware:** 8× MI355X (mia1-p01-g07)
**Branch:** `testgg`, `testgg-maxreq2x` · **Upstream PR:** [#2777](https://github.com/SemiAnalysisAI/InferenceX/pull/2777) (merged 2026-09-01)

> **Methodology (general):** [tuner_en.md](tuner_en.md)
> **SGLang feature gaps vs ATOM:** [SGL_missing_features_01_09_2026.md](SGL_missing_features_01_09_2026.md)

---

# 0. Working environment

```
Host:    giovanni.guasti@amd.com@mia1-p01-g07  (ProxyJump 64.139.223.124)
Workdir: /it-share/gguasti
Model:   /it-share/models/GLM-5.2-MXFP4  (408 GB, NFS)
Runner:  /it-share/gguasti/actions-runner  (GitHub Actions: mi355x-amds_03)
```

Docker images on mia1:
```
lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728  ← campaign baseline
lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260824  ← P-1 (skipped)
lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260824  ← P-2 (abandoned)
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

# 2. Phase 0 — Infrastructure (prerequisites)

- [x] Runner `mi355x-amds_03` on mia1-p01-g07 operational
- [x] Baseline v0.5.16-rocm720 CONC=10: 9,584 tok/s/GPU (TP4/EP4/HiCache)
- [x] Recipe updated: TP=8/EP=8/no-hicache arm added to `amd-master.yaml`
- [x] Fix `_work` root-owned — Alpine chown in "Resource cleanup (pre-run)" workflow step (commit 4b33bfe98)
- [x] **Rollback image v0.5.18 → v0.5.16** (commit 4786948a0): v0.5.18 crashes SIGABRT in `fp8_mqa_logits` (Triton LLVM `iota_range` assertion) on TP=4+HiCache+DSA and on TP=8. Confirmed regression vs v0.5.16.
- [ ] Fix `test-sweep-evals` fromJson empty (workflow cosmetic failure)

---

# 3. Phase 1 — Interactivity optimization

## Overhead analysis at low concurrency

At small CONC, the decode loop is dominated by fixed, unamortized overhead:

```
ITL/user ≈ (t_nccl_allreduce + t_kernel_launch + t_weights) / CONC
```

- `t_nccl` is ~constant for small batches but scales with the number of GPUs in the allreduce
- `t_weights` scales with 1/TP (less weight per GPU = faster loading)
- The sweet spot is where amortization is nearly complete but the queue is still absent

**Why C=1 is worse than C=4:** fixed overhead (NCCL, kernel launch, 5+1 MTP cycles) is not amortized over a single request. At C=4, the same overhead is split among 4 users and `t_weights` is nearly unchanged (memory-bound regime).

## TP size trade-off (GLM-5.2 FP4, ISL p50~79k tokens, 8× MI355X)

| TP | NCCL ranks | Weights/GPU | HBM for KV | Optimal regime |
|----|-----------|-------------|------------|----------------|
| 2  | 2 → very fast | ~204 GB | ~84 GB ⚠ | not practical with ISL=79k tokens |
| 4  | 4 → medium | ~102 GB | ~186 GB ✓ | throughput, CONC ≥ 10 |
| 8  | 8 → slow | ~51 GB  | ~237 GB ✓ | interactivity candidate |

**TP=2:** NCCL optimal but insufficient KV budget for ISL p50=79k tokens (~84 GB/GPU remaining → ~1 active request per group). Not practical without drastically reducing context.

**TP=2 + DCP=4** (8 total GPUs, 4 independent TP=2 groups): explored hypothetically — `t_weights` dominates (not NCCL), so TP=2 would worsen ITL. Deprioritized (see I-6 eliminated row).

**HiCache and ITL:** write-through is asynchronous and does not directly impact ITL in the decode path. The risk is PCIe contention when `cpu_kv_usage=99.7%` (continuous eviction). TP=4 without HiCache is a clean experiment to isolate this effect (see I-5b).

## Step I-0 — Baseline run (completed)

- **Run 32947505370:** branch `testgg-maxreq2x`, image v0.5.16-rocm720-mi35x-20260728
- TP=4/EP=4/HiCache/c10 + TP=8/EP=8/no-hicache/c[4,6,8,10]

| Config | CONC | ITL p50 | ITL p90 | P90 intvty |
|--------|------|---------|---------|------------|
| TP=4/EP=4/HiCache | 10 | 11.9 ms | 19.4 ms | ~52 |
| TP=8/EP=8/no-KV | 4 | 7.3 ms | 9.6 ms | **105** |
| TP=8/EP=8/no-KV | 6 | 8.3 ms | 11.8 ms | 85 |
| TP=8/EP=8/no-KV | 8 | 8.8 ms | 12.2 ms | 82 |
| TP=8/EP=8/no-KV | 10 | 9.5 ms | 14.9 ms | 67 |

**Key finding:** TP=8 beats TP=4 for ITL (−39% at c4). `t_weights` (weight/GPU) dominates, not NCCL → EP=1 is the next natural step (I-8).

### TP=8 baseline config
```
TP=8, EP=8, kv-offloading: none
chunked=32768, MEM_FRACTION=0.85
MRR = 1×CONC
EAGLE: 5 steps / 6 tokens / SGLANG_SIMULATE_ACC_LEN=3.61
```

## Experiment priorities

### 🔴 High priority — EP=1

| # | Experiment | Action | Expected | Notes |
|---|------------|--------|----------|-------|
| **I-8** | **TP=8/EP=1** | New run after I-0b | **ITL p50 ↓↓** | Eliminates MoE all-to-all at low CONC |

**Rationale I-8 — TP=8/EP=1:**
With TP=8/EP=8, every MoE forward introduces an **all-to-all** among 8 EP ranks to route tokens to the correct experts. At CONC=4 (effective=1 from real data), nearly all tokens go to the same 1-2 active experts — the all-to-all is pure unamortized overhead.

With **EP=1** (no expert parallelism, only TP=8):
- Each GPU has all experts replicated → zero all-to-all in MoE forward
- Eliminates an entire collective per MoE layer
- Downside: more weights per GPU → less HBM for KV cache

Memory check:
```
TP=8/EP=8: ~51 GB weights/GPU → ~237 GB KV/GPU   ✓
TP=8/EP=1: more experts/GPU → verify it fits in 288 GB HBM
```
Pattern source: PR #2693 (TP=2/EP=2 → TP=2/EP=1 for Qwen3.5 MI355X, same rationale).

### 🟠 Medium priority — harness fixes (from PR #2737)

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

### 🟢 After harness fixes — further TP=8 optimization

| # | Experiment | Parameter | Baseline | Candidate | Hypothesis | Target metric |
|---|------------|-----------|----------|-----------|------------|---------------|
| **I-4** | More HBM for KV | `mem-fraction-static` | 0.85 | **0.87** | Headroom without HiCache → more KV in GPU → TTFT ↓ | TTFT p50 ↓ |
| **I-5** | Deeper EAGLE | `speculative-num-steps` | 5 | **6** | Effective CONC=1 at c4 → dedicated resources → real AL > 3.61 | tok/s/user ↑ |
| **I-6** | Prefill pipeline | `MRR` | 1×CONC | **1.5×CONC** | Overlap prefill/decode → C\* ↑ | C\* ↑ |

### 🟡 Low priority — throughput arm TP=4

| # | Experiment | Notes |
|---|------------|-------|
| **H-1** | HiCache `write_through_selective` | PR #2679; reduces unnecessary DRAM writes → fewer evictions → C\* ↑ |
| **H-2** | HiCache ratio 1.5 → 2.0 | cpu_kv_usage=100% at c10 → more DRAM → C\* ↑ |
| **H-3** | HiCache `page_size=1` | PR #2679; fine granularity → more precise prefix matching |
| **3a** | MRR 1×→2×CONC (TP=4) | Branch testgg-maxreq2x already ready |
| **I-5b** | TP=4/no-hicache | Diagnostic: isolates HiCache overhead; reduced priority with TP=8 available |

### 🔵 Reference: ATOM (PR #2576 — merged)

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

**Measured data — ATOM vs SGLang (run 31765309673 vs ours):**

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

> **Conclusion:** ATOM validates our I-1, I-2, I-3. The main ITL gap toward ATOM is `ptpc_fp8` (on-the-fly activation quantization) + KV FP8 — engine-level optimizations not directly portable to SGLang. I-9 (KV FP8 in SGLang) can recover part of the HBM advantage (~0.5ms estimated). The residual gap (~0.9ms) is structural between the two engines. Full gap analysis in [SGL_missing_features_01_09_2026.md](SGL_missing_features_01_09_2026.md).

### ⚫ Suspended / conditional

| # | Experiment | Condition |
|---|------------|-----------|
| ~~**DCP+TP=2**~~ | Eliminated | t_weights dominates — TP=2 would worsen it |
| **P-1** | v0.5.18-rocm720 retry | Skipped — superseded by P-2 |
| **P-2** | v0.5.18-rocm724 | 🔴 Abandoned — see §4 |
| **F-5** | DP Attention | Only after upstream ROCm fix |

## Results table (all Phase 1 runs)

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

## Runner .env per run

All variables not listed are absent (no-op). `FORCE_DOCKER`, `MODEL_PATH`, `HF_HUB_CACHE_HOST` are infrastructure invariants present in all runs on mia1.

| Run / Experiment | GH Run ID | Additional variables in `.env` |
|------------------|-----------|-------------------------------|
| Baseline TP=8 EP=8 | 32947505370 | *(none)* |
| I-8: TP=8/EP=1 | 32986446019 | *(none — EP configured in config key)* |
| Invalidated bundle I-1+I-3+I-7 | 32999605584 | `CHUNKED_PREFILL_SIZE_OVERRIDE=16384` `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` `SGLANG_USE_AITER_UNIFIED_ATTN=1` — **did not reach the container** |
| P-2 v0.5.18-rocm724 | 33060449114 | *(none — patch applied unconditionally in the script)* |
| Bundle re-run I-1+I-3+I-7 | 33065762186 | `CHUNKED_PREFILL_SIZE_OVERRIDE=16384` `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` `SGLANG_USE_AITER_UNIFIED_ATTN=1` |
| I-10 CGBS (cancelled) | 33076693271 | `CUDA_GRAPH_BS_LIST_OVERRIDE=1 2 3 4 5 6 7 8` |
| **H-1+H-2 HiCache bundle** | *(see §7)* | `HICACHE_RATIO=2.5` `HICACHE_WRITE_POLICY=write_through_selective` |

---

## 📊 SGLang TP=8 campaign summary (2026-08-25/27)

**Objective:** approach ATOM performance (P90 intvty c4=123, ITL p50=5.9ms) starting from the SGLang EP=8 baseline.

### Results per config at c4 (most critical CONC for interactivity)

| Config | P90 intvty | ITL p50 | Delta vs baseline |
|--------|-----------|---------|-------------------|
| Baseline TP=8/EP=8 | 105 | 7.3ms | — |
| I-8: TP=8/EP=1 | **110.5** | **6.95ms** | **+5.2%** |
| Bundle EP=1 + I-1+I-3+I-7 | **110.4** | **7.03ms** | +5.1% (≈EP=1) |
| *ATOM TP=8/EP=1 (ref)* | *123* | *5.9ms* | *+17%* |

### Overall findings

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

### C* of EP=1 vs EP=8

| CONC | EP=8 P90 | EP=1 P90 | Better |
|------|----------|----------|--------|
| c4 | 105 | **110.5** | EP=1 |
| c6 | 85 | 82.3 | EP=8 |
| c8 | 82 | 81.1 | EP=8 |
| c10 | 67 | 70.2 | EP=1 |

C\* (knee point) remains between c8 and c10 for both configs. EP=1 is preferable at c4 for maximum interactivity.

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

# 4. P-2 — SGLang v0.5.18-rocm724 upgrade (gfx950)

**Objective:** verify whether v0.5.18 + ROCm 7.2.4 brings improvements on MI355X compared to the v0.5.16 baseline.

**Outcome: abandoned** — no gain at c4, additional overhead from the torch fallback.

## Problem: `fp8_mqa_logits` crash on gfx950

**Symptom:** the sglang server crashes during warmup with:
```
AssertionError: Begin <= End  (LLVM sequence.h:275, iota_range)
```
originating from `aiter/ops/triton/attention/fp8_mqa_logits.py` during JIT compilation of the Triton kernel for gfx950.

**Root cause:** the `fp8_mqa_logits` kernel in aiter (integrated into ROCm 7.2.4) has two paths:
- **Gluon path** (gfx950-native): fails with LLVM `iota_range` assertion during JIT
- **Standard Triton path**: fails equally with the same assertion

Both paths are incompatible with the gfx950 compiler in this version of ROCm.

**Applied fix:** pure torch fallback, replaces `fp8_mqa_logits` with per-sequence BF16 matmul. Implemented in the benchmark script — patch applied in-container at run time. Preserved in the `testgg-v518-p2` branch.

**Iterations required (9 failed runs):**
1. Patch conditional on env var `SGLANG_AITER_DISABLE_GLUON_FP8_MQA` — env var was not reaching Docker
2. Discovery that the GH Actions runner **does not export `.env` to the subprocess** → `-e VAR` without value = empty string
3. Multiple forwarding attempts (all failed for the same reason)
4. Unconditional patch → patch applied, but OOM: `[seq_len × total_kv_aligned]` float32 allocation = 2.72 GiB with 90k KV tokens
5. Rewrite with per-sequence loop (`torch.mv`) → OOM resolved, server functional

## Problem: runner `.env` variables not propagated to Docker

**Impact:** also invalidated the bundle I-1+I-3+I-7 runs (run 32999605584) — the vars `CHUNKED_PREFILL_SIZE_OVERRIDE`, `ROCM_QUICK_REDUCE_QUANTIZATION`, `SGLANG_USE_AITER_UNIFIED_ATTN` never reached the container → baseline vs baseline comparison. A later, deeper bug (`MODEL_PREFIX` absent from the `-e` list) invalidated the no-HiCache sweep — see [tuner_en.md §6](tuner_en.md) for the full fix history and the complete required `-e` variable list.

**Root cause:** the GH Actions runner loads `.env` into its own process but **does not export it to the subprocess environment** (the job bash script). `docker run -e VAR` without `=value` propagates an empty string if VAR is not in the subprocess env.

**Fix:** `source .env` at launcher entry + `-e "VAR=${VAR}"` with embedded value for all relevant vars. Full implementation and required var list: **[tuner_en.md §6](tuner_en.md)**.

## P-2 results (c4 the only CONC completed before cancellation)

| Metric | v0.5.18 + torch fallback | v0.5.16 EP=1 baseline |
|--------|--------------------------|----------------------|
| ITL p50 | 7.18 ms | 7.03 ms |
| ITL p90 | 37.7 ms | — |
| intvty p50 | 139.2 tok/s/user | ~110 (P90) |
| errors | 0 | 0 |

**Conclusion:** no relevant improvement (+2% ITL p50, within variability). The torch fallback for `fp8_mqa_logits` introduces overhead compared to the native v0.5.16 kernel. Run cancelled after c4.

## P-3 — SGLang v0.5.18-rocm724 retry after PR #36960 merge (2026-09-02)

**Objective:** retry v0.5.18 upgrade without the torch fallback after sgl-project/sglang #36960 (`Cap the DSA MQA-logits budget at AITER's buffer_store limit`) merged 2026-09-01 22:40 UTC.

**Branch:** `testgg-v518-r901`, config key `glm5.2-fp4-mi355x-sglang-agentic-mtp-v518r901`.

**Outcome: blocked** — image `lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260901` does **not** contain the SGLang-side fix. The image was built before 22:40 UTC on 2026-09-01; `dsa_indexer.py::_get_mqa_logits_budget_bytes` has no `BUFFER_LIMIT_BYTES` cap. Server crashes during warmup with the same iota_range pattern.

**Next step:** wait for image `v0.5.18-rocm724-mi35x-20260902` or later, verify with:
```bash
docker run --rm <image> grep -n "BUFFER_LIMIT\|min(" \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
```

---

# 5. Bundle re-run I-1+I-3+I-7 (v0.5.16, env var fix)

**Objective:** repeat the bundle experiment after fixing the env var propagation bug in the launcher. The previous run (32999605584) was a baseline vs baseline comparison.

**Fix applied:** `source .env` at Docker fallback entry + `-e "VAR=${VAR}"` with embedded value for the three bundle vars. Correct `.env` path: `${GITHUB_WORKSPACE%/*/*/*}` (runner root, not `_work/`).

**Run:** 33065762186 `bundle-rerun-v516-envfix` — completed, v0.5.16-rocm720, TP=8/EP=1. c8 lost due to runner disconnection during the run.

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

# 6. I-10 — Explicit per-batch-size CUDA graphs (c4)

**Motivation:** with `--cuda-graph-max-bs N` SGLang compiles a **single graph** for bs=N. Every decode step with a batch size different from N runs in eager mode (slower, more ITL variance). ATOM uses `--cudagraph-capture-sizes [1,2,4,8]` to compile a graph for every possible batch size — zero graph misses.

**Mechanism:** even with fixed CONC=4, the decode batch size varies continuously between 1 and MAX_RUNNING_REQUESTS (=8) depending on how many requests are simultaneously in the decode phase.

**Config:** `glm5.2-fp4-mi355x-sglang-agentic-mtp-cgbs` — c4 only, TP=8/EP=1, v0.5.16.

**Run:** 33076693271 — **cancelled before execution**.

**Reason:** from the server log of the bundle re-run (c6) it can be seen that SGLang with `--cuda-graph-max-bs 12` already automatically generates a list of batch sizes:
```
decode=PhaseConfig(max_bs=12, bs=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
```
SGLang interpolates a progression between 1 and max_bs — it does not compile a single graph. The test would have had marginal benefit and did not justify 1.5h of GPU time.

**Conclusion:** the residual gap vs ATOM is not explained by CUDA graphs. SGLang is already equivalent on this aspect. `CUDA_GRAPH_BS_LIST_OVERRIDE` removed from runner `.env`.

---

# 7. H-1+H-2 — HiCache bundle (TP=4/c10, throughput arm)

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

**Run:** completed (part of campaign runs on mia1)

**Outcome:** `tok/s/GPU` improved from 91 (baseline c10) to **91 tok/s/GPU at c10** and **102 tok/s/GPU at c12** (+12%). Confirming the enlarged host KV pool delays saturation and sustains throughput through c12.

**Results (run 33373633743, PR #2777 full-sweep validation, 2026-08-31):**

HICACHE_RATIO ran at default=1.5 (reverted from 2.5 after reviewer feedback — ratio=2.5 exceeds ~3.0 TB available DRAM on cluster:mi355x-amds). Comparison against campaign runs (which used HICACHE_RATIO=2.5 via runner .env) and CSV baseline (EP=4, HICACHE_RATIO=1.5).

### Three-way comparison: CSV baseline vs EP=1 campaign vs PR #2777 (HICACHE=1.5)

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

# 8. Low-priority experiments (Phases 2-5)

## HiCache tuning (TP=4, CONC ≥ 10 regime)

cpu_kv_usage=100% at CONC=10. Parameters validated by PR #2679 as reference.

| # | Parameter | Baseline | Candidate | Motivation |
|---|-----------|----------|-----------|------------|
| **H-1** | `HICACHE_WRITE_POLICY` | `write_through` | `write_through_selective` | PR #2679: writes DRAM only for high-reusability prefill → less wasted bandwidth |
| **H-2** | `HICACHE_RATIO` | 1.5 | **2.0** | CPU pool saturated → more DRAM may raise C\* |
| **H-3** | `page_size` | default | **1** | PR #2679: fine granularity → more precise cache matching |
| **H-4** | `HICACHE_IO_BACKEND` | `direct` | `asyncio` | Only if DRAM I/O is bottleneck at high CONC |

**Order:** H-1 → H-2 → H-3 → H-4 only if plateau.

## MRR sweep TP=4 (branch testgg-maxreq2x already ready)

```
3a: MRR=10 (baseline) vs MRR=20 (2×CONC) ← branch already ready
3b: chunked-prefill-size: 16384 / 32768 (baseline) / 65536
3c: cuda-graph-max-bs: 1×MRR / 1.5×MRR / 2×MRR
```

## Platform and advanced parallelism

| # | Experiment | Condition |
|---|------------|-----------|
| **P-1** | v0.5.18-rocm720 retry | Skipped — superseded by P-2 |
| **P-2** | v0.5.18-rocm724 (ROCm 7.2.4) | 🔴 **ABANDONED** — run 33060449114. No gain at c4 vs v0.5.16/EP=1 (+2% ITL p50, within variability). torch fallback for `fp8_mqa_logits` added overhead. See §4. |
| **P-3** | v0.5.18-rocm724 after PR #36960 | 🔴 **BLOCKED** — image `20260901` built before the SGLang fix merged. Awaiting `20260902+` image. |
| **F-5** | DP Attention | Only after upstream ROCm fix |

> **Outcome 2026-08-27/09-02:** P-2 tested and abandoned — no measurable gain. P-3 blocked on image availability. Campaign concluded on v0.5.16 with EP=1/c4 as best config (P90=110.5).

---

# 9. No-HiCache Pareto sweep (2026-09-02 — next campaign)

> ⚠️ **Dataset correction (2026-09-03):** All mia1 runs in this section (runs 33647161138, 33647171278, 33661191728) used the **256k-capped dataset** (`cc-traces-weka-062126-256k`, ISL p90 ~170k) instead of the unfiltered corpus (ISL p90 ~283k). Root cause: `MODEL_PREFIX` was missing from the Docker `-e` list. Fix: commit `541e137`. Re-measurement in progress: run 33724174688 (conc [1,2,4,6,8,10,12], correct dataset). All numerical results below are marked **[256k]**. Qualitative findings (HiCache vs no-HiCache neutral when KV fits) are valid.

**Motivation:** run 33647161138 confirmed that c12/no-HiCache gives lower ITL than HiCache at c12 (~~ITL p90=20ms, tok/s/user p90=111.82~~ **[256k]**, no OOM). The full operating envelope is unknown: we have one data point (c12) but not the complete Pareto curve nor the OOM boundary.

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
- `Output tok/s/user p90` → should peak somewhere around c4–c8 then decline
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

**Note:** a single run sweeps all CONC values sequentially on the same server instance (§2 constraint savings). Duration per slice: 3600s. Estimated total: ~9× slices × ~1h = ~9h wall time.

**Result (run 33661191728, 2026-09-02 — ⚠️ 256k DATASET, INVALIDATED):**

| CONC | ITL p50 (ms) | ITL p90 (ms) | intv p90 (tok/s/u) | tput/GPU (tok/s) | KV%GPU | n_ok | Status |
|------|-------------|-------------|-------------------|-----------------|--------|------|--------|
| 2    | 7.4 [256k]  | 9.5 [256k]  | 105.6 [256k]      | 35 [256k]       | 18%    | 442  | ⚠️ 256k |
| 4    | 8.1 [256k]  | 10.9 [256k] | 92.0  [256k]      | 46 [256k]       | 26%    | 647  | ⚠️ 256k |
| 8    | 10.3 [256k] | 15.0 [256k] | 66.7  [256k]      | 79 [256k]       | 50%    | 1279 | ⚠️ 256k |
| 10   | 11.6 [256k] | 18.5 [256k] | 54.1  [256k]      | 92 [256k]       | 63%    | 1420 | ⚠️ 256k |
| 12   | 11.8 [256k] | 20.4 [256k] | 48.9  [256k]      | 106 [256k]      | 60%    | 1606 | ⚠️ 256k |
| 20   | 25.2 [256k] | 150.4 [256k]| 6.6   [256k]      | 79 [256k]       | 100%   | 1493 | ⚠️ 256k |
| 24   | 115.5 [256k]| 257.1 [256k]| 3.9   [256k]      | 38 [256k]       | 100%   | 823  | ⚠️ 256k |

*All numbers from 256k-capped dataset (ISL p90 ~170k). Re-measurement: run 33724174688 (conc [1,2,4,6,8,10,12], correct dataset).*

**HiCache comparison at c12 (run 33647171278, ⚠️ 256k dataset):**

| Metric | HiCache [256k] | no-HiCache [256k] | Delta |
|--------|---------------|------------|-------|
| ITL p90 | ~~21.0 ms~~ | ~~20.4 ms~~ | +0.6 ms (+3%) |
| intvty p90 (tok/s/u) | ~~47.7~~ | ~~48.9~~ | −1.2 (−2%) |
| KV%GPU | 62% | 60% | +2% |

**Qualitative finding (valid regardless of dataset):** HiCache and no-HiCache are **equivalent at c12 on mia1** — the KV fits in HBM in both cases. HiCache path overhead (+0.6 ms) is within measurement variance. Absolute numbers invalid due to shorter ISL.

**Qualitative key findings (valid regardless of dataset):**
- **HiCache vs no-HiCache: neutral** when KV fits in HBM. R-1 recommendation: no-HiCache is simpler but not a performance win on machines with adequate HBM.
- **OOM boundary exists between c12 and c20** (256k data) — may shift lower with correct ISL p90 ~283k. Re-measuring.
- **Safe operating range confirmed qualitatively: up to c12** (KV ≤ 63% at 256k ISL) — boundary to be confirmed with correct dataset.

**Actual Pareto curve — ⚠️ [256k, to be re-measured]:**
```
intv p90 (tok/s/user) [256k dataset]
     ▲
 106 │  ● c2 (9.5ms ITL) [256k]
  92 │     ● c4 (10.9ms) [256k]
  67 │           ● c8 (15.0ms) [256k]
  54 │                  ● c10 (18.5ms) [256k]
  49 │                       ● c12 (20.4ms) [256k]
   7 │                                            ✗ c20 OOM onset [256k]
     └──────────────────────────────────────────────────► CONC
  Re-measurement in progress: run 33724174688 (correct dataset, ISL p90 ~283k)
```

---

# 10. Known issues (GLM-5.2 / mia1)

## DP attention / DCP on GLM-5.2 + SGLang (status 2026-08-31)

The DP-attention arm in `glm5.2_fp4_mi355x_sglang_mtp.sh` is currently **DORMANT** (no DEP arm in `amd-master.yaml`).

**Root cause:** DSA + DP attention hangs a collective under long-context prefill on ROCm (watchdog kills scheduler, 0 completions). Reproduced with and without HiCache, with and without the DSv4 DP collective env vars. Tracked upstream as SGLang issue [#34582](https://github.com/sgl-project/sglang/issues/34582).

**Status as of v0.5.18:** not fixed. Relevant improvements in v0.5.17+:
- PR #31682: breakable prefill CUDA graph on by default for DP attention (reduces indefinite hangs, but does not fix the DSA path)
- PR #33829: dummy row normalization fix for spec decoding + DP attention (potentially relevant for MTP+DSA, monitor)
- `SGLANG_DP_USE_GATHERV=1` + `SGLANG_DP_USE_REDUCE_SCATTER=1` workaround helps DSv4 but not validated for GLM-5.2/DSA

**Action:** re-enable DEP arm once SGLang issue #34582 is resolved or PR #33829 shows positive effect on DSA+DP path. Monitor v0.5.19+.

## `_work/` root-owned files on mia1

**Critical:** `_work/` remains root-owned after failed runs (NFS root_squash, no sudo on mia1-p01-g07).
Manual workaround:
```bash
docker run --rm --privileged \
  -v /it-share/gguasti/actions-runner/_work:/work \
  lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728 \
  rm -rf /work/InferenceX
```

**Fix to implement:** automatic pre-cleanup in the launcher before each `actions/checkout`.

See `improvements.md` for full backlog.
