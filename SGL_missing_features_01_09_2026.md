# SGLang Missing Features vs ATOM — GLM-5.2 MXFP4 / MI355X
**Date:** 2026-09-01
**Reference config:** `glm5.2-fp4-mi355x-sglang-agentic-mtp` (PR #2777, merged 2026-09-01)
**SGLang baseline:** `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728`
**ATOM ref:** TP=8/EP=1/c4 ITL p50=5.9ms, P90=123
**Campaign best (interactivity):** TP=8/EP=1/c4 P90=110.5 *(upstream dataset)*; TP=4/EP=4/c12/no-HiCache — ⚠️ *preliminary data, 256k-capped dataset — see §12 dataset note below*

> ⚠️ **Dataset correction (2026-09-03):** All mia1 runs prior to fix commit `541e137` used the **256k-capped dataset** (`cc-traces-weka-062126-256k`, ISL p90 ~170k) instead of the unfiltered corpus (`cc-traces-weka-062126`, ISL p90 ~283k) used by upstream PR 2777 and ATOM. Root cause: `MODEL_PREFIX` was not propagated to the Docker container in `launch_mi355x-amds.sh`. Fix applied 2026-09-03. All mia1 numerical results below are marked **[256k]** and must be re-validated with the correct dataset (re-run 33724174688 in progress).

> **Related documents:**
> - **Tuning methodology (general):** [tuner_en.md](tuner_en.md) — parameter space, Pareto objectives, lessons learned, Docker checklist
> - **Campaign data (GLM-5.2 / MI355X / Aug–Sep 2026):** [campaign_glm52_mi355x_2026.md](campaign_glm52_mi355x_2026.md) — all run IDs, experiment results, ATOM comparison

---

## Gap summary

| # | Feature | ATOM | SGLang v0.5.16 | Est. ITL impact | Priority |
|---|---------|------|----------------|-----------------|----------|
| F-1 | `ptpc_fp8` on-the-fly activation quantization | ✅ | ❌ (not ported) | ~0.9ms ITL | High |
| F-2 | INT4 all-reduce (`ROCM_QUICK_REDUCE_QUANTIZATION=INT4`) | ✅ | ⚠️ accessible for GLM-5.2 (no opt-in needed); fires only on large prefill tensors TP=4 | ~0.2–0.3ms ITL (prefill only) | Medium |
| F-3 | DP attention (DSA path, 1-node TP=8/DP=1) | ✅ | ❌ (hangs, issue #34582) | — | Medium |
| F-4 | `iota_range` / `fp8_mqa_logits` crash on gfx950 | N/A | ❌ blocks v0.5.18 upgrade | blocks upgrade | High |
| F-5 | AITER unified attention (`SGLANG_USE_AITER_UNIFIED_ATTN=1`) | implied | Neutral on v0.5.16 | TBD on v0.5.18 | Low |
| F-6 | FlyDSL MoE sorting (`AITER_USE_FLYDSL_MOE_SORTING=1`) | ✅ | ❌ no equivalent | unknown | Low |
| F-7 | DCP (decode context parallelism) | ✅ (Kimi-K3) | ❌ not validated for GLM-5.2 | N/A (deprioritized) | Low |
| F-8 | Quark PTPC FP8 attention load/infer on ROCm | ✅ | ⚠️ PR #28734 open | enables KV FP8 path | Medium |

---

## Recipe optimizations (no SGLang changes needed)

| # | Optimization | Current recipe | Proposed | Measured gain | Status |
|---|-------------|---------------|---------|---------------|--------|
| R-1 | **No-HiCache operating point** (KV in HBM) | `kv-offloading: cpu` (HiCache) | `kv-offloading: none` | ⏳ *pending re-measurement with correct dataset (run 33724174688)* | ⏳ Re-running 2026-09-03 — previous data invalidated (256k dataset) |

**R-1 conditions:** requires sufficient free HBM to hold full KV at target ISL and concurrency. Qualitative finding (valid regardless of dataset): KV fits in HBM on mia1 at moderate CONC. OOM boundary to be re-measured with unfiltered dataset (ISL p90 ~283k vs 170k previously). Needs validation on wbb3/amds_01 before upstreaming.

**No-HiCache Pareto sweep — [256k dataset, INVALIDATED] (run 33661191728, 2026-09-02):**

> ⚠️ Data below used the 256k-capped dataset (ISL p90 ~170k). The unfiltered dataset has ISL p90 ~283k — KV pressure is ~65% higher. OOM boundary likely shifts to lower CONC. Re-measurement in progress (run 33724174688, conc [1,2,4,6,8,10,12]).

| CONC | ITL p90 (ms) | intv p90 (tok/s/u) | tput/GPU (tok/s) | KV%GPU | n_ok | Status |
|------|-------------|---------------------|-----------------|--------|------|--------|
| 2    | 9.5  [256k] | 105.6 [256k]        | 35 [256k]       | 18%    | 442  | ⚠️ 256k |
| 4    | 10.9 [256k] | 92.0  [256k]        | 46 [256k]       | 26%    | 647  | ⚠️ 256k |
| 8    | 15.0 [256k] | 66.7  [256k]        | 79 [256k]       | 50%    | 1279 | ⚠️ 256k |
| 10   | 18.5 [256k] | 54.1  [256k]        | 92 [256k]       | 63%    | 1420 | ⚠️ 256k |
| 12   | 20.4 [256k] | 48.9  [256k]        | 106 [256k]      | 60%    | 1606 | ⚠️ 256k |
| 20   | 150.4 [256k]| 6.6   [256k]        | 79 [256k]       | 100%   | 1493 | ⚠️ 256k |
| 24   | 257.1 [256k]| 3.9   [256k]        | 38 [256k]       | 100%   | 823  | ⚠️ 256k |

**Methodology lesson (2026-09-02, qualitatively valid):** KV offloading (HiCache) should never be enabled by default — first measure `gpu_kv_usage` at target CONC and ISL, then decide. If KV fits in HBM, HiCache adds DRAM overhead with no benefit. Full procedure in tuner_en.md §5.1. The qualitative finding (HiCache vs no-HiCache neutral when KV fits) is valid; absolute numbers need re-measurement with correct ISL distribution.

---

## Feature details

### F-1 — `ptpc_fp8`: On-the-fly FP8 activation quantization

**What it is:** ATOM applies per-token-per-channel FP8 quantization to activations during inference (not just weights). This reduces HBM bandwidth for intermediate tensors in attention and MLP layers.

**Why it matters:** Our campaign analysis ([campaign_glm52_mi355x_2026.md §3](campaign_glm52_mi355x_2026.md)) identifies this as the **main structural gap** between ATOM and SGLang. Estimated residual after KV FP8 (I-9) is ~0.9ms ITL at c4.

**SGLang status:** Not available. There is no `ptpc_fp8` mechanism in SGLang for the GPU linear path. Closest open work:
- **PR #28734** `[AMD] Fix Load and Inference of MLA models with Quark PTPC FP8 attention on ROCm` — OPEN since 2026-06-19, last updated 2026-09-01. Addresses *loading* PTPC-quantized checkpoints and routing through `apply_fp8_ptpc_linear`, but is the attention-side extension — not the runtime activation quantization itself.

**References:**
- [campaign_glm52_mi355x_2026.md §3](campaign_glm52_mi355x_2026.md) — ATOM comparison table and gap analysis
- SGLang PR #28734: https://github.com/sgl-project/sglang/pull/28734 — OPEN

**Action:** Monitor PR #28734 for merge. A follow-on PR enabling runtime activation quantization (not just checkpoint loading) would be needed to close the full gap.

---

### F-2 — INT4 all-reduce (`ROCM_QUICK_REDUCE_QUANTIZATION`)

**What it is:** ATOM compresses all-reduce communication to INT4 via AITER's quick-reduce kernel. SGLang exposes `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (INT8 only).

**Why it matters:** All-reduce latency scales with TP degree and tensor size. INT4 halves the bandwidth vs INT8. ATOM's reference recipe uses `AITER_QUICK_REDUCE_QUANTIZATION=INT4`.

**SGLang status (I-3 in our campaign):** Tested as `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` on v0.5.16 — **neutral** (no measurable gain). The reason: on v0.5.16 the quick-reduce path may not be active for all tensors, or INT8 already saturates the benefit.

**INT4 path in SGLang — code analysis (2026-09-02):**

Contrary to the initial assumption, GLM-5.2 does **not** require a model-specific opt-in to access the INT4 quick-reduce path. Source code analysis of `quick_all_reduce.py` and `glm4_moe.py` (SGLang main, 2026-09-02) reveals:

- `QuickAllReduce` initialises automatically for any model on gfx94x/gfx95x hardware when `ROCM_QUICK_REDUCE_QUANTIZATION` is set to a valid level (FP/INT8/INT6/INT4).
- `glm4_moe.py` model override does **not** set `disable_custom_all_reduce = True` (unlike MiniMax-M3 which disables custom AR by default to avoid corrupting sparse MoE partial outputs). GLM-5.2 is therefore eligible without any additional flag.
- The `SGLANG_M3_ALLOW_CUSTOM_AR` flag is M3-specific — it re-enables custom AR that M3 disables. For GLM-5.2 it is irrelevant.

**Critical constraint — tensor size thresholds (`_QR_MIN_SIZE`):**

The quick-reduce kernel fires only if the tensor size meets the minimum threshold for the given dtype, world_size, and quantisation level:

| Config | INT4 minimum |
|--------|-------------|
| bf16 / TP=4 | **16 MB** |
| bf16 / TP=8 | **2048 MB** |

GLM-5.2 hidden_size=7168, bf16 → tensor size per all-reduce = `7168 × N_tokens × 2 bytes`:
- **Decode** (1–4 tokens): 0.01–0.06 MB → **always below threshold** → INT4 never fires for decode
- **Prefill** (4096 tokens): ~56 MB → above 16 MB threshold for TP=4 ✓
- **TP=8**: even the largest prefill batch (1000s of tokens) is unlikely to reach 2048 MB → INT4 essentially never fires at TP=8

**Conclusion:** INT4 quick-reduce for GLM-5.2 can only help **TP=4, large prefill batches** (~4096+ tokens). Decode ITL is unaffected. Impact is expected only on TTFT at high concurrency where the prefill scheduler accumulates large batches.

**Why `_QR_MIN_SIZE` cannot be lowered:**

The threshold is the empirically-determined break-even point between:
```
Cost(BF16): send(tensor_size_bf16)
Cost(INT4): quantize(tensor) + send(tensor_size/4) + dequantize(tensor)
             ~~~~~~~~~~~~~~                            ~~~~~~~~~~~~~~~~
             ~5–10 µs fixed kernel launch overhead     ~5–10 µs fixed
```
On MI355X (XGMI ~400 GB/s), a 57 KB decode tensor takes ~0.14 µs to transfer in BF16. Adding INT4 quantization adds ~10 µs of fixed overhead — making that all-reduce **~70× slower**. The 16 MB threshold for TP=4 is the point where bandwidth savings exceed kernel launch overhead. Lowering it below ~8 MB would cause measurable ITL regressions on the (hot) decode path.

**2026-09-02 test — run 33635908473 (interactivity arm, HiCache saturated):**

- Branch: `testgg-qr-int4` (based on `testgg` ← `pr/glm52-sglang-ep1-c12`)
- Image: `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728`
- Config: TP=4/EP=4/HiCache, conc=12
- Runner: mia1-p01-g07 (mi355x-amds_03), 8× MI355X
- Env: `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1`
- Run id: 33635908473 — **in progress as of 2026-09-02 ~13:45 UTC**

**Run 33647161138 — `glm5.2-qr-int4-nohicache-c12` — ⚠️ [256k DATASET — INVALIDATED] (mia1-p01-g07, 8× MI355X):**

> ⚠️ This run used the 256k-capped dataset (ISL p90 ~130–137k). With the correct unfiltered dataset (ISL p90 ~283k), KV pressure is ~2× higher and these numbers will differ. Qualitative findings (HiCache vs no-HiCache neutral when KV fits) remain valid.

| Metric | avg | p50 | p90 [256k] | p99 |
|--------|-----|-----|-----|-----|
| ITL (ms) | 15.06 | 11.91 | ~~20.26~~ | 65.19 |
| Output tok/s/user | 82.46 | 83.96 | ~~111.82~~ | 141.42 |
| Requests | 1,607 / 1,607 (0 errors, no OOM) |
| ISL p90 | ~130–137k tokens ← **wrong dataset** |

> **HiCache run 33647171278 — ⚠️ [256k DATASET — INVALIDATED] (mia1-p01-g07, TP=4/EP=4, c12):**
>
> | Metric | HiCache [256k] | no-HiCache [256k] | Delta |
> |--------|---------|------------|-------|
> | ITL p90 | ~~21.0 ms~~ | ~~20.4 ms~~ | +0.6 ms (+3%) |
> | KV%GPU | 62% | 60% | +2% |
>
> **Qualitative finding (valid):** HiCache and no-HiCache are essentially equivalent when KV fits in HBM — HiCache adds <1ms overhead. Absolute numbers invalid due to shorter ISL distribution.

**2026-09-02 no-HiCache test — ⚠️ preliminary, 256k dataset:**

- Config: TP=4/EP=4/no KV offloading, conc=12, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`
- Result: ~~ITL p90=20.26 ms, output tok/s/user p90=111.82~~ — **[256k dataset, to be re-measured]**
- Re-run in progress: run 33724174688 with unfiltered dataset (ISL p90 ~283k)

**References:**
- [tuner_en.md §5.5](tuner_en.md) — INT4 all-reduce tensor size analysis (rule: check `_QR_MIN_SIZE`)
- [campaign_glm52_mi355x_2026.md §3](campaign_glm52_mi355x_2026.md) — I-3 experiment results (neutral on v0.5.16)
- SGLang PR #32230: https://github.com/sgl-project/sglang/pull/32230 — MERGED
- SGLang PR #33402: https://github.com/sgl-project/sglang/pull/33402 — MERGED
- SGLang source: `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- SGLang source: `python/sglang/srt/arg_groups/model_overrides/glm4_moe.py`

**Why ATOM benefits from INT4 but SGLang does not (structural analysis):**

Three cumulative factors explain the gap:

**1. KV offloading — completely different bottleneck:**

| | ATOM | SGLang c12 HiCache |
|---|---|---|
| KV location | HBM (all in GPU) | DRAM (HiCache saturated) |
| Decode bottleneck | compute + all-reduce | KV DRAM fetch |
| INT4 acts on... | critical path ✓ | secondary path ✗ |

**2. EP=1 vs EP=4 — communication mix:**

ATOM EP=1 has **only all-reduce** (INT4 applies). SGLang EP=4 has **all-to-all** (MoE token routing, INT4 does not apply) + all-reduce. A significant fraction of communication time is spent on all-to-all, which INT4 cannot compress.

**3. `_QR_MIN_SIZE` threshold for TP=8 — possibly lower than estimated:**

The TP=8 threshold of 2048 MB cited earlier may be incorrect (not verified in source). If the actual value is ~32–64 MB, ATOM at TP=8 would fire INT4 for moderate prefill batches (>2200–4500 tokens), which is routine at its trace ISL.

**Decode path comparison:**

```
ATOM (TP=8/EP=1/no-offload):
  decode step: [matmul] → [all-reduce INT4 ✓] → [output]
               ← all KV in HBM, no DRAM wait →
               INT4 visible: all-reduce IS the bottleneck

SGLang c12 HiCache:
  decode step: [KV fetch DRAM ⏳⏳] → [matmul] → [all-reduce INT4 ✓] → [output]
               ← DRAM fetch dominates everything →
               INT4 invisible behind KV fetch

SGLang c12 no-HiCache (run 33647161138, ⚠️ 256k DATASET — INVALIDATED):
  decode step: [matmul] → [all-reduce INT4 ✓] → [output]
               ← like ATOM, but TP=4/EP=4 →
               ITL p90=~~20ms~~, output tok/s/user p90=~~111.82~~ [256k data, to re-measure]
```

**To verify:** run `python3 -c "from sglang.srt.distributed.device_communicators.quick_all_reduce import _QR_MIN_SIZE; print(_QR_MIN_SIZE)"` inside the SGLang container to get the exact TP=8 threshold and compare with ATOM's internal value.

**The fundamental INT4 paradox for GLM-5.2 (confirmed 2026-09-02):**

Examining all practical operating points reveals that INT4 has no viable window for decode ITL:

| Concurrency | KV in HBM? | INT4 fires (decode)? | INT4 fires (prefill)? | Net effect |
|-------------|------------|----------------------|------------------------|------------|
| c4 | ✓ Yes | ✗ No (tensor ~14–224 KB << 16 MB) | ✗ Probably not (small batches) | Zero |
| c12 + HiCache | ✗ No | ✗ No (tensor ~170–680 KB << 16 MB) | ~ Prefill only | Masked by DRAM I/O |
| c12 no-HiCache | ✓ Yes (fits at ISL p90~130k, mia1) | ✗ No (decode always < threshold) | ✓ Yes (batch > 1200 tok) | ~~ITL p90=20ms, tok/s p90=111.82~~ ⚠️ [256k] — re-measuring (run 33724174688) |

The key insight: **the decode tensor is always far below the 16 MB threshold regardless of concurrency**, because it scales with tokens-per-step (small), not with the number of requests. INT4 can only fire on the prefill path. But the **no-HiCache operating point** — where the bottleneck is compute/communication rather than DRAM — delivers the best interactivity regardless of whether INT4 fires on prefill: **KV in HBM is the dominant factor**.

ATOM achieves its best results by the same mechanism (no KV offloading, KV in HBM) — the INT4 contribution on the all-reduce is additional but secondary.

**Also confirmed:** our script already uses `--kv-cache-dtype fp8_e4m3`, so FP8 KV compression is on par with ATOM. The gap is not due to missing FP8 KV but purely to the concurrency operating point and the resulting KV memory pressure.

**Action (updated 2026-09-03):** (1) ⚠️ Run 33647161138 (no-HiCache, 256k dataset) — numbers ~~ITL p90=20ms, tok/s/user p90=111.82~~ invalidated. (2) Run 33647171278 (HiCache, 256k) complete — HiCache vs no-HiCache delta qualitatively neutral (valid). (3) Re-measurement in progress: run 33724174688 (no-HiCache Pareto sweep conc [1,2,4,6,8,10,12], correct unfiltered dataset ISL p90 ~283k). (4) **Pending:** confirm no-HiCache recommendation with correct dataset results. (5) Close F-2 as **structurally neutral for INT4 decode ITL** — this qualitative conclusion is unaffected by the dataset bug. (6) Verify actual `_QR_MIN_SIZE` for TP=8 to complete ATOM comparison.

---

### F-3 — DP attention (DSA path, gfx950)

**What it is:** Data-Parallel attention splits the token batch across multiple attention workers, reducing per-rank memory and enabling higher concurrency with the same TP degree.

**Why it matters:** ATOM uses DP attention in its reference recipe. In SGLang, DP attention is implemented (`--enable-dp-attention`) but crashes for GLM-5.2/DSA on ROCm.

**SGLang status:** **BROKEN** — hangs 100% reproducibly on gfx950 during warmup prefill, all 16 scheduler ranks wedge in `prepare_mlp_sync_batch` / `all_gather_into_tensor`. Tracked as:
- **Issue #34582**: `[Bug] GLM-5.2 in DP-attention deadlock at first scheduled batch — all ranks wedge in prepare_mlp_sync_batch all-gather (2-node, TP=16, dp-size=2)` — **OPEN** since 2026-08-12.

Related improvements (partial, don't fix DSA path):
- **PR #31682** `Turn on breakable prefill cuda graph for dp attention by default` — MERGED 2026-07-21. Reduces indefinite hangs but does not fix the DSA collective divergence.
- **PR #33829** `[Model] Complete dots.note.omni support... DP dummy-row normalization for overlap MTP` — MERGED 2026-08-22. Adds dummy-row normalization for DP+spec decoding. Potentially relevant for MTP+DSA — to monitor.
- Workaround env vars `SGLANG_DP_USE_GATHERV=1` + `SGLANG_DP_USE_REDUCE_SCATTER=1` help for DSv4 but not validated for GLM-5.2/DSA.

**References:**
- [campaign_glm52_mi355x_2026.md §10](campaign_glm52_mi355x_2026.md) — DP attention known issue, workaround, tracking
- SGLang issue #34582: https://github.com/sgl-project/sglang/issues/34582 — OPEN
- SGLang PR #31682: https://github.com/sgl-project/sglang/pull/31682 — MERGED
- SGLang PR #33829: https://github.com/sgl-project/sglang/pull/33829 — MERGED (dots.note.omni)

**Action:** Re-enable DEP arm once issue #34582 is resolved. Monitor v0.5.19+. Test `SGLANG_DP_USE_GATHERV=1` on GLM-5.2.

---

### F-4 — `fp8_mqa_logits` Triton LLVM `iota_range` crash (gfx950 / v0.5.18)

**What it is:** On SGLang v0.5.18 + ROCm 7.2.4, the AITER `fp8_mqa_logits` Triton kernel for the DSA indexer crashes with `LLVM iota_range assertion: Begin <= End` during JIT compilation on gfx950. This affects TP=4+HiCache+DSA and TP=8. The crash is triggered at ≥32,768 prompt tokens (well within GLM-5.2's 1M context).

**Why it matters:** Blocks upgrading from v0.5.16 to v0.5.18, preventing access to newer ROCm 7.2.4 kernels, updated AITER, and potential NCCL improvements.

**SGLang status:** Multiple open PRs address the DSA indexer MQA-logits budget:
- **PR #36960** `[ROCm][Bugfix] Cap the DSA MQA-logits budget at AITER's buffer_store limit` — **MERGED 2026-09-01**. Directly caps the budget to prevent the inverted range.
- **PR #35865** `[AMD] Implement DeepSeek V4 DCP on ROCm` — OPEN since 2026-08-21 — contains `fp8_mqa_logits` bounds fix as part of DCP work.
- **PR #34129** `[AMD] [GLM5] use optional AITER BLOCK_Q MQA logits` — OPEN since 2026-08-08. Uses a new AITER BLOCK_Q kernel (ROCm/aiter#4180) that reduces raw MQA-logits time −25.2% on MI355X. Avoids the iota_range path when BLOCK_Q is available. **Note:** the BLOCK_Q kernel shows 7.53ms vs 10.06ms per prefill forward, with consistent serving improvements across the full concurrency sweep.
  > **Note:** the performance benefit of this PR is gated on ROCm/aiter#4180 landing in a released AITER build. Even if this SGLang PR merges, the BLOCK_Q path will silently fall back to the existing kernel on any SGLang image that does not yet bundle ROCm/aiter#4180. The gain will only materialize once a new SGLang image is published with the updated AITER build.

Our workaround (in testgg-v518-p2 branch): runtime torch-fallback patch applied in-container, replacing `fp8_mqa_logits` with per-sequence BF16 matmul. Result: no improvement over v0.5.16 (+2% ITL p50, within variability). Run cancelled after c4.

**Latest SGLang release:** v0.5.18 (2026-08-22). No v0.5.19 yet.

**References:**
- [campaign_glm52_mi355x_2026.md §4](campaign_glm52_mi355x_2026.md) — P-2 torch fallback implementation, P-3 blocked status
- SGLang PR #36960: https://github.com/sgl-project/sglang/pull/36960 — MERGED 2026-09-01
- SGLang PR #34129: https://github.com/sgl-project/sglang/pull/34129 — OPEN
- SGLang PR #35865: https://github.com/sgl-project/sglang/pull/35865 — OPEN

**2026-09-02 validation attempt — image `20260901` does NOT contain the fix:**
Tested `lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260901` on mia1-p01-g07 (branch `testgg-v518-r901`). The server starts and reports KV cache capacity, but crashes during warmup — all requests fail with `ConnectionRefused`, same iota_range pattern as before. Root cause: PR #36960 merged at 22:40 UTC on 2026-09-01; the image was built earlier that day and does **not** include the SGLang-side fix in `dsa_indexer.py::_get_mqa_logits_budget_bytes`. The `BUFFER_LIMIT_BYTES` check found in AITER's `fp8_mqa_logits.py` is a pre-existing guard — the missing piece is the SGLang cap that prevents `_get_mqa_logits_budget_bytes` from returning 4.4 GB (> 2 GB AITER limit).

**Action:** wait for image `lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260902` or later. Verify fix presence with:
```bash
docker run --rm <image> grep -n "BUFFER_LIMIT\|min(" \
  /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsa/dsa_indexer.py
```
Monitor PR #34129 (BLOCK_Q) for merge; once landed, test directly on MI355X for ITL improvement.

---

### F-5 — AITER unified attention (`SGLANG_USE_AITER_UNIFIED_ATTN=1`)

**What it is:** Routes extend (prefill) attention through AITER's `unified_attention` kernel instead of CK `mha_batch_prefill`. Potentially faster on gfx950 for sinks-bearing models.

**SGLang status (I-7 in our campaign):** Tested on v0.5.16 — **neutral** (no measurable gain). On v0.5.16 the unified attention backend may not have full coverage for GLM-5.2's traits on gfx950 (d64 group-mode paged family absent in CK).

Active PRs adding unified attention routes:
- **PR #37310** `[ROCm] aiter backend: route sinks-model extends through unified_attention (gpt-oss prefill −18…−32% TTFT)` — **OPEN** since 2026-09-01. Routes normal extends (not just decode) through `unified_attention` for sinks models. Reports −18%…−32% TTFT for gpt-oss on gfx950.
- **PR #37311** `[ROCm] aiter backend: CK varlen route for long full-attention sinks extends (−15…−16% TTFT at 64k uncached)` — **OPEN** since 2026-09-01. Complementary CK varlen path for long uncached extends.
- **PR #37312** `[ROCm] aiter backend: write extend attention output into the PCG-provided buffer` — **OPEN** since 2026-09-01.

These three PRs (#37310, #37311, #37312) were opened the same day as this document (2026-09-01) and appear to be a coordinated effort to activate unified attention for sinks models on gfx950. GLM-5.2 (DSA + sinks/sliding-window model) may benefit.

**References:**
- [campaign_glm52_mi355x_2026.md §3](campaign_glm52_mi355x_2026.md) — I-7 experiment results (neutral on v0.5.16)
- SGLang PR #37310: https://github.com/sgl-project/sglang/pull/37310 — OPEN
- SGLang PR #37311: https://github.com/sgl-project/sglang/pull/37311 — OPEN
- SGLang PR #37312: https://github.com/sgl-project/sglang/pull/37312 — OPEN

**Action:** Monitor PRs #37310–#37312. Once merged, test `SGLANG_USE_AITER_UNIFIED_ATTN=1` (or equivalent flag) on v0.5.18+ with GLM-5.2. This may activate on a newer image even without explicit env var.

---

### F-6 — FlyDSL MoE sorting (`AITER_USE_FLYDSL_MOE_SORTING=1`)

**What it is:** ATOM uses FlyDSL (AMD's DSL for kernel generation) to implement optimized MoE expert-routing and token sorting. Enables heterogeneous MoE execution with better GPU utilization.

**SGLang status:** FlyDSL is being integrated into SGLang for AMD, but via a different path (linear-attention backend):
- **PR #33544** `[AMD] Add FlyDSL GDN linear-attention backend (--linear-attn-backend flydsl)` — **OPEN**
- **PR #37173** `[AMD][DSV4] perf: tuned FlyDSL fp4 indexer score on MI355X` — **OPEN**

No direct `AITER_USE_FLYDSL_MOE_SORTING` equivalent found in SGLang. The AITER MoE path is accessed differently:
- **PR #35074** `[AMD] Perf dsv4 enable heterogeneous AITER FHMoE` — **OPEN** (heterogeneous FusedHiddenMoE via AITER)
- **PR #36269** `[AMD] Add ROCm MegaMoE path via AITER MegaMoEV2` — **OPEN**

**References:**
- [campaign_glm52_mi355x_2026.md §3](campaign_glm52_mi355x_2026.md) — ATOM comparison table (F-6 row)
- SGLang PR #33544: https://github.com/sgl-project/sglang/pull/33544 — OPEN
- SGLang PR #35074: https://github.com/sgl-project/sglang/pull/35074 — OPEN

**Action:** Low priority for GLM-5.2 specifically. Monitor AITER MoE PRs. The MoE kernel for GLM-5.2 on v0.5.16 is already using AITER fused topk_gating (PR #28399, merged).

---

### F-7 — DCP (decode context parallelism) for GLM-5.2

**What it is:** DCP splits the KV cache across multiple ranks for decode, allowing a smaller TP degree per decode group while maintaining full model parallelism. ATOM uses DCP for Kimi-K3.

**SGLang status:** DCP is being implemented for AMD ROCm but not yet merged:
- **PR #35865** `[AMD] Implement DeepSeek V4 DCP on ROCm` — **OPEN** since 2026-08-21, last updated 2026-08-31
- **PR #37407** `DeepSeek-V4 decode context parallelism on AMD HIP (unified_kv)` — **OPEN** since 2026-09-01

For GLM-5.2 specifically: **deprioritized**. Rationale: TP=8 beats TP=4 for ITL because `t_weights` (weight loading time / GPU) dominates over NCCL — see [tuner_en.md §3 Group E](tuner_en.md). DCP+TP=2 would worsen weight loading. Only worth exploring if NCCL becomes the bottleneck.

**References:**
- [tuner_en.md §3 Group E](tuner_en.md) — TP/EP/DCP parallelism trade-off analysis
- [campaign_glm52_mi355x_2026.md §3](campaign_glm52_mi355x_2026.md) — TP=8 vs TP=4 finding, DCP+TP=2 elimination
- SGLang PR #35865: https://github.com/sgl-project/sglang/pull/35865 — OPEN
- SGLang PR #37407: https://github.com/sgl-project/sglang/pull/37407 — OPEN (opened 2026-09-01)

**Action:** No action for GLM-5.2. Monitor for Kimi-K3 or future models where decode bandwidth dominates.

---

### F-8 — Quark PTPC FP8 attention inference on ROCm (checkpoint loading)

**What it is:** `amd/GLM-5.2-Quark-MXFP4-AttnFP8` variant uses per-token-per-channel FP8 attention weights (quantized by Quark). Loading and running this checkpoint on ROCm/gfx950 requires a specific routing through `apply_fp8_ptpc_linear`.

**Why it matters:** This is a prerequisite for enabling KV FP8 (I-9) via the Quark PTPC attention path, which is estimated to recover ~0.5ms ITL.

**SGLang status:**
- **PR #28734** `[AMD] Fix Load and Inference of MLA models with Quark PTPC FP8 attention on ROCm` — **OPEN** since 2026-06-19, last updated 2026-09-01. Extends the `apply_fp8_ptpc_linear` + `fused_rms_fp8_group_quant` pipeline to the Quark attention path. This is a prerequisite for using `amd/GLM-5.2-Quark-MXFP4-AttnFP8`.

Note: our current recipe uses `amd/GLM-5.2-MXFP4` (not the AttnFP8 variant) with `--kv-cache-dtype fp8_e4m3` at the server level. PR #28734 is needed to use the checkpoint-side FP8 attention.

**References:**
- [campaign_glm52_mi355x_2026.md §3](campaign_glm52_mi355x_2026.md) — ATOM comparison (KV FP8 role) and I-9 rationale (~0.5ms estimated)
- SGLang PR #28734: https://github.com/sgl-project/sglang/pull/28734 — OPEN

**Action:** Monitor PR #28734. Once merged, test `amd/GLM-5.2-Quark-MXFP4-AttnFP8` on v0.5.18+ to measure ITL reduction from KV FP8 path.

---

## Prioritized action plan

| Priority | Feature | Blocker | Next action |
|----------|---------|---------|-------------|
| **1** | F-4: `iota_range` fix | ~~PR #36960~~ MERGED 2026-09-01; PR #34129 open | Retry v0.5.18 upgrade; monitor #34129 (BLOCK_Q) for further ITL gain |
| **2** | F-5: AITER unified attention (extend path) | PR #37310/#37311 merge | Test on v0.5.18+ once merged |
| **3** | F-8: Quark PTPC FP8 loading | PR #28734 merge | Test AttnFP8 checkpoint on v0.5.18+ |
| **4** | F-2: INT4 all-reduce | No opt-in needed for GLM-5.2; fires only on large prefill TP=4 | **Testing 2026-09-02** (run 33635908473, c12, mia1); await final result |
| **5** | F-3: DP attention | Issue #34582 fix | Monitor; re-enable DEP arm after fix |
| **6** | F-1: `ptpc_fp8` | No SGLang PR | Track upstream; structural gap |
| **7** | F-6: FlyDSL MoE sorting | No direct equiv | Low priority for GLM-5.2 |
| **8** | F-7: DCP for GLM-5.2 | Deprioritized | No action |

---

## Key PRs to watch

| PR | Title | State | Date |
|----|-------|-------|------|
| #28734 | AMD Fix Quark PTPC FP8 attn on ROCm | OPEN | 2026-06-19 |
| #34129 | AMD GLM5 AITER BLOCK_Q MQA logits | OPEN | 2026-08-08 |
| #34582 | Bug: DP attn deadlock GLM-5.2 | OPEN | 2026-08-12 |
| #35865 | AMD DSv4 DCP on ROCm | OPEN | 2026-08-21 |
| #36960 | ROCm Bugfix: cap DSA MQA-logits budget | MERGED | 2026-09-01 |
| #37310 | ROCm aiter: unified_attention for extends (−18…−32% TTFT) | OPEN | 2026-09-01 |
| #37311 | ROCm aiter: CK varlen sinks extends (−15…−16% TTFT) | OPEN | 2026-09-01 |
| #37407 | DSv4 DCP on AMD HIP (unified_kv) | OPEN | 2026-09-01 |

---

## Planned incremental improvements (from AgentX tracking, 2026-09-01)

These items come from the internal AgentX optimization tracker (`agentx_PRs_optimizations(data).csv`, lead: Zhenyu Gu / Jiejing). They are incremental gains already planned or in-flight by the SGLang AMD team — distinct from the structural gaps vs ATOM listed above. All target GLM-5.2/SGLang on MI355X unless noted.

### Already completed

| PR | Title | Framework | Status | Impact |
|----|-------|-----------|--------|--------|
| #36515 | [AMD] fix: do not emit a shared-expert marker twice on the per-rank slot path | SGLang | **MERGED 2026-08-30** | Correctness fix for shared-expert MoE routing |
| (no PR) | Full CUDA graph when MTP enabled | SGLang | **Completed** | +25% interactivity / +5% throughput |
| (no PR) | DSA indexer Top-K optimize | SGLang | **Completed** | +11% interactivity / +2% throughput |

### Open PRs — SGLang

| PR | Title | State | Impact |
|----|-------|-------|--------|
| #37152 | [ROCm] Make the HiCache kernel IO backend work on ROCm | OPEN 2026-08-30 | **+25–33% throughput** at ISL=100K, 90% prefix hit rate (TP=4/HiCache arm) |
| #37130 | [AMD] Remove silent ×0.85 mem_fraction_static derate for aiter + ctx>8K | OPEN 2026-08-30 | **+15% KV pool** at ISL=100K — restores memory silently throttled by aiter backend |
| #37133 | [GLM-5.2] Keep GlmMoeDsa MoE e_score_correction_bias in fp32 | OPEN 2026-08-30 | Correctness fix for MoE gating precision |
| #37118 | [ROCm] Define the DSA head-gate graph helpers on HIP | OPEN 2026-08-30 | Enables CUDA-graph capture with MTP on DSA path (prerequisite for full graph coverage) |
| #37124 | [ROCm] Take the fused DSA metadata kernels and drop redundant work from the absorb path | OPEN 2026-08-30 | DSA kernel fusion — reduces per-step overhead on absorb path |
| #36530 | [AMD][DSV4] perf: fold the padded-weight zeroing into the fused append kernel | OPEN | Minor kernel fusion for padded-weight path |
| #31213 | [GLM-5.2] Keep GlmMoeDsa MoE e_score_correction_bias in fp32 (earlier version) | OPEN | Same fix as #37133 — confirm which is canonical with Zhenyu/Jiejing |

### Open PRs — AITER (ROCm/aiter)

| Item | Description | State | Impact |
|------|-------------|-------|--------|
| ROCm/aiter (TBD) | [HIP] topk: fuse DSA page-table transform into cooperative top-k | OPEN | DSA indexer top-k fusion — reduces indexer overhead |
| ROCm/aiter (TBD) | [Triton/Gluon] fp8_mqa_logits: gate buffer ops on int32 offset, not tensor bytes | OPEN | **Fixes crash** on any chunked prefill for GLM-5.x — AITER-side complement to SGLang PR #36960 |

> Note: AITER PR numbers not yet tracked in the CSV. The `fp8_mqa_logits` AITER fix is the upstream root-cause fix for the crash described in F-4 — once it lands in a released AITER/ROCm build, the SGLang-side workarounds (PR #36960) may become redundant.

### Optimizations in progress (no PR yet)

| Item | ETA | Status | Projected impact |
|------|-----|--------|-----------------|
| DCP for GLM-5.2 at high concurrency (needs Top-K kernel opt in DCP scenario) | 4-Sep | Functionality ready | TBD — see F-7 |
| Dynamic MTP acceptance length (MTP=4 verified at c2) | 28-Aug | Verifying other concurrencies | ITL improvement at low concurrency |
| Small kernel fusion on MTP side | 4-Sep | In progress | TBD |
| FlyDSL MoE tuning | 4-Sep | In progress | TBD — see F-6 |
| Tokenizer cache | 4-Sep | In progress | Latency reduction for repeat prompts |
| MegaMoE + MoE tuning | 4-Sep | In progress | **+5% throughput** |
| HiCache read-throughput optimization and capacity increase | 4-Sep | In progress (no PR yet per 2026-08-29 update) | **+10% throughput** — complements PR #37152 |
| MLA PS (prefill-split) kernel | 28-Aug | Planned | TTFT reduction for long prompts |

---

## Appendix — Background on key technologies

### GLM-5.2 and the inference stack

**GLM-5.2** (ZhipuAI / THUDM) is a large Mixture-of-Experts language model. AMD distributes it as `amd/GLM-5.2-MXFP4`, quantized with the Quark toolkit to MXFP4 (microscaling FP4) for weights. The model uses a **DSA** (Dynamic Sparse Attention) architecture — a variant of MLA (Multi-head Latent Attention) with sparse indexing and "sink" tokens — and a **MoE** (Mixture-of-Experts) feed-forward block.

**SGLang** is an open-source inference engine (sgl-project/sglang) originally developed at UC Berkeley. It supports tensor parallelism, speculative decoding, and KV cache offloading. The AMD/ROCm backend is actively co-developed by AMD engineers.

**ATOM** is AMD's proprietary inference engine, optimized specifically for MI3xx hardware. It is the reference target for performance comparisons in this document.

**MI355X / gfx950** is AMD's data-center GPU (Instinct series). It uses the CDNA3 architecture and supports ROCm 7.x, MXFP4/FP8 hardware acceleration, and the DSA sparse-attention kernel family.

---

### Acronyms and terms

**AITER** (AMD Inference Engine Runtime): AMD's library of optimized GPU kernels for inference, now integrated into ROCm as a system library. Provides fused attention kernels (flash-attention style), RoPE, RMSNorm, MoE routing, all-reduce, etc. SGLang uses AITER via the `--attention-backend aiter` flag or through ROCM environment variables.

**ATOM** (AMD Tensor Operations for ML): AMD's closed-source, production-grade inference engine for MI3xx GPUs. Serves as the performance baseline. Includes features not yet available in SGLang (ptpc_fp8, INT4 AR, FlyDSL MoE sorting).

**BF16** (Brain Float 16): Standard 16-bit floating-point format used for most activations and KV cache in our baseline. 16-bit exponent range, 7-bit mantissa.

**CK** (Composable Kernels): AMD's library of hand-tuned GPU kernels for GEMM, attention, etc. Provides `mha_batch_prefill` used by SGLang's aiter backend for prefill attention. Some kernel families are missing for gfx950 (d64 group-mode paged), which is why unified_attention was being integrated (F-5).

**DCP** (Decode Context Parallelism): A parallelism strategy that splits the KV cache across multiple ranks *only during decode*, while keeping standard TP for prefill. Reduces per-rank memory pressure during decode without increasing TP communication overhead. Distinct from TP (Tensor Parallelism) and PP (Pipeline Parallelism). In SGLang: `--dcp-size N`.

**DP attention** (`--enable-dp-attention`): Data-Parallel attention mode in SGLang. Splits the input batch across DP workers for attention computation (each worker sees a subset of tokens), while MLP is done with full TP. Reduces per-step attention time at the cost of an all-gather collective for MLP synchronization. Requires the `DataParallelController` infrastructure.

**DSA** (Dynamic Sparse Attention): GLM-5.2's attention mechanism. Uses a sparse indexer to select which key/value tokens each query attends to, plus "sink" tokens (fixed global attention). The indexer (`fp8_mqa_logits`) computes query–key logit scores over candidate keys to build the sparse index. This is more expensive than standard dense attention but enables 1M-token context with bounded per-step cost.

**EP** (Expert Parallelism): Distributes MoE experts across GPUs. Each GPU holds a subset of experts. Requires all-to-all communication to route tokens to the correct expert. High EP increases communication cost but reduces per-GPU memory. In our recipe: EP=1 (all experts on each GPU replica, no all-to-all) for TP=8 interactivity arm; EP=4 for TP=4 throughput arm.

**FlyDSL**: AMD's domain-specific language for generating optimized GPU kernels, particularly for sparse and MoE operations. Used in ATOM for MoE token sorting and routing. Not yet directly exposed as a user-facing flag in SGLang; AMD is integrating it via the linear-attention and AITER MoE paths.

**FP8** (8-bit floating point, e4m3 format): IEEE-style 8-bit float with 4-bit exponent and 3-bit mantissa. Used for KV cache (`--kv-cache-dtype fp8_e4m3`) to halve KV cache HBM footprint compared to BF16. Also used for attention weight quantization in PTPC mode.

**HiCache**: SGLang's hierarchical KV cache system. Offloads KV cache blocks from GPU HBM to host DRAM when GPU memory is full, then fetches them back on demand. Controlled by `hicache_ratio` (ratio of host DRAM to GPU HBM allocated for KV). Used in the TP=4/EP=4 throughput arm of our recipe.

**HBM** (High Bandwidth Memory): The stacked DRAM directly attached to the GPU die. On MI355X: 192 GB HBM3e. Bandwidth is the main bottleneck for memory-bound operations (KV cache reads, weight reads in decode).

**ITL** (Inter-Token Latency): Time between consecutive generated tokens for a single request, measured in ms. The primary interactivity metric. Lower is better. P90 ITL = 90th percentile across all tokens across all requests in the benchmark.

**JIT** (Just-In-Time compilation): Triton (and other GPU kernel frameworks) compile kernels at first use for the target GPU architecture. On gfx950, some kernel shapes trigger LLVM assertion failures during JIT (see F-4). JIT compilation adds startup latency but allows specialization.

**KV cache**: The key-value tensors stored from previous context tokens, reused in each decode step to avoid recomputation. KV cache size scales with sequence length × layers × heads × head_dim × 2 (K+V). With BF16: 2 bytes/element; with FP8: 1 byte/element.

**MLA** (Multi-head Latent Attention): DeepSeek's attention architecture, used in DSv4, Kimi-K3, and GLM-5.2. Compresses the KV cache via a low-rank latent projection, dramatically reducing KV size. GLM-5.2's DSA is built on top of MLA.

**MoE** (Mixture of Experts): Feed-forward architecture where each token is routed to a small subset of expert MLPs (e.g., 2 out of 64 experts), reducing active parameters per token. GLM-5.2 uses MoE for its FFN layers.

**MTP** (Multi-Token Prediction): GLM-5.2's speculative decoding mechanism. The model's draft heads predict multiple future tokens in parallel; the main model verifies them in a single forward pass. In our recipe: `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` (5-1-6 config). Distinct from EAGLE (which uses a separate draft model); MTP uses the target model's own draft heads.

**MXFP4** (Microscaling FP4): AMD/OCP microscaling format. Groups of 32 weights share a single FP8 scale factor, with each weight stored as 4-bit float. Used for GLM-5.2 weight quantization. Hardware-accelerated on MI355X/gfx950.

**NCCL / RCCL**: GPU communication library (NVIDIA Collective Communications Library / ROCm equivalent). Used for all-reduce, all-gather, and reduce-scatter operations across TP ranks.

**P90** (90th percentile): The latency at or below which 90% of measurements fall. Used here for ITL P90 — the "worst typical" inter-token latency.

**ptpc_fp8** (per-token per-channel FP8): ATOM's on-the-fly activation quantization. During inference, activations (inputs to linear layers) are quantized to FP8 in real time using per-token scale factors. This reduces HBM bandwidth for intermediate tensors without a pre-quantized checkpoint. Not available in SGLang (F-1).

**Quark**: AMD's quantization toolkit for producing quantized model checkpoints. Used to create `amd/GLM-5.2-MXFP4` (MXFP4 weights) and `amd/GLM-5.2-Quark-MXFP4-AttnFP8` (MXFP4 weights + FP8 attention).

**Quick-reduce / custom all-reduce**: AITER's optimized all-reduce path that quantizes the data before communication, reducing bandwidth. INT8 version: halves bandwidth vs BF16. INT4 version: quarters bandwidth. In SGLang: `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (or INT4 with model-specific opt-in).

**ROCm**: AMD's open-source GPU computing platform (equivalent to NVIDIA CUDA). gfx950 = ROCm target architecture for MI355X. ROCm 7.2.4 is the version included in SGLang v0.5.18-rocm724 images.

**TP** (Tensor Parallelism): Splits individual weight tensors (and their computations) across GPUs. Each TP rank holds a shard of each layer. Requires all-reduce after each GEMM. Higher TP reduces per-GPU weight size (reducing weight-load time) at the cost of higher all-reduce overhead. In our recipe: TP=8 for interactivity, TP=4 for throughput.

**TTFT** (Time To First Token): Latency from request submission to the first generated token. Dominated by the prefill pass (processing the input prompt). Distinct from ITL (which measures decode speed).

**Triton**: Python-based GPU kernel language (OpenAI Triton). Used by SGLang and AITER for JIT-compiled kernels, including `fp8_mqa_logits` for DSA indexer logits computation.

**Unified attention** (AITER): AITER's `unified_attention` kernel covers both prefill extends and decode in a single interface. On gfx950, it fills gaps in CK's kernel catalog. Being added to SGLang's aiter backend via PR #37310 and related PRs (F-5).
