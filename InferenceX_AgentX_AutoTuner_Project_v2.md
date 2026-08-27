# InferenceX AgentX Auto-Tuner
## GLM-5.2 on MI355X with SGLang

**Status:** Active development
**Model:** GLM-5.2 (FP4)
**Framework:** SGLang (solo — ATOM rimosso)
**Hardware:** 8× MI355X (mia1-p01-g07)
**Scenario:** agentic-coding
**Goal:** Pareto-optimal serving configurations per latency/interactivity/throughput, minimizzando il tempo di tuning.

---

# 0. Working environment

```
Host:    giovanni.guasti@amd.com@mia1-p01-g07  (ProxyJump 64.139.223.124)
Workdir: /it-share/gguasti
Model:   /it-share/models/GLM-5.2-MXFP4  (408 GB, NFS)
Runner:  /it-share/gguasti/actions-runner  (GitHub Actions: mi355x-amds_03)
```

Docker images disponibili sulla macchina:
```
lmsysorg/sglang-rocm:v0.5.18-rocm720-mi35x-20260824  ← target primario
lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260824  ← challenger ROCm 7.2.4
```

Config recipe: `configs/amd-master.yaml` · key: `glm5.2-fp4-mi355x-sglang-agentic-mtp`
Script: `benchmarks/single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh`

---

# 1. Risultati baseline (2026-08-25)

**Upstream PR:** [SemiAnalysisAI/InferenceX#2570](https://github.com/SemiAnalysisAI/InferenceX/pull/2570) — merged 2026-08-12
(Tuning follow-up a #2488: MTP steps 3→5, draft tokens 4→6, SGLANG_SIMULATE_ACC_LEN 2.99→3.61; TP8 low-conc: hicache rimosso)

**Run:** 32867048910 · branch `testgg` · runner `mi355x-amds_03`
**Image:** v0.5.16-rocm720 · TP4/EP4 · CONC=10 · MRR=10 · chunked=32768 · HiCache ratio=1.5 · EAGLE 5-steps/6-tokens

| Metrica | Valore |
|---------|--------|
| Durata | 3601s |
| Richieste completate | 1405 |
| Throughput totale | 38,337 tok/s |
| **Throughput/GPU** | **9,584 tok/s/GPU** |
| Output throughput | 367 tok/s |
| TTFT mean / p50 / p90 | 1.341s / 0.577s / 3.500s |
| ITL mean / p50 / p90 | 14.82ms / 11.86ms / 18.42ms |
| Interattività mean / p50 / p90 | 67.5 / 84.3 / 54.3 tok/s/user |
| GPU KV cache hit | 88.9% |
| CPU KV cache hit | 0.4% |
| KV GPU usage | 61% |
| **KV CPU usage** | **100% (saturato)** |

> **Nota:** il CPU KV pool è al 100% — il tier host DRAM HiCache è saturo. Potenziale bottleneck per CONC più alti.

Risultati storici completi: `progress.csv`

---

# 2. Obiettivi di ottimizzazione (Pareto)

Il campo di valutazione è suddiviso in **tre regimi** distinti per CONC:

| Regime | CONC indicativo | Metrica primaria | Metrica secondaria |
|--------|-----------------|------------------|--------------------|
| **Interactivity** | 1 – 6 | P90 tok/s/user (maximize) | P90 TTFT (minimize) |
| **Crossover** | 6 – 12 | Crossover CONC C\* (maximize) | tok/s/GPU @ C\* (maximize) |
| **Throughput** | 12 – 24 | tok/s/GPU (maximize) | P90 ITL (minimize) |

**Definizione operativa di C\*:** il CONC minimo in cui `tok/s/user` scende sotto l'80% del valore misurato a CONC=1. Configurazioni con C\* alto mantengono l'interattività per più utenti simultanei.

Minimizzare/massimizzare simultaneamente (Pareto tri-obiettivo):

| Obiettivo | Regime | Direzione |
|-----------|--------|-----------|
| P90 interattività @ CONC=4 (tok/s/user) | Interactivity | maximize |
| Crossover CONC C\* | Crossover | maximize |
| Throughput/chip @ CONC=16 (tok/s/GPU) | Throughput | maximize |

Configurazione `A` domina `B` se migliora almeno uno dei tre obiettivi senza peggiorare gli altri.

> **Implicazione per il sweep:** ogni server lifetime deve campionare CONC in modo denso nella zona crossover (es. [4, 6, 8, 10, 12]) per localizzare C\* con precisione, non solo i punti estremi.

---

# 3. Vincolo architetturale SGLang

**SGLang non supporta mutation runtime** dei parametri scheduler. Tutti i knob seguenti sono fissi all'avvio del server:
- `--chunked-prefill-size`
- `--max-running-requests`
- `--cuda-graph-max-bs`
- `--mem-fraction-static`
- `--speculative-*`
- `--hicache-*`

**Unica variabile client-side (no restart):** CONC (concorrenza).

**Implicazione per il tuner:** un server lifetime → sweep di più CONC. Per ogni configurazione strutturale diversa: restart obbligatorio.

```
Per ogni config C (chunked-prefill, max-running-requests, ...):
  → avvia server (1 restart, ~15-20 min warmup)
  → misura CONC = [4, 6, 8, 10, 12] sequenzialmente
  → ferma server
  = risparmio: 1 restart × N_conc invece di N_conc restart
```

---

# 4. Spazio dei parametri

## Gruppo A — Solo client (nessun riavvio)

| Parametro | Range | Effetto atteso (+) | Rischio (−) | Dipendenze |
|-----------|-------|--------------------|-------------|------------|
| **CONC** | 1, 2, 4, 8, **10**, 12, 16, 24 | Maggiore utilizzo GPU, throughput più alto | ITL peggiora, TTFT aumenta, OOM a CONC alti | `max-running-requests ≥ CONC`; `HICACHE_RATIO` da ricalibrate a CONC alti |
| **DURATION** | 120s (smoke), 600s (tuning), 3600s (full) | — | P90/P99 inaffidabili sotto ~100 richieste | — |

## Gruppo B — Scheduler / Prefill (riavvio)

| Parametro | Range | Effetto atteso (+) | Rischio (−) | Dipendenze |
|-----------|-------|--------------------|-------------|------------|
| **`--chunked-prefill-size`** | 8192, 16384, **32768**, 65536, 131072 | Valori bassi: decode interleave più spesso → ITL e interattività migliorano. Valori alti: prefill efficiente, throughput input | 131072 → OOM osservato (run 29751563205). Valori bassi: overhead scheduling | `mem-fraction-static`: 131k chunk richiede ~7 GiB/rank headroom vs ~1.7 GiB/rank a 32k |
| **`--max-running-requests`** | `0.5×C`, **`1×C`**, `1.5×C`, `2×C`, `3×C` | Più richieste in volo → throughput più alto se server non è bottleneck | Troppo alto: KV cache esaurita, OOM. ITL peggiora | Deve essere ≥ CONC; `cuda-graph-max-bs` ≥ `max-running-requests` |
| **`--cuda-graph-max-bs`** | `1×MRR`, **`1×MRR`**, `1.5×MRR`, `2×MRR` | Riduce "graph capture miss" con EAGLE draft+verify | Più HBM per grafi, startup più lento | Con EAGLE num-steps=5 il batch draft può temporaneamente superare `max-running-requests` |
| **`--mem-fraction-static`** | 0.80, 0.83, **0.85**, 0.87 | Più HBM alla KV → hit rate più alto | Lascia meno headroom attivazioni. OOM con chunked-prefill alto | Interagisce con `chunked-prefill-size` e `HICACHE_RATIO` |

## Gruppo C — Speculative Decoding EAGLE (riavvio)

| Parametro | Range | Effetto atteso (+) | Rischio (−) | Dipendenze |
|-----------|-------|--------------------|-------------|------------|
| **MTP on/off** | **EAGLE** / off | ON: acceptance length ~3.6 → throughput decode 2-3× | Draft overhead aumenta ITL per richieste corte | `cuda-graph-max-bs` deve coprire batch draft |
| **`--speculative-num-steps`** | 3, 4, **5** | Più step → più token accettati → throughput più alto | Overhead cresce linearmente; a CONC alti draft compete per HBM | Golden AL osservato: 3.61 (simulato) |
| **`--speculative-num-draft-tokens`** | 3, 4, **6** | Più draft = più accept potenziali | Se acceptance rate scende, costo senza beneficio | `num-steps × num-draft-tokens` = dimensione batch draft max |
| **`--speculative-eagle-topk`** | **1**, 2 | topk>1: sampling più diversificato | Overhead quadratico, non testato su GLM-5.2 | Da esplorare solo dopo stabilizzazione degli altri parametri |

## Gruppo D — KV Cache / HiCache (riavvio)

| Parametro | Range | Effetto atteso (+) | Rischio (−) | Dipendenze |
|-----------|-------|--------------------|-------------|------------|
| **KV offload backend** | none, **hicache**, mooncake | HiCache: supporta CONC alti con contesti lunghi senza OOM | none: massima velocità. HiCache: bandwidth CPU-GPU bottleneck. Mooncake: infrastruttura aggiuntiva | — |
| **`HICACHE_RATIO`** | 0.5, 1.0, **1.5** (TP arm), 2.0 | Più DRAM host → hit rate più alto a CONC alti | OOM host a ratio troppo alti (osservato: CONC 48 con DP-arm). **CPU pool già al 100% a CONC=10** → valutare aumento | Device pool per rank: ~182.7 GB a TP4 |
| **`HICACHE_WRITE_POLICY`** | **write_through**, write_back | write_through: massimo hit rate | Più bandwidth CPU-GPU durante prefill | — |
| **`HICACHE_IO_BACKEND`** | **direct**, asyncio | direct: latenza bassa | asyncio: potenzialmente meglio su I/O intenso | — |
| **`--kv-cache-dtype`** | **fp8_e4m3**, bf16 | fp8: 2× meno HBM → più token cached | bf16: qualità KV migliore | Strutturale: invalida cache |
| **Mooncake L3** (`L3_PER_RANK_GB`) | 20, **40**, 60, 80 | Più L3 → hit rate su contesti molto lunghi | Richiede infrastruttura mooncake | Solo con `KV_OFFLOAD_BACKEND=mooncake` |

## Gruppo E — Parallelismo / Topologia (riavvio)

| Parametro | Range | Effetto atteso (+) | Rischio (−) | Dipendenze |
|-----------|-------|--------------------|-------------|------------|
| **TP** | **4**, 8 | TP8: meno HBM/rank → più KV cached | Più overhead RCCL, latency collectives | `HICACHE_RATIO` da ricalibrate per rank |
| **EP** | 2, **4**, 8 | EP alto: meno HBM/rank per expert weights | Communication overhead routing esperti | Deve dividere numero esperti GLM-5.2 |
| **DP Attention** | **OFF**, ON | ON: KV/rank ridotta, scalabile a CONC alti | **ROTTO su SGLang ROCm** (hang collective su long-context prefill, v0.5.14 e v0.5.16). Re-enable solo dopo fix upstream confermato | `SGLANG_DP_USE_GATHERV=1`, `SGLANG_DP_USE_REDUCE_SCATTER=1` |
| **DCP** | **OFF**, ON | Separa prefill/decode → decode non bloccato | Infrastruttura complessa. Rilevante per Kimi-K3, non esplorato per GLM-5.2 | Seconda fase |

## Gruppo F — Piattaforma Docker (restart, costo alto)

| Parametro | Valori | Note |
|-----------|--------|------|
| **Docker image** | v0.5.16-rocm720 (baseline), **v0.5.18-rocm720** (prossimo target), v0.5.18-rocm724 (challenger) | Trattare come variabile di piattaforma. A/B test controllato. ROCm 7.2.4 modifica il path HIP graph che impatta chunked-prefill e CUDA graph coverage |

---

# 5. Piano di tuning (fasi)

## Fase 0 — Infrastruttura (prerequisiti)
- [x] Runner `mi355x-amds_03` su mia1-p01-g07 funzionante
- [x] Baseline v0.5.16-rocm720 CONC=10: 9,584 tok/s/GPU (TP4/EP4/HiCache)
- [x] Recipe aggiornata: arm TP=8/EP=8/no-hicache aggiunto in `amd-master.yaml`
- [x] Fix `_work` root-owned — Alpine chown nel "Resource cleanup (pre-run)" del workflow (commit 4b33bfe98)
- [x] **Rollback immagine v0.5.18 → v0.5.16** (commit 4786948a0): v0.5.18 crash SIGABRT in `fp8_mqa_logits` (Triton LLVM `iota_range` assertion) su TP=4+HiCache+DSA e su TP=8. Confermato regression vs v0.5.16.
- [ ] Fix `test-sweep-evals` fromJson empty (workflow cosmetic failure)

---

## Fase 1 — Ottimizzazione Interactivity

### Analisi overhead a bassa concorrenza

A CONC piccola, il ciclo decode è dominato da overhead fissi non ammortizzati:

```
ITL/utente ≈ (t_nccl_allreduce + t_kernel_launch + t_weights) / CONC
```

- `t_nccl` è ~costante per batch piccole ma scala col numero di GPU nell'allreduce
- `t_weights` scala con 1/TP (meno peso per GPU = caricamento più rapido)
- Il sweet spot è dove l'ammortizzazione è quasi completa ma la coda è ancora assente

**Perché C=1 è peggio di C=4:** overhead fisso (NCCL, kernel launch, 5+1 MTP cycles) non si ammortizza su 1 sola richiesta. A C=4, lo stesso overhead è diviso tra 4 utenti e `t_weights` è quasi uguale (regime memory-bound).

### Trade-off dimensione TP

| TP | NCCL ranks | Pesi/GPU | HBM per KV | Regime ottimale |
|----|-----------|----------|------------|-----------------|
| 2  | 2 → velocissimo | ~204 GB | ~84 GB ⚠ | non pratico con ISL=79k token |
| 4  | 4 → medio | ~102 GB | ~186 GB ✓ | throughput, CONC ≥ 10 |
| 8  | 8 → lento | ~51 GB  | ~237 GB ✓ | candidato interactivity |

**TP=2:** NCCL ottimale ma KV budget insufficiente per ISL p50=79k token (rimane ~84 GB/GPU → ~1 request attiva per gruppo). Non pratico senza ridurre drasticamente il contesto.

**TP=2 + DCP=4** (8 GPU totali, 4 gruppi TP=2 indipendenti): idea interessante — NCCL tra 2 GPU ma usando tutte le 8 GPU. Da esplorare **solo** dopo aver dati TP=4 vs TP=8: se TP=8 peggiora ITL rispetto TP=4 (NCCL domina), allora DCP+TP=2 è candidato forte. Se TP=8 migliora (t_weights domina), DCP non aiuta.

**HiCache e ITL:** il write-through è asincrono e non impatta direttamente l'ITL nel decode path. Il rischio è contention PCIe quando `cpu_kv_usage=99.7%` (eviction continua). TP=4 senza HiCache è un esperimento pulito per isolare questo effetto (vedi I-5).

### Step I-0 — Baseline run corrente
- **Run 32947505370:** branch `testgg-maxreq2x`, immagine v0.5.16-rocm720-mi35x-20260728
- TP=4/EP=4/HiCache/c10 + TP=8/EP=8/no-hicache/c[4,6,8,10] in parallelo
- **Risultati:** da compilare

### Config baseline TP=8
```
TP=8, EP=8, kv-offloading: none
chunked=32768, MEM_FRACTION=0.85
MRR = 1×CONC
EAGLE: 5 steps / 6 tokens / SGLANG_SIMULATE_ACC_LEN=3.61
```

### Priorità esperimenti (aggiornata con dati c4/c6)

**Finding chiave:** TP=8/no-HiCache domina per interactivity (-39% ITL vs TP=4). Bottleneck = `t_weights` (peso per GPU), non NCCL.
**Strategia:** massimizzare KV GPU-resident + minimizzare overhead per-step → TP=8, EP ridotto, CONC basso.

#### 🔴 Alta priorità — completare baseline + EP=1

| # | Esperimento | Azione | Attesa | Note |
|---|-------------|--------|--------|------|
| **I-0b** | Baseline TP=8/EP=8 c8+c10 | Run in corso (32947505370) | C\* di TP=8/EP=8 | — |
| **I-8** | **TP=8/EP=1** | Nuovo run dopo I-0b | **ITL p50 ↓↓** | Elimina MoE all-to-all a bassa CONC — vedi sotto |

**Razionale I-8 — TP=8/EP=1:**
Con TP=8/EP=8 ogni forward MoE introduce un **all-to-all** tra 8 rank EP per distribuire i token agli expert corretti. A CONC=4 (effective=1 dal dato reale), quasi tutti i token vanno agli stessi 1-2 expert attivi — l'all-to-all è overhead puro che non si ammortizza.

Con **EP=1** (nessun expert parallelism, solo TP=8):
- Ogni GPU ha tutti gli expert replicati → zero all-to-all nel forward MoE
- Elimina un'intera collettiva per ogni layer MoE
- Contro: più pesi per GPU → meno HBM per KV cache

Verifica memoria prima di lanciare:
```
TP=8/EP=8: ~51 GB pesi/GPU → ~237 GB KV/GPU   ✓
TP=8/EP=1: più expert/GPU → da verificare se entra in 288 GB HBM
```
Fonte pattern: PR #2693 (TP=2/EP=2 → TP=2/EP=1 per Qwen3.5 MI355X, stesso motivo).

#### 🟠 Media priorità — fix harness TP=8 (da PR #2737)

Tre fix mancanti nel nostro script, derivati da PR #2737 (Qwen3.5 MI355X, stesso stack, 2026-08-26). Lanciabili insieme in un solo run dopo I-8.

| # | Esperimento | Parametro | Baseline | Candidato | Ipotesi | Metrica target |
|---|-------------|-----------|----------|-----------|---------|----------------|
| **I-1** | Chunk più piccolo | `chunked-prefill-size` | 32768 | **16384** | PR #2737: sibling B200; interleave decode più frequente → ITL ↓ | ITL p50 ↓ |
| **I-2** | CUDA graph max BS | `CUDA_GRAPH_MAX_BS` | `1×CONC` | **`min(2×CONC, 128)`** | PR #2737: batch > CONC cadono su eager path. Fix critico quando MRR > CONC. | ITL p50 ↓ |
| **I-3** | All-reduce INT8 | `ROCM_QUICK_REDUCE_QUANTIZATION` | non settato | **`INT8`** | PR #2737: cookbook MI355X MXFP4; riduce latency collettive NCCL | ITL p50 ↓ |
| **I-7** | AITER unified attn | `ROCM_AITER_UNIFIED_ATTN` | non settato | **`1`** | AITER integrato in ROCm; provare su v0.5.16 (se ignorato → nessun danno); pieno effetto su v0.5.18 stabile | ITL p50 ↓ |

> **Nota I-2:** va implementato prima o insieme a I-6 (MRR pipeline). Con MRR=1×CONC, impatto limitato; con MRR=1.5×CONC, diventa critico.

**Ordine:** I-1+I-2+I-3+I-7 insieme → I-4 → I-5 → I-6.
**Stop condition:** salto se ΔP90 tok/s/user < 5%.

#### 🟢 Dopo i fix harness — ulteriore ottimizzazione TP=8

| # | Esperimento | Parametro | Baseline | Candidato | Ipotesi | Metrica target |
|---|-------------|-----------|----------|-----------|---------|----------------|
| **I-4** | Più HBM per KV | `mem-fraction-static` | 0.85 | **0.87** | Headroom senza HiCache → più KV in GPU → TTFT ↓ | TTFT p50 ↓ |
| **I-5** | EAGLE più profondo | `speculative-num-steps` | 5 | **6** | Effective CONC=1 a c4 → risorse dedicate → AL reale > 3.61 | tok/s/user ↑ |
| **I-6** | Pipeline prefill | `MRR` | 1×CONC | **1.5×CONC** | Sovrappone prefill/decode → C\* ↑ | C\* ↑ |

#### 🟡 Bassa priorità — throughput arm TP=4

| # | Esperimento | Note |
|---|-------------|------|
| **H-1** | HiCache `write_through_selective` | PR #2679; riduce write inutili su DRAM → meno eviction → C\* ↑ |
| **H-2** | HiCache ratio 1.5 → 2.0 | cpu_kv_usage=100% a c10 → più DRAM → C\* ↑ |
| **H-3** | HiCache `page_size=1` | PR #2679; granularità fine → matching prefix più preciso |
| **3a** | MRR 1×→2×CONC (TP=4) | Branch testgg-maxreq2x già pronto |
| **I-5b** | TP=4/no-hicache | Diagnostico: isola overhead HiCache; ridotta priorità con TP=8 disponibile |

#### 🔵 Riferimento: ATOM (PR #2576 — merged)

ATOM è un inference engine AMD-native (non SGLang) che ha già eseguito lo stesso workload GLM-5.2 FP4 MI355X con risultati significativi in interactivity. L'analisi è utile per validare le nostre scelte e identificare gap.

**Script:** `benchmarks/single_node/agentic/glm5.2_fp4_mi355x_atom_mtp.sh`
**Config:** TP=4 c[2,4,8,10] con LMCache DRAM + TP=8 c[1,2,4] GPU-resident
**Run validazione:** [31765309673](https://github.com/SemiAnalysisAI/InferenceX/actions/runs/31765309673)

| Parametro | SGLang (nostro baseline) | ATOM (PR #2576) | Rilevanza |
|-----------|--------------------------|-----------------|-----------|
| **Engine** | SGLang | `atom.entrypoints.openai_server` | — |
| **Online quant** | nessuna | **`ptpc_fp8`** (attivazioni → FP8 on-the-fly) | SGLang non supporta |
| **KV cache dtype** | BF16 | **FP8** → 2× meno HBM | Abilita TP=8 GPU-resident a c4 |
| **All-reduce quant** | non settato | **`AITER_QUICK_REDUCE_QUANTIZATION=INT4`** | Più aggressivo di I-3 (INT8) |
| **CUDA graph sizes** | `CUDA_GRAPH_MAX_BS=1×CONC` | **per-CONC espliciti** `[1,2,4,8,12,16,20]` | Valida I-2 (2×CONC) |
| **max-num-batched-tokens** | 32768 | **16384** | Valida I-1 (chunk piccolo) |
| **max-num-seqs** | 1×CONC | **2×CONC** | Valida I-2 (MRR 2×) |
| **MoE sorting** | default | `AITER_USE_FLYDSL_MOE_SORTING=1` | SGLang non supporta |
| **TP=8 KV offload** | DRAM HiCache | **GPU-resident** (FP8 → entra in HBM) | Stesso obiettivo I-8 |
| **KV offload backend** | HiCache | LMCache | — |

**Insight chiave da ATOM:**
- **FP8 KV** è il motivo per cui TP=8 GPU-resident funziona a c4: dimezza l'uso HBM KV, permettendo di tenere più token in GPU senza offload. SGLang supporta `--kv-cache-dtype fp8_e4m3` → da aggiungere come **I-9** (bassa priorità finché TP=8/EP=1 non è validato).
- **CUDA graph per-CONC** conferma che la nostra I-2 (`min(2×CONC,128)`) è la direzione giusta — ATOM lo fa in modo ancora più granulare.
- **`max-num-batched-tokens=16384`** conferma I-1.
- **INT4 all-reduce** non è disponibile in SGLang via `ROCM_QUICK_REDUCE_QUANTIZATION` (solo INT8) — differenza strutturale di ATOM.

**Dati misurati — confronto ATOM vs SGLang (run 31765309673 vs 31006984179 vs nostri):**

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
| **Nostro TP=8/c4** (EP=8, no KV) | 2026-08-26 | **7.3ms** | 9.6ms | **105** ✓ | 3.61 |
| **Nostro TP=8/c6** (EP=8, no KV) | 2026-08-26 | **8.3ms** | 11.8ms | **85** ✓ | 3.61 |
| **Nostro TP=8/c8** (EP=8, no KV) | 2026-08-26 | **8.8ms** | 12.2ms | **82** ✓ | 3.61 |
| **Nostro TP=8/c10** (EP=8, no KV) | 2026-08-26 | **9.5ms** | 14.9ms | **67** ✓ | 3.61 |
| **Nostro TP=4/c10** (HiCache) | 2026-08-26 | **11.9ms** | 19.4ms | **~52** ✓ | 3.61 |

**Finding corretto:** ATOM TP=8/EP=1/c4 ha P90=123 vs nostri EP=8/c4 P90=105 → ATOM è +17% su P90 intvty. Su ITL p50: ATOM 5.9ms vs nostri 7.3ms → -19%. Il vantaggio ATOM è reale e si spiega con EP=1 (niente all-to-all MoE) + ptpc_fp8 + KV FP8 + INT4 AR. Nota: AL diverso (2.99 vs 3.61) non incide su P90 intvty che è basato su ITL, non su tok/s/user da acceptance.

**Gap da colmare:** ~1.4ms ITL e ~18 tok/s/user P90 a c4. Obiettivo I-8 (EP=1) + I-3 (INT8 AR) + I-1 (chunk 16384).

> **Conclusione:** ATOM valida le nostre I-1, I-2, I-3. Il gap ITL principale verso ATOM è `ptpc_fp8` (attivazioni on-the-fly) + KV FP8 — ottimizzazioni engine-level non portabili direttamente su SGLang. I-9 (KV FP8 in SGLang) può recuperare parte del vantaggio HBM (~0.5ms stimato). Il gap residuo (~0.9ms) è strutturale tra i due engine.

#### ⚫ Sospesi / condizionali

| # | Esperimento | Condizione |
|---|-------------|------------|
| ~~**DCP+TP=2**~~ | Eliminato | t_weights domina — TP=2 peggiorerebbe |
| **P-1** | v0.5.18-rocm720 retry | Solo dopo fix Triton LLVM `iota_range` upstream confermato |
| **P-2** | v0.5.18-rocm724 | Alternativa a P-1 — potrebbe contenere il fix |
| **F-5** | DP Attention | Solo dopo fix upstream ROCm |

### Tabella risultati

> **Finding:** TP=8 batte TP=4 per ITL (-39% a c4). Domina `t_weights` (peso/GPU), non NCCL → **I-6 DCP+TP=2 deprioritizzato** (TP=2 caricherebbe più pesi/GPU, peggiorerebbe).

Colonne riferimento: CONC=10 per TP=4 (throughput arm); CONC=4/6 per TP=8 (interactivity arm).

> **Nota metriche:** `P90 intvty` = `intvty.p90` dal JSON = 1/ITL_p90 = valore che il 90% degli utenti supera (floor garantito). Le p90 intvty scendono al crescere di CONC perché la coda dell'ITL peggiora.

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
| Bundle c6 | 6 | *(in corso)* | — | — | — | — |
| I-9: KV FP8 | 4-8 | — | — | — | — | — |
| ~~I-6: TP=2+DCP=4~~ | — | deprioritizzato | | | | |

**Finding aggiornato:** C\* per TP=8/EP=8 è tra c8 e c10 — P90 intvty cala da 82 (c8) a 67 (c10), -18%. Throughput continua a crescere (+18% output tok/s/GPU). ATOM TP=8/EP=1/c4 ha P90=123 vs nostri 105 → gap del 17% da colmare con I-8 (EP=1) + I-3 (INT8 AR).

**I-8 finding (2026-08-26):** EP=1 migliora soprattutto a bassa CONC, come atteso:
- **c4: P90=110.5 (+5.2% vs EP=8=105)** — guadagno maggiore, all-to-all era overhead a bassa CONC. ITL p50 6.95ms (-4.8%), ITL p90 9.05ms (-5.7%). kv_usage=17%.
- c6: ITL p50 7.87ms (-5.2%), P90 82.3 (-3%) — quasi invariata rispetto a EP=8=85
- c10: ITL p50 9.27ms (-2.6%), P90 70.2 (+4.8%) vs EP=8=67
- **Pattern:** beneficio EP=1 decresce con CONC (5.2% → -3% → +4.8%). A c4 è il massimo; a c6+ la pressione decode/scheduler domina e EP=1 da solo non basta.
- **Gap residuo vs ATOM c4:** P90 110.5 vs 123 → ancora -10%. Attesa dal bundle.

**Bundle I-1+I-3+I-7 finding (2026-08-27):** nessun guadagno aggiuntivo rispetto a EP=1 puro — risultati praticamente identici su tutti i CONC misurati:
- c4: P90 110.4 (vs EP=1 110.5, vs baseline 105) — nessun delta
- c8: P90 81.0 (vs EP=1 81.1, vs baseline 82) — nessun delta
- c10: P90 68.8 (vs EP=1 70.2, vs baseline 67) — nessun delta
- **Conclusione:** I-1 (chunk 16384), I-3 (INT8 AR), I-7 (AITER unified attn) sono tutti neutri su v0.5.16-rocm720. Il bottleneck non è il chunk size né l'all-reduce né l'attention kernel — è strutturale all'engine SGLang su questo stack.
- **Gap vs ATOM c4 rimane -10% (P90 110 vs 123).** Il delta residuo è spiegato da ptpc_fp8 (attivazioni FP8 on-the-fly, non portabile) + CUDA graph per-CONC espliciti (ATOM li compila a ogni avvio, SGLang usa batch size fisso). Questo gap è probabilmente il limite di SGLang v0.5.16 su MI355X con GLM-5.2 MXFP4.
- **Prossimo step:** valutare se passare a v0.5.18 (con fix AITER e nuovi kernel ROCm) può chiudere il gap residuo. Oppure accettare P90≈110 a c4 come ottimum SGLang e concentrarsi sulla curva throughput (C\*).

<details>
<summary>Dettaglio Baseline TP=4/c10 — run 32947505370 (v0.5.16, ~72 min, 1407 req)</summary>

| | p50 | p90 | p99 |
|--|-----|-----|-----|
| ITL (ms) | 11.9 | 19.4 | 49.7 |
| TTFT (ms) | 509 | 1,592 | 8,683 |
| tok/s/user decode | 84 | 114 | 153 |
| tok/s/user E2E | 69 | 98 | — |
| Request latency (ms) | 5,220 | 28,960 | 113,717 |

- **Throughput:** 363 tok/s decode totale → **91 tok/s/GPU** (4 GPU)
- **Effective CONC p50=5** su 10 impostato (trace lunghe → prefill pesante)
- **Prefix cache hit:** 95.9% · `cpu_kv_usage`≈100%
- **EAGLE:** Accept Length=3.61 (=simulate target) · Accept Rate=52.2%
- **ISL** p50=90k token · **OSL** p50=335 token
- Totale: 138M tok input / 1.3M tok output
</details>

---

## Fasi 2-5 — Dettaglio esperimenti bassa priorità

### HiCache tuning (TP=4, regime CONC ≥ 10)

cpu_kv_usage=100% a CONC=10. Parametri validati da PR #2679 come riferimento.

| # | Parametro | Baseline | Candidato | Motivazione |
|---|-----------|----------|-----------|-------------|
| **H-1** | `HICACHE_WRITE_POLICY` | `write_through` | `write_through_selective` | PR #2679: scrive DRAM solo su prefill ad alta riusabilità → meno bandwidth wasted |
| **H-2** | `HICACHE_RATIO` | 1.5 | **2.0** | CPU pool saturo → più DRAM potrebbe alzare C\* |
| **H-3** | `page_size` | default | **1** | PR #2679: granularità fine → matching cache più preciso |
| **H-4** | `HICACHE_IO_BACKEND` | `direct` | `asyncio` | Solo se I/O DRAM è bottleneck a CONC alti |

**Ordine:** H-1 → H-2 → H-3 → H-4 solo se plateau.

### MRR sweep TP=4 (branch testgg-maxreq2x già pronto)

```
3a: MRR=10 (baseline) vs MRR=20 (2×CONC) ← branch già pronto
3b: chunked-prefill-size: 16384 / 32768 (baseline) / 65536
3c: cuda-graph-max-bs: 1×MRR / 1.5×MRR / 2×MRR
```

### Piattaforma e parallelismo avanzato

| # | Esperimento | Condizione |
|---|-------------|------------|
| **P-1** | v0.5.18-rocm720 retry | Solo dopo fix Triton LLVM `iota_range` upstream confermato |
| **P-2** | v0.5.18-rocm724 (ROCm 7.2.4) | Alternativa a P-1 — potrebbe avere il fix |
| **P-3** | AITER unified attention | AITER ora integrato in ROCm; esplorare `ROCM_AITER_UNIFIED_ATTN` quando stabile per DSA/GLM-5.2 |
| **F-5** | DP Attention | Solo dopo fix upstream ROCm |

---

# 6. Primo passo concreto (Milestone 1)

## Obiettivo
Sweep `max-running-requests` (1×C vs 2×C) su v0.5.18-rocm720, con multi-CONC per server lifetime.

## Steps
1. **Fix `_work` pre-cleanup** nel launcher (aggiunge auto-cleanup con docker prima del checkout)
2. **Aggiorna image** nella recipe a v0.5.18-rocm720
3. **Lancia `testgg-maxreq2x`** (MRR=20): legge risultati su CONC=10
4. **Aggiungi CONC multi-sweep**: modifica la recipe per testare CONC=[4, 6, 8, 10, 12] nello stesso server lifetime (zona crossover densa)
5. **Confronta** baseline (MRR=10) vs maxreq2x (MRR=20) su metriche chiave

## Codice tuner (bozza struttura)
```
utils/autotune/
  config.py        — parametri + metadata (restart_required, range, deps)
  runner.py        — lancia job GH Actions, aspetta risultato, legge agg_bmk.json
  pareto.py        — dominance test, frontier tracking
  storage.py       — aggiorna progress.csv
  cli.py           — entry point
```

Il tuner nella prima versione è un **orchestratore di job GH Actions**, non un controller diretto del server SGLang (impossibile con l'infrastruttura attuale).

---

# 7. Parametri rimossi / fuori scope (per ora)

- **ATOM**: fuori scope per esecuzione diretta (framework diverso). Usato come **reference** per validare scelte di tuning — vedi sezione 🔵 in Fase 1. KV FP8 (I-9) è l'unica ottimizzazione ATOM portabile su SGLang.
- **Kimi-K3**: seconda fase dopo GLM-5.2 stabilizzato.
- **DCP**: solo Kimi-K3 nella configurazione attuale.
- **Mooncake L3**: solo dopo HiCache L2 caratterizzato.
- **Bayesian optimization**: solo dopo infrastructure e multi-fidelity validati.

---

# 8. Multi-fidelity

| Tier | Durata | Uso |
|------|--------|-----|
| Screening | 120–300s | Eliminare configurazioni chiaramente peggiori |
| Tuning | 600–1200s | Ranking affidabile dei candidati |
| Validazione | 3600s (full) | Solo configurazioni Pareto-candidate |

Requisiti minimi: ≥ 100 richieste completate per P90 affidabile.

---

# 9. Problemi noti

Vedi `improvements.md` per backlog completo.

**Critico:** `_work/` resta root-owned dopo run falliti (NFS root_squash, no sudo su mia1-p01-g07).
Workaround manuale:
```bash
docker run --rm --privileged \
  -v /it-share/gguasti/actions-runner/_work:/work \
  lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728 \
  rm -rf /work/InferenceX
```

**Fix da implementare:** pre-cleanup automatico nel launcher prima di ogni `actions/checkout`.
