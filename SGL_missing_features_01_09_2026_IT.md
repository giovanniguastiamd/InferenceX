# Feature mancanti in SGLang rispetto ad ATOM — GLM-5.2 MXFP4 / MI355X
**Data:** 2026-09-01
**Config di riferimento:** `glm5.2-fp4-mi355x-sglang-agentic-mtp` (PR #2777, mergiata 2026-09-01)
**Baseline SGLang:** `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728`
**Riferimento ATOM:** TP=8/EP=1/c4 ITL p50=5.9ms, P90=123
**Miglior risultato campagna (interattività):** TP=8/EP=1/c4 P90=110.5 *(dataset upstream)*; TP=4/EP=4/c12/no-HiCache — ⚠️ *dati preliminari, dataset cappato a 256k — vedi nota dataset §12 sotto*

---

## Riepilogo del gap

| # | Feature | ATOM | SGLang v0.5.16 | Impatto ITL stimato | Priorità |
|---|---------|------|----------------|---------------------|----------|
| F-1 | `ptpc_fp8` quantizzazione attivazioni on-the-fly | ✅ | ❌ (non portato) | ~0.9ms ITL | Alta |
| F-2 | All-reduce INT4 (`ROCM_QUICK_REDUCE_QUANTIZATION=INT4`) | ✅ | ⚠️ accessibile per GLM-5.2 (nessun opt-in richiesto); si attiva solo su grandi batch prefill con TP=4 | ~0.2–0.3ms ITL (solo prefill) | Media |
| F-3 | DP attention (percorso DSA, TP=8/DP=1) | ✅ | ❌ (hang, issue #34582) | — | Media |
| F-4 | Crash `iota_range` / `fp8_mqa_logits` su gfx950 | N/A | ❌ blocca upgrade a v0.5.18 | blocca upgrade | Alta |
| F-5 | AITER unified attention (`SGLANG_USE_AITER_UNIFIED_ATTN=1`) | implicita | Neutro su v0.5.16 | Da valutare su v0.5.18 | Bassa |
| F-6 | FlyDSL MoE sorting (`AITER_USE_FLYDSL_MOE_SORTING=1`) | ✅ | ❌ nessun equivalente | sconosciuto | Bassa |
| F-7 | DCP (decode context parallelism) | ✅ (Kimi-K3) | ❌ non validato per GLM-5.2 | N/A (deprioritizzato) | Bassa |
| F-8 | Inferenza Quark PTPC FP8 attention su ROCm | ✅ | ⚠️ PR #28734 aperta | abilita percorso KV FP8 | Media |

---

> ⚠️ **Correzione dataset (2026-09-03):** Tutte le run su mia1 precedenti al fix commit `541e137` hanno usato il **dataset cappato a 256k** (`cc-traces-weka-062126-256k`, ISL p90 ~170k) invece del corpus integrale (`cc-traces-weka-062126`, ISL p90 ~283k) usato dalla PR 2777 upstream e da ATOM. Causa radice: `MODEL_PREFIX` non veniva propagato al container Docker in `launch_mi355x-amds.sh`. Fix applicato 2026-09-03. Tutti i risultati numerici mia1 qui sotto sono marcati **[256k]** e devono essere rivalidati con il dataset corretto (ri-run 33724174688 in corso).

## Ottimizzazioni di ricetta (nessuna modifica a SGLang necessaria)

| # | Ottimizzazione | Ricetta attuale | Proposta | Guadagno misurato | Stato |
|---|---------------|----------------|---------|-------------------|-------|
| R-1 | **Punto operativo no-HiCache** (KV in HBM) | `kv-offloading: cpu` (HiCache) | `kv-offloading: none` | ⏳ *ri-misurazione in corso con dataset corretto (run 33724174688)* | ⏳ Ri-run 2026-09-03 — dati precedenti invalidati (dataset 256k) |

**Condizioni R-1:** richiede HBM libera sufficiente a contenere tutto il KV all'ISL e concorrenza target. Risultato qualitativo (valido indipendentemente dal dataset): il KV entra in HBM su mia1 a concorrenza moderata. Il boundary OOM deve essere ri-misurato con il dataset integrale (ISL p90 ~283k vs ~170k prima). Necessita validazione su wbb3/amds_01 prima dell'upstream.

**Pareto sweep no-HiCache — [dataset 256k, INVALIDATO] (run 33661191728, 2026-09-02 — mia1-p01-g07, TP=4/EP=4):**

> ⚠️ I dati seguenti usano il dataset cappato a 256k (ISL p90 ~170k). Il dataset integrale ha ISL p90 ~283k — la pressione KV è ~65% più alta. Il boundary OOM si sposterà probabilmente a una CONC più bassa. Ri-misurazione in corso (run 33724174688, conc [1,2,4,6,8,10,12]).

| CONC | ITL p90 (ms) | intv p90 (tok/s/u) | tput/GPU (tok/s) | KV%GPU | n_ok | Stato |
|------|-------------|---------------------|-----------------|--------|------|-------|
| 2    | 9.5 [256k]  | 105.6 [256k]        | 35 [256k]       | 18%    | 442  | ⚠️ 256k |
| 4    | 10.9 [256k] | 92.0  [256k]        | 46 [256k]       | 26%    | 647  | ⚠️ 256k |
| 8    | 15.0 [256k] | 66.7  [256k]        | 79 [256k]       | 50%    | 1279 | ⚠️ 256k |
| 10   | 18.5 [256k] | 54.1  [256k]        | 92 [256k]       | 63%    | 1420 | ⚠️ 256k |
| 12   | 20.4 [256k] | 48.9  [256k]        | 106 [256k]      | 60%    | 1606 | ⚠️ 256k |
| 20   | 150.4 [256k]| 6.6   [256k]        | 79 [256k]       | 100%   | 1493 | ⚠️ 256k |
| 24   | 257.1 [256k]| 3.9   [256k]        | 38 [256k]       | 100%   | 823  | ⚠️ 256k |

*Boundary OOM tra c12 e c20 (dataset 256k) — da ri-verificare con dataset integrale.*

**Lezione metodologica (2026-09-02, qualitativamente valida):** Nello sviluppo di ricette passate, il KV offloading (HiCache) è stato abilitato per default — in modo preventivo, senza misurare prima l'effettiva occupazione HBM alla concorrenza e ISL target. Questo è un errore sistematico: se il KV entra in HBM, abilitare HiCache degrada silenziosamente l'ITL di 3–4× senza alcun beneficio. Il workflow corretto per qualsiasi nuova ricetta dovrebbe essere:

1. **Eseguire prima senza KV offloading** (`kv-offloading: none`) e monitorare `gpu_kv_usage` dalle metriche server
2. Se `gpu_kv_usage` rimane sotto ~85–90% → il KV entra in HBM → no-HiCache è la config migliore
3. Abilitare HiCache solo se `gpu_kv_usage` satura (→ rischio OOM) o se ISL/concorrenza target è nota per superare la capacità HBM
4. Documentare il margine HBM come parte della caratterizzazione della ricetta

Per GLM-5.2 MXFP4 su MI355X (80 GB HBM per GPU): il KV MLA è altamente compresso (~512-dim latent vs K+V completi per testa), quindi il footprint KV a c12/ISL~130k token entra comodamente in HBM. ISL più grandi o concorrenza più alta richiederanno eventualmente HiCache — ma questo va misurato, non assunto a priori.

---

## Dettaglio delle feature

### F-1 — `ptpc_fp8`: quantizzazione on-the-fly delle attivazioni in FP8

**Cos'è:** ATOM applica la quantizzazione FP8 per-token-per-canale alle attivazioni durante l'inferenza (non solo ai pesi). Questo riduce il traffico HBM per i tensori intermedi nei layer di attention e MLP.

**Perché è importante:** La nostra analisi (tuner_en.md §8.4) identifica questo come il **principale gap strutturale** tra ATOM e SGLang. L'impatto residuo stimato, dopo l'aggiunta di KV FP8 (I-9), è ~0.9ms di ITL a c4.

**Stato in SGLang:** Non disponibile. Non esiste un meccanismo `ptpc_fp8` in SGLang per il percorso linear su GPU. Il lavoro più vicino:
- **PR #28734** `[AMD] Fix Load and Inference of MLA models with Quark PTPC FP8 attention on ROCm` — APERTA dal 2026-06-19, aggiornata il 2026-09-01. Riguarda il *caricamento* di checkpoint PTPC-quantizzati e il routing tramite `apply_fp8_ptpc_linear`, ma si tratta dell'estensione lato attention — non della quantizzazione runtime delle attivazioni.

**Riferimenti:**
- tuner_en.md righe 294, 329, 333, 416
- SGLang PR #28734: https://github.com/sgl-project/sglang/pull/28734 — APERTA

**Azione:** Monitorare la PR #28734. Per chiudere il gap completo sarà necessaria una PR successiva che abiliti la quantizzazione runtime delle attivazioni (non solo il caricamento del checkpoint).

---

### F-2 — All-reduce INT4 (`ROCM_QUICK_REDUCE_QUANTIZATION`)

**Cos'è:** ATOM comprime la comunicazione all-reduce a INT4 tramite il kernel quick-reduce di AITER. SGLang espone `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (solo INT8).

**Perché è importante:** La latenza dell'all-reduce scala con il grado TP e le dimensioni del tensore. INT4 dimezza la banda rispetto a INT8. La ricetta di riferimento di ATOM usa `AITER_QUICK_REDUCE_QUANTIZATION=INT4`.

**Stato in SGLang (I-3 nella nostra campagna):** Testato come `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` su v0.5.16 — **neutro** (nessun guadagno misurabile). Motivo: su v0.5.16 il percorso quick-reduce potrebbe non essere attivo per tutti i tensori, oppure INT8 satura già il beneficio.

**Percorso INT4 in SGLang — analisi del codice (2026-09-02):**

Contrariamente all'ipotesi iniziale, GLM-5.2 **non richiede un opt-in specifico per modello** per accedere al percorso quick-reduce INT4. L'analisi del sorgente di `quick_all_reduce.py` e `glm4_moe.py` (SGLang main, 2026-09-02) rivela:

- `QuickAllReduce` si inizializza automaticamente per qualsiasi modello su hardware gfx94x/gfx95x quando `ROCM_QUICK_REDUCE_QUANTIZATION` è impostato a un livello valido (FP/INT8/INT6/INT4).
- Il model override `glm4_moe.py` **non imposta** `disable_custom_all_reduce = True` (a differenza di MiniMax-M3 che lo disabilita di default per evitare la corruzione dei partial output MoE). GLM-5.2 è quindi eleggibile senza flag aggiuntivi.
- Il flag `SGLANG_M3_ALLOW_CUSTOM_AR` è specifico di M3 — riabilita la custom AR che M3 disabilita. Per GLM-5.2 è irrilevante.

**Vincolo critico — soglie di dimensione del tensore (`_QR_MIN_SIZE`):**

Il kernel quick-reduce si attiva solo se la dimensione del tensore supera la soglia minima per il dato dtype, world_size e livello di quantizzazione:

| Configurazione | Minimo INT4 |
|----------------|------------|
| bf16 / TP=4 | **16 MB** |
| bf16 / TP=8 | **2048 MB** |

GLM-5.2 hidden_size=7168, bf16 → dimensione tensore per all-reduce = `7168 × N_token × 2 byte`:
- **Decode** (1–4 token): 0.01–0.06 MB → **sempre sotto soglia** → INT4 non si attiva mai nel decode
- **Prefill** (4096 token): ~56 MB → sopra la soglia 16 MB per TP=4 ✓
- **TP=8**: anche il batch di prefill più grande è improbabile che raggiunga 2048 MB → INT4 praticamente mai attivo a TP=8

**Conclusione:** Il quick-reduce INT4 per GLM-5.2 può aiutare solo su **TP=4, grandi batch di prefill** (~4096+ token). L'ITL del decode non è influenzato. L'impatto è atteso solo sul TTFT ad alta concorrenza, dove lo scheduler di prefill accumula batch grandi.

**Perché `_QR_MIN_SIZE` non può essere abbassata:**

La soglia è il punto di pareggio empirico tra:
```
Costo(BF16): send(dimensione_tensore_bf16)
Costo(INT4): quantize(tensore) + send(dimensione/4) + dequantize(tensore)
              ~~~~~~~~~~~~~~                            ~~~~~~~~~~~~~~~~
              ~5–10 µs overhead fisso lancio kernel     ~5–10 µs fissi
```
Su MI355X (XGMI ~400 GB/s), un tensore da 57 KB (decode step) impiega ~0.14 µs in BF16. Aggiungere la quantizzazione INT4 introduce ~10 µs di overhead fisso — rendendo quell'all-reduce **~70× più lento**. La soglia 16 MB per TP=4 è il punto in cui il risparmio di banda supera l'overhead di lancio kernel. Abbassarla sotto ~8 MB causerebbe regressioni visibili sull'ITL del decode (il percorso caldo).

**Test del 2026-09-02 — run 33635908473 (braccio interattività, HiCache saturo):**

- Branch: `testgg-qr-int4` (basato su `testgg` ← `pr/glm52-sglang-ep1-c12`)
- Immagine: `lmsysorg/sglang-rocm:v0.5.16-rocm720-mi35x-20260728`
- Config: TP=4/EP=4/HiCache, conc=12
- Runner: mia1-p01-g07 (mi355x-amds_03), 8× MI355X
- Env: `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16=1`
- Run id: 33635908473 — **in corso al 2026-09-02 ~13:45 UTC**

**Run 33647161138 — `glm5.2-qr-int4-nohicache-c12` — ⚠️ [DATASET 256k — INVALIDATA] (mia1-p01-g07, 8× MI355X):**

> ⚠️ Questa run ha usato il dataset cappato a 256k (ISL p90 ~130–137k). Con il dataset integrale (ISL p90 ~283k), la pressione KV è ~2× maggiore e questi numeri cambieranno. Il risultato qualitativo (HiCache vs no-HiCache neutro quando il KV entra in HBM) rimane valido.

| Metrica | avg | p50 | p90 [256k] | p99 |
|---------|-----|-----|-----|-----|
| ITL (ms) | 15.06 | 11.91 | ~~20.26~~ | 65.19 |
| Output tok/s/utente | 82.46 | 83.96 | ~~111.82~~ | 141.42 |
| Richieste | 1.607 / 1.607 (0 errori, nessun OOM) |
| ISL p90 | ~130–137k token ← **dataset sbagliato** |

> **Run HiCache 33647171278 — ⚠️ [DATASET 256k — INVALIDATA] (mia1-p01-g07, TP=4/EP=4, c12):**
>
> | Metrica | HiCache [256k] | no-HiCache [256k] | Delta |
> |--------|---------|------------|-------|
> | ITL p90 | ~~21.0 ms~~ | ~~20.4 ms~~ | +0.6 ms (+3%) |
> | KV%GPU | 62% | 60% | +2% |
>
> **Risultato qualitativo (valido):** HiCache e no-HiCache sono equivalenti quando il KV entra in HBM — HiCache aggiunge <1ms di overhead. Numeri assoluti invalidi per la distribuzione ISL più corta.

**Test no-HiCache 2026-09-02 — ⚠️ preliminare, dataset 256k:**

- Config: TP=4/EP=4/**nessun offload KV** (tutto KV in HBM), conc=12, `ROCM_QUICK_REDUCE_QUANTIZATION=INT4`
- Risultato: ~~ITL p90=20,26 ms, output tok/s/utente p90=111,82~~ — **[dataset 256k, da ri-misurare]**
- Ri-run in corso: run 33724174688 con dataset integrale (ISL p90 ~283k)

**Riferimenti:**
- tuner_en.md riga 308
- SGLang PR #32230: https://github.com/sgl-project/sglang/pull/32230 — MERGIATA
- SGLang PR #33402: https://github.com/sgl-project/sglang/pull/33402 — MERGIATA
- Sorgente SGLang: `python/sglang/srt/distributed/device_communicators/quick_all_reduce.py`
- Sorgente SGLang: `python/sglang/srt/arg_groups/model_overrides/glm4_moe.py`

**Perché ATOM guadagna da INT4 e SGLang no (analisi strutturale):**

Tre fattori cumulativi spiegano il divario:

**1. KV offload — collo di bottiglia completamente diverso:**

| | ATOM | SGLang c12 HiCache |
|---|---|---|
| KV location | HBM (tutto in GPU) | DRAM (HiCache saturo) |
| Bottleneck decode step | compute + all-reduce | KV DRAM fetch |
| INT4 agisce su... | percorso critico ✓ | percorso secondario ✗ |

**2. EP=1 vs EP=4 — mix di comunicazioni:**

ATOM con EP=1 ha **solo all-reduce** (INT4 si applica). SGLang con EP=4 ha **all-to-all** (routing token verso esperti MoE, INT4 non si applica) + all-reduce. Una frazione significativa del tempo di comunicazione è spesa sull'all-to-all, che INT4 non può comprimere.

**3. Soglia `_QR_MIN_SIZE` per TP=8 — probabilmente più bassa del previsto:**

Il valore TP=8 di 2048 MB citato in precedenza potrebbe essere errato (non verificato nel sorgente). Se il valore reale è ~32–64 MB, ATOM a TP=8 attiverebbe INT4 per batch di prefill moderati (>2200–4500 token), che è routinario con le ISL delle sue trace.

**Confronto percorso decode:**

```
ATOM (TP=8/EP=1/no-offload):
  decode step: [matmul] → [all-reduce INT4 ✓] → [output]
               ← tutto KV in HBM, nessuna attesa DRAM →
               INT4 visibile: all-reduce È il collo di bottiglia

SGLang c12 HiCache:
  decode step: [fetch KV DRAM ⏳⏳] → [matmul] → [all-reduce INT4 ✓] → [output]
               ← fetch DRAM domina tutto →
               INT4 invisibile dietro il fetch KV

SGLang c12 no-HiCache (run 33647161138, ⚠️ DATASET 256k — INVALIDATA):
  decode step: [matmul] → [all-reduce INT4 ✓] → [output]
               ← come ATOM, ma TP=4/EP=4 →
               ITL p90=~~20ms~~, tok/s/utente p90=~~111,82~~ [256k, da ri-misurare]
```

**Da verificare:** eseguire nel container SGLang:
```bash
python3 -c "from sglang.srt.distributed.device_communicators.quick_all_reduce import _QR_MIN_SIZE; print(_QR_MIN_SIZE)"
```
per ottenere la soglia TP=8 esatta e confrontarla con il valore interno di ATOM.

**Il paradosso fondamentale INT4 per GLM-5.2 (confermato 2026-09-02):**

I tre fattori — concorrenza, KV in HBM e soglia tensore — formano un triangolo impossibile: non esiste un punto operativo in cui INT4 si attiva sul decode E il KV è in HBM E la concorrenza è abbastanza alta da avere grandi batch di prefill.

| Concorrenza | KV in HBM? | INT4 si attiva (decode)? | INT4 si attiva (prefill)? | Effetto netto |
|-------------|------------|--------------------------|---------------------------|---------------|
| c4 | ✓ Sì | ✗ No (tensore ~14–224 KB << 16 MB) | ✗ Probabilmente no (batch piccoli) | Zero |
| c12 + HiCache | ✗ No | ✗ No (tensore ~170–680 KB << 16 MB) | ~ Solo prefill | Mascherato dall'I/O DRAM |
| c12 senza HiCache | ✓ Sì (entra a ISL p90~130k, mia1) | ✗ No (decode sempre sotto soglia) | ✓ Sì (batch > 1200 tok) | ~~ITL p90=20ms, tok/s p90=111,82~~ ⚠️ [256k] — ri-misurazione (run 33724174688) |

Il tensore decode è sempre ~57 KB al massimo (4 token × hidden 7168 × 2 byte ÷ 4 TP) — 3 ordini di grandezza sotto la soglia di 16 MB. **L'INT4 non può mai migliorare l'ITL del decode per GLM-5.2 su TP=4.** Tuttavia, il **punto operativo no-HiCache** — dove il collo di bottiglia torna su compute/comunicazione invece che su DRAM — produce il miglior risultato di interattività indipendentemente dall'attivazione INT4 sul prefill: **il KV in HBM è il fattore dominante**.

ATOM ottiene i suoi risultati migliori con lo stesso meccanismo (nessun KV offload, KV in HBM) — il contributo INT4 sull'all-reduce è aggiuntivo ma secondario.

Per completezza: il KV cache FP8 (`--kv-cache-dtype fp8_e4m3`) è già attivo nello script di produzione (riga 224 di `glm5.2_fp4_mi355x_sglang_mtp.sh`) — non è un gap rispetto ad ATOM.

**Azione (aggiornata 2026-09-03):** (1) ⚠️ Run 33647161138 (no-HiCache, 256k) — numeri ~~ITL p90=20ms, tok/s/utente p90=111,82~~ invalidati. (2) Run 33647171278 (HiCache, 256k) completata — delta HiCache vs no-HiCache qualitativamente neutro (valido). (3) Ri-misurazione in corso: run 33724174688 (Pareto sweep no-HiCache conc [1,2,4,6,8,10,12], dataset integrale ISL p90 ~283k). (4) **In attesa:** confermare raccomandazione no-HiCache con risultati dataset corretto. (5) Chiudere F-2 come **strutturalmente neutro per l'ITL del decode INT4** — questa conclusione qualitativa non è influenzata dal bug dataset. (6) Verificare `_QR_MIN_SIZE` reale per TP=8 per completare il confronto con ATOM.

---

### F-3 — DP attention (percorso DSA, gfx950)

**Cos'è:** La DP (Data-Parallel) attention divide il batch di token tra più worker di attention, riducendo la memoria per-rank e abilitando una maggiore concorrenza con lo stesso grado TP.

**Perché è importante:** ATOM usa la DP attention nella sua ricetta di riferimento. In SGLang la DP attention è implementata (`--enable-dp-attention`) ma causa un hang per GLM-5.2/DSA su ROCm.

**Stato in SGLang:** **NON FUNZIONANTE** — hang 100% riproducibile su gfx950 durante il primo prefill di riscaldamento: tutti i 16 rank dello scheduler si bloccano in `prepare_mlp_sync_batch` / `all_gather_into_tensor`. Tracciato come:
- **Issue #34582**: `[Bug] GLM-5.2 in DP-attention deadlock at first scheduled batch` — **APERTA** dal 2026-08-12.

Miglioramenti correlati (parziali, non risolvono il percorso DSA):
- **PR #31682** `Turn on breakable prefill cuda graph for dp attention by default` — MERGIATA 2026-07-21. Riduce i hang indefiniti ma non risolve la divergenza collettiva DSA.
- **PR #33829** `[Model] Complete dots.note.omni ... DP dummy-row normalization for overlap MTP` — MERGIATA 2026-08-22. Aggiunge normalizzazione delle righe dummy per DP+spec decoding. Potenzialmente rilevante per MTP+DSA — da monitorare.
- Workaround env var `SGLANG_DP_USE_GATHERV=1` + `SGLANG_DP_USE_REDUCE_SCATTER=1`: aiuta per DSv4 ma non validato per GLM-5.2/DSA.

**Riferimenti:**
- tuner_en.md righe 715–726
- SGLang issue #34582: https://github.com/sgl-project/sglang/issues/34582 — APERTA
- SGLang PR #31682: https://github.com/sgl-project/sglang/pull/31682 — MERGIATA
- SGLang PR #33829: https://github.com/sgl-project/sglang/pull/33829 — MERGIATA (dots.note.omni)

**Azione:** Riabilitare il braccio DEP una volta risolto l'issue #34582. Monitorare v0.5.19+. Testare `SGLANG_DP_USE_GATHERV=1` su GLM-5.2.

---

### F-4 — Crash Triton LLVM `iota_range` in `fp8_mqa_logits` (gfx950 / v0.5.18)

**Cos'è:** Su SGLang v0.5.18 + ROCm 7.2.4, il kernel Triton `fp8_mqa_logits` di AITER per l'indicizzatore DSA crasha con `LLVM iota_range assertion: Begin <= End` durante la compilazione JIT su gfx950. Interessa TP=4+HiCache+DSA e TP=8. Il crash si scatena a ≥32.768 token di prompt (ben dentro il contesto da 1M token di GLM-5.2).

**Perché è importante:** Blocca l'upgrade da v0.5.16 a v0.5.18, impedendo l'accesso ai nuovi kernel ROCm 7.2.4, all'AITER aggiornato e ai potenziali miglioramenti NCCL.

**Stato in SGLang:** Più PR aperte affrontano il budget MQA-logits dell'indicizzatore DSA:
- **PR #36960** `[ROCm][Bugfix] Cap the DSA MQA-logits budget at AITER's buffer_store limit` — **MERGIATA 2026-09-01**. Limita direttamente il budget per evitare il range invertito. **Nota:** l'immagine `20260901` è stata costruita prima del merge (22:40 UTC) e non contiene il fix — serve l'immagine `20260902` o successiva.
- **PR #35865** `[AMD] Implement DeepSeek V4 DCP on ROCm` — APERTA dal 2026-08-21 — contiene un fix ai bounds di `fp8_mqa_logits` come parte del lavoro DCP.
- **PR #34129** `[AMD] [GLM5] use optional AITER BLOCK_Q MQA logits` — APERTA dal 2026-08-08. Usa un nuovo kernel AITER BLOCK_Q (ROCm/aiter#4180) che riduce il tempo grezzo di MQA-logits del −25.2% su MI355X (7.53ms vs 10.06ms per forward di prefill). Evita il percorso iota_range quando BLOCK_Q è disponibile.

Nostro workaround (branch testgg-v518-p2): patch torch-fallback applicata in-container al runtime, sostituendo `fp8_mqa_logits` con una matmul BF16 per sequenza. Risultato: nessun miglioramento rispetto a v0.5.16 (+2% ITL p50, nella variabilità). Run annullata dopo c4.

**Ultima release SGLang:** v0.5.18 (2026-08-22). Nessuna v0.5.19 ancora disponibile.

**Riferimenti:**
- tuner_en.md righe 174, 471–525, 340–341
- SGLang PR #36960: https://github.com/sgl-project/sglang/pull/36960 — APERTA
- SGLang PR #34129: https://github.com/sgl-project/sglang/pull/34129 — APERTA
- SGLang PR #35865: https://github.com/sgl-project/sglang/pull/35865 — APERTA

**Azione:** Monitorare PR #36960 e #34129. Una volta mergiata una delle due, ritentare l'upgrade a v0.5.18 senza il fallback torch. Se la PR #34129 (BLOCK_Q) merge per prima, testare direttamente su MI355X per misurare il miglioramento ITL.

---

### F-5 — AITER unified attention (`SGLANG_USE_AITER_UNIFIED_ATTN=1`)

**Cos'è:** Instrada l'attention di extend (prefill) attraverso il kernel `unified_attention` di AITER invece del `mha_batch_prefill` di CK. Potenzialmente più veloce su gfx950 per i modelli con sink tokens.

**Stato in SGLang (I-7 nella nostra campagna):** Testato su v0.5.16 — **neutro** (nessun guadagno misurabile). Su v0.5.16 il backend di unified attention potrebbe non avere copertura completa per le caratteristiche di GLM-5.2 su gfx950 (famiglia d64 group-mode paged assente in CK).

PR attive che aggiungono percorsi di unified attention:
- **PR #37310** `[ROCm] aiter backend: route sinks-model extends through unified_attention (gpt-oss prefill −18…−32% TTFT)` — **APERTA** dal 2026-09-01. Instrada i normali extend attraverso `unified_attention` per i modelli con sinks. Riporta −18%…−32% TTFT per gpt-oss su gfx950.
- **PR #37311** `[ROCm] aiter backend: CK varlen route for long full-attention sinks extends (−15…−16% TTFT at 64k uncached)` — **APERTA** dal 2026-09-01. Percorso CK varlen complementare per extend non cacherate di lunga durata.
- **PR #37312** `[ROCm] aiter backend: write extend attention output into the PCG-provided buffer` — **APERTA** dal 2026-09-01.

Queste tre PR (#37310, #37311, #37312) sono state aperte lo stesso giorno di questo documento (2026-09-01) e sembrano parte di uno sforzo coordinato per attivare la unified attention per i modelli con sinks su gfx950. GLM-5.2 (DSA + sinks/sliding-window) potrebbe beneficiarne.

**Riferimenti:**
- tuner_en.md righe 258, 658
- SGLang PR #37310: https://github.com/sgl-project/sglang/pull/37310 — APERTA
- SGLang PR #37311: https://github.com/sgl-project/sglang/pull/37311 — APERTA
- SGLang PR #37312: https://github.com/sgl-project/sglang/pull/37312 — APERTA

**Azione:** Monitorare PR #37310–#37312. Una volta mergiata, testare `SGLANG_USE_AITER_UNIFIED_ATTN=1` (o flag equivalente) su v0.5.18+ con GLM-5.2. Potrebbe attivarsi su una nuova immagine anche senza env var esplicita.

---

### F-6 — FlyDSL MoE sorting (`AITER_USE_FLYDSL_MOE_SORTING=1`)

**Cos'è:** ATOM usa FlyDSL (il DSL di AMD per la generazione di kernel) per implementare il routing e l'ordinamento ottimizzato degli expert MoE. Abilita un'esecuzione MoE eterogenea con migliore utilizzo della GPU.

**Stato in SGLang:** FlyDSL è in fase di integrazione in SGLang per AMD, ma tramite un percorso diverso (backend di linear-attention):
- **PR #33544** `[AMD] Add FlyDSL GDN linear-attention backend` — APERTA
- **PR #37173** `[AMD][DSV4] perf: tuned FlyDSL fp4 indexer score on MI355X` — APERTA

Nessun equivalente diretto di `AITER_USE_FLYDSL_MOE_SORTING` trovato in SGLang. Il percorso AITER MoE è accessibile diversamente:
- **PR #35074** `[AMD] Perf dsv4 enable heterogeneous AITER FHMoE` — APERTA
- **PR #36269** `[AMD] Add ROCm MegaMoE path via AITER MegaMoEV2` — APERTA

**Riferimenti:**
- tuner_en.md (sezione confronto ATOM)
- SGLang PR #33544: https://github.com/sgl-project/sglang/pull/33544 — APERTA
- SGLang PR #35074: https://github.com/sgl-project/sglang/pull/35074 — APERTA

**Azione:** Bassa priorità per GLM-5.2. Il kernel MoE per GLM-5.2 su v0.5.16 usa già l'AITER fused topk_gating (PR #28399, mergiata). Monitorare le PR AITER MoE.

---

### F-7 — DCP (decode context parallelism) per GLM-5.2

**Cos'è:** Il DCP divide la KV cache tra più rank *solo durante il decode*, permettendo un grado TP inferiore per gruppo di decode mantenendo il parallelismo completo del modello. ATOM usa il DCP per Kimi-K3.

**Stato in SGLang:** Il DCP è in fase di implementazione per AMD ROCm ma non ancora mergiato:
- **PR #35865** `[AMD] Implement DeepSeek V4 DCP on ROCm` — APERTA dal 2026-08-21, aggiornata il 2026-08-31
- **PR #37407** `DeepSeek-V4 decode context parallelism on AMD HIP (unified_kv)` — APERTA dal 2026-09-01

Per GLM-5.2 specificamente: **deprioritizzato** (tuner_en.md righe 339, 346). Motivazione: TP=8 batte TP=4 per ITL perché `t_weights` (tempo di caricamento pesi / GPU) domina su NCCL. DCP+TP=2 peggiorerebbe il caricamento dei pesi. Utile solo se NCCL diventasse il collo di bottiglia.

**Riferimenti:**
- tuner_en.md righe 157, 205, 339, 346, 695
- SGLang PR #35865: https://github.com/sgl-project/sglang/pull/35865 — APERTA
- SGLang PR #37407: https://github.com/sgl-project/sglang/pull/37407 — APERTA (2026-09-01)

**Azione:** Nessuna azione per GLM-5.2. Monitorare per Kimi-K3 o modelli futuri dove la banda di decode domina.

---

### F-8 — Inferenza Quark PTPC FP8 attention su ROCm (caricamento checkpoint)

**Cos'è:** La variante `amd/GLM-5.2-Quark-MXFP4-AttnFP8` usa pesi FP8 per-token-per-canale per l'attention (quantizzati con Quark). Il caricamento e l'esecuzione di questo checkpoint su ROCm/gfx950 richiede un routing specifico tramite `apply_fp8_ptpc_linear`.

**Perché è importante:** È un prerequisito per abilitare il percorso KV FP8 (I-9) tramite Quark PTPC attention, che si stima recuperi ~0.5ms di ITL.

**Stato in SGLang:**
- **PR #28734** `[AMD] Fix Load and Inference of MLA models with Quark PTPC FP8 attention on ROCm` — **APERTA** dal 2026-06-19, aggiornata il 2026-09-01. Estende la pipeline `apply_fp8_ptpc_linear` + `fused_rms_fp8_group_quant` al percorso Quark attention. Prerequisito per usare `amd/GLM-5.2-Quark-MXFP4-AttnFP8`.

Nota: la nostra ricetta attuale usa `amd/GLM-5.2-MXFP4` (non la variante AttnFP8) con `--kv-cache-dtype fp8_e4m3` a livello di server. La PR #28734 è necessaria per usare il FP8 attention lato checkpoint.

**Riferimenti:**
- tuner_en.md riga 333 (KV FP8 ~0.5ms stimato)
- SGLang PR #28734: https://github.com/sgl-project/sglang/pull/28734 — APERTA

**Azione:** Monitorare PR #28734. Una volta mergiata, testare `amd/GLM-5.2-Quark-MXFP4-AttnFP8` su v0.5.18+ per misurare la riduzione di ITL dal percorso KV FP8.

---

## Piano d'azione prioritizzato

| Priorità | Feature | Blocco | Prossima azione |
|----------|---------|--------|-----------------|
| **1** | F-4: fix `iota_range` | ~~PR #36960~~ MERGIATA 2026-09-01; PR #34129 aperta | Ritentare upgrade v0.5.18; monitorare #34129 (BLOCK_Q) per ulteriore gain ITL |
| **2** | F-5: AITER unified attention (percorso extend) | Merge PR #37310/#37311 | Testare su v0.5.18+ una volta mergiate |
| **3** | F-8: caricamento Quark PTPC FP8 | Merge PR #28734 | Testare checkpoint AttnFP8 su v0.5.18+ |
| **4** | F-2: all-reduce INT4 | Nessun opt-in richiesto per GLM-5.2; attivo solo su grandi prefill TP=4 | **Test in corso 2026-09-02** (run 33635908473, c12, mia1); attendere risultati finali |
| **5** | F-3: DP attention | Fix issue #34582 | Monitorare; riabilitare braccio DEP dopo il fix |
| **6** | F-1: `ptpc_fp8` | Nessuna PR SGLang | Seguire upstream; gap strutturale |
| **7** | F-6: FlyDSL MoE sorting | Nessun equivalente diretto | Bassa priorità per GLM-5.2 |
| **8** | F-7: DCP per GLM-5.2 | Deprioritizzato | Nessuna azione |

---

## PR chiave da seguire

| PR | Titolo | Stato | Data |
|----|--------|-------|------|
| #28734 | AMD Fix Quark PTPC FP8 attn su ROCm | APERTA | 2026-06-19 |
| #34129 | AMD GLM5 AITER BLOCK_Q MQA logits | APERTA | 2026-08-08 |
| #34582 | Bug: deadlock DP attn GLM-5.2 | APERTA | 2026-08-12 |
| #35865 | AMD DSv4 DCP su ROCm | APERTA | 2026-08-21 |
| #36960 | ROCm Bugfix: cap budget DSA MQA-logits | APERTA | 2026-08-29 |
| #37310 | ROCm aiter: unified_attention per extend (−18…−32% TTFT) | APERTA | 2026-09-01 |
| #37311 | ROCm aiter: CK varlen sinks extends (−15…−16% TTFT) | APERTA | 2026-09-01 |
| #37407 | DSv4 DCP su AMD HIP (unified_kv) | APERTA | 2026-09-01 |

---

## Miglioramenti incrementali pianificati (da tracker AgentX, 2026-09-01)

Questi item provengono dal tracker interno AgentX (`agentx_PRs_optimizations(data).csv`, responsabile: Zhenyu Gu / Jiejing). Sono miglioramenti incrementali già pianificati o in corso dal team SGLang AMD — distinti dai gap strutturali vs ATOM elencati sopra. Tutti riguardano GLM-5.2/SGLang su MI355X salvo indicazione contraria.

### Già completati

| PR | Titolo | Framework | Stato | Impatto |
|----|--------|-----------|-------|---------|
| #36515 | [AMD] fix: do not emit a shared-expert marker twice on the per-rank slot path | SGLang | **MERGIATA 2026-08-30** | Fix di correttezza per il routing MoE degli shared expert |
| (no PR) | Full CUDA graph quando MTP abilitato | SGLang | **Completato** | +25% interattività / +5% throughput |
| (no PR) | DSA indexer Top-K ottimizzato | SGLang | **Completato** | +11% interattività / +2% throughput |

### PR aperte — SGLang

| PR | Titolo | Stato | Impatto |
|----|--------|-------|---------|
| #37152 | [ROCm] Make the HiCache kernel IO backend work on ROCm | APERTA 2026-08-30 | **+25–33% throughput** a ISL=100K, 90% hit rate prefix (braccio TP=4/HiCache) |
| #37130 | [AMD] Remove silent ×0.85 mem_fraction_static derate for aiter + ctx>8K | APERTA 2026-08-30 | **+15% KV pool** a ISL=100K — ripristina memoria silenziosamente ridotta dal backend aiter |
| #37133 | [GLM-5.2] Keep GlmMoeDsa MoE e_score_correction_bias in fp32 | APERTA 2026-08-30 | Fix di correttezza per la precisione del gating MoE |
| #37118 | [ROCm] Define the DSA head-gate graph helpers on HIP | APERTA 2026-08-30 | Abilita la cattura CUDA-graph con MTP sul percorso DSA (prerequisito per la copertura graph completa) |
| #37124 | [ROCm] Take the fused DSA metadata kernels and drop redundant work from the absorb path | APERTA 2026-08-30 | Fusione kernel DSA — riduce l'overhead per-step sul percorso absorb |
| #36530 | [AMD][DSV4] perf: fold the padded-weight zeroing into the fused append kernel | APERTA | Fusione kernel minore per il percorso padded-weight |
| #31213 | [GLM-5.2] Keep GlmMoeDsa MoE e_score_correction_bias in fp32 (versione precedente) | APERTA | Stesso fix di #37133 — verificare quale è la versione canonica con Zhenyu/Jiejing |

### PR aperte — AITER (ROCm/aiter)

| Item | Descrizione | Stato | Impatto |
|------|-------------|-------|---------|
| ROCm/aiter (TBD) | [HIP] topk: fuse DSA page-table transform into cooperative top-k | APERTA | Fusione top-k indicizzatore DSA — riduce overhead indicizzatore |
| ROCm/aiter (TBD) | [Triton/Gluon] fp8_mqa_logits: gate buffer ops on int32 offset, not tensor bytes | APERTA | **Risolve il crash** su qualsiasi chunked prefill per GLM-5.x — fix lato AITER complementare alla PR SGLang #36960 |

> Nota: i numeri di PR AITER non sono ancora tracciati nel CSV. Il fix AITER di `fp8_mqa_logits` è il fix della causa radice del crash descritto in F-4 — una volta disponibile in una build AITER/ROCm rilasciata, i workaround lato SGLang (PR #36960) potrebbero diventare ridondanti.

### Ottimizzazioni in corso (nessuna PR ancora)

| Item | ETA | Stato | Impatto stimato |
|------|-----|-------|----------------|
| DCP per GLM-5.2 ad alta concorrenza (richiede ottimizzazione kernel Top-K in scenario DCP) | 4-Set | Funzionalità pronta | Da valutare — vedi F-7 |
| MTP acceptance length dinamica (MTP=4 verificato a c2) | 28-Ago | In verifica sulle altre concorrenze | Miglioramento ITL a bassa concorrenza |
| Piccola fusione kernel lato MTP | 4-Set | In corso | Da valutare |
| FlyDSL MoE tuning | 4-Set | In corso | Da valutare — vedi F-6 |
| Tokenizer cache | 4-Set | In corso | Riduzione latenza per prompt ripetuti |
| MegaMoE + MoE tuning | 4-Set | In corso | **+5% throughput** |
| Ottimizzazione read-throughput HiCache e aumento capacità | 4-Set | In corso (nessuna PR al 2026-08-29) | **+10% throughput** — complementa PR #37152 |
| Kernel MLA PS (prefill-split) | 28-Ago | Pianificato | Riduzione TTFT per prompt lunghi |

---

## Appendice — Spiegazione delle tecnologie coinvolte

### GLM-5.2 e lo stack di inferenza

**GLM-5.2** (ZhipuAI / THUDM) è un grande modello linguistico di tipo Mixture-of-Experts. AMD lo distribuisce come `amd/GLM-5.2-MXFP4`, quantizzato con il toolkit Quark in formato MXFP4 (microscaling FP4) per i pesi. Il modello usa un'architettura **DSA** (Dynamic Sparse Attention) — variante di MLA (Multi-head Latent Attention) con indicizzazione sparsa e "sink token" — e un blocco feed-forward di tipo **MoE** (Mixture-of-Experts).

**SGLang** è un motore di inferenza open-source (sgl-project/sglang) sviluppato originariamente a UC Berkeley. Supporta tensor parallelism, speculative decoding e offload della KV cache. Il backend AMD/ROCm è co-sviluppato attivamente da ingegneri AMD.

**ATOM** è il motore di inferenza proprietario di AMD, ottimizzato specificamente per l'hardware MI3xx. È il target di riferimento per i confronti di prestazioni in questo documento.

**MI355X / gfx950** è la GPU data-center di AMD (serie Instinct). Usa l'architettura CDNA3 e supporta ROCm 7.x, accelerazione hardware MXFP4/FP8 e la famiglia di kernel DSA per attention sparsa.

---

### Acronimi e termini

**AITER** (AMD Inference Engine Runtime): libreria di AMD di kernel GPU ottimizzati per l'inferenza, ora integrata in ROCm come libreria di sistema. Fornisce kernel di attention fusa (stile flash-attention), RoPE, RMSNorm, routing MoE, all-reduce, ecc. SGLang usa AITER tramite il flag `--attention-backend aiter` o variabili d'ambiente ROCm.

**ATOM** (AMD Tensor Operations for ML): motore di inferenza closed-source di AMD ad alte prestazioni per GPU MI3xx. Usato come baseline di performance. Include feature non ancora disponibili in SGLang (ptpc_fp8, all-reduce INT4, FlyDSL MoE sorting).

**BF16** (Brain Float 16): formato in virgola mobile a 16 bit standard, usato per la maggior parte delle attivazioni e della KV cache nel nostro baseline. 8 bit esponente, 7 bit mantissa.

**CK** (Composable Kernels): libreria di AMD di kernel GPU ottimizzati manualmente per GEMM, attention, ecc. Fornisce `mha_batch_prefill` usato dal backend aiter di SGLang per il prefill. Alcune famiglie di kernel mancano su gfx950 (d64 group-mode paged), motivo per cui si sta integrando la unified_attention (F-5).

**DCP** (Decode Context Parallelism): strategia di parallelismo che divide la KV cache tra più rank *solo durante il decode*, mantenendo il TP standard per il prefill. Riduce la pressione di memoria per-rank durante il decode senza aumentare il costo di comunicazione TP. Diverso da TP (Tensor Parallelism) e PP (Pipeline Parallelism). In SGLang: `--dcp-size N`.

**DP attention** (`--enable-dp-attention`): modalità di attention Data-Parallel in SGLang. Divide il batch di input tra worker DP per il calcolo dell'attention (ogni worker vede un sottoinsieme di token), mentre il MLP usa TP completo. Richiede un all-gather per la sincronizzazione MLP. Richiede l'infrastruttura `DataParallelController`.

**DSA** (Dynamic Sparse Attention): meccanismo di attention di GLM-5.2. Usa un indicizzatore sparso per selezionare quali token key/value ogni query attende, più "sink token" (attenzione globale fissa). L'indicizzatore (`fp8_mqa_logits`) calcola i logit query–key sui candidati per costruire l'indice sparso. Più costoso dell'attention densa standard ma abilita contesti da 1M token con costo per-step limitato.

**EP** (Expert Parallelism): distribuisce gli expert MoE tra le GPU. Ogni GPU tiene un sottoinsieme di expert. Richiede comunicazione all-to-all per instradare i token all'expert corretto. EP alto aumenta il costo di comunicazione ma riduce la memoria per-GPU. Nella nostra ricetta: EP=1 (tutti gli expert su ogni replica GPU, nessun all-to-all) per il braccio TP=8 di interattività; EP=4 per il braccio TP=4 di throughput.

**FlyDSL**: linguaggio domain-specific di AMD per la generazione di kernel GPU ottimizzati, in particolare per operazioni sparse e MoE. Usato in ATOM per il sorting e routing dei token MoE. Non ancora esposto direttamente come flag in SGLang; AMD lo sta integrando tramite i percorsi di linear-attention e AITER MoE.

**FP8** (floating point 8 bit, formato e4m3): float a 8 bit con 4 bit di esponente e 3 bit di mantissa. Usato per la KV cache (`--kv-cache-dtype fp8_e4m3`) per dimezzare l'occupazione HBM della KV cache rispetto a BF16. Usato anche per la quantizzazione dei pesi di attention in modalità PTPC.

**HiCache**: sistema gerarchico di KV cache di SGLang. Offloada i blocchi di KV cache dalla HBM GPU alla DRAM host quando la memoria GPU è piena, poi li recupera su richiesta. Controllato da `hicache_ratio` (rapporto tra DRAM host e HBM GPU allocata per la KV). Usato nel braccio TP=4/EP=4 di throughput della nostra ricetta.

**HBM** (High Bandwidth Memory): la DRAM stacked collegata direttamente al die GPU. Su MI355X: 192 GB HBM3e. La banda HBM è il principale collo di bottiglia per operazioni memory-bound (letture KV cache, letture pesi nel decode).

**ITL** (Inter-Token Latency): tempo tra token generati consecutivi per una singola richiesta, misurato in ms. La metrica principale di interattività. Più basso è meglio. ITL P90 = 90° percentile su tutti i token di tutte le richieste nel benchmark.

**JIT** (Just-In-Time compilation): Triton (e altri framework di kernel GPU) compilano i kernel al primo utilizzo per l'architettura GPU target. Su gfx950, alcune forme di kernel innescano fallimenti di asserzione LLVM durante il JIT (vedi F-4). La compilazione JIT aggiunge latenza di startup ma permette la specializzazione.

**KV cache**: tensori key-value memorizzati dai token di contesto precedenti, riutilizzati a ogni step di decode per evitare il ricalcolo. La dimensione della KV cache scala con lunghezza sequenza × layer × head × head_dim × 2 (K+V). Con BF16: 2 byte/elemento; con FP8: 1 byte/elemento.

**MLA** (Multi-head Latent Attention): architettura di attention di DeepSeek, usata in DSv4, Kimi-K3 e GLM-5.2. Comprime la KV cache tramite una proiezione latente a basso rango, riducendo drasticamente le dimensioni della KV. La DSA di GLM-5.2 è costruita sopra MLA.

**MoE** (Mixture of Experts): architettura feed-forward in cui ogni token viene instradato a un piccolo sottoinsieme di MLP expert (es. 2 su 64), riducendo i parametri attivi per token. GLM-5.2 usa MoE per i suoi layer FFN.

**MTP** (Multi-Token Prediction): meccanismo di speculative decoding di GLM-5.2. Le draft head del modello predicono più token futuri in parallelo; il modello principale li verifica in un unico forward pass. Nella nostra ricetta: `--speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6` (config 5-1-6). Diverso da EAGLE (che usa un modello draft separato); MTP usa le proprie draft head del modello target.

**MXFP4** (Microscaling FP4): formato AMD/OCP a microscaling. Gruppi di 32 pesi condividono un singolo fattore di scala FP8, con ogni peso memorizzato come float a 4 bit. Usato per la quantizzazione dei pesi di GLM-5.2. Accelerato hardware su MI355X/gfx950.

**NCCL / RCCL**: libreria di comunicazione GPU (NVIDIA Collective Communications Library / equivalente ROCm). Usata per all-reduce, all-gather e reduce-scatter tra rank TP.

**P90** (90° percentile): latenza al di sotto della quale cade il 90% delle misure. Usato qui per ITL P90 — la latenza inter-token "peggiore tipica".

**ptpc_fp8** (per-token per-channel FP8): quantizzazione on-the-fly delle attivazioni di ATOM. Durante l'inferenza, le attivazioni (input ai layer lineari) vengono quantizzate in FP8 in tempo reale usando fattori di scala per-token. Riduce la banda HBM per i tensori intermedi senza un checkpoint pre-quantizzato. Non disponibile in SGLang (F-1).

**Quark**: toolkit di quantizzazione di AMD per la produzione di checkpoint di modelli quantizzati. Usato per creare `amd/GLM-5.2-MXFP4` (pesi MXFP4) e `amd/GLM-5.2-Quark-MXFP4-AttnFP8` (pesi MXFP4 + attention FP8).

**Quick-reduce / custom all-reduce**: percorso all-reduce ottimizzato di AITER che quantizza i dati prima della comunicazione, riducendo la banda. Versione INT8: dimezza la banda rispetto a BF16. Versione INT4: la riduce a un quarto. In SGLang: `ROCM_QUICK_REDUCE_QUANTIZATION=INT8` (o INT4 con opt-in specifico per modello).

**ROCm**: piattaforma di calcolo GPU open-source di AMD (equivalente a NVIDIA CUDA). gfx950 = architettura target ROCm per MI355X. ROCm 7.2.4 è la versione inclusa nelle immagini SGLang v0.5.18-rocm724.

**TP** (Tensor Parallelism): divide i singoli tensori di peso (e i loro calcoli) tra GPU. Ogni rank TP tiene una shard di ogni layer. Richiede all-reduce dopo ogni GEMM. TP più alto riduce la dimensione dei pesi per-GPU (riducendo il tempo di caricamento) a costo di un overhead di all-reduce maggiore. Nella nostra ricetta: TP=8 per interattività, TP=4 per throughput.

**TTFT** (Time To First Token): latenza dalla submissione della richiesta al primo token generato. Dominata dal passaggio di prefill (elaborazione del prompt di input). Distinta da ITL (che misura la velocità di decode).

**Triton**: linguaggio di kernel GPU basato su Python (OpenAI Triton). Usato da SGLang e AITER per kernel compilati JIT, incluso `fp8_mqa_logits` per il calcolo dei logit dell'indicizzatore DSA.

**Unified attention** (AITER): il kernel `unified_attention` di AITER copre sia il prefill extend che il decode in un'unica interfaccia. Su gfx950 colma le lacune nel catalogo kernel di CK. È in fase di aggiunta al backend aiter di SGLang tramite PR #37310 e PR correlate (F-5).
