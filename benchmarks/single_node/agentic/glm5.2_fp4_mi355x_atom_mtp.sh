#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for GLM5.2 FP4 on MI355X using ATOM MTP
#
# Required env vars:
#   MODEL, MODEL_PATH, TP, CONC, KV_OFFLOADING, KV_OFFLOAD_BACKEND,
#   TOTAL_CPU_DRAM_GB, RESULT_DIR, DURATION, EP_SIZE, DP_ATTENTION

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

echo "MODEL=$MODEL TP=$TP CONC=$CONC KV_OFFLOADING=$KV_OFFLOADING TOTAL_CPU_DRAM_GB=$TOTAL_CPU_DRAM_GB RESULT_DIR=$RESULT_DIR DURATION=$DURATION EP_SIZE=$EP_SIZE DP_ATTENTION=$DP_ATTENTION"

if [[ -v SLURM_JOB_ID ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility under slurm cgroups.
if [[ -v ROCR_VISIBLE_DEVICES ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# DCP is enabled for large-concurrency points (C16+) via dcp-size in
# configs/amd-master.yaml; default 1 keeps the small-concurrency TP-only path.
DCP_SIZE="${DCP_SIZE:-1}"

if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

rocm-smi || true
amd-smi || true

resolve_trace_source
install_agentic_deps

# Require the ATOM Prometheus stream in every official result.
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

wait_for_amd_gpu_clean

SERVER_LOG="$RESULT_DIR/server.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"
mkdir -p "$RESULT_DIR"

SERVER_PID=""
LMCACHE_PIDS=()
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$SERVER_PID" "ATOM server" 60
    local i
    for i in "${!LMCACHE_PIDS[@]}"; do
        stop_background_process_tree "${LMCACHE_PIDS[$i]}" "LMCache server $i"
    done
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OFFLOAD_ARGS=()

case "$KV_OFFLOAD_BACKEND" in
    "")
        require_agentic_kv_offload_none
        ;;
    lmcache)
        require_agentic_kv_offload_backend lmcache

        export PYTHONHASHSEED=0
        export LMCACHE_LOCAL_CPU=True
        export LMCACHE_MAX_LOCAL_CPU_SIZE="$TOTAL_CPU_DRAM_GB"
        export LMCACHE_CHUNK_SIZE=256
        export OFFLOAD_MIN_LOAD_TOKENS=8192
        export LMCACHE_NUMA_MODE=auto

        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"lmcache_offload\",\"kv_role\":\"offload\"}"
        )
        ;;
    *)
        echo "Unsupported KV_OFFLOAD_BACKEND: $KV_OFFLOAD_BACKEND (expected empty or lmcache)" >&2
        exit 1
        ;;
esac

echo "Starting atom server..."
export PYTHONNOUSERSITE=1

export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_USE_FLYDSL_MOE_SORTING=1
# GLM-5.2 MLA is nope=192/v=256; FlyDSL gather_kv_b_proj only supports 128/128,
# so force the Triton gather (needed on the DCP prefill-context and MTP verify
# paths).
export ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=0

if (( DCP_SIZE > 1 )); then
    # TP+DCP large-concurrency path: [1,2,4,8] then 12..(2*CONC) step 4.
    CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8'
    for ((size = 12; size <= CONC * 2; size += 4)); do
        CUDAGRAPH_CAPTURE_SIZES+=",${size}"
    done
    CUDAGRAPH_CAPTURE_SIZES+=']'
else
    case "$CONC" in
      1)  CUDAGRAPH_CAPTURE_SIZES='[1,2]' ;;
      2)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4]' ;;
      4)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8]' ;;
      8)  CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16]' ;;
      10) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20]' ;;
      12) CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,12,16,20,24]' ;;
      *)
        echo "Unsupported CONC=$CONC for TP-only path" >&2
        exit 2
        ;;
    esac
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP") #TP
if [ "$DP_ATTENTION" = "true" ]; then
    if [ "$EP_SIZE" -gt 1 ]; then #DP+EP
        PARALLEL_ARGS=(--tensor-parallel-size "$TP" --enable-dp-attention --enable-expert-parallel)
    else 
        PARALLEL_ARGS=(--tensor-parallel-size "$TP" --enable-dp-attention )
    fi
fi
if (( DCP_SIZE > 1 )); then
    PARALLEL_ARGS+=(--decode-context-parallel-size "$DCP_SIZE")
    # Block-level KV interleave: token i lives on DCP rank (i // S) % W, so each
    # rank keeps S consecutive tokens instead of striping one in every W. S=16 is
    # the headline example in ATOM's docs/context_parallel_guide.md and the max
    # allowed here (kv_cache_block_size is 16, and S must divide it). Leaving it
    # unset silently runs the token-level default S=1.
    PARALLEL_ARGS+=(--dcp-config '{"interleave_size": 16, "enable_query_replication": true}')
fi

# Draft depth per concurrency; forced acceptance length is the golden value for
# that depth from
# https://github.com/SemiAnalysisAI/InferenceX/blob/main/golden_al_distribution/glm5.2_mtp.yaml
# (glm-5.2-fp8, thinking_on): K5 -> 3.61, K4 -> 3.33, K3 -> 2.99.
NUM_SPEC_TOKENS=0; SIMULATE_ACC_LEN=0
if (( DCP_SIZE > 1 )); then
    # GLM-5.2 is GlmMoeDsaForCausalLM, i.e. DSA / sparse MLA, and ATOM's
    # docs/context_parallel_guide.md scopes MTP-under-DCP to dense MLA: "DSA /
    # sparse MLA does not support MTP under DCP yet ... Serve DSA + DCP without
    # --method mtp". The same guide validates GLM-5.2 tp4/dcp4 and tp8/dcp8
    # explicitly with no speculative decode. Enabling MTP here collapses prefill
    # (24/444 warmup requests in 2670 s vs 422/444 in 1442 s without it) and
    # also forces interleave_size back to 1.
    SPEC_ARGS=()
else
    case "$CONC" in
      1|2|4|8) NUM_SPEC_TOKENS=5; SIMULATE_ACC_LEN=3.61 ;;
      10|12)   NUM_SPEC_TOKENS=4; SIMULATE_ACC_LEN=3.33 ;;
      *)
        echo "Unsupported CONC=$CONC for TP-only MTP path" >&2
        exit 2
        ;;
    esac
    SPEC_ARGS=(
        --method mtp
        --num-speculative-tokens "$NUM_SPEC_TOKENS"
    )
    if [ "${EVAL_ONLY}" != "true" ]; then
        SPEC_ARGS+=(--spec-decode-acceptance-length "$SIMULATE_ACC_LEN")
    fi
fi
echo "DCP_SIZE=$DCP_SIZE NUM_SPEC_TOKENS=$NUM_SPEC_TOKENS SIMULATE_ACC_LEN=$SIMULATE_ACC_LEN"

ATOM_CMD=(
    python -m atom.entrypoints.openai_server
    --model "$MODEL_PATH"
    # AIPerf addresses the server by the HF id, while --model carries the local
    # checkout path on runners that pre-stage weights. ATOM rejects the mismatch
    # with a 400 on every request, so pin the served name like the dsv4 recipe.
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --server-port "$PORT"
    "${PARALLEL_ARGS[@]}"
    --gpu-memory-utilization 0.95
    --enable_prefix_caching
    --online_quant_config '{"global_quant_config":"ptpc_fp8","exclude_layer":["lm_head","model.embed_tokens","*.mlp.gate","model.layers.[0-9].mlp.*expert*","model.layers.[1-6][0-9].mlp.*expert*","model.layers.7[0-7].mlp.*expert*"]}'
    --max-num-seqs "$((2 * CONC))"
    --cudagraph-capture-sizes "$CUDAGRAPH_CAPTURE_SIZES"
    --max-num-batched-tokens 16384
    --kv_cache_dtype fp8
    "${SPEC_ARGS[@]}"
    "${OFFLOAD_ARGS[@]}"
)
write_command "$RESULT_DIR/server_command.txt" "${ATOM_CMD[@]}"
"${ATOM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --apply-chat-template"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
