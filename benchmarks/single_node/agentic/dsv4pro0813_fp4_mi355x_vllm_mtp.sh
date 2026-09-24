#!/usr/bin/env bash
set -eo pipefail
set -x

# DeepSeek-V4-Pro-0813 FP4 on MI355X with vLLM DSpark K6. Throughput runs fix
# synthetic acceptance to golden AL 3.77; EVAL_ONLY keeps real verification.
#
# Derived from dsv4_fp4_b200_vllm_mtp.sh (same checkpoint, same framework, same
# speculative method) with the ROCm half taken from dsv4_fp4_mi355x_vllm_mtp.sh.
# The config key is dsv4pro0813-fp4-mi355x-vllm-agentic-dspark. The _mtp here
# is not a contradiction: the launcher derives this filename, and SPEC_SUFFIX
# in runners/launch_mi355x-amds.sh emits _mtp for any speculative arm -- no
# launcher has a dspark branch, and spec-decoding is a
# Literal["mtp","draft_model","none"] in infx/matrix/validation.py, so "dspark"
# would fail config validation before a launcher ever ran. Upstream splits it
# the same way: dsv41flash-fp4-mi300x-vllm-agentic-dspark resolves to
# dsv41flash_fp4_mi300x_mtp.sh. DSpark arms declare draft_model.
#
# model-prefix is dsv4pro0813 rather than dsv4 because the launcher builds the
# name from it, and dsv4 would resolve to dsv4_fp4_mi355x_vllm_mtp.sh -- the
# June DeepSeek-V4-Pro arm, whose acceptance curve is a different one (2.49).
#
# DSpark is present in the pinned ROCm build, checked in the image rather than
# assumed: vllm 0.30.1rc1.dev48+g7f1a5398e ships v1/worker/gpu/spec_decode/dspark/,
# SpeculativeConfig accepts every field this script passes, and
# v1/attention/ops/rocm_aiter_mla_sparse.py exists.
#
# Present and schema-valid is not the same as enabled, and reading it that way
# cost run 36005874797: both of the above hold and VllmConfig still refused the
# config with "Model Runner V1 does not support: dspark speculative decoding".
# Enablement is a third gate, handled at VLLM_USE_V2_MODEL_RUNNER below.
#
# What is still unmeasured is
# the acceptance length this checkpoint reaches on this hardware -- 3.77 below
# is B300's, used as a synthetic constant for throughput, not a ROCm result.
#
# Required env vars:
#   MODEL, TP, CONC, KV_OFFLOADING, TOTAL_CPU_DRAM_GB, RESULT_DIR
#
# KV_OFFLOADING=dram requires KV_OFFLOAD_BACKEND=vllm-native or lmcache.

source "$(dirname "$0")/../../benchmark_lib.sh"

check_env_vars \
    MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR \
    DURATION EP_SIZE DP_ATTENTION
check_env_vars EVAL_ONLY

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# DSpark needs the Markov and confidence heads, which only the -0813 checkpoint
# carries. Refuse the June checkpoint here rather than let vLLM fail an hour
# into weight loading, or -- worse -- succeed by silently dropping the draft.
if [[ "$MODEL" != "deepseek-ai/DeepSeek-V4-Pro-0813" ]]; then
    echo "ERROR: DSpark requires the DeepSeek-V4-Pro-0813 checkpoint, got $MODEL" >&2
    exit 1
fi
export DSV4_MODEL_REVISION=72e1d3230f6c080a530b0a1d46f8eb4602340597
if [[ -n "${MODEL_PATH:-}" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --revision "$DSV4_MODEL_REVISION" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL" --revision "$DSV4_MODEL_REVISION"
    export MODEL_PATH="$MODEL"
fi

mkdir -p "$RESULT_DIR"

# MODEL_PATH wins over MODEL whenever it is set, so a directory staged by hand
# -- which is how every run on a node without the shared HF cache starts -- can
# serve a different checkpoint than the one the results are labelled with. This
# reads the index and shard headers and fails on any mismatch, so that trap
# closes before the GPUs are touched.
python3 "$(dirname "$0")/check_dsv4_dspark_checkpoint.py" \
    --model-path "$MODEL_PATH" --revision "$DSV4_MODEL_REVISION" \
    --output "$RESULT_DIR/checkpoint_preflight.json"

rocm-smi || true

resolve_trace_source
install_agentic_deps

# The nightly ROCm image lacks these runtime deps.
agentic_pip_install --quiet Pillow fastapi uvicorn

export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

# vllm-router expands one HTTP backend into a logical worker per DP rank.
# AIPerf's X-Correlation-ID is stable across a conversation's turns; alias it
# to the router's X-Session-ID so every turn lands on the same rank.
USE_VLLM_ROUTER=false
VLLM_BACKEND_PORT="$PORT"
if [ "$DP_ATTENTION" = "true" ]; then
    USE_VLLM_ROUTER=true
    VLLM_BACKEND_PORT=$((PORT + 1))
    VLLM_ROUTER_VERSION=0.1.14
    VLLM_ROUTER_POLICY=consistent_hash
    VLLM_ROUTER_METRICS_PORT=$((PORT + 10000))
    export AIPERF_HTTP_X_SESSION_ID_FROM_CORRELATION_ID=1
    agentic_pip_install --quiet "vllm-router==$VLLM_ROUTER_VERSION"
fi

# AIPerf scrapes the public endpoint's /metrics, which is the router under
# DP-attention; add the engine endpoint explicitly (deduplicated for pure TP).
export AIPERF_SERVER_METRICS_URLS="http://localhost:${VLLM_BACKEND_PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="vllm:"

# ~832 GiB across 66 shards. A cold load from a shared filesystem dominates
# startup; three hours is a ceiling for that case, not an expectation.
export VLLM_ENGINE_READY_TIMEOUT_S=10800

# vllm-project/vllm#43447 keeps local SWA prefix-cache tails sparsely. 32k
# matches the trace-replay tuning validated for this workload.
export VLLM_PREFIX_CACHE_RETENTION_INTERVAL=32768

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
LMCACHE_LOG="$RESULT_DIR/lmcache_server.log"

SERVER_PID=""
ROUTER_PID=""
LMCACHE_PID=""

# Installed before anything is started, and for every offload backend. An
# abandoned engine keeps eight GPUs and its share of host DRAM until someone
# notices, and this node is shared: the next job fails RCCL init on a half
# drained device and the cause is two hours upstream of the symptom.
cleanup_agentic_services() {
    local exit_code=$?
    trap - EXIT INT TERM
    set +e
    stop_background_process_tree "$ROUTER_PID" "vLLM router"
    stop_background_process_tree "$SERVER_PID" "vLLM server" 60
    if [[ -n "$LMCACHE_PID" ]] && kill -0 "$LMCACHE_PID" 2>/dev/null; then
        kill "$LMCACHE_PID" 2>/dev/null || true
        wait "$LMCACHE_PID" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap cleanup_agentic_services EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

OFFLOAD_ARGS=()

if agentic_kv_offload_enabled; then
    check_env_vars KV_OFFLOAD_BACKEND
    case "$KV_OFFLOAD_BACKEND" in
      vllm-native)
        require_agentic_kv_offload_backend vllm-native
        # OffloadingConnector, not SimpleCPUOffloadConnector:
        # VLLM_USE_SIMPLE_KV_OFFLOAD must stay unset.
        unset VLLM_USE_SIMPLE_KV_OFFLOAD
        TOTAL_CPU_DRAM_PARTITION_GB="$((TOTAL_CPU_DRAM_GB / (8 / TP)))"

        OFFLOAD_ARGS=(
            --kv_offloading_backend native
            --kv_offloading_size "$TOTAL_CPU_DRAM_PARTITION_GB"
        )
        ;;
      lmcache)
        require_agentic_kv_offload_backend lmcache

        wait_for_lmcache_ready() {
            { set +x; } 2>/dev/null
            local attempts="120"
            local tail_pid=""

            while [ ! -f "$LMCACHE_LOG" ]; do
                if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
                    echo "LMCache server died before creating log file. Exiting." >&2
                    exit 1
                fi
                sleep 10
            done

            tail -f -n +1 "$LMCACHE_LOG" &
            tail_pid=$!

            for ((i = 1; i <= attempts; i++)); do
                if curl --output /dev/null --silent --fail "http://127.0.0.1:${LMCACHE_HTTP_PORT}/healthcheck"; then
                    kill "$tail_pid" 2>/dev/null || true
                    wait "$tail_pid" 2>/dev/null || true
                    return 0
                fi
                if [[ -n "$LMCACHE_PID" ]] && ! kill -0 "$LMCACHE_PID" 2>/dev/null; then
                    echo "LMCache server died before becoming healthy. Log follows:" >&2
                    kill "$tail_pid" 2>/dev/null || true
                    wait "$tail_pid" 2>/dev/null || true
                    cat "$LMCACHE_LOG" >&2 || true
                    exit 1
                fi
                sleep 1
            done

            echo "Timed out waiting for LMCache server healthcheck. Log follows:" >&2
            kill "$tail_pid" 2>/dev/null || true
            wait "$tail_pid" 2>/dev/null || true
            cat "$LMCACHE_LOG" >&2 || true
            exit 1
        }

        { set +x; } 2>/dev/null
        unset VLLM_USE_SIMPLE_KV_OFFLOAD

        git clone https://github.com/LMCache/LMCache.git
        cd LMCache
        # https://github.com/LMCache/LMCache/pull/3853
        git checkout 9229067cec0b3a63bb8a39368d101db7ac0bc3c1
        pip install -r requirements/build.txt
        pip install grpcio==1.78.0
        CXX=hipcc BUILD_WITH_HIP=1 pip install -e . --no-build-isolation
        cd ..

        python3 -c "import lmcache.integration.vllm.lmcache_mp_connector" >/dev/null

        TOTAL_CPU_DRAM_PARTITION_GB="$((TOTAL_CPU_DRAM_GB / (8 / TP)))"
        # The external MP server owns the pool so vLLM does not split
        # --kv-offloading-size across TP ranks.
        LMCACHE_HOST="127.0.0.1"
        LMCACHE_PORT="5555"
        LMCACHE_HTTP_PORT="8080"
        # LMCacheMPConnector concatenates lmcache.mp.host and port into the
        # ZMQ endpoint, so the connector gets a ZMQ-style host string.
        LMCACHE_CONNECT_HOST="tcp://$LMCACHE_HOST"
        LMCACHE_L1_SIZE_GB="${TOTAL_CPU_DRAM_PARTITION_GB}"
        if [ "$LMCACHE_L1_SIZE_GB" -gt "$TOTAL_CPU_DRAM_GB" ]; then
            echo "Error: LMCACHE_L1_SIZE_GB=$LMCACHE_L1_SIZE_GB exceeds configured capacity $TOTAL_CPU_DRAM_GB" >&2
            exit 1
        fi
        LMCACHE_L1_INIT_SIZE_GB="20"
        # Read locks are leases on chunks lookup promised vLLM can retrieve.
        # TP8/conc32 can spend >300 s between lookup and retrieve while GPU
        # KV is saturated, leaving the object in L1 but unreadable.
        LMCACHE_L1_READ_TTL_SECONDS="7200"
        LMCACHE_CHUNK_SIZE="256"
        LMCACHE_MAX_WORKERS="$TP"
        export PYTHONHASHSEED="0"
        export LMCACHE_BLOCKING_TIMEOUT_SECS=1200
        LMCACHE_TX_MODE="lmcache_driven"

        echo "Starting LMCache MP server..."
        LMCACHE_CMD=(
            lmcache server
            --host "$LMCACHE_HOST"
            --port "$LMCACHE_PORT"
            --http-host "$LMCACHE_HOST"
            --http-port "$LMCACHE_HTTP_PORT"
            --l1-size-gb "$LMCACHE_L1_SIZE_GB"
            --l1-init-size-gb "$LMCACHE_L1_INIT_SIZE_GB"
            --l1-read-ttl-seconds "$LMCACHE_L1_READ_TTL_SECONDS"
            --chunk-size "$LMCACHE_CHUNK_SIZE"
            --max-workers "$LMCACHE_MAX_WORKERS"
            --eviction-policy LRU
            --supported-transfer-mode "$LMCACHE_TX_MODE"
        )
        printf '%q ' "${LMCACHE_CMD[@]}" > "$RESULT_DIR/lmcache_command.txt"
        printf '\n' >> "$RESULT_DIR/lmcache_command.txt"
        "${LMCACHE_CMD[@]}" > "$LMCACHE_LOG" 2>&1 &
        LMCACHE_PID=$!
        echo "LMCache server PID: $LMCACHE_PID"
        wait_for_lmcache_ready

        OFFLOAD_ARGS=(
            --kv-transfer-config
            "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"$LMCACHE_CONNECT_HOST\",\"lmcache.mp.port\":$LMCACHE_PORT,\"lmcache.mp.mq_timeout\":6000.0}}"
        )
        ;;
      *)
        echo "Error: unsupported KV_OFFLOAD_BACKEND '$KV_OFFLOAD_BACKEND' (expected: vllm-native, lmcache)" >&2
        exit 1
        ;;
    esac
fi

PARALLEL_ARGS=(--tensor-parallel-size "$TP" --data-parallel-size 1)
if [ "$DP_ATTENTION" = "true" ]; then
    PARALLEL_ARGS=(--tensor-parallel-size 1 --data-parallel-size "$TP")
fi

EP_ARGS=()
if [ "$EP_SIZE" -gt 1 ]; then
    EP_ARGS=(--enable-expert-parallel)
fi

DP_SCHED_ARGS=()
if [ "$DP_ATTENTION" = "true" ]; then
    DP_SCHED_ARGS=(
        --prefill-schedule-interval 8
        --long-prefill-token-threshold 16384
    )
fi

# AgentX concurrency counts live session trees, not individual requests.
# Subagent fan-out can push instantaneous request concurrency above CONC, so
# leave 2x headroom rather than clipping those bursts at the scheduler.
MAX_NUM_SEQS=$((2 * CONC))
if [ "$DP_ATTENTION" = "true" ]; then
    MAX_NUM_SEQS="$CONC"
fi

# DSpark is implemented only by the V2 GPU model runner: config/vllm.py puts
# "dspark speculative decoding" on the V1 unsupported list, and VllmConfig
# rejects the whole config there rather than falling back. On ROCm this model
# does not reach V2 on its own -- DeepseekV4ForCausalLM is in
# ROCM_DEFAULT_MRV1_ARCHITECTURES, so use_v2_model_runner returns False before
# it ever consults the feature lists. VLLM_USE_V2_MODEL_RUNNER is read first
# and overrides that default, which is why this is set and not merely implied.
#
# Upstream chose V1 for this architecture on ROCm deliberately, so V2 here is
# the unvalidated path: a missing kernel or a throughput regression is a
# plausible outcome and is a finding about V2, not about DSpark. Without it
# there is no DSpark measurement on this hardware at all.
export VLLM_USE_V2_MODEL_RUNNER=1

# Golden AL 3.77: golden_al_distribution/dsv4-pro-0813-dspark.yaml, thinking_on,
# probabilistic drafting, six draft tokens -- the curve's peak. AgentX measures
# the thinking-on regime, which is the regime that curve was measured in.
# EVAL_ONLY drops the synthetic acceptance so eval verifies real drafts.
NUM_SPEC_TOKENS=6
SYNTHETIC_ACCEPT_LEN=3.77
if [ "${EVAL_ONLY}" = "true" ]; then
    SPEC_CONFIG="{\"method\": \"dspark\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"draft_sample_method\": \"probabilistic\"}"
else
    SPEC_CONFIG="{\"method\": \"dspark\", \"num_speculative_tokens\": $NUM_SPEC_TOKENS, \"draft_sample_method\": \"probabilistic\", \"rejection_sample_method\": \"synthetic\", \"synthetic_acceptance_length\": $SYNTHETIC_ACCEPT_LEN}"
fi

# Graph capture sizes are in tokens, not sequences: a decode batch of S seqs
# verifies S*(1+N) tokens. vLLM rounds each size up to a multiple of (1+N) and
# dedups, so a plain 1..MAX_NUM_SEQS list would only cover MAX_NUM_SEQS/(1+N)
# sequences -- at N=6 that is one shape in seven, and every other batch size
# falls back to eager. The list therefore grows with MAX_NUM_SEQS; check the
# capture time in server.log on the first run of a new concurrency band.
TOKENS_PER_SEQ=$((1 + NUM_SPEC_TOKENS))
CAPTURE_SIZE_LIST=()
for ((num_seqs = 1; num_seqs <= MAX_NUM_SEQS; num_seqs++)); do
    CAPTURE_SIZE_LIST+=("$((num_seqs * TOKENS_PER_SEQ))")
done
GRAPH_CAPTURE_SIZES=$(printf '%s\n' "${CAPTURE_SIZE_LIST[@]}" | sort -n -u | paste -sd, -)
COMPILATION_CONFIG="{\"mode\":3,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[${GRAPH_CAPTURE_SIZES}]}"

# Verification slots are needed on top of the prefill budget, or a full decode
# batch cannot be scheduled alongside a prefill chunk.
MAX_NUM_BATCHED_TOKENS=$((8192 + MAX_NUM_SEQS * TOKENS_PER_SEQ))

echo "Starting vllm server..."
set -x
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_MOE=1
# Inherited from the June DeepSeek-V4-Pro arm and not re-derived for -0813:
# that checkpoint mixes packed MXFP4 routed experts with a full-width FP8
# shared expert, and the nightly otherwise admits the combination into the
# fused path and fails while loading incompatible scales/shapes. Leaving the
# fusion off costs some throughput; turning it on is a measurement, not a guess.
export VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0
# vLLM only clamps torch threads after weight loading; cap from process start.
export OMP_NUM_THREADS=1

# Inherited from the June arm. A server killed minutes earlier can still be
# draining HBM -- KFD reclaim takes minutes -- and starting into a half-drained
# node fails RCCL init with an unhandled HIP error whose message names neither.
sleep 180

{ set +x; } 2>/dev/null
VLLM_CMD=(
    vllm serve "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$VLLM_BACKEND_PORT"
    --trust-remote-code
    --async-scheduling
    --distributed-executor-backend mp
    --kv-cache-dtype fp8
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    "${PARALLEL_ARGS[@]}"
    "${EP_ARGS[@]}"
    "${DP_SCHED_ARGS[@]}"
    --gpu-memory-utilization 0.86
    --moe-backend aiter
    --compilation-config "$COMPILATION_CONFIG"
    --speculative-config "$SPEC_CONFIG"
    --tokenizer-mode deepseek_v4
    --tool-call-parser deepseek_v4
    --reasoning-parser deepseek_v4
    --enable-auto-tool-choice
    --enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --max-num-seqs "$MAX_NUM_SEQS"
    "${OFFLOAD_ARGS[@]}"
)

printf '%q ' "${VLLM_CMD[@]}" | tee "$RESULT_DIR/vllm_command.txt"
printf '\n' | tee -a "$RESULT_DIR/vllm_command.txt"
"${VLLM_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$VLLM_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_VLLM_ROUTER" = "true" ]; then
    echo "Starting native vLLM router on port $PORT for $TP DP ranks..."
    vllm-router \
        --worker-urls "http://localhost:$VLLM_BACKEND_PORT" \
        --policy "$VLLM_ROUTER_POLICY" \
        --intra-node-data-parallel-size "$TP" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$VLLM_ROUTER_METRICS_PORT" \
        --request-timeout-secs 14400 \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
