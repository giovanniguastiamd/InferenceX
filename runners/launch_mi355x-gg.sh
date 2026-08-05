#!/usr/bin/env bash
# Launch script for mi355x-gg cluster (Giovanni Guasti - local AMD MI355X node)
# Based on launch_mi355x-amds.sh

scancel_sync() {
    local jobid=$1
    local timeout=${2:-600}
    local interval=10
    local start
    start=$(date +%s)

    echo "[scancel_sync] Requesting cancel of job $jobid"
    scancel "$jobid" || true

    while [[ -n "$(squeue -j "$jobid" --noheader 2>/dev/null)" ]]; do
        local now
        now=$(date +%s)
        if (( now - start >= timeout )); then
            echo "[scancel_sync][WARN] job $jobid still present after ${timeout}s"
            return 1
        fi
        echo "[scancel_sync] waiting for job $jobid to exit. $((timeout-(now-start))) secs remaining..."
        sleep "$interval"
    done
    echo "[scancel_sync] job $jobid exited"
    return 0
}

if [[ "$IS_MULTINODE" == "true" ]]; then
    set -x

    export SLURM_ACCOUNT="$USER"
    export SLURM_PARTITION="compute"                              # TODO: set your Slurm partition name
    export SLURM_JOB_NAME="benchmark-sglang-disagg.job"

    export MODEL_NAME=${MODEL##*/}
    export MODEL_PATH="/it-share/data"                           # TODO: NFS path for pre-staged model weights
    export IBDEVICES="rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7"  # TODO: RDMA device list
    export MORI_RDMA_TC=104

    export MODEL_DIR="$MODEL_PATH"
    export GPUS_PER_NODE=8

    export ISL="$ISL"
    export OSL="$OSL"

    export BENCHMARK_LOGS_DIR="${BENCHMARK_LOGS_DIR:-$GITHUB_WORKSPACE/benchmark_logs}"
    mkdir -p "$BENCHMARK_LOGS_DIR"
    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

    cleanup_and_save_logs() {
        if [[ -n "${GITHUB_ACTIONS:-}" && -n "${JOB_ID:-}" ]]; then
            local art_dir="$GITHUB_WORKSPACE/benchmark_artifacts"
            mkdir -p "$art_dir"
            cp -r "$BENCHMARK_LOGS_DIR"/slurm_job-${JOB_ID}.{out,err} "$art_dir/" 2>/dev/null || true
        fi
        local err_file="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID:-unknown}.err"
        if [[ -s "$err_file" ]]; then
            echo "=== Slurm job stderr ==="
            tail -100 "$err_file"
            echo "========================"
        fi
        sudo rm -rf "$BENCHMARK_LOGS_DIR" 2>/dev/null || true
    }
    if [[ "${KEEP_LOGS:-0}" == "1" ]]; then
        trap '' EXIT
    else
        trap cleanup_and_save_logs EXIT
    fi

    SCRIPT_NAME="${EXP_NAME%%_*}_${PRECISION}_mi355x_${FRAMEWORK}.sh"
    if [[ "$FRAMEWORK" == "sglang-disagg" ]] || [[ "$FRAMEWORK" == "vllm-disagg" ]] || [[ "$FRAMEWORK" == "atom-disagg" ]]; then
        if [[ "${SCENARIO_SUBDIR}" == "agentic/" ]]; then
            BENCHMARK_SUBDIR="multi_node/agentic"
        else
            BENCHMARK_SUBDIR="multi_node"
        fi
    else
        BENCHMARK_SUBDIR="single_node/fixed_seq_len"
    fi
    JOB_ID=$(bash "benchmarks/${BENCHMARK_SUBDIR}/${SCRIPT_NAME}")

    LOG_FILE="$BENCHMARK_LOGS_DIR/slurm_job-${JOB_ID}.out"
    sleep 10

    while ! ls "$LOG_FILE" &>/dev/null; do
        if ! squeue -u "$USER" --noheader --format='%i' | grep -q "$JOB_ID"; then
            echo "ERROR: Job $JOB_ID failed before creating log file"
            scontrol show job "$JOB_ID"
            exit 1
        fi
        sleep 5
    done

    set +x

    (
        while squeue -u $USER --noheader --format='%i' | grep -q "$JOB_ID"; do
            sleep 10
        done
    ) &
    POLL_PID=$!

    tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null
    wait $POLL_PID

    set -x

    if [[ "${EVAL_ONLY:-false}" != "true" && "${IS_AGENTIC:-0}" != "1" ]]; then
        cat > collect_latest_results.py <<'PY'
import os, sys
job_dir, isl, osl, nexp, framework = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
logs_root = f"{job_dir}/logs/"
candidates = []
if os.path.isdir(logs_root):
    for name in os.listdir(logs_root):
        subdir = f"{logs_root}{name}/{framework}_isl_{isl}_osl_{osl}"
        if os.path.isdir(subdir):
            candidates.append(subdir)
for path in sorted(candidates, key=os.path.getmtime, reverse=True)[:nexp]:
    print(path)
PY

        LOGS_DIR=$(python3 collect_latest_results.py "$BENCHMARK_LOGS_DIR" "$ISL" "$OSL" 1 "$FRAMEWORK")
        if [ -z "$LOGS_DIR" ]; then
            echo "No logs directory found for ISL=${ISL}, OSL=${OSL}"
            exit 1
        fi

        echo "Found logs directory: $LOGS_DIR"
        ls -la "$LOGS_DIR"

        for result_file in $(find $LOGS_DIR -type f); do
            file_name=$(basename $result_file)
            if [ -f $result_file ]; then
                WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${file_name}"
                echo "Found result file ${result_file}. Copying it to ${WORKSPACE_RESULT_FILE}"
                cp $result_file $WORKSPACE_RESULT_FILE
            fi
        done
    fi

    if [[ "${RUN_EVAL:-false}" == "true" ]]; then
        EVAL_DIR=$(find "$BENCHMARK_LOGS_DIR/logs" -type d -name eval_results 2>/dev/null | head -1)
        if [ -n "$EVAL_DIR" ] && [ -d "$EVAL_DIR" ]; then
            echo "Extracting eval results from $EVAL_DIR"
            shopt -s nullglob
            for eval_file in "$EVAL_DIR"/*; do
                [ -f "$eval_file" ] || continue
                cp "$eval_file" "$GITHUB_WORKSPACE/"
                echo "Copied eval artifact: $(basename "$eval_file")"
            done
            shopt -u nullglob
        else
            echo "WARNING: RUN_EVAL=true but no eval results found under $BENCHMARK_LOGS_DIR/logs"
        fi
    fi

    if [[ "${IS_AGENTIC:-0}" == "1" ]]; then
        JOB_LOGS_DIR="$BENCHMARK_LOGS_DIR/logs/slurm_job-${JOB_ID}"
        if [ -d "$JOB_LOGS_DIR" ]; then
            AGENTIC_SRC="$JOB_LOGS_DIR/agentic"
            if [ -d "$AGENTIC_SRC" ] && find "$AGENTIC_SRC" -mindepth 1 -maxdepth 1 -type d -name 'conc_*' -print -quit 2>/dev/null | grep -q .; then
                echo "Staging agentic raw artifacts from $AGENTIC_SRC"
                mkdir -p "$GITHUB_WORKSPACE/LOGS/agentic"
                cp -r "$AGENTIC_SRC"/. "$GITHUB_WORKSPACE/LOGS/agentic/"
                sudo chown -R "$(id -u):$(id -g)" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                chmod -R a+rwX "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
                ls -laR "$GITHUB_WORKSPACE/LOGS/agentic"
            else
                echo "WARNING: no agentic conc_*/ artifacts found under $JOB_LOGS_DIR/agentic"
            fi
            if tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$JOB_LOGS_DIR" . 2>/dev/null; then
                echo "Created multinode_server_logs.tar.gz"
            else
                echo "WARNING: failed to create multinode_server_logs.tar.gz"
            fi
        else
            echo "WARNING: agentic staging skipped; $JOB_LOGS_DIR not found"
        fi
    fi

    echo "All result files processed"
    set +x
    scancel_sync $JOB_ID
    set -x
    echo "Canceled the slurm job $JOB_ID"

    sudo rm -rf "$BENCHMARK_LOGS_DIR/logs" 2>/dev/null || true

else

    # -----------------------------------------------------------------------
    # SINGLE-NODE — adattare i path seguenti al nodo g07
    # -----------------------------------------------------------------------
    export HF_HUB_CACHE_MOUNT="/var/lib/hf-hub-cache/"           # TODO: HF cache mount sul compute node
    export AIPERF_MMAP_CACHE_HOST_PATH="/it-share/aiperf-cache/" # TODO: aiperf mmap cache (NFS o locale)
    export PORT_OFFSET=${RUNNER_NAME: -1}
    export PORT=$(( 8888 + ${PORT_OFFSET} ))
    FRAMEWORK_SUFFIX=$([[ "$FRAMEWORK" == "atom" ]] && printf '_atom' || printf '')
    SPEC_SUFFIX=$([[ "$SPEC_DECODING" == "mtp" ]] && printf '_mtp' || printf '')

    PARTITION="compute"                                           # TODO: nome partizione Slurm su g07
    SQUASH_FILE="/var/lib/squash/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"  # TODO: dir squash images
    LOCK_FILE="${SQUASH_FILE}.lock"

    export GPU_COUNT="${GPU_COUNT:-${TP:?TP must be set}}"

    set -x
    salloc --partition=$PARTITION --gres=gpu:$GPU_COUNT --exclusive --cpus-per-task=128 --time=500 --no-shell --job-name="$RUNNER_NAME"
    JOB_ID=$(squeue --name="$RUNNER_NAME" -h -o %A | head -n1)

    srun --jobid=$JOB_ID bash -c "docker stop \$(docker ps -a -q)"

    srun --jobid=$JOB_ID bash -c "
        exec 9>\"$LOCK_FILE\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $SQUASH_FILE'; exit 1; }
        if unsquashfs -l \"$SQUASH_FILE\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import'
        else
            rm -f \"$SQUASH_FILE\"
            enroot import -o \"$SQUASH_FILE\" docker://$IMAGE
        fi
    "

    export VLLM_CACHE_ROOT="/it-share/gharunners/.cache/vllm"    # TODO: vLLM cache root

    if [[ "$FRAMEWORK" == "atom" ]] || [[ "$FRAMEWORK" == "sglang" ]]; then
        SLRUM_HOME_MOUNT=""
    else
        SLRUM_HOME_MOUNT=" --container-mount-home "
    fi

    if [[ ("$FRAMEWORK" == "vllm" || "$FRAMEWORK" == "atom") ]] && [[ "$MODEL" == "deepseek-ai/DeepSeek-V4-Pro" ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"      # TODO: NFS HF cache per modelli grandi
    fi

    if [[ "$MODEL" == MiniMaxAI/MiniMax-M3* || "$MODEL" == amd/MiniMax-M3* ]]; then
        export HF_HUB_CACHE_MOUNT="/it-share/hf-hub-cache/"
    fi

    SCRIPT_BASE="${EXP_NAME%%_*}_${PRECISION}_mi355x"
    SCRIPT_FW="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}_${FRAMEWORK}${SPEC_SUFFIX}.sh"
    SCRIPT_FALLBACK="benchmarks/single_node/${SCENARIO_SUBDIR:-fixed_seq_len/}${SCRIPT_BASE}${FRAMEWORK_SUFFIX}${SPEC_SUFFIX}.sh"
    if [[ -f "$SCRIPT_FW" ]]; then
        BENCHMARK_SCRIPT="$SCRIPT_FW"
    else
        BENCHMARK_SCRIPT="$SCRIPT_FALLBACK"
    fi

    srun --jobid=$JOB_ID \
        --container-image=$SQUASH_FILE \
        --container-mounts=$GITHUB_WORKSPACE:/workspace/,$HF_HUB_CACHE_MOUNT:$HF_HUB_CACHE,$AIPERF_MMAP_CACHE_HOST_PATH:/aiperf_mmap_cache \
        $SLRUM_HOME_MOUNT \
        --container-writable \
        --container-workdir=/workspace/ \
        --container-remap-root \
        --no-container-entrypoint --export=ALL,AIPERF_DATASET_MMAP_CACHE_DIR=/aiperf_mmap_cache \
        bash "$BENCHMARK_SCRIPT"

    scancel $JOB_ID

    if ls gpucore.* 1> /dev/null 2>&1; then
        echo "gpucore files exist. not good"
        rm -f gpucore.*
    fi
fi
