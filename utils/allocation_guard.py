"""Execute one container inside an existing exclusive Spur allocation."""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time


def output(argv):
    return subprocess.check_output(argv, text=True, timeout=30).strip()


def validate_record(record, job, user, node):
    fields = dict(re.findall(r"(\w+)=(\S+)", record))
    expected = {"JobId": job, "UserId": user, "JobState": "RUNNING",
                "NodeList": node, "NumNodes": "1", "Exclusive": "1"}
    for key, value in expected.items():
        if fields.get(key) != value:
            raise ValueError(f"Allocation mismatch: {key}={fields.get(key)!r}, expected {value!r}")
    if fields.get("ReqGPUs") != "8":
        raise ValueError("This execution path requires an exclusive eight-GPU allocation")
    return fields


def check(job, node):
    if not re.fullmatch(r"[0-9]+", job) or not re.fullmatch(r"[a-zA-Z0-9-]+", node):
        raise ValueError("Invalid allocation or node identifier")
    if socket.gethostname().split('.')[0] != node:
        raise ValueError("Runner is not on the requested node")
    return validate_record(output(["scontrol", "show", "job", job]), job, getpass.getuser(), node)


def container_command(env, name, recipe):
    workspace = str(Path(env["GITHUB_WORKSPACE"]).resolve())
    model = str(Path(env["ALLOCATION_MODEL_PATH"]).resolve())
    cache = str(Path(env["ALLOCATION_HF_HOME"]).resolve())
    if any(',' in p or '\n' in p for p in (workspace, model, cache)):
        raise ValueError("Unsupported mount path")
    args = ["docker", "run", "--rm", "--name", name,
            "--label", "owner=inferencex-allocation",
            "--label", "spur_job_id=" + env["ALLOCATION_JOB_ID"],
            "--security-opt", "seccomp=unconfined", "--network=host", "--ipc=host",
            "--shm-size=64g", "--device=/dev/kfd", "--device=/dev/dri",
            "--group-add=video", "--group-add=render",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=bind,src={workspace},dst=/workspace",
            "--mount", f"type=bind,src={model},dst=/model,readonly",
            "--mount", f"type=bind,src={cache},dst=/hf", "-w", "/workspace"]
    names = env["INFERENCEX_RUNTIME_ENV_VARS"].split() + """
        MODEL MODEL_PREFIX THINKING_MODE IMAGE FRAMEWORK PRECISION EXP_NAME
        RECIPE_FINGERPRINT TP PP_SIZE DCP_SIZE PCP_SIZE EP_SIZE DP_ATTENTION CONC
        ISL OSL MAX_MODEL_LEN RANDOM_RANGE_RATIO SPEC_DECODING DISAGG
        KV_OFFLOADING KV_OFFLOAD_BACKEND KV_OFFLOAD_BACKEND_METADATA
        TOTAL_CPU_DRAM_GB DURATION SCENARIO_TYPE SCENARIO_SUBDIR IS_AGENTIC
        IS_MULTINODE KEEP_LOGS RUN_EVAL EVAL_ONLY EVAL_FRAMEWORK EVAL_SUITE
        RESULT_DIR RESULT_FILENAME RESULT_FILENAME_BASE GPU_COUNT
        GPU_MONITOR_INTERVAL GPU_METRICS_CSV PORT RUNNER_NAME RUNNER_TYPE
        PYTHONDONTWRITEBYTECODE PYTHONPYCACHEPREFIX HF_TOKEN
        INFMAX_CONTAINER_WORKSPACE MODEL_DOWNLOAD_LOCK_TIMEOUT
        DSV41_MIN_CUDAGRAPH_CAPTURE_SIZE
    """.split()
    for key in dict.fromkeys(names):
        if key in env:
            args += ["-e", key]
    args += ["-e", "MODEL_PATH=/model", "-e", "HF_HOME=/hf", "-e", "HF_HUB_CACHE=/hf/hub",
             "-e", "HOME=/tmp/inferencex-home", "-e", "SLURM_JOB_ID=" + env["ALLOCATION_JOB_ID"],
             "-e", "SLURMD_NODENAME=" + env["ALLOCATION_NODE"],
             "--entrypoint", "bash", env["IMAGE"], recipe]
    return args


def execute(env, recipe, validate_only=False):
    job, node = env["ALLOCATION_JOB_ID"], env["ALLOCATION_NODE"]
    record = check(job, node)
    model = Path(env["ALLOCATION_MODEL_PATH"])
    index = json.loads((model / "model.safetensors.index.json").read_text())
    shards = set(index["weight_map"].values())
    if not shards or any(not (model / p).is_file() or (model / p).stat().st_size == 0 for p in shards):
        raise ValueError("Missing or empty model shards")
    revision = (model / ".hf-revision").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Model revision marker is invalid")
    workspace = Path(env["GITHUB_WORKSPACE"])
    cache = Path(env["ALLOCATION_HF_HOME"])
    if not cache.is_dir() or not os.access(cache, os.W_OK):
        raise ValueError("Hugging Face cache is not writable")
    name = f"infx-{job}-{env['GITHUB_RUN_ID']}-{env['GITHUB_RUN_ATTEMPT']}"
    if not re.fullmatch(r"infx-[0-9]+-[0-9]+-[0-9]+", name):
        raise ValueError("Invalid execution identity")
    evidence = {"allocation": record, "model_revision_marker": revision,
                "model_shards_present": len(shards), "node": node,
                "commit": output(["git", "rev-parse", "HEAD"]),
                "run_id": env["GITHUB_RUN_ID"], "attempt": env["GITHUB_RUN_ATTEMPT"],
                "validate_only": validate_only}
    path = workspace / "allocation-evidence.json"
    path.write_text(json.dumps(evidence, indent=2))
    if validate_only:
        print("Allocation, model shard presence and cache checks passed; no container started.")
        return 0
    # Fail rather than evict any running container, including a serving smoke.
    if output(["docker", "ps", "-q"]):
        raise ValueError("Running containers detected; finish the existing work first")
    subprocess.run(["docker", "pull", env["IMAGE"]], check=True, timeout=1800)
    check(job, node)
    image_id = output(["docker", "image", "inspect", "--format", "{{.Id}}", env["IMAGE"]])
    evidence["image_id"] = image_id
    env = dict(env, IMAGE=image_id)
    path.write_text(json.dumps(evidence, indent=2))
    command = container_command(env, name, recipe)
    child = None
    def interrupted(signum, frame):
        raise RuntimeError(f"Execution interrupted by signal {signum}")
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        child = subprocess.Popen(command, env=env)
        while child.poll() is None:
            check(job, node)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        check(job, node)
        evidence["exit_code"] = child.returncode
        return child.returncode
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        # Exact unique name only; never prune, cancel the allocation, or sweep containers.
        subprocess.run(["docker", "rm", "-f", name], timeout=30, check=False)
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait(timeout=30)
        evidence["finished_at"] = time.time()
        path.write_text(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(execute(dict(os.environ), args.recipe, args.validate_only))
