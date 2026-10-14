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
import sys
import time

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


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
    dataset = str(Path(env["ALLOCATION_DATASET_PATH"]).resolve())
    arrow_relative = ('semianalysisai___cc-traces-weka-062126/default/0.0.0/'
                      + env['ALLOCATION_DATASET_REVISION'])
    if any(',' in p or '\n' in p for p in (workspace, model, cache, dataset)):
        raise ValueError("Unsupported mount path")
    args = ["docker", "create", "--rm", "--name", name,
            "--label", "owner=inferencex-allocation",
            "--label", "spur_job_id=" + env["ALLOCATION_JOB_ID"],
            "--security-opt", "seccomp=unconfined", "--network=host", "--ipc=host",
            "--shm-size=64g", "--device=/dev/kfd", "--device=/dev/dri",
            "--tmpfs=/tmp/aiter_configs:rw,mode=1777",
            "--tmpfs=/tmp/inferencex-hf-datasets:rw,mode=1777",
            "--group-add=video", "--group-add=render",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=bind,src={workspace},dst=/workspace",
            "--mount", f"type=bind,src={model},dst=/model,readonly",
            "--mount", f"type=bind,src={dataset},dst=/dataset,readonly",
            "--mount", (f"type=bind,src={dataset}/hf/datasets/{arrow_relative},"
                        f"dst=/tmp/inferencex-hf-datasets/{arrow_relative},readonly"),
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
        ALLOCATION_SERVER_EXPECTED ALLOCATION_SERVER_REPORT
        ALLOCATION_MODEL_REVISION ALLOCATION_MODEL_MANIFEST_SHA256
        ALLOCATION_DATASET_REVISION ALLOCATION_DATASET_MANIFEST_SHA256
    """.split()
    for key in dict.fromkeys(names):
        if key in env:
            args += ["-e", key]
    args += ["-e", "MODEL_PATH=/model", "-e", "HF_HOME=/hf",
             "-e", "HF_HUB_CACHE=/dataset/hf/hub", "-e", "HF_DATASETS_CACHE=/tmp/inferencex-hf-datasets",
             "-e", "HF_HUB_OFFLINE=1", "-e", "HF_DATASETS_OFFLINE=1",
             "-e", "AITER_JIT_DIR=/tmp/inferencex-aiter-jit",
             "-e", "FLYDSL_RUNTIME_CACHE_DIR=/tmp/inferencex-flydsl-cache",
             "-e", "USER=inferencex", "-e", "LOGNAME=inferencex",
             "-e", "TORCHINDUCTOR_CACHE_DIR=/tmp/inferencex-torchinductor-cache",
             "-e", "PATH=/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/go/bin",
             "-e", "HOME=/tmp/inferencex-home", "-e", "SLURM_JOB_ID=" + env["ALLOCATION_JOB_ID"],
             "-e", "ALLOCATION_DATASET_PATH=/dataset",
             "-e", "SLURMD_NODENAME=" + env["ALLOCATION_NODE"],
             "--entrypoint", "bash", env["IMAGE"], recipe]
    return args


def resolve_image(env):
    digest = env.get("ALLOCATION_IMAGE_DIGEST", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("Real execution requires ALLOCATION_IMAGE_DIGEST=sha256:<64 hex digits>")
    image = env["IMAGE"]
    repository = image.split('@')[0]
    if ':' in repository.rsplit('/', 1)[-1]:
        repository = repository.rsplit(':', 1)[0]
    reference = repository + '@' + digest
    if '@' in image and image != reference:
        raise ValueError("Requested image reference disagrees with ALLOCATION_IMAGE_DIGEST")
    subprocess.run(["docker", "pull", reference], check=True, timeout=1800)
    metadata = json.loads(output(["docker", "image", "inspect", reference]))[0]
    if reference not in metadata.get("RepoDigests", []):
        raise ValueError("Pulled image does not report the requested registry digest")
    image_id = metadata.get("Id", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Invalid Docker image ID")
    return reference, image_id


def execute(env, recipe, validate_only=False):
    evidence = {"validate_only": validate_only, "status": "preflight",
                "run_id": env.get("GITHUB_RUN_ID"), "attempt": env.get("GITHUB_RUN_ATTEMPT"),
                "node": env.get("ALLOCATION_NODE"), "job_id": env.get("ALLOCATION_JOB_ID")}
    path = Path(env["GITHUB_WORKSPACE"]) / "allocation-evidence.json"
    try:
        (path.parent / "allocation-server-validation.json").unlink(missing_ok=True)
        (path.parent / "allocation-server-expected.json").unlink(missing_ok=True)
        (path.parent / "allocation-replay-validation.json").unlink(missing_ok=True)
        for label in ('model', 'dataset'):
            (path.parent / f'allocation-{label}-validation.json').unlink(missing_ok=True)
        (path.parent / 'allocation-dataset-manifest.json').unlink(missing_ok=True)
        return _execute(env, recipe, validate_only, evidence, path)
    except Exception as exc:
        evidence.update(status="failed", error=str(exc))
        raise
    finally:
        evidence["finished_at"] = time.time()
        path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


def _execute(env, recipe, validate_only, evidence, path):
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
    evidence.update({"allocation": record, "model_revision_marker": revision,
                "model_shards_present": len(shards), "node": node,
                "commit": output(["git", "rev-parse", "HEAD"]),
                "run_id": env["GITHUB_RUN_ID"], "attempt": env["GITHUB_RUN_ATTEMPT"],
                "validate_only": validate_only})
    path.write_text(json.dumps(evidence, indent=2))
    if validate_only:
        evidence["status"] = "validated_only"
        print("Allocation, model shard presence and cache checks passed; no container started.")
        return 0
    # Fail rather than evict any running container, including a serving smoke.
    if output(["docker", "ps", "-q"]):
        raise ValueError("Running containers detected; finish the existing work first")
    from utils.file_provenance import verify_files
    model_revision = env["ALLOCATION_MODEL_REVISION"]
    if revision != model_revision:
        raise ValueError("Model revision marker differs from the requested revision")
    evidence['model'] = verify_files(model, env['ALLOCATION_MODEL_MANIFEST'],
                                    workspace / 'allocation-model-validation.json', model_revision,
                                    env['ALLOCATION_MODEL_MANIFEST_SHA256'], lambda: check(job, node))
    check(job, node)
    dataset = Path(env['ALLOCATION_DATASET_PATH'])
    evidence['dataset'] = verify_files(dataset, dataset / 'manifest.json',
                                      workspace / 'allocation-dataset-validation.json',
                                      env['ALLOCATION_DATASET_REVISION'],
                                      env['ALLOCATION_DATASET_MANIFEST_SHA256'], lambda: check(job, node))
    manifest = json.loads((dataset / 'manifest.json').read_text())
    if manifest.get('offline_cache') is not True:
        raise ValueError('Dataset requires a verified isolated offline HF cache')
    declared_files = {item['path'] for item in manifest['files']}
    actual_files = {p.relative_to(dataset).as_posix() for p in dataset.rglob('*') if p.is_file()}
    if actual_files != declared_files | {'manifest.json'}:
        raise ValueError('Dataset offline cache differs from the verified manifest')
    if any(p.is_symlink() for p in dataset.rglob('*')):
        raise ValueError('Dataset offline cache must not contain symlinks')
    declared_traces = {p for p in declared_files if p.startswith('traces/')}
    actual_traces = {'traces/' + p.name for p in (dataset / 'traces').glob('*.json')}
    if declared_traces != actual_traces:
        raise ValueError('Dataset trace directory differs from the verified manifest')
    hub = dataset / 'hf/hub/datasets--semianalysisai--cc-traces-weka-062126'
    snapshots = {p.name for p in (hub / 'snapshots').iterdir()}
    arrow = dataset / 'hf/datasets/semianalysisai___cc-traces-weka-062126/default/0.0.0'
    if snapshots != {env['ALLOCATION_DATASET_REVISION']} or {p.name for p in arrow.iterdir()} != snapshots:
        raise ValueError('Dataset cache must contain only the requested revision')
    if (hub / 'refs/main').read_text() != env['ALLOCATION_DATASET_REVISION']:
        raise ValueError('Dataset offline main reference differs from requested revision')
    if evidence['dataset']['repo'] != 'semianalysisai/cc-traces-weka-062126':
        raise ValueError('The dedicated GLM baseline requires the full 062126 corpus')
    if evidence['model']['repo'] != env['MODEL']:
        raise ValueError('Model manifest repository differs from requested model')
    (workspace / 'allocation-dataset-manifest.json').write_bytes((dataset / 'manifest.json').read_bytes())
    check(job, node)
    # The caller's request is independent of the recipe command being validated.
    from utils.verify_sglang import validate_expected
    expected = {"served_model_name": env["MODEL"], "model_path": "/model",
                "tp_size": int(env["TP"]), "ep_size": int(env["EP_SIZE"]),
                "speculative_algorithm": env["ALLOCATION_SPEC_ALGORITHM"],
                "speculative_num_steps": int(env["ALLOCATION_SPEC_STEPS"]),
                "speculative_num_draft_tokens": int(env["ALLOCATION_SPEC_DRAFT_TOKENS"])}
    validate_expected(expected)
    (workspace / "allocation-server-expected.json").write_text(json.dumps(expected, indent=2))
    reference, image_id = resolve_image(env)
    check(job, node)
    evidence.update(image_id=image_id, image_digest=env["ALLOCATION_IMAGE_DIGEST"],
                    image_reference=reference, server_expected=expected)
    env = dict(env, IMAGE=image_id,
               ALLOCATION_SERVER_EXPECTED="/workspace/allocation-server-expected.json",
               ALLOCATION_SERVER_REPORT="/workspace/allocation-server-validation.json")
    path.write_text(json.dumps(evidence, indent=2))
    command = container_command(env, name, recipe)
    child = None
    def interrupted(signum, frame):
        raise RuntimeError(f"Execution interrupted by signal {signum}")
    old_handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        subprocess.run(command, env=env, check=True, timeout=60)
        observed_image = output(["docker", "inspect", "--format", "{{.Image}}", name])
        evidence["container_image_id"] = observed_image
        if observed_image != image_id:
            raise ValueError("Created container image differs from the pinned image")
        check(job, node)
        child = subprocess.Popen(["docker", "start", "--attach", name], env=env)
        while child.poll() is None:
            check(job, node)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        check(job, node)
        evidence["exit_code"] = child.returncode
        if child.returncode == 0:
            validation = json.loads((workspace / "allocation-server-validation.json").read_text())
            if validation.get("valid") is not True:
                raise ValueError("Missing successful effective-server validation")
            replay = json.loads((workspace / 'allocation-replay-validation.json').read_text())
            if replay.get('valid') is not True or replay.get('submission_valid') is not True:
                raise ValueError('Missing successful replay preflight')
            if (replay.get('revision') != env['ALLOCATION_DATASET_REVISION']
                    or replay.get('repo') != evidence['dataset']['repo']):
                raise ValueError('Replay preflight corpus identity differs from file proof')
            evidence['replay'] = replay
        evidence["status"] = "completed" if child.returncode == 0 else "failed"
        return child.returncode
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        # Exact unique name only; never prune, cancel the allocation, or sweep containers.
        try:
            cleanup = subprocess.run(["docker", "rm", "-f", name], timeout=30, check=False,
                                     capture_output=True, text=True)
            if cleanup.returncode and "No such container" not in cleanup.stderr:
                print(cleanup.stderr, file=sys.stderr, end="")
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    raise SystemExit(execute(dict(os.environ), args.recipe, args.validate_only))
