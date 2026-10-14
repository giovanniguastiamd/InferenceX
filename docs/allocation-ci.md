# Existing-allocation CI

**English** | [中文](allocation-ci_zh.md)

The fork's `allocation-benchmark.yml` runs one GLM-5.2 FP4 SGLang AgentX
configuration (TP4/EP4, HiCache, concurrency 4), selected by the existing matrix
generator. The reusable benchmark workflow and recipe remain authoritative.

Supply a running, exclusive eight-GPU Spur job and its single node. Register a
temporary runner named `mi355x-amds_<unique-id>` on that node with
`--no-default-labels --labels allocation-<node>-<job-id> --ephemeral`.
Do not apply the generic pool label. One registration accepts one workflow job;
register a new runner for another repetition.

The default is validation only: ownership, node, allocation, cache access and
presence of model shards. It does not launch Docker or claim a measured baseline.
The `.hf-revision` marker is recorded, but it does not prove every weight file
matches that revision. Full provenance and effective-server validation remain
required before baseline measurements are accepted.

For a branch-only pilot before the workflow is on the default branch, commit
an explicit `.github/allocation-request.json` on `tuning/allocation-ci`. Its
`job-id`, `node` and boolean `validate-only` select the run; changing that file
triggers CI on that branch. Never reuse an expired job ID. Other branches are
not activated by this push trigger. Manual dispatch is also supported once
the workflow is available on the repository's default branch.

The allocation path skips generic pre/post resource cleanup and git repair.
`utils/allocation_guard.py` checks ownership before launch and every ten seconds
while the container runs. It never acquires or cancels an allocation. On failure
or cancellation it removes only its uniquely named container. An existing
running container blocks a benchmark instead of being removed.

Weights are mounted read-only at `/model`; the caller's persistent HF cache is
mounted at `/hf`. The container uses the operator's UID/GID and explicit GPU
devices, without privileged mode. Allocation evidence is uploaded separately.
Recipe artifacts retain the reusable workflow's standard upload paths.

The pinned ROCm image contains unreadable AITER build-cache files. The allocation
container sets the native `AITER_JIT_DIR=/tmp/inferencex-aiter-jit` override to
build in a private writable directory, avoiding AITER's automatic copy of that
cache. Kernels and image files are unchanged. Compilation is part of startup,
before readiness and measurement. Cleanup tolerates a container already removed
by Docker's `--rm`; other cleanup errors remain visible.

## Effective configuration gate (M2e)

Real execution requires `image-digest: sha256:<64 hex digits>` in the request
or dispatch input. Resolve it from the intended registry image before submitting
the run; a tag or local Docker image ID is insufficient. The guard pulls the
repository by that digest, checks `RepoDigests`, creates the container using
the resolved image ID, and checks the container's `.Image` before starting it.

The dedicated workflow explicitly requests EAGLE, five speculative steps and
six draft tokens. The six-token value includes the verification token and must
not be confused with five draft steps. After readiness, before AgentX warmup,
the recipe calls `utils/verify_sglang.py`. It reads `/get_server_info` and
`/v1/models`, checking served model name, model path, TP/EP, speculative
algorithm, steps and draft-token count against the caller's request. Missing
fields, wrong types, unavailable endpoints and mismatches fail the job.
Flat fields and a nested `server_args` object are supported; an unsupported
API layout must be inspected on the pinned image rather than guessed.

The allocation artifact contains `allocation-evidence.json`,
`allocation-server-expected.json` and, once the gate runs,
`allocation-server-validation.json`. Validation stores only selected fields,
with requested/observed values and failure details; raw API responses may
contain credentials and are not saved. Old reports are removed at the start
of an attempt. A zero process exit without a successful new validation report
does not pass the guard. Preflight and allocation-loss failures retain run
identity and an error in the evidence document.

Local tests cover API comparison and lifecycle branches; they do not establish
compatibility of the pinned image or a measured baseline. Ingestion and board
integration remain pending. File provenance is checked as described below.

## Model and dataset file provenance

The dedicated workflow pins model revision
`386bd0e4ec821f7b07975701cec3c3b953a5576a` and the byte digest of
[`campaign/glm52-model-manifest.json`](../campaign/glm52-model-manifest.json).
The manifest comes from the public HF revision API and lists all 282 shards
plus configuration/tokenizer files. The guard hashes every listed file before
container creation, using SHA256 for LFS files and Git blob SHA1 otherwise.
The `.hf-revision` marker must also match the caller's revision. Changes,
missing files, manifest mismatches and allocation loss stop preparation.
This reads about 438 GB per attempt; an earlier validation report is not reused
as proof that shared model files still match.

The pinned AIPerf checkout has no CLI revision input for this public Weka
loader. Export the `train` split from dataset revision
`23f152f6f0f9399a85901b89a6458def0ef16729` using
[`utils/prepare_dataset.py`](../utils/prepare_dataset.py), in an environment
with `datasets` installed:

```bash
python utils/prepare_dataset.py \
  --repo semianalysisai/cc-traces-weka-062126 \
  --revision 23f152f6f0f9399a85901b89a6458def0ef16729 \
  --output /share_nfs/gguasti/inferencex/datasets/cc-traces-weka-062126-23f152f6f0f9399a85901b89a6458def0ef16729
```

The destination must not exist. The export preserves row order in numbered
JSON filenames and emits `manifest.json` plus its SHA256. Supply that digest
as `dataset-manifest-sha256` in the request/dispatch. The guard checks all
listed bytes and rejects extra trace JSON files, mounts the corpus read-only
at `/dataset`, and the recipe selects AIPerf's native
`--input-file /dataset/traces --custom-dataset-type weka_trace` path. No mutable
public-dataset download occurs in this allocation path. AIPerf retains its
existing Weka reconstruction, context filtering and trace selection behavior;
end-to-end consumption remains to be demonstrated by the real CI run.
The replay tokenizer is also loaded from the verified `/model` mount instead
of resolving the HF repository's current revision independently.

`allocation-model-validation.json` and `allocation-dataset-validation.json`
are uploaded with the allocation evidence. The preparation on 2026-10-14
verified all 288 model files and all 393 exported traces. These are file
integrity checks, not serving or performance measurements.

Runner lifecycle supervision must stop the listener when the reservation ends
and remove its registration, including an idle runner that never accepted a job.
The container guard does not replace that supervisor. Test allocation loss,
runner termination, uploads and rootless image compatibility before declaring
this integration complete.
