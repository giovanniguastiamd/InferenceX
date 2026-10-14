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

Runner lifecycle supervision must stop the listener when the reservation ends
and remove its registration, including an idle runner that never accepted a job.
The container guard does not replace that supervisor. Test allocation loss,
runner termination, uploads and rootless image compatibility before declaring
this integration complete.
