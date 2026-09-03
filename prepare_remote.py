#!/usr/bin/env python3
"""
prepare_remote.py — Set up a remote machine as a GitHub Actions runner
for InferenceX agentic benchmarks.

Steps performed:
  1. Check disk space and connectivity
  2. Fix permissions on HF hub cache
  3. Download HuggingFace model (skipped if already present)
  4. Download agentic traces dataset
  5. Configure and start the GitHub Actions runner
  6. Write MODEL_PATH into runner .env

Usage:
  python prepare_remote.py \\
    --ssh-host gbt350-odcdh5-wbb3 \\
    --runner-name mi355x-amds_01 \\
    --hf-token hf_xxx \\
    --github-repo giovanniguastiamd/InferenceX

  python prepare_remote.py --help
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULTS = {
    "ssh_config": r"C:\Users\gguasti\Documents\lavoro\config\config.txt",
    "runner_dir": "/mnt/it_share/runner",
    "runner_labels": "cluster:mi355x-amds,{runner_name},self-hosted,Linux,X64",
    "github_repo": "giovanniguastiamd/InferenceX",
    "model": "amd/GLM-5.2-MXFP4",
    "model_path": "/home/gguasti/models/GLM-5.2-MXFP4",
    "dataset": "semianalysisai/cc-traces-weka-062126",
    "hf_hub_cache": "/mnt/hf_hub_cache",
    "runner_log": "/tmp/runner_remote.log",
}

# ── SSH helpers ──────────────────────────────────────────────────────────────

def ssh(host: str, cmd: str, ssh_config: str, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a command on the remote host via SSH."""
    full = ["ssh", "-F", ssh_config, host, cmd]
    print(f"  $ {cmd}")
    result = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Remote command failed (exit {result.returncode}): {cmd}")
    return result


def ssh_background(host: str, cmd: str, ssh_config: str) -> None:
    """Launch a command on the remote host and detach immediately."""
    wrapped = f"setsid bash -c {shquote(cmd)} </dev/null &"
    full = ["ssh", "-F", ssh_config, "-n", host, wrapped]
    print(f"  [bg] $ {cmd}")
    subprocess.Popen(full, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def shquote(s: str) -> str:
    """Minimal single-quote escaping for bash."""
    return "'" + s.replace("'", "'\\''") + "'"

# ── gh CLI helpers ───────────────────────────────────────────────────────────

def gh(*args: str) -> str:
    """Run a gh CLI command without GITHUB_TOKEN in env."""
    env = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True, env=env, timeout=30
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh command failed: {' '.join(args)}\n{result.stderr}")
    return result.stdout.strip()


def get_registration_token(repo: str) -> str:
    print("  Fetching GitHub Actions registration token...")
    return gh(
        "api", "--method", "POST",
        f"repos/{repo}/actions/runners/registration-token",
        "--jq", ".token",
    )


def get_runner_labels(repo: str, runner_name: str) -> list[str]:
    data = gh(
        "api", f"repos/{repo}/actions/runners",
        "--jq", f'.runners[] | select(.name=="{runner_name}") | [.labels[].name]',
    )
    return json.loads(data) if data else []

# ── Steps ────────────────────────────────────────────────────────────────────

def step_check_connectivity(host: str, ssh_config: str) -> None:
    print("\n[1/6] Checking connectivity and disk space...")
    out = ssh(host, "whoami; hostname; df -h /", ssh_config).stdout
    # df output already printed by ssh()


def step_fix_hf_cache_permissions(host: str, ssh_config: str, hf_hub_cache: str) -> None:
    print(f"\n[2/6] Fixing permissions on {hf_hub_cache}...")
    ssh(host, f"sudo chmod -R 777 {hf_hub_cache} 2>/dev/null || true", ssh_config, check=False)
    ssh(host, f"mkdir -p {hf_hub_cache}", ssh_config)


def step_download_model(
    host: str, ssh_config: str, model: str, model_path: str, hf_token: str
) -> None:
    print(f"\n[3/6] Checking model at {model_path}...")
    result = ssh(
        host,
        f"test -d {shquote(model_path)} && ls -A {shquote(model_path)} | head -1",
        ssh_config,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"  Model already present at {model_path}, skipping download.")
        return

    print(f"  Downloading {model} → {model_path} (background)...")
    cmd = (
        f"mkdir -p {shquote(model_path)} && "
        f"HF_TOKEN={shquote(hf_token)} hf download {shquote(model)} "
        f"--local-dir {shquote(model_path)} "
        f"> /tmp/hf_model_download.log 2>&1"
    )
    ssh_background(host, cmd, ssh_config)
    print("  Download started in background. Monitor with: tail -f /tmp/hf_model_download.log")


def step_download_dataset(
    host: str, ssh_config: str, dataset: str, hf_hub_cache: str, hf_token: str
) -> None:
    print(f"\n[4/6] Downloading agentic traces dataset: {dataset}...")
    # Check if already cached
    cache_dir = f"{hf_hub_cache}/datasets--{dataset.replace('/', '--')}"
    result = ssh(host, f"test -d {shquote(cache_dir)}", ssh_config, check=False)
    if result.returncode == 0:
        print(f"  Dataset already cached at {cache_dir}, skipping.")
        return

    cmd = (
        f"HF_TOKEN={shquote(hf_token)} "
        f"HF_HUB_CACHE={shquote(hf_hub_cache)} "
        f"hf download --repo-type dataset {shquote(dataset)} "
        f"> /tmp/hf_dataset_download.log 2>&1"
    )
    ssh(host, cmd, ssh_config, timeout=600)
    print("  Dataset downloaded successfully.")


def step_configure_runner(
    host: str,
    ssh_config: str,
    runner_dir: str,
    runner_name: str,
    runner_labels: str,
    github_repo: str,
    model_path: str,
) -> None:
    print(f"\n[5/6] Configuring GitHub Actions runner '{runner_name}'...")

    # Stop existing runner if running
    pid_result = ssh(host, "pgrep -f 'Runner.Listener' | head -1", ssh_config, check=False)
    pid = pid_result.stdout.strip()
    if pid:
        print(f"  Stopping existing runner (PID {pid})...")
        ssh(host, f"kill {pid}", ssh_config, check=False)
        time.sleep(3)

    # Remove existing config if present
    runner_file = f"{runner_dir}/.runner"
    existing = ssh(host, f"test -f {shquote(runner_file)} && echo exists || echo absent", ssh_config, check=False)
    if "exists" in existing.stdout:
        print("  Removing existing runner configuration...")
        token = get_registration_token(github_repo)
        ssh(host, f"cd {shquote(runner_dir)} && ./config.sh remove --token {token}", ssh_config)

    # Register runner
    token = get_registration_token(github_repo)
    ssh(
        host,
        (
            f"cd {shquote(runner_dir)} && "
            f"./config.sh "
            f"--url https://github.com/{github_repo} "
            f"--token {token} "
            f"--name {shquote(runner_name)} "
            f"--labels {shquote(runner_labels)} "
            f"--unattended"
        ),
        ssh_config,
    )

    # Write MODEL_PATH to .env
    env_file = f"{runner_dir}/.env"
    print(f"  Writing MODEL_PATH to {env_file}...")
    ssh(
        host,
        (
            f"grep -v '^MODEL_PATH=' {shquote(env_file)} > /tmp/runner_env.tmp 2>/dev/null || true && "
            f"echo 'MODEL_PATH={model_path}' >> /tmp/runner_env.tmp && "
            f"mv /tmp/runner_env.tmp {shquote(env_file)}"
        ),
        ssh_config,
    )

    # Start runner
    print("  Starting runner in background...")
    run_cmd = f"cd {shquote(runner_dir)} && ./run.sh >> /tmp/runner_remote.log 2>&1"
    ssh_background(host, run_cmd, ssh_config)

    time.sleep(6)

    # Verify
    pid_result = ssh(host, "pgrep -f 'Runner.Listener' | head -1", ssh_config, check=False)
    if pid_result.stdout.strip():
        print(f"  Runner started (PID {pid_result.stdout.strip()})")
    else:
        print("  WARNING: runner process not found after start. Check /tmp/runner_remote.log")


def step_verify(host: str, ssh_config: str, github_repo: str, runner_name: str) -> None:
    print(f"\n[6/6] Verifying runner status on GitHub...")
    try:
        data = gh(
            "api", f"repos/{github_repo}/actions/runners",
            "--jq", f'.runners[] | select(.name=="{runner_name}") | {{name, status, labels: [.labels[].name]}}',
        )
        info = json.loads(data)
        status = info.get("status", "unknown")
        labels = info.get("labels", [])
        mark = "✓" if status == "online" else "✗"
        print(f"  {mark} Runner '{runner_name}': {status}")
        print(f"    Labels: {', '.join(labels)}")
        if status != "online":
            print("  WARNING: runner is not online yet. It may take a few seconds.")
    except Exception as e:
        print(f"  Could not verify runner status: {e}")

# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare a remote machine as an InferenceX GitHub Actions runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ssh-host", required=True,
                   help="SSH host alias from ssh config (e.g. gbt350-odcdh5-wbb3)")
    p.add_argument("--ssh-config", default=DEFAULTS["ssh_config"],
                   help="Path to SSH config file")
    p.add_argument("--runner-name", required=True,
                   help="GitHub Actions runner name (e.g. mi355x-amds_01)")
    p.add_argument("--runner-dir", default=DEFAULTS["runner_dir"],
                   help="Path to the runner installation on the remote host")
    p.add_argument("--runner-labels", default=None,
                   help="Comma-separated runner labels. Default includes runner-name automatically.")
    p.add_argument("--github-repo", default=DEFAULTS["github_repo"],
                   help="GitHub repo in owner/name format")
    p.add_argument("--hf-token", required=True,
                   help="HuggingFace token for model/dataset download")
    p.add_argument("--model", default=DEFAULTS["model"],
                   help="HuggingFace model ID to download")
    p.add_argument("--model-path", default=DEFAULTS["model_path"],
                   help="Local path on remote host where model is stored")
    p.add_argument("--dataset", default=DEFAULTS["dataset"],
                   help="HuggingFace dataset ID for agentic traces")
    p.add_argument("--hf-hub-cache", default=DEFAULTS["hf_hub_cache"],
                   help="HF_HUB_CACHE path on the remote host")
    p.add_argument("--skip-model", action="store_true",
                   help="Skip model download (useful if already present)")
    p.add_argument("--skip-dataset", action="store_true",
                   help="Skip dataset download")
    p.add_argument("--skip-runner", action="store_true",
                   help="Skip runner configuration and start")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build labels
    if args.runner_labels:
        labels = args.runner_labels
    else:
        labels = DEFAULTS["runner_labels"].format(runner_name=args.runner_name)

    print("=" * 60)
    print("InferenceX Remote Machine Setup")
    print("=" * 60)
    print(f"  Host:        {args.ssh_host}")
    print(f"  Runner:      {args.runner_name}  [{labels}]")
    print(f"  Model:       {args.model} → {args.model_path}")
    print(f"  Dataset:     {args.dataset}")
    print(f"  HF cache:    {args.hf_hub_cache}")
    print(f"  GitHub repo: {args.github_repo}")

    step_check_connectivity(args.ssh_host, args.ssh_config)
    step_fix_hf_cache_permissions(args.ssh_host, args.ssh_config, args.hf_hub_cache)

    if not args.skip_model:
        step_download_model(args.ssh_host, args.ssh_config, args.model, args.model_path, args.hf_token)

    if not args.skip_dataset:
        step_download_dataset(args.ssh_host, args.ssh_config, args.dataset, args.hf_hub_cache, args.hf_token)

    if not args.skip_runner:
        step_configure_runner(
            args.ssh_host, args.ssh_config,
            args.runner_dir, args.runner_name, labels,
            args.github_repo, args.model_path,
        )

    step_verify(args.ssh_host, args.ssh_config, args.github_repo, args.runner_name)

    print("\n" + "=" * 60)
    print("Setup complete.")
    print(f"Monitor runner log: ssh -F {args.ssh_config} {args.ssh_host} 'tail -f /tmp/runner_remote.log'")
    if not args.skip_model:
        print(f"Monitor model download: ssh -F {args.ssh_config} {args.ssh_host} 'tail -f /tmp/hf_model_download.log'")
    print("=" * 60)


if __name__ == "__main__":
    main()
