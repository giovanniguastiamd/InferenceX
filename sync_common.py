"""sync_common.py — keep files listed in .gitcommon in sync across branches.

Modes
-----
Pull (default)
    python sync_common.py [canonical-branch]
    Pull files FROM <canonical-branch> INTO the current branch.
    canonical-branch defaults to "main".

Push
    python sync_common.py --push [--branches b1 b2 ...]
    Push files FROM the current branch TO all other local branches
    (or to the explicitly listed ones).
    Asks for confirmation before touching any branch.
    Skips branches that already have the same content (no-op commit).

Options
-------
--dry-run   Print what would happen without doing anything.
"""

import argparse
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git(*args, capture=False):
    result = subprocess.run(
        ["git"] + list(args),
        check=True,
        text=True,
        capture_output=capture,
    )
    return result.stdout.strip() if capture else None


def current_branch():
    return git("rev-parse", "--abbrev-ref", "HEAD", capture=True)


def all_local_branches():
    out = git("branch", "--format=%(refname:short)", capture=True)
    return [b.strip() for b in out.splitlines() if b.strip()]


def read_gitcommon():
    gitcommon = Path(".gitcommon")
    if not gitcommon.exists():
        print(f"Error: {gitcommon} not found in {Path.cwd()}", file=sys.stderr)
        sys.exit(1)
    return [
        line.strip()
        for line in gitcommon.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


# ---------------------------------------------------------------------------
# Pull mode: bring files from canonical branch into current branch
# ---------------------------------------------------------------------------

def do_pull(canonical, dry_run):
    files = read_gitcommon()
    if not files:
        print("No files listed in .gitcommon.")
        return

    cur = current_branch()
    print(f"Current branch : {cur}")
    print(f"Syncing FROM   : {canonical}")
    print(f"Files          : {', '.join(files)}")
    print()

    if dry_run:
        print("[dry-run] Would run: git checkout", canonical, "--", *files)
        return

    git("checkout", canonical, "--", *files)

    print()
    print(f"Done. Files staged from '{canonical}'.")
    print("Review : git diff --cached")
    print(f"Commit : git commit -m \"chore: sync common files from {canonical}\"")


# ---------------------------------------------------------------------------
# Push mode: push files from current branch to other branches
# ---------------------------------------------------------------------------

def do_push(target_branches, dry_run):
    files = read_gitcommon()
    if not files:
        print("No files listed in .gitcommon.")
        return

    source = current_branch()
    available = all_local_branches()

    if target_branches:
        missing = [b for b in target_branches if b not in available]
        if missing:
            print(f"Error: branch(es) not found locally: {', '.join(missing)}", file=sys.stderr)
            sys.exit(1)
        targets = [b for b in target_branches if b != source]
    else:
        targets = [b for b in available if b != source]

    if not targets:
        print("No target branches to push to.")
        return

    print(f"Source branch  : {source}")
    print(f"Target branches: {', '.join(targets)}")
    print(f"Files          : {', '.join(files)}")
    print()

    if not dry_run:
        answer = input(f"Push {len(files)} file(s) to {len(targets)} branch(es)? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return
        print()

    # Stash any uncommitted changes so we can freely switch branches
    stashed = False
    if not dry_run:
        stash_out = subprocess.run(
            ["git", "stash", "push", "--include-untracked", "-m", "sync_common auto-stash"],
            capture_output=True, text=True,
        )
        if "No local changes" not in stash_out.stdout:
            stashed = True
            print(f"  [stash] {stash_out.stdout.strip()}")
            print()

    try:
        for branch in targets:
            print(f"--- {branch} ---")
            if dry_run:
                print(f"  [dry-run] checkout {branch}")
                print(f"  [dry-run] git checkout {source} -- {' '.join(files)}")
                print(f"  [dry-run] git commit -m 'chore: sync common files from {source}'")
                print(f"  [dry-run] checkout {source}")
                continue

            git("checkout", branch)
            git("checkout", source, "--", *files)

            # Check if there is anything to commit
            diff = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                capture_output=True,
            )
            if diff.returncode == 0:
                print(f"  No changes — already up to date.")
            else:
                git("commit", "-m", f"chore: sync common files from {source}")
                print(f"  Committed.")

    finally:
        git("checkout", source)
        if stashed:
            git("stash", "pop")
            print(f"  [stash] restored local changes.")

    print()
    print(f"Done. Back on '{source}'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sync files listed in .gitcommon across branches.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="Push files FROM current branch TO other branches.",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        metavar="BRANCH",
        help="(push mode) Target branches. Defaults to all local branches except current.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without doing it.",
    )
    parser.add_argument(
        "canonical",
        nargs="?",
        default="main",
        help="(pull mode) Branch to pull files from. Default: main.",
    )
    args = parser.parse_args()

    if args.push:
        do_push(args.branches, args.dry_run)
    else:
        do_pull(args.canonical, args.dry_run)


if __name__ == "__main__":
    main()
