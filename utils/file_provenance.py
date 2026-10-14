"""Verify local model/dataset bytes against an immutable file manifest."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", "")):
        raise ValueError("Manifest requires schema_version=1 and an immutable revision")
    if not isinstance(manifest.get("repo"), str) or not manifest["repo"]:
        raise ValueError("Manifest requires repository identity")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Manifest has no files")
    seen = set()
    for item in files:
        name = item["path"]
        if not isinstance(name, str) or not name or '\\' in name or ':' in name or name.startswith('/') or any(p in ('', '.', '..') for p in name.split('/')):
            raise ValueError("Unsafe manifest path")
        if name in seen:
            raise ValueError("Duplicate manifest path: " + name)
        seen.add(name)
        if type(item["size"]) is not int or item["size"] < 0:
            raise ValueError("Invalid file size")
        pattern = {"sha256": r"[0-9a-f]{64}", "git-sha1": r"[0-9a-f]{40}"}.get(item["algorithm"])
        if pattern is None or not re.fullmatch(pattern, item["digest"]):
            raise ValueError("Invalid file digest")


def verify_files(root, manifest_path, report_path, expected_revision, expected_manifest_digest,
                 check_alive=None):
    report = {"schema_version": 1, "valid": False, "files_verified": 0,
              "bytes_verified": 0, "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        raw = Path(manifest_path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        report["manifest_sha256"] = digest
        if digest != expected_manifest_digest:
            raise ValueError("Manifest digest mismatch")
        manifest = json.loads(raw)
        validate_manifest(manifest)
        report.update(repo=manifest["repo"], revision=manifest["revision"])
        if manifest["revision"] != expected_revision:
            raise ValueError("Manifest revision mismatch")
        root = Path(root).resolve(strict=True)
        checked = time.monotonic()
        if check_alive is not None:
            check_alive()
        for item in manifest["files"]:
            path = (root / item["path"]).resolve(strict=True)
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("File escapes manifest root: " + item["path"])
            before = path.stat()
            if before.st_size != item["size"]:
                raise ValueError("File size mismatch: " + item["path"])
            hasher = hashlib.sha256() if item["algorithm"] == "sha256" else hashlib.sha1()
            if item["algorithm"] == "git-sha1":
                hasher.update(b'blob ' + str(before.st_size).encode() + b'\0')
            with path.open('rb') as source:
                while chunk := source.read(8 * 1024 * 1024):
                    hasher.update(chunk)
                    if check_alive is not None and time.monotonic() - checked >= 10:
                        check_alive()
                        checked = time.monotonic()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("File changed during verification: " + item["path"])
            if hasher.hexdigest() != item["digest"]:
                raise ValueError("File hash mismatch: " + item["path"])
            report["files_verified"] += 1
            report["bytes_verified"] += item["size"]
            if report["files_verified"] % 32 == 0:
                print(f"Verified {report['files_verified']}/{len(manifest['files'])} files", flush=True)
        if check_alive is not None:
            check_alive()
        report["valid"] = True
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        Path(report_path).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    report = verify_files(args.root, args.manifest, args.report, args.revision, args.manifest_sha256)
    print(f"Verified {report['files_verified']} files at {report['revision']}", flush=True)


if __name__ == '__main__':
    main()
