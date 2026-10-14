"""Prepare an isolated HF cache whose decoded rows match a pinned Weka export."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from utils.file_provenance import verify_files


def prepare(export, cache, destination, expected_manifest_digest):
    export, cache, destination = Path(export), Path(cache), Path(destination)
    # The independently pinned export binds Arrow contents to the corpus identity.
    manifest = json.loads((export / 'manifest.json').read_text())
    if manifest['repo'] != 'semianalysisai/cc-traces-weka-062126':
        raise ValueError('Only the full 062126 corpus is supported')
    revision = manifest['revision']
    destination.mkdir()  # Never overwrite an existing preparation.
    verify_files(export, export / 'manifest.json', destination / 'source-validation.json',
                 revision, expected_manifest_digest)
    shutil.copytree(export / 'traces', destination / 'traces')
    hub_repo = 'datasets--semianalysisai--cc-traces-weka-062126'
    arrow_repo = 'semianalysisai___cc-traces-weka-062126'
    snapshot = cache / 'hub' / hub_repo / 'snapshots' / revision
    target_snapshot = destination / 'hf/hub' / hub_repo / 'snapshots' / revision
    target_snapshot.mkdir(parents=True)
    for name in ('README.md', 'traces.jsonl'):
        shutil.copyfile(snapshot / name, target_snapshot / name)
    arrow_path = Path('datasets') / arrow_repo / 'default/0.0.0' / revision
    shutil.copytree(cache / arrow_path, destination / 'hf' / arrow_path)
    refs = destination / 'hf/hub' / hub_repo / 'refs'
    refs.mkdir()
    (refs / 'main').write_text(revision, encoding='utf-8')
    from datasets import load_dataset
    dataset = load_dataset(manifest['repo'], split='train', streaming=False,
                           trust_remote_code=False, cache_dir=str(destination / 'hf/datasets'))
    trace_files = [item for item in manifest['files'] if item['path'].startswith('traces/')]
    if len(dataset) != len(trace_files):
        raise ValueError('Offline Arrow row count differs from pinned export')
    for row, item in zip(dataset, trace_files):
        raw = (json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
        if hashlib.sha256(raw).hexdigest() != item['digest']:
            raise ValueError('Offline Arrow row differs from pinned export: ' + item['path'])
    files = []
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            hasher = hashlib.sha256()
            with path.open('rb') as source:
                while chunk := source.read(8 * 1024 * 1024):
                    hasher.update(chunk)
            files.append({'path': path.relative_to(destination).as_posix(),
                          'size': path.stat().st_size, 'algorithm': 'sha256',
                          'digest': hasher.hexdigest()})
    prepared = dict(manifest, files=files, source_export_manifest_sha256=expected_manifest_digest,
                    source='isolated offline HF cache; every decoded row checked against pinned export',
                    dataset_fingerprint=dataset._fingerprint, offline_cache=True)
    raw = (json.dumps(prepared, indent=2, ensure_ascii=False) + '\n').encode()
    (destination / 'manifest.json').write_bytes(raw)
    return {'path': str(destination), 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'revision': revision, 'trace_count': len(dataset), 'files': len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-manifest-sha256', required=True)
    args = parser.parse_args()
    import os
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['HF_HUB_CACHE'] = str(args.output / 'hf/hub')
    os.environ['HF_DATASETS_CACHE'] = str(args.output / 'hf/datasets')
    print(json.dumps(prepare(args.export, args.cache, args.output, args.export_manifest_sha256)))


if __name__ == '__main__':
    main()
