"""Export a revision-pinned HF Weka corpus for AIPerf's native file loader."""

import argparse
import hashlib
import json
from pathlib import Path
import re


def export_rows(rows, root, repo, revision, dataset_fingerprint):
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Dataset revision must be an immutable commit')
    root = Path(root)
    root.mkdir()  # Never overwrite an existing corpus.
    trace_dir = root / 'traces'
    trace_dir.mkdir()
    files = []
    ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('Dataset row requires a trace id')
        if row['id'] in ids:
            raise ValueError('Duplicate trace id: ' + row['id'])
        ids.add(row['id'])
        # Filename order preserves HF row order for filter-then-cap selection.
        name = f'traces/{index:08d}.json'
        raw = (json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
        (root / name).write_bytes(raw)
        files.append({'path': name, 'size': len(raw), 'algorithm': 'sha256',
                      'digest': hashlib.sha256(raw).hexdigest()})
    if not files:
        raise ValueError('Empty dataset')
    manifest = {'schema_version': 1, 'repo': repo, 'revision': revision,
                'source': 'datasets.load_dataset(revision=<commit>), train split',
                'dataset_fingerprint': dataset_fingerprint, 'files': files,
                'trace_count': len(files)}
    raw = (json.dumps(manifest, indent=2, ensure_ascii=False) + '\n').encode()
    (root / 'manifest.json').write_bytes(raw)
    return {'manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'trace_count': len(files), 'path': str(root), 'revision': revision, 'repo': repo}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', args.revision):
        raise ValueError('Dataset revision must be an immutable commit')
    from datasets import load_dataset
    dataset = load_dataset(args.repo, revision=args.revision, split='train', streaming=False)
    print(json.dumps(export_rows(dataset, args.output, args.repo, args.revision, dataset._fingerprint)))


if __name__ == '__main__':
    main()
