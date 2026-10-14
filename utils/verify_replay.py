"""Validate the real AIPerf command and offline corpus before serving startup."""

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shlex


def verify(command, dataset_root, report_path):
    report = {'valid': False, 'submission_valid': False}
    try:
        root = Path(dataset_root)
        if os.environ.get('HF_HUB_OFFLINE') != '1' or os.environ.get('HF_DATASETS_OFFLINE') != '1':
            raise ValueError('Replay preflight requires offline HF mode')
        from aiperf.cli_commands.profile import app
        from aiperf.config.flags.resolver import resolve_config
        from aiperf.config.loader import build_benchmark_plan
        from aiperf.config.resolution.plan import BenchmarkRun
        from aiperf.config.resolution.resolvers import build_default_resolver_chain
        from aiperf.dataset.loader.base_hf_dataset import BaseHFDatasetLoader
        from aiperf.dataset.loader.semianalysis_cc_traces_weka import SemiAnalysisCCTracesWekaLoader
        args = shlex.split(command)
        if len(args) < 3 or args[1] != 'profile':
            raise ValueError('Expected an AIPerf profile command')
        _, bound, _ = app.parse_args(args[2:], exit_on_error=False, verbose=False)
        plan = build_benchmark_plan(resolve_config(bound.arguments['cli_config']))
        if len(plan.configs) != 1:
            raise ValueError('Allocation preflight requires exactly one benchmark configuration')
        run = BenchmarkRun(benchmark_id='allocation-preflight', cfg=plan.configs[0],
                           artifact_dir=Path('/tmp/inferencex-replay-preflight'))
        build_default_resolver_chain().resolve_all(run)
        outcome = run.resolved.scenario_outcome
        if outcome.submission_valid is not True:
            raise ValueError('AIPerf scenario did not accept this command for submission')
        dataset_config = run.cfg.get_default_dataset()
        if str(dataset_config.dataset) != 'semianalysis_cc_traces_weka_062126':
            raise ValueError('Allocation preflight requires the full public 062126 loader')
        manifest = json.loads((root / 'manifest.json').read_text())
        async def load():
            loader = SemiAnalysisCCTracesWekaLoader(run=run, hf_dataset_name=manifest['repo'])
            try:
                data = (await BaseHFDatasetLoader.load_dataset(loader))['dataset']
                selected = await asyncio.to_thread(loader._validate_rows, data)
                return data, len(selected)
            finally:
                await loader.http_client.close()
        dataset, selected_count = asyncio.run(load())
        traces = sorted((item for item in manifest['files'] if item['path'].startswith('traces/')),
                        key=lambda item: item['path'])
        if len(dataset) != len(traces):
            raise ValueError('AIPerf corpus row count differs from verified export')
        for row, item in zip(dataset, traces):
            raw = (json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode()
            if hashlib.sha256(raw).hexdigest() != item['digest']:
                raise ValueError('AIPerf corpus row differs from verified export: ' + item['path'])
        report.update(valid=True, submission_valid=True, trace_count=len(dataset),
                      selected_trace_count=selected_count,
                      revision=manifest['revision'], repo=manifest['repo'],
                      dataset_fingerprint=dataset._fingerprint,
                      loader=str(dataset_config.dataset), scenario=run.cfg.scenario)
        return report
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        Path(report_path).write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command', required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.command, args.dataset_root, args.report)))
