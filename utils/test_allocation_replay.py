import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


class AllocationReplayTests(unittest.TestCase):
    def test_pinned_files_and_tokenizer_reach_real_replay_command(self):
        root = Path(__file__).resolve().parents[1]
        git_bash = Path('C:/Program Files/Git/bin/bash.exe')
        bash = str(git_bash) if git_bash.exists() else shutil.which('bash')
        self.assertIsNotNone(bash)
        env = dict(os.environ)
        env.update(INFMAX_CONTAINER_WORKSPACE='/workspace', MODEL='org/model', PORT='30000',
                   CONC='4', DURATION='3600', AIPERF_FAILED_REQUEST_THRESHOLD='0.01',
                   AIPERF_LIVE_FAILED_REQUEST_THRESHOLD='0.01', AIPERF_TRACE_IDLE_GAP_CAP_SECONDS='300',
                   AGENTIC_WARMUP_GRACE_PERIOD='3600', AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES='0',
                   AIPERF_DYNAMO_SESSION_TIMEOUT_SECONDS='3600', AIPERF_EXPERIMENTAL_FAST='0',
                   AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID='false', AIPERF_UNSAFE_OVERRIDE='false',
                   AIPERF_USE_DYNAMO_CONV_AWARE_ROUTING='false', AIPERF_WARMUP_REQUESTS_PER_LANE='1',
                   ALLOCATION_MODEL_REVISION='a' * 40, MODEL_PATH='/model', ALLOCATION_DATASET_PATH='/dataset',
                   ALLOCATION_DATASET_REVISION='b' * 40, ALLOCATION_DATASET_MANIFEST_SHA256='c' * 64,
                   MSYS2_ARG_CONV_EXCL='*')
        with tempfile.TemporaryDirectory() as tmp:
            env.update(PYTHONPYCACHEPREFIX=tmp, AIPERF_RUNTIME_DIR=tmp)
            result = subprocess.run([bash, '-c',
                                     'source "$1"; resolve_trace_source; build_replay_cmd results; printf "REPLAY:%s\\n" "$REPLAY_CMD"',
                                     'bash', str(root / 'benchmarks/benchmark_lib.sh')],
                                    env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        command = next(line[7:] for line in result.stdout.splitlines() if line.startswith('REPLAY:'))
        args = shlex.split(command)
        self.assertEqual(args[args.index('--tokenizer') + 1], '/model')
        self.assertEqual(args[args.index('--input-file') + 1], '/dataset/traces')
        self.assertEqual(args[args.index('--custom-dataset-type') + 1], 'weka_trace')
        self.assertEqual(args[args.index('--model') + 1], 'org/model')
        self.assertEqual(args[args.index('--benchmark-duration') + 1], '3600')
        self.assertNotIn('--public-dataset', args)


if __name__ == '__main__':
    unittest.main()
