import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from utils import allocation_guard as guard


RECORD = 'JobId=42 UserId=alice JobState=RUNNING NodeList=g10 NumNodes=1 Exclusive=1 ReqGPUs=8'


class AllocationGuardTests(unittest.TestCase):
    def test_accepts_only_owned_running_exclusive_node(self):
        self.assertEqual(guard.validate_record(RECORD, '42', 'alice', 'g10')['ReqGPUs'], '8')
        for old, new in [('alice', 'bob'), ('RUNNING', 'PENDING'), ('g10', 'g11'),
                         ('Exclusive=1', 'Exclusive=0'), ('ReqGPUs=8', 'ReqGPUs=4'),
                         ('JobId=42', 'JobId=43'), ('NumNodes=1', 'NumNodes=2')]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                guard.validate_record(RECORD.replace(old, new), '42', 'alice', 'g10')

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        model = root / 'model'; model.mkdir()
        (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'one.safetensors'}}))
        (model / 'one.safetensors').write_bytes(b'weights')
        (model / '.hf-revision').write_text('a' * 40)
        self.env = dict(ALLOCATION_JOB_ID='42', ALLOCATION_NODE='g10',
                        ALLOCATION_MODEL_PATH=str(model), ALLOCATION_HF_HOME=str(root),
                        GITHUB_WORKSPACE=str(root), GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1',
                        INFERENCEX_RUNTIME_ENV_VARS='OPENAI_API_KEY', IMAGE='test-image')

    def test_validate_only_writes_evidence_without_docker(self):
        with patch.object(guard, 'check', return_value={'JobId': '42'}), \
             patch.object(guard, 'output', return_value='b' * 40), \
             patch.object(guard.subprocess, 'run') as run:
            self.assertEqual(guard.execute(self.env, 'recipe.sh', True), 0)
            run.assert_not_called()
        data = json.loads((Path(self.tmp.name) / 'allocation-evidence.json').read_text())
        self.assertEqual(data['model_shards_present'], 1)
        self.assertTrue(data['validate_only'])

    def test_incomplete_model_cannot_pass_preflight(self):
        (Path(self.env['ALLOCATION_MODEL_PATH']) / 'one.safetensors').unlink()
        with patch.object(guard, 'check', return_value={}), self.assertRaisesRegex(ValueError, 'shards'):
            guard.execute(self.env, 'recipe.sh', True)

    def test_command_preserves_readonly_weights_and_excludes_github_secret(self):
        self.env.update(GITHUB_TOKEN='not-for-container', OPENAI_API_KEY='also-not-in-command')
        with patch.object(os, 'getuid', return_value=1000, create=True), \
             patch.object(os, 'getgid', return_value=1000, create=True):
            argv = guard.container_command(self.env, 'infx-42-123-1', 'recipe.sh')
        self.assertIn('spur_job_id=42', argv)
        self.assertTrue(any(v.endswith('dst=/model,readonly') for v in argv))
        self.assertNotIn('GITHUB_TOKEN', argv)
        self.assertNotIn('also-not-in-command', argv)
        self.assertIn('OPENAI_API_KEY', argv)
        self.assertNotIn('--privileged', argv)

    def test_allocation_loss_stops_only_this_container(self):
        child = Mock()
        child.poll.side_effect = [None, None]
        with patch.object(guard, 'check', side_effect=[{}, {}, ValueError('allocation lost')]), \
             patch.object(guard, 'output', side_effect=['commit', '', 'sha256:image']), \
             patch.object(guard, 'container_command', return_value=['docker', 'run']), \
             patch.object(guard.subprocess, 'Popen', return_value=child), \
             patch.object(guard.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'allocation lost'):
                guard.execute(self.env, 'recipe.sh')
        self.assertEqual(run.call_args_list[-1].args[0], ['docker', 'rm', '-f', 'infx-42-123-1'])
        child.terminate.assert_called_once()
        data = json.loads((Path(self.tmp.name) / 'allocation-evidence.json').read_text())
        self.assertNotIn('exit_code', data)

    def test_existing_container_blocks_execution(self):
        with patch.object(guard, 'check', return_value={}), \
             patch.object(guard, 'output', side_effect=['commit', 'foreign-container']), \
             patch.object(guard.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Running containers'):
                guard.execute(self.env, 'recipe.sh')
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
