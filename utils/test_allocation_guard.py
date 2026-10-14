import json
import hashlib
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
        manifest = {'schema_version': 1, 'repo': 'test-model', 'revision': 'a' * 40,
                    'files': [{'path': 'one.safetensors', 'size': 7, 'algorithm': 'sha256',
                               'digest': hashlib.sha256(b'weights').hexdigest()}]}
        manifest_path = root / 'model-manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        from utils.prepare_dataset import export_rows
        dataset = root / 'dataset'
        exported = export_rows([{'id': 'trace'}], dataset, 'semianalysisai/cc-traces-weka-062126', 'e' * 40, 'fp')
        self.env = dict(ALLOCATION_JOB_ID='42', ALLOCATION_NODE='g10',
                        ALLOCATION_MODEL_PATH=str(model), ALLOCATION_HF_HOME=str(root),
                        GITHUB_WORKSPACE=str(root), GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1',
                        INFERENCEX_RUNTIME_ENV_VARS='OPENAI_API_KEY', IMAGE='test-image',
                        MODEL='test-model', TP='4', EP_SIZE='4', ALLOCATION_SPEC_ALGORITHM='EAGLE',
                        ALLOCATION_SPEC_STEPS='5', ALLOCATION_SPEC_DRAFT_TOKENS='6',
                        ALLOCATION_IMAGE_DIGEST='sha256:' + 'c' * 64,
                        ALLOCATION_MODEL_MANIFEST=str(manifest_path), ALLOCATION_MODEL_REVISION='a' * 40,
                        ALLOCATION_MODEL_MANIFEST_SHA256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                        ALLOCATION_DATASET_PATH=str(dataset), ALLOCATION_DATASET_REVISION='e' * 40,
                        ALLOCATION_DATASET_MANIFEST_SHA256=exported['manifest_sha256'])

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
        with patch.object(guard, 'check', side_effect=[{}] * 9 + [ValueError('allocation lost')]), \
             patch.object(guard, 'output', side_effect=['commit', '', 'sha256:' + 'd' * 64]), \
             patch.object(guard, 'resolve_image', return_value=('test-image@sha256:' + 'c' * 64, 'sha256:' + 'd' * 64)), \
             patch.object(guard, 'container_command', return_value=['docker', 'run']), \
             patch.object(guard.subprocess, 'Popen', return_value=child), \
             patch.object(guard.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'allocation lost'):
                guard.execute(self.env, 'recipe.sh')
        self.assertEqual(run.call_args_list[-1].args[0], ['docker', 'rm', '-f', 'infx-42-123-1'])
        child.terminate.assert_called_once()
        data = json.loads((Path(self.tmp.name) / 'allocation-evidence.json').read_text())
        self.assertNotIn('exit_code', data)
        self.assertEqual(data['status'], 'failed')
        self.assertEqual(data['error'], 'allocation lost')

    def test_registry_digest_is_resolved_independently_of_image_id(self):
        self.env['IMAGE'] = 'registry.example:5000/org/server:mutable'
        reference = 'registry.example:5000/org/server@sha256:' + 'c' * 64
        image_id = 'sha256:' + 'd' * 64
        with patch.object(guard.subprocess, 'run'), \
             patch.object(guard, 'output', return_value=json.dumps([{'Id': image_id, 'RepoDigests': [reference]}])):
            self.assertEqual(guard.resolve_image(self.env), (reference, image_id))

    def test_missing_or_unreported_registry_digest_is_rejected(self):
        with patch.object(guard.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'ALLOCATION_IMAGE_DIGEST'):
                guard.resolve_image(dict(self.env, ALLOCATION_IMAGE_DIGEST=''))
            run.assert_not_called()
        with patch.object(guard.subprocess, 'run'), \
             patch.object(guard, 'output', return_value='[{"Id": "sha256:x", "RepoDigests": []}]'):
            with self.assertRaisesRegex(ValueError, 'registry digest'):
                guard.resolve_image(self.env)

    def test_container_image_mismatch_never_starts_recipe(self):
        with patch.object(guard, 'check', return_value={}), \
             patch.object(guard, 'output', side_effect=['commit', '', 'unexpected-image']), \
             patch.object(guard, 'resolve_image', return_value=('test-image@sha256:' + 'c' * 64, 'sha256:' + 'd' * 64)), \
             patch.object(guard, 'container_command', return_value=['docker', 'create']), \
             patch.object(guard.subprocess, 'run'), \
             patch.object(guard.subprocess, 'Popen') as start:
            with self.assertRaisesRegex(ValueError, 'Created container image'):
                guard.execute(self.env, 'recipe.sh')
            start.assert_not_called()
        data = json.loads((Path(self.tmp.name) / 'allocation-evidence.json').read_text())
        self.assertEqual(data['container_image_id'], 'unexpected-image')
        self.assertEqual(data['status'], 'failed')

    def test_old_validation_cannot_authorize_new_successful_process(self):
        path = Path(self.tmp.name) / 'allocation-server-validation.json'
        path.write_text('{"valid": true}')
        child = Mock(returncode=0)
        child.poll.return_value = 0
        with patch.object(guard, 'check', return_value={}), \
             patch.object(guard, 'output', side_effect=['commit', '', 'sha256:' + 'd' * 64]), \
             patch.object(guard, 'resolve_image', return_value=('test-image@sha256:' + 'c' * 64, 'sha256:' + 'd' * 64)), \
             patch.object(guard, 'container_command', return_value=['docker', 'create']), \
             patch.object(guard.subprocess, 'run'), \
             patch.object(guard.subprocess, 'Popen', return_value=child):
            with self.assertRaises(FileNotFoundError):
                guard.execute(self.env, 'recipe.sh')
        self.assertFalse(path.exists())

    def test_success_retains_container_and_server_provenance(self):
        path = Path(self.tmp.name) / 'allocation-server-validation.json'
        child = Mock(returncode=0)
        child.poll.return_value = 0
        def started(*args, **kwargs):
            path.write_text('{"valid": true, "checks": []}')
            return child
        with patch.object(guard, 'check', return_value={'JobId': '42'}), \
             patch.object(guard, 'output', side_effect=['b' * 40, '', 'sha256:' + 'd' * 64]), \
             patch.object(guard, 'resolve_image', return_value=('test-image@sha256:' + 'c' * 64, 'sha256:' + 'd' * 64)), \
             patch.object(guard, 'container_command', return_value=['docker', 'create']), \
             patch.object(guard.subprocess, 'run'), \
             patch.object(guard.subprocess, 'Popen', side_effect=started):
            self.assertEqual(guard.execute(self.env, 'recipe.sh'), 0)
        data = json.loads((Path(self.tmp.name) / 'allocation-evidence.json').read_text())
        self.assertEqual(data['status'], 'completed')
        self.assertEqual(data['image_digest'], 'sha256:' + 'c' * 64)
        self.assertEqual(data['container_image_id'], 'sha256:' + 'd' * 64)
        self.assertEqual(data['server_expected']['speculative_num_draft_tokens'], 6)
        self.assertEqual(data['model']['files_verified'], 1)
        self.assertEqual(data['dataset']['revision'], 'e' * 40)

    def test_failed_preflight_still_records_execution_identity(self):
        with patch.object(guard, 'check', side_effect=ValueError('not owned')):
            with self.assertRaisesRegex(ValueError, 'not owned'):
                guard.execute(self.env, 'recipe.sh', True)
        data = json.loads((Path(self.tmp.name) / 'allocation-evidence.json').read_text())
        self.assertEqual(data['job_id'], '42')
        self.assertEqual(data['run_id'], '123')
        self.assertEqual(data['error'], 'not owned')

    def test_existing_container_blocks_execution(self):
        with patch.object(guard, 'check', return_value={}), \
             patch.object(guard, 'output', side_effect=['commit', 'foreign-container']), \
             patch.object(guard.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'Running containers'):
                guard.execute(self.env, 'recipe.sh')
            run.assert_not_called()

    def test_corrupt_model_or_extra_trace_blocks_before_image_pull(self):
        model = Path(self.env['ALLOCATION_MODEL_PATH']) / 'one.safetensors'
        for kind in ('model', 'dataset'):
            with self.subTest(kind=kind):
                if kind == 'model':
                    model.write_bytes(b'changed')
                else:
                    model.write_bytes(b'weights')
                    (Path(self.env['ALLOCATION_DATASET_PATH']) / 'traces/extra.json').write_text('{"id":"extra"}')
                with patch.object(guard, 'check', return_value={}), \
                     patch.object(guard, 'output', side_effect=['commit', '']), \
                     patch.object(guard, 'resolve_image') as pull:
                    with self.assertRaisesRegex(ValueError, 'hash mismatch|trace directory'):
                        guard.execute(self.env, 'recipe.sh')
                    pull.assert_not_called()


if __name__ == '__main__':
    unittest.main()
