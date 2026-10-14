import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from utils.file_provenance import verify_files
from utils.prepare_dataset import export_rows


class FileProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.revision = 'a' * 40
        self.report = self.root / 'report.json'

    def corpus(self):
        path = self.root / 'corpus'
        result = export_rows([{'id': 'z', 'requests': [1]}, {'id': 'a', 'requests': [2]}],
                             path, 'org/dataset', self.revision, 'fp')
        return path, result

    def test_export_preserves_row_order_and_verified_bytes(self):
        path, result = self.corpus()
        report = verify_files(path, path / 'manifest.json', self.report, self.revision, result['manifest_sha256'])
        self.assertTrue(report['valid'])
        self.assertEqual(report['files_verified'], 2)
        rows = [json.loads(p.read_text())['id'] for p in sorted((path / 'traces').glob('*.json'))]
        self.assertEqual(rows, ['z', 'a'])

    def test_same_size_content_corruption_cannot_pass(self):
        path, result = self.corpus()
        trace = path / 'traces/00000000.json'
        trace.write_bytes(trace.read_bytes().replace(b'"z"', b'"y"'))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_files(path, path / 'manifest.json', self.report, self.revision, result['manifest_sha256'])
        self.assertFalse(json.loads(self.report.read_text())['valid'])

    def test_git_blob_hash_verifies_small_model_files(self):
        raw = b'{"config":1}\n'
        (self.root / 'config.json').write_bytes(raw)
        digest = hashlib.sha1(b'blob 13\0' + raw).hexdigest()
        manifest = {'schema_version': 1, 'repo': 'org/model', 'revision': self.revision,
                    'files': [{'path': 'config.json', 'size': 13, 'algorithm': 'git-sha1', 'digest': digest}]}
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        report = verify_files(self.root, path, self.report, self.revision, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(report['bytes_verified'], 13)

    def test_wrong_revision_and_modified_manifest_fail(self):
        path, result = self.corpus()
        for revision, digest, message in [('b' * 40, result['manifest_sha256'], 'revision'),
                                          (self.revision, 'c' * 64, 'digest')]:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                verify_files(path, path / 'manifest.json', self.report, revision, digest)

    def test_path_escape_is_rejected(self):
        path, _ = self.corpus()
        manifest_path = path / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['files'][0]['path'] = '../outside'
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            verify_files(path, manifest_path, self.report, self.revision, hashlib.sha256(manifest_path.read_bytes()).hexdigest())

    def test_duplicate_ids_and_overwrite_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            export_rows([{'id': 'x'}, {'id': 'x'}], self.root / 'duplicate', 'org/data', self.revision, 'fp')
        with self.assertRaises(FileExistsError):
            export_rows([{'id': 'x'}], self.root / 'duplicate', 'org/data', self.revision, 'fp')

    def test_allocation_loss_aborts_file_verification_with_evidence(self):
        path, result = self.corpus()
        check = Mock(side_effect=[None, ValueError('allocation lost')])
        with self.assertRaisesRegex(ValueError, 'allocation lost'):
            verify_files(path, path / 'manifest.json', self.report, self.revision,
                         result['manifest_sha256'], check)
        self.assertFalse(json.loads(self.report.read_text())['valid'])


if __name__ == '__main__':
    unittest.main()
