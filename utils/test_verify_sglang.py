import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from utils import verify_sglang as verifier


EXPECTED = {"served_model_name": "test-model", "model_path": "/model",
            "tp_size": 4, "ep_size": 4, "speculative_algorithm": "EAGLE",
            "speculative_num_steps": 5, "speculative_num_draft_tokens": 6}


class VerifySglangTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.report = Path(self.tmp.name) / "validation.json"

    def run_check(self, info, names=("test-model",), expected=None):
        with patch.object(verifier, "fetch_json", side_effect=[info, {"data": [{"id": n} for n in names]}]):
            return verifier.verify("http://127.0.0.1:1", expected or EXPECTED, self.report)

    def test_http_endpoints_prove_identity_without_recording_api_secrets(self):
        paths = []
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                paths.append(self.path)
                payload = dict(EXPECTED, api_key="private") if self.path == "/get_server_info" else {"data": [{"id": "test-model"}]}
                self.send_response(200)
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            result = verifier.verify(f"http://127.0.0.1:{server.server_port}", EXPECTED, self.report)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
        self.assertTrue(result["valid"])
        self.assertEqual(paths, ["/get_server_info", "/v1/models"])
        self.assertEqual(result["checks"][2], {"field": "tp_size", "requested": 4, "observed": 4, "passed": True})
        self.assertNotIn("private", self.report.read_text())

    def test_nested_args_supported(self):
        self.assertTrue(self.run_check({"server_args": EXPECTED})["valid"])

    def test_mismatch_and_wrong_types_preserve_requested_and_observed(self):
        for key, value in (("served_model_name", "wrong"), ("tp_size", 8),
                           ("ep_size", 1), ("speculative_algorithm", None),
                           ("speculative_num_steps", 3), ("speculative_num_draft_tokens", 4),
                           ("model_path", "/other"), ("tp_size", "4")):
            with self.subTest(key=key, value=value):
                info = dict(EXPECTED, **{key: value})
                with self.assertRaisesRegex(ValueError, key):
                    self.run_check(info)
                report = json.loads(self.report.read_text())
                check = next(c for c in report["checks"] if c["field"] == key)
                self.assertFalse(report["valid"])
                self.assertEqual(check["observed"], value)
                self.assertEqual(check["requested"], EXPECTED[key])

    def test_missing_fields_cannot_validate_disabled_speculation(self):
        expected = dict(EXPECTED, speculative_algorithm=None,
                        speculative_num_steps=None, speculative_num_draft_tokens=None)
        self.assertTrue(self.run_check(expected, expected=expected)["valid"])
        info = copy.copy(expected)
        del info["speculative_num_steps"]
        with self.assertRaisesRegex(ValueError, "speculative_num_steps"):
            self.run_check(info, expected=expected)

    def test_unavailable_client_model_blocks_measurement(self):
        with self.assertRaisesRegex(ValueError, "client_model_available"):
            self.run_check(EXPECTED, names=("path-derived-name",))

    def test_transport_error_overwrites_previous_success(self):
        self.report.write_text('{"valid": true}')
        with patch.object(verifier, "fetch_json", side_effect=TimeoutError("server timeout")):
            with self.assertRaises(TimeoutError):
                verifier.verify("http://127.0.0.1:1", EXPECTED, self.report)
        report = json.loads(self.report.read_text())
        self.assertFalse(report["valid"])
        self.assertEqual(report["error"], "server timeout")

    def test_malformed_response_preserves_failed_report(self):
        with self.assertRaisesRegex(ValueError, "JSON objects"):
            self.run_check([])
        self.assertFalse(json.loads(self.report.read_text())["valid"])

    def test_invalid_caller_configuration_fails_before_http(self):
        with patch.object(verifier, "fetch_json") as fetch:
            with self.assertRaisesRegex(ValueError, "tp_size"):
                verifier.verify("unused", dict(EXPECTED, tp_size=True), self.report)
            fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
