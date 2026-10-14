"""Compare the running SGLang API with the caller's requested configuration."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request


FIELDS = {
    "served_model_name": str,
    "model_path": str,
    "tp_size": int,
    "ep_size": int,
    "speculative_algorithm": (str, type(None)),
    "speculative_num_steps": (int, type(None)),
    "speculative_num_draft_tokens": (int, type(None)),
}


def validate_expected(expected):
    if not isinstance(expected, dict) or set(expected) != set(FIELDS):
        raise ValueError("Expected configuration must contain exactly the required M2e fields")
    for key, types in FIELDS.items():
        if type(expected[key]) not in (types if isinstance(types, tuple) else (types,)):
            raise ValueError(f"Invalid requested {key}")
    for key in ("served_model_name", "model_path"):
        if not expected[key]:
            raise ValueError(f"Empty requested {key}")
    for key in ("tp_size", "ep_size"):
        if expected[key] < 1:
            raise ValueError(f"Invalid requested {key}")
    enabled = expected["speculative_algorithm"] is not None
    for key in ("speculative_num_steps", "speculative_num_draft_tokens"):
        if enabled and (expected[key] is None or expected[key] < 1):
            raise ValueError(f"Enabled speculative decoding requires positive {key}")
        if not enabled and expected[key] is not None:
            raise ValueError(f"Disabled speculative decoding requires null {key}")
    if enabled and not expected["speculative_algorithm"]:
        raise ValueError("Empty speculative algorithm")


def fetch_json(url):
    # Local control-plane calls must not be redirected through a host HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=30) as response:
        return json.load(response)


def verify(base_url, expected, report_path):
    report = {"schema_version": 1, "valid": False, "checks": [],
              "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        validate_expected(expected)
        info = fetch_json(base_url.rstrip("/") + "/get_server_info")
        models = fetch_json(base_url.rstrip("/") + "/v1/models")
        if not isinstance(info, dict) or not isinstance(models, dict):
            raise ValueError("Server endpoints must return JSON objects")
        args = info.get("server_args", info)
        if not isinstance(args, dict):
            raise ValueError("Invalid server_args response")
        for key, requested in expected.items():
            observed = args.get(key)
            report["checks"].append({"field": key, "requested": requested,
                                     "observed": observed,
                                     "passed": key in args and type(observed) is type(requested)
                                     and observed == requested})
        data = models.get("data")
        if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
            raise ValueError("Invalid /v1/models response")
        names = [row.get("id") for row in data]
        report["checks"].append({"field": "client_model_available",
                                 "requested": expected["served_model_name"],
                                 "observed": names,
                                 "passed": expected["served_model_name"] in names})
        report["valid"] = all(item["passed"] for item in report["checks"])
        if not report["valid"]:
            failed = ", ".join(item["field"] for item in report["checks"] if not item["passed"])
            raise ValueError("SGLang configuration mismatch: " + failed)
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        # Persist selected fields only: API responses can contain credentials.
        Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    verify(args.url, json.loads(args.expected.read_text(encoding="utf-8")), args.report)
    print("M2e: SGLang effective configuration verified before warmup.")


if __name__ == "__main__":
    main()
