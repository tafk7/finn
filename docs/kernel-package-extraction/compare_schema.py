"""Verify that datatype codec ownership alone explains the two golden hashes."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).parent
before = json.loads((root / "before-schema.json").read_text())
after = json.loads((root / "after-schema.json").read_text())
for key, old in before.items():
    payload = old["payload"]
    changes = []
    for field in payload["problem"]:
        if field["codec"] == "finn.dataflow.qonnx_datatype@1":
            assert field["name"] in ("accumulator_type", "output_type")
            field["codec"] = "finn.kernels.qonnx_datatype@1"
            changes.append(field["name"])
    assert changes == ["accumulator_type", "output_type"]
    assert payload == after[key]["payload"]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == after[key]["fingerprint"]
print("Both golden hashes follow solely from the two declared codec-identity changes")
