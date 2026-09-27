"""Explain the two intentional dataflow golden-fingerprint changes."""

import importlib
import json
from pathlib import Path
import sys

phase, output = sys.argv[1:]
original = json.dumps
payloads = []


def capture(value, *args, **kwargs):
    if isinstance(value, dict) and set(value) == {"space", "problem"}:
        payloads.append(value)
    return original(value, *args, **kwargs)


json.dumps = capture
from dataflow.ops.mvau.test_source_semantics import _bound, _model
from dataflow.ops.test_graph_context import (
    Build,
    _mvau_model,
    InferShapes,
    InferDataTypes,
    bind_operations,
)

results = {}
operation = _bound(_model())
results["source_semantics"] = {
    "fingerprint": operation.local_problem_fingerprint,
    "payload": payloads[-1],
}
model = _mvau_model().transform(InferShapes()).transform(InferDataTypes())
operation = bind_operations(model, Build())[0]
results["graph_context"] = {
    "fingerprint": operation.local_problem_fingerprint,
    "payload": payloads[-1],
}
Path(output).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
print({key: value["fingerprint"] for key, value in results.items()})
