"""Capture complete construction values and emitted sources for a relocation comparison."""

import argparse
import dataclasses
from enum import Enum
import importlib
import json
import gzip
from pathlib import Path
import tempfile

from qonnx.core.datatype import DataType

parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=("before", "after"))
parser.add_argument("output", type=Path)
args = parser.parse_args()
old = args.phase == "before"
prefix = "finn.dataflow" if old else "finn.kernels"
build = importlib.import_module(prefix + ".artifacts.build")
store_mod = importlib.import_module(prefix + ".artifacts.store")
dotp = importlib.import_module(
    "finn.dataflow.kernels.dotp_axi_minimal" if old else "finn.kernels.dotp"
)
mvau = importlib.import_module(
    "finn.dataflow.kernels.mvau" if old else "finn.kernels.mvau"
)
target = importlib.import_module(
    "finn.dataflow.kernels.target" if old else "finn.kernels.target"
)
helpers = importlib.import_module(
    "dataflow.model.contract_helpers" if old else "kernels.helpers"
)
root = Path.cwd()


def encode(v):
    if isinstance(v, Enum):
        return {"enum": type(v).__module__ + "." + type(v).__qualname__, "name": v.name}
    if dataclasses.is_dataclass(v):
        return {
            "type": type(v).__module__ + "." + type(v).__qualname__,
            "fields": {
                f.name: encode(getattr(v, f.name)) for f in dataclasses.fields(v)
            },
        }
    if isinstance(v, (tuple, list)):
        return [encode(x) for x in v]
    if isinstance(v, dict):
        return {k: encode(x) for k, x in v.items()}
    if v is None or isinstance(v, (bool, int, str, float)):
        return v
    if type(v).__module__.startswith("qonnx.core.datatype"):
        return {"qonnx_dtype": v.name}
    raise TypeError((type(v), repr(v)))


records = {}
with tempfile.TemporaryDirectory(prefix="kernel-equivalence-") as temp:
    store = store_mod.ArtifactStore(Path(temp))
    if old:
        roots = {"finn": root, "finnlib": root / "deps/finnlib"}
        templates = (root / "src/finn/dataflow/kernels/matmul/templates",)
    else:
        resources = importlib.import_module("finn.kernels.resources")
        roots = {"kernels": resources.resource_root(), "finnlib": root / "deps/finnlib"}
        templates = (resources.template_root(),)

    def record(key, req, assembly=None):
        prepared = build.prepare_module_build(
            req, roots=roots, template_roots=templates, blobs=store
        )
        rendered = build.render_module_sources(prepared, store)
        records[key] = {
            "requirements": encode(req),
            "assembly": encode(assembly),
            "prepared_abi": encode(prepared.abi),
            "entry_point": prepared.abi.entry_point,
            "sources": {p: b.decode() for p, b in rendered.contents},
            "source_order": [p for p, b in rendered.contents],
            "fingerprint": build.module_build_fingerprint(req),
        }

    for dsp in target.DspBlock:
        for pumping in (False, True):
            facts = dict(
                pe=2,
                simd=4,
                activation_dtype=DataType["INT3"],
                weights_dtype=DataType["INT3"],
                result_dtype=DataType["INT9"],
                target_dsp=dsp,
                segment_length=0,
            )
            point = helpers.point_for(
                dotp.DotpAxiKernel, facts, compute_pumping=pumping
            )
            record(
                f"dotp/{dsp.value}/{pumping}",
                helpers.value(point.physical.accepted_answer),
            )
            for delivery in mvau.WeightDelivery:
                options = dict(
                    repetitions=4,
                    matrix_width=8,
                    matrix_height=4,
                    activation_dtype=facts["activation_dtype"],
                    weights_dtype=facts["weights_dtype"],
                    pe=2,
                    simd=4,
                    target_dsp=dsp,
                    segment_length=0,
                    compute_pumping=pumping,
                    weight_delivery=delivery,
                )
                if delivery is mvau.WeightDelivery.CYCLIC:
                    options["weights"] = [
                        [(i + j) % 8 - 4 for j in range(8)] for i in range(4)
                    ]
                a = mvau.mvau_assembly(**options)
                record(
                    f"mvau/{dsp.value}/{pumping}/{delivery.value}", a.requirements, a
                )
    harness = importlib.import_module(
        "dataflow.kernels.rtlsim.pure_dot_product_numeric"
        if old
        else "kernels.rtlsim.pure_dot_product_numeric"
    )
    for c in harness.STRESS_CASES:
        # Same full-range width proof as the numerical harness; no latency model.
        activation, weight = DataType[c.activation], DataType[c.weight]
        products = [
            int(a) * int(w)
            for a in (activation.min(), activation.max())
            for w in (weight.min(), weight.max())
        ]
        low, high = min(products) * c.width, max(products) * c.width
        bits = next(
            b for b in range(1, 129) if -(1 << (b - 1)) <= low and high < (1 << (b - 1))
        )
        facts = dict(
            pe=c.pe,
            simd=c.simd,
            activation_dtype=activation,
            weights_dtype=weight,
            result_dtype=DataType[f"INT{bits}"],
            target_dsp=c.target,
            segment_length=c.segment,
        )
        point = helpers.point_for(dotp.DotpAxiKernel, facts, compute_pumping=c.pumping)
        record("stress/" + c.label, helpers.value(point.physical.accepted_answer))
payload = (json.dumps(records, indent=2, sort_keys=True) + "\n").encode()
args.output.write_bytes(
    gzip.compress(payload, mtime=0) if args.output.suffix == ".gz" else payload
)
print(f"Recorded {len(records)} complete constructions and source sets")
