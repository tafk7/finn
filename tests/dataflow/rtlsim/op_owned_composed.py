# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Build accepted source Ops, then observe their stored production component."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path

from finn.dataflow._engine import Decided
from finn.dataflow.artifacts.build import (
    materialize_module_sources,
    module_source_derivation,
    portable_module_component,
    prepare_module_build,
)
from finn.dataflow.artifacts.formats import RtlModuleDirectory
from finn.dataflow.artifacts.formats.rtl_module import RtlModuleOptions
from finn.dataflow.artifacts.packaging import Target, plan_package
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.kernels.physical import capture_kernel_realization
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.persistence import CommitmentStage
from finn.dataflow.ops.physical import (
    capture_op_physical,
    install_physical_component,
    materialize_build_request,
    prepare_build_request,
)

from dataflow.physical_fixture import (
    ACTIVATIONS,
    ALTERNATE_ACTIVATIONS,
    ALTERNATE_WEIGHTS,
    WEIGHTS,
    configure,
    packed_activation,
    packed_output,
    packed_weights,
    roots,
    source_model,
    template_roots,
)
from dataflow.rtlsim.rtl_transport import drive_observed


def _record(value):
    """Lossless readable evidence for the closed immutable artifact records."""
    if isinstance(value, Enum):
        return {
            "enum": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": value.value,
        }
    if is_dataclass(value):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _record(getattr(value, field.name)) for field in fields(value)},
        }
    if isinstance(value, tuple):
        return [_record(item) for item in value]
    if value is None or type(value) in (bool, str, int, float):
        return value
    raise TypeError(f"unexpected artifact evidence value {type(value)}")


class CountingStore(ArtifactStore):
    def __init__(self, root):
        super().__init__(root)
        self.published = []
        self.hits = []

    def publish(self, derivation, workspace, **kwargs):
        result = super().publish(derivation, workspace, **kwargs)
        self.published.append((derivation.kind, result.key))
        return result

    def lookup(self, derivation):
        result = super().lookup(derivation)
        if result is not None:
            self.hits.append((derivation.kind, result.key))
        return result


def _package(component, store):
    package = plan_package(
        RtlModuleDirectory(),
        component,
        Target("xcvc1902-vsva2197-2MP-e-S"),
        RtlModuleOptions(),
        store,
    )
    found = store.lookup(package.derivation)
    if found is not None:
        return found
    workspace = store.workspace(package.derivation)
    for name, content in package.contents:
        path = workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return store.publish(package.derivation, workspace, entry_points=(component.entry_point,))


def build_artifacts(directory):
    store = CountingStore(directory / "store")
    components, requests, associations = [], [], []
    leaf_components = {"replay": [], "compute": []}
    packages = []
    for rows in (1, 2):
        for invocation, weights in enumerate((WEIGHTS, ALTERNATE_WEIGHTS)):
            prefix = f"r{rows}_inv{invocation}"
            model, build, context = source_model(prefix=prefix, rows=rows, weights=weights)
            for index, node in enumerate(list(model.graph.node)):
                operation = configure(
                    MvauDataflowOp(node).bind(model, build, graph_context=context)
                )
                operation = operation.commit(
                    model, build, require=CommitmentStage.PHYSICAL, graph_context=context
                )
                capture = capture_op_physical(operation)
                request = prepare_build_request(
                    operation,
                    capture,
                    model=model,
                    build=build,
                    graph_context=context,
                    roots=roots(),
                    template_roots=template_roots(),
                    blobs=store,
                )
                component = materialize_build_request(
                    operation, request, model=model, build=build, graph_context=context, store=store
                )
                instance = install_physical_component(
                    operation,
                    request,
                    component,
                    outer_instance_id=f"{prefix}_{index}",
                    model=model,
                    build=build,
                    graph_context=context,
                    store=store,
                )
                components.append(component)
                requests.append(request)
                associations.append(
                    {
                        "outer_instance_id": instance.outer_instance_id,
                        "source_scope": operation.recorded_scope_id(),
                        "prepared_fingerprint": instance.prepared_fingerprint,
                        "source_artifact": _record(component.artifact),
                        "rows": rows,
                        "weights": weights,
                    }
                )
                packages.append(_package(component, store))
                for role in ("replay", "compute"):
                    kernel = operation.selected_design().kernel(role)
                    assert isinstance(kernel, Decided)
                    facts = capture_kernel_realization(kernel.value)
                    prepared = prepare_module_build(
                        facts.requirements, roots=roots(), template_roots=(), blobs=store
                    )
                    leaf = portable_module_component(
                        prepared, materialize_module_sources(prepared, store)
                    )
                    leaf_components[role].append(leaf)
                    _package(leaf, store)
            model.save(str(directory / f"{prefix}.onnx"))
    assert all(value == components[0] for value in components)
    assert all(value.prepared == requests[0].prepared for value in requests)
    assert all(value == packages[0] for value in packages)
    for values in leaf_components.values():
        assert all(value == values[0] for value in values)
    # Three source definitions (two leaves, one Op) and their three packages.
    assert len(store.published) == 6, store.published
    assert len(set(store.published)) == 6
    source = store.lookup(module_source_derivation(requests[0].prepared))
    assert source is not None
    paths = [str(Path(source.directory) / path) for path in source.files]
    manifest = {
        "prepared": _record(requests[0].prepared),
        "component": _record(components[0]),
        "associations": associations,
        "published": store.published,
        "verified_hits": store.hits,
        "source_files": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths
        },
        "package_directory": packages[0].directory,
    }
    (directory / "build-evidence.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return components[0], paths


def observation_wrapper(component, directory, *, dual):
    top = "observe_dual" if dual else "observe_single"
    ports = ["input wire ap_clk", "input wire ap_clk2x", "input wire ap_rst_n"]
    instances, connections, observations = [], {}, {}
    for side in ("left", "right") if dual else ("single",):
        instance = f"u_{side}_matmul" if dual else "dut"
        prefix = f"{side}_" if dual else ""
        wiring = [".ap_clk(ap_clk)", ".ap_clk2x(ap_clk2x)", ".ap_rst_n(ap_rst_n)"]
        for bus, width, input_bus in (
            ("in0_V", 8, True),
            ("in1_V", 16, True),
            ("out0_V", 32, False),
        ):
            name = prefix + bus
            for suffix, bits, incoming in (
                ("tdata", width, input_bus),
                ("tvalid", 1, input_bus),
                ("tready", 1, not input_bus),
            ):
                span = f" [{bits - 1}:0]" if bits > 1 else ""
                ports.append(f"{'input' if incoming else 'output'} wire{span} {name}_{suffix}")
                wiring.append(f".{bus}_{suffix}({name}_{suffix})")
        monitor = {}
        for role, pin, width in (
            ("data", "odat", 6),
            ("valid", "ovld", 1),
            ("ready", "ordy", 1),
            ("last", "olast", 1),
        ):
            debug = f"debug_{side}_{role}"
            span = f" [{width - 1}:0]" if width > 1 else ""
            ports.append(f"output wire{span} {debug}")
            connections[debug] = f"{instance}.u_replay.{pin}"
            monitor[role] = debug
        observations[f"{side}.replay"] = monitor
        instances.append(f"{component.entry_point} {instance} (" + ", ".join(wiring) + ");")
    text = "module " + top + " (\n  " + ",\n  ".join(ports) + ");\n"
    text += "\n".join(instances) + "\n"
    text += "\n".join(f"assign {name} = {value};" for name, value in connections.items())
    text += "\nendmodule\n"
    path = directory / f"{top}.sv"
    path.write_text(text)
    return top, str(path), observations


def exercise(component, sources, directory, *, dual, rows, stalls):
    label = f"{'dual' if dual else 'single'}-r{rows}-{'stalls' if stalls else 'free'}"
    top, observer, observations = observation_wrapper(component, directory, dual=dual)
    stimulus, expected, replay_words, last_flags = {}, {}, {}, {}
    for side, position in (("left", 0), ("right", 1)) if dual else (("single", 0),):
        prefix = f"{side}_" if dual else ""
        inputs, weights_stream, output, internal = [], [], [], []
        for activations, weights in (
            (ACTIVATIONS, WEIGHTS),
            (ALTERNATE_ACTIVATIONS, ALTERNATE_WEIGHTS),
        ):
            for row in range(rows):
                activation = activations[(position + row) % 2]
                # Drive nonzero top padding; production wrapper must normalize it.
                inputs.extend(word | 0xC0 for word in packed_activation(activation))
                weights_stream.extend(word | 0xF000 for word in packed_weights(weights))
                output.extend(packed_output(activation, weights))
                internal.extend(packed_activation(activation) * 2)
        stimulus[prefix + "in0_V"] = inputs
        stimulus[prefix + "in1_V"] = weights_stream
        expected[prefix + "out0_V"] = output
        replay_words[f"{side}.replay"] = internal
        last_flags[f"{side}.replay"] = [0, 1] * (len(internal) // 2)
    result = drive_observed(
        top,
        sources + [observer],
        stimulus,
        {name: len(values) for name, values in expected.items()},
        observations,
        stalls=stalls,
        directory=directory / label,
    )
    assert result["outputs"] == expected, (label, result["outputs"], expected)
    assert result["input_counts"] == {name: len(values) for name, values in stimulus.items()}
    for name in observations:
        assert result["observations"][name]["words"] == replay_words[name]
        assert result["observations"][name]["last"] == last_flags[name]
    assert result["drain_cycles"] == 4000
    result["case"] = label
    print(
        f"PASS {label}: exact outputs, replay words/last, transfer counts, 4000-cycle drain",
        flush=True,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    directory = arguments.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    component, sources = build_artifacts(directory)
    print(
        "PASS build: two Kernel components and one Op component each materialized once; "
        "8 distinct associations",
        flush=True,
    )
    results = []
    for dual, rows in ((False, 1), (False, 2), (True, 1)):
        for stalls in (False, True):
            results.append(
                exercise(component, sources, directory, dual=dual, rows=rows, stalls=stalls)
            )
    (directory / "hardware-results.json").write_text(json.dumps(results, indent=2) + "\n")
    print("PASS all 6 production composed simulations", flush=True)


if __name__ == "__main__":
    main()
