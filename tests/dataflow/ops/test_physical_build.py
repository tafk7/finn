# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Current source-to-component use, serial reuse, and association refusals."""

from dataclasses import replace
import pickle
import subprocess
import sys

import pytest
from qonnx.core.modelwrapper import ModelWrapper

import finn.dataflow.artifacts.build as build_module
from finn.dataflow._engine import Decided, Unresolved
from finn.dataflow.artifacts.build import module_source_derivation
from finn.dataflow.artifacts.store import ArtifactStore
from finn.dataflow.model.region import BeatSequence
from finn.dataflow.ops.base import DataflowOpError
from finn.dataflow.ops.mvau.op import MvauDataflowOp
from finn.dataflow.ops.persistence import CommitmentStage
from finn.dataflow.ops.physical import (
    capture_op_physical,
    install_physical_component,
    materialize_build_request,
    prepare_build_request,
    validate_physical_build_association,
)

from dataflow.physical_fixture import (
    ALTERNATE_WEIGHTS,
    configure,
    roots,
    source_model,
    template_roots,
)


def _prepare(op, model, build, context, store):
    return prepare_build_request(
        op,
        capture_op_physical(op),
        model=model,
        build=build,
        graph_context=context,
        roots=roots(),
        template_roots=template_roots(),
        blobs=store,
    )


def test_two_committed_occurrences_reuse_one_component_and_keep_distinct_associations(
    tmp_path, monkeypatch
):
    model, build, context = source_model()
    store = ArtifactStore(tmp_path / "store")
    original = build_module.render_module_sources
    renders = []

    def counted(prepared, contents):
        renders.append(prepared)
        return original(prepared, contents)

    monkeypatch.setattr(build_module, "render_module_sources", counted)
    requests, components, instances = [], [], []
    for index, node in enumerate(list(model.graph.node)):
        op = configure(MvauDataflowOp(node).bind(model, build, graph_context=context))
        op = op.commit(model, build, require=CommitmentStage.PHYSICAL, graph_context=context)
        request = _prepare(op, model, build, context, store)
        component = materialize_build_request(
            op, request, model=model, build=build, graph_context=context, store=store
        )
        instance = install_physical_component(
            op,
            request,
            component,
            outer_instance_id=f"u_{index}",
            model=model,
            build=build,
            graph_context=context,
            store=store,
        )
        requests.append(request)
        components.append(component)
        instances.append(instance)
    assert len(renders) == 1
    assert requests[0].prepared == requests[1].prepared
    assert requests[0].capture.requirements == requests[1].capture.requirements
    assert components[0] == components[1]
    assert instances[0].outer_instance_id != instances[1].outer_instance_id
    assert (
        instances[0].association.logical.source_origin
        != instances[1].association.logical.source_origin
    )
    path = tmp_path / "source.onnx"
    model.save(str(path))
    restored = ModelWrapper(str(path))
    for index, node in enumerate(restored.graph.node):
        op = MvauDataflowOp(node).bind(restored, build, graph_context=context)
        assert capture_op_physical(op).requirements == requests[index].capture.requirements
        rebound_request = _prepare(op, restored, build, context, store)
        assert (
            materialize_build_request(
                op, rebound_request, model=restored, build=build, graph_context=context, store=store
            )
            == components[index]
        )
    assert len(renders) == 1


def test_precommit_capture_requires_recapture_after_strong_commit_and_can_reuse(tmp_path):
    model, build, context = source_model()
    store = ArtifactStore(tmp_path / "store")
    op = configure(MvauDataflowOp(model.graph.node[0]).bind(model, build, graph_context=context))
    old = _prepare(op, model, build, context, store)
    component = materialize_build_request(
        op, old, model=model, build=build, graph_context=context, store=store
    )
    op = op.commit(model, build, require_graph=True, graph_context=context)
    assert validate_physical_build_association(
        op, old.capture, model=model, build=build, graph_context=context
    )
    with pytest.raises(DataflowOpError):
        materialize_build_request(
            op, old, model=model, build=build, graph_context=context, store=store
        )
    new = _prepare(op, model, build, context, store)
    assert new.prepared == old.prepared
    assert (
        materialize_build_request(
            op, new, model=model, build=build, graph_context=context, store=store
        )
        == component
    )


def test_changed_context_refuses_before_a_populated_cache_lookup(tmp_path, monkeypatch):
    model, build, context = source_model()
    store = ArtifactStore(tmp_path / "store")
    op = configure(MvauDataflowOp(model.graph.node[0]).bind(model, build, graph_context=context))
    request = _prepare(op, model, build, context, store)
    materialize_build_request(
        op, request, model=model, build=build, graph_context=context, store=store
    )
    first = context.graph_inputs[0]
    changed = replace(
        context,
        graph_inputs=(
            replace(
                first,
                contract=replace(
                    first.contract,
                    beat_sequence=BeatSequence(4, (((0, 0), (0, 1), (0, 2), (0, 3)),)),
                ),
            ),
            *context.graph_inputs[1:],
        ),
    )

    def forbidden(*args):
        raise AssertionError("stale context reached the store")

    monkeypatch.setattr(store, "lookup", forbidden)
    with pytest.raises(DataflowOpError):
        materialize_build_request(
            op, request, model=model, build=build, graph_context=changed, store=store
        )


def test_mixed_prepared_request_and_component_refuse(tmp_path):
    model, build, context = source_model()
    store = ArtifactStore(tmp_path / "store")
    left = configure(MvauDataflowOp(model.graph.node[0]).bind(model, build, graph_context=context))
    right = configure(
        MvauDataflowOp(model.graph.node[1]).bind(model, build, graph_context=context), pumping=False
    )
    first, second = (_prepare(op, model, build, context, store) for op in (left, right))
    a = materialize_build_request(
        left, first, model=model, build=build, graph_context=context, store=store
    )
    b = materialize_build_request(
        right, second, model=model, build=build, graph_context=context, store=store
    )
    assert a != b
    with pytest.raises(DataflowOpError, match="pair"):
        materialize_build_request(
            left,
            replace(first, prepared=second.prepared),
            model=model,
            build=build,
            graph_context=context,
            store=store,
        )
    with pytest.raises(DataflowOpError, match="component differs"):
        install_physical_component(
            left,
            first,
            b,
            outer_instance_id="mixed",
            model=model,
            build=build,
            graph_context=context,
            store=store,
        )
    assert store.lookup(module_source_derivation(first.prepared)) is not None


def test_external_weights_are_invariant_only_with_equal_consumed_physical_facts(tmp_path):
    captures, prepared = [], []
    store = ArtifactStore(tmp_path / "store")
    narrow_changed = ((-4, *ALTERNATE_WEIGHTS[0][1:]), *ALTERNATE_WEIGHTS[1:])
    for kwargs in ({}, {"weights": ALTERNATE_WEIGHTS}, {"weights": narrow_changed}):
        model, build, context = source_model(**kwargs)
        op = configure(
            MvauDataflowOp(model.graph.node[0]).bind(model, build, graph_context=context)
        )
        request = _prepare(op, model, build, context, store)
        captures.append(request.capture)
        prepared.append(request.prepared)
    assert captures[0].association.logical.incoming != captures[1].association.logical.incoming
    assert captures[0].requirements == captures[1].requirements
    assert prepared[0] == prepared[1]
    assert captures[1].requirements != captures[2].requirements
    assert prepared[1] != prepared[2]


def test_physical_choice_unset_keeps_graph_logical_acceptance():
    model, build, context = source_model()
    op = configure(
        MvauDataflowOp(model.graph.node[0]).bind(model, build, graph_context=context), pumping=None
    )
    assert isinstance(op.graph_dataflow.accepted_answer, Decided)
    assert isinstance(op.physical.accepted_answer, Unresolved)


def test_prepared_production_component_renders_in_fresh_process_without_compiler_reads(tmp_path):
    model, build, context = source_model()
    store = ArtifactStore(tmp_path / "store")
    op = configure(MvauDataflowOp(model.graph.node[0]).bind(model, build, graph_context=context))
    request = _prepare(op, model, build, context, store)
    payload = tmp_path / "prepared.pkl"
    payload.write_bytes(pickle.dumps(request.prepared))
    child = r"""
import builtins, pathlib, pickle, sys
original_import = builtins.__import__
forbidden = ('finn.dataflow.ops', 'finn.dataflow.kernels', 'finn.dataflow.designs',
             'finn.dataflow.model', 'finn.dataflow.space', 'finn.dataflow._engine',
             'onnx', 'qonnx')
def guarded_import(name, *args, **kwargs):
    if any(name == prefix or name.startswith(prefix + '.') for prefix in forbidden):
        raise AssertionError('compiler import during detached rendering: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from finn.dataflow.artifacts.build import render_module_sources
from finn.dataflow.artifacts.store import ArtifactStore
prepared = pickle.loads(pathlib.Path(sys.argv[1]).read_bytes())
store_root = pathlib.Path(sys.argv[2]).resolve()
original_read = pathlib.Path.read_bytes
original_open = pathlib.Path.open
def check_path(path):
    if not pathlib.Path(path).resolve().is_relative_to(store_root):
        raise AssertionError('checkout read during detached rendering: ' + str(path))
def guarded_read(path):
    check_path(path)
    return original_read(path)
def guarded_open(path, *args, **kwargs):
    check_path(path)
    return original_open(path, *args, **kwargs)
pathlib.Path.read_bytes = guarded_read
pathlib.Path.open = guarded_open
rendered = render_module_sources(prepared, ArtifactStore(store_root))
assert len(rendered.contents) == 7
assert rendered.definition.origin == ''
print('PASS fresh-process production rendering from prepared values/store only')
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(payload), str(store.root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS fresh-process" in result.stdout
