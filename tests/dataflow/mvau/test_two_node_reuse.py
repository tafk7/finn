# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 5 gate: two MVAU nodes in one graph, one build.

Everything before this is the value and the seam in isolation.  This is the
sentence the phase exists to make true, on a real graph: two MVAUs at different
positions, configured identically, produce one artifact -- and the second node
gets it back from the store instead of building it again.

It is a *gate* rather than another unit test because every piece has to hold at
once.  A key that leaked the node id would build twice.  A build path that did
not consult the seam would build twice.  A module name carrying the placement
would produce two modules, and the second build would still be a build.  Only
the whole chain being right makes this pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np  # type: ignore[import-not-found]
from onnx import TensorProto, helper  # type: ignore[import-not-found]
import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from qonnx.core.modelwrapper import ModelWrapper  # type: ignore[import-not-found]
from qonnx.util.basic import qonnx_make_model  # type: ignore[import-not-found]

from dataflow.mvau.test_decomposed_op import (
    MATRIX_HEIGHT,
    MATRIX_WIDTH,
    _choices,
    _context,
)
from finn.dataflow.artifacts import ArtifactKey, StoredArtifact
from finn.dataflow.ops.mvau.artifacts.package import package_decomposed_artifact
from finn.dataflow.ops.mvau.artifacts.source import (
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
)
from finn.dataflow.ops.mvau.elaboration import elaborate_mvau
from finn.dataflow.ops.mvau.op import MvauDataflowOp

FINN_ROOT = Path(__file__).resolve().parents[3]

REPETITIONS = 4
NODE_IDS = ("mvau_first", "mvau_second")

WEIGHTS = np.asarray(
    [[-128, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]], dtype=np.float32
)


@dataclass
class _CountingStore:
    """A store that remembers, so "built once" is a number and not a claim."""

    entries: dict[str, StoredArtifact] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def lookup(self, identity: ArtifactKey) -> StoredArtifact | None:
        found = self.entries.get(identity.key)
        if found is None:
            self.misses += 1
        else:
            self.hits += 1
        return found

    def record(self, identity: ArtifactKey, directory: str, files: tuple[str, ...]) -> None:
        self.entries[identity.key] = StoredArtifact(identity.key, directory, files)


def _two_node_model() -> ModelWrapper:
    """Two independent MVAUs of identical shape and datatype, in one graph.

    Independent rather than chained on purpose: chaining would make the second
    node's activation the first's output, and the test would then be partly
    about the datatypes agreeing across the join.  The claim is about reuse.
    """

    nodes = [
        helper.make_node(
            "MvauDataflowOp",
            [f"activation_{index}", f"weights_{index}"],
            [f"output_{index}"],
            name=node_id,
            domain="finn.custom_op.dataflow",
            dataflow_scope_id=f"{node_id}_scope",
            accDataType="INT16",
            ActVal=0,
            noActivation=1,
            binaryXnorMode=0,
        )
        for index, node_id in enumerate(NODE_IDS)
    ]
    model = ModelWrapper(
        qonnx_make_model(
            helper.make_graph(
                nodes,
                "two-mvau",
                [
                    helper.make_tensor_value_info(
                        f"activation_{index}", TensorProto.FLOAT, [REPETITIONS, MATRIX_WIDTH]
                    )
                    for index in range(len(NODE_IDS))
                ]
                + [
                    helper.make_tensor_value_info(
                        f"weights_{index}", TensorProto.FLOAT, [MATRIX_WIDTH, MATRIX_HEIGHT]
                    )
                    for index in range(len(NODE_IDS))
                ],
                [
                    helper.make_tensor_value_info(
                        f"output_{index}", TensorProto.FLOAT, [REPETITIONS, MATRIX_HEIGHT]
                    )
                    for index in range(len(NODE_IDS))
                ],
            ),
            producer_name="two-mvau-test",
            opset_imports=[
                helper.make_opsetid("", 21),
                helper.make_opsetid("finn.custom_op.dataflow", 1),
            ],
        )
    )
    for index in range(len(NODE_IDS)):
        model.set_tensor_datatype(f"activation_{index}", DataType["INT8"])
        model.set_tensor_datatype(f"weights_{index}", DataType["INT8"])
        model.set_tensor_datatype(f"output_{index}", DataType["INT16"])
        model.set_initializer(f"weights_{index}", WEIGHTS)
    return model


def _operations(model: ModelWrapper) -> tuple[MvauDataflowOp, ...]:
    operations = []
    for node in model.graph.node:
        operation = model.get_customop_wrapper(node)
        assert isinstance(operation, MvauDataflowOp)
        operation.initialize_dataflow_scope_id()
        operation.commit_dataflow_assignments(_context(), _choices())
        operations.append(operation)
    return tuple(operations)


def _requirements(operation: MvauDataflowOp) -> MVAUDecomposedArtifactRequirements:
    resolved = operation.resolve_dataflow(_context())
    return build_decomposed_artifact_requirements(resolved, elaborate_mvau(resolved), FINN_ROOT)


@pytest.fixture(name="built")
def _built_fixture() -> tuple[MVAUDecomposedArtifactRequirements, ...]:
    requirements = tuple(_requirements(item) for item in _operations(_two_node_model()))
    if any(not Path(path).is_file() for path in requirements[0].finnlib_sources):
        pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
    return requirements


# -- the gate ----------------------------------------------------------------


def test_two_equal_nodes_build_one_artifact_and_reuse_it(
    built: tuple[MVAUDecomposedArtifactRequirements, ...], tmp_path: Path
) -> None:
    """The Phase 5 gate, in one assertion sequence.

    The second node must be a *hit*, and "hit" is not "produced the same
    files": it is that nothing was built the second time.  Counting the misses
    is what makes that checkable, which is why the double counts rather than
    just answering.
    """

    store = _CountingStore()
    packaged = []
    for requirements in built:
        unit = package_decomposed_artifact(requirements, tmp_path, store=store)
        store.record(unit.identity, unit.directory, unit.files)
        packaged.append(unit)

    assert (store.misses, store.hits) == (1, 1)
    assert packaged[0].directory == packaged[1].directory
    assert packaged[0].files == packaged[1].files
    assert packaged[0].key == packaged[1].key
    # One build on disk, not two directories that happen to agree.
    assert [item.name for item in tmp_path.iterdir()] == [Path(packaged[0].directory).name]


def test_the_two_nodes_really_are_two_distinct_placements(
    built: tuple[MVAUDecomposedArtifactRequirements, ...],
) -> None:
    """Otherwise the gate above would pass by testing one node twice.

    This is the assertion that keeps the fixture honest: the graph positions,
    the scopes, the source tensors, and the elaborated component ids all
    differ, and only the artifact is shared.
    """

    scopes = tuple(item.elaboration.source_scope_id for item in built)
    assert len(set(scopes)) == 2
    assert set(scopes) == {f"{node_id}_scope" for node_id in NODE_IDS}

    components = tuple(
        frozenset(component.id for component in item.elaboration.components) for item in built
    )
    assert components[0] != components[1]
    assert not components[0] & components[1]

    associations = tuple(
        frozenset(
            owner
            for item in requirements.elaboration.associations
            for owner in item.source_owner_ids
        )
        for requirements in built
    )
    assert associations[0] != associations[1]


def test_the_shared_artifact_names_neither_node(
    built: tuple[MVAUDecomposedArtifactRequirements, ...], tmp_path: Path
) -> None:
    """The artifact belongs to both, so it may name neither."""

    packaged = package_decomposed_artifact(built[0], tmp_path)
    text = Path(packaged.files[-1]).read_text()

    for node_id in NODE_IDS:
        assert node_id not in packaged.top_module_name
        assert node_id not in Path(packaged.directory).name
        assert node_id not in text
        assert node_id not in packaged.identity.serialization


def test_a_differently_configured_node_is_a_second_build(tmp_path: Path) -> None:
    """The other side of the gate, so reuse is not just "everything hits".

    Change one node's folding and the store misses twice.  Without this, a
    store that returned the first entry for every key would pass the gate.
    """

    model = _two_node_model()
    operations = []
    for index, node in enumerate(model.graph.node):
        operation = model.get_customop_wrapper(node)
        assert isinstance(operation, MvauDataflowOp)
        operation.initialize_dataflow_scope_id()
        operation.commit_dataflow_assignments(_context(), _choices(pe=2 if index == 0 else 4))
        operations.append(operation)

    store = _CountingStore()
    directories = set()
    for operation in operations:
        requirements = _requirements(operation)
        if any(not Path(path).is_file() for path in requirements.finnlib_sources):
            pytest.skip("FinnLib is not fetched; set FINNLIB_ROOT or run fetch-repos.sh")
        unit = package_decomposed_artifact(requirements, tmp_path, store=store)
        store.record(unit.identity, unit.directory, unit.files)
        directories.add(unit.directory)

    assert (store.misses, store.hits) == (2, 0)
    assert len(directories) == 2
