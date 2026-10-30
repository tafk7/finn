# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Distribution metadata and a bounded installed logical API smoke.

Set FINN_DISTRIBUTION_ARTIFACTS to a wheel/sdist directory to inspect real
artifacts. Logical tests also run in an isolated interpreter against an installed
wheel; they need neither an editable checkout nor hardware/timing profiles.
"""

from __future__ import annotations

import os
import tarfile
import tomllib
import zipfile
from dataclasses import replace
from email.parser import BytesParser
from importlib.metadata import distribution, version
from pathlib import Path

import pytest

from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.descriptors import dump, load
from finn.dataflow.logical import (
    Available,
    Edge,
    InvalidPlan,
    LogicalPlan,
    Node,
    Port,
    Source,
    adapters,
    packing,
)
from finn.dataflow.profiles import Attachment, TimedPlan, Unavailable, readiness
from finn.dataflow.steps import Step, Steps
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.dataflow.traversal import BeatSequence, vector_major


def tiny() -> LogicalPlan:
    tensor = Tensor((2, 6), ScalarEncoding(resolve_qonnx_datatype_name("INT8")))
    producer = BeatSequence(vector_major(tensor.shape, 2))
    consumer = BeatSequence(vector_major(tensor.shape, 1))
    return LogicalPlan(
        "distribution",
        (Node("producer"), Node("consumer")),
        (
            Port("out", "producer", "write", "data", tensor, producer),
            Port("in", "consumer", "read", "data", tensor, consumer),
        ),
        (Edge("edge", "out", "in"),),
        (Source("data", tensor),),
    )


def test_logical_route_and_descriptor_roundtrip() -> None:
    plan = tiny()
    restored = load(dump(plan))
    assert isinstance(restored, LogicalPlan) and restored == plan
    assert restored.digest == plan.digest
    packed = packing(restored, "out")
    assert isinstance(packed, Available)
    assert (packed.value.lanes, packed.value.payload_bits, packed.value.beats_per_pass) == (
        2,
        16,
        6,
    )
    route = adapters(restored, "edge")
    assert isinstance(route, Available) and isinstance(route.value, Steps)
    assert route.value.kinds == (Step.WIDTH,)


def test_malformed_shape_and_direction_are_refused() -> None:
    plan = tiny()
    with pytest.raises(InvalidPlan) as shape:
        replace(plan.ports[0], tensor=Tensor((3, 6), plan.ports[0].tensor.element))
    assert shape.value.findings[0].code == "shape-mismatch"
    with pytest.raises(InvalidPlan) as direction:
        # Mutate an untyped durable input; do not weaken the typed constructor.
        raw = dump(plan).replace(b'"write"', b'"invalid"')
        load(raw)
    assert direction.value.findings[0].code == "direction-invalid"


def test_missing_profiles_leave_logical_queries_available() -> None:
    plan = tiny()
    timed = TimedPlan(plan, Attachment(plan.digest))
    missing = readiness(timed)
    assert isinstance(missing, Unavailable) and missing.missing
    assert isinstance(packing(plan, "out"), Available)
    assert isinstance(adapters(plan, "edge"), Available)
    assert load(dump(plan)) == plan


def test_installed_dependency_and_typing_marker() -> None:
    assert version("islpy") == "2026.2.2"
    info = distribution("finn")
    assert "islpy==2026.2.2" in (info.requires or [])
    assert Path(str(info.locate_file("finn/dataflow/py.typed"))).is_file()


def test_wheel_and_sdist_metadata_and_markers() -> None:
    location = os.environ.get("FINN_DISTRIBUTION_ARTIFACTS")
    if location is None:
        pytest.skip("set FINN_DISTRIBUTION_ARTIFACTS to inspect built distributions")
    root = Path(location)
    wheels, sdists = list(root.glob("*.whl")), list(root.glob("*.tar.gz"))
    assert len(wheels) == len(sdists) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert "finn/dataflow/py.typed" in wheel.namelist()
        assert not any(name.startswith("finn/dataflow_next/") for name in wheel.namelist())
        metadata_files = [n for n in wheel.namelist() if n.endswith(".dist-info/METADATA")]
        assert len(metadata_files) == 1
        wheel_meta = BytesParser().parsebytes(wheel.read(metadata_files[0]))
    with tarfile.open(sdists[0]) as sdist:
        names = sdist.getnames()
        assert any(n.endswith("/src/finn/dataflow/py.typed") for n in names)
        assert not any("/src/finn/dataflow_next/" in name for name in names)
        package_info = [n for n in names if n.count("/") == 1 and n.endswith("/PKG-INFO")]
        assert len(package_info) == 1
        member = sdist.extractfile(package_info[0])
        assert member is not None
        sdist_meta = BytesParser().parsebytes(member.read())
        project_file = sdist.extractfile(next(n for n in names if n.endswith("/pyproject.toml")))
        assert project_file is not None
        project = tomllib.loads(project_file.read().decode())
    source = Path(os.environ["FINN_DISTRIBUTION_SOURCE"])
    lock = tomllib.loads((source / "uv.lock").read_text())
    for metadata in (wheel_meta, sdist_meta):
        assert "islpy==2026.2.2" in metadata.get_all("Requires-Dist", [])
    assert wheel_meta["Version"] == sdist_meta["Version"] == version("finn")
    assert "islpy==2026.2.2" in project["project"]["dependencies"]
    locked = [p for p in lock["package"] if p["name"] == "islpy"]
    assert len(locked) == 1 and locked[0]["version"] == "2026.2.2"
    supported = [w for w in locked[0]["wheels"] if "cp312-cp312-manylinux" in w["url"]]
    assert len(supported) == 1
    assert supported[0]["hash"] == (
        "sha256:9c9ba278c1ae2dc4976195630a51303b434f6eccfe86bf94d49a42a43d678751"
    )
