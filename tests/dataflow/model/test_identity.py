# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import dataclass
import pickle

import pytest

import finn.dataflow.model as model
from finn.dataflow.kernels.matmul.base import DspBlock, MvauComputationProfile
from finn.dataflow.kernels.matmul.supply import WeightSupply
from finn.dataflow.model.kernel import Kernel
from finn.dataflow.model.identity import (
    TYPE_IDENTITY_RELOCATIONS,
    comparison_type_identity,
    implementation_identity,
)
from finn.dataflow.model.logical.region import DataflowRegion
from finn.dataflow.model.physical.layout import PackedBeatLayout
from finn.dataflow.model.physical.interface import KernelStreamBinding
from finn.dataflow.artifacts import build, contribution_types, contributions, requirements
from finn.dataflow.kernels import target
from finn.dataflow.model.logical import composition, results, semantics
from finn.dataflow.model.logical import datatype_semantics, result_semantics


class ExampleKernel(Kernel):
    id = "example"
    version = "7"


def test_model_facade_exposes_only_domain_framework_vocabulary() -> None:
    assert set(model.__all__) == {
        "EdgeSink",
        "ImplementationIdentity",
        "Kernel",
        "KernelChoice",
        "KernelEndpoint",
        "LogicalView",
        "ModuleParameter",
        "NetworkBoundary",
        "NetworkEdge",
        "PhysicalView",
        "PhysicallyUnsupported",
        "RegionDeclaration",
        "PublicOperand",
        "OperandExport",
        "OperandTarget",
        "PublicOperandDeclaration",
        "implementation_identity",
        "kernel_dataflow",
        "kernel_physical",
    }


def test_implementation_identity_is_owned_by_the_domain() -> None:
    assert implementation_identity(ExampleKernel).family == "example"
    assert implementation_identity(ExampleKernel).version == "7"


@pytest.mark.parametrize(
    "value,old_identity",
    (
        (DataflowRegion, "finn.dataflow.model.region.DataflowRegion"),
        (Kernel, "finn.dataflow.kernels.kernel.Kernel"),
        (PackedBeatLayout, "finn.dataflow.kernels.physical.PackedBeatLayout"),
        (KernelStreamBinding, "finn.dataflow.kernels.physical.KernelStreamBinding"),
        (
            MvauComputationProfile,
            "finn.dataflow.ops.mvau.computation.MvauComputationProfile",
        ),
        (DspBlock, "finn.dataflow.kernels.dotp_axi.DspBlock"),
        (WeightSupply, "finn.dataflow.ops.mvau.kernels.supply.WeightSupply"),
    ),
)
def test_comparison_type_identity_preserves_pre_kp_names(
    value: type[object], old_identity: str
) -> None:
    assert comparison_type_identity(value) == old_identity


def test_relocation_is_symbol_specific_for_mixed_origin_modules() -> None:
    assert (
        TYPE_IDENTITY_RELOCATIONS["finn.dataflow.kernels.matmul.base.DspBlock"]
        == "finn.dataflow.kernels.dotp_axi.DspBlock"
    )
    assert (
        TYPE_IDENTITY_RELOCATIONS["finn.dataflow.kernels.matmul.base.MvauComputationProfile"]
        == "finn.dataflow.ops.mvau.computation.MvauComputationProfile"
    )
    assert (
        TYPE_IDENTITY_RELOCATIONS["finn.dataflow.kernels.matmul.supply.WeightSupply"]
        == "finn.dataflow.ops.mvau.kernels.supply.WeightSupply"
    )


def test_unmoved_type_keeps_its_real_identity() -> None:
    @dataclass(frozen=True)
    class Local:
        value: int

    assert comparison_type_identity(Local) == f"{Local.__module__}.{Local.__qualname__}"


@pytest.mark.parametrize(
    "current,compatibility,module",
    [
        (target.DspBlock, DspBlock, "finn.dataflow.kernels.matmul.base"),
        (results.RegionResult, composition.RegionResult, "finn.dataflow.model.logical.composition"),
        (
            results.NetworkResult,
            composition.NetworkResult,
            "finn.dataflow.model.logical.composition",
        ),
        (
            requirements.ModuleBuildRequirements,
            build.ModuleBuildRequirements,
            "finn.dataflow.artifacts.build",
        ),
        (
            requirements.ModuleABIRequirements,
            build.ModuleABIRequirements,
            "finn.dataflow.artifacts.build",
        ),
        (
            contribution_types.CopiedSource,
            contributions.CopiedSource,
            "finn.dataflow.artifacts.contributions",
        ),
    ],
)
def test_split_contracts_preserve_nominal_and_serialized_identity(current, compatibility, module):
    assert current is compatibility
    assert current.__module__ == module
    assert pickle.loads(pickle.dumps(current)) is current


def test_split_semantics_reexport_the_same_tokens_and_codecs() -> None:
    assert (
        semantics.QONNX_DATATYPE_VALUE_SEMANTICS
        is datatype_semantics.QONNX_DATATYPE_VALUE_SEMANTICS
    )
    assert semantics.QONNX_DATATYPE_CODEC is datatype_semantics.QONNX_DATATYPE_CODEC
    assert (
        semantics.DATAFLOW_LOGICAL_RESULT_SEMANTICS
        is result_semantics.DATAFLOW_LOGICAL_RESULT_SEMANTICS
    )
