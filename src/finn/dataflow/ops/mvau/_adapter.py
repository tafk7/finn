# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Temporary AC3 bridge from class-authored MVAU facts to current Designs.

This module is intentionally private and disappears when the two MVAU Design
classes own their declarations directly.  It lets the new Op projection and
persistence compiler be exercised against the exact current semantic and
physical inventory without making either path authoritative prematurely.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from finn.dataflow.authoring.compiler import AdaptedDesignCompilation
from finn.dataflow.authoring.declarations import CompiledClassDeclarations
from finn.dataflow.authoring.scope import Ref
from finn.dataflow.ops.mvau.inventory import declare_mvau_design_inventory
from finn.dataflow.ops.mvau.problem import (
    MVAUComputationProfile,
    MVAUProblem,
    MVAUSourceDescription,
)
from finn.dataflow.parameters.cyclic.definition import CyclicTargetMemoryCapabilities
from finn.dataflow.kernels.dsp import DspBlock
from finn.dataflow.region import BeatSequence, NumericElementType


@dataclass(frozen=True, slots=True)
class MVAUPersistenceReferences:
    design: object
    dot_product_pe: object
    dot_product_simd: object
    batch_interleaved_pe: object
    batch_interleaved_simd: object
    batch_interleaved_interleave: object
    weight_supply: object
    compute_pumping: object
    ram_style: object
    pumped_memory: object


class MVAUDesignAdapter:
    """Private compatibility lowerer used until AC5 removes scope callbacks."""

    def __init__(self) -> None:
        self.refs = MVAUPersistenceReferences(*(object() for _index in range(10)))

    def compile(self, declarations: CompiledClassDeclarations) -> AdaptedDesignCompilation:
        ref = declarations.ref
        problem = MVAUProblem(
            repetitions=cast("Ref[int]", ref("repetitions")),
            matrix_width=cast("Ref[int]", ref("matrix_width")),
            matrix_height=cast("Ref[int]", ref("matrix_height")),
            activation_element_type=cast("Ref[NumericElementType]", ref("activation.datatype")),
            weight_element_type=cast("Ref[NumericElementType]", ref("weight.datatype")),
            accumulator_element_type=cast(
                "Ref[NumericElementType]", ref("accumulator_element_type")
            ),
            output_element_type=cast("Ref[NumericElementType]", ref("output.datatype")),
            threshold_element_type=cast("Ref[NumericElementType]", ref("threshold.datatype")),
            threshold_initializer_available=cast("Ref[bool]", ref("threshold.initializer_present")),
            computation_profile=cast("Ref[MVAUComputationProfile]", ref("computation_profile")),
            weight_initializer_available=cast("Ref[bool]", ref("weight.initializer_present")),
            weight_initializer_fingerprint=cast("Ref[str]", ref("weight.initializer_fingerprint")),
            threshold_initializer_fingerprint=cast(
                "Ref[str]", ref("threshold.initializer_fingerprint")
            ),
            source_description=cast("Ref[MVAUSourceDescription]", ref("source_description")),
            initializer_excludes_minimum=cast("Ref[bool]", ref("initializer_excludes_minimum")),
            runtime_weight_range_contract=cast("Ref[bool]", ref("runtime_weight_range_contract")),
            runtime_writable=cast("Ref[bool]", ref("runtime_writable")),
            external_weight_sequence=cast("Ref[BeatSequence]", ref("external_weight_sequence")),
            accumulator_type_analysis_owner=cast(
                "Ref[str]", ref("accumulator_type_analysis_owner")
            ),
            target_dsp_block=cast("Ref[DspBlock]", ref("target_dsp_block")),
            target_fpga_part=cast("Ref[str]", ref("target_fpga_part")),
            target_clock_period_ns=cast("Ref[float]", ref("target_clock_period_ns")),
            target_memory_capabilities=cast(
                "Ref[CyclicTargetMemoryCapabilities]", ref("target_memory_capabilities")
            ),
        )
        assembly = declare_mvau_design_inventory(
            problem,
            narrow_weights=cast("Ref[bool]", ref("effective_narrow_weights")),
            problem_spec=declarations.scope.spec(),
        )
        design = assembly.inventory.design_selection
        if design is None:
            raise AssertionError("MVAU requires a two-Design selection")
        mapping = {
            id(self.refs.design): cast("Ref[object]", design),
            id(self.refs.dot_product_pe): cast("Ref[object]", assembly.dot_product.pe),
            id(self.refs.dot_product_simd): cast("Ref[object]", assembly.dot_product.simd),
            id(self.refs.batch_interleaved_pe): cast("Ref[object]", assembly.batch_interleaved.pe),
            id(self.refs.batch_interleaved_simd): cast(
                "Ref[object]", assembly.batch_interleaved.simd
            ),
            id(self.refs.batch_interleaved_interleave): cast(
                "Ref[object]", assembly.batch_interleaved.interleave
            ),
            id(self.refs.weight_supply): cast(
                "Ref[object]", assembly.input_supply.declaration.choice
            ),
            id(self.refs.compute_pumping): cast("Ref[object]", assembly.compute_pumping),
            id(self.refs.ram_style): cast("Ref[object]", assembly.input_supply.settings.ram_style),
            id(self.refs.pumped_memory): cast(
                "Ref[object]", assembly.input_supply.settings.pumped_memory
            ),
        }
        return AdaptedDesignCompilation(
            assembly.authoring,
            MappingProxyType(mapping),
        )


MVAU_DESIGN_ADAPTER = MVAUDesignAdapter()


__all__: list[str] = []
