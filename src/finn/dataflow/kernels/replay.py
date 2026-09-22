# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Standalone activation replay with one meaningful SIMD choice.

The logical wrapper exposes one ReplayBuffer child without an implementation
selector. Its matrix height denotes the requested copies directly, so the child
receives one processing element as a semantic constant.
"""

from __future__ import annotations

from finn.dataflow.model.children import KernelChoice
from finn.dataflow.model.kernel import Kernel
from finn.dataflow.model.logical.authoring import NetworkBoundary
from finn.dataflow.model.logical.composition import LogicalResult, logical_network
from finn.dataflow.model.logical.interface import OperandExport, OperandTarget, PublicOperand
from finn.dataflow.model.logical.interface_authoring import PublicOperandDeclaration
from finn.dataflow.model.logical.maps import RectangularDomain
from finn.dataflow.model.logical.network import PositionMap
from finn.dataflow.model.logical.refs import DataflowOperandRef, RegionInputRef, RegionOutputRef
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.space.declarations import (
    Decision,
    Input,
    Subspace,
    Projection,
    Readiness,
    derived,
    divisors_of,
)
from finn.dataflow.model.logical.semantics import QONNX_DATATYPE_VALUE_SEMANTICS


def _replay_export(public: PublicOperand, logical: LogicalResult) -> OperandExport:
    network = logical_network(logical)
    boundary_id = "activation" if public.direction == "input" else "expanded"
    endpoint = next(item.endpoint for item in network.boundaries if item.id == boundary_id)
    region = network.node(endpoint.node_id).region
    ref: DataflowOperandRef
    if public.direction == "input":
        operand = region.input_interface(endpoint.port_id).operand
        ref = RegionInputRef(endpoint.node_id, operand.id)
    else:
        operand = region.output_interface(endpoint.port_id).port.operand
        ref = RegionOutputRef(endpoint.node_id, operand.id)
    return OperandExport(
        public,
        (
            OperandTarget(
                ref,
                PositionMap.row_major_reshape(public.domain, operand.position_domain),
                (endpoint,),
            ),
        ),
    )


class ActivationReplayKernel(Kernel):
    """Present each activation row once per neuron fold, and nothing else."""

    id = "activation_replay"
    version = "2"

    repetitions = Input(int)
    matrix_width = Input(int)
    matrix_height = Input(int)
    activation_type = Input(QONNX_DATATYPE_VALUE_SEMANTICS)

    public_type_ready = Readiness()
    public_type = Projection(activation_type, readiness=public_type_ready)

    @derived(RectangularDomain, rows=repetitions, width=matrix_width)
    def activation_domain(*, rows: int, width: int) -> RectangularDomain:
        return RectangularDomain((rows, width))

    @derived(RectangularDomain, rows=repetitions, width=matrix_width, replays=matrix_height)
    def result_domain(*, rows: int, width: int, replays: int) -> RectangularDomain:
        return RectangularDomain((rows * replays, width))

    activation_domain_ready = Readiness(properties=(activation_domain,))
    result_domain_ready = Readiness(properties=(result_domain,))
    public_activation_domain = Projection(activation_domain, readiness=activation_domain_ready)
    public_result_domain = Projection(result_domain, readiness=result_domain_ready)
    public_operands = (
        PublicOperandDeclaration(
            "activation", "input", public_type, public_activation_domain, _replay_export
        ),
        PublicOperandDeclaration(
            "result", "output", public_type, public_result_domain, _replay_export
        ),
    )

    @derived(int)
    def processing_elements() -> int:
        """Standalone matrix height already denotes the number of requested copies."""
        return 1

    simd = Decision(int, domain=divisors_of(matrix_width))

    replay = KernelChoice(
        Subspace(
            ReplayBufferKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            pe=processing_elements,
            simd=simd,
        ),
    )

    activation = NetworkBoundary(replay.input("activation_in"))
    expanded = NetworkBoundary(replay.output("activation_out"))


KERNEL_INPUTS = ("repetitions", "matrix_width", "matrix_height", "activation_type")

__all__ = ["KERNEL_INPUTS", "ActivationReplayKernel"]
