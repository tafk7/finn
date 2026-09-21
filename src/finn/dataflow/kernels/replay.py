# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One segment, one Kernel, two boundaries.

The smallest Kernel that is still a Kernel.  Its value here is negative: it has
no SubspaceChoice, so nothing in the operation layer may assume a selector
exists; it has one node, so nothing may assume an edge; and it has no weight
path, so nothing may assume a matrix.
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
    domain,
    reject,
)
from finn.dataflow.model.logical.semantics import QONNX_DATATYPE_VALUE_SEMANTICS


def _standalone_pe(*, candidate: object) -> object:
    if type(candidate) is int and candidate == 1:
        return True
    return reject(
        "activation-replay-pe-not-one",
        "standalone activation replay requires PE=1",
    )


def _standalone_pe_candidates() -> tuple[object, ...]:
    return (1,)


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

    # The standalone operation's matrix height is the requested replay count,
    # not an MVAU output height to fold across processing elements.  Keeping
    # this Decision in the persisted shape preserves the existing path while
    # making every accepted point spell that contract exactly.
    pe = Decision(
        int,
        domain=domain(accepts=_standalone_pe, candidates=_standalone_pe_candidates),
    )
    simd = Decision(int, domain=divisors_of(matrix_width))

    replay = KernelChoice(
        Subspace(
            ReplayBufferKernel,
            repetitions=repetitions,
            matrix_width=matrix_width,
            matrix_height=matrix_height,
            activation_type=activation_type,
            pe=pe,
            simd=simd,
        ),
    )

    activation = NetworkBoundary(replay.input("activation_in"))
    expanded = NetworkBoundary(replay.output("activation_out"))


KERNEL_INPUTS = ("repetitions", "matrix_width", "matrix_height", "activation_type")

__all__ = ["KERNEL_INPUTS", "ActivationReplayKernel"]
