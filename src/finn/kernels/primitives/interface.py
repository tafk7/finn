############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
#
# Distilled variant of the interface value-objects in microsoft/brainsmith
# @ 38faaf9 (MIT) — not a pristine mirror. Temporary vendored copy — see Stage 3
# (source model) in kernel-design/kernel-final-design/KERNEL_REFACTOR_PLAN.md.
############################################################################
"""Interface design-space + design-point value objects (shape/width/folding).

An ``InterfaceDesignSpace`` holds the structure constant across a DSE sweep
(tensor/block shapes, datatype, the stream-tiling template). An
``InterfaceDesignPoint`` is a flyweight over it carrying one resolved
``stream_shape`` and deriving all folding/width quantities from it.

Provides the shape/width primitives the resolve engine's Derived layer reads:
``tensor_folding_factor`` (channel-fold / TMEM), ``stream_width_bits``,
``block_folding_factor``, and the tensor/block/stream shape hierarchy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qonnx.core.datatype import BaseDataType

from .ordered_parameter import OrderedParameter
from .types import Shape, ShapeHierarchy, TilingSpec, prod


@dataclass(frozen=True)
class InterfaceDesignSpace:
    """Interface structure constant during DSE.

    Stream tiling preserved as a template for resolution with specific
    parallelization parameters.
    """

    name: str
    tensor_shape: Shape
    block_shape: Shape
    stream_tiling: TilingSpec
    datatype: BaseDataType
    is_weight: bool = False
    tensor_name: str | None = None  # ONNX tensor name for initializer lookups

    # Parallelism metadata (None if no stream parameters)
    parallelism_dimension: OrderedParameter | None = None
    parallelism_param: str | None = None


@dataclass(frozen=True)
class InterfaceDesignPoint:
    """Interface instance with a resolved stream_shape (flyweight over the space)."""

    design_space: InterfaceDesignSpace
    stream_shape: Shape

    @property
    def name(self) -> str:
        return self.design_space.name

    @property
    def tensor_shape(self) -> Shape:
        return self.design_space.tensor_shape

    @property
    def block_shape(self) -> Shape:
        return self.design_space.block_shape

    @property
    def datatype(self) -> BaseDataType:
        return self.design_space.datatype

    @property
    def is_weight(self) -> bool:
        return self.design_space.is_weight

    def get_shape(self, hierarchy: ShapeHierarchy) -> Shape:
        if hierarchy == ShapeHierarchy.STREAM:
            return self.stream_shape
        elif hierarchy == ShapeHierarchy.BLOCK:
            return self.design_space.block_shape
        elif hierarchy == ShapeHierarchy.TENSOR:
            return self.design_space.tensor_shape
        else:
            raise ValueError(f"Invalid hierarchy: {hierarchy}")

    @property
    def tensor_blocks_shape(self) -> Shape:
        """Per-dimension blocks to tile the tensor: ceil(tensor_dim / block_dim)."""
        return tuple(
            math.ceil(tensor_dim / block_dim)
            for tensor_dim, block_dim in zip(self.tensor_shape, self.block_shape)
        )

    @property
    def stream_cycles_shape(self) -> Shape:
        """Per-dimension cycles to stream one block: ceil(block_dim / stream_dim)."""
        return tuple(
            math.ceil(block_dim / stream_dim)
            for block_dim, stream_dim in zip(self.block_shape, self.stream_shape)
        )

    @property
    def tensor_folding_factor(self) -> int:
        """Blocks to cover the full tensor (product of tensor_blocks_shape)."""
        return math.prod(self.tensor_blocks_shape)

    @property
    def block_folding_factor(self) -> int:
        """Cycles to stream one block (product of stream_cycles_shape)."""
        return math.prod(self.stream_cycles_shape)

    @property
    def streaming_bandwidth(self) -> int:
        """Elements streamed per cycle."""
        return prod(self.stream_shape)

    @property
    def stream_width_bits(self) -> int:
        """Stream width in bits."""
        return self.streaming_bandwidth * self.datatype.bitwidth()
