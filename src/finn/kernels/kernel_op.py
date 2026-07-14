############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""Op identity.

A :class:`KernelOp` is the *identity* half of the seam: what an op computes and
how its shape/width/folding contract is derived — independent of any backend.
It is a plain class (single inheritance, ``ThresholdingOp(KernelOp)``), **not**
an ``HWCustomOp``: shedding that base is exactly what dissolves the diamond and
the op_type leak that Brainsmith inherited.

The identity is graph-aware only for *derivation* — it reads a ``ModelWrapper``
to build the ``KernelDesignSpace`` and to extract constant initializers into a
:class:`ParamBundle`. Its outputs (a ``KernelDesignPoint`` + a ``ParamBundle``)
are the hermetic values an :class:`~finn.kernels.implementation.Implementation`
consumes. The identity never generates code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping

from onnx import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from .derivation import (
    BuildContext,
    DesignSpaceBuilder,
    KernelDesignPoint,
    KernelDesignSpace,
    KernelSchema,
)
from .implementation import ParamBundle


class KernelOp(ABC):
    """Base class for op identities.

    Subclasses declare structure via :meth:`build_schema`, the golden model via
    :meth:`reference`, and (optionally) ONNX inference via :meth:`can_infer_from`
    / :meth:`infer_from`. Design-space derivation is inherited here, so per-op
    classes carry no duplicated shape logic.
    """

    #: Human/netron op kind; matches the schema name and the node ``op_type``
    #: projection root. Set by subclasses.
    op_kind: str

    # ------------------------------------------------------------------ schema
    @classmethod
    @abstractmethod
    def build_schema(cls, node: NodeProto, model: ModelWrapper | None) -> KernelSchema:
        """Build the :class:`KernelSchema` for this op (static or node-derived)."""

    # --------------------------------------------------------------- reference
    @abstractmethod
    def reference(self, inputs: Mapping[str, Any], attrs: Mapping[str, Any]) -> Mapping[str, Any]:
        """The named Python golden model: pure function from named input arrays
        + resolved attributes to named output arrays. Used for ``python``
        execution and as the correctness oracle for cppsim/rtlsim."""

    # --------------------------------------------------------------- inference
    @classmethod
    def can_infer_from(cls, node: NodeProto, model: ModelWrapper) -> bool:
        """Whether an ONNX ``node`` can be lowered to this kernel (default no)."""
        return False

    @classmethod
    def infer_from(cls, node: NodeProto, model: ModelWrapper) -> Any:
        raise NotImplementedError(f"{cls.__name__}.infer_from()")

    # -------------------------------------------------------------- derivation
    def build_design_space(
        self,
        node: NodeProto,
        model: ModelWrapper,
        param_getter: Callable[[str], Any],
        param_setter: Callable[[str, Any], None],
    ) -> KernelDesignSpace:
        """Construct the design space from the graph + current nodeattrs."""
        schema = self.build_schema(node, model)
        ctx = BuildContext(
            schema=schema,
            model_w=model,
            node_inputs=list(node.input),
            node_outputs=list(node.output),
            param_getter=param_getter,
            param_setter=param_setter,
            node_name=node.name,
        )
        return DesignSpaceBuilder().build(ctx)

    def derive_design_point(
        self,
        design_space: KernelDesignSpace,
        param_getter: Callable[[str], Any],
    ) -> KernelDesignPoint:
        """Configure a concrete design point from current nodeattr values."""
        config = {name: param_getter(name) for name in design_space.parameters}
        return design_space.configure(config)

    # ----------------------------------------------------------- param extract
    @abstractmethod
    def extract_params(self, node: NodeProto, model: ModelWrapper) -> ParamBundle:
        """Pull the constant initializers this op bakes into hardware (weights,
        thresholds, …) into a frozen :class:`ParamBundle`, so the implementation
        never has to read the ``ModelWrapper``."""
