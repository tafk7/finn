############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""The adapter logic, free of the FINN graph base class.

``KernelCore`` is everything the graph adapter *does* — resolve config from
nodeattrs, derive the design point, delegate the FINN shape/width surface to it,
select an implementation, and run codegen/execution — expressed against plain
callables rather than ``HWCustomOp``. The thin ``KernelCustomOp(HWCustomOp)``
shell in ``adapter.py`` owns only the qonnx-resolution plumbing and forwards to
a ``KernelCore``. Splitting it this way keeps the one graph boundary minimal and
lets the real logic be unit-tested without the full FINN runtime.

The core is the single place that touches the node (via the injected
``get_nodeattr``) — exactly as intended: ``emit`` stays hermetic because the
core resolves values and hands them in as data.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping

from onnx import NodeProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from .implementation import Artifacts, Implementation, ParamBundle, SelectionContext
from .kernel_op import KernelOp
from .registry import KernelRegistry, registry as default_registry

#: The ``implementation`` nodeattr: which backend was selected. Empty until
#: specialization. This is the by-name key the registry resolves on reload.
IMPLEMENTATION_ATTR = "implementation"


class KernelCore:
    """Binds an op identity to the graph node, deriving everything else.

    Parameters
    ----------
    identity:
        The :class:`KernelOp` for this op kind.
    node, model:
        The ONNX node and its model — used only for derivation and param
        extraction, never passed onward to ``emit``.
    get_nodeattr, set_nodeattr:
        Callables onto the node's attributes (the adapter supplies
        ``HWCustomOp.get_nodeattr``; tests supply a dict's methods).
    registry:
        The implementation pool (defaults to the process-wide registry).
    """

    def __init__(
        self,
        identity: KernelOp,
        node: NodeProto,
        model: ModelWrapper,
        get_nodeattr: Callable[[str], Any],
        set_nodeattr: Callable[[str, Any], None],
        registry: KernelRegistry | None = None,
    ) -> None:
        self.identity = identity
        self.node = node
        self.model = model
        self._get = get_nodeattr
        self._set = set_nodeattr
        self.registry = registry or default_registry
        self._space = None
        self._point = None

    # ----------------------------------------------------------- derivation
    @property
    def design_point(self):
        if self._point is None:
            if self._space is None:
                self._space = self.identity.build_design_space(
                    self.node, self.model, self._get, self._set
                )
            self._point = self.identity.derive_design_point(self._space, self._get)
        return self._point

    def invalidate(self) -> None:
        self._space = None
        self._point = None

    # ------------------------------------------------------- config resolve
    def resolve_config(self, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Read the resolved nodeattr map ``emit`` and ``precondition`` consume.

        This is the *one* place nodeattrs are read for codegen. Kernel params
        (``act_val``, ``num_steps``), interface datatype strings, the module
        name, and any selected backend's knobs are gathered here and passed on
        as plain data.
        """
        schema = self.identity.build_schema(self.node, self.model)
        config: dict[str, Any] = {}
        for name in schema.kernel_params:
            config[name] = self._get(name)
        for i in range(len(self.node.input)):
            key = f"input{i}Datatype"
            config[key] = self._get(key)
        for i in range(len(self.node.output)):
            key = f"output{i}Datatype"
            config[key] = self._get(key)
        config["module_name"] = self.node.name
        # Selected backend's knobs, if any. Fall back to each knob's declared
        # default when the nodeattr is unset (the spec's 3rd tuple element),
        # rather than passing an empty string downstream.
        impl_name = self._get(IMPLEMENTATION_ATTR)
        if impl_name:
            impl_cls = self.registry.get_by_name(impl_name).__class__
            for knob, spec in getattr(impl_cls, "knob_specs", {}).items():
                value = self._get(knob)
                if value in ("", None):
                    value = spec[2] if len(spec) > 2 else value
                config[knob] = value
        if extra:
            config.update(extra)
        return config

    # --------------------------------------------------------- selection
    def selection_context(self, fpgapart: str) -> SelectionContext:
        return SelectionContext(
            fpgapart=fpgapart,
            design_point=self.design_point,
            params=self.identity.extract_params(self.node, self.model),
            config=self.resolve_config(),
        )

    def select(self, fpgapart: str, cost_fn=None) -> Implementation:
        """Pick and persist an implementation for this node's context."""
        ctx = self.selection_context(fpgapart)
        impl = self.registry.select(self.identity.op_kind, ctx, cost_fn=cost_fn)
        self._set(IMPLEMENTATION_ATTR, impl.name)
        return impl

    @property
    def implementation(self) -> Implementation:
        """The currently-selected implementation, resolved by name (nothing
        pickled — this is the ONNX round-trip path)."""
        name = self._get(IMPLEMENTATION_ATTR)
        if not name:
            raise RuntimeError(f"{self.node.name}: not specialized (no implementation)")
        return self.registry.get_by_name(name)

    # --------------------------------------------------------- codegen
    def emit(self) -> Artifacts:
        """Run the selected backend's hermetic codegen for this node."""
        params = self.identity.extract_params(self.node, self.model)
        return self.implementation.emit(self.design_point, params, self.resolve_config())

    # ------------------------------------------ FINN shape/width delegation
    def get_input_datatype(self, ind: int = 0) -> DataType:
        return self.design_point.input_list[ind].datatype

    def get_output_datatype(self, ind: int = 0) -> DataType:
        return self.design_point.output_list[ind].datatype

    def get_normal_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        return tuple(self.design_point.input_list[ind].tensor_shape)

    def get_normal_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        return tuple(self.design_point.output_list[ind].tensor_shape)

    def get_folded_input_shape(self, ind: int = 0) -> tuple[int, ...]:
        return self._folded(self.design_point.input_list[ind])

    def get_folded_output_shape(self, ind: int = 0) -> tuple[int, ...]:
        return self._folded(self.design_point.output_list[ind])

    def get_instream_width(self, ind: int = 0) -> int:
        return self.design_point.input_list[ind].stream_width_bits

    def get_outstream_width(self, ind: int = 0) -> int:
        return self.design_point.output_list[ind].stream_width_bits

    def get_number_output_values(self) -> int:
        folded = self.get_folded_output_shape(0)
        return int(math.prod(folded[:-1]))

    def get_exp_cycles(self) -> int:
        return int(self.design_point.initiation_interval)

    @staticmethod
    def _folded(iface) -> tuple[int, ...]:
        fold = [t // s for t, s in zip(iface.tensor_shape, iface.stream_shape)]
        return tuple(fold + [int(math.prod(iface.stream_shape))])

    # --------------------------------------------------------- execution
    def execute_python(self, context: dict) -> None:
        """Golden execution via the identity's reference model."""
        node = self.node
        inputs = {}
        schema = self.identity.build_schema(node, self.model)
        for iface, tensor_name in zip(schema.inputs, node.input):
            inputs[iface.name] = context[tensor_name]
        attrs = self.resolve_config()
        outputs = self.identity.reference(inputs, attrs)
        for iface, tensor_name in zip(schema.outputs, node.output):
            context[tensor_name] = outputs[iface.name]
