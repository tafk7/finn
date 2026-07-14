############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
############################################################################
"""The single boundary to the FINN graph.

``KernelCustomOp`` is the *one* generic adapter that fronts every dataflow
kernel on the ONNX graph — formalizing the transitional
``getCustomOp(node).set_nodeattr(...)`` bridge the prototype needed into a clean
``CustomOp`` subclass. It carries no op-specific logic: it resolves the op
identity by name, delegates the whole FINN shape/width/exec surface to a
:class:`~finn.kernels.core.KernelCore`, and persists all state as nodeattrs so a
graph round-trips to ONNX with nothing pickled.

The ``HWCustomOp`` base pulls the full FINN runtime (xsi/torch/scipy), which is
only present in the FINN Docker image. To keep the *logic* (in
``finn.kernels.core``) unit-testable outside that image, the heavy import is
guarded here: importing this module without FINN installed yields a
``KernelCustomOp`` whose base is a small stand-in, and the real ``HWCustomOp``
is used whenever it is importable.
"""

from __future__ import annotations

from typing import Any

from qonnx.core.datatype import DataType

from .core import IMPLEMENTATION_ATTR, KernelCore
from .registry import registry as default_registry

try:  # real base in the FINN container
    from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp as _Base

    _HAVE_HWCUSTOMOP = True
except Exception:  # pragma: no cover - exercised only outside the FINN image
    from qonnx.custom_op.base import CustomOp as _Base

    _HAVE_HWCUSTOMOP = False


#: Maps op kind -> KernelOp identity class. Concrete kernels register here so the
#: one adapter can front many op kinds. Populated at import of each kernel.
_IDENTITIES: dict[str, type] = {}


def register_identity(op_kind: str, identity_cls: type) -> None:
    _IDENTITIES[op_kind] = identity_cls


class KernelCustomOp(_Base):
    """Generic graph adapter for dataflow kernels.

    op_type/domain resolution: qonnx resolves ``(domain, op_type)`` to this class
    for every registered kernel op_type; the class keys its own behavior on the
    node's op_type via :attr:`_IDENTITIES`.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)
        self._core: KernelCore | None = None

    # ---------------------------------------------------------------- identity
    def _identity(self):
        op_kind = self.onnx_node.op_type.split("_")[0]
        try:
            return _IDENTITIES[op_kind]()
        except KeyError:
            raise KeyError(f"no kernel identity registered for op_kind {op_kind!r}")

    def core(self, model) -> KernelCore:
        """Bind (or rebind) a KernelCore against the given model."""
        if self._core is None or self._core.model is not model:
            self._core = KernelCore(
                identity=self._identity(),
                node=self.onnx_node,
                model=model,
                get_nodeattr=self.get_nodeattr,
                set_nodeattr=self.set_nodeattr,
                registry=default_registry,
            )
        return self._core

    # --------------------------------------------------------------- nodeattrs
    def get_nodeattr_types(self):
        attrs = super().get_nodeattr_types() if _HAVE_HWCUSTOMOP else {}
        attrs[IMPLEMENTATION_ATTR] = ("s", False, "")
        # Schema-derived nodeattrs (datatypes, tiling params, kernel params).
        schema = self._identity().build_schema(self.onnx_node, None)
        attrs.update(schema.build_nodeattr_registry())
        # Any selected backend's knobs.
        impl_name = _safe_get(self, IMPLEMENTATION_ATTR)
        if impl_name:
            impl_cls = default_registry.get_by_name(impl_name).__class__
            attrs.update(getattr(impl_cls, "knob_specs", {}))
        return attrs

    # ---------------------------------------------------- FINN shape delegation
    def get_input_datatype(self, ind=0):
        return self._require_core().get_input_datatype(ind)

    def get_output_datatype(self, ind=0):
        return self._require_core().get_output_datatype(ind)

    def get_normal_input_shape(self, ind=0):
        return self._require_core().get_normal_input_shape(ind)

    def get_normal_output_shape(self, ind=0):
        return self._require_core().get_normal_output_shape(ind)

    def get_folded_input_shape(self, ind=0):
        return self._require_core().get_folded_input_shape(ind)

    def get_folded_output_shape(self, ind=0):
        return self._require_core().get_folded_output_shape(ind)

    def get_instream_width(self, ind=0):
        return self._require_core().get_instream_width(ind)

    def get_outstream_width(self, ind=0):
        return self._require_core().get_outstream_width(ind)

    def get_number_output_values(self):
        return self._require_core().get_number_output_values()

    def get_exp_cycles(self):
        return self._require_core().get_exp_cycles()

    # --------------------------------------------------------------- CustomOp
    def make_shape_compatible_op(self, model):
        oshape = self.core(model).get_normal_output_shape(0)
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        core = self.core(model)
        for i, out_name in enumerate(self.onnx_node.output):
            if out_name:
                model.set_tensor_datatype(out_name, core.get_output_datatype(i))

    def execute_node(self, context, graph):
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.util.basic import qonnx_make_model

        model = ModelWrapper(qonnx_make_model(graph))
        self.core(model).execute_python(context)

    def verify_node(self):
        return []

    # ------------------------------------------------------------------ helpers
    def _require_core(self) -> KernelCore:
        if self._core is None:
            raise RuntimeError(
                f"{self.onnx_node.name}: call core(model) before shape queries"
            )
        return self._core


def _safe_get(op, name) -> Any:
    try:
        return op.get_nodeattr(name)
    except Exception:
        return ""
