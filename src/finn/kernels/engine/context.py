############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""``Context`` — everything known at t=0 that is NOT a design choice.

Tensor shapes and datatypes come from the ONNX graph; ``initializer`` exposes
the actual weight/threshold VALUES (what makes data-dependent derivation and
feasibility expressible); ``fpgapart`` and ``toolchain_version`` are build-config
globals. There is no phase gating — a predicate or derivation reads whatever it
needs (design-space-model.md §1.1).

Two constructors: :meth:`from_model` wraps a QONNX ``ModelWrapper`` for real use;
the default constructor takes plain dicts so tests need no ONNX graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Context:
    """The immutable givens a schema resolves against.

    Attributes:
        shapes: name -> tensor shape (tuple[int, ...]).
        datatypes: name -> QONNX DataType (graph dtype, a given).
        initializers: name -> ndarray of actual values, or absent if dynamic.
        fpgapart: target device string (e.g. "xc7z020clg400-1", "xcvc1902-...").
        toolchain_version: toolchain version string, or None.
        clk_ns: target clock period in ns (toolchain global; drives SEGMENTLEN).
        arities: variadic-group name -> concrete slot count for this node (read by
            :meth:`arity`). Empty for the all-fixed-arity common case (MVAU/Thresholding).
    """

    shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    datatypes: Mapping[str, Any] = field(default_factory=dict)
    initializers: Mapping[str, np.ndarray] = field(default_factory=dict)
    fpgapart: str = ""
    toolchain_version: str | None = None
    clk_ns: float | None = None
    arities: Mapping[str, int] = field(default_factory=dict)

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def tensor_datatype(self, name: str):
        return self.datatypes[name]

    def initializer(self, name: str) -> np.ndarray | None:
        return self.initializers.get(name)

    def arity(self, group: str) -> int:
        """How many concrete node slots a VARIADIC interface group expands to for this node —
        the count :meth:`Kernel.expanded_interfaces` reads to expand a
        :class:`~finn.kernels.model.ports.Variadic` ``(count_from=group)``. Sourced from the
        instantiated node's wired slots (the ONNX variadic slot count), so it is a Context
        given, not a design choice. The adapter builds a Context that records each variadic
        group's count under ``group``; the default plain-dict Context reads it from
        ``self.arities``."""
        return self.arities[group]

    def has_tensor(self, name: str) -> bool:
        """Whether a tensor is WIRED for this node — the ARITY primitive an op reads to
        answer "is this OPTIONAL operand present?". Keyed on the shape map (every wired tensor
        has a shape), so it is uniform whether the operand is a live activation or a baked
        constant — unlike ``initializer(name) is not None``, which only sees constants. The
        schema declares a slot MAY exist (``InterfaceSchema.optional``); this answers whether
        it DOES on the instantiated node's Context. Same presence rule as
        :meth:`Kernel.present_interfaces`."""
        return name in self.shapes

    @property
    def clk(self) -> float | None:
        return self.clk_ns

    @classmethod
    def from_model(
        cls,
        model,
        fpgapart: str,
        toolchain_version: str | None = None,
        clk_ns: float | None = None,
    ) -> "Context":
        """Build a Context from a QONNX ModelWrapper.

        Reads shapes, datatypes and initializers lazily-materialised into dicts
        keyed by tensor name, so downstream code never needs the graph itself.
        """
        shapes: dict[str, tuple[int, ...]] = {}
        datatypes: dict[str, Any] = {}
        initializers: dict[str, np.ndarray] = {}
        for tensor in model.graph.input:
            name = tensor.name
            shapes[name] = tuple(model.get_tensor_shape(name))
            datatypes[name] = model.get_tensor_datatype(name)
        for node in model.graph.node:
            for name in list(node.input) + list(node.output):
                if not name:
                    continue
                shape = model.get_tensor_shape(name)
                if shape is not None:
                    shapes[name] = tuple(shape)
                datatypes[name] = model.get_tensor_datatype(name)
                init = model.get_initializer(name)
                if init is not None:
                    initializers[name] = init
        return cls(
            shapes=shapes,
            datatypes=datatypes,
            initializers=initializers,
            fpgapart=fpgapart,
            toolchain_version=toolchain_version,
            clk_ns=clk_ns,
        )
