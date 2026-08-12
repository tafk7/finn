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

**On the two arrival phases (F11), and why this type survives.** The givens here come from
two owners at two different times. GRAPH givens (shapes, datatypes, initializers) are
derivable from ``(ModelWrapper, node)`` the moment the node exists. DEVICE facts
(``fpgapart``, ``clk_ns``, ``toolchain_version``) belong to the build config and arrive from
whichever step holds it. Bundling them meant constructing a Context demanded a part before
any caller had one — and the workaround for that, an ``fpgapart`` nodeattr no op declares,
is what made the field universally empty on every node.

The taxonomy pass fixed the CAUSE by making device facts an INPUT — ``attach_device(fpgapart,
clk_ns, toolchain_version)``, threaded from ``SpecializeKernels`` — and then deliberately did
NOT delete this dataclass. Churning 267 references buys no behavioural
change: once the facts arrive correctly, one backend carrying both is a grouping, not a phase
confusion. The finding was the phantom nodeattr, not the struct.

Context reads are audited by ``RecordingContext`` in ``tests/engine/test_deps_audit.py`` —
added because they previously were not, which is precisely how F11 stayed invisible to the
harness meant to catch it.
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
        runtime_writeable: interface name -> whether the deployed artifact must let the
            driver rewrite that parameter's values (read by :meth:`is_runtime_writeable`).
            A phase-0 MANDATE, not a design choice — see the method docstring.
    """

    shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    datatypes: Mapping[str, Any] = field(default_factory=dict)
    initializers: Mapping[str, np.ndarray] = field(default_factory=dict)
    fpgapart: str = ""
    toolchain_version: str | None = None
    clk_ns: float | None = None
    arities: Mapping[str, int] = field(default_factory=dict)
    sparsity: Mapping[str, Any] = field(default_factory=dict)
    runtime_writeable: Mapping[str, bool] = field(default_factory=dict)

    def tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def tensor_datatype(self, name: str):
        return self.datatypes[name]

    def initializer(self, name: str) -> np.ndarray | None:
        return self.initializers.get(name)

    def tensor_sparsity(self, name: str):
        """The tensor's declared sparsity annotation, or ``None`` for a dense tensor.

        A given, like a shape or a dtype — carried here so a rule about it can be a declared
        :class:`~finn.kernels.engine.constraints.Constraint` rather than a hand-written check
        in a frontend claim, which is how it could drift from the pool it is meant to describe."""
        return self.sparsity.get(name)

    def is_runtime_writeable(self, name: str) -> bool:
        """Whether the deployed artifact must let the DRIVER rewrite this parameter's values
        after the bitstream ships. Default ``False``.

        A phase-0 MANDATE on the design space, not a folding knob: it does not describe *how
        to build*, it describes *what the deployed artifact must support*. Baseline FINN files
        it with the folding config purely by accident of its step list (there is no field for
        it in ``DataflowBuildConfig``), and treats it as a per-node attribute that later passes
        may flip — which is what let a capability gate consult a value chosen after
        specialization.

        Carried here because it determines OWNERSHIP of the parameter values: a runtime-writable
        parameter is one the kernel cannot see at build time, so no narrowing of it is sound.
        With this as a given, ``ParamDatatype.values_visible`` — and every dtype derived from it
        — settles at INFER rather than after delivery is chosen. See ``decisions.md``,
        *"An owned initializer is DOWNSTREAM of its kernel"*."""
        return bool(self.runtime_writeable.get(name, False))

    def arity(self, group: str) -> int:
        """How many concrete node slots a VARIADIC interface group expands to for this node —
        the count :meth:`DataflowOp.expanded_interfaces` reads to expand a
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
        :meth:`DataflowOp.present_interfaces`."""
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
        sparsity: dict[str, Any] = {}
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
                sp = model.get_tensor_sparsity(name)
                if sp is not None:
                    sparsity[name] = sp
        return cls(
            shapes=shapes,
            datatypes=datatypes,
            initializers=initializers,
            fpgapart=fpgapart,
            toolchain_version=toolchain_version,
            clk_ns=clk_ns,
            sparsity=sparsity,
        )
