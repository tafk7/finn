############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
############################################################################

"""The resolve engine: Context, Axis, Derived, Predicate, Schema, and resolve.

A design space is data — a :class:`Schema` of guarded :class:`Axis` choices,
computed :class:`Derived` quantities, and :class:`Predicate` legality checks —
resolved against a :class:`Context` of givens into a :class:`Point` or an
:class:`Illegal`. See ``kernel-design/kernel-final-design/design-space-model.md``.
"""

from ..engine.axis import (
    Axis,
    Domain,
    PredicateDomain,
    discrete_axis,
    divisor_axis,
    fixed_axis,
    predicate_axis,
)
from ..emit.artifacts import (
    Artifacts,
    BindError,
    BitWidth,
    Bool,
    DataFile,
    Dim,
    GeneratedFile,
    IPICommands,
    Raw,
    RtlModule,
    StaticFile,
    Template,
    TemplateError,
    bind,
)
from ..engine.context import Context
from ..engine.derived import Derived
from .folding import threshold_fold_depth, weight_fold_depth
from ..emit.manifest import ArtifactManifest, SourceFile, read_abc
from .backend import (
    BACKEND_AXIS,
    Backend,
    EmitError,
    PoolError,
    emit_point,
    pool_schema,
)
from ..engine.point import AbsentAxisError, Illegal, Point
from .ports import (
    Direction,
    Kind,
    Port,
    Role,
    STANDARD_BINDINGS,
)
from ..engine.predicate import Predicate, predicate
from ..engine.resolve import resolve
from ..emit.stitch import Cell, StitchError, stitch
from ..engine.schema import Schema, SchemaError
from .demand import ParamDemand
from .delivery import DeliveredParam
from .backend_interface import Interface, backend_interface_for
from .kernel import InterfaceSchema, Kernel, KernelError, KernelSchema
from .tiling import (
    FULL,
    BlockExtent,
    BroadcastAware,
    Const,
    Div,
    GeneratedTiling,
    Mul,
    Ref,
    StreamFold,
    TileError,
    TileExpr,
    broadcast_aware,
    const,
    derive,
    entry_deps,
    eval_entry,
    generate_tiling,
    param,
    stream_width_key,
)

__all__ = [
    "Context",
    "Axis",
    "Domain",
    "PredicateDomain",
    "discrete_axis",
    "divisor_axis",
    "fixed_axis",
    "predicate_axis",
    "Derived",
    "weight_fold_depth",
    "threshold_fold_depth",
    "ArtifactManifest",
    "SourceFile",
    "read_abc",
    "Predicate",
    "predicate",
    "Backend",
    "pool_schema",
    "PoolError",
    "emit_point",
    "EmitError",
    "Template",
    "TemplateError",
    "RtlModule",
    "bind",
    "BindError",
    "BitWidth",
    "Dim",
    "Bool",
    "Raw",
    "GeneratedFile",
    "DataFile",
    "StaticFile",
    "IPICommands",
    "Artifacts",
    "Point",
    "Illegal",
    "AbsentAxisError",
    "Direction",
    "Kind",
    "Role",
    "Port",
    "STANDARD_BINDINGS",
    "Cell",
    "stitch",
    "StitchError",
    "Schema",
    "SchemaError",
    "resolve",
    "BACKEND_AXIS",
    "Kernel",
    "KernelSchema",
    "InterfaceSchema",
    "KernelError",
    "ParamDemand",
    "DeliveredParam",
    "Interface",
    "backend_interface_for",
    "TileExpr",
    "TileError",
    "Ref",
    "Const",
    "Mul",
    "Div",
    "BroadcastAware",
    "FULL",
    "BlockExtent",
    "StreamFold",
    "GeneratedTiling",
    "generate_tiling",
    "stream_width_key",
    "derive",
    "param",
    "const",
    "broadcast_aware",
    "eval_entry",
    "entry_deps",
]
