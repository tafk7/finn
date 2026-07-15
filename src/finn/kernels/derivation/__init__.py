############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
#
# Vendored into FINN from brainsmith.dataflow for the dataflow-kernel backend.
# The declarative schema-derivation + DSE value objects are reused verbatim;
# the HWCustomOp-coupled `kernel_op.py` is intentionally NOT vendored (the new
# FINN-native `KernelOp` identity replaces it).
############################################################################
"""Core dataflow modeling components (vendored derivation layer).

Schemas define STRUCTURE, not storage. ModelWrapper is the single source of
truth for shapes. Only datatypes and user parameters persist in nodeattrs.

    KernelSchema (defines + validates)
      -> DesignSpaceBuilder (constructs)
      -> KernelDesignSpace -> KernelDesignPoint
"""

# QONNX types (direct from QONNX)
from qonnx.core.datatype import BaseDataType, DataType

from .builder import BuildContext, DesignSpaceBuilder
from .constraints import (
    Constraint,
    DatatypeInteger,
    DimensionDivisible,
    IsDynamic,
    IsStatic,
)
from .dse_models import (
    InterfaceDesignPoint,
    InterfaceDesignSpace,
    KernelDesignPoint,
    KernelDesignSpace,
)
from .inference_helpers import lift_scalar_to_rank1
from .ordered_parameter import OrderedParameter
from .schemas import InputSchema, KernelSchema, OutputSchema, ParameterSpec
from .spec_helpers import constant_datatype, derive_dim
from .transformation import TransformationResult
from .types import (
    FULL_DIM,
    FULL_SHAPE,
    VALUE_OPTIMIZED,
    Shape,
    ShapeHierarchy,
)
from .validation import (
    ConfigurationValidationContext,
    DesignSpaceValidationContext,
    RealizationValidationContext,
    ValidationError,
)

__all__ = [
    # Schema
    "KernelSchema",
    "InputSchema",
    "OutputSchema",
    "ParameterSpec",
    # Builder
    "DesignSpaceBuilder",
    "BuildContext",
    # Immutable models
    "KernelDesignSpace",
    "KernelDesignPoint",
    "InterfaceDesignSpace",
    "InterfaceDesignPoint",
    "OrderedParameter",
    # Validation / constraints
    "Constraint",
    "ValidationError",
    "DesignSpaceValidationContext",
    "ConfigurationValidationContext",
    "RealizationValidationContext",
    "DatatypeInteger",
    "DimensionDivisible",
    "IsDynamic",
    "IsStatic",
    # Transformation
    "TransformationResult",
    # Schema helpers
    "derive_dim",
    "constant_datatype",
    "lift_scalar_to_rank1",
    # Types
    "Shape",
    "ShapeHierarchy",
    "FULL_DIM",
    "FULL_SHAPE",
    "VALUE_OPTIMIZED",
    # QONNX re-exports
    "DataType",
    "BaseDataType",
]
