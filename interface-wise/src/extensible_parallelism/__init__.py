"""Extensible Parallelism System for FINN.

A flexible framework for implementing and optimizing parallelizable operations
with constraint-based configuration and resource estimation.
"""

from .core.base import (
    ParallelizableOperation,
    TensorSpec,
    ParallelismConfig,
    TensorParallelism,
    ResourceEstimate,
    ParallelismStrategy,
    MemoryStrategy
)
from .core.registry import OperationRegistry, register_operation
from .core.constraints import (
    Constraint,
    ConstraintViolation,
    DivisibilityConstraint,
    ResourceConstraint,
    MemoryBandwidthConstraint
)

# Import operations to trigger registration
from .operations import (
    MatrixVectorOperation,
    ElementWiseOperation,
    ConvolutionOperation
)

__version__ = "0.1.0"

__all__ = [
    # Core base classes
    "ParallelizableOperation",
    "TensorSpec",
    "ParallelismConfig",
    "TensorParallelism",
    "ResourceEstimate",
    "ParallelismStrategy",
    "MemoryStrategy",

    # Registry system
    "OperationRegistry",
    "register_operation",

    # Constraint system
    "Constraint",
    "ConstraintViolation",
    "DivisibilityConstraint",
    "ResourceConstraint",
    "MemoryBandwidthConstraint",

    # Operations
    "MatrixVectorOperation",
    "ElementWiseOperation",
    "ConvolutionOperation"
]
