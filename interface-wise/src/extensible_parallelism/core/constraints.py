"""
Constraint system for parallelism configuration validation.

Provides various types of constraints that can be applied to tensor parallelism
configurations to ensure validity and optimize resource usage.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Union
from enum import Enum

from .base import ParallelismConfig, TensorSpec, ResourceEstimate


class ConstraintType(Enum):
    """Types of constraints that can be applied."""
    DIVISIBILITY = "divisibility"
    RESOURCE_LIMIT = "resource_limit"
    DEPENDENCY = "dependency"
    PERFORMANCE = "performance"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    CUSTOM = "custom"


@dataclass
class ConstraintViolation:
    """Information about a constraint violation."""
    constraint_name: str
    message: str
    severity: str = "error"  # "error", "warning", "info"
    suggestions: List[str] = None

    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


class Constraint(ABC):
    """
    Abstract base class for all parallelism constraints.

    Constraints validate that a given parallelism configuration is valid
    and can provide suggestions for fixing violations.
    """

    def __init__(self, name: str, constraint_type: ConstraintType,
                 description: str = ""):
        """
        Initialize constraint.

        Args:
            name: Unique name for this constraint
            constraint_type: Type of constraint
            description: Human-readable description
        """
        self.name = name
        self.constraint_type = constraint_type
        self.description = description

    @abstractmethod
    def validate(self, config: ParallelismConfig,
                 tensors: Dict[str, TensorSpec]) -> List[ConstraintViolation]:
        """
        Validate a parallelism configuration against this constraint.

        Args:
            config: Parallelism configuration to validate
            tensors: Tensor specifications for the operation

        Returns:
            List of constraint violations (empty if valid)
        """
        pass

    @abstractmethod
    def suggest_fixes(self, config: ParallelismConfig,
                     tensors: Dict[str, TensorSpec],
                     violation: ConstraintViolation) -> List[ParallelismConfig]:
        """
        Suggest alternative configurations that fix a violation.

        Args:
            config: Current configuration with violation
            tensors: Tensor specifications
            violation: The specific violation to fix

        Returns:
            List of suggested configurations that fix the violation
        """
        pass

    def __str__(self) -> str:
        return f"{self.name} ({self.constraint_type.value})"


class DivisibilityConstraint(Constraint):
    """
    Constraint ensuring tensor dimensions are divisible by parallelism factors.

    This is the most fundamental constraint - if a dimension has size N and
    parallelism factor P, then N must be divisible by P.
    """

    def __init__(self, tensor_name: str, tensor_type: str = "input",
                 dimension_idx: Optional[int] = None):
        """
        Initialize divisibility constraint.

        Args:
            tensor_name: Name of tensor to check
            tensor_type: Type of tensor ("input", "output", "internal")
            dimension_idx: Specific dimension index (None for all dimensions)
        """
        name = f"divisibility_{tensor_type}_{tensor_name}"
        if dimension_idx is not None:
            name += f"_dim{dimension_idx}"

        super().__init__(
            name=name,
            constraint_type=ConstraintType.DIVISIBILITY,
            description=f"Tensor {tensor_type}:{tensor_name} dimensions must be divisible by parallelism factors"
        )

        self.tensor_name = tensor_name
        self.tensor_type = tensor_type
        self.dimension_idx = dimension_idx

    def validate(self, config: ParallelismConfig,
                 tensors: Dict[str, TensorSpec]) -> List[ConstraintViolation]:
        """Validate divisibility constraints."""
        violations = []

        # Get tensor spec
        tensor_key = f"{self.tensor_type}:{self.tensor_name}"
        if tensor_key not in tensors:
            return [ConstraintViolation(
                constraint_name=self.name,
                message=f"Tensor {tensor_key} not found",
                severity="error"
            )]

        tensor_spec = tensors[tensor_key]

        # Get parallelism configuration
        parallelism_list = config.get_tensor_parallelism(self.tensor_name, self.tensor_type)

        for tensor_parallelism in parallelism_list:
            dim_idx = tensor_parallelism.dimension_idx

            # Check if this dimension is constrained
            if self.dimension_idx is not None and dim_idx != self.dimension_idx:
                continue

            # Check divisibility
            if dim_idx >= len(tensor_spec.shape):
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    message=f"Dimension index {dim_idx} out of range for tensor {tensor_key} with shape {tensor_spec.shape}",
                    severity="error"
                ))
                continue

            dim_size = tensor_spec.shape[dim_idx]
            factor = tensor_parallelism.factor

            if dim_size % factor != 0:
                # Find valid factors for suggestions
                valid_factors = []
                for f in range(1, dim_size + 1):
                    if dim_size % f == 0:
                        valid_factors.append(f)

                # Suggest factors close to current factor
                suggestions = []
                for valid_factor in sorted(valid_factors, key=lambda x: abs(x - factor)):
                    if len(suggestions) < 3:  # Limit suggestions
                        suggestions.append(f"Try factor {valid_factor} ({dim_size}/{valid_factor} = {dim_size//valid_factor} chunks)")

                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    message=f"Dimension {dim_idx} of tensor '{tensor_key}' has size {dim_size}, which is not divisible by factor {factor}",
                    severity="error",
                    suggestions=suggestions
                ))

        return violations

    def suggest_fixes(self, config: ParallelismConfig,
                     tensors: Dict[str, TensorSpec],
                     violation: ConstraintViolation) -> List[ParallelismConfig]:
        """Suggest fixed configurations by adjusting parallelism factors."""
        suggestions = []

        tensor_key = f"{self.tensor_type}:{self.tensor_name}"
        if tensor_key not in tensors:
            return suggestions

        tensor_spec = tensors[tensor_key]
        parallelism_list = config.get_tensor_parallelism(self.tensor_name, self.tensor_type)

        for tensor_parallelism in parallelism_list:
            dim_idx = tensor_parallelism.dimension_idx

            if self.dimension_idx is not None and dim_idx != self.dimension_idx:
                continue

            if dim_idx >= len(tensor_spec.shape):
                continue

            dim_size = tensor_spec.shape[dim_idx]
            current_factor = tensor_parallelism.factor

            if dim_size % current_factor != 0:
                # Find valid factors
                valid_factors = [f for f in range(1, dim_size + 1) if dim_size % f == 0]

                # Try factors close to current factor
                for valid_factor in sorted(valid_factors, key=lambda x: abs(x - current_factor)):
                    if valid_factor != current_factor:
                        # Create new configuration with adjusted factor
                        new_config = ParallelismConfig(
                            input_parallelism=dict(config.input_parallelism),
                            output_parallelism=dict(config.output_parallelism),
                            internal_parallelism=dict(config.internal_parallelism),
                            memory_strategy=config.memory_strategy,
                            pipeline_depth=config.pipeline_depth
                        )

                        # Update the specific parallelism factor
                        new_parallelism_list = []
                        for tp in parallelism_list:
                            if tp.dimension_idx == dim_idx:
                                new_tp = TensorParallelism(
                                    dimension_idx=tp.dimension_idx,
                                    factor=valid_factor,
                                    strategy=tp.strategy,
                                    constraints=tp.constraints.copy()
                                )
                                new_parallelism_list.append(new_tp)
                            else:
                                new_parallelism_list.append(tp)

                        new_config.set_tensor_parallelism(self.tensor_name, new_parallelism_list, self.tensor_type)
                        suggestions.append(new_config)

                        if len(suggestions) >= 3:  # Limit number of suggestions
                            break

        return suggestions


class ResourceConstraint(Constraint):
    """
    Constraint limiting hardware resource usage.

    Ensures that the estimated resource usage doesn't exceed available resources.
    """

    def __init__(self, max_dsps: Optional[int] = None,
                 max_brams: Optional[int] = None,
                 max_luts: Optional[int] = None,
                 max_ffs: Optional[int] = None):
        """
        Initialize resource constraint.

        Args:
            max_dsps: Maximum DSP blocks allowed
            max_brams: Maximum BRAM blocks allowed
            max_luts: Maximum LUTs allowed
            max_ffs: Maximum flip-flops allowed
        """
        super().__init__(
            name="resource_limit",
            constraint_type=ConstraintType.RESOURCE_LIMIT,
            description="Hardware resource usage must not exceed limits"
        )

        self.max_dsps = max_dsps
        self.max_brams = max_brams
        self.max_luts = max_luts
        self.max_ffs = max_ffs

    def validate(self, config: ParallelismConfig,
                 tensors: Dict[str, TensorSpec]) -> List[ConstraintViolation]:
        """Validate resource constraints (requires operation to estimate resources)."""
        # This would typically be called by the operation's validate_config method
        # which has access to resource estimation
        violations = []

        # Check configuration-level resource limits
        if self.max_dsps is not None and config.max_dsps is not None:
            if config.max_dsps > self.max_dsps:
                violations.append(ConstraintViolation(
                    constraint_name=self.name,
                    message=f"Configuration requests {config.max_dsps} DSPs but limit is {self.max_dsps}",
                    severity="error",
                    suggestions=[f"Reduce parallelism to use ≤{self.max_dsps} DSPs"]
                ))

        return violations

    def suggest_fixes(self, config: ParallelismConfig,
                     tensors: Dict[str, TensorSpec],
                     violation: ConstraintViolation) -> List[ParallelismConfig]:
        """Suggest configurations with reduced resource usage."""
        suggestions = []

        # Create a configuration with reduced parallelism
        new_config = ParallelismConfig(
            input_parallelism=dict(config.input_parallelism),
            output_parallelism=dict(config.output_parallelism),
            internal_parallelism=dict(config.internal_parallelism),
            memory_strategy=config.memory_strategy,
            pipeline_depth=config.pipeline_depth
        )

        # Reduce parallelism factors by half
        for tensor_name, parallelism_list in new_config.input_parallelism.items():
            for tp in parallelism_list:
                tp.factor = max(1, tp.factor // 2)

        for tensor_name, parallelism_list in new_config.output_parallelism.items():
            for tp in parallelism_list:
                tp.factor = max(1, tp.factor // 2)

        suggestions.append(new_config)
        return suggestions


class MemoryBandwidthConstraint(Constraint):
    """
    Constraint limiting memory bandwidth requirements.
    """

    def __init__(self, max_bandwidth_bits_per_cycle: int):
        """
        Initialize memory bandwidth constraint.

        Args:
            max_bandwidth_bits_per_cycle: Maximum memory bandwidth in bits per cycle
        """
        super().__init__(
            name="memory_bandwidth",
            constraint_type=ConstraintType.MEMORY_BANDWIDTH,
            description=f"Memory bandwidth must not exceed {max_bandwidth_bits_per_cycle} bits/cycle"
        )

        self.max_bandwidth = max_bandwidth_bits_per_cycle

    def validate(self, config: ParallelismConfig,
                 tensors: Dict[str, TensorSpec]) -> List[ConstraintViolation]:
        """Validate memory bandwidth constraints."""
        violations = []

        # Estimate bandwidth requirements (simplified)
        total_bandwidth = 0

        for tensor_name, parallelism_list in config.input_parallelism.items():
            tensor_key = f"input:{tensor_name}"
            if tensor_key in tensors:
                tensor_spec = tensors[tensor_key]

                # Estimate bits per access
                dtype_bits = {"int8": 8, "int16": 16, "float32": 32}.get(tensor_spec.dtype, 32)

                # Calculate parallel access width
                parallel_elements = 1
                for tp in parallelism_list:
                    if tp.dimension_idx < len(tensor_spec.shape):
                        parallel_elements *= tp.factor

                tensor_bandwidth = parallel_elements * dtype_bits
                total_bandwidth += tensor_bandwidth

        if total_bandwidth > self.max_bandwidth:
            violations.append(ConstraintViolation(
                constraint_name=self.name,
                message=f"Estimated memory bandwidth {total_bandwidth} bits/cycle exceeds limit {self.max_bandwidth}",
                severity="error",
                suggestions=["Reduce parallelism factors to decrease bandwidth requirements"]
            ))

        return violations

    def suggest_fixes(self, config: ParallelismConfig,
                     tensors: Dict[str, TensorSpec],
                     violation: ConstraintViolation) -> List[ParallelismConfig]:
        """Suggest configurations with reduced bandwidth requirements."""
        suggestions = []

        # Reduce parallelism to meet bandwidth constraints
        new_config = ParallelismConfig(
            input_parallelism=dict(config.input_parallelism),
            output_parallelism=dict(config.output_parallelism),
            internal_parallelism=dict(config.internal_parallelism),
            memory_strategy=config.memory_strategy,
            pipeline_depth=config.pipeline_depth
        )

        # Systematically reduce parallelism until bandwidth is acceptable
        reduction_factor = 0.8  # Reduce by 20%

        for tensor_name, parallelism_list in new_config.input_parallelism.items():
            for tp in parallelism_list:
                tp.factor = max(1, int(tp.factor * reduction_factor))

        suggestions.append(new_config)
        return suggestions
