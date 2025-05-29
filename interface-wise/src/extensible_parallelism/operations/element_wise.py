"""Element-wise operation implementation."""

from typing import Dict, List, Optional, Callable, Any
import math
from dataclasses import dataclass
from enum import Enum

from ..core.base import (
    ParallelizableOperation,
    TensorSpec,
    ParallelismConfig,
    TensorParallelism,
    ResourceEstimate,
    ParallelismStrategy,
    MemoryStrategy
)
from ..core.registry import register_operation
from ..core.constraints import Constraint, DivisibilityConstraint
from ..core.constants import (
    VALID_DTYPES, 
    DTYPE_BITS, 
    BRAM_BITS_PER_BLOCK,
    DEFAULT_ACTIVATION_DTYPE
)


class ElementWiseOpType(Enum):
    """Supported element-wise operation types."""
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    DIV = "div"
    MAX = "max"
    MIN = "min"
    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    ABS = "abs"
    SQRT = "sqrt"
    EXP = "exp"
    LOG = "log"


@dataclass
class ElementWiseParams:
    """Parameters for Element-wise operation."""
    operation_type: ElementWiseOpType
    tensor_shape: tuple
    input_dtype: str = DEFAULT_ACTIVATION_DTYPE
    output_dtype: str = DEFAULT_ACTIVATION_DTYPE
    num_inputs: int = 1  # 1 for unary ops, 2 for binary ops
    activation_params: Optional[Dict[str, Any]] = None  # For parameterized activations

    def __post_init__(self):
        """Validate parameters."""
        # Validate tensor dimensions
        if not all(dim > 0 for dim in self.tensor_shape):
            raise ValueError("All tensor dimensions must be positive")

        # Validate data types
        if self.input_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid input dtype: {self.input_dtype}")
        if self.output_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid output dtype: {self.output_dtype}")

        # Validate num_inputs based on operation type
        unary_ops = {ElementWiseOpType.RELU, ElementWiseOpType.SIGMOID, ElementWiseOpType.TANH}
        binary_ops = {ElementWiseOpType.ADD, ElementWiseOpType.SUB, ElementWiseOpType.MUL,
                     ElementWiseOpType.DIV, ElementWiseOpType.MAX, ElementWiseOpType.MIN}

        if self.operation_type in unary_ops and self.num_inputs != 1:
            raise ValueError(f"Unary operation {self.operation_type.value} requires 1 input, got {self.num_inputs}")
        elif self.operation_type in binary_ops and self.num_inputs != 2:
            raise ValueError(f"Binary operation {self.operation_type.value} requires 2 inputs, got {self.num_inputs}")


@register_operation("element_wise", "computation")
class ElementWiseOperation(ParallelizableOperation):
    """
    Element-wise operation (equivalent to VVAU).

    Performs element-wise operations on tensors such as:
    - Binary operations: addition, subtraction, multiplication, division, max, min
    - Unary operations: ReLU, sigmoid, tanh, absolute value, square root, exp, log

    This operation can be parallelized along:
    - SIMD: Parallel processing of multiple elements
    """

    def __init__(self, name: str, params: ElementWiseParams):
        """Initialize Element-wise operation.

        Args:
            name: Unique name for this operation instance
            params: Operation parameters including operation type and tensor shape
        """
        super().__init__(name)
        self.params = params
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate operation parameters."""
        if not all(dim > 0 for dim in self.params.tensor_shape):
            raise ValueError("All tensor dimensions must be positive")

        # Validate data types
        if self.params.input_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid input dtype: {self.params.input_dtype}")
        if self.params.output_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid output dtype: {self.params.output_dtype}")

        # Validate number of inputs based on operation type
        binary_ops = {ElementWiseOpType.ADD, ElementWiseOpType.SUB, ElementWiseOpType.MUL,
                     ElementWiseOpType.DIV, ElementWiseOpType.MAX, ElementWiseOpType.MIN}
        unary_ops = {ElementWiseOpType.RELU, ElementWiseOpType.SIGMOID, ElementWiseOpType.TANH,
                    ElementWiseOpType.ABS, ElementWiseOpType.SQRT, ElementWiseOpType.EXP,
                    ElementWiseOpType.LOG}

        if self.params.operation_type in binary_ops and self.params.num_inputs != 2:
            raise ValueError(f"Binary operation {self.params.operation_type.value} requires 2 inputs")
        elif self.params.operation_type in unary_ops and self.params.num_inputs != 1:
            raise ValueError(f"Unary operation {self.params.operation_type.value} requires 1 input")

    def get_input_tensors(self) -> Dict[str, TensorSpec]:
        """Get input tensor specifications."""
        tensors = {}

        # Create dimension names based on tensor shape
        dim_names = [f"dim_{i}" for i in range(len(self.params.tensor_shape))]

        for i in range(self.params.num_inputs):
            tensors[f"input_{i}"] = TensorSpec(
                name=f"input_{i}",
                shape=self.params.tensor_shape,
                dtype=self.params.input_dtype,
                dimension_names=dim_names
            )

        return tensors

    def get_output_tensors(self) -> Dict[str, TensorSpec]:
        """Get output tensor specifications."""
        dim_names = [f"dim_{i}" for i in range(len(self.params.tensor_shape))]

        return {
            "output": TensorSpec(
                name="output",
                shape=self.params.tensor_shape,
                dtype=self.params.output_dtype,
                dimension_names=dim_names
            )
        }

    def get_internal_tensors(self) -> Dict[str, TensorSpec]:
        """Get internal tensor specifications."""
        # Most element-wise operations don't need internal tensors
        # Complex operations might need temporary storage
        if self._needs_internal_storage():
            dim_names = [f"dim_{i}" for i in range(len(self.params.tensor_shape))]
            return {
                "temp": TensorSpec(
                    name="temp",
                    shape=self.params.tensor_shape,
                    dtype=self.params.output_dtype,
                    dimension_names=dim_names
                )
            }
        return {}

    def get_constraints(self) -> List[Constraint]:
        """Get operation constraints."""
        constraints = []

        # SIMD dimension must divide each tensor dimension
        # For simplicity, assume parallelization along the last dimension
        if len(self.params.tensor_shape) > 0:
            last_dim_idx = len(self.params.tensor_shape) - 1
            constraints.append(DivisibilityConstraint(
                tensor_name="output",
                tensor_type="output",
                dimension_idx=last_dim_idx
            ))

        return constraints

    def estimate_resources(self, config: ParallelismConfig) -> ResourceEstimate:
        """Estimate hardware resource usage."""
        # Get parallelism factor (SIMD)
        total_elements = math.prod(self.params.tensor_shape)
        simd_factor = self._get_simd_factor(config)

        # Calculate resource usage based on operation type
        resources = self._estimate_operation_resources(simd_factor)

        # Add memory resources for buffering
        if self._needs_buffering():
            buffer_brams = math.ceil(total_elements *
                                   self._get_dtype_bits(self.params.input_dtype) / BRAM_BITS_PER_BLOCK)
            resources.brams += buffer_brams

        return resources

    def validate_config(self, config: ParallelismConfig) -> bool:
        """Validate parallelism configuration."""
        try:
            # Get actual tensors from the config (use new API with fallback to old API)
            input_tensors = getattr(config, 'input_tensors', {})
            output_tensors = getattr(config, 'output_tensors', {})

            # Check that all required tensors have parallelism specified
            required_input_tensors = set(self.get_input_tensors().keys())
            required_output_tensors = set(self.get_output_tensors().keys())

            config_input_tensors = set(input_tensors.keys())
            config_output_tensors = set(output_tensors.keys())

            if not required_input_tensors.issubset(config_input_tensors):
                return False
            if not required_output_tensors.issubset(config_output_tensors):
                return False

            # Gather all tensor specifications for constraint validation
            all_tensors = {}
            for name, spec in self.get_input_tensors().items():
                all_tensors[f"input:{name}"] = spec
            for name, spec in self.get_output_tensors().items():
                all_tensors[f"output:{name}"] = spec
            for name, spec in self.get_internal_tensors().items():
                all_tensors[f"internal:{name}"] = spec

            # Validate constraints
            constraints = self.get_constraints()
            for constraint in constraints:
                violations = constraint.validate(config, all_tensors)
                if violations:  # If there are any violations, config is invalid
                    return False

            return True
        except Exception:
            return False

    def get_default_parallelism_config(self) -> ParallelismConfig:
        """Get default parallelism configuration."""
        dim_names = [f"dim_{i}" for i in range(len(self.params.tensor_shape))]
        default_factors = {name: 1 for name in dim_names}

        input_tensors = {}
        for i in range(self.params.num_inputs):
            input_tensors[f"input_{i}"] = TensorParallelism(
                dimension_factors=default_factors.copy(),
                strategy=ParallelismStrategy.SPATIAL,
                memory_strategy=MemoryStrategy.STREAMING
            )

        output_tensors = {
            "output": TensorParallelism(
                dimension_factors=default_factors.copy(),
                strategy=ParallelismStrategy.SPATIAL,
                memory_strategy=MemoryStrategy.STREAMING
            )
        }

        internal_tensors = {}
        if self._needs_internal_storage():
            internal_tensors["temp"] = TensorParallelism(
                dimension_factors=default_factors.copy(),
                strategy=ParallelismStrategy.SPATIAL,
                memory_strategy=MemoryStrategy.REGISTERS
            )

        return ParallelismConfig(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            internal_tensors=internal_tensors
        )

    def get_parallelism_bounds(self) -> Dict[str, tuple]:
        """Get bounds for parallelism factors."""
        # For element-wise operations, SIMD can be up to the total number of elements
        max_simd = math.prod(self.params.tensor_shape)
        return {
            "SIMD": (1, max_simd)
        }

    def _get_simd_factor(self, config: ParallelismConfig) -> int:
        """Extract SIMD factor from configuration."""
        # Use the parallelism factor of the last dimension of the output tensor
        if "output" in config.output_tensors:
            last_dim = f"dim_{len(self.params.tensor_shape) - 1}"
            return config.output_tensors["output"].dimension_factors.get(last_dim, 1)
        return 1

    def _estimate_operation_resources(self, simd_factor: int) -> ResourceEstimate:
        """Estimate resources for the specific operation type."""
        op_type = self.params.operation_type

        # Resource estimates per SIMD lane
        if op_type in [ElementWiseOpType.ADD, ElementWiseOpType.SUB]:
            # Simple arithmetic: minimal LUT usage, no DSPs
            luts_per_lane = 10
            dsps_per_lane = 0
        elif op_type in [ElementWiseOpType.MUL]:
            # Multiplication: uses DSPs
            luts_per_lane = 20
            dsps_per_lane = 1
        elif op_type in [ElementWiseOpType.DIV]:
            # Division: expensive in LUTs
            luts_per_lane = 200
            dsps_per_lane = 0
        elif op_type in [ElementWiseOpType.MAX, ElementWiseOpType.MIN]:
            # Comparison: moderate LUT usage
            luts_per_lane = 30
            dsps_per_lane = 0
        elif op_type == ElementWiseOpType.RELU:
            # ReLU: simple comparison and mux
            luts_per_lane = 15
            dsps_per_lane = 0
        elif op_type in [ElementWiseOpType.SIGMOID, ElementWiseOpType.TANH]:
            # Complex functions: lookup tables or polynomial approximation
            luts_per_lane = 100
            dsps_per_lane = 2  # For polynomial evaluation
        elif op_type in [ElementWiseOpType.ABS]:
            # Absolute value: simple logic
            luts_per_lane = 8
            dsps_per_lane = 0
        elif op_type in [ElementWiseOpType.SQRT, ElementWiseOpType.EXP, ElementWiseOpType.LOG]:
            # Complex mathematical functions
            luts_per_lane = 150
            dsps_per_lane = 3
        else:
            # Default conservative estimate
            luts_per_lane = 50
            dsps_per_lane = 1

        return ResourceEstimate(
            dsps=dsps_per_lane * simd_factor,
            brams=0,  # Will be added separately if needed
            luts=luts_per_lane * simd_factor,
            ffs=luts_per_lane * simd_factor  # Roughly equal for pipelined designs
        )

    def _needs_internal_storage(self) -> bool:
        """Check if operation needs internal storage."""
        # Complex operations might need temporary storage
        complex_ops = {ElementWiseOpType.SIGMOID, ElementWiseOpType.TANH,
                      ElementWiseOpType.SQRT, ElementWiseOpType.EXP, ElementWiseOpType.LOG}
        return self.params.operation_type in complex_ops

    def _needs_buffering(self) -> bool:
        """Check if operation needs input buffering."""
        # Binary operations with different input patterns might need buffering
        return self.params.num_inputs > 1

    def _get_dtype_bits(self, dtype: str) -> int:
        """Get bit width for data type."""
        return DTYPE_BITS.get(dtype, 32)

    def __str__(self) -> str:
        """String representation of the operation."""
        return (f"ElementWiseOperation(op_type={self.params.operation_type.value}, "
                f"shape={self.params.tensor_shape}, num_inputs={self.params.num_inputs}, "
                f"input_dtype={self.params.input_dtype}, output_dtype={self.params.output_dtype})")

    def __repr__(self) -> str:
        """Detailed representation of the operation."""
        return self.__str__()
