"""Matrix-Vector multiplication operation implementation."""

from typing import Dict, List, Optional, Tuple
import math
from dataclasses import dataclass

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
from ..core.constraints import Constraint, DivisibilityConstraint, ResourceConstraint
from ..core.constants import (
    VALID_DTYPES, 
    DTYPE_BITS, 
    BRAM_BITS_PER_BLOCK,
    DEFAULT_INPUT_DTYPE,
    DEFAULT_WEIGHT_DTYPE,
    DEFAULT_OUTPUT_DTYPE
)


@dataclass
class MatrixVectorParams:
    """Parameters for Matrix-Vector operation."""
    matrix_height: int
    matrix_width: int
    input_dtype: str = DEFAULT_INPUT_DTYPE
    weight_dtype: str = DEFAULT_WEIGHT_DTYPE
    output_dtype: str = DEFAULT_OUTPUT_DTYPE
    bias_dtype: Optional[str] = None

    def __post_init__(self):
        """Validate parameters."""
        if self.matrix_height <= 0 or self.matrix_width <= 0:
            raise ValueError("Matrix dimensions must be positive")

        # Validate data types
        if self.input_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid input dtype: {self.input_dtype}")
        if self.weight_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid weight dtype: {self.weight_dtype}")
        if self.output_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid output dtype: {self.output_dtype}")
        if self.bias_dtype is not None and self.bias_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid bias dtype: {self.bias_dtype}")


@register_operation("matrix_vector", "computation")
class MatrixVectorOperation(ParallelizableOperation):
    """
    Matrix-Vector multiplication operation (equivalent to MVAU).

    Performs Y = M @ X + B where:
    - M is a matrix of shape (output_features, input_features)
    - X is an input vector of shape (input_features,)
    - B is an optional bias vector of shape (output_features,)
    - Y is the output vector of shape (output_features,)

    This operation can be parallelized along:
    - PE (Processing Elements): Parallel computation of output elements
    - SIMD (Single Instruction Multiple Data): Parallel MAC operations per PE
    """

    def __init__(self, name: str, params: MatrixVectorParams):
        """Initialize Matrix-Vector operation.

        Args:
            name: Unique name for this operation instance
            params: Operation parameters including matrix dimensions and data types
        """
        super().__init__(name)
        self.params = params
        # Validation is handled in MatrixVectorParams.__post_init__

    def get_input_tensors(self) -> Dict[str, TensorSpec]:
        """Get input tensor specifications."""
        tensors = {
            "input": TensorSpec(
                name="input",
                shape=(self.params.matrix_width,),
                dtype=self.params.input_dtype,
                dimension_names=["input_features"]
            ),
            "weights": TensorSpec(
                name="weights",
                shape=(self.params.matrix_height, self.params.matrix_width),
                dtype=self.params.weight_dtype,
                dimension_names=["output_features", "input_features"]
            )
        }

        if self.params.bias_dtype is not None:
            tensors["bias"] = TensorSpec(
                name="bias",
                shape=(self.params.matrix_height,),
                dtype=self.params.bias_dtype,
                dimension_names=["output_features"]
            )

        return tensors

    def get_output_tensors(self) -> Dict[str, TensorSpec]:
        """Get output tensor specifications."""
        return {
            "output": TensorSpec(
                name="output",
                shape=(self.params.matrix_height,),
                dtype=self.params.output_dtype,
                dimension_names=["output_features"]
            )
        }

    def get_internal_tensors(self) -> Dict[str, TensorSpec]:
        """Get internal tensor specifications (accumulator buffers)."""
        return {
            "accumulator": TensorSpec(
                name="accumulator",
                shape=(self.params.matrix_height,),
                dtype=self.params.output_dtype,
                dimension_names=["output_features"]
            )
        }

    def get_constraints(self) -> List[Constraint]:
        """Get operation constraints."""
        constraints = []

        # PE dimension must divide output features
        constraints.append(DivisibilityConstraint(
            tensor_name="output",
            tensor_type="output",
            dimension_idx=0  # output_features is dimension 0
        ))

        # SIMD dimension must divide input features
        constraints.append(DivisibilityConstraint(
            tensor_name="input",
            tensor_type="input",
            dimension_idx=0  # input_features is dimension 0
        ))

        # Resource constraints will be added by the optimizer

        return constraints

    def estimate_resources(self, config: ParallelismConfig) -> ResourceEstimate:
        """Estimate hardware resource usage."""
        # Get parallelism factors
        pe_factor = self._get_parallelism_factor(config, "output", "output_features")
        simd_factor = self._get_parallelism_factor(config, "input", "input_features")

        # Calculate DSP usage (one DSP per MAC operation)
        dsps = pe_factor * simd_factor

        # Calculate BRAM usage for weight storage
        # Assume 18Kb BRAM blocks, each storing weights for one PE
        weight_bits_per_pe = (self.params.matrix_width *
                             self._get_dtype_bits(self.params.weight_dtype))
        bram_per_pe = math.ceil(weight_bits_per_pe / BRAM_BITS_PER_BLOCK)  # 18Kb per BRAM
        brams = pe_factor * bram_per_pe

        # Calculate LUT usage (estimated based on data width and control logic)
        input_bits = self._get_dtype_bits(self.params.input_dtype)
        weight_bits = self._get_dtype_bits(self.params.weight_dtype)
        output_bits = self._get_dtype_bits(self.params.output_dtype)

        # Rough estimation: LUTs for address generation, muxing, and accumulation
        luts_per_pe = (input_bits + weight_bits + output_bits) * 10
        luts = pe_factor * luts_per_pe

        # Calculate FF usage (roughly equal to LUTs for pipelined designs)
        ffs = luts

        return ResourceEstimate(
            dsps=dsps,
            brams=brams,
            luts=luts,
            ffs=ffs
        )

    def validate_config(self, config: ParallelismConfig) -> bool:
        """Validate parallelism configuration."""
        try:
            # Check that all required tensors have parallelism specified
            required_tensors = set(self.get_input_tensors().keys()) | set(self.get_output_tensors().keys())
            config_tensors = set(config.input_tensors.keys()) | set(config.output_tensors.keys())

            if not required_tensors.issubset(config_tensors):
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
        # Default to minimal parallelism (PE=1, SIMD=1)
        return ParallelismConfig(
            input_tensors={
                "input": TensorParallelism(
                    dimension_factors={"input_features": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                ),
                "weights": TensorParallelism(
                    dimension_factors={"output_features": 1, "input_features": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.BLOCK_RAM
                )
            },
            output_tensors={
                "output": TensorParallelism(
                    dimension_factors={"output_features": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                )
            },
            internal_tensors={
                "accumulator": TensorParallelism(
                    dimension_factors={"output_features": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.REGISTERS
                )
            }
        )

    def get_parallelism_bounds(self) -> Dict[str, Tuple[int, int]]:
        """Get bounds for parallelism factors."""
        return {
            "PE": (1, self.params.matrix_height),
            "SIMD": (1, self.params.matrix_width)
        }

    def _get_parallelism_factor(self, config: ParallelismConfig,
                               tensor_name: str, dimension_name: str) -> int:
        """Extract parallelism factor from configuration."""
        if tensor_name in config.input_tensors:
            return config.input_tensors[tensor_name].dimension_factors.get(dimension_name, 1)
        elif tensor_name in config.output_tensors:
            return config.output_tensors[tensor_name].dimension_factors.get(dimension_name, 1)
        elif tensor_name in config.internal_tensors:
            return config.internal_tensors[tensor_name].dimension_factors.get(dimension_name, 1)
        return 1

    def _get_dtype_bits(self, dtype: str) -> int:
        """Get bit width for data type."""
        return DTYPE_BITS.get(dtype, 32)

    def __str__(self) -> str:
        """String representation of the operation."""
        return (f"MatrixVectorOperation(matrix_shape=({self.params.matrix_height}, "
                f"{self.params.matrix_width}), input_dtype={self.params.input_dtype}, "
                f"weight_dtype={self.params.weight_dtype}, output_dtype={self.params.output_dtype})")

    def __repr__(self) -> str:
        """Detailed representation of the operation."""
        return self.__str__()
