"""Convolution operation implementation."""

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
from ..core.constraints import Constraint, DivisibilityConstraint
from ..core.constants import (
    VALID_DTYPES, 
    DTYPE_BITS, 
    BRAM_BITS_PER_BLOCK,
    DEFAULT_INPUT_DTYPE,
    DEFAULT_WEIGHT_DTYPE,
    DEFAULT_OUTPUT_DTYPE
)


@dataclass
class ConvolutionParams:
    """Parameters for Convolution operation."""
    # Input tensor: (batch, channels, height, width) or (channels, height, width)
    input_shape: Tuple[int, ...]
    # Kernel: (out_channels, in_channels, kernel_height, kernel_width)
    kernel_shape: Tuple[int, int, int, int]
    stride: Tuple[int, int] = (1, 1)
    padding: Tuple[int, int, int, int] = (0, 0, 0, 0)  # (top, bottom, left, right)
    dilation: Tuple[int, int] = (1, 1)
    groups: int = 1
    input_dtype: str = DEFAULT_INPUT_DTYPE
    weight_dtype: str = DEFAULT_WEIGHT_DTYPE
    output_dtype: str = DEFAULT_OUTPUT_DTYPE
    bias_dtype: Optional[str] = None

    def __post_init__(self):
        """Validate parameters."""
        # Validate input shape
        if len(self.input_shape) not in [3, 4]:
            raise ValueError("Input shape must be 3D (C,H,W) or 4D (N,C,H,W)")
        if not all(dim > 0 for dim in self.input_shape):
            raise ValueError("All input dimensions must be positive")

        # Validate kernel shape
        if len(self.kernel_shape) != 4:
            raise ValueError("Kernel shape must be 4D (C_out, C_in, K_h, K_w)")
        if not all(dim > 0 for dim in self.kernel_shape):
            raise ValueError("All kernel dimensions must be positive")

        # Validate groups first (before channel validation)
        if self.groups <= 0:
            raise ValueError("Groups must be positive")

        # Validate channel compatibility
        if len(self.input_shape) == 3:
            input_channels = self.input_shape[0]
        else:
            input_channels = self.input_shape[1]

        kernel_input_channels = self.kernel_shape[1] * self.groups
        if input_channels != kernel_input_channels:
            raise ValueError(f"Input channels ({input_channels}) must equal kernel input channels ({kernel_input_channels})")

        # Validate stride
        if not all(s > 0 for s in self.stride):
            raise ValueError("Stride values must be positive")

        # Validate dilation
        if not all(d > 0 for d in self.dilation):
            raise ValueError("Dilation values must be positive")

        # Validate data types
        if self.input_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid input dtype: {self.input_dtype}")
        if self.weight_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid weight dtype: {self.weight_dtype}")
        if self.output_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid output dtype: {self.output_dtype}")
        if self.bias_dtype is not None and self.bias_dtype not in VALID_DTYPES:
            raise ValueError(f"Invalid bias dtype: {self.bias_dtype}")


@register_operation("convolution", "computation")
class ConvolutionOperation(ParallelizableOperation):
    """
    Convolution operation for neural networks.

    Performs 2D convolution: Y = Conv(X, W) + B where:
    - X is input tensor of shape (C_in, H_in, W_in) or (N, C_in, H_in, W_in)
    - W is weight tensor of shape (C_out, C_in, K_h, K_w)
    - B is optional bias tensor of shape (C_out,)
    - Y is output tensor of shape (C_out, H_out, W_out) or (N, C_out, H_out, W_out)

    This operation can be parallelized along:
    - PE (Processing Elements): Parallel computation of output channels
    - SIMD: Parallel MAC operations within each convolution window
    - MMV (Matrix-Matrix-Vector): For im2col-based implementations
    """

    def __init__(self, name: str, params: ConvolutionParams):
        """Initialize Convolution operation.

        Args:
            name: Unique name for this operation instance
            params: Operation parameters including shapes, strides, padding, etc.
        """
        super().__init__(name)
        self.params = params
        self._validate_params()
        self._compute_output_shape()

    def _validate_params(self) -> None:
        """Validate operation parameters."""
        # Validate input shape
        if len(self.params.input_shape) not in [3, 4]:
            raise ValueError("Input shape must be 3D (C,H,W) or 4D (N,C,H,W)")

        # Validate kernel shape
        if len(self.params.kernel_shape) != 4:
            raise ValueError("Kernel shape must be 4D (C_out, C_in, K_h, K_w)")

        # Check channel consistency
        input_channels = self.params.input_shape[-3]  # C dimension
        kernel_in_channels = self.params.kernel_shape[1]

        if input_channels != kernel_in_channels * self.params.groups:
            raise ValueError(f"Input channels ({input_channels}) must equal "
                           f"kernel input channels ({kernel_in_channels}) * groups ({self.params.groups})")

        # Validate other parameters
        if any(s <= 0 for s in self.params.stride):
            raise ValueError("Stride values must be positive")

        if any(d <= 0 for d in self.params.dilation):
            raise ValueError("Dilation values must be positive")

        if self.params.groups <= 0:
            raise ValueError("Groups must be positive")

        # Validate data types
        valid_dtypes = ["int8", "int16", "int32", "uint8", "uint16", "uint32", "float32"]
        if self.params.input_dtype not in valid_dtypes:
            raise ValueError(f"Invalid input dtype: {self.params.input_dtype}")
        if self.params.weight_dtype not in valid_dtypes:
            raise ValueError(f"Invalid weight dtype: {self.params.weight_dtype}")
        if self.params.output_dtype not in valid_dtypes:
            raise ValueError(f"Invalid output dtype: {self.params.output_dtype}")

    def _compute_output_shape(self) -> None:
        """Compute output tensor shape."""
        if len(self.params.input_shape) == 3:
            _, h_in, w_in = self.params.input_shape
        else:
            _, _, h_in, w_in = self.params.input_shape

        c_out, _, k_h, k_w = self.params.kernel_shape
        stride_h, stride_w = self.params.stride
        pad_top, pad_bottom, pad_left, pad_right = self.params.padding
        dil_h, dil_w = self.params.dilation

        # Calculate output dimensions
        h_out = ((h_in + pad_top + pad_bottom - dil_h * (k_h - 1) - 1) // stride_h) + 1
        w_out = ((w_in + pad_left + pad_right - dil_w * (k_w - 1) - 1) // stride_w) + 1

        if len(self.params.input_shape) == 3:
            self.output_shape = (c_out, h_out, w_out)
        else:
            batch_size = self.params.input_shape[0]
            self.output_shape = (batch_size, c_out, h_out, w_out)

    def get_input_tensors(self) -> Dict[str, TensorSpec]:
        """Get input tensor specifications."""
        # Create dimension names
        if len(self.params.input_shape) == 3:
            input_dims = ["channels", "height", "width"]
        else:
            input_dims = ["batch", "channels", "height", "width"]

        tensors = {
            "input": TensorSpec(
                name="input",
                shape=self.params.input_shape,
                dtype=self.params.input_dtype,
                dimension_names=input_dims
            ),
            "weights": TensorSpec(
                name="weights",
                shape=self.params.kernel_shape,
                dtype=self.params.weight_dtype,
                dimension_names=["out_channels", "in_channels", "kernel_height", "kernel_width"]
            )
        }

        if self.params.bias_dtype is not None:
            tensors["bias"] = TensorSpec(
                name="bias",
                shape=(self.params.kernel_shape[0],),  # C_out
                dtype=self.params.bias_dtype,
                dimension_names=["out_channels"]
            )

        return tensors

    def get_output_tensors(self) -> Dict[str, TensorSpec]:
        """Get output tensor specifications."""
        if len(self.output_shape) == 3:
            output_dims = ["channels", "height", "width"]
        else:
            output_dims = ["batch", "channels", "height", "width"]

        return {
            "output": TensorSpec(
                name="output",
                shape=self.output_shape,
                dtype=self.params.output_dtype,
                dimension_names=output_dims
            )
        }

    def get_internal_tensors(self) -> Dict[str, TensorSpec]:
        """Get internal tensor specifications."""
        # Line buffers for sliding window
        _, _, k_h, k_w = self.params.kernel_shape
        if len(self.params.input_shape) == 3:
            c_in, h_in, w_in = self.params.input_shape
        else:
            _, c_in, h_in, w_in = self.params.input_shape

        # Line buffer for streaming implementation
        line_buffer_shape = (c_in, k_h - 1, w_in)

        # Window buffer for current convolution window
        window_buffer_shape = (c_in, k_h, k_w)

        internal_dims_line = ["channels", "buffer_lines", "width"]
        internal_dims_window = ["channels", "kernel_height", "kernel_width"]

        return {
            "line_buffer": TensorSpec(
                name="line_buffer",
                shape=line_buffer_shape,
                dtype=self.params.input_dtype,
                dimension_names=internal_dims_line
            ),
            "window_buffer": TensorSpec(
                name="window_buffer",
                shape=window_buffer_shape,
                dtype=self.params.input_dtype,
                dimension_names=internal_dims_window
            )
        }

    def get_constraints(self) -> List[Constraint]:
        """Get operation constraints."""
        constraints = []

        # PE dimension must divide output channels
        constraints.append(DivisibilityConstraint(
            tensor_name="output",
            tensor_type="output",
            dimension_idx=1 if len(self.output_shape) == 4 else 0  # channels dimension
        ))

        # SIMD dimension must divide input channels per group
        input_channels_per_group = self.params.kernel_shape[1]
        constraints.append(DivisibilityConstraint(
            tensor_name="weights",
            tensor_type="input",
            dimension_idx=1  # in_channels dimension
        ))

        return constraints

    def estimate_resources(self, config: ParallelismConfig) -> ResourceEstimate:
        """Estimate hardware resource usage."""
        # Get parallelism factors
        pe_factor = self._get_parallelism_factor(config, "output", "channels")
        simd_factor = self._get_parallelism_factor(config, "weights", "in_channels")

        # Calculate DSP usage (one DSP per MAC operation)
        dsps_per_pe = simd_factor
        dsps = pe_factor * dsps_per_pe

        # Calculate BRAM usage for weights and line buffers
        weight_brams = self._estimate_weight_brams(pe_factor)
        buffer_brams = self._estimate_buffer_brams()
        brams = weight_brams + buffer_brams

        # Calculate LUT and FF usage
        luts, ffs = self._estimate_logic_resources(pe_factor, simd_factor)

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
        # Default to minimal parallelism
        if len(self.params.input_shape) == 3:
            input_factors = {"channels": 1, "height": 1, "width": 1}
            output_factors = {"channels": 1, "height": 1, "width": 1}
        else:
            input_factors = {"batch": 1, "channels": 1, "height": 1, "width": 1}
            output_factors = {"batch": 1, "channels": 1, "height": 1, "width": 1}

        weight_factors = {"out_channels": 1, "in_channels": 1, "kernel_height": 1, "kernel_width": 1}

        input_tensors = {
            "input": TensorParallelism(
                dimension_factors=input_factors,
                strategy=ParallelismStrategy.SPATIAL,
                memory_strategy=MemoryStrategy.STREAMING
            ),
            "weights": TensorParallelism(
                dimension_factors=weight_factors,
                strategy=ParallelismStrategy.SPATIAL,
                memory_strategy=MemoryStrategy.BLOCK_RAM
            )
        }

        if self.params.bias_dtype is not None:
            input_tensors["bias"] = TensorParallelism(
                dimension_factors={"out_channels": 1},
                strategy=ParallelismStrategy.SPATIAL,
                memory_strategy=MemoryStrategy.BLOCK_RAM
            )

        return ParallelismConfig(
            input_tensors=input_tensors,
            output_tensors={
                "output": TensorParallelism(
                    dimension_factors=output_factors,
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                )
            },
            internal_tensors={
                "line_buffer": TensorParallelism(
                    dimension_factors={"channels": 1, "buffer_lines": 1, "width": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.BLOCK_RAM
                ),
                "window_buffer": TensorParallelism(
                    dimension_factors={"channels": 1, "kernel_height": 1, "kernel_width": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.REGISTERS
                )
            }
        )

    def get_parallelism_bounds(self) -> Dict[str, Tuple[int, int]]:
        """Get bounds for parallelism factors."""
        c_out = self.params.kernel_shape[0]
        c_in = self.params.kernel_shape[1]

        return {
            "PE": (1, c_out),
            "SIMD": (1, c_in)
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

    def _estimate_weight_brams(self, pe_factor: int) -> int:
        """Estimate BRAM usage for weight storage."""
        c_out, c_in, k_h, k_w = self.params.kernel_shape
        weight_bits_per_pe = (c_in * k_h * k_w * self._get_dtype_bits(self.params.weight_dtype))
        bram_per_pe = math.ceil(weight_bits_per_pe / BRAM_BITS_PER_BLOCK)  # 18Kb per BRAM
        return pe_factor * bram_per_pe

    def _estimate_buffer_brams(self) -> int:
        """Estimate BRAM usage for line buffers."""
        if len(self.params.input_shape) == 3:
            c_in, h_in, w_in = self.params.input_shape
        else:
            _, c_in, h_in, w_in = self.params.input_shape

        k_h = self.params.kernel_shape[2]

        # Line buffer storage
        line_buffer_bits = c_in * (k_h - 1) * w_in * self._get_dtype_bits(self.params.input_dtype)
        return math.ceil(line_buffer_bits / BRAM_BITS_PER_BLOCK)

    def _estimate_logic_resources(self, pe_factor: int, simd_factor: int) -> Tuple[int, int]:
        """Estimate LUT and FF usage."""
        input_bits = self._get_dtype_bits(self.params.input_dtype)
        weight_bits = self._get_dtype_bits(self.params.weight_dtype)
        output_bits = self._get_dtype_bits(self.params.output_dtype)

        # Control logic, address generation, and data path
        luts_per_pe = (input_bits + weight_bits + output_bits) * 15
        luts = pe_factor * luts_per_pe

        # Pipeline registers and control
        ffs = luts

        return luts, ffs

    def _get_dtype_bits(self, dtype: str) -> int:
        """Get bit width for data type."""
        return DTYPE_BITS.get(dtype, 32)

    def __str__(self) -> str:
        """String representation of the operation."""
        return (f"ConvolutionOperation(input_shape={self.params.input_shape}, "
                f"kernel_shape={self.params.kernel_shape}, stride={self.params.stride}, "
                f"padding={self.params.padding}, groups={self.params.groups})")

    def __repr__(self) -> str:
        """Detailed representation of the operation."""
        return self.__str__()
