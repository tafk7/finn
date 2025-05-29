"""
Core base classes and data structures for the extensible parallelism system.

This module defines the fundamental abstractions that all operations must implement
to participate in the tensor-centric parallelism framework.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum


class ParallelismStrategy(Enum):
    """Strategy for parallelizing a tensor dimension."""
    SPATIAL = "spatial"      # Parallel compute units
    TEMPORAL = "temporal"    # Time-multiplexed
    PIPELINE = "pipeline"    # Pipelined execution


class MemoryStrategy(Enum):
    """Strategy for memory organization."""
    STREAMING = "streaming"    # Continuous data flow
    BUFFERED = "buffered"     # Local buffering
    CACHED = "cached"         # Cached access patterns
    BLOCK_RAM = "block_ram"   # FPGA block RAM storage
    REGISTERS = "registers"   # Register-based storage


@dataclass
class TensorSpec:
    """Specification for a tensor in the operation."""
    name: str                              # Tensor identifier
    shape: List[int]                       # Tensor dimensions
    dtype: str                             # Data type (e.g., "int8", "float32")
    dimension_names: List[str]             # Semantic dimension names
    is_weight: bool = False                # Whether this is a weight tensor
    is_activation: bool = True             # Whether this is an activation tensor
    reduction_dims: List[int] = field(default_factory=list)  # Dimensions with reductions
    broadcast_dims: List[int] = field(default_factory=list)  # Broadcast dimensions

    def __post_init__(self):
        """Validate tensor specification."""
        if len(self.shape) != len(self.dimension_names):
            raise ValueError(f"Shape length {len(self.shape)} must match dimension_names length {len(self.dimension_names)}")

        if any(dim >= len(self.shape) for dim in self.reduction_dims):
            raise ValueError("Reduction dimensions must be valid indices")

        if any(dim >= len(self.shape) for dim in self.broadcast_dims):
            raise ValueError("Broadcast dimensions must be valid indices")


@dataclass
class TensorParallelism:
    """Parallelism configuration for a tensor."""
    dimension_factors: Dict[str, int] = field(default_factory=dict)  # Per-dimension parallelism factors
    strategy: ParallelismStrategy = ParallelismStrategy.SPATIAL      # How to parallelize
    memory_strategy: MemoryStrategy = MemoryStrategy.STREAMING       # Memory access pattern
    constraints: List[str] = field(default_factory=list)            # Constraints on this parallelism


    def __post_init__(self):
        """Validate parallelism configuration."""
        for dim_name, factor in self.dimension_factors.items():
            if factor <= 0:
                raise ValueError(f"Parallelism factor for dimension '{dim_name}' must be positive")

    def get_factor(self, dimension_name: str) -> int:
        """Get parallelism factor for a specific dimension."""
        return self.dimension_factors.get(dimension_name, 1)

    # Backward compatibility properties for old constraint API
    @property
    def dimension_idx(self) -> int:
        """Get the first dimension index (for backward compatibility)."""
        if self.dimension_factors:
            # Extract the dimension index from the first dimension name
            first_dim = list(self.dimension_factors.keys())[0]
            if first_dim.startswith('dim_'):
                return int(first_dim.split('_')[1])
        return 0

    @property
    def factor(self) -> int:
        """Get the first dimension's factor (for backward compatibility)."""
        if self.dimension_factors:
            return list(self.dimension_factors.values())[0]
        return 1


@dataclass
class ParallelismConfig:
    """Complete parallelism configuration for an operation."""
    # Per-tensor parallelism configurations (old API names for backward compatibility)
    input_tensors: Dict[str, TensorParallelism] = field(default_factory=dict)
    output_tensors: Dict[str, TensorParallelism] = field(default_factory=dict)
    internal_tensors: Dict[str, TensorParallelism] = field(default_factory=dict)

    # New API names
    input_parallelism: Dict[str, List[TensorParallelism]] = field(default_factory=dict)
    output_parallelism: Dict[str, List[TensorParallelism]] = field(default_factory=dict)
    internal_parallelism: Dict[str, List[TensorParallelism]] = field(default_factory=dict)

    # Global configuration
    memory_strategy: MemoryStrategy = MemoryStrategy.STREAMING
    pipeline_depth: int = 1
    clock_frequency: Optional[float] = None  # MHz

    # Resource constraints
    max_dsps: Optional[int] = None
    max_brams: Optional[int] = None
    max_luts: Optional[int] = None

    def __post_init__(self):
        """Convert between old and new API formats."""
        # Convert old API to new API if old API is used
        if self.input_tensors and not self.input_parallelism:
            self.input_parallelism = {k: [v] for k, v in self.input_tensors.items()}
        if self.output_tensors and not self.output_parallelism:
            self.output_parallelism = {k: [v] for k, v in self.output_tensors.items()}
        if self.internal_tensors and not self.internal_parallelism:
            self.internal_parallelism = {k: [v] for k, v in self.internal_tensors.items()}

    def get_tensor_parallelism(self, tensor_name: str, tensor_type: str = "input") -> List[TensorParallelism]:
        """Get parallelism configuration for a specific tensor."""
        if tensor_type == "input":
            return self.input_parallelism.get(tensor_name, [])
        elif tensor_type == "output":
            return self.output_parallelism.get(tensor_name, [])
        elif tensor_type == "internal":
            return self.internal_parallelism.get(tensor_name, [])
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")

    def set_tensor_parallelism(self, tensor_name: str, parallelism: List[TensorParallelism],
                              tensor_type: str = "input"):
        """Set parallelism configuration for a specific tensor."""
        if tensor_type == "input":
            self.input_parallelism[tensor_name] = parallelism
        elif tensor_type == "output":
            self.output_parallelism[tensor_name] = parallelism
        elif tensor_type == "internal":
            self.internal_parallelism[tensor_name] = parallelism
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")


@dataclass
class ResourceEstimate:
    """Estimated resource usage for a parallelism configuration."""
    dsps: int = 0                         # Number of DSP blocks
    brams: int = 0                        # Number of BRAM blocks
    luts: int = 0                         # Number of LUTs
    ffs: int = 0                          # Number of flip-flops
    memory_bandwidth_bits: int = 0        # Memory bandwidth required (bits/cycle)
    compute_cycles: int = 0               # Compute cycles required
    latency_cycles: int = 0               # Total latency in cycles

    def __add__(self, other: 'ResourceEstimate') -> 'ResourceEstimate':
        """Add two resource estimates."""
        return ResourceEstimate(
            dsps=self.dsps + other.dsps,
            brams=self.brams + other.brams,
            luts=self.luts + other.luts,
            ffs=self.ffs + other.ffs,
            memory_bandwidth_bits=max(self.memory_bandwidth_bits, other.memory_bandwidth_bits),
            compute_cycles=max(self.compute_cycles, other.compute_cycles),
            latency_cycles=self.latency_cycles + other.latency_cycles
        )

    def efficiency_score(self, total_available: 'ResourceEstimate') -> float:
        """Calculate resource efficiency score (0-1)."""
        if total_available.dsps == 0:
            return 0.0

        dsp_util = self.dsps / total_available.dsps if total_available.dsps > 0 else 0
        bram_util = self.brams / total_available.brams if total_available.brams > 0 else 0
        lut_util = self.luts / total_available.luts if total_available.luts > 0 else 0

        return max(dsp_util, bram_util, lut_util)  # Bottleneck resource


class ParallelizableOperation(ABC):
    """
    Base class for all operations that can be parallelized.

    This abstract base class defines the interface that all neural network operations
    must implement to participate in the tensor-centric parallelism framework.
    """

    def __init__(self, name: str):
        """Initialize the operation with a unique name."""
        self.name = name
        self._input_tensors: Dict[str, TensorSpec] = {}
        self._output_tensors: Dict[str, TensorSpec] = {}
        self._internal_tensors: Dict[str, TensorSpec] = {}

    @abstractmethod
    def get_input_tensors(self) -> Dict[str, TensorSpec]:
        """Return specifications for all input tensors."""
        pass

    @abstractmethod
    def get_output_tensors(self) -> Dict[str, TensorSpec]:
        """Return specifications for all output tensors."""
        pass

    @abstractmethod
    def get_internal_tensors(self) -> Dict[str, TensorSpec]:
        """Return specifications for internal computation tensors."""
        pass

    @abstractmethod
    def get_constraints(self) -> List['Constraint']:
        """Return list of constraints for this operation."""
        pass

    @abstractmethod
    def estimate_resources(self, config: ParallelismConfig) -> ResourceEstimate:
        """Estimate hardware resource usage for given parallelism configuration."""
        pass

    @abstractmethod
    def validate_config(self, config: ParallelismConfig) -> Tuple[bool, List[str]]:
        """
        Validate a parallelism configuration.

        Returns:
            (is_valid, error_messages)
        """
        pass

    def get_all_tensors(self) -> Dict[str, TensorSpec]:
        """Get all tensors (input, output, internal) with type prefixes."""
        all_tensors = {}

        for name, spec in self.get_input_tensors().items():
            all_tensors[f"input:{name}"] = spec

        for name, spec in self.get_output_tensors().items():
            all_tensors[f"output:{name}"] = spec

        for name, spec in self.get_internal_tensors().items():
            all_tensors[f"internal:{name}"] = spec

        return all_tensors

    def get_compute_complexity(self) -> int:
        """Get computational complexity (number of operations)."""
        # Default implementation - subclasses should override
        input_tensors = self.get_input_tensors()
        if not input_tensors:
            return 0

        # Simple heuristic: product of all input tensor sizes
        complexity = 1
        for tensor_spec in input_tensors.values():
            complexity *= max(1, sum(tensor_spec.shape) if tensor_spec.shape else 1)

        return complexity

    def suggest_default_parallelism(self) -> ParallelismConfig:
        """Suggest a reasonable default parallelism configuration."""
        config = ParallelismConfig()

        # Default: no parallelism (factor 1 for all dimensions)
        for tensor_name, tensor_spec in self.get_input_tensors().items():
            dimension_factors = {}
            for dim_idx, dim_name in enumerate(tensor_spec.dimension_names):
                if tensor_spec.shape[dim_idx] > 1:  # Only parallelize non-trivial dimensions
                    dimension_factors[dim_name] = 1  # Start with no parallelism

            if dimension_factors:  # Only create parallelism if there are parallelizable dimensions
                parallelism = TensorParallelism(
                    dimension_factors=dimension_factors,
                    strategy=ParallelismStrategy.SPATIAL
                )
                config.input_tensors[tensor_name] = parallelism

        for tensor_name, tensor_spec in self.get_output_tensors().items():
            dimension_factors = {}
            for dim_idx, dim_name in enumerate(tensor_spec.dimension_names):
                if tensor_spec.shape[dim_idx] > 1:
                    dimension_factors[dim_name] = 1

            if dimension_factors:
                parallelism = TensorParallelism(
                    dimension_factors=dimension_factors,
                    strategy=ParallelismStrategy.SPATIAL
                )
                config.output_tensors[tensor_name] = parallelism

        return config

    def __str__(self) -> str:
        """String representation of the operation."""
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        """Detailed string representation."""
        input_shapes = {name: spec.shape for name, spec in self.get_input_tensors().items()}
        output_shapes = {name: spec.shape for name, spec in self.get_output_tensors().items()}

        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"inputs={input_shapes}, outputs={output_shapes})")
