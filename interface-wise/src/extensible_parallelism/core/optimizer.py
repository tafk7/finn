"""
Basic parallelism optimizer for the extensible parallelism system.

Provides basic optimization strategies to find efficient parallelism configurations
for operations, including grid search, heuristics, and constraint-based optimization.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
from enum import Enum
import math
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from .base import (
    ParallelizableOperation,
    ParallelismConfig,
    TensorParallelism,
    ResourceEstimate,
    ParallelismStrategy,
    MemoryStrategy
)


class OptimizationStrategy(Enum):
    """Optimization strategies for parallelism configuration search."""
    GRID_SEARCH = "grid_search"           # Exhaustive search over all possibilities
    HEURISTIC = "heuristic"               # Fast heuristic-based search
    GENETIC = "genetic"                   # Genetic algorithm optimization
    SIMULATED_ANNEALING = "simulated_annealing"  # Simulated annealing optimization


class OptimizationObjective(Enum):
    """Optimization objectives for parallelism configuration."""
    MINIMIZE_LATENCY = "minimize_latency"
    MINIMIZE_RESOURCES = "minimize_resources"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCED = "balanced"                 # Balance between latency and resources


@dataclass
class OptimizationConstraints:
    """Constraints for parallelism optimization."""
    max_dsps: Optional[int] = None        # Maximum DSP blocks
    max_brams: Optional[int] = None       # Maximum BRAM blocks
    max_luts: Optional[int] = None        # Maximum LUTs
    max_ffs: Optional[int] = None         # Maximum flip-flops
    max_latency_cycles: Optional[int] = None  # Maximum acceptable latency
    min_throughput: Optional[float] = None    # Minimum required throughput
    clock_frequency: float = 100.0        # Target clock frequency (MHz)

    def is_satisfied_by(self, resources: ResourceEstimate) -> bool:
        """Check if resource estimate satisfies constraints."""
        if self.max_dsps is not None and resources.dsps > self.max_dsps:
            return False
        if self.max_brams is not None and resources.brams > self.max_brams:
            return False
        if self.max_luts is not None and resources.luts > self.max_luts:
            return False
        if self.max_ffs is not None and resources.ffs > self.max_ffs:
            return False
        if self.max_latency_cycles is not None and resources.latency_cycles > self.max_latency_cycles:
            return False
        return True


@dataclass
class OptimizationResult:
    """Result of parallelism optimization."""
    best_config: ParallelismConfig       # Best configuration found
    best_score: float                    # Score of best configuration
    resource_estimate: ResourceEstimate  # Resource usage of best config
    search_time: float                   # Time spent searching (seconds)
    configurations_evaluated: int        # Number of configurations tested
    optimization_log: List[str]         # Log of optimization process

    def __str__(self) -> str:
        """String representation of optimization result."""
        return (f"OptimizationResult(score={self.best_score:.3f}, "
                f"dsps={self.resource_estimate.dsps}, "
                f"brams={self.resource_estimate.brams}, "
                f"luts={self.resource_estimate.luts}, "
                f"evaluated={self.configurations_evaluated}, "
                f"time={self.search_time:.3f}s)")


class ParallelismOptimizer(ABC):
    """Abstract base class for parallelism optimizers."""

    def __init__(self, objective: OptimizationObjective = OptimizationObjective.BALANCED,
                 constraints: Optional[OptimizationConstraints] = None):
        """
        Initialize optimizer.

        Args:
            objective: Optimization objective
            constraints: Resource and performance constraints
        """
        self.objective = objective
        self.constraints = constraints or OptimizationConstraints()
        self.optimization_log = []

    @abstractmethod
    def optimize(self, operation: ParallelizableOperation,
                initial_config: Optional[ParallelismConfig] = None) -> OptimizationResult:
        """
        Optimize parallelism configuration for an operation.

        Args:
            operation: Operation to optimize
            initial_config: Starting configuration (optional)

        Returns:
            Optimization result with best configuration found
        """
        pass

    def evaluate_config(self, operation: ParallelizableOperation,
                       config: ParallelismConfig) -> float:
        """
        Evaluate a parallelism configuration and return a score.

        Higher scores are better. Returns -1 if configuration is invalid.

        Args:
            operation: Operation being optimized
            config: Configuration to evaluate

        Returns:
            Score for the configuration (-1 if invalid)
        """
        # Check if configuration is valid
        if not operation.validate_config(config):
            return -1.0

        # Get resource estimate
        try:
            resources = operation.estimate_resources(config)
        except Exception:
            return -1.0

        # Check constraints
        if not self.constraints.is_satisfied_by(resources):
            return -1.0

        # Calculate score based on objective
        return self._calculate_score(resources)

    def _calculate_score(self, resources: ResourceEstimate) -> float:
        """Calculate score based on optimization objective."""
        if self.objective == OptimizationObjective.MINIMIZE_LATENCY:
            # Lower latency is better, but avoid division by zero
            if resources.latency_cycles <= 0:
                return 1000.0
            return 1000.0 / resources.latency_cycles

        elif self.objective == OptimizationObjective.MINIMIZE_RESOURCES:
            # Lower total resource usage is better
            total_resources = resources.dsps + resources.brams + resources.luts + resources.ffs
            if total_resources <= 0:
                return 1000.0
            return 1000.0 / total_resources

        elif self.objective == OptimizationObjective.MAXIMIZE_THROUGHPUT:
            # Higher throughput is better
            if resources.compute_cycles <= 0:
                return 1000.0
            throughput = self.constraints.clock_frequency / resources.compute_cycles
            return throughput

        else:  # BALANCED
            # Balance between latency and resources
            if resources.latency_cycles <= 0:
                latency_score = 1000.0
            else:
                latency_score = 100.0 / resources.latency_cycles

            total_resources = resources.dsps + resources.brams + resources.luts + resources.ffs
            if total_resources <= 0:
                resource_score = 1000.0
            else:
                resource_score = 100.0 / total_resources

            # Weighted combination
            return 0.6 * latency_score + 0.4 * resource_score

    def _log(self, message: str):
        """Add message to optimization log."""
        self.optimization_log.append(message)


class GridSearchOptimizer(ParallelismOptimizer):
    """
    Grid search optimizer that exhaustively searches parallelism configurations.

    This optimizer tries all possible combinations of parallelism factors
    within specified bounds. It's thorough but can be slow for large search spaces.
    """

    def __init__(self, objective: OptimizationObjective = OptimizationObjective.BALANCED,
                 constraints: Optional[OptimizationConstraints] = None,
                 max_parallelism_factor: int = 16,
                 max_evaluations: int = 1000):
        """
        Initialize grid search optimizer.

        Args:
            objective: Optimization objective
            constraints: Resource constraints
            max_parallelism_factor: Maximum parallelism factor to consider
            max_evaluations: Maximum number of configurations to evaluate
        """
        super().__init__(objective, constraints)
        self.max_parallelism_factor = max_parallelism_factor
        self.max_evaluations = max_evaluations

    def optimize(self, operation: ParallelizableOperation,
                initial_config: Optional[ParallelismConfig] = None) -> OptimizationResult:
        """Optimize using exhaustive grid search."""
        start_time = time.time()
        self.optimization_log = []

        self._log(f"Starting grid search optimization with max factor {self.max_parallelism_factor}")

        # Get parallelism bounds from the operation
        bounds = operation.get_parallelism_bounds()
        self._log(f"Parallelism bounds: {bounds}")

        # Generate all possible configurations (limited by max_evaluations)
        configs = list(itertools.islice(self._generate_configurations(operation, bounds), self.max_evaluations))
        self._log(f"Generated {len(configs)} configurations to evaluate (limited to {self.max_evaluations})")

        best_config = None
        best_score = -1.0
        best_resources = None
        evaluated_count = 0

        # Evaluate each configuration
        for config in configs:
            score = self.evaluate_config(operation, config)
            evaluated_count += 1

            if score > best_score:
                best_score = score
                best_config = config
                best_resources = operation.estimate_resources(config)
                self._log(f"New best config found: score={score:.3f}, "
                         f"dsps={best_resources.dsps}, brams={best_resources.brams}, "
                         f"luts={best_resources.luts}")

            # Early termination if we found a very good solution
            if best_score > 900.0:  # Very high score
                self._log(f"Early termination: found excellent solution with score {best_score:.3f}")
                break

        # Fallback to default config if no valid config found
        if best_config is None:
            self._log("No valid configuration found, trying default")
            default_config = operation.get_default_parallelism_config()
            default_score = self.evaluate_config(operation, default_config)

            if default_score > 0:  # Default config is valid
                best_config = default_config
                best_score = default_score
                best_resources = operation.estimate_resources(default_config)
            else:
                # Even default config fails constraints
                raise ValueError("No feasible configuration found that satisfies the given constraints")

        search_time = time.time() - start_time
        self._log(f"Grid search completed in {search_time:.3f} seconds")

        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            resource_estimate=best_resources,
            search_time=search_time,
            configurations_evaluated=evaluated_count,
            optimization_log=self.optimization_log.copy()
        )

    def _generate_configurations(self, operation: ParallelizableOperation,
                               bounds: Dict[str, tuple]) -> Iterator[ParallelismConfig]:
        """Generate all possible parallelism configurations."""
        # Get tensor specifications
        input_tensors = operation.get_input_tensors()
        output_tensors = operation.get_output_tensors()
        internal_tensors = operation.get_internal_tensors()

        # Create dimension factor ranges for each tensor
        tensor_ranges = {}

        # Input tensors
        for tensor_name, tensor_spec in input_tensors.items():
            tensor_ranges[f"input:{tensor_name}"] = self._get_dimension_ranges(tensor_spec)

        # Output tensors
        for tensor_name, tensor_spec in output_tensors.items():
            tensor_ranges[f"output:{tensor_name}"] = self._get_dimension_ranges(tensor_spec)

        # Internal tensors
        for tensor_name, tensor_spec in internal_tensors.items():
            tensor_ranges[f"internal:{tensor_name}"] = self._get_dimension_ranges(tensor_spec)

        # Generate all combinations
        tensor_names = list(tensor_ranges.keys())
        if not tensor_names:
            # No tensors, return default config
            yield operation.get_default_parallelism_config()
            return

        # Generate all combinations across tensors (lazily to avoid memory issues)
        def generate_tensor_combinations():
            # Get all factor combinations for each tensor
            tensor_factor_combinations = []
            for tensor_name in tensor_names:
                dim_ranges = tensor_ranges[tensor_name]
                if dim_ranges:
                    # Limit combinations per tensor to avoid explosion
                    dim_combinations = list(itertools.product(*dim_ranges.values()))
                    # Take only first few combinations if too many
                    if len(dim_combinations) > 10:
                        dim_combinations = dim_combinations[:10]
                    tensor_factor_combinations.append(dim_combinations)
                else:
                    tensor_factor_combinations.append([{}])

            # Generate combinations with early termination
            count = 0
            for tensor_factors in itertools.product(*tensor_factor_combinations):
                if count >= 100:  # Hard limit to prevent infinite loops
                    break
                count += 1
                yield tensor_factors

        for tensor_factors in generate_tensor_combinations():
            config_dict = {
                "input_tensors": {},
                "output_tensors": {},
                "internal_tensors": {}
            }

            for i, tensor_name in enumerate(tensor_names):
                tensor_type, name = tensor_name.split(":", 1)
                dim_names = tensor_ranges[tensor_name].keys() if tensor_ranges[tensor_name] else []

                if dim_names:
                    dimension_factors = dict(zip(dim_names, tensor_factors[i]))
                else:
                    dimension_factors = {}

                tensor_parallelism = TensorParallelism(
                    dimension_factors=dimension_factors,
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                )

                if tensor_type == "input":
                    config_dict["input_tensors"][name] = tensor_parallelism
                elif tensor_type == "output":
                    config_dict["output_tensors"][name] = tensor_parallelism
                elif tensor_type == "internal":
                    config_dict["internal_tensors"][name] = tensor_parallelism

            yield ParallelismConfig(**config_dict)

    def _get_dimension_ranges(self, tensor_spec) -> Dict[str, List[int]]:
        """Get valid parallelism factor ranges for each dimension of a tensor."""
        ranges = {}

        for i, (dim_name, dim_size) in enumerate(zip(tensor_spec.dimension_names, tensor_spec.shape)):
            # Find a reasonable set of factors, prioritizing powers of 2 and small factors
            factors = [1]  # Always include 1
            max_factor = min(dim_size, self.max_parallelism_factor)

            # Add powers of 2 that are valid factors
            power = 2
            while power <= max_factor:
                if dim_size % power == 0:
                    factors.append(power)
                power *= 2

            # Add a few other small factors for completeness (but limit to avoid explosion)
            for factor in [3, 4, 5, 6, 8, 12, 16]:
                if factor <= max_factor and dim_size % factor == 0 and factor not in factors:
                    factors.append(factor)
                    if len(factors) >= 8:  # Limit to max 8 factors per dimension
                        break

            # Sort factors
            factors.sort()
            ranges[dim_name] = factors

        return ranges


class HeuristicOptimizer(ParallelismOptimizer):
    """
    Heuristic-based optimizer that uses domain knowledge for fast optimization.

    This optimizer applies heuristics based on operation type and characteristics
    to quickly find good parallelism configurations without exhaustive search.
    """

    def optimize(self, operation: ParallelizableOperation,
                initial_config: Optional[ParallelismConfig] = None) -> OptimizationResult:
        """Optimize using heuristics."""
        start_time = time.time()
        self.optimization_log = []

        self._log("Starting heuristic optimization")

        # Start with default or initial config
        if initial_config is not None:
            current_config = initial_config
            self._log("Using provided initial configuration")
        else:
            current_config = operation.get_default_parallelism_config()
            self._log("Using default configuration as starting point")

        best_config = current_config
        best_score = self.evaluate_config(operation, current_config)
        best_resources = operation.estimate_resources(current_config)
        evaluated_count = 1

        self._log(f"Initial score: {best_score:.3f}")

        # Apply heuristics to improve configuration
        improved_configs = self._apply_heuristics(operation, current_config)

        for config in improved_configs:
            score = self.evaluate_config(operation, config)
            evaluated_count += 1

            if score > best_score:
                best_score = score
                best_config = config
                best_resources = operation.estimate_resources(config)
                self._log(f"Improved config found: score={score:.3f}")

        search_time = time.time() - start_time
        self._log(f"Heuristic optimization completed in {search_time:.3f} seconds")

        # Check if we found any valid configuration
        if best_score <= 0:
            raise ValueError("No feasible configuration found that satisfies the given constraints")

        return OptimizationResult(
            best_config=best_config,
            best_score=best_score,
            resource_estimate=best_resources,
            search_time=search_time,
            configurations_evaluated=evaluated_count,
            optimization_log=self.optimization_log.copy()
        )

    def _apply_heuristics(self, operation: ParallelizableOperation,
                         base_config: ParallelismConfig) -> List[ParallelismConfig]:
        """Apply heuristics to generate improved configurations."""
        configs = []

        # Heuristic 1: Increase parallelism in largest dimensions
        configs.extend(self._increase_largest_dimensions(operation, base_config))

        # Heuristic 2: Balance parallelism across dimensions
        configs.extend(self._balance_parallelism(operation, base_config))

        # Heuristic 3: Use operation-specific optimizations
        configs.extend(self._operation_specific_optimizations(operation, base_config))

        return configs

    def _increase_largest_dimensions(self, operation: ParallelizableOperation,
                                   base_config: ParallelismConfig) -> List[ParallelismConfig]:
        """Increase parallelism in the largest tensor dimensions."""
        configs = []

        # Get tensor specifications
        output_tensors = operation.get_output_tensors()

        for tensor_name, tensor_spec in output_tensors.items():
            if tensor_name in base_config.output_tensors:
                current_tp = base_config.output_tensors[tensor_name]

                # Find largest dimension
                largest_dim_idx = 0
                largest_dim_size = tensor_spec.shape[0]

                for i, size in enumerate(tensor_spec.shape):
                    if size > largest_dim_size:
                        largest_dim_size = size
                        largest_dim_idx = i

                if largest_dim_idx < len(tensor_spec.dimension_names):
                    largest_dim_name = tensor_spec.dimension_names[largest_dim_idx]
                    current_factor = current_tp.dimension_factors.get(largest_dim_name, 1)

                    # Try increasing parallelism factor
                    for new_factor in [current_factor * 2, current_factor * 4]:
                        if largest_dim_size % new_factor == 0:
                            new_factors = current_tp.dimension_factors.copy()
                            new_factors[largest_dim_name] = new_factor

                            new_tp = TensorParallelism(
                                dimension_factors=new_factors,
                                strategy=current_tp.strategy,
                                memory_strategy=current_tp.memory_strategy
                            )

                            new_config = ParallelismConfig(
                                input_tensors=base_config.input_tensors.copy(),
                                output_tensors=base_config.output_tensors.copy(),
                                internal_tensors=base_config.internal_tensors.copy()
                            )
                            new_config.output_tensors[tensor_name] = new_tp

                            configs.append(new_config)

        return configs

    def _balance_parallelism(self, operation: ParallelizableOperation,
                           base_config: ParallelismConfig) -> List[ParallelismConfig]:
        """Create configurations with balanced parallelism across dimensions."""
        configs = []

        # For now, return the base config
        # In a full implementation, this would balance factors across dimensions

        return configs

    def _operation_specific_optimizations(self, operation: ParallelizableOperation,
                                        base_config: ParallelismConfig) -> List[ParallelismConfig]:
        """Apply operation-specific optimization heuristics."""
        configs = []

        # Use the operation's class name to determine type
        op_type = operation.__class__.__name__

        if "MatrixVector" in op_type:
            configs.extend(self._optimize_matrix_vector(operation, base_config))
        elif "Convolution" in op_type:
            configs.extend(self._optimize_convolution(operation, base_config))
        elif "ElementWise" in op_type:
            configs.extend(self._optimize_element_wise(operation, base_config))

        return configs

    def _optimize_matrix_vector(self, operation: ParallelizableOperation,
                              base_config: ParallelismConfig) -> List[ParallelismConfig]:
        """Optimize matrix-vector operations."""
        # For matrix-vector, prioritize PE parallelism (output features)
        return []

    def _optimize_convolution(self, operation: ParallelizableOperation,
                            base_config: ParallelismConfig) -> List[ParallelismConfig]:
        """Optimize convolution operations."""
        # For convolution, balance channel and spatial parallelism
        return []

    def _optimize_element_wise(self, operation: ParallelizableOperation,
                             base_config: ParallelismConfig) -> List[ParallelismConfig]:
        """Optimize element-wise operations."""
        # For element-wise, prioritize SIMD parallelism
        return []


def create_optimizer(strategy: OptimizationStrategy = OptimizationStrategy.HEURISTIC,
                    objective: OptimizationObjective = OptimizationObjective.BALANCED,
                    constraints: Optional[OptimizationConstraints] = None,
                    **kwargs) -> ParallelismOptimizer:
    """
    Factory function to create optimizers.

    Args:
        strategy: Optimization strategy to use
        objective: Optimization objective
        constraints: Resource constraints
        **kwargs: Additional arguments for specific optimizers

    Returns:
        Configured optimizer instance
    """
    if strategy == OptimizationStrategy.GRID_SEARCH:
        return GridSearchOptimizer(objective, constraints, **kwargs)
    elif strategy == OptimizationStrategy.HEURISTIC:
        return HeuristicOptimizer(objective, constraints)
    else:
        raise ValueError(f"Optimization strategy {strategy} not implemented yet")
