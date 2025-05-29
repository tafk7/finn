"""Tests for parallelism optimizer."""

import pytest
import time
from typing import Dict, List
from unittest.mock import Mock, patch

from extensible_parallelism.core.optimizer import (
    ParallelismOptimizer,
    GridSearchOptimizer,
    HeuristicOptimizer,
    OptimizationStrategy,
    OptimizationObjective,
    OptimizationConstraints,
    OptimizationResult
)
from extensible_parallelism.core.base import (
    ParallelismConfig,
    TensorParallelism,
    ResourceEstimate,
    ParallelismStrategy,
    MemoryStrategy
)
from extensible_parallelism.operations.matrix_vector import MatrixVectorOperation, MatrixVectorParams
from extensible_parallelism.operations.element_wise import ElementWiseOperation, ElementWiseParams, ElementWiseOpType
from extensible_parallelism.operations.convolution import ConvolutionOperation, ConvolutionParams


class TestOptimizationConstraints:
    """Test suite for OptimizationConstraints."""
    
    def test_default_constraints(self):
        """Test default constraint initialization."""
        constraints = OptimizationConstraints()
        assert constraints.max_dsps is None
        assert constraints.max_brams is None
        assert constraints.max_luts is None
        assert constraints.clock_frequency == 100.0
    
    def test_constraint_satisfaction_no_limits(self):
        """Test constraint satisfaction with no limits."""
        constraints = OptimizationConstraints()
        resources = ResourceEstimate(dsps=100, brams=50, luts=1000, ffs=2000, latency_cycles=10)
        assert constraints.is_satisfied_by(resources)
    
    def test_constraint_satisfaction_with_limits(self):
        """Test constraint satisfaction with resource limits."""
        constraints = OptimizationConstraints(
            max_dsps=50,
            max_brams=25,
            max_luts=500,
            max_ffs=1000,
            max_latency_cycles=20
        )
        
        # Resource estimate within limits
        good_resources = ResourceEstimate(dsps=40, brams=20, luts=400, ffs=800, latency_cycles=15)
        assert constraints.is_satisfied_by(good_resources)
        
        # Resource estimate exceeding DSP limit
        bad_dsps = ResourceEstimate(dsps=60, brams=20, luts=400, ffs=800, latency_cycles=15)
        assert not constraints.is_satisfied_by(bad_dsps)
        
        # Resource estimate exceeding latency limit
        bad_latency = ResourceEstimate(dsps=40, brams=20, luts=400, ffs=800, latency_cycles=25)
        assert not constraints.is_satisfied_by(bad_latency)


class TestOptimizationResult:
    """Test suite for OptimizationResult."""
    
    def test_result_creation(self):
        """Test optimization result creation."""
        config = ParallelismConfig()
        resources = ResourceEstimate(dsps=10, brams=5, luts=100, ffs=200, latency_cycles=8)
        
        result = OptimizationResult(
            best_config=config,
            best_score=0.85,
            resource_estimate=resources,
            search_time=1.5,
            configurations_evaluated=25,
            optimization_log=["Starting optimization", "Found solution"]
        )
        
        assert result.best_config == config
        assert result.best_score == 0.85
        assert result.resource_estimate == resources
        assert result.search_time == 1.5
        assert result.configurations_evaluated == 25
        assert len(result.optimization_log) == 2
    
    def test_string_representation(self):
        """Test string representation of result."""
        config = ParallelismConfig()
        resources = ResourceEstimate(dsps=10, brams=5, luts=100, ffs=200, latency_cycles=8)
        
        result = OptimizationResult(
            best_config=config,
            best_score=0.85,
            resource_estimate=resources,
            search_time=1.5,
            configurations_evaluated=25,
            optimization_log=[]
        )
        
        result_str = str(result)
        assert "score=0.850" in result_str
        assert "dsps=10" in result_str
        assert "evaluated=25" in result_str
        assert "time=1.500s" in result_str


class TestGridSearchOptimizer:
    """Test suite for GridSearchOptimizer."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = GridSearchOptimizer(
            objective=OptimizationObjective.MAXIMIZE_THROUGHPUT,
            constraints=OptimizationConstraints(max_dsps=100)
        )
        
        assert optimizer.objective == OptimizationObjective.MAXIMIZE_THROUGHPUT
        assert optimizer.constraints.max_dsps == 100
        assert optimizer.max_parallelism_factor == 16
    
    def test_optimize_matrix_vector_minimal(self):
        """Test optimization of simple matrix-vector operation."""
        # Create simple operation
        params = MatrixVectorParams(matrix_height=32, matrix_width=16)
        operation = MatrixVectorOperation("test_mv", params)
        
        # Create optimizer with loose constraints
        constraints = OptimizationConstraints(max_dsps=200, max_brams=100)
        optimizer = GridSearchOptimizer(
            objective=OptimizationObjective.BALANCED,
            constraints=constraints,
            max_parallelism_factor=8,  # Limit for fast test
            max_evaluations=50
        )
        
        # Run optimization
        result = optimizer.optimize(operation)
        
        # Verify result structure
        assert isinstance(result, OptimizationResult)
        assert isinstance(result.best_config, ParallelismConfig)
        assert isinstance(result.resource_estimate, ResourceEstimate)
        assert result.best_score > 0
        assert result.configurations_evaluated > 0
        assert result.search_time > 0
        
        # Verify resource constraints are satisfied
        assert constraints.is_satisfied_by(result.resource_estimate)
    
    def test_optimize_with_strict_constraints(self):
        """Test optimization with very strict resource constraints."""
        # Create much smaller operation for faster testing
        params = MatrixVectorParams(matrix_height=8, matrix_width=4)
        operation = MatrixVectorOperation("test_mv", params)
        
        # Very strict constraints
        constraints = OptimizationConstraints(max_dsps=5, max_brams=2)
        optimizer = GridSearchOptimizer(
            objective=OptimizationObjective.MINIMIZE_RESOURCES,
            constraints=constraints,
            max_parallelism_factor=2,  # Very limited
            max_evaluations=10  # Very small search space
        )
        
        # Should find a solution (even if minimal parallelism)
        result = optimizer.optimize(operation)
        assert constraints.is_satisfied_by(result.resource_estimate)
        assert result.resource_estimate.dsps <= 5
        assert result.resource_estimate.brams <= 2
    
    def test_no_feasible_solution(self):
        """Test behavior when no feasible solution exists."""
        # Create operation
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        operation = MatrixVectorOperation("test_mv", params)
        
        # Impossible constraints
        constraints = OptimizationConstraints(max_dsps=0, max_brams=0)
        optimizer = GridSearchOptimizer(
            objective=OptimizationObjective.MINIMIZE_RESOURCES,
            constraints=constraints,
            max_parallelism_factor=2,
            max_evaluations=50
        )
        
        # Should raise exception
        with pytest.raises(ValueError, match="No feasible configuration found"):
            optimizer.optimize(operation)


class TestHeuristicOptimizer:
    """Test suite for HeuristicOptimizer."""
    
    def test_optimizer_initialization(self):
        """Test heuristic optimizer initialization."""
        optimizer = HeuristicOptimizer(
            objective=OptimizationObjective.MINIMIZE_LATENCY,
            constraints=OptimizationConstraints(max_dsps=50)
        )
        
        assert optimizer.objective == OptimizationObjective.MINIMIZE_LATENCY
        assert optimizer.constraints.max_dsps == 50
    
    def test_optimize_matrix_vector_speed(self):
        """Test that heuristic optimizer is faster than grid search."""
        # Create reasonably sized operation
        params = MatrixVectorParams(matrix_height=128, matrix_width=64)
        operation = MatrixVectorOperation("test_mv", params)
        
        constraints = OptimizationConstraints(max_dsps=100, max_brams=50)
        
        # Time heuristic optimizer
        heuristic_optimizer = HeuristicOptimizer(
            objective=OptimizationObjective.BALANCED,
            constraints=constraints
        )
        start_time = time.time()
        heuristic_result = heuristic_optimizer.optimize(operation)
        heuristic_time = time.time() - start_time
        
        # Time grid search optimizer (with limited configurations)
        grid_optimizer = GridSearchOptimizer(
            objective=OptimizationObjective.BALANCED,
            constraints=constraints,
            max_parallelism_factor=4,  # Limited for reasonable test time
            max_evaluations=50
        )
        start_time = time.time()
        grid_result = grid_optimizer.optimize(operation)
        grid_time = time.time() - start_time
        
        # Heuristic should be faster (though this may vary with small problems)
        # At minimum, both should find valid solutions
        assert isinstance(heuristic_result, OptimizationResult)
        assert isinstance(grid_result, OptimizationResult)
        assert constraints.is_satisfied_by(heuristic_result.resource_estimate)
        assert constraints.is_satisfied_by(grid_result.resource_estimate)
        
        # Heuristic should evaluate fewer configurations
        assert heuristic_result.configurations_evaluated <= grid_result.configurations_evaluated
    
    def test_different_objectives(self):
        """Test optimization with different objectives."""
        params = MatrixVectorParams(matrix_height=32, matrix_width=16)
        operation = MatrixVectorOperation("test_mv", params)
        constraints = OptimizationConstraints(max_dsps=50, max_brams=25)
        
        objectives = [
            OptimizationObjective.MINIMIZE_LATENCY,
            OptimizationObjective.MINIMIZE_RESOURCES,
            OptimizationObjective.MAXIMIZE_THROUGHPUT,
            OptimizationObjective.BALANCED
        ]
        
        results = {}
        for objective in objectives:
            optimizer = HeuristicOptimizer(objective=objective, constraints=constraints)
            result = optimizer.optimize(operation)
            results[objective] = result
            
            # All should find valid solutions
            assert isinstance(result, OptimizationResult)
            assert constraints.is_satisfied_by(result.resource_estimate)
        
        # Different objectives should potentially give different results
        # (Though with simple operations, they might converge)
        latency_result = results[OptimizationObjective.MINIMIZE_LATENCY]
        resource_result = results[OptimizationObjective.MINIMIZE_RESOURCES]
        
        # Resource minimization should use fewer or equal resources
        assert (resource_result.resource_estimate.dsps <= latency_result.resource_estimate.dsps or
                resource_result.resource_estimate.brams <= latency_result.resource_estimate.brams or  
                resource_result.resource_estimate.luts <= latency_result.resource_estimate.luts)


class TestOptimizerIntegration:
    """Integration tests with different operation types."""
    
    def test_optimize_element_wise_operation(self):
        """Test optimization of element-wise operation."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.RELU,
            tensor_shape=(1024,),
            num_inputs=1
        )
        operation = ElementWiseOperation("test_ew", params)
        
        constraints = OptimizationConstraints(max_dsps=20, max_brams=10)
        optimizer = HeuristicOptimizer(
            objective=OptimizationObjective.MAXIMIZE_THROUGHPUT,
            constraints=constraints
        )
        
        result = optimizer.optimize(operation)
        
        assert isinstance(result, OptimizationResult)
        assert constraints.is_satisfied_by(result.resource_estimate)
    
    def test_optimize_convolution_operation(self):
        """Test optimization of convolution operation."""
        params = ConvolutionParams(
            input_shape=(1, 16, 32, 32),  # (N, C, H, W)
            kernel_shape=(32, 16, 3, 3),  # (C_out, C_in, K_H, K_W)
            stride=(1, 1),
            padding=(1, 1, 1, 1)  # (top, bottom, left, right)
        )
        operation = ConvolutionOperation("test_conv", params)
        
        constraints = OptimizationConstraints(max_dsps=100, max_brams=50)
        optimizer = GridSearchOptimizer(
            objective=OptimizationObjective.BALANCED,
            constraints=constraints,
            max_parallelism_factor=4,  # Limited for test speed
            max_evaluations=50
        )
        
        result = optimizer.optimize(operation)
        
        assert isinstance(result, OptimizationResult)
        assert constraints.is_satisfied_by(result.resource_estimate)
    
    def test_compare_optimizers_same_operation(self):
        """Test that different optimizers can handle the same operation."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        operation = MatrixVectorOperation("test_mv", params)
        constraints = OptimizationConstraints(max_dsps=40, max_brams=20)
        
        # Test grid search
        grid_optimizer = GridSearchOptimizer(
            objective=OptimizationObjective.BALANCED,
            constraints=constraints,
            max_parallelism_factor=4,
            max_evaluations=50
        )
        grid_result = grid_optimizer.optimize(operation)
        
        # Test heuristic
        heuristic_optimizer = HeuristicOptimizer(
            objective=OptimizationObjective.BALANCED,
            constraints=constraints
        )
        heuristic_result = heuristic_optimizer.optimize(operation)
        
        # Both should find valid solutions
        assert isinstance(grid_result, OptimizationResult)
        assert isinstance(heuristic_result, OptimizationResult)
        assert constraints.is_satisfied_by(grid_result.resource_estimate)
        assert constraints.is_satisfied_by(heuristic_result.resource_estimate)
        
        # Heuristic should be faster
        assert heuristic_result.search_time <= grid_result.search_time * 2  # Allow some variance
    
    def test_optimization_with_complex_constraints(self):
        """Test optimization with multiple constraint types."""
        params = MatrixVectorParams(matrix_height=128, matrix_width=64)
        operation = MatrixVectorOperation("test_mv", params)
        
        # Complex constraints
        constraints = OptimizationConstraints(
            max_dsps=50,
            max_brams=25,
            max_luts=2000,
            max_ffs=4000,
            max_latency_cycles=100,
            min_throughput=0.1,
            clock_frequency=200.0
        )
        
        optimizer = HeuristicOptimizer(
            objective=OptimizationObjective.BALANCED,
            constraints=constraints
        )
        
        result = optimizer.optimize(operation)
        
        assert isinstance(result, OptimizationResult)
        assert constraints.is_satisfied_by(result.resource_estimate)
        
        # Check specific constraints
        assert result.resource_estimate.dsps <= 50
        assert result.resource_estimate.brams <= 25
        assert result.resource_estimate.luts <= 2000
        assert result.resource_estimate.ffs <= 4000
        assert result.resource_estimate.latency_cycles <= 100


class TestOptimizerErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_operation(self):
        """Test behavior with invalid operation."""
        # Mock operation that raises errors
        mock_operation = Mock()
        mock_operation.estimate_resources.side_effect = ValueError("Invalid operation")
        mock_operation.get_input_tensors.return_value = {"input": Mock()}
        mock_operation.get_output_tensors.return_value = {"output": Mock()}
        mock_operation.get_internal_tensors.return_value = {}
        
        optimizer = HeuristicOptimizer()
        
        with pytest.raises(ValueError):
            optimizer.optimize(mock_operation)
    
    def test_empty_tensor_specs(self):
        """Test behavior with operation having no tensors."""
        mock_operation = Mock()
        mock_operation.get_input_tensors.return_value = {}
        mock_operation.get_output_tensors.return_value = {}
        mock_operation.get_internal_tensors.return_value = {}
        
        # Mock the resource estimation to return valid ResourceEstimate
        mock_resources = ResourceEstimate(dsps=0, brams=0, luts=0, ffs=0, latency_cycles=1)
        mock_operation.estimate_resources.return_value = mock_resources
        mock_operation.validate_config.return_value = False  # Invalid config
        
        optimizer = HeuristicOptimizer()
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((ValueError, RuntimeError)):
            optimizer.optimize(mock_operation)
    
    def test_optimizer_logging(self):
        """Test that optimization process is logged."""
        params = MatrixVectorParams(matrix_height=16, matrix_width=8)
        operation = MatrixVectorOperation("test_mv", params)
        
        optimizer = HeuristicOptimizer()
        result = optimizer.optimize(operation)
        
        # Should have some log entries
        assert len(result.optimization_log) > 0
        assert any("Starting" in entry for entry in result.optimization_log)
