"""Tests for MatrixVector operation."""

import pytest
from typing import Dict

from extensible_parallelism.operations.matrix_vector import (
    MatrixVectorOperation, 
    MatrixVectorParams
)
from extensible_parallelism.core.base import (
    TensorSpec, 
    ParallelismConfig, 
    TensorParallelism,
    ParallelismStrategy,
    MemoryStrategy
)
from extensible_parallelism.core.registry import OperationRegistry


class TestMatrixVectorOperation:
    """Test suite for MatrixVector operation."""
    
    def test_basic_initialization(self):
        """Test basic operation initialization."""
        params = MatrixVectorParams(
            matrix_height=64,
            matrix_width=32,
            input_dtype="int8",
            weight_dtype="int8",
            output_dtype="int32"
        )
        
        op = MatrixVectorOperation("test_mv", params)
        assert op.params.matrix_height == 64
        assert op.params.matrix_width == 32
        assert op.params.input_dtype == "int8"
        assert op.params.weight_dtype == "int8"
        assert op.params.output_dtype == "int32"
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="Matrix dimensions must be positive"):
            MatrixVectorParams(matrix_height=0, matrix_width=32)
        
        with pytest.raises(ValueError, match="Matrix dimensions must be positive"):
            MatrixVectorParams(matrix_height=64, matrix_width=-1)
    
    def test_invalid_dtypes(self):
        """Test validation of invalid data types."""
        with pytest.raises(ValueError, match="Invalid input dtype"):
            MatrixVectorParams(
                matrix_height=64, 
                matrix_width=32, 
                input_dtype="invalid"
            )
    
    def test_input_tensors(self):
        """Test input tensor specifications."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        input_tensors = op.get_input_tensors()
        
        assert "input" in input_tensors
        assert "weights" in input_tensors
        
        input_spec = input_tensors["input"]
        assert input_spec.shape == (32,)
        assert input_spec.dtype == "int8"
        assert input_spec.dimension_names == ["input_features"]
        
        weight_spec = input_tensors["weights"]
        assert weight_spec.shape == (64, 32)
        assert weight_spec.dtype == "int8"
        assert weight_spec.dimension_names == ["output_features", "input_features"]
    
    def test_output_tensors(self):
        """Test output tensor specifications."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        output_tensors = op.get_output_tensors()
        
        assert "output" in output_tensors
        
        output_spec = output_tensors["output"]
        assert output_spec.shape == (64,)
        assert output_spec.dtype == "int32"
        assert output_spec.dimension_names == ["output_features"]
    
    def test_bias_tensor(self):
        """Test bias tensor when specified."""
        params = MatrixVectorParams(
            matrix_height=64, 
            matrix_width=32,
            bias_dtype="int32"
        )
        op = MatrixVectorOperation("test_mv", params)
        
        input_tensors = op.get_input_tensors()
        assert "bias" in input_tensors
        
        bias_spec = input_tensors["bias"]
        assert bias_spec.shape == (64,)
        assert bias_spec.dtype == "int32"
        assert bias_spec.dimension_names == ["output_features"]
    
    def test_internal_tensors(self):
        """Test internal tensor specifications."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        internal_tensors = op.get_internal_tensors()
        
        assert "accumulator" in internal_tensors
        
        acc_spec = internal_tensors["accumulator"]
        assert acc_spec.shape == (64,)
        assert acc_spec.dtype == "int32"
        assert acc_spec.dimension_names == ["output_features"]
    
    def test_constraints(self):
        """Test operation constraints."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        constraints = op.get_constraints()
        assert len(constraints) == 2
        
        # Should have PE and SIMD divisibility constraints
        constraint_names = [c.__class__.__name__ for c in constraints]
        assert "DivisibilityConstraint" in constraint_names
    
    def test_default_parallelism_config(self):
        """Test default parallelism configuration."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        config = op.get_default_parallelism_config()
        
        assert "input" in config.input_tensors
        assert "weights" in config.input_tensors
        assert "output" in config.output_tensors
        assert "accumulator" in config.internal_tensors
        
        # Check default values (should be 1 for minimal parallelism)
        input_parallelism = config.input_tensors["input"]
        assert input_parallelism.dimension_factors["input_features"] == 1
        
        output_parallelism = config.output_tensors["output"]
        assert output_parallelism.dimension_factors["output_features"] == 1
    
    def test_resource_estimation(self):
        """Test resource estimation."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        config = op.get_default_parallelism_config()
        resources = op.estimate_resources(config)
        
        assert resources.dsps >= 0
        assert resources.brams >= 0
        assert resources.luts >= 0
        assert resources.ffs >= 0
        
        # With default config (PE=1, SIMD=1), should use 1 DSP
        assert resources.dsps == 1
    
    def test_resource_estimation_with_parallelism(self):
        """Test resource estimation with increased parallelism."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        # Create config with PE=4, SIMD=2
        config = ParallelismConfig(
            input_tensors={
                "input": TensorParallelism(
                    dimension_factors={"input_features": 2},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                ),
                "weights": TensorParallelism(
                    dimension_factors={"output_features": 4, "input_features": 2},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.BLOCK_RAM
                )
            },
            output_tensors={
                "output": TensorParallelism(
                    dimension_factors={"output_features": 4},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                )
            },
            internal_tensors={
                "accumulator": TensorParallelism(
                    dimension_factors={"output_features": 4},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.REGISTERS
                )
            }
        )
        
        resources = op.estimate_resources(config)
        
        # Should use PE * SIMD = 4 * 2 = 8 DSPs
        assert resources.dsps == 8
        assert resources.brams > 0  # Should need BRAMs for weights
    
    def test_parallelism_bounds(self):
        """Test parallelism bounds."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        bounds = op.get_parallelism_bounds()
        
        assert "PE" in bounds
        assert "SIMD" in bounds
        
        pe_min, pe_max = bounds["PE"]
        assert pe_min == 1
        assert pe_max == 64  # matrix_height
        
        simd_min, simd_max = bounds["SIMD"]
        assert simd_min == 1
        assert simd_max == 32  # matrix_width
    
    def test_config_validation_valid(self):
        """Test validation of valid configuration."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        config = op.get_default_parallelism_config()
        assert op.validate_config(config) is True
    
    def test_config_validation_invalid_missing_tensors(self):
        """Test validation fails with missing tensors."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        # Create incomplete config
        config = ParallelismConfig(
            input_tensors={},  # Missing required tensors
            output_tensors={},
            internal_tensors={}
        )
        
        assert op.validate_config(config) is False
    
    def test_operation_registration(self):
        """Test that operation is properly registered."""
        registry = OperationRegistry()
        
        # Check that matrix_vector operation is registered
        operations = registry.list_operations()
        assert "matrix_vector" in operations
        
        # Test creation through registry
        op = registry.create(
            "matrix_vector",
            name="test_registry_mv",
            params=MatrixVectorParams(matrix_height=32, matrix_width=16)
        )
        assert isinstance(op, MatrixVectorOperation)
        assert op.params.matrix_height == 32
        assert op.params.matrix_width == 16
    
    def test_string_representation(self):
        """Test string representation."""
        params = MatrixVectorParams(matrix_height=64, matrix_width=32)
        op = MatrixVectorOperation("test_mv", params)
        
        str_repr = str(op)
        assert "MatrixVectorOperation" in str_repr
        assert "64" in str_repr
        assert "32" in str_repr
        assert "int8" in str_repr
        assert "int32" in str_repr
        
        repr_str = repr(op)
        assert repr_str == str_repr
