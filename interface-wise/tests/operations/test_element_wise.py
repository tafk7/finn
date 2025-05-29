"""Tests for ElementWise operation."""

import pytest
from typing import Dict

from extensible_parallelism.operations.element_wise import (
    ElementWiseOperation, 
    ElementWiseParams,
    ElementWiseOpType
)
from extensible_parallelism.core.base import (
    TensorSpec, 
    ParallelismConfig, 
    TensorParallelism,
    ParallelismStrategy,
    MemoryStrategy
)
from extensible_parallelism.core.registry import OperationRegistry


class TestElementWiseOperation:
    """Test suite for ElementWise operation."""
    
    def test_basic_initialization_unary(self):
        """Test basic unary operation initialization."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.RELU,
            tensor_shape=(128,),
            input_dtype="int16",
            output_dtype="int16",
            num_inputs=1
        )
        
        op = ElementWiseOperation("test_ew", params)
        assert op.params.operation_type == ElementWiseOpType.RELU
        assert op.params.tensor_shape == (128,)
        assert op.params.input_dtype == "int16"
        assert op.params.output_dtype == "int16"
        assert op.params.num_inputs == 1
    
    def test_basic_initialization_binary(self):
        """Test basic binary operation initialization."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.ADD,
            tensor_shape=(64, 32),
            input_dtype="int16",
            output_dtype="int16",
            num_inputs=2
        )
        
        op = ElementWiseOperation("test_ew", params)
        assert op.params.operation_type == ElementWiseOpType.ADD
        assert op.params.tensor_shape == (64, 32)
        assert op.params.num_inputs == 2
    
    def test_invalid_dimensions(self):
        """Test validation of invalid dimensions."""
        with pytest.raises(ValueError, match="All tensor dimensions must be positive"):
            ElementWiseParams(
                operation_type=ElementWiseOpType.RELU,
                tensor_shape=(0, 32),
                num_inputs=1
            )
        
        with pytest.raises(ValueError, match="All tensor dimensions must be positive"):
            ElementWiseParams(
                operation_type=ElementWiseOpType.ADD,
                tensor_shape=(64, -1),
                num_inputs=2
            )
    
    def test_invalid_dtypes(self):
        """Test validation of invalid data types."""
        with pytest.raises(ValueError, match="Invalid input dtype"):
            ElementWiseParams(
                operation_type=ElementWiseOpType.RELU,
                tensor_shape=(128,),
                input_dtype="invalid",
                num_inputs=1
            )
    
    def test_invalid_num_inputs_binary(self):
        """Test validation of incorrect number of inputs for binary operations."""
        with pytest.raises(ValueError, match="Binary operation add requires 2 inputs"):
            ElementWiseParams(
                operation_type=ElementWiseOpType.ADD,
                tensor_shape=(128,),
                num_inputs=1
            )
    
    def test_invalid_num_inputs_unary(self):
        """Test validation of incorrect number of inputs for unary operations."""
        with pytest.raises(ValueError, match="Unary operation relu requires 1 input"):
            ElementWiseParams(
                operation_type=ElementWiseOpType.RELU,
                tensor_shape=(128,),
                num_inputs=2
            )
    
    def test_input_tensors_unary(self):
        """Test input tensor specifications for unary operation."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.RELU,
            tensor_shape=(128,),
            num_inputs=1
        )
        op = ElementWiseOperation("test_ew", params)
        
        input_tensors = op.get_input_tensors()
        
        assert len(input_tensors) == 1
        assert "input_0" in input_tensors
        
        input_spec = input_tensors["input_0"]
        assert input_spec.shape == (128,)
        assert input_spec.dtype == "int16"
        assert input_spec.dimension_names == ["dim_0"]
    
    def test_input_tensors_binary(self):
        """Test input tensor specifications for binary operation."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.ADD,
            tensor_shape=(64, 32),
            num_inputs=2
        )
        op = ElementWiseOperation("test_ew", params)
        
        input_tensors = op.get_input_tensors()
        
        assert len(input_tensors) == 2
        assert "input_0" in input_tensors
        assert "input_1" in input_tensors
        
        input_spec_0 = input_tensors["input_0"]
        assert input_spec_0.shape == (64, 32)
        assert input_spec_0.dtype == "int16"
        assert input_spec_0.dimension_names == ["dim_0", "dim_1"]
        
        input_spec_1 = input_tensors["input_1"]
        assert input_spec_1.shape == (64, 32)
        assert input_spec_1.dtype == "int16"
        assert input_spec_1.dimension_names == ["dim_0", "dim_1"]
    
    def test_output_tensors(self):
        """Test output tensor specifications."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.RELU,
            tensor_shape=(64, 32),
            num_inputs=1
        )
        op = ElementWiseOperation("test_ew", params)
        
        output_tensors = op.get_output_tensors()
        
        assert len(output_tensors) == 1
        assert "output" in output_tensors
        
        output_spec = output_tensors["output"]
        assert output_spec.shape == (64, 32)
        assert output_spec.dtype == "int16"
        assert output_spec.dimension_names == ["dim_0", "dim_1"]
    
    def test_internal_tensors_simple_op(self):
        """Test internal tensors for simple operations (should be empty)."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.RELU,
            tensor_shape=(128,),
            num_inputs=1
        )
        op = ElementWiseOperation("test_ew", params)
        
        internal_tensors = op.get_internal_tensors()
        assert len(internal_tensors) == 0
    
    def test_internal_tensors_complex_op(self):
        """Test internal tensors for complex operations."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.SIGMOID,
            tensor_shape=(128,),
            num_inputs=1
        )
        op = ElementWiseOperation("test_ew", params)
        
        internal_tensors = op.get_internal_tensors()
        
        # Sigmoid needs internal storage
        assert len(internal_tensors) == 1
        assert "temp" in internal_tensors
        
        temp_spec = internal_tensors["temp"]
        assert temp_spec.shape == (128,)
        assert temp_spec.dtype == "int16"
        assert temp_spec.dimension_names == ["dim_0"]
    
    def test_constraints(self):
        """Test operation constraints."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.ADD,
            tensor_shape=(64, 32),
            num_inputs=2
        )
        op = ElementWiseOperation("test_ew", params)
        
        constraints = op.get_constraints()
        assert len(constraints) == 1
        
        # Should have SIMD divisibility constraint on last dimension
        constraint = constraints[0]
        assert constraint.__class__.__name__ == "DivisibilityConstraint"
    
    def test_default_parallelism_config_unary(self):
        """Test default parallelism configuration for unary operation."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.RELU,
            tensor_shape=(64, 32),
            num_inputs=1
        )
        op = ElementWiseOperation("test_ew", params)
        
        config = op.get_default_parallelism_config()
        
        assert "input_0" in config.input_tensors
        assert "output" in config.output_tensors
        
        # Check default values (should be 1 for minimal parallelism)
        input_parallelism = config.input_tensors["input_0"]
        assert input_parallelism.dimension_factors["dim_0"] == 1
        assert input_parallelism.dimension_factors["dim_1"] == 1
        
        output_parallelism = config.output_tensors["output"]
        assert output_parallelism.dimension_factors["dim_0"] == 1
        assert output_parallelism.dimension_factors["dim_1"] == 1
    
    def test_default_parallelism_config_binary(self):
        """Test default parallelism configuration for binary operation."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.ADD,
            tensor_shape=(64, 32),
            num_inputs=2
        )
        op = ElementWiseOperation("test_ew", params)
        
        config = op.get_default_parallelism_config()
        
        assert "input_0" in config.input_tensors
        assert "input_1" in config.input_tensors
        assert "output" in config.output_tensors
        
        # Both inputs should have same default parallelism
        input0_parallelism = config.input_tensors["input_0"]
        input1_parallelism = config.input_tensors["input_1"]
        assert input0_parallelism.dimension_factors == input1_parallelism.dimension_factors
    
    def test_resource_estimation_simple_op(self):
        """Test resource estimation for simple operations."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.ADD,
            tensor_shape=(128,),
            num_inputs=2
        )
        op = ElementWiseOperation("test_ew", params)
        
        config = op.get_default_parallelism_config()
        resources = op.estimate_resources(config)
        
        assert resources.dsps >= 0
        assert resources.brams >= 0
        assert resources.luts >= 0
        assert resources.ffs >= 0
        
        # Addition should not use DSPs
        assert resources.dsps == 0
    
    def test_resource_estimation_multiply_op(self):
        """Test resource estimation for multiply operations."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.MUL,
            tensor_shape=(128,),
            num_inputs=2
        )
        op = ElementWiseOperation("test_ew", params)
        
        config = op.get_default_parallelism_config()
        resources = op.estimate_resources(config)
        
        # Multiplication should use DSPs
        assert resources.dsps > 0
    
    def test_resource_estimation_with_parallelism(self):
        """Test resource estimation with increased parallelism."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.MUL,
            tensor_shape=(128,),
            num_inputs=2
        )
        op = ElementWiseOperation("test_ew", params)
        
        # Create config with SIMD=4
        config = ParallelismConfig(
            input_tensors={
                "input_0": TensorParallelism(
                    dimension_factors={"dim_0": 4},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                ),
                "input_1": TensorParallelism(
                    dimension_factors={"dim_0": 4},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                )
            },
            output_tensors={
                "output": TensorParallelism(
                    dimension_factors={"dim_0": 4},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                )
            },
            internal_tensors={}
        )
        
        resources = op.estimate_resources(config)
        
        # Should use 4 DSPs (one per SIMD lane)
        assert resources.dsps == 4
    
    def test_parallelism_bounds(self):
        """Test parallelism bounds."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.ADD,
            tensor_shape=(64, 32),
            num_inputs=2
        )
        op = ElementWiseOperation("test_ew", params)
        
        bounds = op.get_parallelism_bounds()
        
        assert "SIMD" in bounds
        
        simd_min, simd_max = bounds["SIMD"]
        assert simd_min == 1
        assert simd_max == 64 * 32  # Total number of elements
    
    def test_config_validation_valid(self):
        """Test validation of valid configuration."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.RELU,
            tensor_shape=(128,),
            num_inputs=1
        )
        op = ElementWiseOperation("test_ew", params)
        
        config = op.get_default_parallelism_config()
        assert op.validate_config(config) is True
    
    def test_operation_registration(self):
        """Test that operation is properly registered."""
        registry = OperationRegistry()
        
        # Check that element_wise operation is registered
        operations = registry.list_operations()
        assert "element_wise" in operations
         # Test creation through registry
        op = registry.create(
            "element_wise",
            name="test_registry_ew",
            params=ElementWiseParams(
                operation_type=ElementWiseOpType.RELU,
                tensor_shape=(64,),
                num_inputs=1
            )
        )
        assert isinstance(op, ElementWiseOperation)
        assert op.params.operation_type == ElementWiseOpType.RELU
    
    def test_string_representation(self):
        """Test string representation."""
        params = ElementWiseParams(
            operation_type=ElementWiseOpType.ADD,
            tensor_shape=(64, 32),
            num_inputs=2
        )
        op = ElementWiseOperation("test_ew", params)
        
        str_repr = str(op)
        assert "ElementWiseOperation" in str_repr
        assert "add" in str_repr
        assert "(64, 32)" in str_repr
        assert "num_inputs=2" in str_repr
        
        repr_str = repr(op)
        assert repr_str == str_repr
    
    def test_all_operation_types(self):
        """Test that all operation types can be instantiated."""
        for op_type in ElementWiseOpType:
            num_inputs = 1 if op_type in [
                ElementWiseOpType.RELU, ElementWiseOpType.SIGMOID, ElementWiseOpType.TANH,
                ElementWiseOpType.ABS, ElementWiseOpType.SQRT, ElementWiseOpType.EXP,
                ElementWiseOpType.LOG
            ] else 2
            
            params = ElementWiseParams(
                operation_type=op_type,
                tensor_shape=(32,),
                num_inputs=num_inputs
            )
            
            # Should not raise any exceptions
            op = ElementWiseOperation("test_ew", params)
            assert op.params.operation_type == op_type
