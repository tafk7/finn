"""Tests for Convolution operation."""

import pytest
from typing import Dict

from extensible_parallelism.operations.convolution import (
    ConvolutionOperation, 
    ConvolutionParams
)
from extensible_parallelism.core.base import (
    TensorSpec, 
    ParallelismConfig, 
    TensorParallelism,
    ParallelismStrategy,
    MemoryStrategy
)
from extensible_parallelism.core.registry import OperationRegistry


class TestConvolutionOperation:
    """Test suite for Convolution operation."""
    
    def test_basic_initialization_3d(self):
        """Test basic 3D convolution initialization."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),  # (C, H, W)
            kernel_shape=(32, 16, 3, 3),  # (C_out, C_in, K_h, K_w)
            stride=(1, 1),
            padding=(1, 1, 1, 1),
            input_dtype="int8",
            weight_dtype="int8",
            output_dtype="int32"
        )
        
        op = ConvolutionOperation("test_conv", params)
        assert op.params.input_shape == (16, 32, 32)
        assert op.params.kernel_shape == (32, 16, 3, 3)
        assert op.params.stride == (1, 1)
        assert op.params.padding == (1, 1, 1, 1)
        assert op.output_shape == (32, 32, 32)  # Same size due to padding
    
    def test_basic_initialization_4d(self):
        """Test basic 4D convolution initialization."""
        params = ConvolutionParams(
            input_shape=(1, 16, 32, 32),  # (N, C, H, W)
            kernel_shape=(32, 16, 3, 3),  # (C_out, C_in, K_h, K_w)
            stride=(2, 2),
            padding=(0, 0, 0, 0)
        )
        
        op = ConvolutionOperation("test_conv", params)
        assert op.params.input_shape == (1, 16, 32, 32)
        assert op.output_shape == (1, 32, 15, 15)  # (32-3+1)/2 = 15
    
    def test_invalid_input_shape(self):
        """Test validation of invalid input shapes."""
        with pytest.raises(ValueError, match="Input shape must be 3D \\(C,H,W\\) or 4D \\(N,C,H,W\\)"):
            ConvolutionParams(
                input_shape=(32, 32),  # Only 2D
                kernel_shape=(16, 32, 3, 3)
            )
        
        with pytest.raises(ValueError, match="Input shape must be 3D \\(C,H,W\\) or 4D \\(N,C,H,W\\)"):
            ConvolutionParams(
                input_shape=(1, 2, 3, 4, 5),  # 5D
                kernel_shape=(16, 32, 3, 3)
            )
    
    def test_invalid_kernel_shape(self):
        """Test validation of invalid kernel shapes."""
        with pytest.raises(ValueError, match="Kernel shape must be 4D"):
            ConvolutionParams(
                input_shape=(16, 32, 32),
                kernel_shape=(32, 16, 3)  # Only 3D
            )
    
    def test_channel_mismatch(self):
        """Test validation of channel dimension mismatch."""
        with pytest.raises(ValueError, match="Input channels .* must equal kernel input channels"):
            ConvolutionParams(
                input_shape=(16, 32, 32),  # 16 input channels
                kernel_shape=(32, 8, 3, 3)  # Kernel expects 8 input channels
            )
    
    def test_invalid_stride(self):
        """Test validation of invalid stride values."""
        with pytest.raises(ValueError, match="Stride values must be positive"):
            ConvolutionParams(
                input_shape=(16, 32, 32),
                kernel_shape=(32, 16, 3, 3),
                stride=(0, 1)
            )
    
    def test_invalid_dilation(self):
        """Test validation of invalid dilation values."""
        with pytest.raises(ValueError, match="Dilation values must be positive"):
            ConvolutionParams(
                input_shape=(16, 32, 32),
                kernel_shape=(32, 16, 3, 3),
                dilation=(1, 0)
            )
    
    def test_invalid_groups(self):
        """Test validation of invalid groups."""
        with pytest.raises(ValueError, match="Groups must be positive"):
            ConvolutionParams(
                input_shape=(16, 32, 32),
                kernel_shape=(32, 16, 3, 3),
                groups=0
            )
    
    def test_input_tensors_3d(self):
        """Test input tensor specifications for 3D input."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        input_tensors = op.get_input_tensors()
        
        assert "input" in input_tensors
        assert "weights" in input_tensors
        
        input_spec = input_tensors["input"]
        assert input_spec.shape == (16, 32, 32)
        assert input_spec.dtype == "int8"
        assert input_spec.dimension_names == ["channels", "height", "width"]
        
        weight_spec = input_tensors["weights"]
        assert weight_spec.shape == (32, 16, 3, 3)
        assert weight_spec.dtype == "int8"
        assert weight_spec.dimension_names == ["out_channels", "in_channels", "kernel_height", "kernel_width"]
    
    def test_input_tensors_4d(self):
        """Test input tensor specifications for 4D input."""
        params = ConvolutionParams(
            input_shape=(2, 16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        input_tensors = op.get_input_tensors()
        
        input_spec = input_tensors["input"]
        assert input_spec.shape == (2, 16, 32, 32)
        assert input_spec.dimension_names == ["batch", "channels", "height", "width"]
    
    def test_bias_tensor(self):
        """Test bias tensor when specified."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3),
            bias_dtype="int32"
        )
        op = ConvolutionOperation("test_conv", params)
        
        input_tensors = op.get_input_tensors()
        assert "bias" in input_tensors
        
        bias_spec = input_tensors["bias"]
        assert bias_spec.shape == (32,)  # C_out
        assert bias_spec.dtype == "int32"
        assert bias_spec.dimension_names == ["out_channels"]
    
    def test_output_tensors_3d(self):
        """Test output tensor specifications for 3D input."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3),
            padding=(1, 1, 1, 1)  # Same padding
        )
        op = ConvolutionOperation("test_conv", params)
        
        output_tensors = op.get_output_tensors()
        
        assert "output" in output_tensors
        
        output_spec = output_tensors["output"]
        assert output_spec.shape == (32, 32, 32)  # (C_out, H_out, W_out)
        assert output_spec.dtype == "int32"
        assert output_spec.dimension_names == ["channels", "height", "width"]
    
    def test_output_tensors_4d(self):
        """Test output tensor specifications for 4D input."""
        params = ConvolutionParams(
            input_shape=(2, 16, 32, 32),
            kernel_shape=(32, 16, 3, 3),
            padding=(1, 1, 1, 1)
        )
        op = ConvolutionOperation("test_conv", params)
        
        output_tensors = op.get_output_tensors()
        
        output_spec = output_tensors["output"]
        assert output_spec.shape == (2, 32, 32, 32)  # (N, C_out, H_out, W_out)
        assert output_spec.dimension_names == ["batch", "channels", "height", "width"]
    
    def test_output_shape_calculation_stride(self):
        """Test output shape calculation with stride."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3),
            stride=(2, 2),
            padding=(0, 0, 0, 0)
        )
        op = ConvolutionOperation("test_conv", params)
        
        # (32 - 3 + 1) / 2 = 15
        assert op.output_shape == (32, 15, 15)
    
    def test_output_shape_calculation_dilation(self):
        """Test output shape calculation with dilation."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3),
            stride=(1, 1),
            padding=(0, 0, 0, 0),
            dilation=(2, 2)
        )
        op = ConvolutionOperation("test_conv", params)
        
        # Effective kernel size: (k-1)*d + 1 = (3-1)*2 + 1 = 5
        # (32 - 5 + 1) = 28
        assert op.output_shape == (32, 28, 28)
    
    def test_internal_tensors(self):
        """Test internal tensor specifications."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        internal_tensors = op.get_internal_tensors()
        
        assert "line_buffer" in internal_tensors
        assert "window_buffer" in internal_tensors
        
        line_buffer_spec = internal_tensors["line_buffer"]
        assert line_buffer_spec.shape == (16, 2, 32)  # (C_in, K_h-1, W_in)
        assert line_buffer_spec.dimension_names == ["channels", "buffer_lines", "width"]
        
        window_buffer_spec = internal_tensors["window_buffer"]
        assert window_buffer_spec.shape == (16, 3, 3)  # (C_in, K_h, K_w)
        assert window_buffer_spec.dimension_names == ["channels", "kernel_height", "kernel_width"]
    
    def test_constraints(self):
        """Test operation constraints."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        constraints = op.get_constraints()
        assert len(constraints) == 2
        
        # Should have PE and SIMD divisibility constraints
        constraint_names = [c.__class__.__name__ for c in constraints]
        assert "DivisibilityConstraint" in constraint_names
    
    def test_default_parallelism_config_3d(self):
        """Test default parallelism configuration for 3D input."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        config = op.get_default_parallelism_config()
        
        assert "input" in config.input_tensors
        assert "weights" in config.input_tensors
        assert "output" in config.output_tensors
        assert "line_buffer" in config.internal_tensors
        assert "window_buffer" in config.internal_tensors
        
        # Check default values (should be 1 for minimal parallelism)
        input_parallelism = config.input_tensors["input"]
        assert input_parallelism.dimension_factors["channels"] == 1
        assert input_parallelism.dimension_factors["height"] == 1
        assert input_parallelism.dimension_factors["width"] == 1
        
        output_parallelism = config.output_tensors["output"]
        assert output_parallelism.dimension_factors["channels"] == 1
    
    def test_resource_estimation(self):
        """Test resource estimation."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        config = op.get_default_parallelism_config()
        resources = op.estimate_resources(config)
        
        assert resources.dsps >= 0
        assert resources.brams >= 0
        assert resources.luts >= 0
        assert resources.ffs >= 0
        
        # With default config (PE=1, SIMD=1), should use 1 DSP
        assert resources.dsps == 1
        assert resources.brams > 0  # Should need BRAMs for weights and line buffers
    
    def test_resource_estimation_with_parallelism(self):
        """Test resource estimation with increased parallelism."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        # Create config with PE=4, SIMD=2
        config = ParallelismConfig(
            input_tensors={
                "input": TensorParallelism(
                    dimension_factors={"channels": 2, "height": 1, "width": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                ),
                "weights": TensorParallelism(
                    dimension_factors={
                        "out_channels": 4, "in_channels": 2, 
                        "kernel_height": 1, "kernel_width": 1
                    },
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.BLOCK_RAM
                )
            },
            output_tensors={
                "output": TensorParallelism(
                    dimension_factors={"channels": 4, "height": 1, "width": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.STREAMING
                )
            },
            internal_tensors={
                "line_buffer": TensorParallelism(
                    dimension_factors={"channels": 2, "buffer_lines": 1, "width": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.BLOCK_RAM
                ),
                "window_buffer": TensorParallelism(
                    dimension_factors={"channels": 2, "kernel_height": 1, "kernel_width": 1},
                    strategy=ParallelismStrategy.SPATIAL,
                    memory_strategy=MemoryStrategy.REGISTERS
                )
            }
        )
        
        resources = op.estimate_resources(config)
        
        # Should use PE * SIMD = 4 * 2 = 8 DSPs
        assert resources.dsps == 8
        assert resources.brams > 0
    
    def test_parallelism_bounds(self):
        """Test parallelism bounds."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        bounds = op.get_parallelism_bounds()
        
        assert "PE" in bounds
        assert "SIMD" in bounds
        
        pe_min, pe_max = bounds["PE"]
        assert pe_min == 1
        assert pe_max == 32  # C_out
        
        simd_min, simd_max = bounds["SIMD"]
        assert simd_min == 1
        assert simd_max == 16  # C_in
    
    def test_config_validation_valid(self):
        """Test validation of valid configuration."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3)
        )
        op = ConvolutionOperation("test_conv", params)
        
        config = op.get_default_parallelism_config()
        assert op.validate_config(config) is True
    
    def test_grouped_convolution(self):
        """Test grouped convolution."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 8, 3, 3),  # 8 input channels per group
            groups=2  # 16 / 2 = 8 channels per group
        )
        
        # Should not raise an exception
        op = ConvolutionOperation("test_conv", params)
        assert op.params.groups == 2
    
    def test_operation_registration(self):
        """Test that operation is properly registered."""
        registry = OperationRegistry()
        
        # Check that convolution operation is registered
        operations = registry.list_operations()
        assert "convolution" in operations
        
        # Test creation through registry
        op = registry.create(
            "convolution", 
            name="test_conv",
            params=ConvolutionParams(
                input_shape=(16, 32, 32),
                kernel_shape=(32, 16, 3, 3)
            )
        )
        assert isinstance(op, ConvolutionOperation)
        assert op.params.input_shape == (16, 32, 32)
        assert op.params.kernel_shape == (32, 16, 3, 3)
    
    def test_string_representation(self):
        """Test string representation."""
        params = ConvolutionParams(
            input_shape=(16, 32, 32),
            kernel_shape=(32, 16, 3, 3),
            stride=(2, 2),
            padding=(1, 1, 1, 1)
        )
        op = ConvolutionOperation("test_conv", params)
        
        str_repr = str(op)
        assert "ConvolutionOperation" in str_repr
        assert "(16, 32, 32)" in str_repr
        assert "(32, 16, 3, 3)" in str_repr
        assert "(2, 2)" in str_repr
        assert "(1, 1, 1, 1)" in str_repr
        
        repr_str = repr(op)
        assert repr_str == str_repr
