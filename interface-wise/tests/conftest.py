"""Pytest configuration for extensible parallelism tests."""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture
def sample_matrix_vector_params():
    """Provide sample MatrixVector parameters for testing."""
    from extensible_parallelism.operations.matrix_vector import MatrixVectorParams
    
    return MatrixVectorParams(
        matrix_height=64,
        matrix_width=32,
        input_dtype="int8",
        weight_dtype="int8",
        output_dtype="int32"
    )


@pytest.fixture
def sample_element_wise_params():
    """Provide sample ElementWise parameters for testing."""
    from extensible_parallelism.operations.element_wise import (
        ElementWiseParams, 
        ElementWiseOpType
    )
    
    return ElementWiseParams(
        operation_type=ElementWiseOpType.RELU,
        tensor_shape=(128,),
        input_dtype="int16",
        output_dtype="int16",
        num_inputs=1
    )


@pytest.fixture
def sample_convolution_params():
    """Provide sample Convolution parameters for testing."""
    from extensible_parallelism.operations.convolution import ConvolutionParams
    
    return ConvolutionParams(
        input_shape=(16, 32, 32),
        kernel_shape=(32, 16, 3, 3),
        stride=(1, 1),
        padding=(1, 1, 1, 1)
    )


@pytest.fixture
def operation_registry():
    """Provide a fresh operation registry for testing."""
    from extensible_parallelism.core.registry import OperationRegistry
    
    return OperationRegistry()
