#!/usr/bin/env python3
"""
Example demonstrating the Extensible Parallelism System.

This script shows how to:
1. Create operations using the registry
2. Configure parallelism settings
3. Estimate resource usage
4. Validate configurations
5. Apply constraints
"""

import sys
import os

# Add src to path for running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from extensible_parallelism import (
    OperationRegistry,
    MatrixVectorOperation,
    ElementWiseOperation,
    ConvolutionOperation,
    ParallelismConfig,
    TensorParallelism,
    ParallelismStrategy,
    MemoryStrategy,
    DivisibilityConstraint,
    ResourceConstraint
)

from extensible_parallelism.operations.matrix_vector import MatrixVectorParams
from extensible_parallelism.operations.element_wise import ElementWiseParams, ElementWiseOpType
from extensible_parallelism.operations.convolution import ConvolutionParams


def main():
    """Main example function."""
    print("=== FINN Extensible Parallelism System Demo ===\n")
    
    # 1. Show available operations
    print("1. Available Operations:")
    registry = OperationRegistry()
    operations = registry.list_operations()
    
    for op_info in operations:
        print(f"   - {op_info['name']}: {op_info['category']}")
    print()
    
    # 2. Create a MatrixVector operation
    print("2. Creating MatrixVector Operation:")
    mv_params = MatrixVectorParams(
        matrix_height=256,
        matrix_width=128,
        input_dtype="int8",
        weight_dtype="int8",
        output_dtype="int32",
        bias_dtype="int32"
    )
    
    mv_op = MatrixVectorOperation(mv_params)
    print(f"   Operation: {mv_op}")
    
    # Show tensor specifications
    print("   Input tensors:")
    for name, spec in mv_op.get_input_tensors().items():
        print(f"     {name}: {spec.shape} ({spec.dtype})")
    
    print("   Output tensors:")
    for name, spec in mv_op.get_output_tensors().items():
        print(f"     {name}: {spec.shape} ({spec.dtype})")
    print()
    
    # 3. Configure parallelism
    print("3. Configuring Parallelism:")
    
    # Default configuration
    default_config = mv_op.get_default_parallelism_config()
    print("   Default configuration (PE=1, SIMD=1):")
    resources_default = mv_op.estimate_resources(default_config)
    print(f"     Resources: {resources_default.dsps} DSPs, {resources_default.brams} BRAMs, "
          f"{resources_default.luts} LUTs, {resources_default.ffs} FFs")
    
    # High parallelism configuration
    high_parallel_config = ParallelismConfig(
        input_tensors={
            "input": TensorParallelism(
                dimension_factors={"input_features": 8},  # SIMD=8
                strategy=ParallelismStrategy.PARALLEL,
                memory_strategy=MemoryStrategy.STREAMING
            ),
            "weights": TensorParallelism(
                dimension_factors={"output_features": 16, "input_features": 8},  # PE=16, SIMD=8
                strategy=ParallelismStrategy.PARALLEL,
                memory_strategy=MemoryStrategy.BLOCK_RAM
            ),
            "bias": TensorParallelism(
                dimension_factors={"output_features": 16},  # PE=16
                strategy=ParallelismStrategy.PARALLEL,
                memory_strategy=MemoryStrategy.BLOCK_RAM
            )
        },
        output_tensors={
            "output": TensorParallelism(
                dimension_factors={"output_features": 16},  # PE=16
                strategy=ParallelismStrategy.PARALLEL,
                memory_strategy=MemoryStrategy.STREAMING
            )
        },
        internal_tensors={
            "accumulator": TensorParallelism(
                dimension_factors={"output_features": 16},  # PE=16
                strategy=ParallelismStrategy.PARALLEL,
                memory_strategy=MemoryStrategy.REGISTERS
            )
        }
    )
    
    print("   High parallelism configuration (PE=16, SIMD=8):")
    resources_high = mv_op.estimate_resources(high_parallel_config)
    print(f"     Resources: {resources_high.dsps} DSPs, {resources_high.brams} BRAMs, "
          f"{resources_high.luts} LUTs, {resources_high.ffs} FFs")
    print()
    
    # 4. Constraint validation
    print("4. Constraint Validation:")
    
    # Create constraints
    div_constraint = DivisibilityConstraint()
    resource_constraint = ResourceConstraint(max_dsps=64, max_bram=512, max_luts=50000, max_ffs=50000)
    
    constraints = [div_constraint, resource_constraint]
    
    # Validate configurations
    print("   Validating default configuration:")
    violations_default = []
    for constraint in constraints:
        violation = constraint.validate(mv_op, default_config)
        if violation:
            violations_default.append(violation)
    
    if violations_default:
        print("     Violations found:")
        for violation in violations_default:
            print(f"       - {violation.message}")
    else:
        print("     ✓ Configuration is valid")
    
    print("   Validating high parallelism configuration:")
    violations_high = []
    for constraint in constraints:
        violation = constraint.validate(mv_op, high_parallel_config)
        if violation:
            violations_high.append(violation)
    
    if violations_high:
        print("     Violations found:")
        for violation in violations_high:
            print(f"       - {violation.message}")
            suggestions = violation.get_suggestions()
            if suggestions:
                print(f"       Suggestions: {suggestions}")
    else:
        print("     ✓ Configuration is valid")
    print()
    
    # 5. Element-wise operation example
    print("5. Element-wise Operation Example:")
    
    ew_params = ElementWiseParams(
        operation_type=ElementWiseOpType.RELU,
        tensor_shape=(1024, 512),
        input_dtype="int16",
        output_dtype="int16",
        num_inputs=1
    )
    
    ew_op = ElementWiseOperation(ew_params)
    print(f"   Operation: {ew_op}")
    
    ew_config = ew_op.get_default_parallelism_config()
    ew_resources = ew_op.estimate_resources(ew_config)
    print(f"   Resources: {ew_resources.dsps} DSPs, {ew_resources.brams} BRAMs, "
          f"{ew_resources.luts} LUTs, {ew_resources.ffs} FFs")
    print()
    
    # 6. Convolution operation example
    print("6. Convolution Operation Example:")
    
    conv_params = ConvolutionParams(
        input_shape=(64, 224, 224),  # (C, H, W)
        kernel_shape=(128, 64, 3, 3),  # (C_out, C_in, K_h, K_w)
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        input_dtype="int8",
        weight_dtype="int8",
        output_dtype="int32"
    )
    
    conv_op = ConvolutionOperation(conv_params)
    print(f"   Operation: {conv_op}")
    print(f"   Output shape: {conv_op.output_shape}")
    
    conv_config = conv_op.get_default_parallelism_config()
    conv_resources = conv_op.estimate_resources(conv_config)
    print(f"   Resources: {conv_resources.dsps} DSPs, {conv_resources.brams} BRAMs, "
          f"{conv_resources.luts} LUTs, {conv_resources.ffs} FFs")
    print()
    
    # 7. Registry-based operation creation
    print("7. Registry-based Operation Creation:")
    
    # Create operations through registry
    registry_mv_op = registry.create_operation("matrix_vector", params=mv_params)
    registry_ew_op = registry.create_operation("element_wise", params=ew_params)
    registry_conv_op = registry.create_operation("convolution", params=conv_params)
    
    print("   Operations created through registry:")
    print(f"     MatrixVector: {type(registry_mv_op).__name__}")
    print(f"     ElementWise: {type(registry_ew_op).__name__}")
    print(f"     Convolution: {type(registry_conv_op).__name__}")
    print()
    
    # 8. Parallelism bounds
    print("8. Parallelism Bounds:")
    
    for op_name, op in [("MatrixVector", mv_op), ("ElementWise", ew_op), ("Convolution", conv_op)]:
        bounds = op.get_parallelism_bounds()
        print(f"   {op_name}:")
        for param, (min_val, max_val) in bounds.items():
            print(f"     {param}: [{min_val}, {max_val}]")
    print()
    
    print("=== Demo Complete ===")


if __name__ == "__main__":
    main()
