#!/usr/bin/env python3
"""
Demonstration of the completed Phase 1 Extensible Parallelism System
"""

import sys
import time
sys.path.insert(0, 'src')

from extensible_parallelism.operations.matrix_vector import MatrixVectorOperation, MatrixVectorParams
from extensible_parallelism.operations.element_wise import ElementWiseOperation, ElementWiseParams, ElementWiseOpType
from extensible_parallelism.operations.convolution import ConvolutionOperation, ConvolutionParams
from extensible_parallelism.core.optimizer import (
    GridSearchOptimizer, HeuristicOptimizer, 
    OptimizationConstraints, OptimizationObjective
)

def demo_matrix_vector():
    """Demonstrate MatrixVector operation optimization"""
    print("🔹 Matrix-Vector Operation Demo")
    print("=" * 50)
    
    # Create a typical MVAU-style operation
    params = MatrixVectorParams(matrix_height=512, matrix_width=256)
    operation = MatrixVectorOperation("demo_mvau", params)
    
    print(f"Operation: {operation}")
    print(f"Input tensors: {list(operation.get_input_tensors().keys())}")
    print(f"Output tensors: {list(operation.get_output_tensors().keys())}")
    
    # Optimize with resource constraints
    constraints = OptimizationConstraints(max_dsps=100, max_brams=50)
    optimizer = GridSearchOptimizer(
        objective=OptimizationObjective.BALANCED,
        constraints=constraints,
        max_parallelism_factor=8,
        max_evaluations=100
    )
    
    start_time = time.time()
    result = optimizer.optimize(operation)
    optimization_time = time.time() - start_time
    
    print(f"\n✅ Optimization Result:")
    print(f"  Score: {result.best_score:.3f}")
    print(f"  DSPs used: {result.resource_estimate.dsps}")
    print(f"  BRAMs used: {result.resource_estimate.brams}")
    print(f"  LUTs used: {result.resource_estimate.luts}")
    print(f"  Latency: {result.resource_estimate.latency_cycles} cycles")
    print(f"  Configurations evaluated: {result.configurations_evaluated}")
    print(f"  Optimization time: {optimization_time:.3f}s")
    print()

def demo_element_wise():
    """Demonstrate ElementWise operation optimization"""
    print("🔹 Element-Wise Operation Demo")
    print("=" * 50)
    
    # Create a RELU activation layer
    params = ElementWiseParams(
        operation_type=ElementWiseOpType.RELU,
        tensor_shape=(1024, 512),
        num_inputs=1
    )
    operation = ElementWiseOperation("demo_relu", params)
    
    print(f"Operation: {operation}")
    print(f"Operation type: {params.operation_type}")
    print(f"Tensor shape: {params.tensor_shape}")
    
    # Optimize for maximum throughput
    optimizer = HeuristicOptimizer(
        objective=OptimizationObjective.MAXIMIZE_THROUGHPUT,
        constraints=OptimizationConstraints(max_dsps=50)
    )
    
    start_time = time.time()
    result = optimizer.optimize(operation)
    optimization_time = time.time() - start_time
    
    print(f"\n✅ Optimization Result:")
    print(f"  Score: {result.best_score:.3f}")
    print(f"  Resources: DSPs={result.resource_estimate.dsps}, LUTs={result.resource_estimate.luts}")
    print(f"  Latency: {result.resource_estimate.latency_cycles} cycles")
    print(f"  Optimization time: {optimization_time:.3f}s")
    print()

def demo_convolution():
    """Demonstrate Convolution operation optimization"""
    print("🔹 Convolution Operation Demo")
    print("=" * 50)
    
    # Create a typical CNN layer
    params = ConvolutionParams(
        input_shape=(1, 64, 56, 56),    # Batch=1, Channels=64, H=56, W=56
        kernel_shape=(128, 64, 3, 3),   # OutCh=128, InCh=64, KH=3, KW=3
        stride=(1, 1),
        padding=(1, 1, 1, 1)
    )
    operation = ConvolutionOperation("demo_conv", params)
    
    print(f"Operation: {operation}")
    print(f"Input shape: {params.input_shape}")
    print(f"Kernel shape: {params.kernel_shape}")
    print(f"Output shape: {operation.get_output_tensors()['output'].shape}")
    
    # Optimize with generous resource budget
    constraints = OptimizationConstraints(max_dsps=200, max_brams=100)
    optimizer = GridSearchOptimizer(
        objective=OptimizationObjective.MINIMIZE_LATENCY,
        constraints=constraints,
        max_parallelism_factor=4,  # Limited for demo speed
        max_evaluations=50
    )
    
    start_time = time.time()
    result = optimizer.optimize(operation)
    optimization_time = time.time() - start_time
    
    print(f"\n✅ Optimization Result:")
    print(f"  Score: {result.best_score:.3f}")
    print(f"  Resource usage: DSPs={result.resource_estimate.dsps}, BRAMs={result.resource_estimate.brams}")
    print(f"  Latency: {result.resource_estimate.latency_cycles} cycles")
    print(f"  Configurations evaluated: {result.configurations_evaluated}")
    print(f"  Optimization time: {optimization_time:.3f}s")
    print()

def demo_comparison():
    """Compare grid search vs heuristic optimization"""
    print("🔹 Optimizer Comparison Demo")
    print("=" * 50)
    
    # Same operation for both optimizers
    params = MatrixVectorParams(matrix_height=128, matrix_width=64)
    operation = MatrixVectorOperation("comparison_test", params)
    
    constraints = OptimizationConstraints(max_dsps=80, max_brams=40)
    
    # Grid search optimizer
    grid_optimizer = GridSearchOptimizer(
        objective=OptimizationObjective.BALANCED,
        constraints=constraints,
        max_parallelism_factor=4,
        max_evaluations=50
    )
    
    start_time = time.time()
    grid_result = grid_optimizer.optimize(operation)
    grid_time = time.time() - start_time
    
    # Heuristic optimizer
    heuristic_optimizer = HeuristicOptimizer(
        objective=OptimizationObjective.BALANCED,
        constraints=constraints
    )
    
    start_time = time.time()
    heuristic_result = heuristic_optimizer.optimize(operation)
    heuristic_time = time.time() - start_time
    
    print(f"Grid Search Results:")
    print(f"  Score: {grid_result.best_score:.3f}")
    print(f"  Configs evaluated: {grid_result.configurations_evaluated}")
    print(f"  Time: {grid_time:.3f}s")
    
    print(f"\nHeuristic Results:")
    print(f"  Score: {heuristic_result.best_score:.3f}")
    print(f"  Configs evaluated: {heuristic_result.configurations_evaluated}")
    print(f"  Time: {heuristic_time:.3f}s")
    
    print(f"\n📊 Comparison:")
    print(f"  Speedup: {grid_time/heuristic_time:.1f}x faster with heuristic")
    print(f"  Efficiency: {heuristic_result.configurations_evaluated}/{grid_result.configurations_evaluated} evaluations")
    print()

def main():
    print("🚀 Extensible Parallelism System - Phase 1 Demo")
    print("=" * 60)
    print("Demonstrating completed core framework with optimization\n")
    
    try:
        demo_matrix_vector()
        demo_element_wise()
        demo_convolution()
        demo_comparison()
        
        print("🎉 All demos completed successfully!")
        print("Phase 1 is ready for production use and Phase 2 development!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
