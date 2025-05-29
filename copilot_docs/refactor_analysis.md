# Current PE/SIMD Implementation vs Refactor Proposal Analysis

## Overview

This document analyzes the current PE and SIMD implementation in FINN against the proposed refactor outlined in `parallelism_refactor_proposal.md`. It identifies alignment opportunities, gaps, and implementation strategies.

## Current Implementation Analysis

### Existing Parallelism Model

#### Current Parameters (Per Operation)
```python
# Current FINN model - operation-centric
current_model = {
    "PE": "Output parallelism (processing elements)",
    "SIMD": "Input parallelism (single instruction, multiple data)",
    "MW": "Matrix width (input features)",  
    "MH": "Matrix height (output features)",
    "numInputVectors": "Batch/sequence dimension"
}
```

#### Current Calculation Pattern
```python
# MVAU cycle calculation
def get_exp_cycles(self):
    pe = self.get_nodeattr("PE")
    simd = self.get_nodeattr("SIMD") 
    num_inp_vec = self.get_nodeattr("numInputVectors")
    mh = self.get_nodeattr("MH")
    mw = self.get_nodeattr("MW")
    mmv = 1  # Fixed memory macroprocessing value
    
    exp_cycles = (mh / pe) * (mw / simd) * np.prod(num_inp_vec) / mmv
    return int(exp_cycles)
```

## Proposed Refactor Model

### Interface-Centric Approach

#### Proposed Parameters (Per Interface)
```python
# Proposed model - interface-centric  
proposed_model = {
    # Kernel Parameters (universal constants)
    "n_intf": "Number of interfaces",
    
    # Model Parameters (from ONNX)
    "TDIM_i": "Tensor dimensionality (per interface i)",
    "n_batch": "Query batch size", 
    "n_exec_i": "Number of executions (query width)",
    "n_calc_i": "Number of calculations per execution",
    
    # DSE Parameters (design space exploration)
    "SDIM_i": "Stream dimensionality (formerly SIMD)",
    "PE_i": "Parallelism (calculations in parallel)"
}
```

#### Proposed Calculation Pattern
```python
# Interface-level calculations
def calculate_latencies(interface_i):
    L_C_i = np.prod(TDIM_i / SDIM_i)  # Calculation latency
    L_E_i = (n_calc_i / PE_i) * L_C_i  # Execution latency  
    L_Q_i = n_exec_i * L_E_i  # Query (inference) latency
    return L_C_i, L_E_i, L_Q_i

# Kernel-level calculations
def calculate_kernel_metrics(all_interfaces):
    L_E_k = max([L_E_i for all interfaces])
    L_Q_k = max([L_Q_i for all interfaces]) 
    T_k = n_batch / L_Q_k  # Throughput
    return L_E_k, L_Q_k, T_k
```

## Alignment Analysis

### Strengths of Current Implementation

#### 1. **Mature Mathematical Foundation**
```python
# Current model has well-established cycle calculations
# Proven in hardware with extensive validation
mvau_cycles = (MH/PE) * (MW/SIMD) * batch_size
vvau_cycles = ((channels * k_h * k_w) / PE / SIMD) * (dim_h * dim_w)
```

#### 2. **Hardware-Proven Architecture**
```systemverilog
// Direct hardware parameter mapping
module mvu_vvu_axi #(
    int unsigned PE,     // Maps directly to current PE
    int unsigned SIMD,   // Maps directly to current SIMD
    int unsigned MW,     // Matrix dimensions
    int unsigned MH
);
```

#### 3. **Comprehensive Tooling**
```python
# Existing tools that work well
- SetFolding transformation (automated optimization)
- Resource estimation functions
- Performance analysis passes
- Hardware generation pipeline
```

### Gaps Identified

#### 1. **Multi-Interface Operations**
```python
# Current limitation: Single interface assumption
# Example: MatMul with 2 inputs not well modeled
current_matmul = {
    "inputs": ["A[M,N]", "B[N,P]"],  # Two interfaces
    "PE": "Single PE value",         # Should be per-interface
    "SIMD": "Single SIMD value"      # Should be per-interface
}

# Proposed improvement
proposed_matmul = {
    "interface_0": {"SDIM": [s1], "PE": pe1},  # For input A
    "interface_1": {"SDIM": [s2], "PE": pe2}   # For input B
}
```

#### 2. **Inconsistent Tensor Handling**
```python
# Current: Mixed approaches to tensor dimensionality
mvau_approach = {
    "tensors": "Handled via numInputVectors",
    "folding": "2D folding (PE × SIMD)",
    "constraints": "MW % SIMD == 0, MH % PE == 0"
}

conv_approach = {
    "tensors": "Handled via Dim attributes", 
    "folding": "Channel + spatial folding",
    "constraints": "Different per operation type"
}

# Proposed: Unified tensor chunking
unified_approach = {
    "chunking": "All data viewed as list of tensors",
    "folding": "Consistent TDIM/SDIM across operations",
    "constraints": "Uniform divisibility rules"
}
```

#### 3. **Limited Batch Handling**
```python
# Current: Batch conflated with other dimensions
current_batch = {
    "representation": "Embedded in numInputVectors",
    "optimization": "Not explicitly modeled",
    "constraints": "Implicit in tensor shapes"
}

# Proposed: Explicit batch modeling
proposed_batch = {
    "representation": "Separate n_batch parameter",
    "optimization": "Explicit batch-aware optimization", 
    "throughput": "T_k = n_batch / L_Q_k"
}
```

## Migration Strategy

### Phase 1: Extend Current Model (Low Risk)

#### Add Interface Awareness
```python
class HWCustomOp_Extended(HWCustomOp):
    def get_num_interfaces(self):
        """Number of input interfaces for this operation"""
        return len(self.onnx_node.input)
    
    def get_interface_parallelism(self, interface_idx):
        """Get PE/SIMD for specific interface"""
        if interface_idx == 0:
            return self.get_nodeattr("PE"), self.get_nodeattr("SIMD")
        else:
            # Extended model: per-interface parameters
            return (self.get_nodeattr(f"PE_{interface_idx}"), 
                   self.get_nodeattr(f"SIMD_{interface_idx}"))
```

#### Unified Cycle Calculation
```python
def get_exp_cycles_unified(self):
    """Unified cycle calculation using interface model"""
    interface_latencies = []
    
    for i in range(self.get_num_interfaces()):
        # Map current parameters to proposed model
        TDIM_i = self.get_tensor_dimensions(i)
        SDIM_i = self.get_stream_dimensions(i)  # From SIMD
        n_calc_i = self.get_num_calculations(i)
        PE_i = self.get_interface_parallelism(i)[0]
        
        # Calculate using proposed formulas
        L_C_i = np.prod(TDIM_i / SDIM_i)
        L_E_i = (n_calc_i / PE_i) * L_C_i
        interface_latencies.append(L_E_i)
    
    # Return maximum (bottleneck)
    return int(max(interface_latencies))
```

### Phase 2: Implement Tensor Chunking (Medium Risk)

#### Tensor Dimension Mapping
```python
def map_onnx_to_tensor_chunks(self, onnx_shape, layout_hint):
    """Map ONNX tensor shapes to proposed chunking model"""
    
    layout_mappings = {
        "[N,C,H,W]": {
            "query_width": "C",           # n_exec
            "tensor_dims": "[N*H*W]",     # TDIM
            "batch_size": "1 or N"       # n_batch
        },
        "[N,L,C]": {
            "query_width": "C", 
            "tensor_dims": "[N*L]",
            "batch_size": "1 or N"
        }
    }
    
    return layout_mappings.get(layout_hint, {})
```

#### Stream Width Unification
```python
def calculate_stream_widths_unified(self):
    """Calculate stream widths using unified model"""
    stream_widths = {}
    
    for i in range(self.get_num_interfaces()):
        SDIM_i = self.get_stream_dimensions(i)
        datatype = self.get_input_datatype(i)
        
        # Unified calculation
        stream_widths[f"interface_{i}"] = np.prod(SDIM_i) * datatype.bitwidth()
    
    return stream_widths
```

### Phase 3: Full Refactor (High Risk, High Reward)

#### New Base Class Architecture
```python
class HWKernel_Refactored:
    """Refactored HW kernel following proposed model"""
    
    def __init__(self, onnx_node):
        # Parse kernel parameters from RTL/HLS
        self.kernel_params = self.parse_kernel_params()
        
        # Parse model parameters from ONNX
        self.model_params = self.parse_model_params(onnx_node)
        
        # Initialize DSE parameters
        self.dse_params = self.initialize_dse_params()
    
    def parse_kernel_params(self):
        """Extract universal constants from kernel definition"""
        return {
            "n_intf": self.get_num_interfaces(),
            "supported_datatypes": self.get_supported_datatypes(),
            "resource_model": self.get_resource_model()
        }
    
    def parse_model_params(self, onnx_node):
        """Extract model-specific parameters from ONNX"""
        params = {}
        for i in range(self.kernel_params["n_intf"]):
            params[f"interface_{i}"] = {
                "TDIM": self.extract_tensor_dims(onnx_node, i),
                "n_exec": self.extract_query_width(onnx_node, i),
                "n_calc": self.extract_num_calculations(onnx_node, i)
            }
        params["n_batch"] = self.extract_batch_size(onnx_node)
        return params
    
    def calculate_performance_metrics(self):
        """Calculate performance using proposed formulas"""
        interface_metrics = {}
        
        for i in range(self.kernel_params["n_intf"]):
            TDIM_i = self.model_params[f"interface_{i}"]["TDIM"]
            SDIM_i = self.dse_params[f"interface_{i}"]["SDIM"]
            n_calc_i = self.model_params[f"interface_{i}"]["n_calc"] 
            PE_i = self.dse_params[f"interface_{i}"]["PE"]
            n_exec_i = self.model_params[f"interface_{i}"]["n_exec"]
            
            # Proposed calculations
            L_C_i = np.prod(np.array(TDIM_i) / np.array(SDIM_i))
            L_E_i = (n_calc_i / PE_i) * L_C_i
            L_Q_i = n_exec_i * L_E_i
            
            interface_metrics[f"interface_{i}"] = {
                "L_C": L_C_i, "L_E": L_E_i, "L_Q": L_Q_i
            }
        
        # Kernel-level metrics
        L_E_k = max([m["L_E"] for m in interface_metrics.values()])
        L_Q_k = max([m["L_Q"] for m in interface_metrics.values()])
        T_k = self.model_params["n_batch"] / L_Q_k
        
        return {
            "kernel_metrics": {"L_E": L_E_k, "L_Q": L_Q_k, "T": T_k},
            "interface_metrics": interface_metrics
        }
```

## Implementation Recommendations

### Near-Term (6 months)
1. **Extend existing HWCustomOp** with interface-aware methods
2. **Add unified cycle calculation** as alternative to current methods
3. **Implement tensor chunking utilities** for common layouts
4. **Create compatibility layer** between old and new models

### Medium-Term (12 months) 
1. **Refactor SetFolding** to use interface-centric optimization
2. **Update resource estimation** with per-interface modeling
3. **Enhance hardware generation** with unified parameter mapping
4. **Comprehensive testing** of both models in parallel

### Long-Term (18+ months)
1. **Full migration** to interface-centric model
2. **Deprecate legacy** PE/SIMD handling where possible
3. **Advanced optimization** leveraging multi-interface awareness
4. **Documentation and training** for new model usage

## Risk Assessment

### Technical Risks
| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing designs | High | Medium | Maintain compatibility layer |
| Performance regression | Medium | Low | Extensive validation testing |
| Increased complexity | Medium | High | Gradual migration, good documentation |
| Hardware generation issues | High | Low | Parallel development and testing |

### Benefits Assessment
| Benefit | Impact | Confidence | Timeline |
|---------|--------|------------|----------|
| Multi-interface support | High | High | Medium-term |
| Unified tensor handling | Medium | High | Near-term |
| Better batch modeling | Medium | Medium | Medium-term |
| Simplified maintenance | High | Medium | Long-term |

## Conclusion

The proposed refactor represents a natural evolution of FINN's parallelism model that addresses real limitations in the current implementation. The interface-centric approach provides better support for multi-input operations, cleaner tensor handling, and more explicit batch modeling.

The migration can be accomplished incrementally with low risk by:
1. **Extending** the current model rather than replacing it
2. **Adding** interface awareness gradually
3. **Maintaining** backward compatibility throughout transition
4. **Validating** each phase thoroughly before proceeding

The key insight is that the current PE/SIMD model is fundamentally sound and well-proven - the refactor should build upon this foundation rather than replacing it entirely. The proposed interface-centric view can coexist with and enhance the current model, providing a path for future optimization while preserving existing functionality.
