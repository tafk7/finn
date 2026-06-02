# Kernel Specialization Guide

## Overview

FINN supports flexible backend specialization through `step_specialize_layers`, which operates sequentially in two phases:

1. **Explicit Specialization** (when `kernel_selections` is set): Priority-based backend selection for specified kernels with automatic fallback
2. **Automatic Specialization** (always runs): Built-in heuristics specialize remaining generic nodes

This sequential approach allows combining explicit priority control for high-value kernels while ensuring all other nodes are specialized automatically.

## Key Architectural Changes

### Backend Attribute Semantics

The `backend` node attribute now indicates implementation style rather than just being a marker:

- `backend="fpgadataflow"`: Generic HWCustomOp (not yet specialized)
- `backend="hls"`: HLS-specialized implementation
- `backend="rtl"`: RTL-specialized implementation

### How Specialization Works

When specializing with `SpecializeKernel`:
1. Takes a base kernel class (e.g., `MVAU`) and variant classes (e.g., `[MVAU_rtl, MVAU_hls]`)
2. Finds nodes with matching base kernel name (`op_type="MVAU"`)
3. Tries each variant in priority order, checking constraints
4. Updates the node to use the selected variant's:
   - **op_type**: Changes to variant name (e.g., `MVAU_rtl`)
   - **domain**: Changes to variant's registered domain (e.g., `finn.custom_op.fpgadataflow.rtl`)
   - **backend**: Changes to backend style (e.g., `"rtl"`)

| Aspect | Before Specialization | After Specialization |
|--------|----------------------|---------------------|
| Domain | `finn.custom_op.fpgadataflow` | `finn.custom_op.fpgadataflow.rtl` |
| Backend | `"fpgadataflow"` | `"rtl"` |
| Op Type | `MVAU` | `MVAU_rtl` |

## Using Kernel Specialization

### Configuration

Add `kernel_selections` to your `DataflowBuildConfig`:

```python
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU
from finn.custom_op.fpgadataflow.hls.vectorvectoractivation_hls import VVAU_hls
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import StreamingDataWidthConverter
from finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls import StreamingDataWidthConverter_hls
from finn.custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl import StreamingDataWidthConverter_rtl

cfg = DataflowBuildConfig(
    output_dir="output",
    synth_clk_period_ns=5.0,
    board="ZCU104",
    generate_outputs=[...],
    # Specify kernel specialization preferences using class references
    kernel_selections=[
        (MVAU, [MVAU_rtl, MVAU_hls]),    # Try RTL first, fall back to HLS
        (VVAU, [VVAU_hls]),               # Only use HLS
        (StreamingDataWidthConverter, [StreamingDataWidthConverter_rtl, StreamingDataWidthConverter_hls]),  # Prefer RTL
    ]
)
```

### Build Steps

The `step_specialize_layers` step runs sequentially:

```python
from finn.builder.build_dataflow_config import default_build_dataflow_steps

# Option 1: Use default steps (includes step_specialize_layers)
cfg.steps = default_build_dataflow_steps

# Option 2: Custom step order
cfg.steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",        # Phase 1: explicit (if configured) + Phase 2: automatic
    "step_target_fps_parallelization",
    # ... rest of steps
]
```

**Phase 1** (if `kernel_selections` is set): Explicit priority-based selection for specified kernels
**Phase 2** (always runs): Automatic specialization for remaining generic nodes

## Priority-Based Backend Selection

The `SpecializeKernel` transformation tries backend variants in the order specified and selects the first one that meets all constraints:

### Example: MVAU Specialization

```python
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl

kernel_selections = [(MVAU, [MVAU_rtl, MVAU_hls])]
```

The transform will:

1. **Try RTL first**:
   - Check if `MVAU_rtl` exists in domain
   - Check RTL constraints:
     - `noActivation == 1` (no embedded thresholds)
     - Input/weight bitwidths 2-8 bits (or 9-bit signed for DSP58)
     - Narrow weights for DSP48E1
   - If all pass → select RTL, set `backend="rtl"`, `op_type="MVAU_rtl"`

2. **Fall back to HLS** if RTL constraints not met:
   - Check if `MVAU_hls` exists
   - HLS is more flexible, usually succeeds
   - Set `backend="hls"`, `op_type="MVAU_hls"`

3. **Warn if neither works**:
   - Node stays generic (`backend="fpgadataflow"`)
   - Warning emitted for user

## Constraint Checking

### MVAU RTL Constraints

- No embedded activation (`noActivation == 1`)
- No binary XNOR mode (`binaryXnorMode == 0`)
- Input datatype: 2-8 bits (or 9-bit signed)
- Weight datatype: 2-8 bits, signed
- Narrow weights if DSP48E1 is the only available DSP

### VVAU RTL Constraints

- No embedded activation
- Versal platform only (requires DSP58)
- Input: ≤8 bits or 9-bit signed
- Weight: ≤8 bits, signed

### StreamingDataWidthConverter RTL Constraints

- Integer width ratios: `inWidth % outWidth == 0` OR `outWidth % inWidth == 0`

## Multiple Backends for One Kernel

You can register multiple backend implementations for the same kernel in custom domains:

```python
# In your custom kernel module structure:
brainsmith/
  kernels/
    matmul.py         # Generic MatMul (backend="fpgadataflow")
    hls/
      matmul_hls.py   # HLS variant (backend="hls")
    rtl/
      matmul_rtl.py   # RTL variant (backend="rtl")
    dsp/
      matmul_dsp.py   # DSP-optimized variant (backend="dsp")
```

Then configure using class references:

```python
from brainsmith.kernels.matmul import MatMul
from brainsmith.kernels.dsp.matmul_dsp import MatMul_dsp
from brainsmith.kernels.rtl.matmul_rtl import MatMul_rtl
from brainsmith.kernels.hls.matmul_hls import MatMul_hls

cfg.kernel_selections = [
    (MatMul, [MatMul_dsp, MatMul_rtl, MatMul_hls]),  # Try custom DSP first, then RTL, then HLS
]
```

## Backwards Compatibility

The new system maintains full backwards compatibility:

### Legacy Mode (Domain-Based)

Nodes with `backend="fpgadataflow"` and domains like `finn.custom_op.fpgadataflow.hls` are still recognized by `is_hls_node()` and `is_rtl_node()`.

### Modern Mode (Attribute-Based)

Nodes with `backend="hls"` or `backend="rtl"` are recognized regardless of domain.

### Detection Priority

Helper functions check `backend` attribute first, then fall back to domain checking:

```python
def is_hls_node(node):
    # Check backend attribute first (modern)
    if backend_value == "hls":
        return True
    # Fall back to domain (legacy)
    elif backend_value == "fpgadataflow":
        if node.domain.endswith(".hls"):
            return True
```

## Comparison with SpecializeLayers

| Feature | SpecializeLayers | SpecializeKernel |
|---------|------------------|------------------|
| Scope | All eligible nodes | Specific kernel types |
| Backend Selection | Automatic heuristics | Priority list with class references |
| Domain Changes | Yes (adds `.hls`/`.rtl`) | Yes (uses variant's domain) |
| Multiple Backends | No | Yes (tries in order) |
| Custom Domains | Limited support | Full support |
| Use Case | Standard FINN layers | Priority-based selection |
| Type Safety | String-based | Class-based |

## Example: End-to-End Build

```python
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    ShellFlowType,
)
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.hls.matrixvectoractivation_hls import MVAU_hls
from finn.custom_op.fpgadataflow.rtl.matrixvectoractivation_rtl import MVAU_rtl
from finn.custom_op.fpgadataflow.vectorvectoractivation import VVAU
from finn.custom_op.fpgadataflow.hls.vectorvectoractivation_hls import VVAU_hls
from finn.custom_op.fpgadataflow.rtl.vectorvectoractivation_rtl import VVAU_rtl
from finn.custom_op.fpgadataflow.streamingdatawidthconverter import StreamingDataWidthConverter
from finn.custom_op.fpgadataflow.rtl.streamingdatawidthconverter_rtl import StreamingDataWidthConverter_rtl
# Import your custom kernels
from brainsmith.kernels.custom_kernel import CustomKernel
from brainsmith.kernels.dsp.custom_kernel_dsp import CustomKernel_dsp
from brainsmith.kernels.hls.custom_kernel_hls import CustomKernel_hls

# Configuration
cfg = DataflowBuildConfig(
    output_dir="output_dir",
    synth_clk_period_ns=5.0,
    board="ZCU104",
    shell_flow_type=ShellFlowType.VIVADO_ZYNQ,
    generate_outputs=[
        DataflowOutputType.BITFILE,
        DataflowOutputType.PYNQ_DRIVER,
    ],
    # Kernel-specific backend preferences using class references
    kernel_selections=[
        (MVAU, [MVAU_rtl, MVAU_hls]),                     # Prefer RTL for MVAUs
        (VVAU, [VVAU_rtl, VVAU_hls]),                     # Prefer RTL for VVAUs
        (StreamingDataWidthConverter, [StreamingDataWidthConverter_rtl]),  # RTL only for DWCs
        (CustomKernel, [CustomKernel_dsp, CustomKernel_hls]),  # Custom backend
    ],
    # Other settings
    target_fps=1000,
    auto_fifo_depths=True,
)

# Build
model = build_dataflow_cfg("model.onnx", cfg)
```

## Debugging

### Check Backend Assignments

After specialization, inspect nodes to see which backends were selected:

```python
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import get_by_name

model = ModelWrapper("intermediate_models/step_specialize_layers.onnx")

for node in model.graph.node:
    backend_attr = get_by_name(node.attribute, "backend")
    if backend_attr:
        backend_value = backend_attr.s.decode("UTF-8")
        print(f"{node.name} ({node.op_type}): backend={backend_value}, domain={node.domain}")
```

### Enable Warnings

Specialization warnings are emitted when:
- A requested backend variant doesn't exist
- No backends in the priority list met constraints
- Falling back from preferred backend to alternative

Check the build log for these warnings.

## Migration Guide

### From SpecializeLayers Only

If you're currently using only `step_specialize_layers`:

**No changes needed** - your builds will continue to work as before.

### Adding Kernel Specialization

To start using kernel-specific specialization:

1. Add `kernel_selections` to your config with class references
2. Keep `step_specialize_layers` in your build steps (no changes needed)
3. The step will run sequentially:
   - **Phase 1**: Explicit priority-based selection for kernels in `kernel_selections`
   - **Phase 2**: Automatic specialization for all remaining generic nodes
4. Result: High-value kernels get priority control, everything else gets specialized automatically

### Custom Kernels in Non-Standard Domains

If you have kernels in custom domains (e.g., `brainsmith.kernels.*`):

1. Ensure your HLS/RTL variants are registered with appropriate domains
2. Use `kernel_selections` to specify backend preferences
3. The domain will remain unchanged; only `backend` attribute and `op_type` will change

## Summary

The kernel specialization system provides:

- **Flexibility**: Support for kernels in any domain
- **Control**: Explicit priority-based backend selection via `kernel_selections`
- **Completeness**: Sequential phases ensure all nodes get specialized
- **Extensibility**: Easy to add custom backends
- **Compatibility**: Full backwards compatibility with existing flows
- **Simplicity**: Single build step with two-phase sequential execution

Use `kernel_selections` to specify priority order for high-value kernels (e.g., RTL-first for MVAU). All other kernels automatically specialized with built-in heuristics. No nodes left behind.
