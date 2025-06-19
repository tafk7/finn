# FINN Simplified Templates

This directory contains simplified, self-contained templates for FINN code generation. These templates are designed to be easy to understand and modify by open source contributors.

## Templates

### 1. `hls_basic.cpp.j2`
Basic HLS template for simple operations. Features:
- Direct variable substitution
- No complex macros or includes
- Clear structure: includes → defines → function
- Suitable for operations like Thresholding, Activation functions

**Key Variables:**
- `function_name`: Name of the compute function
- `input_ports`, `output_ports`: Stream port definitions
- `pragmas`: HLS optimization directives
- `compute_body`: Main computation code

### 2. `thresholding_rtl.v.j2`
RTL template specifically for thresholding operations. Features:
- Simple Verilog wrapper
- Direct parameter mapping
- Optional AXI-Lite interface for runtime configuration
- Instantiates the finn-rtllib thresholding core

**Key Variables:**
- `module_name`: Top module name
- `n`, `wi`, `wt`, `c`, `pe`: Core parameters
- `use_axilite`: Enable AXI-Lite interface
- Stream width calculations handled automatically

### 3. `hls_improved.cpp.j2`
Enhanced HLS template for complex operations. Features:
- Better code organization with sections
- Support for weights, types, and constants
- Optional testbench generation
- Suitable for MVAU, Convolution, Pooling

**Key Variables:**
- All from basic template plus:
- `weight_precision`, `accumulator_precision`: Custom types
- `array_partitions`: HLS array optimizations
- `inner_loops`: Nested loop structures
- `generate_wrapper`: IP export wrapper

## Usage Example

```python
# Simple operation
template_values = {
    'function_name': 'my_threshold',
    'input_ports': [{'name': 'in', 'type': 'ap_uint<8>'}],
    'output_ports': [{'name': 'out', 'type': 'ap_uint<4>'}],
    'pragmas': ['#pragma HLS PIPELINE II=1'],
    'compute_body': 'out.write(threshold_function(in.read()));'
}

# Render template
code = template_engine.render('hls_basic.cpp.j2', template_values)
```

## Guidelines for Contributors

1. **Keep it Simple**: These templates are intentionally simple. Avoid adding complexity.
2. **Self-Contained**: Each template should be understandable without referencing other files.
3. **Clear Variables**: Use descriptive variable names, not abstract placeholders.
4. **Document**: Add comments in templates explaining non-obvious sections.
5. **Test**: Always test generated code before committing template changes.

## Migration from Complex Templates

If you need features from the old complex template system:
1. Check if the feature is really needed
2. If yes, add it directly to the template (no includes/macros)
3. Consider if it should be in basic or improved template
4. Document the addition in this README