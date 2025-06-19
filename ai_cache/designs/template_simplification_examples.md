# Template Simplification Examples

**Purpose**: Show concrete examples of how to simplify FINN templates for better usability

## Example 1: Thresholding Operation

### Current Complex Approach

**File structure required:**
```
templates/
├── base/hls_base.cpp.j2
├── components/includes/operation_specific.j2
├── components/includes/standard_hls.j2
├── components/streams/declarations.j2
├── components/hls/pragmas.j2
└── thresholding/hls/docompute.cpp.j2
```

**Main template** (`thresholding/hls/docompute.cpp.j2`):
```jinja2
{% extends "base/hls_base.cpp.j2" %}
{% block operation_specific %}
{% include "components/includes/thresholding_includes.j2" %}
{% endblock %}
```

**Base template** (`base/hls_base.cpp.j2`):
```jinja2
{% include 'components/includes/standard_hls.j2' %}
{% block operation_specific %}{% endblock %}
{{ defines }}
{% include 'components/streams/declarations.j2' %}
int main() {
    {{ pragmas }}
    {{ stream_declarations }}
    {{ do_compute }}
}
```

**Total lines across files**: ~200+
**Files to understand**: 6+

### Simplified Approach

**File structure required:**
```
templates/
└── thresholding_hls.cpp
```

**Single complete template**:
```cpp
// Thresholding HLS Implementation
// Node: {{ node_name }}
// Channels: {{ num_channels }}, PE: {{ pe }}

#define AP_INT_MAX_W {{ ap_int_max_w }}
#include "ap_int.h"
#include "hls_stream.h"
#include "thresholding.hpp"

void {{ node_name }}(
    hls::stream<ap_uint<{{ input_width }}>> &in0,
    hls::stream<ap_uint<{{ output_width }}>> &out0
) {
    #pragma HLS INTERFACE axis port=in0
    #pragma HLS INTERFACE axis port=out0
    #pragma HLS INTERFACE ap_ctrl_none port=return
    
    // Thresholds memory
    {{ threshold_type }} thresholds[{{ num_channels }}/{{ pe }}][{{ pe }}][{{ num_thresholds }}];
    #pragma HLS ARRAY_PARTITION variable=thresholds complete dim=2
    {% if ram_style == "distributed" %}
    #pragma HLS BIND_STORAGE variable=thresholds type=RAM_2P impl=LUTRAM
    {% endif %}
    
    // Initialize thresholds
    #include "thresholds_{{ node_name }}.hpp"
    
    // Main processing
    Thresholding_Batch<{{ num_channels }}, {{ pe }}, {{ num_thresholds }}>(
        in0, out0, thresholds, {{ num_samples }}
    );
}
```

**Total lines**: ~30
**Files to understand**: 1

## Example 2: Matrix-Vector Unit (MVAU)

### Current Complex Approach

Uses multiple template fragments, macros for different PE/SIMD configurations, conditional includes based on memory mode, etc.

### Simplified Approach

**Template for embedded weights**:
```cpp
// MVAU HLS Implementation - {{ node_name }}
// Matrix: {{ mh }}x{{ mw }}, SIMD: {{ simd }}, PE: {{ pe }}

#include "mvau.hpp"

// Embedded weight array
const {{ weight_type }} weights[{{ mh }}/{{ pe }}][{{ pe }}][{{ mw }}/{{ simd }}][{{ simd }}] = {
#include "weights_{{ node_name }}.hpp"
};

void {{ node_name }}(
    hls::stream<ap_uint<{{ input_width }}>> &in0,
    hls::stream<ap_uint<{{ output_width }}>> &out0
) {
    #pragma HLS INTERFACE axis port=in0
    #pragma HLS INTERFACE axis port=out0
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weights complete dim=4
    
    Matrix_Vector_Activate_Batch<{{ mw }}, {{ mh }}, {{ simd }}, {{ pe }}>(
        in0, out0, weights, {{ activation_function }}, {{ num_samples }}
    );
}
```

**Template for streaming weights**:
```cpp
// MVAU HLS Implementation - {{ node_name }} (Streaming Weights)
// Matrix: {{ mh }}x{{ mw }}, SIMD: {{ simd }}, PE: {{ pe }}

#include "mvau.hpp"

void {{ node_name }}(
    hls::stream<ap_uint<{{ input_width }}>> &in0,
    hls::stream<ap_uint<{{ weight_width }}>> &weights,
    hls::stream<ap_uint<{{ output_width }}>> &out0
) {
    #pragma HLS INTERFACE axis port=in0
    #pragma HLS INTERFACE axis port=weights  
    #pragma HLS INTERFACE axis port=out0
    
    Matrix_Vector_Activate_Stream<{{ mw }}, {{ mh }}, {{ simd }}, {{ pe }}>(
        in0, weights, out0, {{ activation_function }}, {{ num_samples }}
    );
}
```

## Example 3: RTL Template Simplification

### Current Approach

**Multiple files with macros**:
```jinja2
{# base/rtl_base.v.j2 #}
{% from "components/rtl/signal_declarations.j2" import io_signals %}
{% from "components/rtl/process_blocks.j2" import clk_rst_process %}

module {{ module_name }} (
    {{ io_signals(input_width, output_width) }}
);
{{ clk_rst_process("state", "IDLE") }}
{% block operation_logic %}{% endblock %}
endmodule
```

### Simplified Approach

**Single clear template**:
```verilog
// {{ operation_name }} RTL Implementation
// Node: {{ node_name }}

module {{ node_name }}_rtl #(
    parameter INPUT_WIDTH = {{ input_width }},
    parameter OUTPUT_WIDTH = {{ output_width }},
    parameter PE = {{ pe }}
)(
    // Clock and Reset
    input  wire clk,
    input  wire rst,
    
    // AXI Stream Input
    input  wire [INPUT_WIDTH-1:0] s_axis_tdata,
    input  wire s_axis_tvalid,
    output wire s_axis_tready,
    
    // AXI Stream Output  
    output reg [OUTPUT_WIDTH-1:0] m_axis_tdata,
    output reg m_axis_tvalid,
    input  wire m_axis_tready
);

    // Implementation specific to this operation
    // Clear, readable, no hidden complexity
    
endmodule
```

## Benefits of Simplification

### For New Contributors
- **Before**: Study 6+ files, learn macro system, trace through includes
- **After**: Read one file, understand immediately, start modifying

### For Debugging
- **Before**: Error in macro expansion across multiple files
- **After**: Error points to exact line in single template

### For Custom Operations
- **Before**: Figure out which base template to extend, which macros to use
- **After**: Copy similar operation template, modify values

### For Maintenance
- **Before**: Change in base template affects all operations
- **After**: Each operation template is independent

## Migration Strategy

### Phase 1: Document Simple Template Patterns
- Create examples for each operation type
- Show variable naming conventions
- Provide copy-paste starting points

### Phase 2: New Operations Use Simple Templates
- All new operations use self-contained templates
- No requirement to use base templates
- Direct variable substitution

### Phase 3: Gradual Migration of Existing Operations
- As operations are updated, simplify templates
- Keep old templates for compatibility
- Mark complex templates as "legacy"

### Phase 4: Community Templates
- Encourage users to share simple templates
- Build library of template patterns
- Focus on clarity over cleverness

## Template Guidelines

### DO:
- ✅ Use descriptive variable names: `{{ num_channels }}` not `{{ nc }}`
- ✅ Include comments showing example values
- ✅ Keep templates self-contained
- ✅ Use simple conditionals when needed
- ✅ Provide complete working examples

### DON'T:
- ❌ Create deep inheritance hierarchies  
- ❌ Hide logic in macros
- ❌ Split simple operations across files
- ❌ Over-abstract common patterns
- ❌ Require extensive documentation to understand

## Conclusion

The best template is one that a hardware engineer can understand in 5 minutes and modify in 10 minutes. Complex template systems create barriers that discourage contributions and slow development. By adopting simple, self-contained templates, FINN can become more accessible to the broader FPGA community.