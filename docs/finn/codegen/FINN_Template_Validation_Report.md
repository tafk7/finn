# FINN Template Validation Report
**Overall Status**: VALID

## Base Templates
- ✅ `hls_base.cpp.j2`
- ✅ `rtl_base.v.j2`

## Component Templates
- ✅ `components/rtl/process_blocks.j2`
  - **Macros**: clk_rst_process, state_machine_comb, counter_process, pipeline_stage, memory_read_process, memory_write_process, handshake_logic, pe_array_process, output_assignment
  - **Components**: rtl_always
- ✅ `components/rtl/signal_declarations.j2`
  - **Macros**: io_signals, multi_io_signals, processing_signals, memory_signals, counter_signals, state_machine_signals, fifo_signals, pe_array_signals
  - **Components**: rtl_signals
- ✅ `components/includes/operation_specific.j2`
- ✅ `components/hls/pragmas.j2`
  - **Macros**: stream_interface, array_partition, resource_bind, memory_allocation, pipeline_with_dependence, dataflow_interval, inline_control, top_function_interface, latency_constraint, clock_domain, thresholding_pragmas, mvau_pragmas
  - **Components**: hls_pragmas
- ✅ `components/hls/loop_utils.j2`
  - **Macros**: hls_loop, tile_loop, pe_loop, simd_loop, nested_loops, close_loop, close_nested
  - **Components**: hls_pragmas
- ✅ `components/streams/declarations.j2`

## Summary
- **Total Templates**: 8
- **Valid Templates**: 8
- **Success Rate**: 100.0%