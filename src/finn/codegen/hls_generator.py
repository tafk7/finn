"""
Modern HLS Generator - Operation-Specific HLS Code Generation

This module provides the modern HLS generator that replaces the mega-template
approach with operation-specific templates, adopting RTL's successful
operation-driven pattern for HLS code generation.
"""

from typing import Dict, List, Any, Optional
import os
from pathlib import Path

from .base import BaseCodeGenerator


class ModernHLSGenerator(BaseCodeGenerator):
    """
    Modern HLS code generator using operation-specific templates.
    
    Replaces the rigid mega-template approach with flexible, operation-driven
    code generation that allows each operation to define its own structure
    and requirements.
    """
    
    def __init__(self, operation):
        """
        Initialize the modern HLS generator.
        
        Args:
            operation: HWCustomOp instance for HLS code generation
        """
        super().__init__(operation)
        self.backend_type = "hls"
        
    def get_template_name(self) -> str:
        """
        Get operation-specific template name for HLS generation.
        
        Returns:
            Template filename based on operation type and configuration
        """
        op_type = self.operation.onnx_node.op_type
        
        # Operation-specific template selection logic
        if op_type == "MatrixVectorActivation":
            return self._get_mvau_template_name()
        elif op_type == "Thresholding_Batch":
            return "hls/thresholding_batch.cpp.j2"
        elif op_type == "SlidingWindow":
            return "hls/sliding_window.cpp.j2"
        elif op_type == "Pool_Batch":
            return "hls/pool_batch.cpp.j2"
        elif op_type == "StreamingDataWidthConverter":
            return "hls/data_width_converter.cpp.j2"
        elif op_type == "ConvolutionInputGenerator":
            return "hls/convolution_input_generator.cpp.j2"
        elif op_type == "FMPadding":
            return "hls/fm_padding.cpp.j2"
        elif op_type == "StreamingFIFO":
            return "hls/streaming_fifo.cpp.j2"
        else:
            # Generic template for unknown operations
            return "hls/generic_operation.cpp.j2"
    
    def _get_mvau_template_name(self) -> str:
        """
        Get template name for MatrixVectorActivation based on configuration.
        
        Returns:
            Specific MVAU template name
        """
        # Template selection based on operation attributes
        mem_mode = self.operation.get_nodeattr("mem_mode")
        pe = self.operation.get_nodeattr("PE")
        simd = self.operation.get_nodeattr("SIMD")
        
        if mem_mode == "internal_embedded":
            return "hls/mvau_embedded.cpp.j2"
        elif mem_mode == "internal_decoupled":
            return "hls/mvau_decoupled.cpp.j2"
        elif pe == 1 and simd == 1:
            return "hls/mvau_sequential.cpp.j2"
        else:
            return "hls/mvau_streaming.cpp.j2"
    
    def prepare_context(self, model, fpgapart: str, clk: str) -> Dict[str, Any]:
        """
        Prepare HLS-specific template context.
        
        Args:
            model: FINN model containing the operation
            fpgapart: Target FPGA part string
            clk: Clock specification
            
        Returns:
            Dictionary containing HLS template variables
        """
        context = {
            # Basic operation information
            'op_type': self.operation.onnx_node.op_type,
            'node_name': self.operation.onnx_node.name,
            
            # HLS-specific configuration
            'defines': self._generate_defines(),
            'includes': self._generate_includes(),
            'pragmas': self._generate_pragmas(),
            'function_name': self._get_function_name(),
            'namespace': self._get_namespace(),
            
            # Data types and shapes
            'input_shapes': self._get_input_shapes(),
            'output_shapes': self._get_output_shapes(),
            'data_types': self._get_data_types(),
            
            # Operation-specific parameters
            'operation_params': self._get_operation_params(),
            
            # Memory and parallelization
            'memory_config': self._get_memory_config(),
            'parallelization': self._get_parallelization_config(),
            
            # Templates and code generation
            'template_params': self._get_template_params(),
            'code_variants': self._get_code_variants(),
        }
        
        return context
    
    def _generate_defines(self) -> List[str]:
        """Generate C++ #define statements for this operation."""
        defines = []
        
        # Common defines for all HLS operations
        defines.append(('AP_INT_MAX_W', '8192'))
        
        # Operation-specific defines
        op_type = self.operation.onnx_node.op_type
        
        if op_type == "MatrixVectorActivation":
            defines.extend(self._get_mvau_defines())
        elif op_type == "Thresholding_Batch":
            defines.extend(self._get_thresholding_defines())
        elif op_type == "SlidingWindow":
            defines.extend(self._get_sliding_window_defines())
        
        return defines
    
    def _get_mvau_defines(self) -> List[tuple]:
        """Get defines specific to MatrixVectorActivation."""
        defines = []
        
        # Matrix dimensions
        mw = self.operation.get_nodeattr("MW")
        mh = self.operation.get_nodeattr("MH")
        defines.append(('MW', str(mw)))
        defines.append(('MH', str(mh)))
        
        # Parallelization
        pe = self.operation.get_nodeattr("PE")
        simd = self.operation.get_nodeattr("SIMD")
        defines.append(('PE', str(pe)))
        defines.append(('SIMD', str(simd)))
        
        # Memory mode
        mem_mode = self.operation.get_nodeattr("mem_mode")
        defines.append(('MEM_MODE', f'"{mem_mode}"'))
        
        # Activation function - handle missing ActType gracefully
        try:
            act_type = self.operation.get_nodeattr("ActType")
            if act_type:
                defines.append(('ACT_TYPE', f'"{act_type}"'))
        except (AttributeError, KeyError):
            # Try noActivation attribute as fallback
            try:
                no_activation = self.operation.get_nodeattr("noActivation")
                act_type = "none" if no_activation else "relu"
                defines.append(('ACT_TYPE', f'"{act_type}"'))
            except (AttributeError, KeyError):
                defines.append(('ACT_TYPE', '"relu"'))
        
        return defines
    
    def _get_thresholding_defines(self) -> List[tuple]:
        """Get defines specific to Thresholding_Batch."""
        defines = []
        
        # Thresholding parameters
        num_channels = self.operation.get_nodeattr("NumChannels")
        pe = self.operation.get_nodeattr("PE") 
        
        defines.append(('NUM_CHANNELS', str(num_channels)))
        defines.append(('PE', str(pe)))
        
        # Activation type
        act_type = self.operation.get_nodeattr("ActType")
        if act_type:
            defines.append(('ACT_TYPE', f'"{act_type}"'))
        
        return defines
    
    def _get_sliding_window_defines(self) -> List[tuple]:
        """Get defines specific to SlidingWindow."""
        defines = []
        
        # Window parameters
        dim_h = self.operation.get_nodeattr("DimH")
        dim_w = self.operation.get_nodeattr("DimW")
        kernel_h = self.operation.get_nodeattr("KernelH")
        kernel_w = self.operation.get_nodeattr("KernelW")
        
        defines.extend([
            ('DIM_H', str(dim_h)),
            ('DIM_W', str(dim_w)),
            ('KERNEL_H', str(kernel_h)),
            ('KERNEL_W', str(kernel_w))
        ])
        
        return defines
    
    def _generate_includes(self) -> List[str]:
        """Generate include statements for this operation."""
        includes = []
        
        # Get includes from library resolver
        lib_includes = self.get_include_files()
        includes.extend(lib_includes)
        
        # Add operation-specific includes
        op_type = self.operation.onnx_node.op_type
        
        if op_type == "MatrixVectorActivation":
            includes.extend(['mvau.hpp', 'activations.hpp'])
        elif op_type == "Thresholding_Batch":
            includes.extend(['thresholding.hpp'])
        elif op_type == "SlidingWindow":
            includes.extend(['sliding_window.hpp'])
        
        return list(set(includes))  # Remove duplicates
    
    def _generate_pragmas(self) -> List[str]:
        """Generate HLS pragma statements for this operation."""
        pragmas = []
        
        # Interface pragmas
        pragmas.extend([
            '#pragma HLS INTERFACE axis port=in0',
            '#pragma HLS INTERFACE axis port=out',
            '#pragma HLS INTERFACE ap_ctrl_none port=return'
        ])
        
        # Operation-specific pragmas
        op_type = self.operation.onnx_node.op_type
        
        if op_type == "MatrixVectorActivation":
            pragmas.extend(self._get_mvau_pragmas())
        elif op_type == "Thresholding_Batch":
            pragmas.extend(self._get_thresholding_pragmas())
        
        return pragmas
    
    def _get_mvau_pragmas(self) -> List[str]:
        """Get pragmas specific to MatrixVectorActivation."""
        pragmas = []
        
        # Array partitioning based on parallelization
        pe = self.operation.get_nodeattr("PE")
        simd = self.operation.get_nodeattr("SIMD")
        
        if pe > 1:
            pragmas.append(f'#pragma HLS ARRAY_PARTITION variable=weights dim=1 factor={pe}')
        if simd > 1:
            pragmas.append(f'#pragma HLS ARRAY_PARTITION variable=weights dim=2 factor={simd}')
        
        # Pipeline pragma
        pragmas.append('#pragma HLS PIPELINE II=1')
        
        return pragmas
    
    def _get_thresholding_pragmas(self) -> List[str]:
        """Get pragmas specific to Thresholding_Batch."""
        pragmas = []
        
        pe = self.operation.get_nodeattr("PE")
        if pe > 1:
            pragmas.append(f'#pragma HLS ARRAY_PARTITION variable=thresholds factor={pe}')
        
        pragmas.append('#pragma HLS PIPELINE II=1')
        return pragmas
    
    def _get_function_name(self) -> str:
        """Get the HLS function name for this operation."""
        node_name = self.operation.onnx_node.name
        return f"{node_name}_hls"
    
    def _get_namespace(self) -> str:
        """Get the namespace for this operation."""
        return "finn_hls"
    
    def _get_input_shapes(self) -> List[List[int]]:
        """Get input tensor shapes."""
        shapes = []
        for idx, input_name in enumerate(self.operation.onnx_node.input):
            shape = self.operation.get_normal_input_shape(idx)
            if shape:
                shapes.append(shape)
        return shapes
    
    def _get_output_shapes(self) -> List[List[int]]:
        """Get output tensor shapes."""
        shapes = []
        for idx, output_name in enumerate(self.operation.onnx_node.output):
            shape = self.operation.get_normal_output_shape(idx)
            if shape:
                shapes.append(shape)
        return shapes
    
    def _get_data_types(self) -> Dict[str, str]:
        """Get data types for inputs and outputs."""
        data_types = {}
        
        # Input data types
        for i, input_name in enumerate(self.operation.onnx_node.input):
            dt = self.operation.get_input_datatype(i)
            data_types[f"input_{i}_type"] = str(dt) if dt else "ap_int<8>"
        
        # Output data types
        for i, output_name in enumerate(self.operation.onnx_node.output):
            dt = self.operation.get_output_datatype(i)
            data_types[f"output_{i}_type"] = str(dt) if dt else "ap_int<8>"
        
        return data_types
    
    def _get_operation_params(self) -> Dict[str, Any]:
        """Get operation-specific parameters."""
        params = {}
        
        # Extract all node attributes
        for attr_name in self.operation.get_nodeattr_types().keys():
            params[attr_name] = self.operation.get_nodeattr(attr_name)
        
        return params
    
    def _get_memory_config(self) -> Dict[str, Any]:
        """Get memory configuration for this operation."""
        return {
            'mem_mode': self.operation.get_nodeattr("mem_mode"),
            'ram_style': self.operation.get_nodeattr("ram_style"),
        }
    
    def _get_parallelization_config(self) -> Dict[str, Any]:
        """Get parallelization configuration."""
        return {
            'pe': self.operation.get_nodeattr("PE"),
            'simd': self.operation.get_nodeattr("SIMD"),
        }
    
    def _get_template_params(self) -> Dict[str, Any]:
        """Get template parameters for C++ templates."""
        return {
            'use_templates': True,
            'template_args': self._get_template_args(),
        }
    
    def _get_template_args(self) -> List[str]:
        """Get C++ template arguments."""
        args = []
        
        # Add data width template arguments
        for i, input_name in enumerate(self.operation.onnx_node.input):
            dt = self.operation.get_input_datatype(i)
            if dt:
                args.append(f"unsigned int IN{i}_WIDTH = {dt.bitwidth()}")
        
        return args
    
    def _get_code_variants(self) -> Dict[str, bool]:
        """Get code generation variants/options."""
        return {
            'use_pragma_unroll': True,
            'use_array_partition': True,
            'use_pipeline': True,
            'generate_testbench': False,
        }
    
    def get_generated_files(self, code_gen_dir: str) -> List[str]:
        """
        Get list of files that will be generated.
        
        Args:
            code_gen_dir: Directory where files will be generated
            
        Returns:
            List of file paths that will be generated
        """
        node_name = self.operation.onnx_node.name
        files = [
            os.path.join(code_gen_dir, f"{node_name}.cpp"),
            os.path.join(code_gen_dir, f"{node_name}.h"),
        ]
        
        # Add additional files based on operation type
        op_type = self.operation.onnx_node.op_type
        if op_type == "MatrixVectorActivation":
            files.append(os.path.join(code_gen_dir, f"{node_name}_weights.h"))
        
        return files
    
    def validate_operation(self) -> bool:
        """
        Validate that this operation can be handled by the HLS generator.
        
        Returns:
            True if operation is supported, False otherwise
        """
        if not super().validate_operation():
            return False
        
        # Check for HLS-specific requirements
        op_type = self.operation.onnx_node.op_type
        supported_ops = [
            "MatrixVectorActivation",
            "Thresholding_Batch", 
            "SlidingWindow",
            "Pool_Batch",
            "StreamingDataWidthConverter",
            "ConvolutionInputGenerator",
            "FMPadding",
            "StreamingFIFO"
        ]
        
        return op_type in supported_ops