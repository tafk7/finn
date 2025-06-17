"""
Modern RTL Generator - Enhanced RTL Code Generation

This module provides the modern RTL generator that builds upon RTL's successful
operation-driven pattern while adding shared infrastructure benefits like
advanced template processing and unified file management.
"""

from typing import Dict, List, Any, Optional
import os
from pathlib import Path

from .base import BaseCodeGenerator


class ModernRTLGenerator(BaseCodeGenerator):
    """
    Modern RTL code generator that enhances RTL's successful approach.
    
    Builds upon RTL's proven operation-driven pattern while providing
    modern template engine capabilities and shared infrastructure benefits.
    Maintains RTL's simplicity while adding Jinja2 features.
    """
    
    def __init__(self, operation):
        """
        Initialize the modern RTL generator.
        
        Args:
            operation: HWCustomOp instance for RTL code generation
        """
        super().__init__(operation)
        self.backend_type = "rtl"
        
    def get_template_name(self) -> str:
        """
        Get operation-specific template name for RTL generation.
        
        Returns:
            Template filename based on operation type
        """
        op_type = self.operation.onnx_node.op_type
        
        # Operation-specific template selection (similar to current RTL approach)
        if op_type == "StreamingFIFO_rtl":
            return "rtl/streaming_fifo_wrapper.v.j2"
        elif op_type == "Thresholding_rtl":
            return "rtl/thresholding_wrapper.v.j2"
        elif op_type == "MatrixVectorActivation_rtl":
            return "rtl/mvau_wrapper.v.j2"
        elif op_type == "StreamingDataWidthConverter_rtl":
            return "rtl/dwc_wrapper.v.j2"
        elif op_type == "Pool_Batch_rtl":
            return "rtl/pool_wrapper.v.j2"
        elif op_type == "ConvolutionInputGenerator_rtl":
            return "rtl/swg_wrapper.v.j2"
        else:
            # Generic RTL wrapper template
            return "rtl/generic_wrapper.v.j2"
    
    def prepare_context(self, model, fpgapart: str, clk: str) -> Dict[str, Any]:
        """
        Prepare RTL-specific template context.
        
        Enhanced version of current RTL prepare_codegen_rtl_values method,
        providing more comprehensive context for Jinja2 templates.
        
        Args:
            model: FINN model containing the operation
            fpgapart: Target FPGA part string
            clk: Clock specification
            
        Returns:
            Dictionary containing RTL template variables
        """
        # Start with basic RTL context (similar to current prepare_codegen_rtl_values)
        context = self._prepare_basic_rtl_context(model, fpgapart, clk)
        
        # Add enhanced context for Jinja2 templates
        context.update({
            # Module and instance information
            'module_name': self._get_module_name(),
            'instance_name': self._get_instance_name(),
            'top_module': self._get_top_module_name(),
            
            # Port definitions
            'input_ports': self._get_input_ports(),
            'output_ports': self._get_output_ports(),
            'control_ports': self._get_control_ports(),
            
            # Parameters and generics
            'parameters': self._get_parameters(),
            'localparam': self._get_localparams(),
            
            # Data path configuration
            'data_widths': self._get_data_widths(),
            'fifo_depths': self._get_fifo_depths(),
            'pipeline_stages': self._get_pipeline_stages(),
            
            # Memory configuration
            'memory_config': self._get_rtl_memory_config(),
            'ram_style': self._get_ram_style(),
            
            # File references
            'rtl_files': self._get_rtl_file_references(),
            'include_files': self._get_rtl_includes(),
            
            # Operation-specific context
            'operation_config': self._get_operation_specific_config(),
        })
        
        return context
    
    def _prepare_basic_rtl_context(self, model, fpgapart: str, clk: str) -> Dict[str, Any]:
        """
        Prepare basic RTL context similar to current prepare_codegen_rtl_values.
        
        This maintains compatibility with existing RTL operations while
        providing a foundation for enhanced template features.
        """
        context = {}
        
        # Basic node information
        node = self.operation.onnx_node
        context['node_name'] = node.name
        context['op_type'] = node.op_type
        
        # Input/output shapes and data types
        context['input_shapes'] = []
        context['output_shapes'] = []
        context['input_datatypes'] = []
        context['output_datatypes'] = []
        
        for i, input_name in enumerate(node.input):
            ishape = self.operation.get_normal_input_shape(i)
            itype = self.operation.get_input_datatype(i)
            if ishape:
                context['input_shapes'].append(ishape)
            if itype:
                context['input_datatypes'].append(str(itype))
        
        for i, output_name in enumerate(node.output):
            oshape = self.operation.get_normal_output_shape(i)
            otype = self.operation.get_output_datatype(i)
            if oshape:
                context['output_shapes'].append(oshape)
            if otype:
                context['output_datatypes'].append(str(otype))
        
        # Basic operation attributes
        for attr_name in self.operation.get_nodeattr_types().keys():
            context[attr_name] = self.operation.get_nodeattr(attr_name)
        
        return context
    
    def _get_module_name(self) -> str:
        """Get the RTL module name for this operation."""
        node_name = self.operation.onnx_node.name
        return f"{node_name}_wrapper"
    
    def _get_instance_name(self) -> str:
        """Get the RTL instance name for this operation."""
        node_name = self.operation.onnx_node.name
        return f"{node_name}_inst"
    
    def _get_top_module_name(self) -> str:
        """Get the top-level module name."""
        node_name = self.operation.onnx_node.name
        return f"{node_name}_top"
    
    def _get_input_ports(self) -> List[Dict[str, Any]]:
        """Get input port definitions for the RTL module."""
        ports = []
        
        # Standard AXI4-Stream input ports
        ports.append({
            'name': 'ap_clk',
            'direction': 'input',
            'width': 1,
            'type': 'logic',
            'description': 'Clock signal'
        })
        
        ports.append({
            'name': 'ap_rst_n',
            'direction': 'input', 
            'width': 1,
            'type': 'logic',
            'description': 'Active-low reset'
        })
        
        # Data input ports based on operation inputs
        for i, input_name in enumerate(self.operation.onnx_node.input):
            ishape = self.operation.get_normal_input_shape(i)
            itype = self.operation.get_input_datatype(i)
            
            if ishape and itype:
                width = self._calculate_stream_width(ishape, itype)
                ports.extend([
                    {
                        'name': f'in{i}_V_TDATA',
                        'direction': 'input',
                        'width': width,
                        'type': 'logic',
                        'description': f'Input {i} data'
                    },
                    {
                        'name': f'in{i}_V_TVALID',
                        'direction': 'input',
                        'width': 1,
                        'type': 'logic',
                        'description': f'Input {i} valid'
                    },
                    {
                        'name': f'in{i}_V_TREADY',
                        'direction': 'output',
                        'width': 1,
                        'type': 'logic',
                        'description': f'Input {i} ready'
                    }
                ])
        
        return ports
    
    def _get_output_ports(self) -> List[Dict[str, Any]]:
        """Get output port definitions for the RTL module."""
        ports = []
        
        # Data output ports based on operation outputs
        for i, output_name in enumerate(self.operation.onnx_node.output):
            oshape = self.operation.get_normal_output_shape(i)
            otype = self.operation.get_output_datatype(i)
            
            if oshape and otype:
                width = self._calculate_stream_width(oshape, otype)
                ports.extend([
                    {
                        'name': f'out_V_TDATA',
                        'direction': 'output',
                        'width': width,
                        'type': 'logic',
                        'description': f'Output data'
                    },
                    {
                        'name': f'out_V_TVALID',
                        'direction': 'output',
                        'width': 1,
                        'type': 'logic',
                        'description': f'Output valid'
                    },
                    {
                        'name': f'out_V_TREADY',
                        'direction': 'input',
                        'width': 1,
                        'type': 'logic',
                        'description': f'Output ready'
                    }
                ])
        
        return ports
    
    def _get_control_ports(self) -> List[Dict[str, Any]]:
        """Get control port definitions for the RTL module."""
        ports = []
        
        # Add control ports if needed by operation
        op_type = self.operation.onnx_node.op_type
        
        if "MatrixVectorActivation" in op_type:
            # MVAU may need weight loading control
            ports.extend([
                {
                    'name': 'weights_V_TDATA',
                    'direction': 'input',
                    'width': 32,  # Configurable
                    'type': 'logic',
                    'description': 'Weight data'
                },
                {
                    'name': 'weights_V_TVALID',
                    'direction': 'input',
                    'width': 1,
                    'type': 'logic',
                    'description': 'Weight valid'
                }
            ])
        
        return ports
    
    def _get_parameters(self) -> List[Dict[str, Any]]:
        """Get module parameters for the RTL module."""
        params = []
        
        # Common parameters
        params.extend([
            {
                'name': 'C_S_AXI_DATA_WIDTH',
                'value': 32,
                'type': 'integer',
                'description': 'AXI data width'
            },
            {
                'name': 'C_S_AXI_ADDR_WIDTH',
                'value': 32,
                'type': 'integer', 
                'description': 'AXI address width'
            }
        ])
        
        # Operation-specific parameters
        op_type = self.operation.onnx_node.op_type
        
        if "MatrixVectorActivation" in op_type:
            params.extend([
                {
                    'name': 'MW',
                    'value': self.operation.get_nodeattr("MW"),
                    'type': 'integer',
                    'description': 'Matrix width'
                },
                {
                    'name': 'MH', 
                    'value': self.operation.get_nodeattr("MH"),
                    'type': 'integer',
                    'description': 'Matrix height'
                },
                {
                    'name': 'PE',
                    'value': self.operation.get_nodeattr("PE"),
                    'type': 'integer',
                    'description': 'Processing elements'
                },
                {
                    'name': 'SIMD',
                    'value': self.operation.get_nodeattr("SIMD"),
                    'type': 'integer',
                    'description': 'SIMD parallelism'
                }
            ])
        elif "Thresholding" in op_type:
            params.extend([
                {
                    'name': 'NumChannels',
                    'value': self.operation.get_nodeattr("NumChannels"),
                    'type': 'integer',
                    'description': 'Number of channels'
                },
                {
                    'name': 'PE',
                    'value': self.operation.get_nodeattr("PE"),
                    'type': 'integer',
                    'description': 'Processing elements'
                }
            ])
        
        return params
    
    def _get_localparams(self) -> List[Dict[str, Any]]:
        """Get local parameters for the RTL module."""
        localparams = []
        
        # Add commonly used local parameters
        localparams.extend([
            {
                'name': 'IDLE',
                'value': "2'b00",
                'description': 'State machine idle state'
            },
            {
                'name': 'PROCESSING',
                'value': "2'b01", 
                'description': 'State machine processing state'
            },
            {
                'name': 'DONE',
                'value': "2'b10",
                'description': 'State machine done state'
            }
        ])
        
        return localparams
    
    def _get_data_widths(self) -> Dict[str, int]:
        """Get data width specifications."""
        widths = {}
        
        # Input data widths
        for i, input_name in enumerate(self.operation.onnx_node.input):
            itype = self.operation.get_input_datatype(i)
            if itype:
                widths[f'input_{i}_width'] = itype.bitwidth()
        
        # Output data widths
        for i, output_name in enumerate(self.operation.onnx_node.output):
            otype = self.operation.get_output_datatype(i)
            if otype:
                widths[f'output_{i}_width'] = otype.bitwidth()
        
        return widths
    
    def _get_fifo_depths(self) -> Dict[str, int]:
        """Get FIFO depth specifications."""
        depths = {}
        
        # Default FIFO depths
        depths['input_fifo_depth'] = 32
        depths['output_fifo_depth'] = 32
        
        # Operation-specific FIFO depths
        if hasattr(self.operation, 'get_instream_width'):
            depths['instream_fifo_depth'] = 64
        if hasattr(self.operation, 'get_outstream_width'):
            depths['outstream_fifo_depth'] = 64
        
        return depths
    
    def _get_pipeline_stages(self) -> int:
        """Get number of pipeline stages."""
        # Default pipeline depth
        return 2
    
    def _get_rtl_memory_config(self) -> Dict[str, Any]:
        """Get RTL memory configuration."""
        return {
            'ram_style': self._get_ram_style(),
            'use_bram': True,
            'use_uram': False,
        }
    
    def _get_ram_style(self) -> str:
        """Get RAM style for memory inference."""
        ram_style = self.operation.get_nodeattr("ram_style")
        return ram_style if ram_style else "auto"
    
    def _get_rtl_file_references(self) -> List[str]:
        """Get RTL file references needed by this operation."""
        files = []
        
        # Get files from current RTL operation (maintaining compatibility)
        if hasattr(self.operation, 'get_verilog_top_module_intf_names'):
            # This operation has Verilog interfaces
            files.append('axis_interfaces.sv')
        
        # Add operation-specific files
        op_type = self.operation.onnx_node.op_type
        
        if "StreamingFIFO" in op_type:
            files.extend(['Q_srl.v', 'Q_srl_no_reserve.v'])
        elif "Thresholding" in op_type:
            files.extend(['thresholding_template.sv'])
        elif "MatrixVectorActivation" in op_type:
            files.extend(['mvu_template.sv', 'vvu_template.sv'])
        
        return files
    
    def _get_rtl_includes(self) -> List[str]:
        """Get RTL include files."""
        includes = []
        
        # Get includes from library resolver
        lib_includes = self.get_include_files()
        includes.extend(lib_includes)
        
        return includes
    
    def _get_operation_specific_config(self) -> Dict[str, Any]:
        """Get operation-specific configuration."""
        config = {}
        
        op_type = self.operation.onnx_node.op_type
        
        if "MatrixVectorActivation" in op_type:
            config.update({
                'use_weights': True,
                'weight_memory_type': 'distributed',
                'activation_type': self._get_safe_activation_type(),
            })
        elif "Thresholding" in op_type:
            config.update({
                'use_thresholds': True,
                'threshold_memory_type': 'distributed',
                'narrow_thresholds': False,
            })
        elif "StreamingFIFO" in op_type:
            config.update({
                'use_output_register': True,
                'bypass_when_empty': False,
            })
        
        return config
    
    def _get_safe_activation_type(self) -> str:
        """Get activation type safely, handling missing ActType attribute."""
        try:
            return self.operation.get_nodeattr("ActType")
        except (AttributeError, KeyError):
            # Try noActivation attribute as fallback
            try:
                no_activation = self.operation.get_nodeattr("noActivation")
                return "none" if no_activation else "relu"
            except (AttributeError, KeyError):
                return "relu"
    
    def _calculate_stream_width(self, shape: List[int], datatype) -> int:
        """Calculate the stream width for given shape and datatype."""
        if not shape or not datatype:
            return 32  # Default width
        
        # Calculate total bits needed
        elements_per_cycle = shape[-1] if len(shape) > 0 else 1
        bits_per_element = datatype.bitwidth() if hasattr(datatype, 'bitwidth') else 8
        
        return elements_per_cycle * bits_per_element
    
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
            os.path.join(code_gen_dir, f"{node_name}_wrapper.v"),
        ]
        
        # Add additional files based on operation type
        op_type = self.operation.onnx_node.op_type
        
        if "MatrixVectorActivation" in op_type:
            files.extend([
                os.path.join(code_gen_dir, f"{node_name}_mvu.sv"),
                os.path.join(code_gen_dir, f"{node_name}_vvu.sv"),
            ])
        elif "Thresholding" in op_type:
            files.append(os.path.join(code_gen_dir, f"{node_name}_thresholding.sv"))
        
        return files
    
    def validate_operation(self) -> bool:
        """
        Validate that this operation can be handled by the RTL generator.
        
        Returns:
            True if operation is supported, False otherwise
        """
        if not super().validate_operation():
            return False
        
        # Check for RTL-specific requirements
        op_type = self.operation.onnx_node.op_type
        
        # RTL operations typically end with "_rtl"
        if not op_type.endswith("_rtl"):
            return False
        
        # Check for required RTL methods
        required_methods = ['get_verilog_top_module_intf_names']
        for method in required_methods:
            if not hasattr(self.operation, method):
                return False
        
        return True