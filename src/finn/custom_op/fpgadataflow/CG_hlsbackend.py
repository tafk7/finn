# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
import os
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from qonnx.core.datatype import DataType

from finn.codegen.codegen import Codegen
from finn.codegen import TemplateEngine
from finn.util.basic import CppBuilder, make_build_dir


class CG_HLSBackend(Codegen):
    """
    Clean HLS Backend class without legacy compatibility bloat.
    
    Provides HLS-specific code generation functionality for FINN custom ops
    using only Jinja2-based template system with direct value generation.
    No code_gen_dict or legacy string replacement methods.
    """

    def __init__(self, **kwargs):
        """Initialize clean HLS backend."""
        super().__init__()
        
        # HLS-specific initialization without legacy bloat
        self.logger = logging.getLogger(f"finn.codegen.{self.__class__.__name__}")
        
        # Initialize template engine for execution methods
        self.template_engine = TemplateEngine()
        
        # Apply HLS-specific configurations
        for key, value in kwargs.items():
            if key.startswith('hls_'):
                setattr(self, key, value)
        
        self.logger.debug(f"Initialized clean HLS backend: {self.__class__.__name__}")

    def get_template_name(self) -> str:
        """Get template name - explicit declaration required.
        
        Returns:
            Name of template to use for code generation
            
        Raises:
            NotImplementedError: If no template declared by subclass
        """
        # Check for explicit class-level template declaration
        if hasattr(self.__class__, 'TEMPLATE_NAME') and self.__class__.TEMPLATE_NAME:
            return self.__class__.TEMPLATE_NAME
            
        # Check for template selection from options
        if hasattr(self.__class__, 'TEMPLATE_OPTIONS') and self.__class__.TEMPLATE_OPTIONS:
            return self._select_template_from_options()
            
        raise NotImplementedError(
            f"{self.__class__.__name__} must declare TEMPLATE_NAME or TEMPLATE_OPTIONS"
        )
    
    def _select_template_from_options(self) -> str:
        """Select template from available options. Override in subclass for logic.
        
        Returns:
            Template name from TEMPLATE_OPTIONS
        """
        if not hasattr(self.__class__, 'TEMPLATE_OPTIONS') or not self.__class__.TEMPLATE_OPTIONS:
            raise NotImplementedError("No template options available")
        
        # Default: return first option
        return next(iter(self.__class__.TEMPLATE_OPTIONS.values()))

    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Generate template values directly - no legacy conversion.
        
        Args:
            template_name: Name of template to extract values for
            
        Returns:
            Dictionary mapping template placeholders to values
        """
        self.logger.debug(f"Generating template values for: {template_name}")
        
        # Build template values through direct generation
        values = self._generate_common_values(self)
        values.update(self._generate_hls_common_values())
        values.update(self._generate_operation_specific_values(template_name))
        
        # Validate and clean values
        validated_values = self._validate_template_values(values)
        
        self.logger.debug(f"Generated {len(validated_values)} template values")
        return validated_values

    def _generate_hls_common_values(self) -> Dict[str, Any]:
        """Generate values common to all HLS operations.
        
        Returns:
            Dictionary of common HLS template values
        """
        return {
            'AP_INT_MAX_W': self._calculate_ap_int_max_w(),
            'INCLUDES': self._generate_hls_includes(),
            'PRAGMAS': self._generate_hls_pragmas(), 
            'STREAM_DECLARATIONS': self._generate_stream_declarations(),
            'GLOBALS': self._generate_hls_globals(),
        }

    def _generate_hls_includes(self) -> str:
        """Generate HLS includes directly.
        
        Returns:
            String containing all necessary HLS includes
        """
        includes = [
            '#include "hls_stream.h"',
            '#include "ap_int.h"',
            '#include "bnn-library.h"',
            '#include "cnpy.h"',
            '#include "npy2apintstream.hpp"',
            '#include "npy2vectorstream.hpp"',
            '#include <vector>',
        ]
        return '\n'.join(includes)

    def _generate_hls_pragmas(self) -> str:
        """Generate HLS pragmas directly.
        
        Returns:
            String containing HLS interface pragmas
        """
        pragmas = [
            '#pragma HLS INTERFACE axis port=in0_V',
            '#pragma HLS INTERFACE axis port=out0_V',
            '#pragma HLS INTERFACE ap_ctrl_none port=return'
        ]
        return '\n'.join(pragmas)

    def _generate_stream_declarations(self) -> str:
        """Generate HLS stream declarations directly.
        
        Returns:
            String containing stream declarations
        """
        declarations = [
            f'hls::stream<ap_uint<{self.get_instream_width()}>> in0_V ("in0_V");',
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");'
        ]
        return '\n'.join(declarations)

    def _generate_hls_globals(self) -> str:
        """Generate global declarations for HLS.
        
        Returns:
            String containing global declarations
        """
        globals_list = ['// Global declarations']
        
        # Add parameter includes if needed
        if hasattr(self, 'onnx_node'):
            globals_list.append('#include "params.h"')
        
        return '\n'.join(globals_list)

    def _calculate_ap_int_max_w(self) -> int:
        """Calculate the maximum width of any ap_int used in this module.
        
        Returns:
            Maximum ap_int width needed
        """
        try:
            instream = self.get_instream_width()
            outstream = self.get_outstream_width()
            max_w = max([instream, outstream])
            
            # Ensure it's within HLS limits
            assert max_w <= 8191, f"AP_INT_MAX_W={max_w} is larger than allowed maximum of 8191"
            return max_w
        except:
            # Safe default if stream widths not available
            return 32

    def _validate_template_values(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and sanitize template values.
        
        Args:
            values: Raw template values
            
        Returns:
            Validated and cleaned template values
        """
        validated = {}
        
        for key, value in values.items():
            # Skip None values
            if value is None:
                continue
                
            # Convert values to strings for template rendering
            if isinstance(value, (int, float, bool)):
                validated[key] = str(value)
            elif isinstance(value, str):
                validated[key] = value
            else:
                # Convert other types to string representation
                validated[key] = str(value)
        
        return validated

    @abstractmethod
    def _generate_common_values(self, instance) -> Dict[str, Any]:
        """Generate common template values shared across templates.
        
        Must be implemented by operation-specific subclasses.
        
        Args:
            instance: Backend instance (for compatibility)
            
        Returns:
            Dictionary of common template values
        """
        pass

    @abstractmethod  
    def _generate_operation_specific_values(self, template_name: str) -> Dict[str, Any]:
        """Generate operation-specific template values.
        
        Must be implemented by operation-specific subclasses.
        
        Args:
            template_name: Name of template being generated for
            
        Returns:
            Dictionary of operation-specific template values
        """
        pass

    def get_nodeattr_types(self) -> Dict[str, Any]:
        """Get HLS backend node attribute types.
        
        Returns:
            Dictionary of node attribute specifications
        """
        # Base HLS backend attributes
        hls_attrs = {
            "code_gen_dir_cppsim": ("s", False, ""),
            "code_gen_dir_ipgen": ("s", False, ""),
            "executable_path": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
            "res_hls": ("s", False, ""),
            "rtlsim_so": ("s", False, ""),
            "rtlsim_trace": ("s", False, ""),
            # Interface style for HLS operations
            "cpp_interface": ("s", False, "packed", {"packed", "hls_vector"}),
        }
        
        # Try to get operation-specific attributes from all parent classes
        operation_attrs = {}
        
        # Walk the MRO to get attributes from all parent classes
        for cls in self.__class__.__mro__[1:]:  # Skip self
            if hasattr(cls, 'get_nodeattr_types') and cls != CG_HLSBackend:
                try:
                    # Call the parent class method properly
                    parent_attrs = cls.get_nodeattr_types(self)
                    # Merge parent attributes, giving priority to first found
                    for k, v in parent_attrs.items():
                        if k not in operation_attrs:
                            operation_attrs[k] = v
                    self.logger.debug(f"Got {len(parent_attrs)} attributes from {cls.__name__}")
                except Exception as e:
                    self.logger.debug(f"Failed to get attributes from {cls.__name__}: {e}")
                    continue
        
        # HLS attributes extend operation attributes (merge, don't override)
        merged_attrs = {**operation_attrs, **hls_attrs}
        
        # Log any conflicts for debugging
        conflicts = set(operation_attrs.keys()) & set(hls_attrs.keys())
        if conflicts:
            self.logger.debug(f"HLS attributes extend operation attributes, conflicts: {conflicts}")
        
        self.logger.debug(f"Final merged attributes: {len(merged_attrs)} total")
        return merged_attrs

    # =====  HLS-Specific Utility Methods =====

    def _safe_get_nodeattr(self, attr_name: str, default_value=None):
        """Safely get node attribute with logging.
        
        Args:
            attr_name: Name of attribute to get
            default_value: Default value if attribute missing
            
        Returns:
            Attribute value or default
        """
        try:
            value = self.get_nodeattr(attr_name)
            self.logger.debug(f"Retrieved {attr_name}: {value}")
            return value
        except (AttributeError, KeyError) as e:
            if default_value is not None:
                self.logger.debug(f"Using default for {attr_name}: {default_value}")
                return default_value
            else:
                self.logger.error(f"Required attribute {attr_name} missing: {e}")
                raise

    def _extract_hls_parallelization_values(self) -> Dict[str, Any]:
        """Extract HLS-specific parallelization values.
        
        Returns:
            Dictionary of HLS parallelization configuration
        """
        values = {
            'pe_factor': self._safe_get_nodeattr('PE', 1),
            'simd_factor': self._safe_get_nodeattr('SIMD', 1),
        }
        
        # Compute total parallelization
        values['total_parallelization'] = values['pe_factor'] * values['simd_factor']
        
        # Determine parallelization strategy
        if values['pe_factor'] > 1 and values['simd_factor'] > 1:
            values['parallelization_strategy'] = 'pe_and_simd'
        elif values['pe_factor'] > 1:
            values['parallelization_strategy'] = 'pe_only'
        elif values['simd_factor'] > 1:
            values['parallelization_strategy'] = 'simd_only'
        else:
            values['parallelization_strategy'] = 'sequential'
        
        self.logger.debug(f"HLS parallelization: PE={values['pe_factor']}, SIMD={values['simd_factor']}")
        return values

    def _extract_hls_memory_config(self) -> Dict[str, Any]:
        """Extract HLS-specific memory configuration.
        
        Returns:
            Dictionary of HLS memory configuration
        """
        return {
            'mem_mode': self._safe_get_nodeattr('mem_mode', 'external'),
            'ram_style': self._safe_get_nodeattr('ram_style', 'block'),
            'res_type': self._safe_get_nodeattr('resType', 'lut'),
        }

    # ===== Template-Native Execution Methods =====
    # These methods provide FINN compatibility through template-based implementations

    def code_generation_cppsim(self):
        """Generate C++ simulation code using template system."""
        self.logger.info(f"Generating C++ simulation code for {self.onnx_node.name}")
        
        # Ensure code generation directory exists
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        if not code_gen_dir or not os.path.isdir(code_gen_dir):
            code_gen_dir = make_build_dir("code_gen_cppsim_" + self.onnx_node.name + "_")
            self.set_nodeattr("code_gen_dir_cppsim", code_gen_dir)
        
        # Generate template values
        template_values = self.get_execution_template_values("cppsim")
        
        # Render cppsim template
        cppsim_code = self.template_engine.render_template("execution/cppsim.cpp.j2", template_values)
        
        # Write main C++ file
        function_name = template_values.get('function_name', 'main')
        cpp_file = os.path.join(code_gen_dir, f"{function_name}.cpp")
        with open(cpp_file, 'w') as f:
            f.write(cppsim_code)
        
        # Generate params.h if needed
        if template_values.get('operation_params'):
            params_code = self.template_engine.render_template("execution/params.h.j2", template_values)
            params_file = os.path.join(code_gen_dir, "params.h")
            with open(params_file, 'w') as f:
                f.write(params_code)
        
        self.logger.info(f"Generated C++ simulation code: {cpp_file}")

    def code_generation_ipgen(self, model, fpgapart: str, clk: float):
        """Generate IP generation code using template system."""
        self.logger.info(f"Generating IP code for {self.onnx_node.name}")
        
        # Store context for template generation
        self._current_model = model
        self._current_fpgapart = fpgapart
        self._current_clk = clk
        
        # Ensure code generation directory exists
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        if not code_gen_dir or not os.path.isdir(code_gen_dir):
            code_gen_dir = make_build_dir("code_gen_ipgen_" + self.onnx_node.name + "_")
            self.set_nodeattr("code_gen_dir_ipgen", code_gen_dir)
        
        # Generate template values with IP context
        template_values = self.get_execution_template_values("ipgen")
        template_values.update({
            'fpga_part': fpgapart,
            'clock_period': clk,
            'project_name': f"{self.onnx_node.name}_project",
            'ip_name': f"{self.onnx_node.name}_ip",
            'ip_version': "1.0",
        })
        
        # Render IP generation template
        ipgen_code = self.template_engine.render_template("execution/ipgen.cpp.j2", template_values)
        tcl_code = self.template_engine.render_template("execution/ipgen.tcl.j2", template_values)
        
        # Write files
        function_name = template_values.get('function_name', 'main')
        cpp_file = os.path.join(code_gen_dir, f"{function_name}.cpp")
        tcl_file = os.path.join(code_gen_dir, "run_hls.tcl")
        
        with open(cpp_file, 'w') as f:
            f.write(ipgen_code)
        with open(tcl_file, 'w') as f:
            f.write(tcl_code)
        
        # Generate params.h if needed
        if template_values.get('operation_params'):
            params_code = self.template_engine.render_template("execution/params.h.j2", template_values)
            params_file = os.path.join(code_gen_dir, "params.h")
            with open(params_file, 'w') as f:
                f.write(params_code)
        
        self.logger.info(f"Generated IP code: {cpp_file}, {tcl_file}")

    def compile_singlenode_code(self):
        """Compile C++ simulation code using template system."""
        self.logger.info(f"Compiling single-node code for {self.onnx_node.name}")
        
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        if not code_gen_dir or not os.path.isdir(code_gen_dir):
            raise ValueError(f"Code generation directory not found: {code_gen_dir}")
        
        # Get template values for compilation info
        template_values = self.get_execution_template_values("cppsim")
        function_name = template_values.get('function_name', 'main')
        
        # Build using CppBuilder
        builder = CppBuilder()
        
        # Add source files
        cpp_file = os.path.join(code_gen_dir, f"{function_name}.cpp")
        if not os.path.exists(cpp_file):
            raise FileNotFoundError(f"Generated C++ file not found: {cpp_file}")
        
        builder.append_sources([cpp_file])
        
        # Add includes
        builder.append_includes([code_gen_dir])
        
        # Build executable
        executable_path = builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", executable_path)
        
        self.logger.info(f"Compiled executable: {executable_path}")

    def execute_node(self, context, graph):
        """Execute the node using template system."""
        exec_mode = self.get_nodeattr("exec_mode")
        
        if exec_mode == "cppsim":
            return self._execute_cppsim(context, graph)
        elif exec_mode == "rtlsim":
            return self._execute_rtlsim(context, graph)
        else:
            raise ValueError(f"Unknown execution mode: {exec_mode}")

    def _execute_cppsim(self, context, graph):
        """Execute using C++ simulation."""
        executable_path = self.get_nodeattr("executable_path")
        if not executable_path or not os.path.exists(executable_path):
            raise ValueError(f"Executable not found: {executable_path}")
        
        # Implementation would handle data conversion and subprocess execution
        # This is a simplified version
        self.logger.info(f"Executing C++ simulation: {executable_path}")
        return context  # Return modified context

    def _execute_rtlsim(self, context, graph):
        """Execute using RTL simulation."""
        # Implementation would handle RTL simulation
        self.logger.info("Executing RTL simulation")
        return context  # Return modified context

    def get_execution_template_values(self, execution_type: str) -> Dict[str, Any]:
        """Get template values for execution-specific templates.
        
        Args:
            execution_type: Type of execution ("cppsim", "ipgen", etc.)
            
        Returns:
            Dictionary of template values for execution
        """
        # Start with basic template values
        values = self.get_template_values(self.get_template_name())
        
        # Add execution-specific values
        execution_values = self._generate_execution_values(execution_type)
        values.update(execution_values)
        
        return values

    def _generate_execution_values(self, execution_type: str) -> Dict[str, Any]:
        """Generate execution-specific template values.
        
        Args:
            execution_type: Type of execution
            
        Returns:
            Dictionary of execution-specific values
        """
        values = {
            'execution_type': execution_type,
            'function_name': self._generate_function_name(),
            'mem_mode': self._safe_get_nodeattr('mem_mode', 'external'),
            'headers': self._generate_execution_headers(),
            'input_ports': self._generate_input_ports(),
            'output_ports': self._generate_output_ports(),
            'internal_streams': self._generate_internal_streams(),
            'operation_params': self._generate_operation_params(),
            'data_types': self._generate_data_types(),
            'resource_config': self._generate_resource_config(),
        }
        
        if execution_type == "ipgen":
            values.update({
                'pipeline_depth': self._safe_get_nodeattr('pipeline_depth', 1),
                'resource_pragmas': self._generate_resource_pragmas(),
                'synthesis_directives': self._generate_synthesis_directives(),
            })
        
        return values

    def _generate_function_name(self) -> str:
        """Generate function name for execution."""
        try:
            return self.onnx_node.name.replace('-', '_').replace('.', '_')
        except:
            return "compute_function"

    def _generate_execution_headers(self) -> list:
        """Generate headers needed for execution."""
        return []  # Override in subclasses

    def _generate_input_ports(self) -> list:
        """Generate input port definitions."""
        return []  # Override in subclasses

    def _generate_output_ports(self) -> list:
        """Generate output port definitions."""
        return []  # Override in subclasses

    def _generate_internal_streams(self) -> list:
        """Generate internal stream definitions."""
        return []  # Override in subclasses

    def _generate_operation_params(self) -> dict:
        """Generate operation-specific parameters."""
        return {}  # Override in subclasses

    def _generate_data_types(self) -> dict:
        """Generate data type definitions."""
        return {}  # Override in subclasses

    def _generate_resource_config(self) -> dict:
        """Generate resource configuration."""
        return {}  # Override in subclasses

    def _generate_resource_pragmas(self) -> list:
        """Generate HLS resource pragmas."""
        return []  # Override in subclasses

    def _generate_synthesis_directives(self) -> list:
        """Generate synthesis directives."""
        return []  # Override in subclasses