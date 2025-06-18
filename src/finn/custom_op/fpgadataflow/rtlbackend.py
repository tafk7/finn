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
from abc import ABC, abstractmethod
from typing import Dict, Any, List

from finn.codegen.codegen import Codegen
from finn.util.basic import make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

try:
    import pyxsi_utils
except ModuleNotFoundError:
    pyxsi_utils = None


class CG_RTLBackend(Codegen):
    """
    Clean RTL Backend class without legacy compatibility bloat.
    
    Provides RTL-specific code generation functionality for FINN custom ops
    using only Jinja2-based template system with direct value generation.
    No legacy compatibility methods or complex string replacement.
    """

    def __init__(self, **kwargs):
        """Initialize clean RTL backend."""
        super().__init__()
        
        # RTL-specific initialization without legacy bloat
        self.logger = logging.getLogger(f"finn.codegen.{self.__class__.__name__}")
        
        # Apply RTL-specific configurations
        for key, value in kwargs.items():
            if key.startswith('rtl_'):
                setattr(self, key, value)
        
        self.logger.debug(f"Initialized clean RTL backend: {self.__class__.__name__}")

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
        self.logger.debug(f"Generating RTL template values for: {template_name}")
        
        # Build template values through direct generation
        values = self._generate_common_values(self)
        values.update(self._generate_rtl_common_values())
        values.update(self._generate_operation_specific_values(template_name))
        
        # Validate and clean values
        validated_values = self._validate_template_values(values)
        
        self.logger.debug(f"Generated {len(validated_values)} RTL template values")
        return validated_values

    def _generate_rtl_common_values(self) -> Dict[str, Any]:
        """Generate values common to all RTL operations.
        
        Returns:
            Dictionary of common RTL template values
        """
        return {
            'MODULE_NAME': self._generate_module_name(),
            'CLK_SIGNAL': 'clk',
            'RST_SIGNAL': 'rst_n',
            'DATA_WIDTH': self._extract_data_width(),
            'INTERFACE_TYPE': 'axi_stream',
            'RTL_PARAMETERS': self._generate_rtl_parameters(),
            'PORT_DECLARATIONS': self._generate_port_declarations(),
        }

    def _generate_module_name(self) -> str:
        """Generate clean module name.
        
        Returns:
            Generated RTL module name
        """
        try:
            base_name = self.onnx_node.name.replace('-', '_').replace('.', '_')
            op_type = self.onnx_node.op_type.lower()
            return f"{base_name}_{op_type}"
        except:
            return "rtl_module"

    def _extract_data_width(self) -> int:
        """Extract data width for RTL.
        
        Returns:
            Data width in bits
        """
        try:
            if hasattr(self, 'get_input_datatype'):
                return self.get_input_datatype().bitwidth()
            else:
                return self._safe_get_nodeattr('DataWidth', 8)
        except:
            self.logger.warning("Could not extract data width, using default 8")
            return 8

    def _generate_rtl_parameters(self) -> str:
        """Generate RTL parameter declarations.
        
        Returns:
            String containing RTL parameter declarations
        """
        params = []
        
        # Common RTL parameters
        params.append(f"parameter DATA_WIDTH = {self._extract_data_width()}")
        
        # Add operation-specific parameters
        try:
            if hasattr(self, 'get_nodeattr'):
                pe_count = self._safe_get_nodeattr('PE', 1)
                params.append(f"parameter PE_COUNT = {pe_count}")
        except:
            pass
        
        return ',\n    '.join(params)

    def _generate_port_declarations(self) -> str:
        """Generate RTL port declarations.
        
        Returns:
            String containing RTL port declarations
        """
        data_width = self._extract_data_width()
        
        ports = [
            "// Clock and Reset",
            "input wire clk",
            "input wire rst_n",
            "",
            "// AXI Stream Input",
            f"input wire [{data_width-1}:0] s_axis_tdata",
            "input wire s_axis_tvalid",
            "output wire s_axis_tready",
            "",
            "// AXI Stream Output", 
            f"output wire [{data_width-1}:0] m_axis_tdata",
            "output wire m_axis_tvalid",
            "input wire m_axis_tready"
        ]
        
        return '\n    '.join(ports)

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
                
            # Convert values to appropriate types for template rendering
            if isinstance(value, (int, float, bool)):
                validated[key] = str(value)
            elif isinstance(value, str):
                validated[key] = value
            else:
                # Convert other types to string representation
                validated[key] = str(value)
        
        return validated

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
        """Get RTL backend node attribute types.
        
        Returns:
            Dictionary of node attribute specifications
        """
        # Base RTL backend attributes
        rtl_attrs = {
            # RTL generation attributes
            "gen_top_module": ("s", False, ""),
            "code_gen_dir_ipgen": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
            
            # RTL simulation attributes
            "rtlsim_so": ("s", False, ""),
            "rtlsim_trace": ("s", False, ""),
        }
        
        # Try to get operation-specific attributes from parent classes
        operation_attrs = {}
        for base in self.__class__.__bases__:
            if hasattr(base, 'get_nodeattr_types') and base != CG_RTLBackend:
                try:
                    operation_attrs = base.get_nodeattr_types(self)
                    break
                except Exception:
                    pass
        
        # RTL attributes override operation attributes for clean separation
        merged_attrs = {**operation_attrs, **rtl_attrs}
        
        # Log any conflicts for debugging
        conflicts = set(operation_attrs.keys()) & set(rtl_attrs.keys())
        if conflicts:
            self.logger.debug(f"RTL attributes override operation attributes: {conflicts}")
        
        return merged_attrs

    # ===== RTL-Specific Utility Methods =====

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

    def _extract_rtl_interface_values(self) -> Dict[str, Any]:
        """Extract RTL-specific interface values.
        
        Returns:
            Dictionary of RTL interface configuration
        """
        return {
            'module_name': self._generate_module_name(),
            'data_width': self._extract_data_width(),
            'clock_enable': True,
            'reset_style': 'async_neg',  # Active low async reset
            'interface_type': 'axi_stream',
            'pipeline_depth': self._safe_get_nodeattr('pipeline_depth', 1),
        }

    def _extract_rtl_timing_values(self) -> Dict[str, Any]:
        """Extract RTL-specific timing values.
        
        Returns:
            Dictionary of RTL timing configuration
        """
        return {
            'clock_period': self._safe_get_nodeattr('clock_period', 10.0),  # ns
            'setup_time': 0.5,  # ns
            'hold_time': 0.1,   # ns
            'max_delay': self._safe_get_nodeattr('max_delay', 8.0),  # ns
        }

    # ===== RTL Simulation Support =====

    def prepare_rtlsim(self):
        """Creates a xsi emulation library for RTL simulation.
        
        Sets the rtlsim_so attribute to the path of the compiled simulation library.
        """
        if pyxsi_utils is None:
            raise RuntimeError("pyxsi_utils not available for RTL simulation")
        
        verilog_files = self.get_rtl_file_list(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        
        ret = pyxsi_utils.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir, debug
        )
        
        # Save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])
        self.logger.debug(f"RTL simulation library created: {ret[0]}/{ret[1]}")

    def get_verilog_paths(self) -> List[str]:
        """Returns paths containing Verilog files for this operation.
        
        Returns:
            List of directory paths containing Verilog files
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        return [code_gen_dir] if code_gen_dir else []

    @abstractmethod
    def get_rtl_file_list(self, abspath: bool = False) -> List[str]:
        """Returns list of RTL files for this operation.
        
        Must be implemented by operation-specific subclasses.
        
        Args:
            abspath: Whether to return absolute paths
            
        Returns:
            List of RTL file paths
        """
        pass

    @abstractmethod
    def get_verilog_top_module_name(self) -> str:
        """Get the name of the top-level Verilog module.
        
        Must be implemented by operation-specific subclasses.
        
        Returns:
            Name of top-level module
        """
        pass

    # ===== Code Generation Methods =====

    def generate_hdl(self, model, fpgapart: str, clk: float):
        """Generate HDL code using template system.
        
        Args:
            model: FINN model
            fpgapart: Target FPGA part
            clk: Clock period in ns
        """
        self.logger.info(f"Generating HDL for {self.onnx_node.name}")
        
        # Store context for template generation
        self._current_model = model
        self._current_fpgapart = fpgapart
        self._current_clk = clk
        
        # Generate using template system
        rtl_code = self.generate_code()
        
        # Write to file
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        module_name = self._generate_module_name()
        rtl_path = os.path.join(code_gen_dir, f"{module_name}.sv")
        
        with open(rtl_path, "w") as f:
            f.write(rtl_code)
        
        # Set node attributes
        self.set_nodeattr("gen_top_module", module_name)
        self.logger.info(f"Generated HDL: {rtl_path}")

    def code_generation_ipgen(self, model, fpgapart: str, clk: float):
        """Generate IP for this operation.
        
        Args:
            model: FINN model
            fpgapart: Target FPGA part
            clk: Clock period in ns
        """
        self.generate_hdl(model, fpgapart, clk)

    @abstractmethod
    def code_generation_ipi(self) -> List[str]:
        """Generate IPI (IP Integrator) TCL commands.
        
        Must be implemented by operation-specific subclasses.
        
        Returns:
            List of TCL commands for IP Integrator
        """
        pass

    # ===== REMOVED: All Legacy Methods =====
    # The following methods are intentionally NOT implemented to eliminate legacy bloat:
    # - Any string replacement template logic
    # - Complex template selection mechanisms  
    # - Legacy compatibility methods
    # - Manual Verilog generation methods
    #
    # These are replaced by the direct template value generation methods above.