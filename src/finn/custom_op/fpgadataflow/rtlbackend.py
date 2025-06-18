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

import numpy as np
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Optional

from finn.util.basic import make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.codegen.codegen import Codegen
from finn.codegen import TemplateEngine

try:
    import pyxsi_utils
except ModuleNotFoundError:
    pyxsi_utils = None


class RTLBackend(Codegen):
    """Clean RTL Backend class using direct template value generation.
    
    Provides RTL-specific code generation functionality for FINN custom ops
    that correspond to modules in finn-rtllib. Uses explicit template declaration
    and direct value generation without legacy compatibility layers.
    """

    # ===== Explicit Template Declaration =====
    # These should be overridden by concrete operation classes
    TEMPLATE_NAME: Optional[str] = None
    TEMPLATE_OPTIONS: Optional[Dict[str, str]] = None

    def __init__(self, **kwargs):
        """Initialize RTL backend with clean Codegen infrastructure."""
        # Extract RTL-specific kwargs to avoid conflicts
        rtl_kwargs = {k: v for k, v in kwargs.items() if k.startswith('rtl_')}
        
        # Initialize parent Codegen class
        super().__init__()
        
        # Initialize template engine
        self.template_engine = TemplateEngine()
        
        # RTL-specific initialization
        self.rtl_template_path = "rtl/"
        
        # Context for template values
        self._current_model = None
        self._current_fpgapart = None
        self._current_clk = None
        
        # Apply RTL-specific configurations
        for key, value in rtl_kwargs.items():
            setattr(self, key, value)

    # ===== Explicit Template Interface Implementation =====

    def get_template_name(self) -> str:
        """Get template name - explicit declaration required.
        
        Returns:
            Name of template to use for code generation
            
        Raises:
            NotImplementedError: If no template declared
        """
        # Check for instance override first
        if hasattr(self, '_template_override'):
            return self._template_override
            
        # Use class-level declaration
        if self.TEMPLATE_NAME:
            return self.TEMPLATE_NAME
            
        if self.TEMPLATE_OPTIONS:
            return self._select_template_from_options()
            
        raise NotImplementedError(
            f"{self.__class__.__name__} must declare TEMPLATE_NAME or TEMPLATE_OPTIONS"
        )
    
    def _select_template_from_options(self) -> str:
        """Select template from available options. Override in subclass for logic.
        
        Returns:
            Template name from TEMPLATE_OPTIONS
        """
        if not self.TEMPLATE_OPTIONS:
            raise NotImplementedError("No template options available")
        
        # Default: return first option
        return next(iter(self.TEMPLATE_OPTIONS.values()))
    
    def set_template_override(self, template_name: str):
        """Allow runtime template override.
        
        Args:
            template_name: Template name to use instead of class declaration
        """
        self._template_override = template_name

    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for RTL template.
        
        Args:
            template_name: Name of template to extract values for
            
        Returns:
            Dictionary mapping template placeholders to values
        """
        # Base RTL values
        template_values = self._extract_common_values(self)
        
        # Template-specific values
        if 'thresholding' in template_name:
            template_values.update(self._get_thresholding_template_values())
        elif 'swg' in template_name:
            template_values.update(self._get_swg_template_values())
        else:
            # Generic RTL wrapper values
            template_values.update(self._get_generic_wrapper_values())
        
        return template_values

    def _get_thresholding_template_values(self) -> Dict[str, Any]:
        """Get values for thresholding wrapper template."""
        return {
            'MODULE_NAME_AXI_WRAPPER': f"{self.onnx_node.name}_wrapper",
            'N': self._safe_extract_value(self, 'NumSteps', 8),
            'WI': self._safe_extract_value(self, 'inputDataType', 8),
            'WT': self._safe_extract_value(self, 'weightDataType', 8),
            'C': self._safe_extract_value(self, 'NumChannels', 32),
            'PE': self._safe_extract_value(self, 'PE', 4),
            'SIGNED': 0,
            'FPARG': 0,
            'BIAS': 0,
            'THRESHOLDS_PATH': '""',
            'USE_AXILITE': 1,
            'DEPTH_TRIGGER_URAM': 0,
            'DEPTH_TRIGGER_BRAM': 0,
            'DEEP_PIPELINE': 0,
            'O_BITS': self._safe_extract_value(self, 'NumSteps', 8),
        }

    def _get_swg_template_values(self) -> Dict[str, Any]:
        """Get values for SWG wrapper template."""
        return {
            'TOP_MODULE_NAME': f"{self.onnx_node.name}_wrapper",
            'BIT_WIDTH': 8,
            'SIMD': self._safe_extract_value(self, 'SIMD', 4),
            'MMV_IN': 32,
            'MMV_OUT': 32,
            'IN_WIDTH_PADDED': 256,
            'OUT_WIDTH_PADDED': 256,
        }

    def _get_generic_wrapper_values(self) -> Dict[str, Any]:
        """Get values for generic RTL wrapper."""
        return {
            'MODULE_NAME': f"{self.onnx_node.name}_wrapper",
            'DATA_WIDTH': self._extract_data_width(self),
            'PE_COUNT': self._safe_extract_value(self, 'PE', 1),
        }

    # ===== RTL-Specific Methods =====

    def generate_rtl_code(self) -> str:
        """Generate RTL code - uses inherited generate_code().
        
        Returns:
            Generated RTL code as string
        """
        return self.generate_code()

    def _extract_rtl_interface_values(self, operation) -> Dict[str, Any]:
        """Extract RTL-specific interface values.
        
        Args:
            operation: Operation instance to extract from
            
        Returns:
            Dictionary of RTL interface values
        """
        return {
            'module_name': self._generate_module_name(operation),
            'data_width': self._extract_data_width(operation),
            'clock_enable': True,
            'reset_style': 'sync',
            'interface_type': 'axi_stream',
        }

    def _generate_module_name(self, operation) -> str:
        """Generate RTL module name.
        
        Args:
            operation: Operation instance
            
        Returns:
            Generated module name
        """
        op_type = operation.onnx_node.op_type.lower()
        pe_factor = self._safe_extract_value(operation, 'PE', 1)
        return f"{op_type}_{pe_factor}pe"

    def _extract_data_width(self, operation) -> int:
        """Extract data width for RTL.
        
        Args:
            operation: Operation instance
            
        Returns:
            Data width in bits
        """
        try:
            return operation.get_input_datatype().bitwidth()
        except:
            self.logger.warning("Could not extract data width, using default 8")
            return 8  # Default data width

    # ===== Legacy RTL Backend Functionality =====
    # All existing methods preserved for backward compatibility

    def get_nodeattr_types(self) -> Dict[str, Any]:
        """Get RTL backend node attribute types.
        
        Merges any operation-specific attributes with RTL backend attributes.
        RTL attributes take precedence in case of conflicts.
        
        Returns:
            Dictionary of node attribute specifications
        """
        # Base RTL backend attributes
        rtl_attrs = {
            # attribute to save top module name - not user configurable
            "gen_top_module": ("s", False, ""),
            "code_gen_dir_ipgen": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
            # Template override support
            "rtl_template_override": ("s", False, ""),
        }
        
        # Try to get operation-specific attributes if this is a multiple inheritance case
        operation_attrs = {}
        for base in self.__class__.__bases__:
            if hasattr(base, 'get_nodeattr_types') and base != RTLBackend:
                try:
                    operation_attrs = base.get_nodeattr_types(self)
                    break
                except Exception:
                    pass
        
        # RTL attributes override operation attributes (explicit policy)
        merged_attrs = {**operation_attrs, **rtl_attrs}
        
        # Log any conflicts for debugging
        conflicts = set(operation_attrs.keys()) & set(rtl_attrs.keys())
        if conflicts:
            self.logger.debug(f"RTL attributes override operation attributes: {conflicts}")
        
        return merged_attrs

    def generate_hdl(self, model, fpgapart, clk):
        """Generate HDL code using template system.
        
        Args:
            model: FINN model
            fpgapart: Target FPGA part
            clk: Clock period
        """
        # Store context
        self._current_model = model
        self._current_fpgapart = fpgapart
        self._current_clk = clk
        
        # Generate using template system
        rtl_code = self.generate_code()  # Uses get_template_name() and get_template_values()
        
        # Write to file
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        module_name = self._generate_module_name(self)
        rtl_path = os.path.join(code_gen_dir, f"{module_name}.sv")
        
        with open(rtl_path, "w") as f:
            f.write(rtl_code)
        
        # Set node attributes
        self.set_nodeattr("gen_top_module", module_name)

    def prepare_rtlsim(self):
        """Creates a xsi emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path."""

        verilog_files = self.get_rtl_file_list(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        ret = pyxsi_utils.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir, debug
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

    def get_verilog_paths(self):
        """Returns path to code gen directory. Can be overwritten to
        return additional paths to relevant verilog files"""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        return [code_gen_dir]

    @abstractmethod
    def get_rtl_file_list(self, abspath=False):
        """Returns list of rtl files. Needs to be filled by each node.
        
        Args:
            abspath: Whether to return absolute paths
            
        Returns:
            List of RTL file paths
        """
        pass

    @abstractmethod
    def code_generation_ipi(self):
        """Generate IPI (IP Integrator) code for this operation."""
        pass

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate IP for this operation.
        
        Args:
            model: FINN model
            fpgapart: Target FPGA part
            clk: Clock period
        """
        self.generate_hdl(model, fpgapart, clk)

    def execute_node(self, context, graph):
        """Execute this node in the given context.
        
        Args:
            context: Execution context
            graph: Model graph
        """
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        if mode == "rtlsim":
            node = self.onnx_node
            inputs = {}
            for i, inp in enumerate(node.input):
                exp_ishape = tuple(self.get_normal_input_shape(i))
                folded_ishape = self.get_folded_input_shape(i)
                inp_val = context[inp]
                assert str(inp_val.dtype) == "float32", "Input datatype is not float32"
                assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
                export_idt = self.get_input_datatype(i)

                reshaped_input = inp_val.reshape(folded_ishape)
                np.save(os.path.join(code_gen_dir, "input_%s.npy" % i), reshaped_input)
                nbits = self.get_instream_width(i)
                rtlsim_inp = npy_to_rtlsim_input(
                    "{}/input_{}.npy".format(code_gen_dir, i), export_idt, nbits
                )
                inputs["in%s" % i] = rtlsim_inp
            outputs = {}
            for o, outp in enumerate(node.output):
                outputs["out%s" % o] = []
            # assembled execution context
            io_dict = {"inputs": inputs, "outputs": outputs}

            sim = self.get_rtlsim()
            self.reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            self.close_rtlsim(sim)
            for o, outp in enumerate(node.output):
                rtlsim_output = io_dict["outputs"]["out%s" % o]
                odt = self.get_output_datatype(o)
                target_bits = odt.bitwidth()
                packed_bits = self.get_outstream_width(o)
                out_npy_path = "{}/output.npy".format(code_gen_dir)
                out_shape = self.get_folded_output_shape(o)
                rtlsim_output_to_npy(
                    rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
                )
                # load and reshape output
                exp_oshape = tuple(self.get_normal_output_shape(o))
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
                context[outp] = output

                assert (
                    context[outp].shape == exp_oshape
                ), "Output shape doesn't match expected shape."

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
