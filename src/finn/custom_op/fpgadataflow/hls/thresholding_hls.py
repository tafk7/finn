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

import os
from typing import Dict, Any
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.custom_op.fpgadataflow.CG_hlsbackend import CG_HLSBackend


class CG_ThresholdingHLS(Thresholding, CG_HLSBackend):
    """
    Clean thresholding HLS implementation with current inheritance.
    
    Inherits from both Thresholding and CG_HLSBackend to maintain compatibility
    while using clean Jinja2-based template generation without legacy bloat.
    """
    
    # Explicit template declaration
    TEMPLATE_NAME = "thresholding/hls/docompute.cpp.j2"
    TEMPLATE_FALLBACKS = ["hls/docompute.cpp.j2"]
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize clean thresholding HLS backend with current inheritance."""
        # Maintain current inheritance structure
        Thresholding.__init__(self, onnx_node, **kwargs)
        CG_HLSBackend.__init__(self, **kwargs)
        
        self.logger.debug(f"Initialized CG_ThresholdingHLS for node: {onnx_node.name}")

    def _generate_operation_specific_values(self, template_name: str) -> Dict[str, Any]:
        """Generate thresholding-specific template values.
        
        Args:
            template_name: Name of template being generated for
            
        Returns:
            Dictionary of thresholding-specific template values
        """
        values = {}
        
        # Generate core thresholding values
        values['DEFINES'] = self._generate_thresholding_defines()
        values['DOCOMPUTE'] = self._generate_thresholding_compute()
        
        # Add template-specific values
        if 'docompute' in template_name:
            values.update({
                'READNPYDATA': self._generate_read_npy_data(),
                'DATAOUTSTREAM': self._generate_data_out_stream(),
                'SAVEASCNPY': self._generate_save_as_npy(),
            })
            
            # Add timeout values for timeout template variant
            if 'timeout' in template_name:
                values.update({
                    'TIMEOUT_VALUE': self._generate_timeout_value(),
                    'TIMEOUT_CONDITION': self._generate_timeout_condition(),
                    'TIMEOUT_READ_STREAM': self._generate_timeout_read_stream(),
                })
        
        elif 'ipgen' in template_name:
            if template_name.endswith('.cpp.j2'):
                values['BLACKBOXFUNCTION'] = self._generate_blackbox_function()
            elif template_name.endswith('.tcl.j2'):
                values.update(self._generate_ipgen_tcl_values())
        
        return values

    def _generate_thresholding_defines(self) -> str:
        """Generate thresholding defines directly.
        
        Returns:
            String containing thresholding-specific #define statements
        """
        num_input_vectors = self.get_nodeattr('numInputVectors')
        num_reps = num_input_vectors[0] if num_input_vectors else 1
        
        defines = [
            f"#define NumChannels1 {self.get_nodeattr('NumChannels')}",
            f"#define PE1 {self.get_nodeattr('PE')}",
            f"#define numReps {num_reps}",
            f"#define numSteps {self.get_nodeattr('numSteps')}",
        ]
        
        # Add threshold memory size if needed
        tmem = self.calc_tmem()
        if tmem > 0:
            defines.append(f"#define TMEM {tmem}")
        
        return '\n'.join(defines)

    def _generate_thresholding_compute(self) -> str:
        """Generate thresholding compute logic directly.
        
        Returns:
            String containing the thresholding computation call
        """
        inp_hls_str = self.get_input_datatype().get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        
        # Get threshold info
        tmem = self.calc_tmem()
        
        if tmem == 0:
            raise Exception("Unexpected thresholding config: #thresholds=0")
        
        tdt = self.get_input_datatype(1)
        thold_hls_str = tdt.get_hls_datatype_str()
        
        compute_call = f"""Thresholding_Batch<{inp_hls_str}, {out_hls_str}, {thold_hls_str}, NumChannels1, PE1, {tmem}>
        (in0_V, out0_V, threshs.parameters, numReps);"""
        
        return compute_call

    def _generate_read_npy_data(self) -> str:
        """Generate read npy data logic directly.
        
        Returns:
            String containing npy data reading code
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        
        # Get input data type and stream info
        dtype = self.get_input_datatype()
        if dtype.name == "BIPOLAR":
            dtype = dtype.get_binary_equivalent()
        
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = f"{code_gen_dir}/input_0.npy"
        
        # Stream configuration
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        
        cpp_interface = self._safe_get_nodeattr("cpp_interface", "packed")
        
        if cpp_interface == "packed":
            read_code = f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("{npy_in}", in0_V);'
        else:
            folded_shape = self.get_folded_input_shape()
            read_code = f'npy2vectorstream<{elem_hls_type}, {npy_type}, {folded_shape[-1]}>("{npy_in}", in0_V, false);'
        
        return read_code

    def _generate_data_out_stream(self) -> str:
        """Generate data output stream logic directly.
        
        Returns:
            String containing data output streaming code
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        
        # Get output data type and stream info
        dtype = self.get_output_datatype()
        if dtype.name == "BIPOLAR":
            dtype = dtype.get_binary_equivalent()
        
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = f"{code_gen_dir}/output_0.npy"
        oshape = self.get_folded_output_shape()
        oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")
        
        cpp_interface = self._safe_get_nodeattr("cpp_interface", "packed")
        
        if cpp_interface == "packed":
            elem_bits = dtype.bitwidth()
            packed_bits = self.get_outstream_width()
            packed_hls_type = f"ap_uint<{packed_bits}>"
            
            out_code = f'apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>(out0_V, {oshape_cpp_str}, "{npy_out}");'
        else:
            folded_shape = self.get_folded_output_shape()
            out_code = f'vectorstream2npy<{elem_hls_type}, {npy_type}, {folded_shape[-1]}>(strm, {oshape_cpp_str}, "{npy_out}");'
        
        return out_code

    def _generate_save_as_npy(self) -> str:
        """Generate save as npy logic directly.
        
        Returns:
            String containing save functionality (usually empty for thresholding)
        """
        return "// Save functionality handled by dataoutstream"

    def _generate_blackbox_function(self) -> str:
        """Generate blackbox function signature directly.
        
        Returns:
            String containing blackbox function signature
        """
        return f"""void {self.onnx_node.name}(hls::stream<ap_uint<{self.get_instream_width()}>> &in0_V,
    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V)"""

    def _generate_timeout_value(self) -> str:
        """Generate timeout value for timeout templates.
        
        Returns:
            String containing timeout value
        """
        return "1000"

    def _generate_timeout_condition(self) -> str:
        """Generate timeout condition for timeout templates.
        
        Returns:
            String containing timeout condition
        """
        return "out0_V.empty()"

    def _generate_timeout_read_stream(self) -> str:
        """Generate timeout read stream for timeout templates.
        
        Returns:
            String containing timeout read stream code
        """
        return "strm << out0_V.read();"

    def _generate_ipgen_tcl_values(self) -> Dict[str, Any]:
        """Generate values for IPGen TCL template.
        
        Returns:
            Dictionary containing TCL template values
        """
        return {
            'PROJECTNAME': f"project_{self.onnx_node.name}",
            'HWSRCDIR': self.get_nodeattr("code_gen_dir_ipgen"),
            'FPGAPART': getattr(self, '_current_fpgapart', "xc7z020clg400-1"),
            'TOPFXN': self.onnx_node.name,
            'CLKPERIOD': getattr(self, '_current_clk', 10),
            'DEFAULT_DIRECTIVES': self._generate_default_directives(),
            'EXTRA_DIRECTIVES': self._generate_extra_directives(),
        }

    def _generate_default_directives(self) -> str:
        """Generate default HLS directives.
        
        Returns:
            String containing default HLS directives
        """
        directives = [
            "set_param hls.enable_hidden_option_error false",
            "config_compile -disable_unroll_code_size_check -pipeline_style flp",
            "config_interface -m_axi_addr64",
            "config_rtl -module_auto_prefix",
            "config_rtl -deadlock_detection none",
        ]
        return '\n'.join(directives)

    def _generate_extra_directives(self) -> str:
        """Generate extra HLS directives.
        
        Returns:
            String containing extra HLS directives (empty by default)
        """
        return ""

    def _generate_hls_pragmas(self) -> str:
        """Override to add thresholding-specific pragmas.
        
        Returns:
            String containing HLS pragmas with thresholding-specific additions
        """
        pragmas = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V", 
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        ]
        
        # Add array partition pragmas for thresholds
        ram_style = self._safe_get_nodeattr("ram_style", "auto")
        if self.calc_tmem() != 0:
            pragmas.extend([
                "#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=1",
                "#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=3"
            ])
            
            if ram_style == "distributed":
                pragmas.append("#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_LUTRAM")
            elif ram_style == "block":
                pragmas.append("#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_BRAM")
        
        return '\n'.join(pragmas)

    def _generate_hls_globals(self) -> str:
        """Override to add thresholding-specific globals.
        
        Returns:
            String containing thresholding-specific global declarations
        """
        globals_list = [
            '#include "activations.hpp"',
            '#include "params.h"'
        ]
        return '\n'.join(globals_list)

    # ===== Code Generation Entry Points =====

    def code_generation_cppsim(self, model):
        """Generate C++ simulation code using Jinja2 templates.
        
        Args:
            model: FINN model containing this operation
        """
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        
        self.logger.info(f"Generating C++ simulation code for {node.name}")
        
        # Generate parameter files first
        self.generate_params(model, path)
        
        # Determine template based on interface
        cpp_interface = self._safe_get_nodeattr("cpp_interface", "packed")
        if cpp_interface == "hls_vector":
            template_name = "thresholding/hls/docompute_timeout.cpp.j2"
        else:
            template_name = "thresholding/hls/docompute.cpp.j2"
        
        # Generate code using template engine
        cpp_code = self.generate_code()
        
        # Write file
        cpp_path = os.path.join(path, f"execute_{node.op_type}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        
        self.logger.info(f"Generated C++ simulation: {cpp_path}")

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate IP generation files using Jinja2 templates.
        
        Args:
            model: FINN model containing this operation
            fpgapart: Target FPGA part
            clk: Clock period
        """
        # Store context for template generation
        self._current_fpgapart = fpgapart
        self._current_clk = clk
        
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_ipgen")
        
        self.logger.info(f"Generating IP generation files for {node.name}")
        
        # Generate parameter files first
        self.generate_params(model, path)
        
        # Generate C++ file
        self.__class__.TEMPLATE_NAME = "thresholding/hls/ipgen.cpp.j2"
        cpp_code = self.generate_code()
        
        cpp_path = os.path.join(path, f"top_{node.name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        
        # Generate TCL file
        self.__class__.TEMPLATE_NAME = "thresholding/hls/ipgen.tcl.j2"
        tcl_code = self.generate_code()
        
        tcl_path = os.path.join(path, f"hls_syn_{node.name}.tcl")
        with open(tcl_path, "w") as f:
            f.write(tcl_code)
        
        # Reset template name
        self.__class__.TEMPLATE_NAME = "thresholding/hls/docompute.cpp.j2"
        
        self.logger.info(f"Generated IP files: {cpp_path}, {tcl_path}")