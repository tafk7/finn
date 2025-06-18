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
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.codegen import TemplateEngine


class ThresholdingHLS(Thresholding, HLSBackend):
    """Clean HLS backend for Thresholding operations using direct template value generation."""
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize Thresholding HLS backend with template engine."""
        super().__init__(onnx_node, **kwargs)
        
        # Initialize template engine
        self.template_engine = TemplateEngine()
        
        # Context for code generation
        self._current_fpgapart = None
        self._current_clk = None
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Extract values for Jinja2 templates - NO code_gen_dict."""
        if 'docompute' in template_name:
            return self._get_docompute_values(template_name)
        elif 'ipgen.cpp' in template_name:
            return self._get_ipgen_cpp_values()
        elif 'ipgen.tcl' in template_name:
            return self._get_ipgen_tcl_values()
        else:
            raise ValueError(f"Unsupported template: {template_name}")
    
    def _get_docompute_values(self, template_name: str) -> Dict[str, Any]:
        """Get values for docompute templates."""
        values = {
            'AP_INT_MAX_W': self.get_ap_int_max_w(),
            'GLOBALS': self._generate_globals(),
            'DEFINES': self._generate_defines("cppsim"),
            'PRAGMAS': self._generate_pragmas(),
            'STREAMDECLARATIONS': self._generate_stream_declarations(),
            'READNPYDATA': self._generate_read_npy_data(),
            'DOCOMPUTE': self._generate_docompute(),
            'DATAOUTSTREAM': self._generate_data_out_stream(),
            'SAVEASCNPY': self._generate_save_as_npy(),
        }
        
        if 'timeout' in template_name:
            values.update({
                'TIMEOUT_VALUE': self._generate_timeout_value(),
                'TIMEOUT_CONDITION': self._generate_timeout_condition(),
                'TIMEOUT_READ_STREAM': self._generate_timeout_read_stream(),
            })
        
        return values
    
    def _get_ipgen_cpp_values(self) -> Dict[str, Any]:
        """Get values for ipgen C++ template."""
        return {
            'AP_INT_MAX_W': self.get_ap_int_max_w(),
            'GLOBALS': self._generate_globals(),
            'DEFINES': self._generate_defines("ipgen"),
            'BLACKBOXFUNCTION': self._generate_blackbox_function(),
            'PRAGMAS': self._generate_pragmas(),
            'DOCOMPUTE': self._generate_docompute(),
        }
    
    def _get_ipgen_tcl_values(self) -> Dict[str, Any]:
        """Get values for ipgen TCL template."""
        return {
            'PROJECTNAME': f"project_{self.onnx_node.name}",
            'HWSRCDIR': self.get_nodeattr("code_gen_dir_ipgen"),
            'FPGAPART': self._current_fpgapart,
            'TOPFXN': self.onnx_node.name,
            'CLKPERIOD': self._current_clk,
            'DEFAULT_DIRECTIVES': '\n'.join(self.ipgen_default_directives()),
            'EXTRA_DIRECTIVES': '\n'.join(self.ipgen_extra_directives()),
        }
    
    # Direct template value generation methods (replace code_gen_dict usage)
    def _generate_globals(self) -> str:
        """Generate global includes directly."""
        includes = [
            '#include "activations.hpp"',
            '#include "params.h"'
        ]
        return '\n'.join(includes)
    
    def _generate_defines(self, mode: str) -> str:
        """Generate defines directly."""
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        numReps = numInputVectors[0]
        
        defines = [
            f"#define NumChannels1 {self.get_nodeattr('NumChannels')}",
            f"#define PE1 {self.get_nodeattr('PE')}",
            f"#define numReps {numReps}"
        ]
        return '\n'.join(defines)
    
    def _generate_pragmas(self) -> str:
        """Generate pragmas directly."""
        pragmas = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V", 
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        ]
        
        # Add array partition pragmas for thresholds
        ram_style = self.get_nodeattr("ram_style")
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
    
    def _generate_stream_declarations(self) -> str:
        """Generate stream declarations directly."""
        declarations = [
            f'hls::stream<ap_uint<{self.get_instream_width()}>> in0_V ("in0_V");',
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");'
        ]
        return '\n'.join(declarations)
    
    def _generate_read_npy_data(self) -> str:
        """Generate read npy data directly."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        
        # Get input data type and stream info
        dtype = self.get_input_datatype()
        if dtype.name == "BIPOLAR":
            dtype = self.get_input_datatype().get_binary_equivalent()
        
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = f"{code_gen_dir}/input_0.npy"
        
        # Stream configuration
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        
        return f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>("{npy_in}", in0_V);'
    
    def _generate_docompute(self) -> str:
        """Generate docompute directly."""
        node = self.onnx_node
        inp_hls_str = self.get_input_datatype().get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        
        # Get threshold info
        tmem = self.calc_tmem()
        
        if tmem == 0:
            raise Exception("Unexpected thresholding config: #thresholds=0")
        
        tdt = self.get_input_datatype(1)
        thold_hls_str = tdt.get_hls_datatype_str()
        
        return f"""Thresholding_Batch<{inp_hls_str}, {out_hls_str}, {thold_hls_str}, NumChannels1, PE1, {tmem}>
        (in0_V, out0_V, threshs.parameters, numReps);"""
    
    def _generate_data_out_stream(self) -> str:
        """Generate data output stream directly."""
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
        
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        
        return f'apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, {npy_type}>(out0_V, {oshape_cpp_str}, "{npy_out}");'
    
    def _generate_save_as_npy(self) -> str:
        """Generate save as npy directly."""
        return ""  # Usually empty for thresholding
    
    def _generate_blackbox_function(self) -> str:
        """Generate blackbox function directly."""
        return f"""void {self.onnx_node.name}(hls::stream<ap_uint<{self.get_instream_width()}>> &in0_V,
    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V)"""
    
    def _generate_timeout_value(self) -> str:
        """Generate timeout value directly."""
        return "1000"
    
    def _generate_timeout_condition(self) -> str:
        """Generate timeout condition directly.""" 
        return "out0_V.empty()"
    
    def _generate_timeout_read_stream(self) -> str:
        """Generate timeout read stream directly."""
        return "strm << out0_V.read();"
    
    def code_generation_cppsim(self, model):
        """Generate C++ simulation code using Jinja2 templates."""
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        
        # Generate parameter files first
        self.generate_params(model, path)
        
        # Determine template
        if self.get_nodeattr("cpp_interface") == "hls_vector":
            template_name = "thresholding/hls/docompute_timeout.cpp.j2"
        else:
            template_name = "thresholding/hls/docompute.cpp.j2"
        
        # Generate code using template engine
        template_values = self.get_template_values(template_name)
        cpp_code = self.template_engine.render_template(template_name, template_values)
        
        # Write file
        cpp_path = os.path.join(path, f"execute_{node.op_type}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
    
    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate IP generation files using Jinja2 templates."""
        # Store context
        self._current_fpgapart = fpgapart
        self._current_clk = clk
        
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_ipgen")
        
        # Generate parameter files first
        self.generate_params(model, path)
        
        # Generate C++ file
        cpp_values = self.get_template_values("thresholding/hls/ipgen.cpp.j2")
        cpp_code = self.template_engine.render_template("thresholding/hls/ipgen.cpp.j2", cpp_values)
        
        cpp_path = os.path.join(path, f"top_{node.name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        
        # Generate TCL file
        tcl_values = self.get_template_values("thresholding/hls/ipgen.tcl.j2")
        tcl_code = self.template_engine.render_template("thresholding/hls/ipgen.tcl.j2", tcl_values)
        
        tcl_path = os.path.join(path, f"hls_syn_{node.name}.tcl")
        with open(tcl_path, "w") as f:
            f.write(tcl_code)
