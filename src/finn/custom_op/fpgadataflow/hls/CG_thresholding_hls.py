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


class CG_Thresholding_hls(Thresholding, CG_HLSBackend):
    """
    Clean thresholding HLS implementation with current inheritance.
    
    Inherits from both Thresholding and CG_HLSBackend to maintain compatibility
    while using clean Jinja2-based template generation without legacy bloat.
    """
    
    # Use simplified HLS template
    TEMPLATE_NAME = "hls_basic.cpp.j2"
    
    def __init__(self, onnx_node, **kwargs):
        """Initialize clean thresholding HLS backend with current inheritance."""
        # Maintain current inheritance structure
        Thresholding.__init__(self, onnx_node, **kwargs)
        CG_HLSBackend.__init__(self, **kwargs)
        
        self.logger.debug(f"Initialized CG_Thresholding_hls for node: {onnx_node.name}")

    def get_nodeattr_types(self):
        """Get node attribute types from both parent classes."""
        my_attrs = {}
        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(CG_HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def get_template_name(self) -> str:
        """Return the template name for this backend."""
        return self.TEMPLATE_NAME
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Generate template values for simplified HLS template.
        
        Args:
            template_name: Name of template being generated for
            
        Returns:
            Dictionary of template values for hls_basic.cpp.j2
        """
        # Get basic parameters
        pe = self.get_nodeattr('PE')
        num_channels = self.get_nodeattr('NumChannels')
        num_steps = self.get_nodeattr('numSteps')
        
        # Data types
        input_dtype = self.get_input_datatype()
        output_dtype = self.get_output_datatype()
        threshold_dtype = self.get_input_datatype(1)
        
        # Generate includes
        global_includes = [
            '#include "activations.hpp"',
            '#include "params.h"'
        ]
        
        # Generate defines
        num_input_vectors = self.get_nodeattr('numInputVectors')
        num_reps = num_input_vectors[0] if num_input_vectors else 1
        tmem = self.calc_tmem()
        
        defines = [
            f"#define NumChannels {num_channels}",
            f"#define PE {pe}",
            f"#define numReps {num_reps}",
            f"#define numSteps {num_steps}",
        ]
        if tmem > 0:
            defines.append(f"#define TMEM {tmem}")
        
        # Generate input/output ports
        input_ports = [{
            'name': 'in0_V',
            'type': f'ap_uint<{self.get_instream_width()}>'
        }]
        
        output_ports = [{
            'name': 'out0_V',
            'type': f'ap_uint<{self.get_outstream_width()}>'
        }]
        
        # Generate pragmas
        pragmas = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        ]
        
        # Add array partition pragmas if using memory
        if tmem != 0:
            pragmas.extend([
                "#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=1",
                "#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=3"
            ])
            
            ram_style = self._safe_get_nodeattr("ram_style", "auto")
            if ram_style == "distributed":
                pragmas.append("#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_LUTRAM")
            elif ram_style == "block":
                pragmas.append("#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_BRAM")
        
        # Generate compute body
        inp_hls_str = input_dtype.get_hls_datatype_str()
        out_hls_str = output_dtype.get_hls_datatype_str()
        thold_hls_str = threshold_dtype.get_hls_datatype_str()
        
        compute_body = f"""// Thresholding computation
Thresholding_Batch<{inp_hls_str}, {out_hls_str}, {thold_hls_str}, NumChannels, PE, {tmem}>(
    in0_V, out0_V, threshs.parameters, numReps);"""
        
        # Return values for simplified template
        return {
            'function_name': f'{self.onnx_node.name}_compute',
            'ap_int_max_w': 8192,
            'global_includes': global_includes,
            'defines': defines,
            'input_ports': input_ports,
            'output_ports': output_ports,
            'pragmas': pragmas,
            'compute_body': compute_body,
            'include_wrapper': False  # We'll handle wrapper separately if needed
        }

    def _safe_get_nodeattr(self, attr_name: str, default_value=None):
        """Safely get node attribute with default value.
        
        Args:
            attr_name: Name of attribute to get
            default_value: Default value if attribute not found
            
        Returns:
            Attribute value or default
        """
        try:
            return self.get_nodeattr(attr_name)
        except:
            return default_value
    
    # Abstract methods required by CG_HLSBackend
    def _generate_common_values(self, instance) -> Dict[str, Any]:
        """Generate common template values (required by parent class).
        
        Args:
            instance: Backend instance (for compatibility)
            
        Returns:
            Empty dict - we handle everything in get_template_values
        """
        return {}
    
    def _generate_operation_specific_values(self, template_name: str) -> Dict[str, Any]:
        """Generate operation-specific values (required by parent class).
        
        Args:
            template_name: Name of template being generated for
            
        Returns:
            Empty dict - we handle everything in get_template_values
        """
        return {}

    # ===== Code Generation Entry Points =====

    def code_generation_cppsim(self, model):
        """Generate C++ simulation code using simplified template.
        
        Args:
            model: FINN model containing this operation
        """
        # Use parent class implementation which uses generate_code()
        super().code_generation_cppsim(model)

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate IP generation files using simplified template.
        
        Args:
            model: FINN model containing this operation
            fpgapart: Target FPGA part
            clk: Clock period
        """
        # For IP generation, we need a wrapper function
        # Temporarily switch to use wrapper in template
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_ipgen")
        
        # Generate parameter files first
        self.generate_params(model, path)
        
        # Get template values with wrapper enabled
        template_values = self.get_template_values(self.TEMPLATE_NAME)
        template_values['include_wrapper'] = True
        template_values['top_function_name'] = node.name
        template_values['input_width'] = self.get_instream_width()
        template_values['output_width'] = self.get_outstream_width()
        
        # Generate code with wrapper
        from finn.codegen.template_engine import TemplateEngine
        engine = TemplateEngine()
        cpp_code = engine.render(self.TEMPLATE_NAME, template_values)
        
        # Write file
        cpp_path = os.path.join(path, f"top_{node.name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        
        # For TCL, we can use a simple script
        tcl_content = f"""# HLS synthesis script for {node.name}
open_project project_{node.name}
add_files {cpp_path}
set_top {node.name}
open_solution sol1
set_part {fpgapart}
create_clock -period {clk} -name default
csynth_design
export_design -format ip_catalog
exit 0
"""
        
        tcl_path = os.path.join(path, f"hls_syn_{node.name}.tcl")
        with open(tcl_path, "w") as f:
            f.write(tcl_content)