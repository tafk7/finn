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
import subprocess
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Set, Optional
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow import templates
from finn.util.basic import CppBuilder, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.hls import CallHLS
from finn.codegen.codegen import Codegen
from finn.codegen import TemplateEngine

try:
    import pyxsi_utils
except ModuleNotFoundError:
    pyxsi_utils = None


class HLSBackend(Codegen):
    """Clean HLS Backend class using direct template value generation.
    
    Provides HLS-specific code generation functionality for FINN custom ops
    that correspond to finn-hlslib functions. Uses explicit template declaration
    and direct value generation without legacy compatibility layers.
    """

    # ===== Explicit Template Declaration =====
    # These should be overridden by concrete operation classes
    TEMPLATE_NAME: Optional[str] = None
    TEMPLATE_OPTIONS: Optional[Dict[str, str]] = None

    def __init__(self, **kwargs):
        """Initialize HLS backend with clean Codegen infrastructure."""
        # Extract HLS-specific kwargs to avoid conflicts
        hls_kwargs = {k: v for k, v in kwargs.items() if k.startswith('hls_')}
        
        # Initialize parent Codegen class
        super().__init__()
        
        # Initialize template engine
        self.template_engine = TemplateEngine()
        
        # HLS-specific initialization
        self.hls_template_path = "hls/"
        
        # Context for template values
        self._current_fpgapart = None
        self._current_clk = None
        
        # Apply HLS-specific configurations
        for key, value in hls_kwargs.items():
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
    
    def get_nodeattr_types(self) -> Dict[str, Any]:
        """Get HLS backend node attribute types.
        
        Merges any operation-specific attributes with HLS backend attributes.
        HLS attributes take precedence in case of conflicts.
        
        Returns:
            Dictionary of node attribute specifications
        """
        # Base HLS backend attributes
        hls_attrs = {
            "code_gen_dir_cppsim": ("s", False, ""),
            "executable_path": ("s", False, ""),
            "res_hls": ("s", False, ""),
            # Temporary node attribute to keep track of interface style of HLS ops
            "cpp_interface": ("s", False, "packed", {"packed", "hls_vector"}),
            # Template override support
            "hls_template_override": ("s", False, ""),
        }
        
        # Try to get operation-specific attributes if this is a multiple inheritance case
        operation_attrs = {}
        for base in self.__class__.__bases__:
            if hasattr(base, 'get_nodeattr_types') and base != HLSBackend:
                try:
                    operation_attrs = base.get_nodeattr_types(self)
                    break
                except Exception:
                    pass
        
        # HLS attributes override operation attributes (explicit policy)
        merged_attrs = {**operation_attrs, **hls_attrs}
        
        # Log any conflicts for debugging
        conflicts = set(operation_attrs.keys()) & set(hls_attrs.keys())
        if conflicts:
            self.logger.debug(f"HLS attributes override operation attributes: {conflicts}")
        
        return merged_attrs

    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Generate values for HLS template using direct value generation.
        
        Args:
            template_name: Name of template to extract values for
            
        Returns:
            Dictionary mapping template placeholders to values
        """
        # Generate template values directly without legacy conversion
        template_values = {
            'AP_INT_MAX_W': self.get_ap_int_max_w(),
        }
        
        # Template-specific values based on template type
        if 'ipgen' in template_name:
            if template_name.endswith('.tcl.j2'):
                template_values.update({
                    'PROJECTNAME': f"project_{self.onnx_node.name}",
                    'HWSRCDIR': self.get_nodeattr("code_gen_dir_ipgen"),
                    'FPGAPART': self._current_fpgapart or "xc7z020clg400-1",
                    'TOPFXN': self.onnx_node.name,
                    'CLKPERIOD': self._current_clk or 10,
                    'DEFAULT_DIRECTIVES': '\n'.join(self.ipgen_default_directives()),
                    'EXTRA_DIRECTIVES': '\n'.join(self.ipgen_extra_directives()),
                })
        
        elif 'ip_package' in template_name:
            template_values.update({
                'TOPNAME': self.onnx_node.name,
                'VERILOG_DIR': self.get_nodeattr("code_gen_dir_ipgen"),
                'HLS_SNAME': 'V',  # Default stream name
            })
        
        return template_values


    # ===== HLS-Specific Methods =====

    def generate_hls_code(self) -> str:
        """Generate HLS code - uses inherited generate_code().
        
        Returns:
            Generated HLS code as string
        """
        return self.generate_code()

    def _extract_hls_parallelization_values(self, operation) -> Dict[str, Any]:
        """Extract HLS-specific parallelization values.
        
        Args:
            operation: Operation instance to extract from
            
        Returns:
            Dictionary of HLS parallelization values
        """
        values = {
            'pe_factor': self._safe_extract_value(operation, 'PE', 1),
            'simd_factor': self._safe_extract_value(operation, 'SIMD', 1),
        }
        values['parallelization_strategy'] = self._infer_hls_parallelization_strategy(operation)
        return values

    def _infer_hls_parallelization_strategy(self, operation) -> str:
        """Infer HLS parallelization strategy.
        
        Args:
            operation: Operation instance to analyze
            
        Returns:
            Parallelization strategy string
        """
        has_pe = self._has_attr(operation, 'PE')
        has_simd = self._has_attr(operation, 'SIMD')
        
        if has_pe and has_simd:
            return 'pe_simd'
        elif has_pe:
            return 'pe_only'
        elif has_simd:
            return 'simd_only'
        else:
            return 'sequential'

    def _extract_hls_memory_config(self, operation) -> Dict[str, Any]:
        """Extract HLS-specific memory configuration.
        
        Args:
            operation: Operation instance to extract from
            
        Returns:
            Dictionary of HLS memory configuration
        """
        return {
            'mem_mode': self._safe_extract_value(operation, 'mem_mode', 'const_embedded'),
            'ram_style': self._safe_extract_value(operation, 'ram_style', 'auto'),
        }

    # ===== Legacy HLS Backend Functionality =====
    # All existing methods preserved for backward compatibility


    def get_all_verilog_paths(self):
        "Return list of all folders containing Verilog code for this node."

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        assert (
            code_gen_dir != ""
        ), """Node attribute "code_gen_dir_ipgen" is
        not set. Please run HLSSynthIP first."""
        verilog_path = "{}/project_{}/sol1/impl/verilog/".format(code_gen_dir, self.onnx_node.name)
        subcore_verilog_path = "{}/project_{}/sol1/impl/ip/hdl/ip/".format(
            code_gen_dir, self.onnx_node.name
        )
        # default impl only returns the HLS verilog codegen dir and subcore (impl/ip/hdl/ip) dir
        # if it exists
        ret = [verilog_path]
        if os.path.isdir(subcore_verilog_path):
            ret += [subcore_verilog_path]
        return ret

    def get_all_verilog_filenames(self, abspath=False):
        "Return list of all Verilog files used for this node."

        verilog_files = []
        verilog_paths = self.get_all_verilog_paths()
        for verilog_path in verilog_paths:
            for f in os.listdir(verilog_path):
                if f.endswith(".v"):
                    if abspath:
                        verilog_files += [verilog_path + "/" + f]
                    else:
                        verilog_files += [f]
        return verilog_files

    def prepare_rtlsim(self):
        """Creates a xsi emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path."""

        verilog_files = self.get_all_verilog_filenames(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        ret = pyxsi_utils.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir, debug
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate HLS code using clean template system."""
        # Store context for template values
        self._current_fpgapart = fpgapart
        self._current_clk = clk
        
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_ipgen")
        
        # Generate parameters first
        self.generate_params(model, path)
        
        # Generate CPP file using template
        self.set_template_override("hls/ipgen.cpp.j2")
        cpp_code = self.generate_code()
        
        # Write CPP file
        cpp_path = os.path.join(path, f"top_{node.name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        
        # Generate TCL script using template
        self.set_template_override("hls/ipgen.tcl.j2")
        tcl_code = self.generate_code()
        
        # Write TCL file
        tcl_path = os.path.join(path, f"hls_syn_{node.name}.tcl")
        with open(tcl_path, "w") as f:
            f.write(tcl_code)

    def ipgen_default_directives(self):
        """Return list of default HLS synthesis directives"""

        default_directives = [
            "set_param hls.enable_hidden_option_error false",
            "config_compile -disable_unroll_code_size_check -pipeline_style flp",
            "config_interface -m_axi_addr64",
            "config_rtl -module_auto_prefix",
            "config_rtl -deadlock_detection none",
        ]
        return default_directives

    def ipgen_extra_directives(self):
        "Return a list of extra tcl directives for HLS synthesis."
        return []

    def ipgen_singlenode_code(self):
        """Builds the bash script for IP generation using the CallHLS utility."""
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        builder = CallHLS()
        builder.append_tcl(code_gen_dir + "/hls_syn_{}.tcl".format(node.name))
        builder.set_ipgen_path(code_gen_dir + "/project_{}".format(node.name))
        builder.build(code_gen_dir)
        ipgen_path = builder.ipgen_path
        assert os.path.isdir(ipgen_path), "IPGen failed: %s not found" % (ipgen_path)
        self.set_nodeattr("ipgen_path", ipgen_path)
        ip_path = ipgen_path + "/sol1/impl/ip"
        assert os.path.isdir(ip_path), "IPGen failed: %s not found. Check log under %s" % (
            ip_path,
            code_gen_dir,
        )
        self.set_nodeattr("ip_path", ip_path)
        vlnv = "xilinx.com:hls:%s:1.0" % node.name
        self.set_nodeattr("ip_vlnv", vlnv)

    def code_generation_cppsim(self, model):
        """Generate C++ simulation code using clean template system."""
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        
        # Generate parameters first
        self.generate_params(model, path)
        
        # Determine template based on interface
        if self.get_nodeattr("cpp_interface") == "hls_vector":
            template_name = "hls/docompute_timeout.cpp.j2"
        else:
            template_name = "hls/docompute.cpp.j2"
        
        # Generate using template
        self.set_template_override(template_name)
        cpp_code = self.generate_code()
        
        # Write file
        cpp_path = os.path.join(path, f"execute_{node.op_type}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        vlnv = self.get_nodeattr("ip_vlnv")
        cmd = ["create_bd_cell -type ip -vlnv %s %s" % (vlnv, self.onnx_node.name)]
        return cmd

    def compile_singlenode_code(self):
        """Builds the bash script for compilation using the CppBuilder from
        finn.util.basic and executes the script to produce the executable."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes("-I$FINN_ROOT/src/finn/qnn-data/cpp")
        builder.append_includes("-I$FINN_DEPS_DIR/cnpy/")
        builder.append_includes("-I$FINN_DEPS_DIR/finn-hlslib")
        builder.append_includes("-I$FINN_ROOT/custom_hls")
        builder.append_includes("-I{}/include".format(os.environ["HLS_PATH"]))
        builder.append_includes("-I{}/include".format(os.environ["VITIS_PATH"]))
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$FINN_DEPS_DIR/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    def npy_to_dynamic_output(self, context):
        """Reads the output from an output.npy file generated from cppsim and
        places its content into the context dictionary."""
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        for o, outp in enumerate(node.output):
            output = np.load("{}/output_{}.npy".format(code_gen_dir, o))
            exp_shape = self.get_normal_output_shape(o)
            context[outp] = output.reshape(exp_shape)

    def exec_precompiled_singlenode_model(self):
        """Executes precompiled executable."""
        executable_path = self.get_nodeattr("executable_path")
        if executable_path == "":
            raise Exception(
                """
Found no executable for this node, did you run the codegen and
compilation transformations?
            """
            )
        process_execute = subprocess.Popen(executable_path, stdout=subprocess.PIPE)
        process_execute.communicate()

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )
        inputs = {}
        for i, inp in enumerate(node.input):
            exp_ishape = tuple(self.get_normal_input_shape(i))
            folded_ishape = self.get_folded_input_shape(i)
            inp_val = context[inp]
            # Make sure the input has the right container datatype
            if inp_val.dtype is not np.float32:
                # Issue a warning to make the user aware of this type-cast
                warnings.warn(
                    f"{node.name}: Changing input container datatype from "
                    f"{inp_val.dtype} to {np.float32}"
                )
                # Convert the input to floating point representation as the
                # container datatype
                inp_val = inp_val.astype(np.float32)
            assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
            export_idt = self.get_input_datatype(i)

            if export_idt == DataType["BIPOLAR"]:
                # store bipolar activations as binary
                inp_val = (inp_val + 1) / 2
                export_idt = DataType["BINARY"]

            reshaped_input = inp_val.reshape(folded_ishape)
            reshaped_input = reshaped_input.copy()
            np.save(os.path.join(code_gen_dir, "input_%s.npy" % i), reshaped_input)
            nbits = self.get_instream_width(i)
            # if the stream is not exposed, it has 0 width and no npy file will be created
            if nbits == 0:
                continue
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_{}.npy".format(code_gen_dir, i), export_idt, nbits
            )
            inputs["in%s" % i] = rtlsim_inp

        if mode == "cppsim":
            # execute the precompiled model
            self.exec_precompiled_singlenode_model()
            # load output npy file
            self.npy_to_dynamic_output(context)
            for o, outp in enumerate(node.output):
                exp_oshape = tuple(self.get_normal_output_shape(o))
                assert (
                    context[outp].shape == exp_oshape
                ), "cppsim did not produce expected output shape"
                # binary -> bipolar if needed
                if self.get_output_datatype(o) == DataType["BIPOLAR"]:
                    out = context[outp]
                    out = 2 * out - 1
                    context[outp] = out
        elif mode == "rtlsim":
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
                out_npy_path = "{}/output_{}.npy".format(code_gen_dir, o)
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


    def get_ap_int_max_w(self):
        """Return the maximum width of any ap_int used in this module. Used to set the
        AP_INT_MAX_W definition for HLS."""
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        ret = max([instream, outstream])
        assert ret <= 8191, "AP_INT_MAX_W=%d is larger than allowed maximum of 8191" % ret
        return ret
