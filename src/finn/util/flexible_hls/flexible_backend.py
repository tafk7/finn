############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN FlexibleHLSBackend - Drop-in replacement for FINN's HLSBackend
############################################################################

import numpy as np
import os
import subprocess
import warnings
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set, Union
from qonnx.core.datatype import DataType

from finn.util.basic import CppBuilder, make_build_dir
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.hls import CallHLS

from .config import FINNHLSConfig, create_finn_config
from .path_resolver import FINNPathResolver
from .template_engine import FINNTemplateEngine, FINNTemplateMigrator, FINNTemplateType
from .compiler_builder import FINNCompilerBuilder, FINNBuildConfiguration

try:
    import pyxsi_utils
except ModuleNotFoundError:
    pyxsi_utils = None


logger = logging.getLogger(__name__)


class FlexibleHLSBackend(ABC):
    """Flexible HLS Backend that replaces FINN's HLSBackend with enhanced capabilities.
    
    This class provides 100% API compatibility with FINN's original HLSBackend while
    adding flexible path resolution, template inheritance, and enhanced compilation.
    All existing FINN kernels work without modification.
    """

    def __init__(self, config: Optional[FINNHLSConfig] = None):
        """Initialize FlexibleHLSBackend with optional configuration."""
        self.config = config or create_finn_config()
        self.path_resolver = FINNPathResolver()
        self.template_engine = FINNTemplateEngine(
            search_paths=self._get_template_search_paths()
        )
        self.compiler_builder = FINNCompilerBuilder(self.path_resolver)
        
        # FINN compatibility: code_gen_dict for template variables
        self.code_gen_dict: Dict[str, List[str]] = {}
        
        # Migration helper for static templates
        self.template_migrator = FINNTemplateMigrator(self.template_engine)
        
        # Initialize FINN template compatibility
        self._setup_finn_template_compatibility()
        
        logger.debug("FlexibleHLSBackend initialized with flexible path resolution")

    def _get_template_search_paths(self) -> List[str]:
        """Get template search paths from configuration and environment."""
        search_paths = []
        
        # Add configured template paths
        expanded_config = self.config.get_expanded_paths()
        search_paths.extend(expanded_config.templates.template_paths)
        
        # Add default FINN template paths
        if self.path_resolver.env_config.finn_root:
            finn_templates = os.path.join(
                self.path_resolver.env_config.finn_root,
                "src", "finn", "custom_op", "fpgadataflow", "templates"
            )
            if os.path.exists(finn_templates):
                search_paths.append(finn_templates)
        
        # Add flexible_hls template paths
        current_dir = os.path.dirname(__file__)
        flexible_templates = os.path.join(current_dir, "templates")
        if os.path.exists(flexible_templates):
            search_paths.append(flexible_templates)
        
        return search_paths

    def _setup_finn_template_compatibility(self):
        """Set up compatibility with FINN's static templates."""
        # Import FINN's static templates for migration
        try:
            from finn.custom_op.fpgadataflow import templates as finn_templates
            
            static_templates = {
                "ipgen_template": getattr(finn_templates, "ipgen_template", ""),
                "ipgentcl_template": getattr(finn_templates, "ipgentcl_template", ""),
                "docompute_template": getattr(finn_templates, "docompute_template", ""),
                "docompute_template_timeout": getattr(finn_templates, "docompute_template_timeout", "")
            }
            
            # Migrate static templates to template engine
            self.template_migrator.migrate_static_templates(static_templates)
            logger.debug("Migrated FINN static templates to template engine")
            
        except ImportError as e:
            logger.warning(f"Could not import FINN templates for migration: {e}")

    # FINN API Compatibility Methods - Identical signatures to original HLSBackend

    def get_nodeattr_types(self):
        """Get node attribute types. Maintains FINN API compatibility."""
        return {
            "code_gen_dir_cppsim": ("s", False, ""),
            "executable_path": ("s", False, ""),
            "res_hls": ("s", False, ""),
            "cpp_interface": ("s", False, "packed", {"packed", "hls_vector"}),
        }

    def get_all_verilog_paths(self):
        """Return list of all folders containing Verilog code for this node."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        assert (
            code_gen_dir != ""
        ), """Node attribute "code_gen_dir_ipgen" is
        not set. Please run HLSSynthIP first."""
        verilog_path = "{}/project_{}/sol1/impl/verilog/".format(code_gen_dir, self.onnx_node.name)
        subcore_verilog_path = "{}/project_{}/sol1/impl/ip/hdl/ip/".format(
            code_gen_dir, self.onnx_node.name
        )
        ret = [verilog_path]
        if os.path.isdir(subcore_verilog_path):
            ret += [subcore_verilog_path]
        return ret

    def get_all_verilog_filenames(self, abspath=False):
        """Return list of all Verilog files used for this node."""
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
        """Creates a xsi emulation library for the RTL code generated for this node."""
        verilog_files = self.get_all_verilog_filenames(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        ret = pyxsi_utils.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir, debug
        )
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates c++ code and tcl script for ip generation using flexible templates."""
        node = self.onnx_node

        # Enhanced: Use flexible template engine instead of static templates
        path = self.get_nodeattr("code_gen_dir_ipgen")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("ipgen")
        self.blackboxfunction()
        self.pragmas()
        self.docompute()

        # Enhanced: Use template engine for ipgen cpp template
        template_vars = dict((k, '\n'.join(v) if isinstance(v, list) else str(v)) 
                           for k, v in self.code_gen_dict.items())
        
        cpp_content = self.template_engine.render_template(
            "ipgen_template", template_vars, FINNTemplateType.IPGEN_CPP
        )
        
        if cpp_content is None:
            logger.error("Failed to render ipgen_cpp template, falling back to default")
            # Fallback to basic template if rendering fails
            cpp_content = self._get_fallback_ipgen_template()
            for key, value_list in self.code_gen_dict.items():
                cpp_content = cpp_content.replace(key, '\n'.join(value_list))

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(os.path.join(code_gen_dir, "top_{}.cpp".format(node.name)), "w") as f:
            f.write(cpp_content)
        self.code_gen_dict.clear()

        # Enhanced: Use template engine for ipgen tcl template
        self.code_gen_dict["$PROJECTNAME$"] = ["project_{}".format(node.name)]
        self.code_gen_dict["$HWSRCDIR$"] = [code_gen_dir]
        self.code_gen_dict["$FPGAPART$"] = [fpgapart]
        self.code_gen_dict["$TOPFXN$"] = [node.name]
        self.code_gen_dict["$CLKPERIOD$"] = [str(clk)]
        self.code_gen_dict["$DEFAULT_DIRECTIVES$"] = self.ipgen_default_directives()
        self.code_gen_dict["$EXTRA_DIRECTIVES$"] = self.ipgen_extra_directives()

        template_vars = dict((k, '\n'.join(v) if isinstance(v, list) else str(v)) 
                           for k, v in self.code_gen_dict.items())
        
        tcl_content = self.template_engine.render_template(
            "ipgentcl_template", template_vars, FINNTemplateType.IPGEN_TCL
        )
        
        if tcl_content is None:
            logger.error("Failed to render ipgen_tcl template, falling back to default")
            tcl_content = self._get_fallback_ipgen_tcl_template()
            for key, value_list in self.code_gen_dict.items():
                tcl_content = tcl_content.replace(key, '\n'.join(value_list))

        with open(os.path.join(code_gen_dir, "hls_syn_{}.tcl".format(node.name)), "w") as f:
            f.write(tcl_content)
        self.code_gen_dict.clear()

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
        """Return a list of extra tcl directives for HLS synthesis."""
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
        """Generates c++ code for simulation (cppsim) using flexible templates."""
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("cppsim")
        self.read_npy_data()
        self.strm_decl()
        self.pragmas()
        self.docompute()
        self.dataoutstrm()
        self.save_as_npy()

        # Enhanced: Use template engine for docompute template
        template_vars = dict((k, '\n'.join(v) if isinstance(v, list) else str(v)) 
                           for k, v in self.code_gen_dict.items())

        if self.get_nodeattr("cpp_interface") == "hls_vector":
            self.timeout_value()
            self.timeout_condition()
            self.timeout_read_stream()
            template_name = "docompute_template_timeout"
            template_type = FINNTemplateType.DOCOMPUTE_TIMEOUT
        else:
            template_name = "docompute_template"
            template_type = FINNTemplateType.DOCOMPUTE

        cpp_content = self.template_engine.render_template(
            template_name, template_vars, template_type
        )
        
        if cpp_content is None:
            logger.error(f"Failed to render {template_name}, falling back to default")
            cpp_content = self._get_fallback_docompute_template()
            for key, value_list in self.code_gen_dict.items():
                cpp_content = cpp_content.replace(key, '\n'.join(value_list))

        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        with open(os.path.join(code_gen_dir, "execute_{}.cpp".format(node.op_type)), "w") as f:
            f.write(cpp_content)
        self.code_gen_dict.clear()

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        vlnv = self.get_nodeattr("ip_vlnv")
        cmd = ["create_bd_cell -type ip -vlnv %s %s" % (vlnv, self.onnx_node.name)]
        return cmd

    def compile_singlenode_code(self):
        """Enhanced compilation using flexible path resolution and build strategies."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        
        # Enhanced: Use flexible compiler builder instead of hardcoded CppBuilder
        try:
            build_config = FINNBuildConfiguration(
                custom_includes=self.path_resolver.get_finn_include_paths(),
                custom_sources=self.path_resolver.get_finn_source_paths()
            )
            
            executable_path = self.compiler_builder.build(
                source_directory=code_gen_dir,
                output_directory=code_gen_dir,
                build_config=build_config
            )
            
            self.set_nodeattr("executable_path", executable_path)
            logger.debug(f"Enhanced compilation completed: {executable_path}")
            
        except Exception as e:
            logger.warning(f"Enhanced compilation failed, falling back to legacy: {e}")
            self._compile_singlenode_code_legacy()

    def _compile_singlenode_code_legacy(self):
        """Fallback to legacy compilation method for compatibility."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()
        
        # Use flexible path resolution for include paths
        include_paths = self.path_resolver.get_finn_include_paths()
        for include_path in include_paths:
            builder.append_includes(f"-I{include_path}")
        
        # Add tool-specific includes if available
        if self.path_resolver.env_config.hls_path:
            builder.append_includes(f"-I{self.path_resolver.env_config.hls_path}/include")
        if self.path_resolver.env_config.vitis_path:
            builder.append_includes(f"-I{self.path_resolver.env_config.vitis_path}/include")
        
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        
        # Add FINN source dependencies
        source_paths = self.path_resolver.get_finn_source_paths()
        for source_path in source_paths:
            builder.append_sources(source_path)
        
        builder.append_includes("-lz")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    # All other methods remain identical to FINN's HLSBackend for 100% compatibility
    
    def npy_to_dynamic_output(self, context):
        """Reads the output from an output.npy file generated from cppsim."""
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
        """Execute node with enhanced error handling and flexible path resolution."""
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
            if inp_val.dtype is not np.float32:
                warnings.warn(
                    f"{node.name}: Changing input container datatype from "
                    f"{inp_val.dtype} to {np.float32}"
                )
                inp_val = inp_val.astype(np.float32)
            assert inp_val.shape == exp_ishape, "Input shape doesn't match expected shape."
            export_idt = self.get_input_datatype(i)

            if export_idt == DataType["BIPOLAR"]:
                inp_val = (inp_val + 1) / 2
                export_idt = DataType["BINARY"]

            reshaped_input = inp_val.reshape(folded_ishape)
            reshaped_input = reshaped_input.copy()
            np.save(os.path.join(code_gen_dir, "input_%s.npy" % i), reshaped_input)
            nbits = self.get_instream_width(i)
            if nbits == 0:
                continue
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_{}.npy".format(code_gen_dir, i), export_idt, nbits
            )
            inputs["in%s" % i] = rtlsim_inp

        if mode == "cppsim":
            self.exec_precompiled_singlenode_model()
            self.npy_to_dynamic_output(context)
            for o, outp in enumerate(node.output):
                exp_oshape = tuple(self.get_normal_output_shape(o))
                assert (
                    context[outp].shape == exp_oshape
                ), "cppsim did not produce expected output shape"
                if self.get_output_datatype(o) == DataType["BIPOLAR"]:
                    out = context[outp]
                    out = 2 * out - 1
                    context[outp] = out
        elif mode == "rtlsim":
            outputs = {}
            for o, outp in enumerate(node.output):
                outputs["out%s" % o] = []
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
                exp_oshape = tuple(self.get_normal_output_shape(o))
                output = np.load(out_npy_path)
                output = np.asarray([output], dtype=np.float32).reshape(*exp_oshape)
                context[outp] = output

                assert (
                    context[outp].shape == exp_oshape
                ), "Output shape doesn't match expected shape."

    # Abstract methods that must be implemented by specific kernels (unchanged from FINN)
    
    @abstractmethod
    def global_includes(self):
        """Function to set the global includes for c++ code generation."""
        pass

    @abstractmethod
    def defines(self, var):
        """Function to set the define commands for c++ code generation."""
        pass

    @abstractmethod
    def docompute(self):
        """Function to generate the commands for the computational part of the c++ code."""
        pass

    @abstractmethod
    def blackboxfunction(self):
        """Function to generate a blackbox function in c++ from which an IP block will be generated."""
        pass

    # Additional utility methods (unchanged from FINN)
    
    def read_npy_data(self):
        """Function to generate the commands for reading data from .npy file in c++."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$READNPYDATA$"] = []
        cpp_interface = self.get_nodeattr("cpp_interface")

        for i, inp in enumerate(self.onnx_node.input):
            dtype = self.get_input_datatype(i)
            if dtype == DataType["BIPOLAR"]:
                dtype = DataType["BINARY"]
            elem_hls_type = dtype.get_hls_datatype_str()
            npy_type = "float"
            npy_in = "%s/input_%s.npy" % (code_gen_dir, i)

            iwidth = self.get_instream_width(i)
            if iwidth == 0:
                continue
            if cpp_interface == "packed":
                elem_bits = dtype.bitwidth()
                packed_bits = iwidth
                packed_hls_type = "ap_uint<%d>" % packed_bits
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2apintstream<%s, %s, %d, %s>("%s", in%s_V);'
                    % (
                        packed_hls_type,
                        elem_hls_type,
                        elem_bits,
                        npy_type,
                        npy_in,
                        i,
                    )
                )
            else:
                folded_shape = self.get_folded_input_shape()
                self.code_gen_dict["$READNPYDATA$"].append(
                    'npy2vectorstream<%s, %s, %d>("%s", in%s_V, false);'
                    % (
                        elem_hls_type,
                        npy_type,
                        folded_shape[-1],
                        npy_in,
                        i,
                    )
                )

    def strm_decl(self):
        """Function to generate the commands for the stream declaration in c++."""
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width())
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width())
        )

    def dataoutstrm(self):
        """Function to generate the commands for reading out data from c++."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$DATAOUTSTREAM$"] = []

        for o, outp in enumerate(self.onnx_node.output):
            dtype = self.get_output_datatype(o)
            if dtype == DataType["BIPOLAR"]:
                dtype = DataType["BINARY"]
            elem_hls_type = dtype.get_hls_datatype_str()
            npy_type = "float"
            npy_out = "%s/output_%s.npy" % (code_gen_dir, o)
            oshape = self.get_folded_output_shape(o)
            oshape_cpp_str = str(oshape).replace("(", "{").replace(")", "}")

            cpp_interface = self.get_nodeattr("cpp_interface")

            if cpp_interface == "packed":
                elem_bits = dtype.bitwidth()
                packed_bits = self.get_outstream_width(o)
                packed_hls_type = "ap_uint<%d>" % packed_bits

                self.code_gen_dict["$DATAOUTSTREAM$"].append(
                    'apintstream2npy<%s, %s, %d, %s>(out%s_V, %s, "%s");'
                    % (
                        packed_hls_type,
                        elem_hls_type,
                        elem_bits,
                        npy_type,
                        o,
                        oshape_cpp_str,
                        npy_out,
                    )
                )
            else:
                folded_shape = self.get_folded_output_shape(o)
                self.code_gen_dict["$DATAOUTSTREAM$"].append(
                    'vectorstream2npy<%s, %s, %d>(strm, %s, "%s");'
                    % (
                        elem_hls_type,
                        npy_type,
                        folded_shape[-1],
                        oshape_cpp_str,
                        npy_out,
                    )
                )

    def save_as_npy(self):
        """Function to generate the commands for saving data in .npy file in c++"""
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def pragmas(self):
        """Function to generate the pragma commands in c++."""
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

    def get_ap_int_max_w(self):
        """Return the maximum width of any ap_int used in this module."""
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        ret = max([instream, outstream])
        assert ret <= 8191, "AP_INT_MAX_W=%d is larger than allowed maximum of 8191" % ret
        return ret

    def timeout_value(self):
        """Set timeout value for HLS functions defined for one clock cycle"""
        self.code_gen_dict["$TIMEOUT_VALUE$"] = ["1000"]

    def timeout_condition(self):
        """Set timeout condition for HLS functions defined for one clock cycle"""
        self.code_gen_dict["$TIMEOUT_CONDITION$"] = ["out0_V.empty()"]

    def timeout_read_stream(self):
        """Set reading output stream procedure for HLS functions defined for one clock cycle"""
        self.code_gen_dict["$TIMEOUT_READ_STREAM$"] = ["strm << out0_V.read();"]

    # Fallback templates for compatibility
    
    def _get_fallback_ipgen_template(self):
        """Fallback ipgen template if template engine fails."""
        return """
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$

#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

$BLACKBOXFUNCTION$
{
$PRAGMAS$
$DOCOMPUTE$
}
"""

    def _get_fallback_ipgen_tcl_template(self):
        """Fallback ipgen tcl template if template engine fails."""
        return """
set config_proj_name $PROJECTNAME$
puts "HLS project: $config_proj_name"
set config_hwsrcdir "$HWSRCDIR$"
puts "HW source dir: $config_hwsrcdir"
set config_proj_part "$FPGAPART$"
set config_bnnlibdir "$::env(FINN_DEPS_DIR)/finn-hlslib"
puts "finn-hlslib dir: $config_bnnlibdir"
set config_customhlsdir "$::env(FINN_ROOT)/custom_hls"
puts "custom HLS dir: $config_customhlsdir"
set config_toplevelfxn "$TOPFXN$"
set config_clkperiod $CLKPERIOD$

open_project $config_proj_name
add_files $config_hwsrcdir/top_$TOPFXN$.cpp -cflags "-std=c++14 -I$config_bnnlibdir -I$config_customhlsdir"

set_top $config_toplevelfxn
open_solution sol1
set_part $config_proj_part

$DEFAULT_DIRECTIVES$
$EXTRA_DIRECTIVES$

create_clock -period $config_clkperiod -name default
csynth_design
export_design -format ip_catalog
exit 0
"""

    def _get_fallback_docompute_template(self):
        """Fallback docompute template if template engine fails."""
        return """
#define HLS_CONSTEXPR_ENABLE
#define AP_INT_MAX_W $AP_INT_MAX_W$
#define HLS_NO_XIL_FPO_LIB
#include "cnpy.h"
#include "npy2apintstream.hpp"
#include "npy2vectorstream.hpp"
#include <vector>
#include "bnn-library.h"

// includes for network parameters
$GLOBALS$

// defines for network parameters
$DEFINES$

int main(){
$PRAGMAS$

$STREAMDECLARATIONS$

$READNPYDATA$

$DOCOMPUTE$

$DATAOUTSTREAM$

$SAVEASCNPY$

}
"""