# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import math
import numpy as np
import os
import shutil
from typing import Dict, Any, List
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.CG_rtlbackend import CG_RTLBackend
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.util.basic import get_memutil_alternatives, mem_primitives_versal
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)


class CG_Thresholding_rtl(Thresholding, CG_RTLBackend):
    """Clean implementation of FINN Thresholding RTL backend.
    
    This is a clean implementation that eliminates legacy compatibility bloat
    and uses only direct template value generation methods. No code_gen_dict usage.
    
    RTL backend for Thresholding using clean Jinja2 template architecture.
    """

    def __init__(self, onnx_node, **kwargs):
        # Initialize parent classes explicitly to maintain compatibility
        Thresholding.__init__(self, onnx_node, **kwargs)
        CG_RTLBackend.__init__(self, **kwargs)
        
        self.logger.debug(f"Initialized CG_Thresholding_rtl for node: {onnx_node.name}")

    def get_nodeattr_types(self):
        my_attrs = {
            # memory depth triggers for threshold storage
            "depth_trigger_uram": ("i", False, 0),
            "depth_trigger_bram": ("i", False, 0),
            # enable uniform thres optimization
            # doesn't actually do anything yet, only
            # for resource estimations
            "uniform_thres": ("i", False, 0, {0, 1}),
            # enable deep pipelining for easier timing closure
            # setting to 0 may save some FFs but otherwise leave on
            "deep_pipeline": ("i", False, 1, {0, 1}),
        }
        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(CG_RTLBackend.get_nodeattr_types(self))
        return my_attrs

    # =============================================================================
    # CLEAN TEMPLATE VALUE GENERATION METHODS (No code_gen_dict usage)
    # =============================================================================

    def get_rtl_wrapper_values(self):
        """Generate RTL wrapper template values for thresholding operation."""
        bias = self.get_nodeattr("ActVal")
        output_data_type = self.get_nodeattr("outputDataType")
        input_data_type = self.get_nodeattr("inputDataType")
        o_bitwidth = DataType[output_data_type].bitwidth()
        i_bitwidth = DataType[input_data_type].bitwidth()
        wdt = self.get_input_datatype(1)
        
        # Calculate O_BITS (existing logic)
        if bias >= 0:
            o_bits = math.ceil(math.log2(2**o_bitwidth + bias))
        else:
            o_bits = 1 + math.ceil(
                math.log2(-bias if -bias >= 2 ** (o_bitwidth - 1) else 2**o_bitwidth + bias)
            )
        
        return {
            'MODULE_NAME_AXI_WRAPPER': self.get_verilog_top_module_name(),
            'N': o_bitwidth,
            'WI': i_bitwidth,
            'WT': wdt.bitwidth(),
            'C': self.get_nodeattr("NumChannels"),
            'PE': self.get_nodeattr("PE"),
            'SIGNED': 1 if self.get_input_datatype(0).signed() else 0,
            'FPARG': 0 if self.get_input_datatype().is_integer() else 1,
            'BIAS': bias,
            'THRESHOLDS_PATH': f'"./{self.onnx_node.name}_"',
            'USE_AXILITE': self.get_nodeattr("runtime_writeable_weights"),
            'DEPTH_TRIGGER_URAM': self.get_nodeattr("depth_trigger_uram"),
            'DEPTH_TRIGGER_BRAM': self.get_nodeattr("depth_trigger_bram"),
            'DEEP_PIPELINE': self.get_nodeattr("deep_pipeline"),
            'O_BITS': int(o_bits),
        }

    def get_rtl_module_name(self):
        """Generate the RTL module name."""
        return self.get_verilog_top_module_name()

    def get_rtl_parameters(self):
        """Generate RTL parameters for thresholding module."""
        bias = self.get_nodeattr("ActVal")
        output_data_type = self.get_nodeattr("outputDataType")
        input_data_type = self.get_nodeattr("inputDataType")
        o_bitwidth = DataType[output_data_type].bitwidth()
        i_bitwidth = DataType[input_data_type].bitwidth()
        wdt = self.get_input_datatype(1)
        
        # Calculate O_BITS
        if bias >= 0:
            o_bits = math.ceil(math.log2(2**o_bitwidth + bias))
        else:
            o_bits = 1 + math.ceil(
                math.log2(-bias if -bias >= 2 ** (o_bitwidth - 1) else 2**o_bitwidth + bias)
            )
        
        return {
            'N': o_bitwidth,
            'WI': i_bitwidth,
            'WT': wdt.bitwidth(),
            'C': self.get_nodeattr("NumChannels"),
            'PE': self.get_nodeattr("PE"),
            'O_BITS': int(o_bits),
        }

    def get_port_declarations(self):
        """Generate port declarations for RTL module."""
        runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
        
        ports = {
            'data_input': {
                'name': 'in0_V',
                'direction': 'input',
                'width': self.get_instream_width(),
                'type': 'axis'
            },
            'data_output': {
                'name': 'out0_V', 
                'direction': 'output',
                'width': self.get_outstream_width(),
                'type': 'axis'
            }
        }
        
        if runtime_writeable:
            ports['axilite'] = {
                'name': 's_axilite',
                'direction': 'slave',
                'type': 'axilite'
            }
        
        return ports

    def get_memory_files(self):
        """Generate list of memory initialization files."""
        dat_files = []
        pe = self.get_nodeattr("PE")
        output_data_type = self.get_nodeattr("outputDataType")
        o_bitwidth = DataType[output_data_type].bitwidth()
        
        for stage in range(o_bitwidth):
            for pe_value in range(pe):
                thresh_file = f"{self.onnx_node.name}_threshs_{pe_value}_{stage}.dat"
                dat_files.append(thresh_file)
        
        return dat_files

    def get_source_files(self):
        """Generate list of RTL source files needed."""
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/thresholding/hdl/")
        
        return [
            os.path.join(rtllib_dir, "axilite_if.v"),
            os.path.join(rtllib_dir, "thresholding.sv"),
            os.path.join(rtllib_dir, "thresholding_axi.sv"),
        ]

    # =============================================================================
    # TEMPLATE VALUE GENERATION METHODS
    # =============================================================================
    
    def get_template_name(self) -> str:
        """Return the template name for this backend."""
        return "thresholding_rtl.v.j2"
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Generate template values for simplified RTL template.
        
        Args:
            template_name: Name of template being generated for
            
        Returns:
            Dictionary of template values for thresholding_rtl.v.j2
        """
        # Get base values
        template_values = self.get_rtl_wrapper_values()
        
        # Convert to simplified template format
        simplified_values = {
            'module_name': template_values['MODULE_NAME_AXI_WRAPPER'],
            'n': template_values['N'],
            'wi': template_values['WI'],
            'wt': template_values['WT'],
            'c': template_values['C'],
            'pe': template_values['PE'],
            'signed': template_values['SIGNED'],
            'fparg': template_values['FPARG'],
            'bias': template_values['BIAS'],
            'thresholds_path': template_values['THRESHOLDS_PATH'],
            'use_axilite': template_values['USE_AXILITE'],
            'depth_trigger_uram': template_values['DEPTH_TRIGGER_URAM'],
            'depth_trigger_bram': template_values['DEPTH_TRIGGER_BRAM'],
            'deep_pipeline': template_values['DEEP_PIPELINE'],
            'o_bits': template_values['O_BITS'],
        }
        
        # Calculate stream widths
        pe = template_values['PE']
        wi = template_values['WI']
        o_bits = template_values['O_BITS']
        
        input_width = pe * wi
        output_width = pe * o_bits
        
        # Pad to byte boundaries
        simplified_values['input_stream_width'] = ((input_width + 7) // 8) * 8
        simplified_values['output_stream_width'] = ((output_width + 7) // 8) * 8
        
        # Calculate AXI-Lite address width if needed
        if template_values['USE_AXILITE']:
            import math
            c = template_values['C']
            n = template_values['N']
            addr_width = math.ceil(math.log2(c/pe)) + math.ceil(math.log2(pe)) + n + 1
            simplified_values['axilite_addr_width'] = int(addr_width)
        else:
            simplified_values['axilite_addr_width'] = 0
            
        return simplified_values

    # =============================================================================
    # OPERATION-SPECIFIC METHODS (threshold file generation)
    # =============================================================================

    def generate_threshold_files(self, model):
        """Generate threshold data files for RTL simulation."""
        thresholds = model.get_initializer(self.onnx_node.input[1])
        bias = self.get_nodeattr("ActVal")
        output_data_type = self.get_nodeattr("outputDataType")
        input_data_type = self.get_nodeattr("inputDataType")
        o_bitwidth = DataType[output_data_type].bitwidth()

        t_path = self.get_nodeattr("code_gen_dir_ipgen")
        if self.get_nodeattr("runtime_writeable_weights") == 1:
            thresh_file_name = f"{t_path}/memblock.dat"
            self.make_weight_file(thresholds, "decoupled", thresh_file_name)

        # The RTL expects 2^N-1 thresholds, but narrow range quantization will result in
        # one less threshold, prepending a dummy threshold (minimal possible value determined by
        # input data type) and decrease the bias by 1.
        # Additionally, increase number of threshold steps to reflect new shape
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.get_nodeattr("numSteps")
        wdt = self.get_input_datatype(1)
        if expected_thresholds != n_thres_steps:
            if DataType[output_data_type].signed():
                min_val = wdt.min()
                thresholds = np.insert(thresholds, 0, min_val, axis=1)
                bias = bias - 1
            # TODO: temporary fix for unsigned narrow quantization
            else:
                max_val = wdt.max()
                if max_val > DataType[input_data_type].max():
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
                else:
                    max_val = max_val + 1
                    # increase wdt
                    if not wdt.signed():
                        wdt = DataType.get_smallest_possible(max_val)
                    else:
                        wdt = DataType.get_smallest_possible(-max_val - 1)
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
            n_thres_steps += 1

        # add dummy dimension as final dimension (that's what gets packed with next call)
        t_expand = np.expand_dims(thresholds, axis=-1)
        bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 4)
        t_packed = pack_innermost_dim_as_hex_string(
            t_expand,
            wdt,
            bw_hexdigit,
            prefix="",
        )

        pe = self.get_nodeattr("PE")
        num_channels = self.get_nodeattr("NumChannels")

        # If a single threshold value is found, broadcast the value
        if t_packed.shape[0] == 1:
            t_packed = np.broadcast_to(t_packed, (pe, expected_thresholds))
            num_channels = pe

        channel_fold = int(num_channels / pe)

        for stage in range(o_bitwidth):
            sn = o_bitwidth - stage - 1
            for pe_value in range(pe):
                thresh_file = t_path + "/%s_threshs_%s_%s.dat" % (
                    self.onnx_node.name,
                    pe_value,
                    stage,
                )
                threshs = np.zeros([channel_fold * (2**stage)], dtype="object")
                for ch in range(channel_fold):
                    for i in range(2**stage):
                        threshs[(ch << stage) + i] = t_packed[ch * pe + pe_value][
                            (i << (o_bitwidth - stage)) + 2**sn - 1
                        ]
                with open(thresh_file, "w") as f:
                    for val in threshs:
                        f.write(val + "\n")

    def copy_rtl_library_files(self, code_gen_dir):
        """Copy RTL library files to generation directory."""
        sv_files = ["axilite_if.v", "thresholding.sv", "thresholding_axi.sv"]
        rtlsrc = os.environ["FINN_ROOT"] + "/finn-rtllib/thresholding/hdl"
        for sv_file in sv_files:
            shutil.copy(rtlsrc + "/" + sv_file, code_gen_dir)

    def generate_hdl(self, model, fpgapart, clk):
        """Generate HDL using simplified RTL template."""
        # Generate threshold data files
        self.generate_threshold_files(model)
        
        # Get template values and convert to match simplified template
        template_values = self.get_rtl_wrapper_values()
        
        # Convert keys to match simplified template expectations
        simplified_values = {
            'module_name': template_values['MODULE_NAME_AXI_WRAPPER'],
            'n': template_values['N'],
            'wi': template_values['WI'],
            'wt': template_values['WT'],
            'c': template_values['C'],
            'pe': template_values['PE'],
            'signed': template_values['SIGNED'],
            'fparg': template_values['FPARG'],
            'bias': template_values['BIAS'],
            'thresholds_path': template_values['THRESHOLDS_PATH'],
            'use_axilite': template_values['USE_AXILITE'],
            'depth_trigger_uram': template_values['DEPTH_TRIGGER_URAM'],
            'depth_trigger_bram': template_values['DEPTH_TRIGGER_BRAM'],
            'deep_pipeline': template_values['DEEP_PIPELINE'],
            'o_bits': template_values['O_BITS'],
        }
        
        # Calculate stream widths
        pe = template_values['PE']
        wi = template_values['WI']
        o_bits = template_values['O_BITS']
        
        input_width = pe * wi
        output_width = pe * o_bits
        
        # Pad to byte boundaries
        simplified_values['input_stream_width'] = ((input_width + 7) // 8) * 8
        simplified_values['output_stream_width'] = ((output_width + 7) // 8) * 8
        
        # Calculate AXI-Lite address width if needed
        if template_values['USE_AXILITE']:
            import math
            c = template_values['C']
            n = template_values['N']
            addr_width = math.ceil(math.log2(c/pe)) + math.ceil(math.log2(pe)) + n + 1
            simplified_values['axilite_addr_width'] = int(addr_width)
        else:
            simplified_values['axilite_addr_width'] = 0
        
        # Use simplified template
        from finn.codegen.template_engine import TemplateEngine
        engine = TemplateEngine()
        rtl_code = engine.render("thresholding_rtl.v.j2", simplified_values)
        
        # Write RTL file
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        module_name = simplified_values['module_name']
        rtl_path = os.path.join(code_gen_dir, f"{module_name}.v")
        with open(rtl_path, "w") as f:
            f.write(rtl_code)
        
        # Copy library files
        self.copy_rtl_library_files(code_gen_dir)
        
        # Set node attributes
        self.set_nodeattr("gen_top_module", module_name)
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    # =============================================================================
    # RESOURCE ESTIMATION METHODS (unchanged from original)
    # =============================================================================

    def get_pe_mem_geometries(self):
        """return a list of (bitwidth, depth) for PE memory configurations to be used
        in resource estimation

        for each bitwidth, the depth is calculated as the
        number of thresholds that can be stored in a single
        memory block
        the bitwidth is the bitwidth of the threshold values
        the depth is the number of thresholds that can be stored
        in a single memory block
        the number of memory blocks is calculated as the number
        of thresholds divided by the depth
        the number of memory blocks is then multiplied by the
        number of PEs to get the total number of memory blocks
        required for the entire layer
        """
        pe = self.get_nodeattr("PE")
        wdt = self.get_input_datatype(1)
        wdt_bits = wdt.bitwidth()
        odt = self.get_output_datatype()
        odt_bits = odt.bitwidth()
        t_channels = self.get_nodeattr("NumChannels")
        cf = t_channels / pe
        is_uniform = self.get_nodeattr("uniform_thres")
        if is_uniform:
            ret = [(odt_bits - x, cf * (2**x)) for x in range(1, odt_bits)]
        else:
            ret = [(wdt_bits, (cf) * 2**x) for x in range(odt_bits)]
        return ret

    def get_memory_estimate(self):
        """return the memory estimate for this node"""
        res_dict = {}
        depth_trigger_bram = self.get_nodeattr("depth_trigger_bram")
        depth_trigger_uram = self.get_nodeattr("depth_trigger_uram")
        pe = self.get_nodeattr("PE")
        ret = self.get_pe_mem_geometries()
        for mem_cfg in ret:
            (width, depth) = mem_cfg
            primitives = mem_primitives_versal
            if depth_trigger_bram != 0 or depth_trigger_uram != 0:
                if depth >= depth_trigger_bram and depth < depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "BRAM" in k}
                elif depth >= depth_trigger_uram:
                    primitives = {k: v for (k, v) in mem_primitives_versal.items() if "URAM" in k}
            alts = get_memutil_alternatives(mem_cfg, primitives)
            primary_alt = alts[0]
            res_type = primary_alt[0].split("_")[0]
            res_count, eff, waste = primary_alt[1]
            res_dict[res_type] = res_dict.get(res_type, 0) + pe * res_count
        return res_dict

    def bram_estimation(self):
        """return the number of BRAMs required for this node"""
        res_dict = self.get_memory_estimate()
        return res_dict.get("BRAM", 0)

    def uram_estimation(self):
        """return the number of URAMs required for this node"""
        res_dict = self.get_memory_estimate()
        return res_dict.get("URAM", 0)

    def lut_estimation(self):
        """return the number of LUTs required for this node"""
        res_dict = self.get_memory_estimate()
        return res_dict.get("LUTRAM", 0)

    # =============================================================================
    # INHERITED METHODS (unchanged from original)
    # =============================================================================

    def get_all_meminit_filenames(self, abspath=False):
        "Return a list of all .dat memory initializer files used for this node"
        dat_files = []
        t_path = self.get_nodeattr("code_gen_dir_ipgen") if abspath else "."
        pe = self.get_nodeattr("PE")
        output_data_type = self.get_nodeattr("outputDataType")  # output precision
        o_bitwidth = DataType[output_data_type].bitwidth()
        for stage in range(o_bitwidth):
            for pe_value in range(pe):
                thresh_file = t_path + "/%s_threshs_%s_%s.dat" % (
                    self.onnx_node.name,
                    pe_value,
                    stage,
                )
                dat_files.append(thresh_file)
        return dat_files

    def get_rtl_file_list(self, abspath=False):
        """Thresholding binary search RTL file list"""
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/thresholding/hdl/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [
            rtllib_dir + "axilite_if.v",
            rtllib_dir + "thresholding.sv",
            rtllib_dir + "thresholding_axi.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]
        return verilog_files

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        if mode == "cppsim":
            Thresholding.execute_node(self, context, graph)
        elif mode == "rtlsim":
            node = self.onnx_node
            # create a npy file fore each input of the node (in_ind is input index)
            in_ind = 0
            for inputs in node.input:
                # it is assumed that the first input of the node is the data input
                # the second input are the thresholds
                if in_ind == 0:
                    assert str(context[inputs].dtype) in [
                        "float32",
                        "float16",
                    ], """Input datatype is
                    not float32 or float16 as expected."""
                    expected_inp_shape = self.get_folded_input_shape()
                    reshaped_input = context[inputs].reshape(expected_inp_shape)

                    if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                        # store bipolar activations as binary
                        reshaped_input = (reshaped_input + 1) / 2
                        export_idt = DataType["BINARY"]
                    else:
                        export_idt = self.get_input_datatype(0)

                    # make copy before saving the array
                    reshaped_input = reshaped_input.copy()
                    np.save(
                        os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                        reshaped_input,
                    )
                elif in_ind > 2:
                    raise Exception("Unexpected input found for Thresholding_rtl")
                in_ind += 1

            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []},
            }
            super().reset_rtlsim(sim)
            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)
            rtlsim_output = io_dict["outputs"]["out0"]

            # Manage output data
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()

            rtlsim_output_to_npy(
                rtlsim_output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output
        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def get_rtl_file_list(self, abspath: bool = False) -> List[str]:
        """Returns list of RTL files for thresholding.
        
        Args:
            abspath: Whether to return absolute paths
            
        Returns:
            List of RTL file paths
        """
        rtl_files = []
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        
        # Add wrapper file if it exists
        wrapper_filename = self.get_verilog_top_module_name() + "_wrapper.v"
        wrapper_path = os.path.join(code_gen_dir, wrapper_filename)
        if os.path.exists(wrapper_path):
            rtl_files.append(wrapper_path if abspath else wrapper_filename)
        
        # Add source files from rtllib
        rtl_files.extend(self.get_source_files() if abspath else 
                        [os.path.basename(f) for f in self.get_source_files()])
        
        return rtl_files
    
    def get_verilog_top_module_name(self) -> str:
        """Get the Verilog module name for this operation.
        
        Returns:
            Verilog module name
        """
        # Base it on the node name to ensure uniqueness
        node_name = self.onnx_node.name
        # Ensure valid Verilog identifier
        verilog_name = node_name.replace(".", "_").replace("/", "_")
        return f"{verilog_name}_wrapper"

    def code_generation_ipi(self):
        """Constructs and returns the TCL commands for node instantiation as an RTL
        block."""
        rtl_file_list = self.get_rtl_file_list()
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]

        for rtl_file in rtl_file_list:
            cmd.append(
                "add_files -copy_to %s -norecurse %s"
                % (source_target, os.path.join(code_gen_dir, rtl_file))
            )

        # Create an RTL block, not an IP core (-type ip)
        cmd.append(
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        )

        return cmd

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        if self.get_nodeattr("runtime_writeable_weights") == 1:
            intf_names["axilite"] = ["s_axilite"]

        return intf_names
    
    def _generate_operation_specific_values(self, template_name: str) -> Dict[str, Any]:
        """Generate operation-specific values (required by parent class).
        
        Args:
            template_name: Name of template being generated for
            
        Returns:
            Empty dict - we handle everything in parent class methods
        """
        return {}

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights (thresholds) in appropriate
        format for this layer. This file can be used for either synthesis or
        run-time reconfig of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_name : filename for the weight file to be generated

        """
        thresholds = weights
        pe = self.get_nodeattr("PE")
        ch = self.get_nodeattr("NumChannels")
        output_data_type = self.get_nodeattr("outputDataType")  # output precision
        o_bitwidth = DataType[output_data_type].bitwidth()
        # The RTL expects 2^N-1 thresholds, but narrow range quantization will result in
        # one less threshold, prepending a dummy threshold (minimal possible value determined by
        # input data type) and decrease the bias by 1.
        # Additionally, increase number of threshold steps to reflect new shape
        expected_thresholds = 2**o_bitwidth - 1
        n_thres_steps = self.get_nodeattr("numSteps")
        wdt = self.get_input_datatype(1)
        if expected_thresholds != n_thres_steps:
            if DataType[output_data_type].signed():
                min_val = wdt.min()
                thresholds = np.insert(thresholds, 0, min_val, axis=1)
            # TODO: temporary fix for unsigned narrow quantization
            else:
                max_val = wdt.max()
                if max_val > self.get_input_datatype(0).max():
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
                else:
                    max_val = max_val + 1
                    # increase wdt
                    if not wdt.signed():
                        wdt = DataType.get_smallest_possible(max_val)
                    else:
                        wdt = DataType.get_smallest_possible(-max_val - 1)
                    thresholds = np.insert(thresholds, len(thresholds[0]), max_val, axis=1)
            n_thres_steps += 1

        # If a single threshold value is found, broadcast the value
        if thresholds.shape[0] == 1:
            thresholds = np.broadcast_to(thresholds, (pe, expected_thresholds))
            ch = pe

        width_padded = roundup_to_integer_multiple(thresholds.shape[1], 2**o_bitwidth)
        thresh_padded = np.zeros((thresholds.shape[0], width_padded))
        thresh_padded[: thresholds.shape[0], :n_thres_steps] = thresholds
        thresh_stream = []
        bw_hexdigit = roundup_to_integer_multiple(wdt.bitwidth(), 32)
        padding = np.zeros(width_padded, dtype=np.int32)

        chan_ind = 0
        cf = ch // pe
        for fold in range(cf):
            for c in range(2 ** (pe - 1).bit_length()):
                if (c == 0 or c % pe != 0) and c < pe:
                    for t in thresh_padded[chan_ind]:
                        t_packed = pack_innermost_dim_as_hex_string(
                            [t], wdt, bw_hexdigit, prefix=""
                        ).item()
                        thresh_stream.append(t_packed)
                    chan_ind += 1
                else:
                    for z in padding:
                        t_packed = pack_innermost_dim_as_hex_string(
                            [z], wdt, bw_hexdigit, prefix=""
                        ).item()
                        thresh_stream.append(t_packed)
        with open(weight_file_name, "w") as f:
            for val in thresh_stream:
                f.write(val + "\n")