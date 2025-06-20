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

import numpy as np
import os
import textwrap
from math import ceil, log2
from typing import Dict, Any
from qonnx.core.datatype import DataType
from qonnx.util.basic import roundup_to_integer_multiple

from finn.custom_op.fpgadataflow.CG_hlsbackend import CG_HLSBackend
from finn.custom_op.fpgadataflow.thresholding import Thresholding
from finn.util.data_packing import (
    npy_to_rtlsim_input,
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    rtlsim_output_to_npy,
)

# ONNX i/o tensor shape assumptions for Thresholding:
# input 0 is the input tensor, shape (..., NumChannels)
# input 1 is the threshold tensor, shape (NumChannels, n_thres)
# output 0 is the output tensor, shape (..., NumChannels) - same as input
# the ... here can be any shape (representing groups of vectors)


class CG_Thresholding_hls_Full(Thresholding, CG_HLSBackend):
    """Full-featured clean thresholding HLS implementation with legacy functionality.
    
    This implementation provides complete feature parity with the legacy Thresholding_hls
    while using the modern CG_HLSBackend and template-based code generation system.
    
    Features include:
    - Memory modes: internal_embedded and internal_decoupled
    - Runtime writeable weights via AXI-lite
    - Resource estimation (BRAM/LUT)
    - Full execution support (cppsim/rtlsim)
    - Hardware integration and IP generation
    """
    
    # Use enhanced HLS template that supports memory modes
    TEMPLATE_NAME = "hls_basic.cpp.j2"

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # memory mode for the thresholds
            # internal_embedded -- embedded thresholds
            # internal_decoupled -- default, streaming thresholds with  streamer packaged inside IP
            "mem_mode": (
                "s",
                False,
                "internal_decoupled",
                {"internal_embedded", "internal_decoupled"},
            ),
            # string defining memory type
            "ram_style": ("s", False, "distributed", {"distributed", "block"}),
            # (mem_mode = internal_decoupled only) whether weights (thresholds) will be
            # writable through an AXI-lite interface during runtime
            # 1 for enabled, 0 for disabled.
            # see finn-rtllib/memstream/doc/README for more about the memory
            # address map used for writable weights
            # IMPORTANT: After using AXI lite to either read or write the weights,
            # always "flush" the accelerator by first passing a dummy input
            # vector through the accelerator. This will get rid of any old
            # weight data from the weight FIFOs.
            "runtime_writeable_weights": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(Thresholding.get_nodeattr_types(self))
        my_attrs.update(CG_HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def bram_estimation(self):
        """Calculates BRAM cost if resource set to BRAM"""
        style = self.get_nodeattr("ram_style")
        P = self.get_nodeattr("PE")
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()
        tmem = self.calc_tmem()

        if style == "block" and tmem > 1:
            return int(ceil(A * P / 16)) * int(ceil(tmem / 1024))
        else:
            return 0

    def lut_estimation(self):
        """Calculates LUT cost, taking memory resource type into account"""
        # TODO add in/out FIFO contributions
        style = self.get_nodeattr("ram_style")
        P = self.get_nodeattr("PE")
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()
        tmem = self.calc_tmem()
        # cost of comparators
        comparator_cost = A * P
        # cost of LUTRAM
        if style == "distributed" and tmem > 1:
            lutram_cost = P * A * int(ceil(tmem / 64))
        else:
            lutram_cost = 0
        # total cost
        return comparator_cost + lutram_cost

    def get_ap_int_max_w(self):
        ap_int_max_w = CG_HLSBackend._calculate_ap_int_max_w(self)
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            weightstream = self.get_instream_width(1)
            ap_int_max_w = max([weightstream, ap_int_max_w])
        return ap_int_max_w

    # ===== Template Support Methods =====

    def get_template_name(self) -> str:
        """Return the template name for this backend."""
        return self.TEMPLATE_NAME
    
    def get_template_values(self, template_name: str) -> Dict[str, Any]:
        """Generate template values for enhanced HLS template with memory mode support.
        
        Args:
            template_name: Name of template being generated for
            
        Returns:
            Dictionary of template values for hls_basic.cpp.j2
        """
        # Get basic parameters
        pe = self.get_nodeattr('PE')
        num_channels = self.get_nodeattr('NumChannels')
        num_steps = self.get_nodeattr('numSteps')
        mem_mode = self.get_nodeattr('mem_mode')
        
        # Data types
        input_dtype = self.get_input_datatype()
        output_dtype = self.get_output_datatype()
        threshold_dtype = self.get_input_datatype(1)
        
        # Generate includes - memory mode dependent
        global_includes = [
            '#include "activations.hpp"',
            '#include "params.h"'
        ]
        
        if mem_mode == "internal_decoupled":
            global_includes.extend([
                '#include "memstream.h"',
                '#include "axi_utils.hpp"'
            ])
        
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
        
        # Memory mode specific defines
        if mem_mode == "internal_decoupled":
            defines.extend([
                f"#define MEMSTREAM_WIDTH {self.get_instream_width(1)}",
                f"#define RUNTIME_WRITEABLE {self.get_nodeattr('runtime_writeable_weights')}"
            ])
        
        # Generate input/output ports - memory mode dependent
        input_ports = [{
            'name': 'in0_V',
            'type': f'ap_uint<{self.get_instream_width()}>'
        }]
        
        # Add weight stream port for decoupled mode
        if mem_mode == "internal_decoupled":
            input_ports.append({
                'name': 'in1_V',
                'type': f'ap_uint<{self.get_instream_width(1)}>'
            })
        
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
        
        # Add weight stream pragmas for decoupled mode
        if mem_mode == "internal_decoupled":
            pragmas.append("#pragma HLS INTERFACE axis port=in1_V")
            
            # Runtime writeable weights need AXI-lite interface
            if self.get_nodeattr('runtime_writeable_weights'):
                pragmas.append("#pragma HLS INTERFACE s_axilite port=return bundle=control")
        
        # Add array partition pragmas if using memory
        if tmem != 0:
            ram_style = self._safe_get_nodeattr("ram_style", "auto")
            if mem_mode == "internal_embedded":
                pragmas.extend([
                    "#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=1",
                    "#pragma HLS ARRAY_PARTITION variable=threshs.parameters complete dim=3"
                ])
                
                if ram_style == "distributed":
                    pragmas.append("#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_LUTRAM")
                elif ram_style == "block":
                    pragmas.append("#pragma HLS RESOURCE variable=threshs.parameters core=ROM_2P_BRAM")
        
        # Generate compute body - memory mode dependent
        inp_hls_str = input_dtype.get_hls_datatype_str()
        out_hls_str = output_dtype.get_hls_datatype_str()
        thold_hls_str = threshold_dtype.get_hls_datatype_str()
        
        if mem_mode == "internal_embedded":
            compute_body = f"""// Thresholding computation (embedded mode)
Thresholding_Batch<{inp_hls_str}, {out_hls_str}, {thold_hls_str}, NumChannels, PE, {tmem}>(
    in0_V, out0_V, threshs.parameters, numReps);"""
        else:  # internal_decoupled
            compute_body = f"""// Thresholding computation (decoupled mode)
Thresholding_Stream_Batch<{inp_hls_str}, {out_hls_str}, {thold_hls_str}, NumChannels, PE>(
    in0_V, out0_V, in1_V, numReps);"""
        
        # Return values for template
        return {
            'function_name': f'{self.onnx_node.name}_compute',
            'ap_int_max_w': self.get_ap_int_max_w(),
            'global_includes': global_includes,
            'defines': defines,
            'input_ports': input_ports,
            'output_ports': output_ports,
            'pragmas': pragmas,
            'compute_body': compute_body,
            'include_wrapper': False,
            'mem_mode': mem_mode
        }

    def _safe_get_nodeattr(self, attr_name: str, default_value=None):
        """Safely get node attribute with default value."""
        try:
            return self.get_nodeattr(attr_name)
        except:
            return default_value
    
    # Abstract methods required by CG_HLSBackend
    def _generate_common_values(self, instance) -> Dict[str, Any]:
        """Generate common template values (required by parent class)."""
        return {}
    
    def _generate_operation_specific_values(self, template_name: str) -> Dict[str, Any]:
        """Generate operation-specific values (required by parent class)."""
        return {}

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate IP generation files using template system."""
        # Generate parameter files first
        self.generate_params(model, self.get_nodeattr("code_gen_dir_ipgen"))
        
        # Generate main HLS code using template engine
        from finn.codegen.template_engine import TemplateEngine
        engine = TemplateEngine()
        
        template_values = self.get_template_values(self.TEMPLATE_NAME)
        template_values['include_wrapper'] = True
        template_values['top_function_name'] = self.onnx_node.name
        
        # Generate code
        cpp_code = engine.render(self.TEMPLATE_NAME, template_values)
        
        # Write main C++ file
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        cpp_path = os.path.join(code_gen_dir, f"top_{self.onnx_node.name}.cpp")
        with open(cpp_path, "w") as f:
            f.write(cpp_code)
        
        # Generate TCL script for synthesis
        tcl_content = f"""# HLS synthesis script for {self.onnx_node.name}
open_project project_{self.onnx_node.name}
add_files {cpp_path}
set_top {self.onnx_node.name}
open_solution sol1
set_part {fpgapart}
create_clock -period {clk} -name default
csynth_design
export_design -format ip_catalog
exit 0
"""
        
        tcl_path = os.path.join(code_gen_dir, f"hls_syn_{self.onnx_node.name}.tcl")
        with open(tcl_path, "w") as f:
            f.write(tcl_content)
        
        # Handle memory streaming for decoupled mode
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            # Generate memory streaming wrapper if needed
            try:
                self.generate_hdl_memstream(fpgapart)
            except AttributeError:
                # Method may be inherited from parent - call parent implementation
                super().code_generation_ipgen(model, fpgapart, clk)

    def get_template_param_values(self):
        """Returns the template parameter values according to input, output and weight
        data types."""
        ret = dict()
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        # fill in TSrcI
        ret["TSrcI"] = "Slice<%s>" % inp_hls_str
        # fill in TDstI
        ret["TDstI"] = "Slice<%s>" % out_hls_str

        return ret

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce a file containing given weights (thresholds) in appropriate
        format for this layer. This file can be used for either synthesis or
        run-time reconfig of weights.

        Arguments:

        * weights : numpy array with weights to be put into the file
        * weight_file_mode : one of {hls_header, decoupled_verilog_dat,
          decoupled_runtime}
        * weight_file_name : filename for the weight file to be generated

        """
        threshold_tensor = self.get_hw_compatible_threshold_tensor(weights)
        tdt = self.get_input_datatype(1)
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        if weight_file_mode == "hls_header":
            # save thresholds in thresh.h
            thresholds_hls_code = numpy_to_hls_code(
                threshold_tensor, tdt, "thresholds", False, True
            )
            # write thresholds into thresh.h
            f_thresh = open(weight_file_name, "w")
            tdt_hls = tdt.get_hls_datatype_str()
            # use binary to export bipolar activations
            export_odt = self.get_output_datatype()
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                export_odt = DataType["BINARY"]
            odt_hls = export_odt.get_hls_datatype_str()
            f_thresh.write(
                "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs \
                = ".format(
                    self.calc_tmem(),
                    self.get_nodeattr("PE"),
                    threshold_tensor.shape[-1],
                    tdt_hls,
                    odt_hls,
                    self.get_nodeattr("ActVal"),
                    "comp::less_equal<%s, %s>" % (tdt_hls, tdt_hls),
                )
            )
            f_thresh.write(thresholds_hls_code)
            f_thresh.close()
        elif "decoupled" in weight_file_mode:
            # streaming thresholds need to be organized differently
            # (1, pe, tmem, n_thres_steps) -> (1, tmem, pe, n_thres_steps)
            decoupled_thres = np.transpose(threshold_tensor, (0, 2, 1, 3))
            # TODO add flips/reversals as needed here
            # (1, tmem, pe, n_thres_steps) -(1, tmem, pe * n_thres_steps)
            pe = self.get_nodeattr("PE")
            n_thres_steps = self.get_nodeattr("numSteps")
            decoupled_thres_pe_flipped = np.flip(decoupled_thres, axis=-2)
            decoupled_thres = decoupled_thres.reshape(1, -1, pe * n_thres_steps)
            decoupled_thres = decoupled_thres.copy()
            decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.reshape(
                1, -1, pe * n_thres_steps
            )
            decoupled_thres_pe_flipped = decoupled_thres_pe_flipped.copy()

            if weight_file_mode == "decoupled_npy":
                # save weight stream into npy for cppsim
                np.save(weight_file_name, decoupled_thres)
            elif weight_file_mode == "decoupled_verilog_dat":
                # convert weight values into hexstring
                weight_width = self.get_instream_width(1)
                # pad to nearest 4 bits to get hex strings
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")
            elif weight_file_mode == "decoupled_runtime":
                # memstream axi-lite interface will map each mem line to
                # one or multiple 32-bit words
                weight_width = self.get_instream_width(1)
                words_per_memwidth = 2 ** ceil(log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32
                # first, pack and ensure padding to 32 bits
                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    decoupled_thres_pe_flipped, tdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        # split into groups of 8 hex digits (= 32 bits)
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise Exception("Decoupled weight export not yet implemented")
        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model, path):
        code_gen_dir = path
        thresholds = model.get_initializer(self.onnx_node.input[1])
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            # save thresholds in thresh.h
            weight_filename = "{}/thresh.h".format(code_gen_dir)
            self.make_weight_file(thresholds, "hls_header", weight_filename)
        elif mem_mode == "internal_decoupled":
            # save internal_decoupled weights for cppsim
            weight_filename_sim = "{}/thresholds.npy".format(code_gen_dir)
            self.make_weight_file(thresholds, "decoupled_npy", weight_filename_sim)
            # also save weights as Verilog .dat file
            weight_filename_rtl = "{}/memblock.dat".format(code_gen_dir)
            self.make_weight_file(thresholds, "decoupled_verilog_dat", weight_filename_rtl)
        else:
            raise Exception("Unrecognized mem_mode")

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        # TODO ensure codegen dir exists
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

        # create a npy file fore each input of the node (in_ind is input index)
        in_ind = 0
        for inputs in node.input:
            # it is assumed that the first input of the node is the data input
            # the second input are the weights
            # the third input are the thresholds
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
                raise Exception("Unexpected input found for Thresholding_Batch")
            in_ind += 1

        if mode == "cppsim":
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # load output npy file
            super().npy_to_dynamic_output(context)
            # reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out
            oshape = self.get_normal_output_shape()
            assert context[node.output[0]].shape == oshape, """Output shape is not as expected"""
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
            super().reset_rtlsim(sim)
            if self.get_nodeattr("mem_mode") == "internal_decoupled":
                wnbits = self.get_instream_width(1)
                export_wdt = self.get_input_datatype(1)
                wei = npy_to_rtlsim_input(
                    "{}/thresholds.npy".format(code_gen_dir), export_wdt, wnbits
                )
                num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
                io_dict = {
                    "inputs": {"in0": inp, "in1": wei * num_w_reps},
                    "outputs": {"out0": []},
                }
            elif self.get_nodeattr("mem_mode") == "internal_embedded":
                io_dict = {
                    "inputs": {"in0": inp},
                    "outputs": {"out0": []},
                }
            else:
                raise Exception("Unrecognized mem_mode")
            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)
            output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output_0.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(output, out_npy_path, odt, out_shape, packed_bits, target_bits)

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

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = ['#include "activations.hpp"']
        if self.get_nodeattr("mem_mode") == "internal_embedded":
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    # TODO check and add whatever missing
    def defines(self, var):
        numReps = 1
        numInputVectors = list(self.get_nodeattr("numInputVectors"))
        total_spatial_size = int(np.prod(numInputVectors))

        self.code_gen_dict["$DEFINES$"] = [
            """#define NumChannels1 {}\n #define PE1 {}\n #define numReps {}\n
               #define ImgDim1 {}""".format(
                self.get_nodeattr("NumChannels"),
                self.get_nodeattr("PE"),
                numReps,
                total_spatial_size,
            )
        ]
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$DEFINES$"].append(
                "#define ActVal1 %d" % self.get_nodeattr("ActVal")
            )
            self.code_gen_dict["$DEFINES$"].append(
                "#define ThresType1 %s" % self.get_input_datatype(1).get_hls_datatype_str()
            )
            self.code_gen_dict["$DEFINES$"].append(
                "#define NumSteps1 %d" % self.get_nodeattr("numSteps")
            )

    def read_npy_data(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype(0)
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width(0)
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "half" if dtype == DataType["FLOAT16"] else "float"
        npy_in = "%s/input_0.npy" % code_gen_dir
        self.code_gen_dict["$READNPYDATA$"] = []
        # note: the innermost dim is reversed for the input
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0_V, false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                npy_in,
            )
        )
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            tdt = self.get_input_datatype(1)
            elem_bits = tdt.bitwidth()
            packed_bits = self.get_instream_width(1)
            packed_hls_type = "ap_uint<%d>" % packed_bits
            elem_hls_type = tdt.get_hls_datatype_str()
            npy_type = "half" if tdt == DataType["FLOAT16"] else "float"
            npy_in = "%s/thresholds.npy" % code_gen_dir

            self.code_gen_dict["$READNPYDATA$"].append(
                'npy2apintstream<%s, %s, %d, %s>("%s", in1_V, false, ImgDim1);'
                % (
                    packed_hls_type,
                    elem_hls_type,
                    elem_bits,
                    npy_type,
                    npy_in,
                )
            )

    def strm_decl(self):
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width(0))
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width())
        )
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> in1_V ("in1_V");'.format(self.get_instream_width(1))
            )

    def docompute(self):
        tmpl_args = self.get_template_param_values()
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """Thresholding_Batch<ImgDim1, NumChannels1, PE1, {}, {}>
                (in0_V, out0_V, threshs, numReps);""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                )
            ]
        elif mem_mode == "internal_decoupled":
            # note that numReps is set to 1 in the invocation below, since
            # - for cppsim the repetition comes from the threshold stream reader+input
            # - for synth the unit runs continuously anyway (ap_ctrl_none)
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<ImgDim1, NumChannels1, PE1, {}, {}, ActVal1, ThresType1, NumSteps1>
                (in0_V, out0_V, in1_V, numReps);""".format(
                    "Thresholding_Stream_Batch",
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                )
            ]
        else:
            raise Exception("Unrecognized mem_mode")

    def dataoutstrm(self):
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        if dtype == DataType["BIPOLAR"]:
            # use binary for bipolar storage
            dtype = DataType["BINARY"]
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output_0.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        # note: the innermost dim is not reversed for the output
        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out0_V, %s, "%s", false);'
            % (
                packed_hls_type,
                elem_hls_type,
                elem_bits,
                npy_type,
                shape_cpp_str,
                npy_out,
            )
        ]

    def blackboxfunction(self):
        if self.get_nodeattr("mem_mode") == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0_V,
                    hls::stream<ap_uint<{}>> &out0_V
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(0),
                    self.get_outstream_width(),
                )
            ]
        elif self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0_V,
                    hls::stream<ap_uint<{}>> &in1_V,
                    hls::stream<ap_uint<{}>> &out0_V
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(0),
                    self.get_instream_width(1),
                    self.get_outstream_width(),
                )
            ]
        else:
            raise Exception("Unrecognized mem_mode")

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if self.get_nodeattr("mem_mode") == "internal_embedded":
            # the threshold tensor is acc_type [PE][TMEM][N_THRES]
            # partition for parallel access along PE and N_THRES
            # dimensions (dims 1 and 3)
            self.code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=1")
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                ("#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds " "complete dim=3")
            )
            # set resource type
            ram_style = self.get_nodeattr("ram_style")
            pe = self.get_nodeattr("PE")
            ich = self.get_nodeattr("NumChannels")
            # if PE less than NumChannels, assign cores according to ram_style;
            # otherwise if PE == NumChannels, Vivado HLS will unroll to FFs
            if pe < ich:
                if ram_style == "distributed":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_LUTRAM")
                    )
                elif ram_style == "block":
                    self.code_gen_dict["$PRAGMAS$"].append(
                        ("#pragma HLS RESOURCE variable=threshs.m_thresholds " "core=ROM_2P_BRAM")
                    )
                else:
                    raise Exception(
                        """Invalid value for attribute ram_style! Is currently set to: {}
                    has to be set to one of ("block", "distributed")""".format(
                            ram_style
                        )
                    )
        elif self.get_nodeattr("mem_mode") == "internal_decoupled":
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")

    def code_generation_ipi(self):
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]
        # add streamer if needed
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            node_name = self.onnx_node.name
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            # create a hierarchy for this layer, with the same port names
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]
            cmd.append("create_bd_cell -type hier %s" % node_name)
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )
            # instantiate the hls ip
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s"
                % (self.get_nodeattr("ip_vlnv"), node_name, node_name)
            )
            # instantiate a streamer and connect it to the IP
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            swg_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
            file_suffix = "_memstream_wrapper.v"
            # automatically find memstream verilog component in code generation directory
            for fname in os.listdir(code_gen_dir):
                if fname.endswith(file_suffix):
                    strm_tmpl = fname
            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                os.path.join(code_gen_dir, strm_tmpl),
                swg_rtllib_dir + "axilite_if.v",
                swg_rtllib_dir + "memstream_axi.sv",
                swg_rtllib_dir + "memstream.sv",
            ]
            for f in sourcefiles:
                cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]
            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (strm_tmpl_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/in1_V]" % (node_name, strm_inst, node_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            # 2x clock is not used for decoupled thresholds
            # simply connect input to the 1x clock for now
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, rst_name, node_name, node_name, rst_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, clk_name, node_name, node_name, clk_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, din_name, node_name, node_name, din_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, dout_name, node_name, node_name, dout_name)
            )
            if runtime_writable:
                # expose axi lite interface for writeable weights
                axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s" % (node_name, axilite_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, axilite_name, node_name, strm_inst, axilite_name)
                )
                # TODO calculate and pass in segment size here
                cmd.append("assign_bd_address")
            cmd.append("save_bd_design")
        elif mem_mode == "internal_embedded":
            # base class impl sufficient for internal_embedded mode
            return super().code_generation_ipi()
        else:
            raise Exception("Unrecognized mem_mode for Thresholding_Batch")
        return cmd

    def get_verilog_top_module_intf_names(self):
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            # only expose axilite interface if attribute is set
            runtime_writable = self.get_nodeattr("runtime_writeable_weights") == 1
            if runtime_writable:
                intf_names["axilite"] = ["s_axilite"]
        return intf_names

    def get_op_and_param_counts(self):
        ret_dict = {}
        weight_bits = self.get_input_datatype(1).bitwidth()
        out_features = self.get_nodeattr("NumChannels")
        num_steps = self.get_nodeattr("numSteps")
        # thresholds are called weights in this layer
        thres_param_type = "param_threshold_%db" % (weight_bits)
        thres_count = out_features * num_steps
        ret_dict[thres_param_type] = thres_count
        return ret_dict

    def ipgen_extra_directives(self):
        "Return a list of extra tcl directives for HLS synthesis."

        return ["config_compile -pipeline_style frp"]

    def derive_characteristic_fxns(self, period):
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {
                "in0": [0 for i in range(n_inps)],
            },
            "outputs": {"out0": []},
        }
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["internal_decoupled", "external"]:
            n_weight_inps = self.calc_tmem()
            num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
            io_dict["inputs"]["in1"] = [0 for i in range(num_w_reps * n_weight_inps)]
        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)

    # ===== Execution-Specific Template Methods =====

    def _generate_execution_headers(self) -> list:
        """Generate headers needed for thresholding execution."""
        headers = ["npy2apintstream.hpp", "streamtools.h", "thresholding.h"]
        
        # Add memory mode specific headers
        mem_mode = self._safe_get_nodeattr('mem_mode', 'external')
        if mem_mode == "internal_embedded":
            headers.append("thresh_param.h")
        elif mem_mode == "internal_decoupled":
            headers.append("weights.h")
            
        return headers

    def _generate_input_ports(self) -> list:
        """Generate input port definitions for thresholding."""
        instream_w = self.get_instream_width()
        return [{
            'name': 'in0_V',
            'type': f'ap_uint<{instream_w}>*',
            'stream_type': f'ap_uint<{instream_w}>',
            'size': self._calculate_input_size(),
        }]

    def _generate_output_ports(self) -> list:
        """Generate output port definitions for thresholding."""
        outstream_w = self.get_outstream_width()
        return [{
            'name': 'out0_V',
            'type': f'ap_uint<{outstream_w}>*',
            'stream_name': 'out0_V_stream',
            'size': self._calculate_output_size(),
        }]

    def _generate_internal_streams(self) -> list:
        """Generate internal stream definitions for thresholding."""
        outstream_w = self.get_outstream_width()
        return [{
            'name': 'out0_V_stream',
            'type': f'ap_uint<{outstream_w}>',
        }]

    def _generate_operation_params(self) -> dict:
        """Generate thresholding-specific parameters."""
        return {
            'NumChannels': self.get_nodeattr("NumChannels"),
            'PE': self.get_nodeattr("PE"),
            'numSteps': self.get_nodeattr("numSteps"),
            'inputDataType': str(self.get_input_datatype()),
            'outputDataType': str(self.get_output_datatype()),
            'numThresh': self._get_num_thresh(),
        }

    def _generate_data_types(self) -> dict:
        """Generate thresholding data type definitions."""
        input_dt = self.get_input_datatype()
        output_dt = self.get_output_datatype()
        thresh_dt = self.get_thresholds_datatype()
        
        return {
            'input_t': input_dt.get_hls_datatype_str(),
            'output_t': output_dt.get_hls_datatype_str(),
            'thresh_t': thresh_dt.get_hls_datatype_str(),
            'ActVal': str(output_dt.get_hls_datatype_str()),
        }

    def _generate_resource_config(self) -> dict:
        """Generate thresholding resource configuration."""
        return {
            'PE_COUNT': self.get_nodeattr("PE"),
            'THRESH_COUNT': self._get_num_thresh(),
            'MEM_MODE': self._safe_get_nodeattr('mem_mode', 'external'),
        }

    def _generate_resource_pragmas(self) -> list:
        """Generate HLS resource pragmas for thresholding."""
        pe = self.get_nodeattr("PE")
        pragmas = [
            f'#pragma HLS RESOURCE variable=thresholds core=RAM_2P_BRAM',
            f'#pragma HLS ARRAY_PARTITION variable=thresholds cyclic factor={pe} dim=1',
        ]
        return pragmas

    def _generate_synthesis_directives(self) -> list:
        """Generate synthesis directives for thresholding."""
        return [
            'set_directive_pipeline "Thresholding_hls/Thresholding_hls_label0"',
            'set_directive_unroll "Thresholding_hls/Thresholding_hls_label1"',
        ]

    def _calculate_input_size(self) -> int:
        """Calculate input tensor size."""
        ishape = self.get_normal_input_shape()
        return int(np.prod(ishape))

    def _calculate_output_size(self) -> int:
        """Calculate output tensor size."""
        oshape = self.get_normal_output_shape()
        return int(np.prod(oshape))

    def _get_num_thresh(self) -> int:
        """Get number of thresholds."""
        return self.get_nodeattr("numSteps") - 1
