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

from finn.custom_op.fpgadataflow.matrixvectoractivation import MVAU
from finn.custom_op.fpgadataflow.CG_rtlbackend import CG_RTLBackend
from finn.util.basic import get_dsp_block, is_versal
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

# ONNX i/o tensor shape assumptions for MatrixVectorActivation_rtl:
# input 0 is the input tensor, shape (.., i_size) = (..., MW)
# input 1 is the weight tensor, shape (i_size, o_size) = (MW, MH)
# output 0 is the output tensor, shape (.., o_size) = (..., MH)
# the ... here can be any shape (representing groups of vectors)


class CG_MVAU_rtl(MVAU, CG_RTLBackend):
    """Clean implementation of FINN MVAU RTL backend.
    
    This is a clean implementation that eliminates legacy compatibility bloat
    and uses only direct template value generation methods. No code_gen_dict usage.
    
    Class that corresponds to finn-rtl Matrix Vector Unit using clean template architecture.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
            # Double-pumped DSPs enabled
            "pumpedCompute": ("i", False, 0, {0, 1}),
        }
        my_attrs.update(MVAU.get_nodeattr_types(self))
        my_attrs.update(CG_RTLBackend.get_nodeattr_types(self))
        return my_attrs

    # =============================================================================
    # CLEAN TEMPLATE VALUE GENERATION METHODS (No code_gen_dict usage)
    # =============================================================================

    def get_rtl_wrapper_values(self, fpgapart, clk):
        """Generate RTL wrapper template values for MVAU operation."""
        # check if settings are valid
        pumped_compute = self.get_nodeattr("pumpedCompute")
        simd = self.get_nodeattr("SIMD")
        if pumped_compute and simd == 1:
            raise Exception(
                "Clock pumping an input of SIMD=1 is not meaningful. Please increase SIMD."
            )
        
        dsp_block = get_dsp_block(fpgapart)
        
        return {
            'MODULE_NAME_AXI_WRAPPER': self.get_verilog_top_module_name(),
            'IS_MVU': 1,
            'COMPUTE_CORE': self._resolve_impl_style(dsp_block),
            'PUMPED_COMPUTE': pumped_compute,
            'MW': self.get_nodeattr("MW"),
            'MH': self.get_nodeattr("MH"),
            'PE': self.get_nodeattr("PE"),
            'SIMD': simd,
            'ACTIVATION_WIDTH': self.get_input_datatype(0).bitwidth(),
            'WEIGHT_WIDTH': self.get_input_datatype(1).bitwidth(),
            'ACCU_WIDTH': self.get_output_datatype().bitwidth(),
            'SIGNED_ACTIVATIONS': 1 if (self.get_input_datatype(0).min() < 0) else 0,
            'SEGMENTLEN': self._resolve_segment_len(clk),
            'NARROW_WEIGHTS': 0,  # Will be updated in generate_hdl
            'FORCE_BEHAVIORAL': 0,  # For synthesis
        }

    def get_rtl_module_name(self):
        """Generate the RTL module name."""
        return self.get_verilog_top_module_name()

    def get_rtl_parameters(self, fpgapart, clk):
        """Generate RTL parameters for MVAU module."""
        pumped_compute = self.get_nodeattr("pumpedCompute")
        simd = self.get_nodeattr("SIMD")
        dsp_block = get_dsp_block(fpgapart)
        
        return {
            'IS_MVU': 1,
            'COMPUTE_CORE': self._resolve_impl_style(dsp_block),
            'PUMPED_COMPUTE': pumped_compute,
            'MW': self.get_nodeattr("MW"),
            'MH': self.get_nodeattr("MH"),
            'PE': self.get_nodeattr("PE"),
            'SIMD': simd,
            'ACTIVATION_WIDTH': self.get_input_datatype(0).bitwidth(),
            'WEIGHT_WIDTH': self.get_input_datatype(1).bitwidth(),
            'ACCU_WIDTH': self.get_output_datatype().bitwidth(),
            'SIGNED_ACTIVATIONS': 1 if (self.get_input_datatype(0).min() < 0) else 0,
            'SEGMENTLEN': self._resolve_segment_len(clk),
        }

    def get_port_declarations(self):
        """Generate port declarations for RTL module."""
        mem_mode = self.get_nodeattr("mem_mode")
        
        ports = {
            'data_input': {
                'name': 'in0_V',
                'direction': 'input',
                'width': self.get_instream_width(0),
                'type': 'axis'
            },
            'data_output': {
                'name': 'out0_V', 
                'direction': 'output',
                'width': self.get_outstream_width(),
                'type': 'axis'
            }
        }
        
        if mem_mode in ["internal_decoupled", "external"]:
            ports['weight_input'] = {
                'name': 'in1_V',
                'direction': 'input', 
                'width': self.get_instream_width(1),
                'type': 'axis'
            }
        
        if mem_mode == "internal_decoupled" and self.get_nodeattr("runtime_writeable_weights"):
            ports['axilite'] = {
                'name': 's_axilite',
                'direction': 'slave',
                'type': 'axilite'
            }
        
        return ports

    def get_source_files(self):
        """Generate list of RTL source files needed."""
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
        
        return [
            os.path.join(rtllib_dir, "mvu_vvu_axi.sv"),
            os.path.join(rtllib_dir, "replay_buffer.sv"),
            os.path.join(rtllib_dir, "mvu_4sx4u.sv"),
            os.path.join(rtllib_dir, "mvu_vvu_8sx9_dsp58.sv"),
            os.path.join(rtllib_dir, "mvu_8sx8u_dsp48.sv"),
        ]

    def get_simulation_files(self):
        """Generate list of files for RTL simulation."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        wrapper_name = self.get_nodeattr("gen_top_module")
        
        return [
            os.path.join(code_gen_dir, f"{wrapper_name}_wrapper_sim.v"),
        ]

    # =============================================================================
    # MVAU-SPECIFIC HELPER METHODS (for compute core selection and optimization)
    # =============================================================================

    def _resolve_segment_len(self, clk):
        """Insert pipeline registers in the DSP58 chain to meet target clock frequency."""
        # ~0.741 ns seems the worst-case delay through first DSP
        # ~0.605 ns seems to be (on average) delay for all subsequent DSPs
        # clk >= (critical_path_dsps - 1) * 0.605 + 0.741
        if self.get_nodeattr("pumpedCompute"):
            ref_clk = clk / 2
            simd_factor = 6
        else:
            ref_clk = clk
            simd_factor = 3

        assert (
            ref_clk > 0.741
        ), """Infeasible clk target of {} ns has been set,
        consider lowering the targeted clock frequency!""".format(
            ref_clk
        )
        critical_path_dsps = np.floor((ref_clk - 0.741) / 0.605 + 1)
        max_chain_len = np.ceil(self.get_nodeattr("SIMD") / simd_factor)
        dsp_chain_len = critical_path_dsps if critical_path_dsps < max_chain_len else max_chain_len
        return int(dsp_chain_len)

    def _resolve_impl_style(self, dsp_block):
        """Based on target device and activation/weight-width, choose the supported RTL compute core."""
        assert (
            self.get_nodeattr("resType") != "lut"
        ), """LUT-based RTL-MVU implementation currently not supported!
        Please change resType for {} to 'dsp' or consider switching to HLS-based MVAU!""".format(
            self.onnx_node.name
        )

        act_width = self.get_input_datatype(0).bitwidth()
        weight_width = self.get_input_datatype(1).bitwidth()

        if dsp_block == "DSP58":
            if act_width <= 4 and weight_width <= 4:
                return "mvu_4sx4u_dsp48e2"
            else:
                return "mvu_vvu_8sx9_dsp58"
        else:
            if act_width <= 4 and weight_width <= 4:
                if dsp_block == "DSP48E1":
                    return "mvu_4sx4u_dsp48e1"
                elif dsp_block == "DSP48E2":
                    return "mvu_4sx4u_dsp48e2"
            else:
                return "mvu_8sx8u_dsp48"

    def _determine_narrow_weights(self, model):
        """Determine if weights are narrow range and return parameter value."""
        weights = model.get_initializer(self.onnx_node.input[1])
        wdt = self.get_input_datatype(1)
        return 0 if np.min(weights) == wdt.min() else 1

    def generate_hdl(self, model, fpgapart, clk):
        """Generate HDL using clean template architecture."""
        # Generate params as part of IP preparation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.generate_params(model, code_gen_dir)

        # Get template values using clean architecture
        template_values = self.get_rtl_wrapper_values(fpgapart, clk)
        
        # Add narrow weights determination
        template_values['NARROW_WEIGHTS'] = self._determine_narrow_weights(model)
        
        # Save top module name so we can refer to it after this node has been renamed
        self.set_nodeattr("gen_top_module", template_values['MODULE_NAME_AXI_WRAPPER'])

        # Generate synthesis wrapper
        template_values['FORCE_BEHAVIORAL'] = 0
        synthesis_code = self.render_template("mvau/rtl/wrapper.v.j2", template_values)
        
        wrapper_path = os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v")
        with open(wrapper_path, "w") as f:
            f.write(synthesis_code)

        # Generate simulation wrapper  
        template_values['FORCE_BEHAVIORAL'] = 1
        simulation_code = self.render_template("mvau/rtl/wrapper.v.j2", template_values)
        
        sim_wrapper_path = os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper_sim.v")
        with open(sim_wrapper_path, "w") as f:
            f.write(simulation_code)

        # Handle memory streaming if needed
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            if self.get_nodeattr("ram_style") == "ultra" and not is_versal(fpgapart):
                runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
                assert (
                    runtime_writeable == 1
                ), """Layer with URAM weights must have runtime_writeable_weights=1
                    if Ultrascale device is targeted."""
            self.generate_hdl_memstream(fpgapart, pumped_memory=self.get_nodeattr("pumpedMemory"))
        
        # Set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    # =============================================================================
    # RESOURCE ESTIMATION METHODS (unchanged from original)
    # =============================================================================

    def lut_estimation(self):
        return 0

    def dsp_estimation(self, fpgapart):
        # multiplication
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        dsp_block = get_dsp_block(fpgapart)
        if dsp_block == "DSP58":
            mult_dsp = P * np.ceil(Q / 3)
        else:
            mult_dsp = np.ceil(P / 4) * Q
        return int(mult_dsp)

    # =============================================================================
    # INHERITED METHODS (unchanged from original)
    # =============================================================================

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.get_nodeattr("mem_mode")
        node = self.onnx_node

        if mode == "cppsim":
            MVAU.execute_node(self, context, graph)
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            # create a npy file fore each input of the node (in_ind is input index)
            in_ind = 0
            for inputs in node.input:
                # it is assumed that the first input of the node is the data input
                # the second input are the weights
                if in_ind == 0:
                    assert (
                        str(context[inputs].dtype) == "float32"
                    ), """Input datatype is
                    not float32 as expected."""
                    expected_inp_shape = self.get_folded_input_shape()
                    reshaped_input = context[inputs].reshape(expected_inp_shape)
                    export_idt = self.get_input_datatype(0)
                    # make copy before saving the array
                    reshaped_input = reshaped_input.copy()
                    np.save(
                        os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                        reshaped_input,
                    )
                elif in_ind > 1:
                    raise Exception("Unexpected input found for MatrixVectorActivation_rtl")
                in_ind += 1
                sim = self.get_rtlsim()
                nbits = self.get_instream_width(0)
                inp = npy_to_rtlsim_input("{}/input_0.npy".format(code_gen_dir), export_idt, nbits)
                super().reset_rtlsim(sim)
                if mem_mode in ["external", "internal_decoupled"]:
                    wnbits = self.get_instream_width(1)
                    export_wdt = self.get_input_datatype(1)
                    wei = npy_to_rtlsim_input(
                        "{}/weights.npy".format(code_gen_dir), export_wdt, wnbits
                    )
                    num_w_reps = np.prod(self.get_nodeattr("numInputVectors"))
                    io_dict = {
                        "inputs": {"in0": inp, "in1": wei * num_w_reps},
                        "outputs": {"out0": []},
                    }
                else:
                    io_dict = {
                        "inputs": {"in0": inp},
                        "outputs": {"out0": []},
                    }
                self.rtlsim_multi_io(sim, io_dict)
                super().close_rtlsim(sim)
                output = io_dict["outputs"]["out0"]
                odt = self.get_output_datatype()
                target_bits = odt.bitwidth()
                packed_bits = self.get_outstream_width()
                out_npy_path = "{}/output.npy".format(code_gen_dir)
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

    def instantiate_ip(self, cmd):
        # instantiate the RTL IP
        node_name = self.onnx_node.name
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
        sourcefiles = [
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            rtllib_dir + "mvu_vvu_axi.sv",
            rtllib_dir + "replay_buffer.sv",
            rtllib_dir + "mvu_4sx4u.sv",
            rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",
            rtllib_dir + "mvu_8sx8u_dsp48.sv",
        ]
        for f in sourcefiles:
            cmd.append("add_files -norecurse %s" % (f))
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    node_name,
                    node_name,
                )
            )
            # if using 2x pumped compute, connect the MVU's 2x clk input
            # to the 2x clock port. Otherwise connect 2x clk to regular clk port
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            if self.get_nodeattr("pumpedCompute") or self.get_nodeattr("pumpedMemory"):
                clk2x_name = self.get_verilog_top_module_intf_names()["clk2x"][0]
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                    % (node_name, clk2x_name, node_name, node_name, clk2x_name)
                )
            else:
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                    % (node_name, clk_name, node_name, node_name)
                )
        # external
        else:
            cmd.append(
                "create_bd_cell -type hier -reference %s %s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    node_name,
                )
            )
            # if using 2x pumped compute, connect the MVU's 2x clk input
            # to the 2x clock port. Otherwise connect 2x clk to regular clk port
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            if self.get_nodeattr("pumpedCompute"):
                clk2x_name = self.get_verilog_top_module_intf_names()["clk2x"][0]
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s]"
                    % (node_name, clk2x_name, node_name, clk2x_name)
                )
            else:
                cmd.append(
                    "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/ap_clk2x]"
                    % (node_name, clk_name, node_name)
                )

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""
        verilog_files = [
            code_gen_dir + self.get_nodeattr("gen_top_module") + "_wrapper_sim.v",
            rtllib_dir + "mvu_vvu_axi.sv",
            rtllib_dir + "replay_buffer.sv",
            rtllib_dir + "mvu_4sx4u.sv",
            rtllib_dir + "mvu_vvu_8sx9_dsp58.sv",
            rtllib_dir + "mvu_8sx8u_dsp48.sv",
        ]
        return verilog_files

    def get_verilog_paths(self):
        verilog_paths = super().get_verilog_paths()
        verilog_paths.append(os.environ["FINN_ROOT"] + "/finn-rtllib/mvu")
        return verilog_paths