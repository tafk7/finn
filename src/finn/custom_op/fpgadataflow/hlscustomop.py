# Copyright (c) 2020, Xilinx
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
import re
import shutil
import subprocess
import warnings
from abc import abstractmethod
from pyverilator.util.axi_utils import _read_signal, reset_rtlsim, rtlsim_multi_io, toggle_clk
from qonnx.core.datatype import DataType
from qonnx.custom_op.base import CustomOp
from qonnx.util.basic import roundup_to_integer_multiple

from finn.util.basic import (
    CppBuilder,
    get_rtlsim_trace_depth,
    make_build_dir,
    pyverilate_get_liveness_threshold_cycles,
)
from finn.util.hls import CallHLS
from finn.util.pyverilator import make_single_source_file

from . import templates

try:
    from pyverilator import PyVerilator
except ModuleNotFoundError:
    PyVerilator = None


class HLSCustomOp(CustomOp):
    """HLSCustomOp class all custom ops that correspond to a finn-hlslib
    function are based on. Contains different functions every fpgadataflow
    custom node should have. Some as abstract methods, these have to be filled
    when writing a new fpgadataflow custom op node."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        self.code_gen_dict = {}

        # getting templates from templates.py

        # template for single node execution
        self.docompute_template = templates.docompute_template

        # templates for single node ip generation
        # cpp file
        self.ipgen_template = templates.ipgen_template
        # tcl script
        self.ipgentcl_template = templates.ipgentcl_template

    def get_nodeattr_types(self):
        return {
            "backend": ("s", True, "fpgadataflow"),
            "code_gen_dir_cppsim": ("s", False, ""),
            "code_gen_dir_ipgen": ("s", False, ""),
            "executable_path": ("s", False, ""),
            "ipgen_path": ("s", False, ""),
            "ip_path": ("s", False, ""),
            "ip_vlnv": ("s", False, ""),
            "exec_mode": ("s", False, "", {"", "rtlsim", "cppsim"}),
            "cycles_rtlsim": ("i", False, 0),
            "cycles_estimate": ("i", False, 0),
            "rtlsim_trace": ("s", False, ""),
            "res_estimate": ("s", False, ""),
            "res_hls": ("s", False, ""),
            "res_synth": ("s", False, ""),
            "rtlsim_so": ("s", False, ""),
            # partitioning info
            # ID of SLR to which the Op is attached in Vitis builds
            # Set to -1 as 'don't care'
            "slr": ("i", False, -1),
            # Vitis memory port to which any AXI-MM interface
            # of this Op should be attached in Vitis builds
            # E.g.: "DDR[0]", "HBM[0]", "PLRAM[0]"
            "mem_port": ("s", False, ""),
            # Partition to which the Op belongs; all Ops with the
            # same partition_id are stitched together
            # Users should avoid setting this attribute manually
            # and instead use the floorplan transform to set
            # partition IDs from Vitis design rules and SLR IDs
            "partition_id": ("i", False, 0),
            # ID of FPGA device to which this Op is allocated, in
            # a multi-FPGA setting
            "device_id": ("i", False, 0),
            # input and output FIFO depths for multi-I/O nodes
            "inFIFODepths": ("ints", False, [2]),
            "outFIFODepths": ("ints", False, [2]),
            "output_hook": ("s", False, ""),
            # accumulated characteristic function over two periods
            "io_chrc_in": ("t", False, np.asarray([], dtype=np.int32)),
            "io_chrc_out": ("t", False, np.asarray([], dtype=np.int32)),
            # the period for which the characterization was run
            "io_chrc_period": ("i", False, 0),
            # amount of zero padding inserted during chrc.
            "io_chrc_pads_in": ("ints", False, []),
            "io_chrc_pads_out": ("ints", False, []),
        }

    def get_verilog_top_module_name(self):
        "Return the Verilog top module name for this node."

        node = self.onnx_node
        prefixed_top_name = node.name

        return prefixed_top_name

    def get_verilog_top_module_intf_names(self):
        """Return a dict of names of input and output interfaces.
        The keys reflect the protocols each interface implements:
        'clk', 'rst', 'm_axis', 's_axis', 'aximm', 'axilite'.
        Values are lists of tuples (axis, aximm) or names (axilite):
        'axis' tuples correspond to the list of node inputs in order,
        each tuple is (interface_name, interface_width_bits).
        axilite always assumed to be 32 bits and is not tuple (name only).
        Each block must have at most one aximm and one axilite."""
        intf_names = {}
        intf_names["clk"] = ["ap_clk"]
        intf_names["clk2x"] = ["ap_clk2x"]
        intf_names["rst"] = ["ap_rst_n"]
        sname = self.hls_sname()
        intf_names["s_axis"] = [("in0_" + sname, self.get_instream_width_padded())]
        intf_names["m_axis"] = [("out_" + sname, self.get_outstream_width_padded())]
        intf_names["aximm"] = []
        intf_names["axilite"] = []
        intf_names["ap_none"] = []
        return intf_names

    def get_decoupled_weight_filename(self, abspath):
        """Return the path to decoupled-weight memory init .dat file, if relevant
        for this node, either with absolute path or with relative path depending
        on the abspath parameter. For nonrelevant nodes, returns None."""
        # note that we don't guarantee the existence of the weights file here
        # this is only a utility for returning its path
        # only defined for mem_mode=decoupled
        attr_exists = "mem_mode" in self.get_nodeattr_types()
        if not attr_exists:
            return None
        attr_ok = self.get_nodeattr("mem_mode") == "decoupled"
        if not attr_ok:
            return None
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        weight_filename_rtl = "memblock_{}.dat".format(self.onnx_node.name)
        if abspath:
            weight_filename_rtl = "{}/memblock_{}.dat".format(code_gen_dir, self.onnx_node.name)
        else:
            weight_filename_rtl = "./memblock_{}.dat".format(self.onnx_node.name)
        return weight_filename_rtl

    def get_all_meminit_filenames(self, abspath=False):
        "Return a list of all .dat memory initializer files used for this node"
        # generic implementation:
        # walk the generated verilog and look for references to £readmemh
        verilog_files = self.get_all_verilog_filenames(abspath=True)
        dat_files = []
        # regex to match the filenames from lines containing readmem
        pattern = r'\s*\$readmemh\("([^"]+)",'

        for verilog_file in verilog_files:
            # skip over any non-existing files (even though they shouldn't be in the
            # list, better to handle this gracefully here instead of crashing)
            if not os.path.isfile(verilog_file):
                continue
            with open(verilog_file) as rf:
                vfile_parent = str(Path(verilog_file).parent.absolute())
                lines = rf.read()
                for line in lines.split("\n"):
                    match = re.search(pattern, line)
                    if match:
                        dat_filename = match.group(1)
                        if abspath and dat_filename.startswith("./"):
                            dat_filename = dat_filename.replace("./", vfile_parent + "/")
                        dat_files.append(dat_filename)

        # add decoupled-mode weight memory .dat file separately
        decoupled_w_fn = self.get_decoupled_weight_filename(abspath=abspath)
        if decoupled_w_fn is not None:
            dat_files.append(decoupled_w_fn)
        return dat_files

    def get_verilog_top_filename(self):
        "Return the Verilog top module filename for this node."

        verilog_file = "{}/project_{}/sol1/impl/verilog/{}.v".format(
            self.get_nodeattr("code_gen_dir_ipgen"),
            self.onnx_node.name,
            self.get_verilog_top_module_name(),
        )
        return verilog_file

    def get_all_verilog_paths(self):
        "Return list of all folders containing Verilog code for this node."

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        assert (
            code_gen_dir != ""
        ), """Node attribute "code_gen_dir_ipgen" is
        not set. Please run HLSSynthIP first."""
        verilog_path = "{}/project_{}/sol1/impl/verilog/".format(code_gen_dir, self.onnx_node.name)
        # default impl only returns the HLS verilog codegen dir
        if os.path.isdir(verilog_path):
            return [verilog_path]
        else:
            return []

    def get_all_verilog_filenames(self, abspath=False):
        "Return list of all Verilog files used for this node."

        verilog_files = []
        verilog_paths = self.get_all_verilog_paths()
        for verilog_path in verilog_paths:
            for f in os.listdir(verilog_path):
                if f.endswith(".v") or f.endswith(".sv"):
                    if abspath:
                        verilog_files += [verilog_path + "/" + f]
                    else:
                        verilog_files += [f]
        return verilog_files

    def prepare_rtlsim(self):
        """Creates a Verilator emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path and returns
        a PyVerilator wrapper around it."""

        if PyVerilator is None:
            raise ImportError("Installation of PyVerilator is required.")
        verilog_files = self.get_all_verilog_filenames(abspath=True)
        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        target_file = single_src_dir + "/" + self.get_verilog_top_module_name() + ".v"
        make_single_source_file(verilog_files, target_file)
        dat_files = self.get_all_meminit_filenames(abspath=True)
        for dat_file in dat_files:
            shutil.copy(dat_file, single_src_dir)
        # build the Verilator emu library
        sim = PyVerilator.build(
            self.get_verilog_top_module_name() + ".v",
            build_dir=single_src_dir,
            verilog_path=[single_src_dir],
            trace_depth=get_rtlsim_trace_depth(),
            top_module_name=self.get_verilog_top_module_name(),
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", sim.lib._name)
        return sim

    def get_rtlsim(self):
        """Return a PyVerilator wrapper for the Verilator emulation library
        for this node."""

        rtlsim_so = self.get_nodeattr("rtlsim_so")
        assert os.path.isfile(rtlsim_so), "Cannot find rtlsim library."
        # create PyVerilator wrapper
        sim = PyVerilator(rtlsim_so)
        return sim

    def node_res_estimation(self):
        """Returns summarized resource estimation of BRAMs and LUTs
        of the node as a dictionary."""
        ret = dict()
        ret["BRAM_18K"] = self.bram_estimation()
        ret["BRAM_efficiency"] = self.bram_efficiency_estimation()
        ret["LUT"] = self.lut_estimation()
        ret["URAM"] = self.uram_estimation()
        ret["URAM_efficiency"] = self.uram_efficiency_estimation()
        ret["DSP"] = self.dsp_estimation()
        return ret

    def bram_efficiency_estimation(self):
        """Function for BRAM efficiency estimation: actual parameter storage
        needed divided by the allocated BRAM storage (from estimation)"""
        return 1

    def uram_efficiency_estimation(self):
        """Function for URAM efficiency estimation: actual parameter storage
        needed divided by the allocated URAM storage (from estimation)"""
        return 1

    def bram_estimation(self):
        """Function for BRAM resource estimation, is member function of
        HLSCustomOp class but has to be filled by every node"""
        return 0

    def uram_estimation(self):
        """Function for UltraRAM resource estimation, is member function of
        HLSCustomOp class but has to be filled by every node"""
        return 0

    def lut_estimation(self):
        """Function for LUT resource estimation, is member function of
        HLSCustomOp class but has to be filled by every node"""
        return 0

    def dsp_estimation(self):
        """Function for DSP resource estimation, is member function of
        HLSCustomOp class but has to be filled by every node"""
        return 0

    def get_exp_cycles(self):
        """Function for estimation of expected cycles for set folding,
        is member function of HLSCustomOp class but has to be filled
        by every node"""
        return 0

    def get_op_and_param_counts(self):
        """Return a dictionary with number of ops needed per inference for
        this layer as well as parameter count (weights, thresholds, etc.).
        Entries should be in the format:
        {op_<optype> : <count>, param_<paramtype>: <count>}."""
        return {}

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates c++ code and tcl script for ip generation."""
        node = self.onnx_node

        # generate top cpp file for ip generation
        path = self.get_nodeattr("code_gen_dir_ipgen")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("ipgen")
        self.blackboxfunction()
        self.pragmas()
        self.docompute()

        template = self.ipgen_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        f = open(os.path.join(code_gen_dir, "top_{}.cpp".format(node.name)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

        # generate tcl script for ip generation
        self.code_gen_dict["$PROJECTNAME$"] = ["project_{}".format(node.name)]
        self.code_gen_dict["$HWSRCDIR$"] = [code_gen_dir]
        self.code_gen_dict["$FPGAPART$"] = [fpgapart]
        self.code_gen_dict["$TOPFXN$"] = [node.name]
        self.code_gen_dict["$CLKPERIOD$"] = [str(clk)]
        self.code_gen_dict["$DEFAULT_DIRECTIVES$"] = self.ipgen_default_directives()
        self.code_gen_dict["$EXTRA_DIRECTIVES$"] = self.ipgen_extra_directives()

        template = self.ipgentcl_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        f = open(os.path.join(code_gen_dir, "hls_syn_{}.tcl".format(node.name)), "w")
        f.write(template)
        f.close()
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
        """Generates c++ code for simulation (cppsim)."""
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

        template = self.docompute_template

        for key in self.code_gen_dict:
            # transform list into long string separated by '\n'
            code_gen_line = "\n".join(self.code_gen_dict[key])
            template = template.replace(key, code_gen_line)
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        f = open(os.path.join(code_gen_dir, "execute_{}.cpp".format(node.op_type)), "w")
        f.write(template)
        f.close()
        self.code_gen_dict.clear()

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
        builder.append_includes("-I$FINN_ROOT/deps/cnpy/")
        builder.append_includes("-I$FINN_ROOT/deps/finn-hlslib")
        builder.append_includes("-I$FINN_ROOT/custom_hls")
        builder.append_includes("-I{}/include".format(os.environ["HLS_PATH"]))
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$FINN_ROOT/deps/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    def dynamic_input_to_npy(self, context, count, target_dir=""):
        """Saves input (given context) into .npy files.

        Count indicates the number of inputs that have to be saved."""
        node = self.onnx_node
        if target_dir == "":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
            if code_gen_dir == "":
                raise Exception(
                    """
    Found no codegen dir for this node, did you run the prepare_cppsim transformation?
                """
                )
            target_dir = code_gen_dir
        # create a npy file for each input of the node (in_ind is input index)
        # assuming dynamic inputs start from 0
        for in_ind in range(count):
            current_input_name = node.input[in_ind]
            input_array = context[current_input_name]
            if in_ind == 0:
                expected_inp_shape = self.get_folded_input_shape()
                idt = self.get_input_datatype()
            else:
                expected_inp_shape = self.get_folded_input_shape(in_ind)
                idt = self.get_input_datatype(in_ind)
            reshaped_input = input_array.reshape(expected_inp_shape)
            if idt == DataType["BIPOLAR"]:
                # store bipolar activations as binary
                reshaped_input = (reshaped_input + 1) / 2
            # make copy before saving the array
            reshaped_input = reshaped_input.copy()
            np.save(
                os.path.join(target_dir, "input_{}.npy".format(in_ind)),
                reshaped_input,
            )

    def npy_to_dynamic_output(self, context):
        """Reads the output from an output.npy file generated from cppsim and
        places its content into the context dictionary."""
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        output = np.load("{}/output.npy".format(code_gen_dir))
        exp_shape = self.get_normal_output_shape()
        context[node.output[0]] = output.reshape(exp_shape)

    def npy_to_dynamic_outputs(self, context, npy_list):
        """Reads the output from .npy files generated from cppsim and places
        their content into the context dictionary.
        npy_list is a list specifying which files to read, and its order must
        match the order of node outputs."""
        node = self.onnx_node
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        for i in range(len(npy_list)):
            output = np.load("{}/{}".format(code_gen_dir, npy_list[i]))
            if i == 0:
                exp_shape = self.get_normal_output_shape()
            else:
                exp_shape = self.get_normal_output_shape(i)
            context[node.output[i]] = output.reshape(exp_shape)

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

    def reset_rtlsim(self, sim):
        """Sets reset input in pyverilator to zero, toggles the clock and set it
        back to one"""
        reset_rtlsim(sim)

    def toggle_clk(self, sim):
        """Toggles the clock input in pyverilator once."""
        toggle_clk(sim)

    def hls_sname(self):
        """Get the naming convention used by Vitis HLS for stream signals
        Example: the TDATA for a stream called "out" would be out_V_TDATA.
        """
        return "V"

    def rtlsim(self, sim, inp, inp2=None):
        """Runs the pyverilator simulation by passing the input values to the simulation,
        toggle the clock and observing the execution time. Function contains also an
        observation loop that can abort the simulation if no output value is produced
        after 100 cycles."""

        trace_file = self.get_nodeattr("rtlsim_trace")
        if trace_file != "":
            if trace_file == "default":
                trace_file = self.onnx_node.name + ".vcd"
            sim.start_vcd_trace(trace_file)
        inputs = inp
        outputs = []
        sname = self.hls_sname()
        o_ready = "out_" + sname + "_TREADY"
        o_valid = "out_" + sname + "_TVALID"
        o_data = "out_" + sname + "_TDATA"
        in0_ready = "in0_" + sname + "_TREADY"
        in0_valid = "in0_" + sname + "_TVALID"
        in0_data = "in0_" + sname + "_TDATA"
        in1_ready = "in1_" + sname + "_TREADY"
        in1_valid = "in1_" + sname + "_TVALID"
        in1_data = "in1_" + sname + "_TDATA"

        sim.io[o_ready] = 1

        # observe if output is completely calculated
        # observation_count will contain the number of cycles the calculation ran
        num_out_values = self.get_number_output_values()
        output_observed = False
        observation_count = 0

        # avoid infinite looping of simulation by aborting when there is no change in
        # output values after 100 cycles
        no_change_count = 0
        old_outputs = outputs
        liveness_threshold = pyverilate_get_liveness_threshold_cycles()

        while not (output_observed):
            sim.io[in0_valid] = 1 if len(inputs) > 0 else 0
            sim.io[in0_data] = inputs[0] if len(inputs) > 0 else 0
            if sim.io[in0_ready] == 1 and sim.io[in0_valid] == 1:
                inputs = inputs[1:]

            if inp2 is not None:
                sim.io[in1_valid] = 1 if len(inp2) > 0 else 0
                sim.io[in1_data] = inp2[0] if len(inp2) > 0 else 0
                if sim.io[in1_ready] == 1 and sim.io[in1_valid] == 1:
                    inp2 = inp2[1:]

            if sim.io[o_valid] == 1 and sim.io[o_ready] == 1:
                outputs = outputs + [sim.io[o_data]]
            sim.io.ap_clk = 1
            sim.io.ap_clk = 0

            observation_count = observation_count + 1
            no_change_count = no_change_count + 1

            if len(outputs) == num_out_values:
                self.set_nodeattr("cycles_rtlsim", observation_count)
                output_observed = True

            if no_change_count == liveness_threshold:
                if old_outputs == outputs:
                    if trace_file != "":
                        sim.flush_vcd_trace()
                        sim.stop_vcd_trace()
                    raise Exception(
                        "Error in simulation! Takes too long to produce output. "
                        "Consider setting the LIVENESS_THRESHOLD env.var. to a "
                        "larger value."
                    )
                else:
                    no_change_count = 0
                    old_outputs = outputs
        if trace_file != "":
            sim.flush_vcd_trace()
            sim.stop_vcd_trace()
        return outputs

    def rtlsim_multi_io(self, sim, io_dict):
        "Run rtlsim for this node, supports multiple i/o streams."

        # signal name
        sname = "_" + self.hls_sname() + "_"

        trace_file = self.get_nodeattr("rtlsim_trace")
        if trace_file == "default":
            trace_file = self.onnx_node.name + ".vcd"
        num_out_values = self.get_number_output_values()
        total_cycle_count = rtlsim_multi_io(
            sim,
            io_dict,
            num_out_values,
            trace_file=trace_file,
            sname=sname,
            liveness_threshold=pyverilate_get_liveness_threshold_cycles(),
        )
        self.set_nodeattr("cycles_rtlsim", total_cycle_count)

    def execute_node(self, context, graph):
        """Executes single node using cppsim or rtlsim."""
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            # save input(s)
            self.dynamic_input_to_npy(context, 1)
            # execute the precompiled model
            self.exec_precompiled_singlenode_model()
            # load output npy file
            self.npy_to_dynamic_output(context)
        elif mode == "rtlsim":
            pass

        else:
            raise Exception(
                """Invalid value for attribute exec_mode! Is currently set to: {}
            has to be set to one of the following value ("cppsim", "rtlsim")""".format(
                    mode
                )
            )

    def generate_params(self, model, path):
        """Function to generate parameters (i.e. weights and thresholds),
        is member function of HLSCustomOp class but has to be filled
        by every node."""
        pass

    @abstractmethod
    def get_number_output_values(self):
        """Function to get the number of expected output values,
        is member function of HLSCustomOp class but has to be filled
        by every node."""
        pass

    @abstractmethod
    def global_includes(self):
        """Function to set the global includes for c++ code that has to be generated
        for cppsim or rtlsim, is member function of HLSCustomOp class but has to
        be filled by every node."""
        pass

    @abstractmethod
    def defines(self, var):
        """Function to set the define commands for c++ code that has to be generated
        for cppsim or rtlsim, is member function of HLSCustomOp class but has to
        be filled by every node.

        var: makes it possible to reuse the function for different c++ code generation.
        I.e. if set to "ipgen" in MatrixVectorActivation additional PRAGMA defines are
        added."""
        pass

    @abstractmethod
    def read_npy_data(self):
        """Function to generate the commands for reading data from .npy file in c++,
        is member function of HLSCustomOp class but has to be filled by every node."""
        pass

    @abstractmethod
    def strm_decl(self):
        """Function to generate the commands for the stream declaration in c++,
        is member function of HLSCustomOp class but has to be filled
        by every node."""
        pass

    @abstractmethod
    def docompute(self):
        """Function to generate the commands for the computational part of the
        c++ code, is member function of HLSCustomOp class but has to be filled
        by every node."""
        pass

    @abstractmethod
    def dataoutstrm(self):
        """Function to generate the commands for reading out data from c++ and convert
        into npy format, is member function of HLSCustomOp class but has to be filled
        by every node."""
        pass

    @abstractmethod
    def save_as_npy(self):
        """Function to generate the commands for saving data in .npy file in c++,
        is member function of HLSCustomOp class but has to be filled by every node."""
        pass

    @abstractmethod
    def blackboxfunction(self):
        """Function to generate a blackbock function in c++ from which an IP block
        will be generated, is member function of HLSCustomOp class but has to be filled
        by every node."""
        pass

    @abstractmethod
    def pragmas(self):
        """Function to generate the pragma commands in c++, is member function of
        HLSCustomOp class but has to be filled by every node."""
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input stream ind."""
        raise Exception("get_input_datatype not implemented for this op")

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output stream ind."""
        raise Exception("get_output_datatype not implemented for this op")

    def get_normal_input_shape(self, ind=0):
        """Returns normal input shape if implemented."""
        raise Exception("get_normal_input_shape not implemented for this op")

    def get_normal_output_shape(self, ind=0):
        """Returns folded output shape if implemented."""
        raise Exception("get_normal_output_shape not implemented for this op")

    def get_folded_input_shape(self, ind=0):
        """Returns folded input shape (according to synapse folding), if implemented."""
        raise Exception("get_folded_input_shape not implemented for this op")

    def get_folded_output_shape(self, ind=0):
        """Returns folded output shape (according to neuron folding), if implemented."""
        raise Exception("get_folded_output_shape not implemented for this op")

    def get_instream_width(self, ind=0):
        """Returns input stream width, if implemented."""
        raise Exception("get_instream_width not implemented for this op")

    def get_outstream_width(self, ind=0):
        """Returns output stream width, if implemented."""
        raise Exception("get_outstream_width not implemented for this op")

    def get_instream_width_padded(self, ind=0):
        """Returns input stream width padded to a multiple of 8. This is required
        by the AXI Stream spec."""
        in_width = self.get_instream_width(ind=ind)
        return roundup_to_integer_multiple(in_width, 8)

    def get_outstream_width_padded(self, ind=0):
        """Returns output stream width padded to a multiple of 8. This is required
        by the AXI Stream spec."""
        out_width = self.get_outstream_width(ind=ind)
        return roundup_to_integer_multiple(out_width, 8)

    def get_ap_int_max_w(self):
        """Return the maximum width of any ap_int used in this module. Used to set the
        AP_INT_MAX_W definition for HLS."""
        instream = self.get_instream_width()
        outstream = self.get_outstream_width()
        ret = max([instream, outstream])
        assert ret <= 32768, "AP_INT_MAX_W=%d is larger than allowed maximum of 32768" % ret
        return ret

    def derive_characteristic_fxns(self, period, override_rtlsim_dict=None):
        """Return the unconstrained characteristic functions for this node."""
        # ensure rtlsim is ready
        assert self.get_nodeattr("rtlsim_so") != "", "rtlsim not ready for " + self.onnx_node.name
        if self.get_nodeattr("io_chrc_period") > 0:
            warnings.warn("Skipping node %s: already has FIFO characteristic" % self.onnx_node.name)
            return
        exp_cycles = self.get_exp_cycles()
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        n_outs = np.prod(self.get_folded_output_shape()[:-1])
        if exp_cycles == 0:
            # try to come up with an optimistic estimate
            exp_cycles = min(n_inps, n_outs)
        assert (
            exp_cycles <= period
        ), "Period %d too short to characterize %s : expects min %d cycles" % (
            period,
            self.onnx_node.name,
            exp_cycles,
        )
        sim = self.get_rtlsim()
        # signal name
        sname = "_" + self.hls_sname() + "_"
        if override_rtlsim_dict is not None:
            io_dict = override_rtlsim_dict
        else:
            io_dict = {
                "inputs": {
                    "in0": [0 for i in range(n_inps)],
                },
                "outputs": {"out": []},
            }

        # extra dicts to keep track of cycle-by-cycle transaction behavior
        # note that we restrict key names to filter out weight streams etc
        txns_in = {key: [] for (key, value) in io_dict["inputs"].items() if "in" in key}
        txns_out = {key: [] for (key, value) in io_dict["outputs"].items() if "out" in key}

        def monitor_txns(sim_obj):
            for inp in txns_in:
                in_ready = _read_signal(sim, inp + sname + "TREADY") == 1
                in_valid = _read_signal(sim, inp + sname + "TVALID") == 1
                if in_ready and in_valid:
                    txns_in[inp].append(1)
                else:
                    txns_in[inp].append(0)
            for outp in txns_out:
                if (
                    _read_signal(sim, outp + sname + "TREADY") == 1
                    and _read_signal(sim, outp + sname + "TVALID") == 1
                ):
                    txns_out[outp].append(1)
                else:
                    txns_out[outp].append(0)

        reset_rtlsim(sim)
        total_cycle_count = rtlsim_multi_io(
            sim,
            io_dict,
            n_outs,
            sname=sname,
            liveness_threshold=period,
            hook_preclk=monitor_txns,
        )
        assert (
            total_cycle_count <= period
        ), """Total cycle count from rtl simulation is higher than
            specified period, please set the period higher than {}""".format(
            total_cycle_count
        )
        self.set_nodeattr("io_chrc_period", period)

        def accumulate_char_fxn(chrc):
            p = len(chrc)
            ret = []
            for t in range(2 * p):
                if t == 0:
                    ret.append(chrc[0])
                else:
                    ret.append(ret[-1] + chrc[t % p])
            return np.asarray(ret, dtype=np.int32)

        all_txns_in = np.empty((len(txns_in.keys()), 2 * period), dtype=np.int32)
        all_txns_out = np.empty((len(txns_out.keys()), 2 * period), dtype=np.int32)
        all_pad_in = []
        all_pad_out = []
        for in_idx, in_strm_nm in enumerate(txns_in.keys()):
            txn_in = txns_in[in_strm_nm]
            if len(txn_in) < period:
                pad_in = period - len(txn_in)
                txn_in += [0 for x in range(pad_in)]
            txn_in = accumulate_char_fxn(txn_in)
            all_txns_in[in_idx, :] = txn_in
            all_pad_in.append(pad_in)

        for out_idx, out_strm_nm in enumerate(txns_out.keys()):
            txn_out = txns_out[out_strm_nm]
            if len(txn_out) < period:
                pad_out = period - len(txn_out)
                txn_out += [0 for x in range(pad_out)]
            txn_out = accumulate_char_fxn(txn_out)
            all_txns_out[out_idx, :] = txn_out
            all_pad_out.append(pad_out)

        self.set_nodeattr("io_chrc_in", all_txns_in)
        self.set_nodeattr("io_chrc_out", all_txns_out)
        self.set_nodeattr("io_chrc_pads_in", all_pad_in)
        self.set_nodeattr("io_chrc_pads_out", all_pad_out)
