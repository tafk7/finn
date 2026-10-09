# Copyright (c) 2020, Xilinx
# Copyright (C) 2026, Advanced Micro Devices, Inc.
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
import onnx
import onnx.numpy_helper as nph
from importlib import resources as importlib
from pkgutil import get_data


def get_test_model(netname, wbits, abits, pretrained):
    """Returns the model specified by input arguments from the Brevitas BNN-PYNQ
    test networks. Pretrained weights loaded if pretrained is True."""
    from brevitas_examples import bnn_pynq, imagenet_classification  # noqa: PLC0415

    # map of (wbits,abits) -> model
    example_map = {
        ("CNV", 1, 1): bnn_pynq.cnv_1w1a,
        ("CNV", 1, 2): bnn_pynq.cnv_1w2a,
        ("CNV", 2, 2): bnn_pynq.cnv_2w2a,
        ("LFC", 1, 1): bnn_pynq.lfc_1w1a,
        ("LFC", 1, 2): bnn_pynq.lfc_1w2a,
        ("SFC", 1, 1): bnn_pynq.sfc_1w1a,
        ("SFC", 1, 2): bnn_pynq.sfc_1w2a,
        ("SFC", 2, 2): bnn_pynq.sfc_2w2a,
        ("TFC", 1, 1): bnn_pynq.tfc_1w1a,
        ("TFC", 1, 2): bnn_pynq.tfc_1w2a,
        ("TFC", 2, 2): bnn_pynq.tfc_2w2a,
        ("mobilenet", 4, 4): imagenet_classification.quant_mobilenet_v1_4b,
    }
    model_cfg = (netname, wbits, abits)
    model_def_fxn = example_map[model_cfg]
    fc = model_def_fxn(pretrained)
    return fc.eval()


def get_test_model_trained(netname, wbits, abits):
    "get_test_model with pretrained=True"
    return get_test_model(netname, wbits, abits, pretrained=True)


def get_test_model_untrained(netname, wbits, abits):
    "get_test_model with pretrained=False"
    return get_test_model(netname, wbits, abits, pretrained=False)


def get_example_input(topology):
    "Get example numpy input tensor for given topology."

    if "fc" in topology:
        raw_i = get_data("qonnx.data", "onnx/mnist-conv/test_data_set_0/input_0.pb")
        onnx_tensor = onnx.load_tensor_from_string(raw_i)
        return nph.to_array(onnx_tensor)
    elif topology == "cnv":
        ref = importlib.files("finn.qnn-data") / "cifar10/cifar10-test-data-class3.npz"
        with importlib.as_file(ref) as fn:
            input_tensor = np.load(fn)["arr_0"].astype(np.float32)
        return input_tensor
    else:
        raise Exception("Unknown topology, can't return example input")


def get_trained_network_and_ishape(topology, wbits, abits):
    "Return (trained_model, shape) for given BNN-PYNQ test config."

    topology_to_ishape = {
        "tfc": (1, 1, 28, 28),
        "lfc": (1, 1, 28, 28),
        "cnv": (1, 3, 32, 32),
    }
    ishape = topology_to_ishape[topology]
    model = get_test_model_trained(topology.upper(), wbits, abits)
    return (model, ishape)


def resize_smaller_side(target_pixels, img):
    """Resizes smallest side of image to target pixels and resizes larger side with
    same ratio. Expects a PIL image."""
    import torchvision.transforms.functional as torchvision_util  # noqa: PLC0415

    return torchvision_util.resize(img, target_pixels)


def crop_center(size, img):
    """Crop central size*size window out of a PIL image."""
    import torchvision.transforms.functional as torchvision_util  # noqa: PLC0415

    return torchvision_util.center_crop(img, size)
