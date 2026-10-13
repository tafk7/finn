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

"""The pretrained Brevitas example networks the tests export."""

import torch


def get_test_model_trained(netname: str, wbits: int, abits: int) -> torch.nn.Module:
    """The Brevitas BNN-PYNQ test network ``netname`` (``TFC``, ``CNV``, ...) at ``wbits``
    weight and ``abits`` activation bits, its pretrained weights loaded, in eval mode."""
    from brevitas_examples import (  # type: ignore[import-untyped]  # noqa: PLC0415
        bnn_pynq,
        imagenet_classification,
    )

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
    network: torch.nn.Module = example_map[(netname, wbits, abits)](True)
    return network.eval()
