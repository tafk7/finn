# Copyright (c) 2020-2022, Xilinx
# Copyright (C) 2022-2024, Advanced Micro Devices, Inc.
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
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes


class RoundAndClipThresholds(Transformation):
    """Rounds and clips the thresholds of each MultiThreshold whose input is an
    integer or fixed-point type, keeping every count it computes.

    The thresholds are rounded up to the input type's next representable value,
    clipped to ``[min, max + step]`` of that type, and annotated with the smallest
    type that holds them. They are stored in float64: ONNX gives thresholds no type,
    and float64 holds every integer threshold up to 2**53 exactly, where float32
    rounds past 2**24. InferDataTypes runs afterwards to carry the new annotations
    through the graph."""

    def apply(self, model: ModelWrapper):  # noqa
        graph_modified = False
        for node in model.graph.node:
            if node.op_type != "MultiThreshold":
                continue
            thresholds = model.get_initializer(node.input[1])
            if thresholds is None:
                continue
            dtype = model.get_tensor_datatype(node.input[0])
            if dtype.is_integer():
                new_thresholds = np.clip(
                    np.ceil(thresholds.astype(np.float64)), dtype.min(), dtype.max() + 1
                )
                # The clipped thresholds need one bit more than the input type.
                max_val = dtype.max() + 1
                if not dtype.signed():
                    tdt = DataType.get_smallest_possible(max_val)
                else:
                    tdt = DataType.get_smallest_possible(-(max_val) - 1)
            elif dtype.is_fixed_point():
                new_thresholds = np.clip(
                    np.ceil(thresholds.astype(np.float64) / dtype.scale_factor())
                    * dtype.scale_factor(),
                    dtype.min(),
                    dtype.max() + dtype.scale_factor(),
                )
                max_val = dtype.max() / dtype.scale_factor() + 1
                tdt_int = DataType.get_smallest_possible(-max_val - 1)
                tdt = DataType[
                    f"FIXED<{tdt_int.bitwidth()},{tdt_int.bitwidth() - dtype.frac_bits()}>"
                ]
            else:
                continue
            model.set_initializer(node.input[1], new_thresholds)
            model.set_tensor_datatype(node.input[1], tdt)
            if np.any(new_thresholds != thresholds):
                # Stop at the first change so the new types propagate before the next
                # node is considered; the transform is repeated until nothing changes.
                graph_modified = True
                break

        model = model.transform(InferDataTypes())
        return model, graph_modified
