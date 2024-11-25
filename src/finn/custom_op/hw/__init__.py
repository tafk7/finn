# Copyright (C) 2020-2022, Xilinx, Inc.
# Copyright (C) 2023-2024, Advanced Micro Devices, Inc.
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

from finn.custom_op.hw.addstreams import AddStreams
from finn.custom_op.hw.channelwise_op import ChannelwiseOp
from finn.custom_op.hw.concat import StreamingConcat
from finn.custom_op.hw.convolutioninputgenerator import (
    ConvolutionInputGenerator,
)
from finn.custom_op.hw.downsampler import DownSampler
from finn.custom_op.hw.duplicatestreams import DuplicateStreams
from finn.custom_op.hw.fmpadding import FMPadding
from finn.custom_op.hw.fmpadding_pixel import FMPadding_Pixel
from finn.custom_op.hw.globalaccpool import GlobalAccPool
from finn.custom_op.hw.labelselect import LabelSelect
from finn.custom_op.hw.lookup import Lookup
from finn.custom_op.hw.matrixvectoractivation import MVAU
from finn.custom_op.hw.pool import Pool
from finn.src.finn.custom_op.streamingdataflowpartition import (
    StreamingDataflowPartition,
)
from finn.custom_op.hw.streamingdatawidthconverter import (
    StreamingDataWidthConverter,
)
from finn.custom_op.hw.streamingeltwise import StreamingEltwise
from finn.custom_op.hw.streamingfifo import StreamingFIFO
from finn.custom_op.hw.streamingmaxpool import StreamingMaxPool
from finn.custom_op.hw.thresholding import Thresholding
from finn.custom_op.hw.upsampler import UpsampleNearestNeighbour
from finn.custom_op.hw.vectorvectoractivation import VVAU

custom_op = dict()

# make sure new HLSCustomOp subclasses are imported here so that they get
# registered and plug in correctly into the infrastructure
custom_op["MVAU"] = MVAU
custom_op["StreamingFIFO"] = StreamingFIFO
custom_op["Thresholding"] = Thresholding
custom_op["VVAU"] = VVAU
custom_op["StreamingDataflowPartition"] = StreamingDataflowPartition

custom_op["AddStreams"] = AddStreams
custom_op["ChannelwiseOp"] = ChannelwiseOp
custom_op["ConvolutionInputGenerator"] = ConvolutionInputGenerator
custom_op["DownSampler"] = DownSampler
custom_op["DuplicateStreams"] = DuplicateStreams
custom_op["FMPadding"] = FMPadding
custom_op["FMPadding_Pixel"] = FMPadding_Pixel
custom_op["GlobalAccPool"] = GlobalAccPool
custom_op["LabelSelect"] = LabelSelect
custom_op["Lookup"] = Lookup
custom_op["Pool"] = Pool
custom_op["StreamingConcat"] = StreamingConcat
custom_op["StreamingDataWidthConverter"] = StreamingDataWidthConverter
custom_op["StreamingEltwise"] = StreamingEltwise
custom_op["StreamingMaxPool"] = StreamingMaxPool
custom_op["UpsampleNearestNeighbour"] = UpsampleNearestNeighbour
