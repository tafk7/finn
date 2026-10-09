# Copyright (C) 2020, Xilinx, Inc.
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The pynq shell's PYNQ driver, from the partition's integration export
(``finn.transformation.kernels.integration``): its I/O the export's ends (each end's
tensor and shape, the channel's element, and its beats and lanes); it sets PL0 to the
clock it is given, the one the bitfile delivers, and runs the bitfile it is given
(``write_driver``). ``driver_description`` states what it takes and returns.

The driver's text and the files it runs with are the HWCustomOp flow's PYNQ driver's
(finn.transformation.fpgadataflow.make_driver), copied as the shell's frozen extraction:
the template (``templates.pynq_driver_template``), the base driver and validate.py
(``data/pynq_driver``), and the trimmed qonnx and finn helpers the driver imports
(``write_pynq_driver_support``). A kernel-path driver has no external weights and no
MLO configuration.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
from collections.abc import Callable, Sequence
from types import ModuleType
from typing import Any

import numpy as np
import qonnx
import qonnx.util.basic
from qonnx.core.datatype import BaseDataType
from qonnx.util.basic import gen_finn_dt_tensor

import finn.util
import finn.util.data_packing
from finn.shells.pynq import templates
from finn.shells.pynq.ipgen import data_path
from finn.transformation.kernels.integration import Integration
from finn.util.data_packing import finnpy_to_packed_bytearray


def packed_shape(dtype: BaseDataType, folded_shape: tuple[int, ...]) -> tuple[int, ...]:
    """The shape of a folded tensor of ``dtype`` packed into bytes, as the driver packs
    it (``finnpy_to_packed_bytearray``)."""
    dummy = gen_finn_dt_tensor(dtype, folded_shape)
    packed: np.ndarray[Any, Any] = finnpy_to_packed_bytearray(dummy, dtype)  # type: ignore[no-untyped-call]
    return tuple(packed.shape)


def _extract_license_header(source_module: ModuleType) -> str:
    """Return the leading comment/blank-line block from a module's source file.

    Used so that generated minimal copies of a source file keep the original
    copyright/license notice intact."""
    src_lines = inspect.getsource(source_module).splitlines()
    header = []
    for line in src_lines:
        if line.startswith("#") or line.strip() == "":
            header.append(line)
        else:
            break
    return "\n".join(header).rstrip()


def _generate_minimal_module(
    target_file: str,
    source_module: ModuleType,
    functions: Sequence[Callable[..., Any]],
    import_block: str,
) -> None:
    """Write a lightweight copy of source_module to target_file that contains
    only the given functions, the original license header and a minimal import
    block.

    The generated PYNQ driver only needs a small subset of the helper functions
    in qonnx.util.basic and finn.util.data_packing. Copying those files verbatim
    would pull heavy imports (onnx, bitstring) onto the deployment board, even
    though none of that functionality is exercised by the driver. Emitting a
    trimmed-down module keeps the board dependencies limited to numpy (+pynq)."""
    orig_module = source_module.__name__
    license_header = _extract_license_header(source_module)
    note = (
        "# This is a minimal version of {0}, containing only the subset of\n"
        "# functions required by the generated PYNQ driver. It is trimmed down to\n"
        "# keep the runtime dependencies on the deployment board lightweight\n"
        "# (avoiding heavy imports such as onnx / bitstring). Refer to the full\n"
        "# {0} in the original source tree for the complete implementation."
    ).format(orig_module)
    bodies = "\n\n\n".join(inspect.getsource(fn) for fn in functions)
    content = (
        license_header
        + "\n\n"
        + note
        + "\n\n"
        + import_block.strip()
        + "\n\n\n"
        + bodies.rstrip()
        + "\n"
    )
    with open(target_file, "w") as f:
        f.write(content)


def write_pynq_driver_support(pynq_driver_dir: str) -> None:
    """What every generated PYNQ driver runs with, into ``pynq_driver_dir``: the base
    driver (driver_base.py), validate.py, and the parts of qonnx and finn it imports,
    trimmed to what it uses."""
    # create the base FINN driver -- same for all accels
    shutil.copy(data_path("pynq_driver", "driver_base.py"), pynq_driver_dir + "/driver_base.py")
    # driver depends on qonnx and finn packages
    # extract individual source files and copy to driver folder
    qonnx_target_path = pynq_driver_dir + "/qonnx"
    finn_target_path = pynq_driver_dir + "/finn"
    os.makedirs(qonnx_target_path + "/core", exist_ok=True)
    os.makedirs(qonnx_target_path + "/util", exist_ok=True)
    os.makedirs(finn_target_path + "/util", exist_ok=True)
    qonnx_path = qonnx.__path__[0]
    finn_util_path = finn.util.__path__[0]
    files_to_copy = [
        (qonnx_path + "/core/datatype.py", qonnx_target_path + "/core/datatype.py"),
        (qonnx_path + "/core/__init__.py", qonnx_target_path + "/core/__init__.py"),
        (qonnx_path + "/util/__init__.py", qonnx_target_path + "/util/__init__.py"),
        (finn_util_path + "/__init__.py", finn_target_path + "/util/__init__.py"),
    ]
    for src_file, target_file in files_to_copy:
        shutil.copy(src_file, target_file)

    # qonnx.util.basic and finn.util.data_packing are not copied verbatim:
    # the driver only needs a handful of pure-numpy helpers from each, while
    # the full files import onnx (qonnx.util.basic) and bitstring
    # (finn.util.data_packing). Emitting a trimmed-down module keeps those
    # heavy dependencies off the deployment board.
    _generate_minimal_module(
        qonnx_target_path + "/util/basic.py",
        qonnx.util.basic,
        [
            qonnx.util.basic.roundup_to_integer_multiple,
            qonnx.util.basic.gen_finn_dt_tensor,
        ],
        "import numpy as np\n"
        "from typing import cast\n\n"
        "from qonnx.core.datatype import BaseDataType, DataType, FixedPointType",
    )
    dp = finn.util.data_packing
    _generate_minimal_module(
        finn_target_path + "/util/data_packing.py",
        dp,
        [
            dp.finnpy_to_packed_bytearray,
            dp._pack_whole_byte_container,
            dp._pack_bit_double_reverse,
            dp._pack_general,
            dp.finnpy_to_int_array,
            dp.int_array_to_packed_bytearray,
            dp.packed_bytearray_to_finnpy,
            dp.prepare_values,
            dp.unsiged_array_to_signed,
            dp.packed_bytearray_to_finnpy_fast,
            dp.data_prepared_to_finnpy_bipolar,
            dp.data_prepared_to_finnpy_ternary,
            dp.data_prepared_to_finnpy_fixed,
            dp.data_prepared_to_finnpy_int,
            dp.packed_bytearray_to_finnpy_float,
        ],
        "import numpy as np\n\n"
        "from qonnx.core.datatype import DataType\n"
        "from qonnx.util.basic import roundup_to_integer_multiple",
    )
    # add validate.py to run full top-1 test (only for suitable networks)
    shutil.copy(data_path("pynq_driver", "validate.py"), pynq_driver_dir + "/validate.py")


def pynq_driver_text(platform: str, shapes: dict[str, Any], fclk_mhz: float, bitfile: str) -> str:
    """The generated PYNQ driver (driver.py) for ``platform`` (the host runtime): its I/O
    (``shapes``, ``driver_shapes``' form), the clock in MHz the overlay sets PL0 to
    (``fclk_mhz``), and the bitfile it runs unless told another, ``bitfile``, relative
    to the driver's own directory. It has no external weights and no MLO configuration."""
    default_bitfile = (
        f"os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), {bitfile!r}))"
    )
    driver = templates.pynq_driver_template
    driver = driver.replace("$PLATFORM$", platform)
    driver = driver.replace("$INPUT_FINN_DATATYPE$", str(shapes["idt"]).replace('"', ""))
    driver = driver.replace("$INPUT_SHAPE_NORMAL$", str(shapes["ishape_normal"]))
    driver = driver.replace("$INPUT_SHAPE_FOLDED$", str(shapes["ishape_folded"]))
    driver = driver.replace("$INPUT_SHAPE_PACKED$", str(shapes["ishape_packed"]))
    driver = driver.replace("$OUTPUT_FINN_DATATYPE$", str(shapes["odt"]).replace('"', ""))
    driver = driver.replace("$OUTPUT_SHAPE_NORMAL$", str(shapes["oshape_normal"]))
    driver = driver.replace("$OUTPUT_SHAPE_FOLDED$", str(shapes["oshape_folded"]))
    driver = driver.replace("$OUTPUT_SHAPE_PACKED$", str(shapes["oshape_packed"]))
    driver = driver.replace("$INPUT_DMA_NAME$", "%s" % str(shapes["idma_names"]))
    driver = driver.replace("$OUTPUT_DMA_NAME$", "%s" % str(shapes["odma_names"]))
    driver = driver.replace("$NUM_INPUTS$", str(len(shapes["idma_names"])))
    driver = driver.replace("$NUM_OUTPUTS$", str(len(shapes["odma_names"])))
    driver = driver.replace("$EXT_WEIGHT_NUM$", "0")
    driver = driver.replace("$EXT_WEIGHT_INPUT_SHAPES$", str({}))
    driver = driver.replace("$MLO_WEIGHT_CONFIG$", json.dumps({}, indent=4))
    driver = driver.replace("$FCLK_MHZ$", repr(float(fclk_mhz)))
    driver = driver.replace("$BITFILE$", default_bitfile)
    return driver


def driver_shapes(export: Integration) -> dict[str, Any]:
    """The driver's I/O from the export's ends: each input end's (``i``) and output
    end's (``o``) element as its ``DataType``, its tensor's shape (``normal``), its
    frame as the stream carries it, ``(1, beats, lanes)`` (``folded``), and that packed
    into bytes (``packed``), and its instance."""
    shapes: dict[str, Any] = {}
    for side, direction, dma in (("i", "in", "idma_names"), ("o", "out", "odma_names")):
        ends = [end for end in export.ends if end.contract.direction == direction]
        folded = [(1, end.contract.beats, end.contract.lanes) for end in ends]
        shapes[f"{side}dt"] = [f"DataType['{end.contract.element.dtype.name}']" for end in ends]
        shapes[f"{side}shape_normal"] = [tuple(end.shape) for end in ends]
        shapes[f"{side}shape_folded"] = folded
        shapes[f"{side}shape_packed"] = [
            packed_shape(end.contract.element.dtype, each) for end, each in zip(ends, folded)
        ]
        shapes[dma] = [end.instance for end in ends]
    return shapes


def write_driver(export: Integration, directory: str, fclk_mhz: float, bitfile: str) -> None:
    """The PYNQ driver of the integration ``export`` into ``directory``: its I/O the
    export's (``driver_shapes``), PL0 set to ``fclk_mhz``, the clock the bitfile
    delivers; driver.py and validate.py run ``bitfile``, relative to ``directory``,
    unless told another."""
    os.makedirs(directory, exist_ok=True)
    write_pynq_driver_support(directory)
    host_runtime = export.host_runtime
    if host_runtime is None:
        raise ValueError(f"the {export.shell!r} shell states no host runtime to drive")
    with open(os.path.join(directory, "driver.py"), "w") as f:
        f.write(pynq_driver_text(host_runtime, driver_shapes(export), fclk_mhz, bitfile))


def driver_description(
    export: Integration, fclk_mhz: float, bitfile: str, before: list[str], after: list[str]
) -> dict[str, Any]:
    """What the driver of ``export`` takes and returns (SS9): the bitfile it runs, by
    its path in the build's output directory and in the deployment package, both of
    which hold bitfile/ beside driver/; the partition's inputs and outputs, each by its
    tensor's name, element and shape, and the end that moves it; the clock it sets; and
    the host's nodes of the parent graph that run ``before`` and ``after`` it."""

    def tensors(direction: str) -> list[dict[str, Any]]:
        return [
            {
                "name": end.tensor,
                "element": end.contract.element.dtype.name,
                "shape": list(end.shape),
                "dma": end.instance,
            }
            for end in export.ends
            if end.contract.direction == direction
        ]

    return {
        "host_runtime": export.host_runtime,
        "bitfile": bitfile,
        "fclk_mhz": fclk_mhz,
        "takes": tensors("in"),
        "returns": tensors("out"),
        "host": {"before": list(before), "after": list(after)},
    }


__all__ = [
    "driver_description",
    "driver_shapes",
    "packed_shape",
    "pynq_driver_text",
    "write_driver",
    "write_pynq_driver_support",
]
