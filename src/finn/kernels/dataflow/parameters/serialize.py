############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Re-converges on base FINN's make_weight_file factoring
# (matrixvectoractivation.py: get_hw_compatible_weight_tensor once, then a
# mode-switched artifact form). The kernels rewrite had SPLIT that one method across
# two emit files (emit_hls owns embedded params.h, emit_memstream owns decoupled
# memblock.dat), each re-deriving the identical Part-1 reshape. This module puts it
# back where base FINN had it: ONE serializer keyed by a LayoutConstraint, called by
# both consumers.
############################################################################

"""The unified parameter serializer — ``layout(param_tensor, constraint)``.

A parameter serializer turns a numpy parameter tensor (weights, thresholds) into the
on-chip memory byte layout the hardware reads. It has two parts:

* **Part 1 — geometric reshape** (the ``ByteTraversalOrder``): interleave the tensor into
  the ``(1, PE, DEPTH, INNER)`` hardware fold. This is the piece that was DUPLICATED
  byte-for-byte between the HLS (``params.h``) and decoupled (``memblock.dat``) emits.
* **Part 2 — artifact form** (the ``SerializationForm``): wrap the reshaped tensor as a
  C++ initializer body (``cpp-header``, baked into the compute core) or as hex ``.dat``
  text (``dat-hex``, loaded by a memstream cell). This LEGITIMATELY differs by delivery.

The serializer is PARAMETER-specific, not compute- or delivery-specific — it is the ONE
piece common to fused (embedded) and separable (decoupled) storage, which is why it lives
in ``parameters/`` and both emits import it. The ``LayoutConstraint`` is compute-published
(the byte-traversal order is a property of the compute core + delivery path, not a memory
assumption), keeping ``layout`` indifferent to iteration mechanism.

Serializer-vs-transport orthogonality: ``layout`` ALWAYS runs (embedded bakes the bytes
into ``params.h``, decoupled writes them to ``memblock.dat``). WHICH port the bytes reach
the core through — an AXI-Stream ``PARAM_SOURCE`` (decoupled), no port (embedded/baked),
or an AXI-Lite reload sideband — is the separate delivery decision, not the serializer's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qonnx.core.datatype import DataType
from qonnx.util.basic import (
    interleave_matrix_outer_dim_from_partitions,
    roundup_to_integer_multiple,
)

from finn.util.data_packing import numpy_to_hls_code, pack_innermost_dim_as_hex_string

# --- SerializationForm — the two artifact forms of a single serialized blob ----
# (base FINN also has decoupled_npy (cppsim) + decoupled_runtime (AXI-lite reload); the
# form field accommodates them as further members without reshaping this interface.)
CPP_HEADER = "cpp-header"  # a C++ initializer body baked into the compute core
DAT_HEX = "dat-hex"  # newline-joined hex words loaded by a memstream cell

# --- ByteTraversalOrder — the interleave/flip vocabulary (compute-published) ---
# WEIGHT: (1,PE,WMEM,SIMD) — transpose, bipolar->binary, PE-interleave, SIMD-flip.
# THRESHOLD: (1,PE,TMEM,nSteps) — tile a per-tensor (1,nSteps) matrix up to MH channels,
#   PE-interleave, reshape. The HLS threshold ROM (a SEPARABLE, static-schedule memory —
#   read in full each output beat, indexed by the output-channel LOOP) routes through here.
#   (The RTL binary-search threshold storage is DATA-DEPENDENT-addressed — the ROM address
#   is the runtime comparison outcome, not a loop counter — so it is FUSED to the compute
#   core and stays out of layout(); see ops/thresholding/emit_rtl.py.)
WEIGHT = "weight"
THRESHOLD = "threshold"


@dataclass(frozen=True)
class LayoutConstraint:
    """The compute-published serialization constraint — the ``(byte-order × form ×
    delivery-extras)`` a serializer needs, indifferent to iteration mechanism.

    Fields:
        traversal: the ByteTraversalOrder (which Part-1 reshape). ``WEIGHT`` this pass.
        form: the SerializationForm — ``CPP_HEADER`` or ``DAT_HEX``.
        pe / inner / depth: the hardware fold extents. ``inner`` is the innermost fold
            (SIMD for weights); ``depth`` is the memory depth (WMEM for weights).
        orig_dtype: the tensor's own dtype (drives the bipolar->binary Part-1 step).
        export_dtype: the dtype the values are packed/emitted AS (BINARY for BIPOLAR).
        decoupled_pe_flip: the decoupled-only extra (transpose + PE-flip before packing).
        pumped_split: split each hex word into two half-width entries (pumpedMemory).
        var_name / hls_flags: the ``numpy_to_hls_code`` variable name + (do_declare,
            fill) flags for the CPP_HEADER body.
    """

    traversal: str
    form: str
    pe: int
    inner: int
    depth: int
    orig_dtype: DataType
    export_dtype: DataType
    decoupled_pe_flip: bool = False
    pumped_split: bool = False
    var_name: str = "weights"
    hls_flags: tuple[bool, bool] = (True, True)
    tile_rows: int = 0  # THRESHOLD: tile a per-tensor (1,inner) matrix up to this many rows (MH)


@dataclass(frozen=True)
class SerializedParam:
    """A parameter serialized to a single blob — the ``MemImage``. ``text`` is the
    artifact content (a C++ initializer body for CPP_HEADER, full ``.dat`` text for
    DAT_HEX). Stays a single blob: the per-segment fanout the validation once proposed
    served only the excluded (data-dependent) RTL-threshold case."""

    text: str
    form: str


def weight_constraint(
    pe: int,
    simd: int,
    wmem: int,
    wdt: DataType,
    export_wdt: DataType,
    *,
    form: str,
    decoupled_pe_flip: bool = False,
    pumped_split: bool = False,
) -> LayoutConstraint:
    """Build the WEIGHT-traversal constraint — the single place the weight byte-order is
    declared, shared by the embedded (params.h) and decoupled (memblock.dat) callers so
    they cannot diverge."""
    return LayoutConstraint(
        traversal=WEIGHT,
        form=form,
        pe=pe,
        inner=simd,
        depth=wmem,
        orig_dtype=wdt,
        export_dtype=export_wdt,
        decoupled_pe_flip=decoupled_pe_flip,
        pumped_split=pumped_split,
        var_name="weights",
        hls_flags=(True, True),
    )


def threshold_constraint(
    pe: int,
    tmem: int,
    n_steps: int,
    mh: int,
    tdt: DataType,
    *,
    form: str = CPP_HEADER,
) -> LayoutConstraint:
    """Build the THRESHOLD-traversal constraint for the separable HLS threshold ROM. The
    innermost fold is ``n_steps`` (thresholds per channel), ``depth`` is TMEM, and
    ``tile_rows`` is MH (a per-tensor (1,n_steps) matrix is broadcast up to MH channels
    before interleaving). Emitted as a C++ ``ThresholdsActivation`` initializer body — the
    thresholds are NOT bipolar-adjusted (``do_declare`` off), matching the old
    ``numpy_to_hls_code(tensor, tdt, "thresholds", False, True)``."""
    return LayoutConstraint(
        traversal=THRESHOLD,
        form=form,
        pe=pe,
        inner=n_steps,
        depth=tmem,
        orig_dtype=tdt,
        export_dtype=tdt,
        var_name="thresholds",
        hls_flags=(False, True),
        tile_rows=mh,
    )


def layout(param_tensor, constraint: LayoutConstraint) -> SerializedParam:
    """Serialize ``param_tensor`` per ``constraint`` — Part-1 reshape once, then the
    form-dispatched Part-2 artifact. Pure over its inputs (no point/context/graph)."""
    tensor = _reshape(np.asarray(param_tensor), constraint)
    if constraint.form == CPP_HEADER:
        text = numpy_to_hls_code(
            tensor, constraint.export_dtype, constraint.var_name, *constraint.hls_flags
        )
    elif constraint.form == DAT_HEX:
        text = _to_dat_hex(tensor, constraint)
    else:
        raise ValueError(f"unknown SerializationForm {constraint.form!r}")
    return SerializedParam(text=text, form=constraint.form)


# ------------------------------------------------------------- Part 1: reshape


def _reshape(param_tensor, c: LayoutConstraint):
    if c.traversal == WEIGHT:
        return _weight_reshape(param_tensor, c)
    if c.traversal == THRESHOLD:
        return _threshold_reshape(param_tensor, c)
    raise ValueError(f"unknown ByteTraversalOrder {c.traversal!r}")


def _weight_reshape(weights, c: LayoutConstraint):
    """get_hw_compatible_weight_tensor (matrixvectoractivation.py:602): transpose to
    hlslib layout, bipolar->binary, interleave rows across PEs, reshape
    ``(1, PE, WMEM, SIMD)``, reverse SIMD. The formerly-duplicated Part-1 core."""
    ret = weights.T
    if c.orig_dtype == DataType["BIPOLAR"]:
        ret = (ret + 1) / 2
    ret = interleave_matrix_outer_dim_from_partitions(ret, c.pe)
    ret = ret.reshape(1, c.pe, c.depth, c.inner)
    ret = np.flip(ret, axis=-1)
    return ret


def _threshold_reshape(thresholds, c: LayoutConstraint):
    """get_hw_compatible_threshold_tensor (matrixvectoractivation.py:584): tile a per-tensor
    (1, n_steps) threshold matrix up to MH channels, interleave rows across PEs, reshape
    ``(1, PE, TMEM, n_steps)``. No SIMD-flip (thresholds are not innermost-reversed)."""
    ret = thresholds
    if ret.shape[0] == 1:
        ret = np.tile(ret, (c.tile_rows, 1))
    ret = interleave_matrix_outer_dim_from_partitions(ret, c.pe)
    return ret.reshape(1, c.pe, c.depth, c.inner)


# ------------------------------------------------------------- Part 2: dat-hex


def _to_dat_hex(tensor, c: LayoutConstraint) -> str:
    """make_weight_file "decoupled_verilog_dat" (matrixvectoractivation.py:715-811): the
    decoupled PE-flip, then hex-pack each PE*INNER group at 4-bit-padded width; pumpedMemory
    splits each word into two half-width entries (low half first)."""
    pe, inner = c.pe, c.inner
    if c.decoupled_pe_flip:
        # transpose (1,PE,DEPTH,INNER) -> (1,DEPTH,PE,INNER), then PE-flip.
        unflipped = np.transpose(tensor, (0, 2, 1, 3))
        tensor = np.flip(unflipped, axis=-2)
    tensor = tensor.reshape(1, -1, pe * inner).copy()

    width = pe * inner * c.export_dtype.bitwidth()
    width_padded = roundup_to_integer_multiple(width, 4)
    packed = pack_innermost_dim_as_hex_string(tensor, c.export_dtype, width_padded, prefix="")
    stream = packed.flatten().copy()

    if c.pumped_split:
        split = []
        for w in stream:
            split.append(w[len(w) // 2:])
            split.append(w[: len(w) // 2])
        stream = split

    return "".join(str(v) + "\n" for v in stream)
