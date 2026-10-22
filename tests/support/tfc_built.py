# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""What TFC_W2A2 is built with at 1,000,000 frames a second: SetFolding's folding as the
oracle captured it, every persisted choice of the build before the engine core
(``TFC_BUILT``), and every one stored now (``TFC_STORED``)."""

from __future__ import annotations

from oracle import capture


def set_folding() -> dict[str, dict[str, int]]:
    """FINN's SetFolding on TFC_W2A2 at 1,000,000 frames a second and 5 ns (200 cycles a
    frame), as the finn-dev oracle captured it (``tfc_set_folding``): each layer's folding,
    by the build's node (the oracle's i-th Thresholding is MultiThreshold_i, its i-th
    MVAU MatMul_i; its label select is on the host here)."""
    captured = capture("tfc_set_folding")
    assert captured["target_cycles"] == 200
    folding: dict[str, dict[str, int]] = {}
    for layer in captured["layers"]:
        family = layer["op_type"].partition("_")[0]
        if family == "Thresholding":
            index = sum(name.startswith("MultiThreshold") for name in folding)
            folding[f"MultiThreshold_{index}"] = {"pe": layer["PE"]}
        elif family == "MVAU":
            index = sum(name.startswith("MatMul") for name in folding)
            folding[f"MatMul_{index}"] = {
                "pe": layer["PE"],
                "simd": layer["SIMD"],
            }
    return folding


#: Each layer's folding, which the target throughput strategy reaches by cost.
SET_FOLDING = set_folding()


def tfc_built() -> dict[str, dict[str, object]]:
    """Every value TFC_W2A2 is built with at 1,000,000 frames a second, every one
    persisted, 57 by graph name (on 8 nodes and 13 tensors): the Zynq landing's baseline
    build's (Ultra96, 5 ns, [target_throughput, size_fifos]), whose choices were all
    persisted then, and each MatMul's pumping, at its baseline, off: the doubled clock is the
    shell's, which the kernels no longer read off the platform (SZ11 (e)), so the choice
    is open on pynq too. Each layer's kernel on its node; each channel on its tensor: a
    MultiThreshold's input, a MatMul's activations and weights, the graph's output. Every
    memory takes its baseline, an explicit style (SZ18): the first three layers' weights
    in block RAM (8 192 bits and more), the last layer's and every input buffer in LUTRAM,
    and each threshold table's stages in LUTRAM, no block stage."""
    built: dict[str, dict[str, object]] = {}
    for node, folding in SET_FOLDING.items():
        layer = int(node.rsplit("_", 1)[1])
        if node.startswith("MatMul"):
            built[node] = {
                **folding,
                "compute.packed.compute_pumping": False,
                "compute.packed.reducer": "tree",
            }
            built[f"MultiThreshold_{layer}_out0"] = {
                "adapter.input_gen.input_gen.ram_style": "distributed",
                "transport": "direct",
            }
            built[f"MatMul_{layer}_param0"] = {
                "source.memstream.pumped_memory": False,
                "source.memstream.ram_style": "distributed" if node == "MatMul_3" else "block",
                "transport": "direct",
            }
        else:
            built[node] = {
                **folding,
                "block_stages": 0,
                "deep_pipeline": False,
                "ram_style": "distributed",
            }
            built[f"MatMul_{layer - 1}_out0" if layer else "Reshape_0_out0"] = {
                "transport": "direct"
            }
    built["MatMul_3_out0"] = {"transport": "direct"}
    return built


TFC_BUILT = tfc_built()


#: What conversion, the cut and the fold store on TFC beside ``TFC_BUILT``, 27 values
#: forcing left unstored: each MatMul's core (conversion's first valid), each
#: Thresholding's AXI-Lite and UltraRAM stages (conversion's), each weight's source and
#: each activation's adapter (the interim channel rule's one possible), the mover of each
#: boundary channel the pynq row offers one, and the output adapter the fold at 1e6 frames
#: a second brings into play on each activation and on the first MatMul's output (the
#: interim channel rule's).
STORED: dict[str, dict[str, object]] = {
    **{f"MatMul_{index}": {"compute": "packed"} for index in range(4)},
    **{f"MultiThreshold_{index}": {"use_axilite": 0, "ultra_stages": 0} for index in range(4)},
    **{f"MatMul_{index}_param0": {"source": "memstream"} for index in range(4)},
    **{
        f"MultiThreshold_{index}_out0": {"adapter": "input_gen", "output_adapter": "vpc"}
        for index in range(4)
    },
    "MatMul_0_out0": {"output_adapter": "vpc"},
    "Reshape_0_out0": {"mover": "iodma_hls"},
    "MatMul_3_out0": {"mover": "iodma_hls"},
}

#: Every value TFC_W2A2 is built with and stores at 1,000,000 frames a second, 84 by
#: graph name: ``TFC_BUILT`` and ``STORED``.
TFC_STORED = {owner: {**TFC_BUILT[owner], **STORED.get(owner, {})} for owner in TFC_BUILT}
