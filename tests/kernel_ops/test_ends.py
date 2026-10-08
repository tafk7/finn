# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The ends a shell offers on its boundary channels (``finn.kernels.ends``).

An offer places an ``IODMA_hls`` end on each free side of the shell root that meets
the host: forced, so nothing is persisted; binding no RTL, so the module is the
``ip`` shell's; its cycles a frame the channel's, so an exploration's bottleneck can
name it, and FIFO sizing reads both boundaries through it. On the Chain
(``kernels.chain`` as KernelOps), and on TFC_W2A2 for Ultra96, whose memory port is
128 bits wide.
"""

from __future__ import annotations

from typing import Any

import pytest
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from finn.custom_op.kernels.base import KernelOpError
from finn.custom_op.kernels.shell import persist, shell_root
from finn.dataflow.tensor import ScalarEncoding
from finn.kernels.configure import chosen
from finn.kernels.ends import IODMA_HLS, MEMORY_LATENCY, EndContract, EndOffer, iodma_hls
from finn.kernels.explore import Explorer, SizeFifos, TargetThroughput
from finn.kernels.fifo_sizing import NotModelled, Pattern, ends
from finn.transformation.kernels import (
    InferKernelTensors,
    ToKernelOps,
    explore_kernel_choices,
    kernel_choices_config,
)
from kernel_ops.models import configure_partition, kernel_model
from kernel_ops.tfc import ULTRA96, streamlined

# -- an end's arithmetic -------------------------------------------------------------------


def contract(tdata: int, beats: int, memory: int, call: int, frames: int = 1) -> EndContract:
    return EndContract(
        kind=IODMA_HLS,
        direction="in",
        port="s_axis_0",
        tdata=tdata,
        beats=beats,
        lanes=tdata // 8,
        element=ScalarEncoding(DataType["UINT8"]),
        memory_width=memory,
        words=tdata * beats // memory,
        converter=memory != tdata,
        call_cycles=call,
        frames_per_call=frames,
    )


def test_an_end_takes_its_beats_or_its_words_and_its_call_constant_a_frame() -> None:
    # TFC's input at 1e6 fps: 196 beats of 32 bits from 49 words of 128, a converter.
    tfc = contract(32, 196, 128, 4)
    assert tfc.cycles == max(196, 49) + 4 == 200
    # Sixteen frames a call share the constant: the 16-frame figure beside it.
    assert tfc.cycles_at(16) == 196 + 1
    # One beat a cycle, the memory the wider.
    assert tfc.times == tuple(range(196))
    # A narrower memory: a beat once the word completing it arrived, a word a cycle.
    narrow = contract(112, 56, 64, 4)
    assert narrow.cycles == 98 + 4
    assert narrow.times[:4] == (1, 3, 5, 6) and narrow.times[-1] == 97


def test_the_iodma_offer_states_its_measured_call_constants() -> None:
    offer = iodma_hls(128)
    assert (offer.kind, offer.width_cap, offer.frames_per_call) == (IODMA_HLS, 128, 1)
    assert (offer.call_converted, offer.call_direct) == (4, 13)
    with pytest.raises(ValueError, match="no whole bytes"):
        iodma_hls(12)
    with pytest.raises(ValueError, match="frames a call is none"):
        iodma_hls(64, frames_per_call=0)


# -- on the Chain --------------------------------------------------------------------------


def chain_root(*offers: EndOffer, model: ModelWrapper | None = None) -> Any:
    model = kernel_model() if model is None else model
    return shell_root(model, model.graph.node, name="chain", offers=offers)


def test_an_offer_places_a_forced_end_on_each_free_side() -> None:
    root = chain_root(iodma_hls(64))
    assert root.ends == ("x", "y")
    point = root.point
    # Forced: no key of an end is committed, so none is persisted.
    assert not [key for key in chosen(point) if ".end" in key]
    x, y = point.x.end_contract, point.y.end_contract
    # x: two INT3 lanes padded to a byte, 6 beats; 48 bits a frame share 16 with 64.
    assert (x.direction, x.port, x.tdata, x.beats, x.lanes) == ("in", "s_axis_0", 8, 6, 2)
    assert (x.element.dtype.name, x.memory_width, x.words, x.converter) == ("INT3", 16, 3, True)
    assert (y.direction, y.port, y.memory_width, y.words) == ("out", "m_axis_0", 32, 3)
    # The end's cycles are the channel's, beside its stages': y has none of its own.
    assert point.y.cycles == y.cycles == 6 + 4
    assert point.x.cycles == max(12, x.cycles)


def test_without_offers_the_shell_is_the_ip_shell() -> None:
    root = chain_root()
    assert root.ends == ()
    assert not root.point.x.ended and root.point.y.cycles == 0


def test_an_end_binds_no_rtl() -> None:
    model = kernel_model()
    configure_partition(model)
    ip = chain_root(model=model).point.module
    ended = chain_root(iodma_hls(64), model=model).point.module
    assert ended.fragment == ip.fragment
    assert ended.abi == ip.abi


def test_an_ended_point_persists_and_replays_as_itself() -> None:
    model = kernel_model()
    root = chain_root(iodma_hls(64), model=model)
    held = persist(model, root, root.point)
    assert not [key for values in held.values() for key in values if "end" in key.split(".")]
    again = chain_root(iodma_hls(64), model=model)
    assert chosen(again.point) == chosen(root.point) and not again.dropped


def test_a_boundary_handed_on_to_a_kernel_op_gets_no_end() -> None:
    model = kernel_model()
    first = model.graph.node[0]
    root = shell_root(model, [first], name="first", offers=(iodma_hls(64),))
    # x meets the host; hidden is handed on to activate, a KernelOp outside.
    assert root.ends == ("x",)
    activate = model.graph.node[1:2]
    assert shell_root(model, activate, name="activate", offers=(iodma_hls(64),)).ends == ()


def test_an_offer_of_an_unknown_or_repeated_kind_is_refused() -> None:
    other = EndOffer(kind="dma", width_cap=64, frames_per_call=1, call_converted=0, call_direct=0)
    with pytest.raises(KernelOpError, match="no end of kind dma"):
        chain_root(other)
    with pytest.raises(KernelOpError, match="each kind of end once"):
        chain_root(iodma_hls(64), iodma_hls(128))


def test_sizing_reads_a_free_side_its_end_paces() -> None:
    ended = chain_root(iodma_hls(64)).point
    x = ended.x.end_contract
    supply, _ = ends(ended.x)
    assert supply == Pattern(x.times, x.cycles)
    y = ended.y.end_contract
    _, accept = ends(ended.y)
    assert accept == Pattern(y.times, y.cycles)
    with pytest.raises(NotModelled, match="a boundary: not modelled"):
        ends(chain_root().point.x)


# -- on TFC_W2A2 ---------------------------------------------------------------------------

ULTRA96_END = iodma_hls(128)
"""Ultra96's memory port: 128 bits."""


@pytest.fixture(scope="module")
def tfc(tmp_path_factory: pytest.TempPathFactory) -> ModelWrapper:
    """TFC_W2A2 as KernelOps for Ultra96 at 5 ns (half a minute)."""
    source = streamlined(tmp_path_factory.mktemp("tfc"))
    return source.transform(ToKernelOps(ULTRA96)).transform(InferKernelTensors())


def explored(
    tfc: ModelWrapper, fps: float, *offers: EndOffer, sizing: bool = False
) -> tuple[ModelWrapper, dict[str, Any]]:
    model = ModelWrapper(tfc.model.__deepcopy__())
    strategies: list[Explorer] = [TargetThroughput(fps), *([SizeFifos()] if sizing else [])]
    return model, dict(explore_kernel_choices(model, strategies, offers=offers).report)


def built(model: ModelWrapper, report: dict[str, Any]) -> dict[str, dict[str, object]]:
    """Every value the partition is built with, persisted or completed, by node."""
    found = {node: dict(held) for node, held in kernel_choices_config(model).items()}
    for node, held in report["completed"].items():
        found.setdefault(node, {}).update({key: entry["value"] for key, entry in held.items()})
    return found


@pytest.mark.slow
def test_tfc_with_ultra96_s_ends_at_1e6_fps_names_its_input_end(tfc: ModelWrapper) -> None:
    ip_model, ip = explored(tfc, 1e6, sizing=True)
    model, report = explored(tfc, 1e6, ULTRA96_END, sizing=True)
    # The ends change no choice: persisted and completed alike, 45 values on 8 nodes.
    assert kernel_choices_config(model) == kernel_choices_config(ip_model)
    values = built(model, report)
    assert values == built(ip_model, ip)
    assert (len(values), sum(map(len, values.values()))) == (8, 45)
    persisted = kernel_choices_config(model).values()
    assert not [key for held in persisted for key in held if "end" in key.split(".")]
    # The input end at max(196, 49) + 4 and the output end at max(10, 5) + 4, the
    # partition at its 196; the budget of 200 is met, nothing relaxed.
    rows = report["ends"]
    assert {name: (row["beats"], row["words"], row["cycles"]) for name, row in rows.items()} == {
        "Reshape_0_out0": (196, 49, 200),
        "MatMul_3_out0": (10, 5, 14),
    }
    assert rows["Reshape_0_out0"] == {
        "kind": "iodma_hls",
        "direction": "in",
        "port": "s_axis_0",
        "tdata": 32,
        "beats": 196,
        "lanes": 4,
        "element": "UINT8",
        "memory_width": 128,
        "words": 49,
        "converter": True,
        "call_cycles": 4,
        "frames_per_call": 1,
        "cycles": 200,
        "cycles_16_frames_a_call": 197,
    }
    assert (rows["MatMul_3_out0"]["element"], rows["MatMul_3_out0"]["memory_width"]) == (
        "INT8",
        16,
    )
    assert report["memory_latency"] == MEMORY_LATENCY
    assert report["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": 200}
    target, sizing = report["strategies"]
    assert (target["cycles"], target["relaxed_to"]) == (200, None)
    # SizeFifos reads both boundaries through their ends, at the period they set.
    whys = {name: row["why"] for name, row in sizing["channels"].items()}
    assert sizing["period"] == 200
    assert whys["Reshape_0_out0"] == whys["MatMul_3_out0"] == "direct absorbs it"
    # The ip shell: no end, no row.
    assert "ends" not in ip and "memory_latency" not in ip
    assert ip["bottleneck"]["cycles"] == 196


@pytest.mark.slow
def test_at_3e6_fps_a_64_bit_memory_port_binds_and_128_bits_do_not(tfc: ModelWrapper) -> None:
    """3e6 frames a second, a budget of 66 cycles: on a 64-bit port TFC's 6 272 input
    bits are 98 words, so the input end binds whatever the folding, and the strategy
    relaxes to what the root reaches; on 128 bits, 49 words, MatMul_3 binds at 64."""
    _, narrow = explored(tfc, 3e6, iodma_hls(64))
    end = narrow["ends"]["Reshape_0_out0"]
    assert (end["memory_width"], end["words"]) == (64, 98)
    assert end["cycles"] >= 98 + 4
    (target,) = narrow["strategies"]
    assert target["cycles"] == 66 and target["relaxed_to"] is not None
    assert narrow["bottleneck"] == {"members": ["Reshape_0_out0"], "cycles": end["cycles"]}
    _, wide = explored(tfc, 3e6, iodma_hls(128))
    end = wide["ends"]["Reshape_0_out0"]
    assert (end["memory_width"], end["words"], end["cycles"]) == (128, 49, 60)
    bottleneck = wide["bottleneck"]
    assert bottleneck["cycles"] == 64 and "partition.MatMul_3" in bottleneck["members"]
    assert "Reshape_0_out0" not in bottleneck["members"]
