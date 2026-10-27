# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The thresholding kernel: its result type, its table, its configuration and its refusals."""

from __future__ import annotations

import pytest
from qonnx.core.datatype import DataType
from support.helpers import (
    FULL_DSP48E2,
    THRESHOLDS,
    by_hand,
    controlled,
    integer_tensor,
    set_indices,
    threshold_base,
)
from support.refusals import refused

from finn.core.space import (
    Available,
    DefinitionError,
    Rejected,
    Unresolved,
    design_space,
    inspection,
)
from finn.dataflow.datatypes import resolve_qonnx_datatype_name
from finn.dataflow.tensor import ScalarEncoding, Tensor
from finn.kernels.artifacts.abi import Bus, Endpoint
from finn.kernels.composer import Facts, Port, channel
from finn.kernels.library import library_module
from finn.kernels.port import AxiStreamPort
from finn.kernels.sides import Side
from finn.kernels.thresholding import ThresholdingAxiKernel, c
from finn.kernels.values.domains import Integer
from finn.kernels.values.semantics import IntegerTensor, IntegerTensorValue, integer_shape

AUTO_MEMORY: dict[str, object] = {"ram_style": "auto", "ultra_stages": 0}


def threshold(
    *,
    table: IntegerTensor = THRESHOLDS,
    pe: int | None = 1,
    bias: int = -1,
    input_dtype: str = "INT8",
    threshold_dtype: str = "INT5",
    memory: dict[str, object] | None = None,
    axilite: bool = False,
    deep: bool = False,
) -> ThresholdingAxiKernel:
    """On one row of channels, the table's rows (with one row shared, PE of them); with
    several sets, given their set indices, one an input beat. PE is committed as a choice;
    ``None`` leaves it open."""
    shape = integer_shape(table)
    rows = shape[1] if shape is not None and len(shape) == 3 else 1
    sets = shape[0] if shape is not None and len(shape) == 3 else 1
    channels = rows if rows > 1 or pe is None else pe
    set_tensor = None if sets < 2 else set_indices(sets, channels // (pe or 1))
    base = threshold_base(
        table=table,
        bias=bias,
        input_dtype=input_dtype,
        threshold_dtype=threshold_dtype,
        channels=channels,
        set_tensor=set_tensor,
    )
    factors: dict[str, object] = {} if pe is None else {"pe": pe}
    if pe is not None:
        # The memories, like PE, range over the table (its thresholds' stages).
        factors |= AUTO_MEMORY if memory is None else memory
    if axilite:
        # Runtime-writable thresholds present their bus through a control node.
        element = ScalarEncoding(resolve_qonnx_datatype_name(input_dtype))
        facts = dict(
            input_tensor=Tensor((1, channels), element),
            threshold_dtype=resolve_qonnx_datatype_name(threshold_dtype),
            thresholds=integer_tensor(table),
            bias=bias,
            platform=FULL_DSP48E2,
        )
        if set_tensor is not None:
            facts["set_tensor"] = set_tensor
        return controlled(
            ThresholdingAxiKernel, facts, use_axilite=True, deep_pipeline=deep, **factors
        )
    report = base.try_with_choices(use_axilite=axilite, deep_pipeline=deep, **factors)
    assert report.accepted
    return report.instance


def test_threshold_states_its_result_type_initial_table_and_configuration() -> None:
    point = threshold()
    requirements = library_module(point)
    assert point.result_dtype.name == "INT3"
    # The table is THRESHOLDS_FILE, named by its contents: per row fold (C / PE = 2),
    # the row's thresholds padded to a power of two, WT = 5 bits each.
    initial = point.thresholds_file
    assert initial.data == b"1e\n00\n03\n00\n1f\n01\n04\n00\n"
    assert initial.path.startswith("thresholds_") and initial.path.endswith(".dat")
    assert dict(requirements.parameters)["THRESHOLDS_FILE"] == f'"{initial.path}"'
    assert dict(requirements.parameters)["THRESHOLDS"] == "'{default: '0}"
    assert initial in requirements.data
    assert threshold(bias=0).result_dtype.name == "UINT2"
    assert threshold(bias=-4).result_dtype.name == "INT3"
    narrow_negative = threshold(bias=-5)
    assert narrow_negative.result_dtype.name == "INT33"
    assert isinstance(narrow_negative.inspect(ThresholdingAxiKernel.netlist).result, Rejected)
    assert inspection.validity(narrow_negative).status == "refused"
    enabled = library_module(threshold(axilite=True, deep=True))
    assert dict(enabled.parameters)["USE_AXILITE"] == 1
    assert dict(enabled.parameters)["DEEP_PIPELINE"] == 1
    assert dict(library_module(threshold(pe=2)).parameters)["PE"] == 2
    config = next(
        port for port in requirements.abi.pins if isinstance(port, Bus) and port.name == "s_axilite"
    )
    assert {
        signal.width for signal in config.signals if signal.logical in ("awaddr", "araddr")
    } == {5}


def test_threshold_sets_present_a_selector_bus_and_refuse_axilite_addressing() -> None:
    table = (((-2, 0, 3), (-1, 1, 4)), ((-3, 0, 5), (-2, 0, 6)))
    point = threshold(table=table)
    requirements = library_module(point)
    assert dict(requirements.parameters)["SETS"] == 2
    selector = next(
        port
        for port in requirements.abi.pins
        if isinstance(port, Bus) and port.name == "s_axis_set"
    )
    assert next(signal.width for signal in selector.signals if signal.logical == "tdata") == 8
    # Runtime-writable sets: the AXI-Lite wrapper addresses one set, so setting it is
    # refused by that rule.
    with pytest.raises(ValueError, match="threshold-config-sets"):
        threshold(table=table, axilite=True)


def test_a_set_port_that_does_not_match_the_sets_is_refused_both_ways() -> None:
    """Several sets accept an input beat only beside a set beat, so without set indices
    the hardware takes no beat; a single set leaves the set port's ready undriven."""
    two_sets = (((-2, 0, 3), (-1, 1, 4)), ((-3, 0, 5), (-2, 0, 6)))
    pieces = []
    for space in (
        threshold_base(table=two_sets),
        threshold_base(set_tensor=set_indices(1, 2)),
    ):
        validity = inspection.validity(space)
        assert validity.status == "refused"
        pieces += [
            (piece.owner, finding.code, finding.message)
            for piece in validity.pieces
            for finding in piece.result.findings
        ]
    assert pieces == [
        ("selected", "threshold-set-channel", "several threshold sets take a set channel"),
        ("selected", "threshold-set-channel", "a single threshold set takes no set channel"),
    ]
    # Given its set indices, the two sets stand.
    given = threshold_base(table=two_sets, set_tensor=set_indices(2, 2))
    assert inspection.validity(given).status != "refused"


def test_several_sets_without_their_indices_refuse_the_set_port_s_order() -> None:
    # Its rule refuses such a thresholding (``selected``); the set port's order reads the
    # indices, and refuses the same way rather than read what is not given.
    two_sets = (((-2, 0, 3), (-1, 1, 4)), ((-3, 0, 5), (-2, 0, 6)))
    answer = threshold_base(table=two_sets).query(ThresholdingAxiKernel.set_sequence)
    assert isinstance(answer, Rejected)
    assert [(finding.code, finding.owner) for finding in answer.findings] == [
        ("threshold-set-channel", "set_sequence")
    ]


def test_a_single_set_refuses_its_idle_set_port_s_order() -> None:
    # One set takes no set channel: its set port is idle, and the port's order is refused
    # with the kernel's set-channel code, not marked inapplicable.
    answer = threshold_base().query(ThresholdingAxiKernel.set_sequence)
    assert isinstance(answer, Rejected)
    assert [(finding.code, finding.owner, finding.message) for finding in answer.findings] == [
        ("threshold-set-channel", "set_sequence", "a single threshold set takes no set channel")
    ]


def test_threshold_partial_dtype_query_does_not_adopt_implementation_decisions() -> None:
    base = threshold_base()
    assert base.result_dtype.name == "INT3"
    assert isinstance(base.query(ThresholdingAxiKernel.deep_pipeline), Unresolved)
    assert isinstance(base.inspect(ThresholdingAxiKernel.netlist).result, Unresolved)


def test_threshold_rejects_unsupported_profiles_and_malformed_tables() -> None:
    profiles = (
        threshold(table=(((2, 1),),)),
        threshold(table=(((0, 20),),)),
        threshold(bias=1 << 31),
        threshold(bias=-10),
    )
    assert all(
        isinstance(point.inspect(ThresholdingAxiKernel.netlist).result, Rejected)
        for point in profiles
    )
    assert all(inspection.validity(point).status == "refused" for point in profiles)
    # A table with no sets axis is refused on its facts, before any choice is set.
    flat = threshold_base(table=((-2, 0, 3),), channels=1)
    assert inspection.validity(flat).status == "refused"
    # More stages than the RTL has: the change placing them is refused by the triggers'
    # rule (a commit that breaks a rule is).
    report = threshold_base().try_with_choices(
        pe=1, ram_style="distributed", block_stages=2, ultra_stages=1
    )
    assert not report.accepted and [rule.rule for rule in report.broken] == ["depth_triggers"]
    # Thresholds of another signedness than the input: never written, the table is in
    # their type, which the RTL compares in the input's signedness; refused by that rule.
    unsigned = threshold_base(threshold_dtype="UINT5").try_with_choices(use_axilite=False)
    assert not unsigned.accepted and [rule.rule for rule in unsigned.broken] == ["types_supported"]
    # The table is an integer tensor value: a nest is none, however well formed (it makes
    # one, ``IntegerTensorValue.of``), and a malformed nest makes none.
    facts = dict(
        input_tensor=Tensor((1, 2), ScalarEncoding(DataType["INT8"])),
        threshold_dtype=DataType["INT5"],
        bias=-1,
        platform=FULL_DSP48E2,
    )
    with refused("literal-unrecognized", error=DefinitionError, message="IntegerTensorValue"):
        ThresholdingAxiKernel(**facts, thresholds=(((-2, 0, 3), (-1, 1, 4)),))
    for malformed in ((), (((0,), (0, 1)),), (([-2, 0, 3],),)):
        with pytest.raises(TypeError, match="integer tensor"):
            IntegerTensorValue.of(malformed)
    # PE divides the input's channels, the table's two rows; another is refused where it
    # is committed.
    for pe in (0, 3, 4, 5):
        assert not threshold_base().try_with_choices(pe=pe).accepted
    assert threshold_base().try_with_choices(pe=2).accepted


class NarrowInput(ThresholdingAxiKernel):
    """A thresholding whose input port takes only elements of up to 4 bits."""

    input = AxiStreamPort(
        name="s_axis",
        endpoint=Endpoint.TARGET,
        tensor=ThresholdingAxiKernel.input_tensor,
        schedule=ThresholdingAxiKernel.schedule,
        index=ThresholdingAxiKernel.indices,
        lanes=(c,),
        admits=Integer(max_bits=4),
    )


def test_a_refused_input_port_leaves_its_kernels_domains_and_refuses_its_products() -> None:
    """The port's rule (``input_element_taken``) stands in its kernel's validity: its
    extents read the port's access as a value, so PE keeps its candidates, while the
    kernel's module and cycles refuse with it."""

    def narrow(dtype: str) -> NarrowInput:
        return design_space(
            NarrowInput(
                input_tensor=Tensor((1, 2), ScalarEncoding(resolve_qonnx_datatype_name(dtype))),
                threshold_dtype=resolve_qonnx_datatype_name("INT5"),
                thresholds=THRESHOLDS,
                bias=-1,
                platform=FULL_DSP48E2,
            )
        )

    taken = narrow("INT4")
    assert inspection.validity(taken).status != "refused"
    point = narrow("INT8")
    assert point.field(ThresholdingAxiKernel.pe).candidates() == Available((1, 2))
    validity = inspection.validity(point)
    assert validity.status == "refused"
    assert [piece.owner for piece in validity.pieces] == ["input_element_taken"]
    # A commit is accepted all the same (a refusal elsewhere never refuses a commit).
    committed = point.try_with_choices(pe=1)
    assert committed.accepted
    assert [piece.owner for piece in inspection.validity(committed.instance).pieces] == [
        "input_element_taken"
    ]
    assert isinstance(committed.instance.query(ThresholdingAxiKernel.netlist), Rejected)
    assert isinstance(committed.instance.query(ThresholdingAxiKernel.cycles), Rejected)


def placed_on(channels: int, table: IntegerTensor) -> ThresholdingAxiKernel:
    """The kernel between two boundary channels of three rows of ``channels``, PE open."""
    shape = (3, channels)

    x_tensor = Tensor(shape, ScalarEncoding(DataType["INT8"]))
    y_tensor = Tensor(shape, ScalarEncoding(DataType["INT3"]))

    network = by_hand(
        {
            "x": channel(
                Side.boundary("x"),
                Port("activate", "input"),
                tensor=x_tensor,
                platform=FULL_DSP48E2,
            ),
            "y": channel(
                Port("activate", "output"),
                Side.boundary("y"),
                tensor=y_tensor,
                platform=FULL_DSP48E2,
            ),
            "activate": Facts(
                ThresholdingAxiKernel,
                {
                    "threshold_dtype": DataType["INT5"],
                    "thresholds": integer_tensor(table),
                    "bias": -1,
                    "input_tensor": x_tensor,
                    "platform": FULL_DSP48E2,
                },
            ),
        }
    )
    kernel: ThresholdingAxiKernel = network.member("activate")
    return kernel


def test_a_placed_table_has_one_row_or_a_row_a_channel() -> None:
    """Placed, PE ranges over the input's channels, whatever the table's rows; a table of
    another number of rows is refused by name."""
    row = (-2, 0, 3)
    for table in (((row,),), (tuple(row for _ in range(6)),)):
        point = placed_on(6, table)
        assert point.field(ThresholdingAxiKernel.pe).candidates() == Available((1, 2, 3, 6))
        chosen = point.try_with_choices(pe=3, use_axilite=False, deep_pipeline=False, **AUTO_MEMORY)
        assert chosen.accepted
        assert dict(library_module(chosen.instance).parameters)["C"] == len(table[0])
    other = placed_on(6, ((row, row),)).try_with_choices(pe=1).instance
    found = other.query(ThresholdingAxiKernel.schedule)
    assert isinstance(found, Rejected)
    assert {finding.code for finding in found.findings} == {"threshold-rows"}


def test_runtime_writable_thresholds_declare_the_writes_of_their_table() -> None:
    """Each threshold at (channel fold, lane, threshold) as thresholding_axi decodes it, its
    word masked to the table's bits; presented on the control bus with it."""
    table = (((-2, 0, 3), (-1, 1, 4), (0, 2, 5)),)  # three rows, N = 3
    point = threshold(table=table, pe=1, axilite=True)
    # Writable over an INT8 input, the table is INT9 ([-128, 128]), whatever its INT5
    # initial values: nine bits a word.
    assert point.table_dtype.name == "INT9"
    # PE 1: three folds of one lane; two bits of threshold, no lane bit, two of fold.
    expected = []
    for row_index, row in enumerate(table[0]):
        for index, value in enumerate(row):
            expected.append((((row_index << 2) | index) << 2, value & 0x1FF))
    assert point.register_map.writes == tuple(expected)
    assert point.config_control.registers == point.register_map
    shared = threshold(table=(((-2, 0, 3),),), pe=2, axilite=True)  # C = 1: one row, lane 0
    assert shared.register_map.writes == ((0, 0x1FE), (4, 0), (8, 3))
    assert threshold(pe=1).config_control.bus is None  # held, not presented: nothing to write


def test_a_writable_table_is_sized_from_the_input_never_its_initial_values() -> None:
    """Never written, the table's type is its thresholds' (``threshold_dtype``); runtime-
    writable, it holds any threshold a host may set over the input's clipped range, ``[min,
    max + 1]``: WT, the file's words and the register map's mask follow it."""
    table = (((-2, 0, 3),),)  # over an INT4 input: INT3, as the KernelOp narrows it
    fixed = threshold(table=table, input_dtype="INT4", threshold_dtype="INT3")
    written = threshold(table=table, input_dtype="INT4", threshold_dtype="INT3", axilite=True)
    assert (fixed.table_dtype.name, written.table_dtype.name) == ("INT3", "INT5")  # [-8, 8]
    assert [dict(library_module(point).parameters)["WT"] for point in (fixed, written)] == [3, 5]
    # N = 3, padded to four words: one hex digit of three bits, two of five.
    assert fixed.thresholds_file.data == b"6\n0\n3\n0\n"
    assert written.thresholds_file.data == b"1e\n00\n03\n00\n"
    assert fixed.register_map.writes == ((0, 0x6), (4, 0), (8, 3))
    assert written.register_map.writes == ((0, 0x1E), (4, 0), (8, 3))


def test_a_writable_table_holds_thresholds_of_another_signedness_than_the_input() -> None:
    """``UINT5`` thresholds over an ``INT4`` input. Never written, the table is ``UINT5``,
    which the RTL would compare signed (the input's): refused. Writable, the table holds
    both ranges, ``[0, 31]`` and the input's ``[-8, 8]``, signed: ``INT6``, which compares
    every initial threshold as it is."""
    table = (((1, 9, 17),),)  # none at 0, UINT5's minimum, which a negative input saturates to
    facts = dict(
        input_tensor=Tensor((1, 1), ScalarEncoding(DataType["INT4"])),
        threshold_dtype=DataType["UINT5"],
        thresholds=IntegerTensorValue.of(table),
        bias=0,
        platform=FULL_DSP48E2,
    )
    space = controlled(ThresholdingAxiKernel, facts)
    (viable,) = inspection.viable(space, [ThresholdingAxiKernel.use_axilite])
    assert viable.cases == (True,)
    (refused_value,) = [
        item for item in viable.values or () if isinstance(item, inspection.Refused)
    ]
    assert refused_value.value is False
    assert [rule.rule for rule in refused_value.broken] == ["types_supported"]
    written = threshold(
        table=table, bias=0, input_dtype="INT4", threshold_dtype="UINT5", axilite=True
    )
    assert written.table_dtype.name == "INT6"
    assert dict(library_module(written).parameters)["WT"] == 6
    assert written.register_map.writes == ((0, 1), (4, 9), (8, 17))
    assert inspection.validity(written).status != "refused"


def test_a_threshold_wider_than_a_word_is_written_in_words_low_first() -> None:
    point = threshold(
        table=(((-(1 << 35), 1 << 34),),),
        pe=1,
        input_dtype="INT8",
        threshold_dtype="INT40",
        axilite=True,
    )
    mask = (1 << 40) - 1
    low, high = (-(1 << 35)) & mask, 1 << 34
    # Two words a threshold: the word select below the threshold index; the last commits.
    assert point.register_map.writes == (
        (0, low & 0xFFFFFFFF),
        (4, low >> 32),
        (8, high & 0xFFFFFFFF),
        (12, high >> 32),
    )


def _clog2(value: int) -> int:
    return (value - 1).bit_length()


@pytest.mark.parametrize(
    "sets,rows,count,pe",
    [(1, 1, 3, 4), (1, 6, 3, 2), (1, 4, 4, 4), (2, 6, 5, 2), (3, 2, 7, 1), (2, 1, 1, 4)],
)
def test_the_thresholds_file_holds_each_threshold_where_the_rtl_reads_it(
    sets: int, rows: int, count: int, pe: int
) -> None:
    """``thresholding.sv`` reads memory word ``a`` of stage ``s`` and PE lane ``p`` at a
    file address (``genInitFile``); without a file it takes ``THRESHOLDS[set][c][t]``
    (``genInitParam``). Both, transcribed here, agree for every threshold the RTL loads."""
    table = tuple(
        tuple(tuple(16 * g + 4 * r + t - 30 for t in range(count)) for r in range(rows))
        for g in range(sets)
    )
    point = threshold(table=table, pe=pe, threshold_dtype="INT8")
    words = [int(word, 16) for word in point.thresholds_file.data.split()]
    folds, lanes = (1, rows) if pe >= rows else (rows // pe, pe)
    group = 2 ** _clog2(folds) if sets > 1 else folds
    assert len(words) == sets * group * 2 ** _clog2(lanes) * 2 ** _clog2(count)
    stages = count.bit_length()
    for stage in range(stages):
        below = stages - 1 - stage
        for lane in range(pe):
            for address in range(group * 2**stage):
                upper, index = address % 2**stage, address // 2**stage
                position = upper * 2 ** (below + 1) + 2**below - 1
                fold, chosen = index % 2 ** _clog2(folds), index // 2 ** _clog2(folds)
                if position >= count or fold >= folds:
                    continue  # never loaded from the table: the RTL masks it
                read = (
                    index * 2 ** _clog2(lanes) * 2 ** _clog2(count)
                    + (lane % lanes) * 2 ** _clog2(count)
                    + position
                )
                expected = table[chosen][fold * lanes + lane % lanes][position]
                assert words[read] == expected & 0xFF


def test_a_table_over_a_million_bits_is_no_parameter() -> None:
    """Vivado stops on a parameter over 10**6 bits (issue thresholds-parameter-limit):
    the table is the file's, and no parameter grows with it."""
    rows, count = 512, 255
    table = (tuple(tuple(range(-128, 127)) for _ in range(rows)),)
    point = threshold(table=table, pe=8, input_dtype="INT8", threshold_dtype="INT8")
    assert rows * count * 8 > 10**6
    parameters = dict(library_module(point).parameters)
    assert max(len(str(value)) for value in parameters.values()) < 40
    assert len(point.thresholds_file.data.split()) == (rows // 8) * 8 * 256


@pytest.mark.parametrize("bits", [64, 70])
def test_thresholds_past_64_bits_reach_the_file_and_the_writes_exactly(bits: int) -> None:
    """The table's integers are read as Python ints: a threshold of 64 bits or more is
    masked and written exactly, in the file and in the AXI-Lite writes."""
    least, greatest = -(1 << (bits - 1)) + 1, (1 << (bits - 2)) + 5
    table = IntegerTensorValue.of((((least, greatest),),))
    dtype = f"INT{bits}"
    point = threshold(table=table, pe=1, threshold_dtype=dtype)
    mask, digits = (1 << bits) - 1, (bits + 3) // 4
    assert point.thresholds_file.data == (
        f"{least & mask:0{digits}x}\n{greatest:0{digits}x}\n".encode()
    )
    words = (bits + 31) // 32
    writes = threshold(table=table, pe=1, threshold_dtype=dtype, axilite=True).register_map.writes
    assert [value for _, value in writes] == [
        ((threshold & mask) >> (32 * word)) & 0xFFFFFFFF
        for threshold in (least, greatest)
        for word in range(words)
    ]
