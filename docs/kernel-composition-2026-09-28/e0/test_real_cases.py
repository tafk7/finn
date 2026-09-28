# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause
"""Cases 1-4 over the real kernels: the refined form against today's declaration.

Equivalence evidence: the 14 configurations of the D10 identity dump build
identical module fingerprints (and equal assemblies), the fact sets declare
identical decision keys, and compute and adapter compatibility and refusals
agree case by case.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from core.space._collapse_support import answers, open_space
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]
from real_cases import (
    ADMISSION,
    OwnedRamStyleStream,
    RefinedBufferedStream,
    RefinedMatMul,
    RefinedStream,
    refined_assembly,
)
from refined import Decision, compatible_cases
from spike import engine_spike

from finn.core.space import (
    Available,
    DefinitionError,
    Inapplicable,
    Space,
    View,
    composite,
    design_space,
    inspection,
)
from finn.kernels.adapters import (
    InputGenAdapter,
    RegroupAdapter,
    RegroupMarkersAdapter,
    ReorderWidthAdapter,
    ReorderWidthMarkersAdapter,
    StreamAdapter,
    WidthAdapter,
    WidthReorderAdapter,
)
from finn.kernels.artifacts.build import module_build_fingerprint
from finn.kernels.configure import commit, compatible
from finn.kernels.delivery import CyclicDelivery
from finn.kernels.dotp import DotpAxiKernel, Int8Dsp58DotpKernel, PackedDotpKernel
from finn.kernels.matmul import MatMulKernel, matmul_assembly
from finn.kernels.memstream import MemStreamKernel
from finn.kernels.streams import Stream, StreamFifo, _Direct
from finn.kernels.target import DspBlock

_DUMP = Path(__file__).resolve().parents[2] / "stream-model-2026-09-27" / "identity_dump.py"


def _identity_dump() -> Any:
    """The D10 identity dump's configurations, without running its printing loop."""
    source = _DUMP.read_text().split("replay = next", 1)[0]
    namespace: dict[str, Any] = {"__name__": "identity_dump"}
    exec(compile(source, str(_DUMP), "exec"), namespace)  # noqa: S102 - a checked-in probe input
    return namespace


DUMP = _identity_dump()
CONFIGS: dict[str, dict[str, Any]] = {
    name: {key: value for key, value in {**DUMP["BASE"], **overrides}.items() if key != "replay"}
    for name, overrides in DUMP["CONFIGS"].items()
}
COMMON = dict(
    activation_dtype=DataType["INT8"],
    weights_dtype=DataType["INT8"],
    target_dsp=DspBlock.DSP58,
    target_period_ns=5.0,
)


def _unsourced(message: str) -> str:
    return re.sub(r"set by \w+ at [\w.]+:\d+", "set by <body>", message)


def keys(point: Any) -> list[str]:
    return sorted(item.key for item in inspection.decisions(point))


# -- cases 1-4 together: module parameters and keys ---------------------------------------


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_refined_matmul_builds_todays_module_for_every_identity_configuration(
    name: str,
) -> None:
    today = matmul_assembly(**CONFIGS[name])
    refined, settled = refined_assembly(RefinedMatMul, **CONFIGS[name])
    assert module_build_fingerprint(refined.requirements) == module_build_fingerprint(
        today.requirements
    )
    assert refined == today
    # settle chose what matmul_assembly's compatible() and commit_adapters() chose.
    if "core" not in CONFIGS[name]:
        assert settled.committed["compute"] == "packed"
    assert settled.committed["activations.adapter"] == "input_gen"


@pytest.mark.parametrize("facts", sorted(DUMP["FACTS"]))
def test_the_refined_matmul_declares_todays_decision_keys(facts: str) -> None:
    today = design_space(MatMulKernel(**COMMON, **DUMP["FACTS"][facts]))
    refined = design_space(RefinedMatMul(**COMMON, **DUMP["FACTS"][facts]))
    assert keys(refined) == keys(today)
    for left, right in zip(inspection.choices(today), inspection.choices(refined)):
        assert left.key == right.key
        assert [case.name for case in left.cases] == [case.name for case in right.cases]
        assert [case.space_type for case in left.cases] == [case.space_type for case in right.cases]


# -- case 1: the dot-product core ---------------------------------------------------------


def _core_facts(**changes: Any) -> dict[str, Any]:
    facts = dict(DUMP["FACTS"]["dense"], **COMMON)
    facts.update(changes)
    return facts


CORE_CASES = {
    "int8 on dsp58": ({}, ("packed", "int8_dsp58")),
    "int9 weights on dsp58": (dict(weights_dtype=DataType["INT9"]), ("packed",)),
    "int3 on dsp48e2": (
        dict(
            activation_dtype=DataType["INT3"],
            weights_dtype=DataType["INT3"],
            target_dsp=DspBlock.DSP48E2,
        ),
        ("packed",),
    ),
    "int24 activations on dsp48e2": (
        dict(activation_dtype=DataType["INT24"], target_dsp=DspBlock.DSP48E2),
        (),
    ),
}


@pytest.mark.parametrize("case", sorted(CORE_CASES))
def test_case1_core_compatibility_and_refusals_agree(case: str) -> None:
    changes, expected = CORE_CASES[case]
    choices = {"pe": 2, "simd": 2, "compute_pumping": False, "delivery": "external"}
    today = commit(design_space(MatMulKernel(**_core_facts(**changes))), choices)
    refined = commit(design_space(RefinedMatMul(**_core_facts(**changes))), choices)

    def support(point: Any) -> Any:
        return point.compute.inspect(DotpAxiKernel.support).result

    assert compatible(today, "compute", support) == expected
    assert compatible_cases(refined, "compute", ADMISSION) == expected
    for core in ("packed", "int8_dsp58"):
        left = support(commit(today, {"compute": core}))
        right = support(commit(refined, {"compute": core}))
        assert type(left) is type(right)
        if not isinstance(left, Available):
            # The text agrees up to provenance, which names the body and line that set
            # an input: for a shared binding, the refined Decision's line.
            assert [(f.code, _unsourced(f.message)) for f in left.findings] == [
                (f.code, _unsourced(f.message)) for f in right.findings
            ]


def test_case1_direct_and_qualified_reads_over_the_real_cores() -> None:
    class Reads(RefinedMatMul):
        interfaces = View(RefinedMatMul.compute.interfaces)  # every core declares it
        narrow = View(RefinedMatMul.compute["packed"].narrow_weights)

    with pytest.raises(DefinitionError, match=r"candidates \['int8_dsp58'\] do not declare"):
        RefinedMatMul.compute.narrow_weights  # noqa: B018
    base = commit(
        design_space(Reads(**_core_facts())),
        {"pe": 2, "simd": 2, "compute_pumping": False, "delivery": "external"},
    )
    packed, int8 = commit(base, {"compute": "packed"}), commit(base, {"compute": "int8_dsp58"})
    assert packed.narrow is False  # external weights promise nothing
    assert isinstance(int8.query(Reads.narrow), Inapplicable)
    assert len(packed.interfaces) == len(int8.interfaces) == 3


def test_case1_a_shared_binding_is_checked_against_every_core() -> None:
    with pytest.raises(DefinitionError, match=r"'narrow_weights' is not declared by candidates"):
        Decision(
            {"packed": PackedDotpKernel, "int8_dsp58": Int8Dsp58DotpKernel},
            narrow_weights=True,
        )


# -- case 2: weight memory ------------------------------------------------------------


def test_case2_optional_memory_places_nothing_for_external_weights() -> None:
    base = commit(
        design_space(RefinedMatMul(**_core_facts())),
        {"pe": 2, "simd": 2, "compute_pumping": False},
    )
    (choice,) = [item for item in inspection.choices(base) if item.key == "delivery"]
    assert [case.name for case in choice.cases] == ["external", "cyclic", "memstream"]
    assert [case.space_type for case in choice.cases] == [None, CyclicDelivery, MemStreamKernel]
    assert commit(base, {"delivery": "external"}).delivery is None


def test_case2_values_cannot_be_a_shared_binding() -> None:
    """Both memories take ``values``: the collision Q1 decides (here written per entry)."""
    with pytest.raises(DefinitionError, match=r"shared bindings \['values'\]"):
        Decision(
            {"cyclic": CyclicDelivery, "memstream": MemStreamKernel},
            optional=True,
            values=MatMulKernel.datapath_weights,
        )


# -- case 3: the stream adapter --------------------------------------------------------


def test_case3_adapter_compatibility_agrees_over_configurations() -> None:
    for name in ("external", "padded-output", "per-channel-native", "fifo-cyclic"):
        config = CONFIGS[name]
        today = matmul_assembly(**config)
        refined, settled = refined_assembly(RefinedMatMul, **config)
        instances = [item.instance_id for item in today.structure.instances]
        assert [item.instance_id for item in refined.structure.instances] == instances
        assert settled.open.get("activations.adapter", ()) == ()


def test_case3_a_shared_ram_style_passes_vacuously_because_the_base_declares_it() -> None:
    """All seven adapters inherit ``ram_style`` from StreamAdapter, so strictness passes.

    The rule is only as honest as the declarations: ``WidthAdapter`` has no memory.
    """
    T = Stream

    class Vacuous(Stream):
        adapter: StreamAdapter = Decision(
            {
                "input_gen": InputGenAdapter,
                "vpc": WidthAdapter,
                "vpc_input_gen": WidthReorderAdapter,
                "input_gen_vpc": ReorderWidthAdapter,
                "input_gen_vpc_input_gen": ReorderWidthMarkersAdapter,
                "vpc_input_gen_vpc": RegroupAdapter,
                "vpc_input_gen_vpc_input_gen": RegroupMarkersAdapter,
            },
            tensor=T.tensor,
            plan=T.plan,
            ram_style=T.adapter_ram_style,
            when=T.adapting,
        )

    assert "ram_style" in vars(StreamAdapter) and "ram_style" not in vars(WidthAdapter)
    vpc = Vacuous.adapter._space_decision().candidates["vpc"]
    assert vpc is not None and "ram_style" in vpc.bindings  # a memory style for a chain with none


def test_case3_option_c_adapters_own_their_ram_style() -> None:
    class Owned(RefinedMatMul):
        activations = OwnedRamStyleStream(tensor=MatMulKernel.activation_tensor, port="in0_V")

    config = CONFIGS["external"]
    facts = ("rows", "reduction", "outputs", "activation_dtype", "weights_dtype", "target_dsp")
    base = design_space(Owned(**{key: config[key] for key in facts}, target_period_ns=5.0))
    owned = [key for key in keys(base) if key.startswith("activations.adapter.")]
    assert owned == [
        "activations.adapter.input_gen.ram_style",
        "activations.adapter.input_gen_vpc.ram_style",
        "activations.adapter.input_gen_vpc_input_gen.ram_style",
        "activations.adapter.vpc_input_gen.ram_style",
        "activations.adapter.vpc_input_gen_vpc.ram_style",
        "activations.adapter.vpc_input_gen_vpc_input_gen.ram_style",
    ]
    point = commit(
        base,
        {
            "delivery": "external",
            "weight_stream.transport": "direct",
            "compute_pumping": False,
            "pe": 2,
            "simd": 2,
            "compute": "packed",
            "activations.adapter": "input_gen",
            "activations.adapter.input_gen.ram_style": "auto",
        },
    )
    composed = point.query(MatMulKernel.structure)
    assert isinstance(composed, Available)
    today = matmul_assembly(**config)
    assert module_build_fingerprint(composed.value.requirements) == module_build_fingerprint(
        today.requirements
    )
    # The stream-level Decision is still declared by Stream and now binds nothing (K2 drops it).
    assert "activations.adapter_ram_style" in keys(base)


# -- case 4: transport ------------------------------------------------------------------


def test_case4_transport_keeps_its_keys_and_direct_takes_no_binding() -> None:
    base = design_space(RefinedMatMul(**_core_facts()))
    transport = [key for key in keys(base) if key.startswith("weight_stream.transport")]
    assert transport == [
        "weight_stream.transport",
        "weight_stream.transport.fifo.buffer.depth",
        "weight_stream.transport.fifo.buffer.ram_style",
    ]
    cases = RefinedBufferedStream.transport._space_decision().candidates
    assert list(cases) == ["direct", "fifo"]
    direct, fifo = cases["direct"], cases["fifo"]
    assert direct is not None and direct.bindings == {}
    assert fifo is not None and sorted(fifo.bindings) == ["arriving", "tensor"]
    with pytest.raises(DefinitionError, match=r"'tensor' is not declared by candidates \['direct"):
        Decision({"direct": _Direct, "fifo": StreamFifo}, tensor=Stream.tensor)


# -- narrowing, collapse, programmatic --------------------------------------------------


def test_todays_narrowing_cannot_restate_a_stream_binding_but_the_spike_pins_by_key() -> None:
    with pytest.raises(DefinitionError, match="reaches into another node"):

        class Restated(Space):
            mm = RefinedMatMul(**_core_facts())
            mm.compute = Decision({"packed": PackedDotpKernel}, activation_stream=mm.activations)

    with engine_spike():

        class Pinned(Space):
            mm = RefinedMatMul(**_core_facts())
            mm.compute = "int8_dsp58"  # type: ignore[assignment]

        pinned = design_space(Pinned())

    class Open(Space):
        mm = RefinedMatMul(**_core_facts())

    assert "mm.compute" not in keys(pinned)
    before = {item.key for item in inspection.pinned(design_space(Open()))}
    assert {item.key for item in inspection.pinned(pinned)} - before == {"mm.compute"}
    choices = {"pe": 2, "simd": 2, "compute_pumping": False, "delivery": "external"}
    point = commit(pinned, {f"mm.{key}": value for key, value in choices.items()})
    assert isinstance(point.mm.compute, Int8Dsp58DotpKernel)
    today = commit(
        design_space(RefinedMatMul(**_core_facts())), {**choices, "compute": "int8_dsp58"}
    )
    assert (
        point.mm.compute.inspect(DotpAxiKernel.support).result
        == today.compute.inspect(DotpAxiKernel.support).result
    )


def test_every_node_answers_the_same_with_and_without_collapse() -> None:
    choices = {
        "pe": 2,
        "simd": 2,
        "compute_pumping": False,
        "delivery": "external",
        "weight_stream.transport": "direct",
        "compute": "packed",
        "activations.adapter": "input_gen",
        "activations.adapter_ram_style": "auto",
    }
    facts = dict(
        rows=3,
        reduction=4,
        outputs=4,
        activation_dtype=DataType["INT3"],
        weights_dtype=DataType["INT3"],
        target_dsp=DspBlock.DSP48E2,
        target_period_ns=5.0,
    )
    for applied in ({}, choices):
        collapsed = open_space(RefinedMatMul(**facts), collapsed=True)
        plain = open_space(RefinedMatMul(**facts), collapsed=False)
        if applied:
            collapsed, plain = commit(collapsed, applied), commit(plain, applied)
        assert answers(collapsed) == answers(plain)


def test_the_refined_matmul_declared_at_run_time_keeps_keys_and_modules() -> None:
    """As G1's adapter would: the composite built from data, on today's MatMul base."""
    M, T = MatMulKernel, Stream
    adapters = {
        "input_gen": InputGenAdapter(ram_style=T.adapter_ram_style),
        "vpc": WidthAdapter,
        "vpc_input_gen": WidthReorderAdapter(ram_style=T.adapter_ram_style),
        "input_gen_vpc": ReorderWidthAdapter(ram_style=T.adapter_ram_style),
        "input_gen_vpc_input_gen": ReorderWidthMarkersAdapter(ram_style=T.adapter_ram_style),
        "vpc_input_gen_vpc": RegroupAdapter(ram_style=T.adapter_ram_style),
        "vpc_input_gen_vpc_input_gen": RegroupMarkersAdapter(ram_style=T.adapter_ram_style),
    }
    stream = composite(
        "RunTimeStream",
        {"adapter": Decision(adapters, tensor=T.tensor, plan=T.plan, when=T.adapting)},
        base=Stream,
        annotations={"adapter": StreamAdapter},
    )
    activations = stream(tensor=M.activation_tensor, port="in0_V")
    weight_stream = RefinedBufferedStream(tensor=M.weight_tensor, port="in1_V")
    results = RefinedStream(tensor=M.result_tensor, port="out0_V")
    set_index = RefinedStream(tensor=M.set_tensor, port="in2_V", when=M.multi_set)
    shared = dict(
        activation_dtype=M.activation_dtype,
        weights_dtype=M.weights_dtype,
        result_dtype=M.result_type,
        pe=M.pe,
        simd=M.simd,
        target_dsp=M.target_dsp,
        target_period_ns=M.target_period_ns,
        compute_pumping=M.compute_pumping,
        contraction=M.datapath,
        activation_stream=activations,
        weights_stream=weight_stream,
        result_stream=results,
        iteration=M.iteration,
    )
    family = composite(
        "RunTimeMatMul",
        {
            "activations": activations,
            "weight_stream": weight_stream,
            "results": results,
            "set_index": set_index,
            "compute": Decision(
                {
                    "packed": PackedDotpKernel(narrow_weights=M.narrow_weights),
                    "int8_dsp58": Int8Dsp58DotpKernel,
                },
                **shared,
            ),
            "delivery": Decision(
                {
                    "cyclic": CyclicDelivery(values=M.datapath_weights),
                    "memstream": MemStreamKernel(
                        values=M.datapath_weights,
                        writable=M.writable_weights,
                        sets=M.weight_sets,
                        set_stream=set_index,
                        control=M.config,
                    ),
                },
                optional="external",
                dtype=M.weights_dtype,
                form=M.weight_period,
                output_stream=weight_stream,
            ),
        },
        base=MatMulKernel,
        annotations={
            "compute": PackedDotpKernel | Int8Dsp58DotpKernel,
            "delivery": CyclicDelivery | MemStreamKernel | None,
        },
    )
    for facts in DUMP["FACTS"].values():
        assert keys(design_space(family(**COMMON, **facts))) == keys(
            design_space(MatMulKernel(**COMMON, **facts))
        )
    for name in ("external", "cyclic-block", "memstream-sets"):
        built, _ = refined_assembly(family, **CONFIGS[name])
        assert built == matmul_assembly(**CONFIGS[name])
