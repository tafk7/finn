# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Phase 6b, MVAU half: what the generation stages are allowed to read.

``test_elaboration_audit`` makes the generic claim -- a Kernel's elaboration
cannot emit a value the point did not answer.  This module asks the same
question of the four stages downstream of it on the real MVAU path: wrapper
generation, shim generation, packaging, and synthesis preparation.

The check has two shapes, and they establish different things.

**Structural**, over the function signatures: these stages take strings,
numbers and already-resolved values.  None of them takes a ``ModelWrapper``, a
``DesignPoint``, a resolved design or a build config, so there is no ambient
state for a design choice to be read out of.  That is a property of the
interface and holds for inputs nobody thought to test.

**Textual**, over what they produce: every parameter the generated wrapper
overrides is one of the bound Kernels' declared parameters, carrying the value
the binding resolved.  That is narrower -- it catches specific violations in
this configuration -- but it is the half that would actually notice a
generator computing a width on the way out.

Neither proves "no design choice is made here".  Item 9 is not provable by a
passing test; see ``test_elaboration_audit``'s module docstring for why, and
what is claimed instead.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

import pytest
from qonnx.core.datatype import DataType  # type: ignore[import-not-found]

from dataflow.mvau import test_datatype_continuity as continuity
from finn.dataflow.ops.mvau.artifacts import _implementation as composition
from dataflow.mvau.test_decomposed_op import _committed, _context, _model
from finn.dataflow.artifacts import TargetIdentity
from finn.dataflow.ops.mvau.physical import MVAUElaborationError
from finn.dataflow.ops.mvau.binding import bind_decomposed
from finn.dataflow.kernels.dotp_axi import DotpAxiKernel
from finn.dataflow.kernels.replay_buffer import ReplayBufferKernel
from finn.dataflow.ops.mvau.artifacts._implementation import (
    MVAUDecomposedArtifactRequirements,
    build_decomposed_artifact_requirements,
    elaborate_decomposed,
    package_decomposed_artifact,
    prepare_decomposed_synthesis,
    prepare_ip_package,
    render_clock_constraints,
    render_decomposed_wrapper,
    render_stitch_shim,
    staged_layout,
)
from finn.dataflow.ops.mvau.source import MVAUResolvedDesign

FINN_ROOT = Path(__file__).resolve().parents[3]

#: Types that carry a whole compilation state.  A generation stage taking one
#: could read anything from it, and no signature audit downstream would show
#: which.  Matched by name because the annotations are strings under
#: ``from __future__ import annotations``.
AMBIENT_TYPES = (
    "ModelWrapper",
    "DesignPoint",
    "Engine",
    "MVAUResolvedDesign",
    "MVAUDataflowBuildContext",
    "DataflowBuildConfigView",
    "MvauDataflowOp",
)

#: The stages this module audits, in the order a build runs them.
#:
#: ``prepare_ip_package`` joined in Phase 6f and the drift check below caught
#: its absence -- which is what that check is for: a generator added without
#: being audited is a generator nobody asked what it reads.
GENERATION_STAGES: tuple[Callable[..., object], ...] = (
    render_decomposed_wrapper,
    render_stitch_shim,
    render_clock_constraints,
    staged_layout,
    package_decomposed_artifact,
    prepare_decomposed_synthesis,
    prepare_ip_package,
)


@pytest.fixture(name="requirements")
def _requirements() -> MVAUDecomposedArtifactRequirements:
    operation = _committed(_model())
    resolved = operation.resolve_dataflow(_context())
    return build_decomposed_artifact_requirements(
        resolved, elaborate_decomposed(resolved), FINN_ROOT
    )


# -- the structural half -------------------------------------------------------


@pytest.mark.parametrize("stage", GENERATION_STAGES, ids=lambda item: item.__name__)
def test_no_generation_stage_takes_a_whole_compilation_state(
    stage: Callable[..., object],
) -> None:
    """The strongest thing said here, and it is said by the signature.

    A stage handed a ``ModelWrapper`` or a ``DesignPoint`` could read anything
    at all, and no amount of checking its output would say which.  Refusing the
    argument is what makes "reads only what it was given" checkable rather than
    reviewed.
    """

    signature = inspect.signature(stage)
    for name, parameter in signature.parameters.items():
        annotation = str(parameter.annotation)
        for ambient in AMBIENT_TYPES:
            assert ambient not in annotation, f"{stage.__name__}({name}: {annotation})"


def test_the_stage_list_is_the_whole_public_generation_surface() -> None:
    """A list that drifts behind the module tests nothing about what was added.

    Every public callable in ``composition`` whose name renders, stages or
    prepares is audited above -- so a new generator has to be added here or
    this fails.
    """

    prefixes = ("render_", "staged_", "package_", "prepare_")
    public = {
        name for name in composition.__all__ if any(name.startswith(prefix) for prefix in prefixes)
    }
    assert public == {stage.__name__ for stage in GENERATION_STAGES}


# -- the textual half ----------------------------------------------------------


def _overrides(source: str) -> dict[str, dict[str, int]]:
    """Parameter overrides per instantiated module, from the generated text.

    Only what is inside a ``#( ... )``.  Port connections use the same
    ``.name(value)`` spelling, so a regex over the whole file would report
    ``m_axis_output_tdata`` as an undeclared parameter and be noise rather than
    a finding.
    """

    found: dict[str, dict[str, int]] = {}
    module: str | None = None
    for line in source.splitlines():
        stripped = line.strip()
        # The top's own header ends in ``#(`` too, and what follows it is the
        # port-width parameter *declarations* rather than overrides of anything.
        if stripped.endswith("#(") and not stripped.startswith("module "):
            module = stripped[: -len("#(")].strip()
            found[module] = {}
            continue
        if module is None:
            continue
        if stripped.startswith(")"):
            module = None
            continue
        assert stripped.startswith("."), stripped
        name, _, rest = stripped.partition("(")
        found[module][name.lstrip(".")] = int(rest.rstrip("),"))
    return found


def test_the_generated_wrapper_overrides_only_declared_parameters(
    requirements: MVAUDecomposedArtifactRequirements,
) -> None:
    """Every ``.NAME(value)`` in the top, against the binding that owns it.

    Both directions.  A name the Kernel never declared would be a value with no
    owner; a declared name carrying something else would make the artifact key
    -- which records the resolved value -- describe RTL that does not exist.

    Booleans reach Verilog as ``0``/``1``, so the comparison is over ``int``;
    that is a rendering convention, not a second value.
    """

    bindings = bind_decomposed(_resolved_from(requirements))
    declared = {
        "replay_buffer": dict(bindings.kernel("replay").parameters),
        "dotp_axi": dict(bindings.kernel("compute").parameters),
    }
    overrides = _overrides(requirements.wrapper_source)
    assert set(overrides) == set(declared)
    for module, values in overrides.items():
        assert set(values) == set(declared[module]), module
        for name, value in values.items():
            assert value == int(declared[module][name]), f"{module}.{name}"  # type: ignore[call-overload]


def test_the_shim_declares_the_same_port_widths_as_the_top_it_wraps(
    requirements: MVAUDecomposedArtifactRequirements,
) -> None:
    """Two texts, one set of widths, and no second derivation between them.

    The shim exists because Vivado will not take a ``.sv`` top as a
    ``create_bd_cell`` reference.  It is a second place the widths are written
    down, which is a second place they can be wrong -- fixture 7 would catch a
    mismatch only as a stitching failure, and only if it was run.
    """

    for name in ("WSTREAM", "ISTREAM", "OSTREAM"):
        top = _parameter_value(requirements.wrapper_source, name)
        shim = _parameter_value(requirements.stitch_source, name)
        assert top == shim, name


def test_the_synthesis_script_names_only_the_target_the_clock_and_the_staged_files(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """Preparation writes a script; it must not decide anything in it.

    A synthesis directive that is neither in the packaged unit nor in the
    target -- a strategy, a retiming flag, an effort level -- would be a design
    choice made after the artifact was keyed, and two runs differing only in it
    would share a key.  The recipe is a fixed shape and this checks that what
    fills it comes from the two values the stage was handed.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)
    target = TargetIdentity(requirements.target_fpga_part, requirements.clock_period_ns)
    prepared = prepare_decomposed_synthesis(packaged, target, tmp_path)
    script = Path(prepared.script_path).read_text()

    braced = {item.strip("{}") for item in script.split() if item.startswith("{")}
    allowed = {*packaged.files, prepared.constraints_path, prepared.report_path}
    assert braced <= allowed, braced - allowed
    assert target.fpga_part in script
    assert packaged.top_module_name in script
    constraints = Path(prepared.constraints_path).read_text()
    assert str(target.clock_period_ns) in constraints
    assert str(target.clock_period_ns / 2) in constraints


def test_packaging_stages_exactly_the_layout_and_invents_no_file(
    requirements: MVAUDecomposedArtifactRequirements, tmp_path: Path
) -> None:
    """The packaged unit's contents, against the layout its key is taken over.

    Already enforced in ``PackagedDecomposedArtifact.__post_init__``; asserted
    here because "packaging makes no choice" is one of the four stages item 9
    covers, and an audit that skipped it because another test happens to catch
    it would be relying on a coincidence of coverage.
    """

    packaged = package_decomposed_artifact(requirements, tmp_path)
    on_disk = sorted(item.name for item in Path(packaged.directory).iterdir())
    assert on_disk == sorted(staged_layout(requirements))
    assert on_disk == sorted(packaged.identity.layout)


# -- item 12's remaining half: refused before, not during ----------------------

#: One graph per way the decomposed core can be asked for something it is not.
#: ``test_fused_hardware`` establishes that each is refused; what is open is
#: *where*, and the answer has to be the same for all of them.
UNSUPPORTED = (
    ("ternary weights", {"weight": DataType["TERNARY"]}),
    ("binary weights", {"weight": DataType["BINARY"]}),
    ("bipolar activations", {"activation": DataType["BIPOLAR"]}),
    (
        "floating accumulator",
        {"accumulator": DataType["FLOAT16"], "output": DataType["FLOAT16"]},
    ),
    (
        "unsigned accumulator",
        {"accumulator": DataType["UINT16"], "output": DataType["UINT16"]},
    ),
    ("one-bit weights", {"weight": DataType["UINT1"]}),
    (
        "accumulator past the datapath",
        {
            "weight": DataType["INT32"],
            "accumulator": DataType["INT32"],
            "output": DataType["INT32"],
        },
    ),
)


@pytest.mark.parametrize(("label", "change"), UNSUPPORTED, ids=[item[0] for item in UNSUPPORTED])
def test_an_unsupported_signature_is_refused_before_any_kernel_elaborates(
    label: str, change: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusal that arrives *out of* elaboration has already lost the diagnostic.

    "It is refused" is weaker than it sounds: an exception surfacing from
    wrapper generation is also a refusal, and it names a rendering failure
    rather than an uncovered point, after doing work that should not have
    begun.  So both Kernel elaborations are sabotaged here -- if either runs,
    the test sees the sabotage instead of the coverage finding.
    """

    def refuse_to_run(cls: type, binding: object) -> tuple[object, ...]:
        raise AssertionError(f"{cls.__name__} elaborated an uncovered point")

    for kernel in (DotpAxiKernel, ReplayBufferKernel):
        monkeypatch.setattr(kernel, "elaborate", classmethod(refuse_to_run))

    resolved = continuity._resolved(continuity._model(**change))  # type: ignore[arg-type]
    for entry in (bind_decomposed, elaborate_decomposed):
        with pytest.raises(MVAUElaborationError) as refusal:
            entry(resolved)
        codes = {item.code for item in refusal.value.findings}
        assert "hardware-coverage-refused" in codes, codes


# -- helpers -------------------------------------------------------------------


def _parameter_value(source: str, name: str) -> int:
    for line in source.splitlines():
        stripped = line.strip().rstrip(",")
        if stripped.startswith(f"parameter {name} = "):
            return int(stripped.split(" = ")[1])
    raise AssertionError(f"no parameter {name!r} in this text")


def _resolved_from(requirements: MVAUDecomposedArtifactRequirements) -> MVAUResolvedDesign:
    """The resolved design behind these requirements, rebuilt the same way.

    Rebuilt rather than carried, because ``MVAUDecomposedArtifactRequirements``
    deliberately holds the elaboration and not the point -- which is the same
    separation this module is auditing.
    """

    operation = _committed(_model())
    return operation.resolve_dataflow(_context())
