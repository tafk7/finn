# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The build's steps. Each is a function ``(model, DataflowBuildConfig) -> model``,
registered by name in ``build_dataflow_step_lookup`` so that a configuration's ``steps``
can name it; the phases (finn.builder.build_dataflow_phases) group them in order.

A step with options of its own declares them as a dataclass beside it and registers that
type by the step's name in ``build_dataflow_step_options``. A configuration sets them in
``step_options``, and the step reads its own with ``DataflowBuildConfig.options``.

The steps prepare a Brevitas export into the streamlined graph the kernel path converts,
and check it; convert its nodes to KernelOps (finn.custom_op.kernels) and cut them into
one partition; choose the partition's parameters; and make the outputs the configuration
asks for: the partition's packaged IP on any shell, and the bitfile, driver and
deployment package on a shell that integrates it."""

import copy
import dataclasses
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    VerificationStepType,
)
from finn.builder.build_dataflow_resources import RESOURCES_FILE, shell_resources_report
from finn.builder.build_dataflow_testbench import (
    TESTBENCH_DIR,
    generated_frame,
    partition_frame,
    run_testbench,
    write_testbench,
)
from finn.core.onnx_exec import execute_onnx
from finn.core.space import by_code, finding_record
from finn.custom_op.kernels.base import read_target
from finn.custom_op.kernels.shell import configured_root
from finn.custom_op.partition.kernel_partitions import (
    OUTPUT_BITFILE,
    OUTPUT_HWH,
    OUTPUT_IP,
    OUTPUT_PROJECT,
    OUTPUT_REPORTS,
    partition_body,
)
from finn.harness.preparation import check_equivalence
from finn.platform import refuse_drift
from finn.shells.pynq.driver import DRIVER_DIR, DRIVER_REPORT, MakePynqDriver
from finn.shells.pynq.runner import BITFILE, BuildPynqBitfile, pynq_row, refuse_stale_bitfile
from finn.toolchain.launcher import Toolchain
from finn.transformation.kernels import (
    EXPLORATION_REPORT,
    ExplorePartition,
    ToKernelOps,
    kernel_choices_config,
    kernel_ops_by_anchor,
    kernel_ops_report,
    kernel_ops_summary,
    refuse_host_between,
    strategy,
)
from finn.transformation.kernels.cut import CutKernelPartition
from finn.transformation.kernels.package import (
    ElaboratePartition,
    PackagePartition,
    partition_utilization_report,
)
from finn.transformation.prepare import (
    DEVIATIONS,
    SUB_PHASES,
    GraphPreparation,
    PreparationRefused,
    census,
    checkpoint,
    reference,
)
from finn.transformation.prepare.phase import (
    DEFAULT_STREAMLINING,
    DEFAULT_TOPOLOGY,
    SubPhase,
    containers_exact,
    imported,
    io_explicit,
    quantization_lowered,
    streamlined,
    topology_settled,
)

#: The steps' logger. The build log has every line; the console shows only warnings, such
#: as a failed verification the build continued past, or a limitation.
_LOG = logging.getLogger(__name__)

#: The graph-preparation phase's report, under the output directory.
PREPARATION_REPORT = "report/graph_preparation.json"


def _tools(cfg: DataflowBuildConfig, step: str) -> Toolchain:
    """The build's toolchain for ``step``, writing every tool's transcript into
    ``output_dir/logs/<step>``. The step passes it to each transformation that runs a
    tool."""
    return cfg._resolve_toolchain().transcripts_in(Path(cfg.output_dir) / "logs" / step)


def _verdict(name: str, passed: bool) -> None:
    """Log a verification's result, ``Verification for <name> : SUCCESS`` or ``FAIL``. A
    FAIL is a warning, because the build continues."""
    if passed:
        _LOG.info(f"Verification for {name} : SUCCESS")
    else:
        _LOG.warning(f"Verification for {name} : FAIL")


def _preparation_source(cfg: DataflowBuildConfig) -> str:
    """Where step_prepare_import saves the export, the reference for the
    GRAPH_PREPARATION_PYTHON check."""
    return cfg.output_dir + "/intermediate_models/graph_preparation_source.onnx"


def _preparation_report(cfg: DataflowBuildConfig, fresh: bool = False) -> dict[str, Any]:
    """The graph-preparation report written so far, or an empty one if ``fresh`` or none
    exists."""
    path = Path(cfg.output_dir) / PREPARATION_REPORT
    if fresh or not path.is_file():
        return {"sub_phases": [], "checkpoint": None, "equivalence": None}
    report: dict[str, Any] = json.loads(path.read_text())
    return report


def _write_preparation_report(cfg: DataflowBuildConfig, report: dict[str, Any]) -> None:
    path = Path(cfg.output_dir) / PREPARATION_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


def _preparation(*stated: Any) -> GraphPreparation:
    """The graph-preparation options (finn.transformation.prepare.GraphPreparation) set
    from the preparation steps' options ``stated``; the rest keep their defaults."""
    return GraphPreparation(
        **{
            item.name: getattr(each, item.name)
            for each in stated
            for item in dataclasses.fields(each)
        }
    )


def _sub_phase(
    run: SubPhase,
    model: ModelWrapper,
    cfg: DataflowBuildConfig,
    options: GraphPreparation,
    fresh: bool = False,
) -> ModelWrapper:
    """Run one preparation sub-phase, ``run``, on ``model`` with ``options``, record the
    nodes it added and removed in the report under its name, and log a one-line summary.
    ``fresh`` starts the report afresh."""
    name = next(name for name, sub_phase in SUB_PHASES if sub_phase is run)
    before = copy.deepcopy(model)
    model = run(model, options)
    changed = census(before, model)
    report = _preparation_report(cfg, fresh)
    report["sub_phases"].append({"sub_phase": name, **changed})
    _write_preparation_report(cfg, report)
    counts = ", ".join(
        f"{sign}{count} {op}"
        for sign, side in (("-", "removed"), ("+", "added"))
        for op, count in changed[side].items()
    )
    _LOG.info(f"Graph preparation: {name}: {counts or 'no node changed'}")
    return model


@dataclass(frozen=True)
class PrepareImportOptions:
    """Options of step_prepare_import. ``override_inpsize``: the batch size (an int) or
    the whole input shape (a list) for qonnx's cleanup to set; None keeps the export's."""

    override_inpsize: int | list[int] | None = None


def step_prepare_import(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run qonnx's cleanup on the export, overriding its input size if the options ask.
    Starts report/graph_preparation.json afresh, and saves the export as the reference
    when GRAPH_PREPARATION_PYTHON is configured."""
    options = cfg.options("step_prepare_import", PrepareImportOptions)
    if VerificationStepType.GRAPH_PREPARATION_PYTHON in cfg.verify_steps:
        os.makedirs(os.path.dirname(_preparation_source(cfg)), exist_ok=True)
        model.save(_preparation_source(cfg))
    return _sub_phase(imported, model, cfg, _preparation(options), fresh=True)


@dataclass(frozen=True)
class PrepareQuantizationOptions:
    """Options of step_prepare_quantization. ``max_multithreshold_bit_width``: the
    widest Quant activation lowered to a MultiThreshold; a wider one stays a Quant.
    step_prepare_io lowers the preprocessing model with the same limit."""

    max_multithreshold_bit_width: int = 8


def step_prepare_quantization(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Lower Quant nodes: activations to MultiThresholds, weights to integers."""
    options = cfg.options("step_prepare_quantization", PrepareQuantizationOptions)
    return _sub_phase(quantization_lowered, model, cfg, _preparation(options))


@dataclass(frozen=True)
class PrepareIoOptions:
    """Options of step_prepare_io: the network's inputs and outputs. By default, the
    export's own.

    - ``preprocessing``: an ONNX model (a QONNX export, such as one of
      ``finn.util.pytorch.ToTensor``) merged ahead of the network; its input becomes
      the graph's input.
    - ``input_datatype``: the qonnx name of the graph input's datatype, the range of
      values the network is prepared for. None leaves the input unannotated, which is
      read as FLOAT32.
    - ``topk``: append a TopK that selects the ``topk`` largest outputs; its indices
      become the graph's output."""

    preprocessing: str | None = None
    input_datatype: str | None = None
    topk: int | None = None

    def __post_init__(self) -> None:
        _preparation(self)


def step_prepare_io(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Merge the preprocessing model (lowered with step_prepare_quantization's bit-width
    limit), set the input datatype and append the TopK, as the options say."""
    options = cfg.options("step_prepare_io", PrepareIoOptions)
    quantization = cfg.options("step_prepare_quantization", PrepareQuantizationOptions)
    return _sub_phase(io_explicit, model, cfg, _preparation(options, quantization))


@dataclass(frozen=True)
class PrepareStreamliningOptions:
    """Options of step_prepare_streamlining. ``streamlining``: the recipe, names from
    finn.transformation.prepare.RECIPE_TRANSFORMS, run in order."""

    streamlining: list[str] = field(default_factory=lambda: list(DEFAULT_STREAMLINING))

    def __post_init__(self) -> None:
        _preparation(self)


def step_prepare_streamlining(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run the streamlining recipe."""
    options = cfg.options("step_prepare_streamlining", PrepareStreamliningOptions)
    return _sub_phase(streamlined, model, cfg, _preparation(options))


@dataclass(frozen=True)
class PrepareTopologyOptions:
    """Options of step_prepare_topology. ``topology``: the recipe, names from
    finn.transformation.prepare.RECIPE_TRANSFORMS, run in order."""

    topology: list[str] = field(default_factory=lambda: list(DEFAULT_TOPOLOGY))

    def __post_init__(self) -> None:
        _preparation(self)


def step_prepare_topology(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run the topology recipe and infer the data layouts."""
    options = cfg.options("step_prepare_topology", PrepareTopologyOptions)
    return _sub_phase(topology_settled, model, cfg, _preparation(options))


def step_prepare_containers(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Widen to float64 the integer regions float32 cannot hold exactly, with Casts at
    their edges; the graph's inputs and outputs keep their types. No options."""
    return _sub_phase(containers_exact, model, cfg, GraphPreparation())


def step_prepare_checkpoint(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Check the prepared graph before the kernel path converts it: its structure, that
    its integer annotations are sound and its containers exact, and what would stay on
    the host (float nodes, Transposes, host nodes between KernelOps). Writes the result
    into the preparation report, logs one line per finding code, and refuses the graph
    on any blocker.

    With GRAPH_PREPARATION_PYTHON, also compare the graph with the export on seeded
    random inputs. Differences are reported, not refused, because a build cannot tell an
    intended deviation from a bug: the log prints FAIL and the declared deviations that
    might explain them, and the build continues. A value outside its tensor's annotated
    range does refuse the graph, because the kernels' bit widths trust that annotation.
    Where the export itself rounds (integers that float32 cannot hold exactly), the
    report lists each tensor."""
    checked = checkpoint(model, set(kernel_ops_by_anchor()))
    blockers = list(checked.blockers)
    report = _preparation_report(cfg)
    report["checkpoint"] = {
        "bounded": checked.bounded,
        "unbounded": dict(checked.unbounded),
        "findings": [finding_record(finding) for finding in checked.findings],
    }
    if VerificationStepType.GRAPH_PREPARATION_PYTHON in cfg.verify_steps:
        source_file = _preparation_source(cfg)
        if not os.path.isfile(source_file):
            raise FileNotFoundError(
                f"{source_file}: the export, which step_prepare_import keeps when "
                "GRAPH_PREPARATION_PYTHON is asked, is not there (did the build start after it?)"
            )
        input_name = model.graph.input[0].name
        if not (
            model.has_tensor_datatype(input_name)
            and model.get_tensor_datatype(input_name).is_integer()
        ):
            raise ValueError(
                f"graph_preparation_python draws from the prepared input's annotation, and "
                f"{input_name} states no integer one: state step_prepare_io's input_datatype"
            )
        # The reference is the export after the import and I/O steps only, with their
        # options (and the preprocessing model lowered with the quantization step's limit).
        options = _preparation(
            cfg.options("step_prepare_import", PrepareImportOptions),
            cfg.options("step_prepare_quantization", PrepareQuantizationOptions),
            cfg.options("step_prepare_io", PrepareIoOptions),
        )
        equivalence = check_equivalence(reference(ModelWrapper(source_file), options), model)
        report["equivalence"] = {
            "draws": equivalence.draws,
            "findings": [finding_record(finding) for finding in equivalence.findings],
            "export_rounds": [finding_record(finding) for finding in equivalence.rounds],
        }
        unsound = [f for f in equivalence.findings if f.code == "annotation-unsound"]
        blockers += unsound
    _write_preparation_report(cfg, report)
    _LOG.info(
        f"Graph preparation: checkpoint: {len(checked.findings)} findings, {checked.bounded} "
        f"integer outputs bounded; {PREPARATION_REPORT}"
    )
    for line in by_code(checked.findings):
        _LOG.info(f"Graph preparation:   {line}")
    if VerificationStepType.GRAPH_PREPARATION_PYTHON in cfg.verify_steps:
        _verdict("graph_preparation_python", not equivalence.findings)
        if equivalence.findings:
            declared = ", ".join(DEVIATIONS)
            refused = "annotation-unsound refuses the graph, the rest " if unsound else ""
            _LOG.warning(
                f"Graph preparation: equivalence: {len(equivalence.findings)} findings on "
                f"{equivalence.draws} draws, {refused}reported, not refused; a build runs no "
                f"deviation's predicate, and the phase declares {declared}"
            )
            for line in by_code(equivalence.findings):
                _LOG.warning(f"Graph preparation:   {line}")
        for line in by_code(equivalence.rounds):
            _LOG.info(f"Graph preparation:   {line}")
    if blockers:
        raise PreparationRefused(blockers)
    return model


def _kernel_path_source(cfg: DataflowBuildConfig) -> str:
    """Where step_convert_to_hw saves the model it converts, the reference for the
    PARTITION_PYTHON check."""
    return cfg.output_dir + "/intermediate_models/kernel_path_source.onnx"


def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Record the build's target in the model, and convert each node a KernelOp
    supports (MatMul, MultiThreshold) into that KernelOp. Writes report/kernel_ops.json:
    what each node became, or why it stayed on the host. Refuses a host node between
    KernelOps, since they must form one partition. With PARTITION_PYTHON, the model
    before conversion is saved as that check's reference."""
    if VerificationStepType.PARTITION_PYTHON in cfg.verify_steps:
        os.makedirs(os.path.dirname(_kernel_path_source(cfg)), exist_ok=True)
        model.save(_kernel_path_source(cfg))
    conversion = ToKernelOps(cfg._resolve_target())
    model = model.transform(conversion)
    report = kernel_ops_report(model, conversion.outcomes)
    os.makedirs(cfg.output_dir + "/report", exist_ok=True)
    with open(cfg.output_dir + "/report/kernel_ops.json", "w") as f:
        json.dump(report, f, indent=2)
    for line in kernel_ops_summary(conversion.outcomes):
        _LOG.info(line)
    refuse_host_between(model, conversion.outcomes)
    return model


@dataclass(frozen=True)
class ExploreOptions:
    """Options of step_explore.

    ``kernel_exploration``: the strategies that set the KernelOps' choices, run in
    order, each a spec with its own parameters
    (finn.transformation.kernels.KERNEL_STRATEGIES):

    - ``{"strategy": "pinned", "path": ...}``: the choices in a kernel_choices.json;
    - ``{"strategy": "target_throughput", "fps": ..., "relax": true}``: the least
      parallelism that meets ``fps`` at the target's clock;
    - ``{"strategy": "max_throughput", "within": {"lut": 0.5}}``: the fastest folding
      whose shell's resources stay within the given fractions of the part
      (``within`` is required);
    - ``{"strategy": "size_fifos", "margin": 0}``: each channel's FIFO sized from the
      beat patterns at both its ends;
    - ``{"strategy": "baseline"}``: every choice not pinned set back to its baseline;
    - ``{"strategy": "placeholder"}``: for debugging, every open required choice set
      to its first allowed value.

    Every choice starts at its baseline, set when it is created (a KernelOp's at
    conversion, a channel's at the cut). Each strategy replaces the values it decides,
    and every value is saved. Pins go first, and FIFO sizing after the last strategy
    that changes the folding; a list that folds after its last size_fifos is refused.
    To start over from the baselines, put ``{"strategy": "baseline"}`` first.

    The default, ``[{"strategy": "size_fifos"}]``, keeps the baseline folding and sizes
    its FIFOs. A list you give replaces the default and runs as written: without
    size_fifos, every channel keeps its baseline, a direct connection. The partition is
    packaged and built from exactly the saved values, and a required choice no strategy
    set is refused there, by name."""

    kernel_exploration: list[dict[str, Any]] = field(
        default_factory=lambda: [{"strategy": "size_fifos"}]
    )


def step_explore(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Run the strategies the options list (``kernel_exploration``) over the partition's
    KernelOps, in order, and save every value in the partition's body: a kernel's on its
    node, a channel's on its tensor. Each strategy carries its own objective; what no
    strategy decides keeps its baseline. Writes report/kernel_exploration.json (what each
    strategy changed, who set each value, the choices left open, and the limitations
    found) and kernel_choices.json (every choice by node or tensor name, which the
    ``pinned`` strategy reads back). A limitation, such as the shell's resources
    exceeding the part's, is logged as a warning, never an error."""
    options = cfg.options("step_explore", ExploreOptions)
    strategies = [strategy(spec) for spec in options.kernel_exploration]
    exploring = ExplorePartition(strategies)
    model = model.transform(exploring)
    explored, body = exploring.explored, exploring.body
    assert explored is not None and body is not None
    os.makedirs(cfg.output_dir + "/report", exist_ok=True)
    with open(os.path.join(cfg.output_dir, EXPLORATION_REPORT), "w") as f:
        json.dump(explored.report, f, indent=2)
    with open(cfg.output_dir + "/kernel_choices.json", "w") as f:
        json.dump(kernel_choices_config(body), f, indent=2)
    report = explored.report
    for each in report["strategies"]:
        _LOG.info(f"Kernel choices: {each['strategy']} changed {len(each['changed'])}")
        if each["strategy"] == "max_throughput":
            reached = each["bottleneck"]["cycles"]
            _LOG.info(
                f"Kernel choices: max_throughput reached {reached} cycles a frame within "
                f"{each['budget']} ({'fits' if each['fits'] else 'does not fit'}; binding: "
                f"{each['binding']}; {len(each['tried'])} budgets tried)"
            )
    for finding in explored.findings:
        _LOG.warning(f"Kernel choices: {finding.message}")
    for node, held in report["choices"].items():
        for attribute, entry in held.items():
            if str(entry["by"]).startswith("DEBUG"):
                _LOG.info(f"{entry['by']}: {node}.{attribute} = {entry['value']!r}")
    if report["open"]:
        _LOG.info(f"Kernel choices: open, refused at packaging: {', '.join(report['open'])}")
    _LOG.info(f"FIFOs: {report['fifos']}")
    return model


def _partition_directory(cfg: DataflowBuildConfig) -> Path:
    """Where the build keeps the partition's body (``partition.onnx``) and the scratch
    models the shell's integration generates the ends' IPs from (``ends/``)."""
    return Path(cfg.output_dir) / "partition"


def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Cut the KernelOps into one StreamingDataflowPartition, leaving the nodes before
    and after it on the host. From here on the build's model is the parent graph. The
    partition is named ``partition``: its node, its body (partition/partition.onnx), its
    IP and its block-design instance. Its body carries the target and every choice at
    its baseline. Each channel's choices are created here, and each KernelOp's are
    checked again in context: one that is stale because the model was edited since
    conversion is reset to its baseline, with a warning saying why."""
    cut = CutKernelPartition(_partition_directory(cfg))
    model = model.transform(cut)
    for key, why in cut.dropped.items():
        _LOG.warning(f"Kernel choices: {key} dropped as stale at the cut and re-baselined: {why}")
    return model


def _verify_partition_python(model: ModelWrapper, cfg: DataflowBuildConfig) -> bool:
    """Execute the parent graph ``model``, its partition node running the KernelOps, on
    each input in ``verify_input_npy``, and compare the partition's outputs value for
    value with those of the model saved before conversion (step_convert_to_hw). Saves
    each input's partition outputs as
    verification_output/verify_kernel_partition_python_<index>_<SUCCESS|FAIL>.npz, and
    returns whether all matched."""
    source_file = _kernel_path_source(cfg)
    if not os.path.isfile(source_file):
        raise FileNotFoundError(
            f"{source_file}: the kernel path's source, which step_convert_to_hw keeps when "
            "PARTITION_PYTHON is asked, is not there (did the build start after it?)"
        )
    verify_out_dir = cfg.output_dir + "/verification_output"
    os.makedirs(verify_out_dir, exist_ok=True)
    node, _, _ = partition_body(model)
    source = ModelWrapper(source_file)
    source_input = source.graph.input[0].name
    shape = source.get_tensor_shape(source_input)
    if shape is None:
        raise ValueError(
            f"kernel_partition_python: the source's input {source_input} states no shape"
        )
    ishape = tuple(shape)
    outputs = list(node.output)
    inputs = np.load(cfg.verify_input_npy)
    all_match = True
    for index in range(inputs.shape[0]):
        frame = inputs[index : index + 1].reshape((1,) + ishape[1:])
        built = execute_onnx(model, {model.graph.input[0].name: frame}, True)
        expected = execute_onnx(source, {source_input: frame}, True)
        mismatched = [name for name in outputs if not np.array_equal(built[name], expected[name])]
        for name in mismatched:
            _LOG.warning(
                f"kernel_partition_python: input {index}: {name} differs from the source's in "
                f"{int(np.sum(built[name] != expected[name]))} of {built[name].size} values"
            )
        result = "FAIL" if mismatched else "SUCCESS"
        all_match = all_match and not mismatched
        np.savez(
            f"{verify_out_dir}/verify_kernel_partition_python_{index}_{result}.npz",
            **{name: built[name] for name in outputs},
        )
    _LOG.info(
        f"kernel_partition_python: the partition's outputs {outputs} on {inputs.shape[0]} "
        f"inputs of {cfg.verify_input_npy}, against the model the kernel path started from"
    )
    return all_match


def step_verify_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Check the partition before a shell builds it: its choices are complete and
    current, its ports match its graph's inputs and outputs in order (configured_root),
    and its target is the configuration's (refuse_drift). Then run the optional checks
    ``verify_steps`` asks for: PARTITION_PYTHON compares the partition's outputs with
    the model before conversion (_verify_partition_python), and PARTITION_ELABORATION
    compiles and elaborates its RTL in XSim."""
    node, body, _ = partition_body(model)
    configured_root(body, node.name)
    refuse_drift(read_target(body), cfg._resolve_target(), "build")
    if VerificationStepType.PARTITION_PYTHON in cfg.verify_steps:
        matched = _verify_partition_python(model, cfg)
        _verdict("kernel_partition_python", matched)
    if VerificationStepType.PARTITION_ELABORATION in cfg.verify_steps:
        directory = cfg.output_dir + "/verification_output/kernel_partition_elaboration"
        body.transform(
            ElaboratePartition(
                directory=Path(directory),
                toolchain=_tools(cfg, "step_verify_partition"),
            )
        )
        _verdict("kernel_partition_elaboration", True)
    return model


def _refuse_drift(
    model: ModelWrapper, cfg: DataflowBuildConfig, output: DataflowOutputType
) -> None:
    """Refuse to make ``output`` if the partition's target is not the configuration's."""
    _, body, _ = partition_body(model)
    refuse_drift(read_target(body), cfg._resolve_target(), f"{output.value} build")


def step_create_stitched_ip(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Package the partition as an IP named ``partition`` into ``stitched_ip/``, with its
    interface description (``interface.json``), if STITCHED_IP or OOC_SYNTH is asked.
    The partition's body records the IP, so a shell's integration later in the build
    reuses it instead of packaging again.

    With OOC_SYNTH, the IP is synthesized out of context, and the model's reports list
    its utilization report (``ooc_synth``) for report/resources.json. With STITCHED_IP,
    an XSim testbench is written into ``stitched_ip/testbench``
    (finn.builder.build_dataflow_testbench). It streams the partition's inputs for the
    first frame of ``verify_input_npy`` if that file exists, else a generated frame. The
    testbench is run only if STITCHED_IP_TESTBENCH is in ``verify_steps``, and the step
    then fails (``SimulationFailed``) if the module's outputs differ from the
    partition's in Python."""
    stitched = DataflowOutputType.STITCHED_IP in cfg.generate_outputs
    ooc = DataflowOutputType.OOC_SYNTH in cfg.generate_outputs
    if not (stitched or ooc):
        _LOG.info("STITCHED_IP and OOC_SYNTH not in requested outputs, skipping.")
        return model
    directory = Path(cfg.output_dir) / "stitched_ip"
    toolchain = _tools(cfg, "step_create_stitched_ip")
    node, body, body_file = partition_body(model)
    body = body.transform(
        PackagePartition(
            node.name,
            run_synth=ooc,
            directory=directory,
            toolchain=toolchain,
        )
    )
    body.save(body_file)
    _LOG.info(f"Packaged IP {node.name} and its interface description in {directory}")
    if ooc:
        report = partition_utilization_report(directory)
        model.set(OUTPUT_REPORTS, {**(model.get(OUTPUT_REPORTS) or {}), "ooc_synth": str(report)})
    if stitched:
        if os.path.isfile(cfg.verify_input_npy):
            frame = partition_frame(model, np.load(cfg.verify_input_npy)[0])
            stimulus = f"the first input of {cfg.verify_input_npy}"
        else:
            frame, stimulus = generated_frame(body), "a generated frame"
        testbench = directory / TESTBENCH_DIR
        write_testbench(body, testbench, frame, toolchain=toolchain, label=node.name)
        _LOG.info(f"XSim testbench on {stimulus} written into {testbench} (run.sh runs it)")
        if VerificationStepType.STITCHED_IP_TESTBENCH in cfg.verify_steps:
            run_testbench(testbench, toolchain=toolchain)
            _verdict("stitched_ip_testbench", True)
    return model


def step_synthesize_bitfile(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Build the target's shell around the partition to a bitfile in the output
    directory, if BITFILE is asked. Only the pynq shell has a bitfile build
    (BuildPynqBitfile); ``enable_hw_debug`` is its option. Each tool's transcript goes
    into logs/step_synthesize_bitfile. Refuses a partition built for another target than
    the configuration's. The model records what was made (finn.outputs) and what the
    bitfile was built from. Any shortfall of the delivered clock is logged as a
    warning."""
    if DataflowOutputType.BITFILE not in cfg.generate_outputs:
        _LOG.info("BITFILE not in requested outputs, skipping step_synthesize_bitfile.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.BITFILE)
    building = BuildPynqBitfile(
        Path(cfg.output_dir),
        toolchain=_tools(cfg, "step_synthesize_bitfile"),
        enable_hw_debug=cfg.enable_hw_debug,
    )
    model = model.transform(building)
    for finding in building.findings:
        _LOG.warning(finding.message)
    _LOG.info(f"Bitfile written into {building.bitfile.parent}")
    return model


def step_make_driver(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write the PYNQ driver into driver/ and its description into report/driver.json,
    if PYNQ_DRIVER is asked. The driver sets PL0 to the clock the bitfile actually
    delivers (report/delivered_clock.json), not the one requested, and loads the
    build's bitfile by default. Refuses a bitfile built from another design than the
    current partition, and a partition built for another target than the
    configuration's. The model's reports list the description."""
    if DataflowOutputType.PYNQ_DRIVER not in cfg.generate_outputs:
        _LOG.info("PYNQ_DRIVER not in requested outputs, skipping step_make_driver.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.PYNQ_DRIVER)
    making = MakePynqDriver(Path(cfg.output_dir))
    model = model.transform(making)
    description = making.description
    returned = ", ".join(
        f"{each['name']} ({each['element']}, {each['shape']})" for each in description["returns"]
    )
    driver_dir = os.path.join(cfg.output_dir, DRIVER_DIR)
    _LOG.info(
        f"PYNQ Python driver written into {driver_dir}, at {description['fclk_mhz']} MHz; "
        f"it returns {returned}"
    )
    return model


#: The finn.outputs keys a deployed model drops: they hold paths on the build machine.
_MACHINE_PATHS = (OUTPUT_IP, OUTPUT_PROJECT, OUTPUT_BITFILE, OUTPUT_HWH, OUTPUT_REPORTS)


def _host_model(model: ModelWrapper, directory: Path) -> Path:
    """Write the parent graph ``model`` into ``directory`` as the deployment ships it:
    parent.onnx (the host's nodes and the partition node) and partition.onnx beside it.
    The node refers to its body by a relative path, and neither model keeps a path on
    the build machine (_MACHINE_PATHS). Returns the path of parent.onnx."""
    directory.mkdir(parents=True, exist_ok=True)
    node, body, _ = partition_body(model)
    shipped = ModelWrapper(copy.deepcopy(model.model))
    shipped_body = ModelWrapper(copy.deepcopy(body.model))
    for each in (shipped, shipped_body):
        for key in _MACHINE_PATHS:
            each.delete(key)
    (shipped_node,) = [each for each in shipped.graph.node if each.name == node.name]
    getCustomOp(shipped_node).set_nodeattr("model", "partition.onnx")
    shipped_body.save(str(directory / "partition.onnx"))
    shipped.save(str(directory / "parent.onnx"))
    return directory / "parent.onnx"


def step_deployment_package(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Package the bitfile, the driver, the host-side model (deploy/model/) and the
    driver's description (deploy/driver.json) into deploy/, if DEPLOYMENT_PACKAGE is
    asked; deploy/ is written afresh. Ships the bitfile and driver this output
    directory's own steps made: refuses a model that records neither, or records ones
    from another directory, or a bitfile built from another design than the current
    partition."""
    if DataflowOutputType.DEPLOYMENT_PACKAGE not in cfg.generate_outputs:
        _LOG.info("DEPLOYMENT_PACKAGE not in requested outputs, skipping.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.DEPLOYMENT_PACKAGE)
    pynq_row(model, "deployment_package")
    output = Path(cfg.output_dir)
    made = {
        "bitfile": (model.get(OUTPUT_BITFILE), output / BITFILE),
        "pynq_driver": ((model.get(OUTPUT_REPORTS) or {}).get("driver"), output / DRIVER_REPORT),
    }
    for name, (stated, expected) in made.items():
        if stated is None:
            raise ValueError(
                f"deployment_package: the model states no {name} (finn.outputs): the "
                "deployment ships the bitfile and the driver its build made; ask bitfile "
                "and pynq_driver with it"
            )
        if Path(stated) != expected:
            raise ValueError(
                f"deployment_package: the model's {name} is {stated}, not this output "
                f"directory's ({expected})"
            )
    refuse_stale_bitfile(model, "deployment_package")
    deploy = output / "deploy"
    shutil.rmtree(deploy, ignore_errors=True)
    shutil.copytree(output / BITFILE.parent, deploy / BITFILE.parent)
    shutil.copytree(output / DRIVER_DIR, deploy / DRIVER_DIR, copy_function=shutil.copyfile)
    _host_model(model, deploy / "model")
    shutil.copy(output / DRIVER_REPORT, deploy / "driver.json")
    return model


def step_resources_report(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write report/resources.json: the resources of each member of the shell (the
    partition, each end, each IP of the static region) and their total, as modelled, as
    synthesized out of context and as placed, whichever the build produced
    (finn.builder.build_dataflow_resources). The model's reports list it
    (``resources``)."""
    report = Path(cfg.output_dir) / RESOURCES_FILE
    report.parent.mkdir(parents=True, exist_ok=True)
    stated = shell_resources_report(model)
    report.write_text(json.dumps(stated, indent=2))
    model.set(OUTPUT_REPORTS, {**(model.get(OUTPUT_REPORTS) or {}), "resources": str(report)})
    absent = ", ".join(f"{column} ({why})" for column, why in stated["absent"].items())
    _LOG.info(
        f"Resources per member written to {report}" + (f"; absent: {absent}" if absent else "")
    )
    return model


#: The build's steps by name, for a configuration's ``steps``.
build_dataflow_step_lookup = {
    step.__name__: step
    for step in (
        step_prepare_import,
        step_prepare_quantization,
        step_prepare_io,
        step_prepare_streamlining,
        step_prepare_topology,
        step_prepare_containers,
        step_prepare_checkpoint,
        step_convert_to_hw,
        step_create_dataflow_partition,
        step_explore,
        step_verify_partition,
        step_create_stitched_ip,
        step_synthesize_bitfile,
        step_make_driver,
        step_resources_report,
        step_deployment_package,
    )
}

#: The options type of each step that has options, by step name, for a configuration's
#: ``step_options``.
build_dataflow_step_options: dict[str, type] = {
    "step_prepare_import": PrepareImportOptions,
    "step_prepare_quantization": PrepareQuantizationOptions,
    "step_prepare_io": PrepareIoOptions,
    "step_prepare_streamlining": PrepareStreamliningOptions,
    "step_prepare_topology": PrepareTopologyOptions,
    "step_explore": ExploreOptions,
}
