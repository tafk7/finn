# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The kernel path's steps, each a ``(model, DataflowBuildConfig) -> model`` function,
registered by name in ``build_dataflow_step_lookup`` for a DataflowBuildConfig's ``steps``
(finn.builder.build_dataflow_config). Its phases (finn.builder.build_dataflow_phases) name
them in order. A step with options of its own declares their type, a dataclass, beside
it, and registers it by the step's name in ``build_dataflow_step_options``: a
DataflowBuildConfig's ``step_options`` states them, and the step reads its own
(``DataflowBuildConfig.options``).

``phase_prepare_model`` takes a Brevitas export to the streamlined graph the kernel
path converts, by the sub-phases of finn.transformation.prepare, and checks what it
leaves before anything converts it (its checkpoint, P7). ``phase_convert_to_hardware``,
``phase_create_accelerator`` and ``phase_optimize_hardware`` take that graph to a
partition of KernelOps
(finn.custom_op.kernels): the KernelOps bind kernels to RTL from the model's facts and
the build's target, so there is no specialization, folding config or per-node IP
generation. ``phase_build_hardware`` and ``phase_generate_outputs`` then make the
outputs the configuration asks:
the partition's own on any shell (its packaged IP, interface description and testbench;
its out-of-context resources), then the target's shell's (its bitfile, driver and
deployment package)."""

import copy
import dataclasses
import json
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
from finn.core.space import finding_record
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
from finn.kernels.explore import Completion
from finn.platform import refuse_drift
from finn.shells.pynq.driver import DRIVER_DIR, DRIVER_REPORT, MakePynqDriver
from finn.shells.pynq.runner import BITFILE, BuildPynqBitfile, pynq_row, refuse_stale_bitfile
from finn.toolchain.launcher import Toolchain
from finn.transformation.kernels import (
    EXPLORATION_REPORT,
    ExplorePartition,
    ToKernelOps,
    completion,
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
    summary,
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

#: The graph-preparation phase's report, under the output directory.
PREPARATION_REPORT = "report/graph_preparation.json"


def _tools(cfg: DataflowBuildConfig, step: str) -> Toolchain:
    """The build's toolchain for ``step``: every tool it runs writes its transcript into
    ``output_dir/logs/<step>`` (``Toolchain.transcripts_in``), passed down with the
    toolchain to each transformation and driver that runs one."""
    return cfg._resolve_toolchain().transcripts_in(Path(cfg.output_dir) / "logs" / step)


def _preparation_source(cfg: DataflowBuildConfig) -> str:
    """Where step_prepare_import keeps the export, the reference of the phase's
    equivalence check."""
    return cfg.output_dir + "/intermediate_models/graph_preparation_source.onnx"


def _preparation_report(cfg: DataflowBuildConfig, fresh: bool = False) -> dict[str, Any]:
    """The phase's report so far, as written; an empty one when ``fresh`` or none is."""
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
    """The graph-preparation phase's options (finn.transformation.prepare.GraphPreparation)
    as the preparation steps' options ``stated`` set them, every other at its default."""
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
    """``model`` through the sub-phase ``run`` with ``options``, what it changed recorded
    in the phase's report under the sub-phase's name (finn.transformation.prepare:
    SUB_PHASES, census) and logged."""
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
    print(f"Graph preparation: {name}: {counts or 'no node changed'}")
    return model


@dataclass(frozen=True)
class PrepareImportOptions:
    """step_prepare_import's options. ``override_inpsize``: the batch size (an int) or
    the input's shape (a list) qonnx's cleanup sets; None keeps the export's."""

    override_inpsize: int | list[int] | None = None


def step_prepare_import(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """P0: the export through qonnx's cleanup (finn.transformation.prepare.phase.imported),
    its input size overridden if its options ask (PrepareImportOptions). Starts
    report/graph_preparation.json afresh. When GRAPH_PREPARATION_PYTHON is asked, the
    export is kept as that check's reference (graph_preparation_source.onnx)."""
    options = cfg.options("step_prepare_import", PrepareImportOptions)
    if VerificationStepType.GRAPH_PREPARATION_PYTHON in cfg.verify_steps:
        os.makedirs(os.path.dirname(_preparation_source(cfg)), exist_ok=True)
        model.save(_preparation_source(cfg))
    return _sub_phase(imported, model, cfg, _preparation(options), fresh=True)


@dataclass(frozen=True)
class PrepareQuantizationOptions:
    """step_prepare_quantization's options. ``max_multithreshold_bit_width``: the
    widest Quant activation lowered to a MultiThreshold; a wider one stays a Quant.
    step_prepare_io lowers the preprocessing model by the same width."""

    max_multithreshold_bit_width: int = 8


def step_prepare_quantization(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """P2: Quant activations to MultiThresholds and weights to integers, as finn-dev's
    step_qonnx_to_finn and step_tidy_up
    (finn.transformation.prepare.phase.quantization_lowered), by its options
    (PrepareQuantizationOptions)."""
    options = cfg.options("step_prepare_quantization", PrepareQuantizationOptions)
    return _sub_phase(quantization_lowered, model, cfg, _preparation(options))


@dataclass(frozen=True)
class PrepareIoOptions:
    """step_prepare_io's options, the network's inputs and outputs as the build states
    them; by default the export's own.

    - ``preprocessing``: an ONNX model (a QONNX export, as of
      ``finn.util.pytorch.ToTensor``) merged ahead of the network; its input becomes
      the graph's.
    - ``input_datatype``: the graph input's datatype (a qonnx name), the domain the
      network is prepared for; None leaves the input unannotated, read as FLOAT32.
    - ``topk``: a label select of the ``topk`` largest outputs appended (TopK, its
      indices the graph's output)."""

    preprocessing: str | None = None
    input_datatype: str | None = None
    topk: int | None = None

    def __post_init__(self) -> None:
        _preparation(self)


def step_prepare_io(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """P1: the network's inputs and outputs as its options state them (PrepareIoOptions):
    its preprocessing model merged ahead, lowered by step_prepare_quantization's width,
    its input datatype, its label select (finn.transformation.prepare.phase.io_explicit)."""
    options = cfg.options("step_prepare_io", PrepareIoOptions)
    quantization = cfg.options("step_prepare_quantization", PrepareQuantizationOptions)
    return _sub_phase(io_explicit, model, cfg, _preparation(options, quantization))


@dataclass(frozen=True)
class PrepareStreamliningOptions:
    """step_prepare_streamlining's options. ``streamlining``: its recipe, names of
    finn.transformation.prepare.RECIPE_TRANSFORMS run in order."""

    streamlining: list[str] = field(default_factory=lambda: list(DEFAULT_STREAMLINING))

    def __post_init__(self) -> None:
        _preparation(self)


def step_prepare_streamlining(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """P3: the streamlining recipe (PrepareStreamliningOptions)."""
    options = cfg.options("step_prepare_streamlining", PrepareStreamliningOptions)
    return _sub_phase(streamlined, model, cfg, _preparation(options))


@dataclass(frozen=True)
class PrepareTopologyOptions:
    """step_prepare_topology's options. ``topology``: its recipe, names of
    finn.transformation.prepare.RECIPE_TRANSFORMS run in order."""

    topology: list[str] = field(default_factory=lambda: list(DEFAULT_TOPOLOGY))

    def __post_init__(self) -> None:
        _preparation(self)


def step_prepare_topology(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """P4: the topology recipe (PrepareTopologyOptions), layouts inferred."""
    options = cfg.options("step_prepare_topology", PrepareTopologyOptions)
    return _sub_phase(topology_settled, model, cfg, _preparation(options))


def step_prepare_containers(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """P6: every integer tensor held exactly by its container, the integer regions that
    need it widened to float64, Casts at their edges, the graph's inputs and outputs
    unchanged (finn.transformation.prepare.containers). It has no options."""
    return _sub_phase(containers_exact, model, cfg, GraphPreparation())


def step_prepare_checkpoint(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """P7: the prepared graph checked before the kernel path converts it
    (finn.transformation.prepare.checkpoint): its structure, its integer annotations'
    soundness and its containers' exactness, by bound, against the KernelOps' anchors;
    what it leaves (float nodes, Transposes, host nodes predicted between KernelOps).
    Writes the report's checkpoint, logs one line per finding code, and refuses the
    graph on a checkpoint blocker, each named.

    With GRAPH_PREPARATION_PYTHON, also its equivalence with the export on seeded draws
    (finn.harness.preparation.check_equivalence), which the build has no deviation's
    predicate for: every difference is a finding. That check is a verification, as
    PARTITION_PYTHON is: its findings go to the report's equivalence, the log prints
    FAIL, one line per finding code and the declared deviations that might explain
    them, and the build continues. But a drawn value outside its tensor's annotation
    (annotation-unsound) is no difference from the export: the graph's own
    annotation is wrong, and the kernels' widths trust it, so it refuses the graph
    beside the checkpoint's blockers, named. Where the export itself rounds (its
    integers otherwise in float32 than in float64), the report's equivalence names each
    tensor (export_rounds, limitations)."""
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
        # The reference is the export through P0 and P1 alone, as those steps' options
        # state them (and the preprocessing model lowered by P2's width).
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
    print(
        f"Graph preparation: checkpoint: {len(checked.findings)} findings, {checked.bounded} "
        f"integer outputs bounded; {PREPARATION_REPORT}"
    )
    for line in summary(checked.findings):
        print(line)
    if VerificationStepType.GRAPH_PREPARATION_PYTHON in cfg.verify_steps:
        print(
            "Verification for graph_preparation_python : "
            + ("FAIL" if equivalence.findings else "SUCCESS")
        )
        if equivalence.findings:
            declared = ", ".join(DEVIATIONS)
            refused = "annotation-unsound refuses the graph, the rest " if unsound else ""
            print(
                f"Graph preparation: equivalence: {len(equivalence.findings)} findings on "
                f"{equivalence.draws} draws, {refused}reported, not refused; a build runs no "
                f"deviation's predicate, and the phase declares {declared}"
            )
            for line in summary(equivalence.findings):
                print(line)
        for line in summary(equivalence.rounds):
            print(line)
    if blockers:
        raise PreparationRefused(blockers)
    return model


def _kernel_path_source(cfg: DataflowBuildConfig) -> str:
    """Where step_convert_to_hw keeps the model it converts, the reference of the
    partition's Python check."""
    return cfg.output_dir + "/intermediate_models/kernel_path_source.onnx"


def step_convert_to_hw(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """State the build's target in the model (``cfg.target``, resolved) and rewrite
    each node a KernelOp binds (MatMul, MultiThreshold) and the kernels admit as one,
    every tensor inferred as it goes: ToKernelOps. When PARTITION_PYTHON is asked, the
    model it converts is kept as that check's reference (kernel_path_source.onnx).
    Writes report/kernel_ops.json (every node's outcome: the KernelOp it became, or
    why it stays on the host) and logs its summary; refuses host nodes between
    KernelOps, each named with its findings (refuse_host_between), which partitioning
    could not leave out."""
    if VerificationStepType.PARTITION_PYTHON in cfg.verify_steps:
        os.makedirs(os.path.dirname(_kernel_path_source(cfg)), exist_ok=True)
        model.save(_kernel_path_source(cfg))
    conversion = ToKernelOps(cfg._resolve_target())
    model = model.transform(conversion)
    report = kernel_ops_report(model, conversion.outcomes)
    os.makedirs(cfg.output_dir + "/report", exist_ok=True)
    with open(cfg.output_dir + "/report/kernel_ops.json", "w") as f:
        json.dump(report, f, indent=2)
    for line in kernel_ops_summary(report):
        print(line)
    refuse_host_between(model, conversion.outcomes)
    return model


@dataclass(frozen=True)
class ExploreOptions:
    """step_explore's options.

    - ``kernel_exploration``: the strategies that choose the KernelOps' open choices
      through the DSE seam, run as written, each a spec with its own parameters
      (finn.transformation.kernels.KERNEL_STRATEGIES): ``{"strategy": "pinned", "path":
      ...}`` (a kernel_choices.json), ``{"strategy": "target_throughput", "fps": ...,
      "relax": true}`` (the least parallelism meeting fps at the target's clock),
      ``{"strategy": "max_throughput", "within": {"lut": 0.5}}`` (the fewest cycles a
      frame whose shell's resources stay within the fractions named of the part's;
      ``within`` is required), ``{"strategy": "size_fifos", "margin": 0}`` (each
      channel's FIFO from both ends' beat patterns), ``{"strategy": "baseline"}`` (every
      choice not pinned back at its baseline), ``{"strategy": "placeholder"}``
      (debugging: every open required choice at its first case). Every choice the body
      leaves open takes its baseline first; each strategy replaces the values it
      decides, and every value is saved. Pins go first, sizing after the last strategy
      that folds (a list folding after its last size_fifos is refused). By default
      ``[{"strategy": "size_fifos"}]`` (SR6): the baseline folding, its FIFOs sized. A
      list written replaces the default and runs as written: without size_fifos, every
      transport stays at its baseline, direct.
    - ``kernel_exploration_fresh``: clear the KernelOps' saved choices before
      exploring: re-explore from scratch. Otherwise a saved choice is pinned and the
      exploration fills only open ones.
    - ``kernel_completion``: the completion policy
      (finn.transformation.kernels.KERNEL_COMPLETIONS) that completes what no strategy
      chose, on a copy that is never saved, wherever the partition is costed or built
      (every step after step_explore reads it here, ``_completion``): ``baseline``,
      every open choice at its kernel's baseline (its first viable case: the least
      parallelism, each memory in the explicit style its size orders first, a FIFO in
      FinnLib's ``auto``) and, when the partition is built, its FIFOs sized at that
      folding; a required choice (a FIFO's depth) is refused, named. ``placeholder``,
      for debugging, also takes the first case of a required choice.
      report/kernel_exploration.json lists every value completed."""

    kernel_exploration: list[dict[str, Any]] = field(
        default_factory=lambda: [{"strategy": "size_fifos"}]
    )
    kernel_exploration_fresh: bool = False
    kernel_completion: str = "baseline"


def _completion(cfg: DataflowBuildConfig) -> Completion:
    """The completion policy step_explore's options name (ExploreOptions), which every
    step that costs or builds the partition completes it by."""
    return completion(cfg.options("step_explore", ExploreOptions).kernel_completion)


def step_explore(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Explore the partition's KernelOps through the parent graph's partition node
    (finn.transformation.kernels.ExplorePartition), through the DSE seam by the strategies
    its options' ``kernel_exploration`` lists (ExploreOptions), in order
    (finn.transformation.kernels.strategy: each a spec, {"strategy": name, **parameters}),
    and save them where they are stated (a kernel's on its node, a channel's on its tensor,
    finn.channel), the body saved again in its file (explore_kernel_choices): the choices
    strategies made, never a completed one. What no strategy chose stays open, and its
    options' ``kernel_completion`` completes it wherever the partition is costed or built. A
    strategy carries its own objective. Writes report/kernel_exploration.json (the
    strategies, each with the choices it committed, attempts and time, and the completed
    values it read; every committed choice with the strategy that made it; the completion
    policy, every value it completes and the required choices it leaves open; whether FIFOs
    were sized; the dropped choices with why, per member cycles and buffering and the
    bottleneck, of the completed point, and the shell's resources against the part's) and
    kernel_choices.json (the body's choices, sparse, by graph name: a node's or a tensor's,
    which a "pinned" strategy reads back), and logs what each strategy committed (and where
    max_throughput's search ended), what the completion completed (each value, for the debug
    placeholder), whether FIFOs were sized, and a warning naming the binding resource where
    the shell's resources exceed the part's (RC5: never a refusal)."""
    options = cfg.options("step_explore", ExploreOptions)
    strategies = [strategy(spec) for spec in options.kernel_exploration]
    completing = completion(options.kernel_completion)
    exploring = ExplorePartition(
        strategies, fresh=options.kernel_exploration_fresh, completion=completing
    )
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
        print(f"Kernel choices: {each['strategy']} committed {each['committed']}")
        if each["strategy"] == "max_throughput":
            reached = each["bottleneck"]["cycles"]
            print(
                f"Kernel choices: max_throughput reached {reached} cycles a frame within "
                f"{each['budget']} ({'fits' if each['fits'] else 'does not fit'}; binding: "
                f"{each['binding']}; {len(each['tried'])} budgets tried)"
            )
        if "read_completed" in each:
            print(f"Kernel choices: {each['strategy']} {each['read_completed']}")
    if report["resources"]["warning"] is not None:
        print(f"Kernel choices: WARNING: {report['resources']['warning']}")
    completed = [
        (f"{node}.{attribute}", entry)
        for node, held in report["completed"].items()
        for attribute, entry in held.items()
    ]
    policy = report["completion"]
    print(f"Kernel choices: {policy['policy']} completes {len(completed)}")
    for name, entry in completed:
        if entry["by"].startswith("DEBUG"):
            print(f"{entry['by']}: {name} = {entry['value']!r}")
    if "refused" in policy:
        print(f"Kernel choices: the {policy['policy']} completion refuses: {policy['refused']}")
    if policy.get("open"):
        print(f"Kernel choices: open, to choose before packaging: {', '.join(policy['open'])}")
    print(f"FIFOs: {report['fifos']}")
    return model


def _partition_directory(cfg: DataflowBuildConfig) -> Path:
    """Where the kernel path keeps its partition's body (``partition.onnx``) and the
    scratch models its shell's integration generates its ends' IPs from (``ends/``)."""
    return Path(cfg.output_dir) / "partition"


def step_create_dataflow_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Cut the KernelOps once into one StreamingDataflowPartition, the nodes before and
    after it on the host (CutKernelPartition), before anything explores them. The parent
    graph is the build's model from here on; the partition is named ``partition`` (its
    node, its body partition/partition.onnx, its IP and its block-design instance), and
    its body carries the target."""
    return model.transform(CutKernelPartition(_partition_directory(cfg)))


def _verify_partition_python(model: ModelWrapper, cfg: DataflowBuildConfig) -> bool:
    """The partition's own outputs, for each input ``verify_input_npy`` states: the
    parent graph ``model`` executed, its partition node running its body of KernelOps,
    against the model the kernel path started from (step_convert_to_hw's
    kernel_path_source.onnx) on the same input, value for value. Each input's partition
    outputs are saved as
    verification_output/verify_kernel_partition_python_<index>_<SUCCESS|FAIL>.npz."""
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
            print(
                f"kernel_partition_python: input {index}: {name} differs from the source's in "
                f"{int(np.sum(built[name] != expected[name]))} of {built[name].size} values"
            )
        result = "FAIL" if mismatched else "SUCCESS"
        all_match = all_match and not mismatched
        np.savez(
            f"{verify_out_dir}/verify_kernel_partition_python_{index}_{result}.npz",
            **{name: built[name] for name in outputs},
        )
    print(
        f"kernel_partition_python: the partition's outputs {outputs} on {inputs.shape[0]} "
        f"inputs of {cfg.verify_input_npy}, against the model the kernel path started from"
    )
    return all_match


def step_verify_partition(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Check the kernel path's partition before a shell builds it, its body opened
    through the parent graph's partition node.

    Always: its choices replay to a configured root, completed by step_explore's
    ``kernel_completion`` (none stale, none left open, its ports its graph's
    inputs and outputs in order; configured_root), and it states the configuration's
    target (refuse_drift). As ``verify_steps`` asks: PARTITION_PYTHON compares the
    partition's own outputs, with the parent graph executed, against the model the
    kernel path started from, on each input verify_input_npy states
    (_verify_partition_python); PARTITION_ELABORATION compiles and elaborates
    the partition's emitted RTL in XSim, through the build's toolchain."""
    node, body, _ = partition_body(model)
    configured_root(body, node.name, _completion(cfg))
    refuse_drift(read_target(body), cfg._resolve_target(), "build")
    if VerificationStepType.PARTITION_PYTHON in cfg.verify_steps:
        matched = _verify_partition_python(model, cfg)
        print("Verification for kernel_partition_python : " + ("SUCCESS" if matched else "FAIL"))
    if VerificationStepType.PARTITION_ELABORATION in cfg.verify_steps:
        directory = cfg.output_dir + "/verification_output/kernel_partition_elaboration"
        body.transform(
            ElaboratePartition(
                directory=Path(directory),
                toolchain=_tools(cfg, "step_verify_partition"),
                completion=_completion(cfg),
            )
        )
        print("Verification for kernel_partition_elaboration : SUCCESS")
    return model


def _refuse_drift(
    model: ModelWrapper, cfg: DataflowBuildConfig, output: DataflowOutputType
) -> None:
    """Refuse, for ``output``, a parent graph ``model`` whose partition body states
    another target than the configuration's (refuse_drift)."""
    _, body, _ = partition_body(model)
    refuse_drift(read_target(body), cfg._resolve_target(), f"{output.value} build")


def step_create_stitched_ip(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Package the partition as an IP into ``stitched_ip/``, if STITCHED_IP or OOC_SYNTH
    is asked: PackagePartition on the body the parent graph's partition node opens,
    named as the partition (its node, ``partition``), its choices completed by
    step_explore's ``kernel_completion``, with the interface description beside it
    (``interface.json``). The body, saved, states its IP (finn.outputs), and a shell's
    integration later in the build takes that IP rather than package it again. With
    OOC_SYNTH the module is synthesized out of context and the IP packages the
    checkpoint; the parent graph states its hierarchical utilization report among its
    reports (``ooc_synth``), which report/resources.json reads. With STITCHED_IP, the XSim
    testbench is written into ``stitched_ip/testbench`` (finn.builder.build_dataflow_testbench),
    on the partition's inputs from the parent graph executed on the first frame of
    ``verify_input_npy`` when it exists, a generated frame otherwise. It is an output,
    and not run, unless STITCHED_IP_TESTBENCH is among ``verify_steps``: then its run.sh
    runs once, in the build's toolchain's environment, and the step fails
    (``SimulationFailed``) if the module's outputs differ from the partition's in
    Python."""
    stitched = DataflowOutputType.STITCHED_IP in cfg.generate_outputs
    ooc = DataflowOutputType.OOC_SYNTH in cfg.generate_outputs
    if not (stitched or ooc):
        print("STITCHED_IP and OOC_SYNTH not in requested outputs, skipping.")
        return model
    kernel_completion = _completion(cfg)
    directory = Path(cfg.output_dir) / "stitched_ip"
    toolchain = _tools(cfg, "step_create_stitched_ip")
    node, body, body_file = partition_body(model)
    body = body.transform(
        PackagePartition(
            node.name,
            run_synth=ooc,
            directory=directory,
            toolchain=toolchain,
            completion=kernel_completion,
        )
    )
    body.save(body_file)
    print(f"Packaged IP {node.name} and its interface description in {directory}")
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
        write_testbench(body, testbench, frame, completion=kernel_completion, label=node.name)
        print(f"XSim testbench on {stimulus} written into {testbench} (run.sh runs it)")
        if VerificationStepType.STITCHED_IP_TESTBENCH in cfg.verify_steps:
            run_testbench(testbench, toolchain=toolchain)
            print("Verification for stitched_ip_testbench : SUCCESS")
    return model


def step_synthesize_bitfile(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Build the shell the parent graph states (its partition's target) around the
    partition to a bitfile, if BITFILE is asked, into the output directory: the pynq
    shell's Vivado block design (finn.shells.pynq.runner.BuildPynqBitfile, which refuses
    a shell without that build), the partition completed by step_explore's
    ``kernel_completion``, every tool's transcript in logs/step_synthesize_bitfile, and
    ``cfg.enable_hw_debug`` its shell option. A parent graph stating another target than
    the configuration's is refused. The parent graph stays the build's model and states
    what was made (finn.outputs: the project, bitfile, hwh, reports and host runtime) and
    what the bitfile was made from (the design's digest, the shell's options); the
    delivered clock's warning, if any, is logged."""
    if DataflowOutputType.BITFILE not in cfg.generate_outputs:
        print("BITFILE not in requested outputs, skipping step_synthesize_bitfile.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.BITFILE)
    building = BuildPynqBitfile(
        Path(cfg.output_dir),
        toolchain=_tools(cfg, "step_synthesize_bitfile"),
        enable_hw_debug=cfg.enable_hw_debug,
        completion=_completion(cfg),
    )
    model = model.transform(building)
    if "warning" in building.clock:
        print("WARNING: " + building.clock["warning"])
    print(f"Bitfile written into {building.bitfile.parent}")
    return model


def step_make_driver(model: ModelWrapper, cfg: DataflowBuildConfig) -> ModelWrapper:
    """Write the driver the parent graph's shell's host runtime runs into driver/, and its
    description into report/driver.json, if PYNQ_DRIVER is asked: the PYNQ driver
    (finn.shells.pynq.driver.MakePynqDriver), PL0 set to the clock the bitfile delivers
    (report/delivered_clock.json), the bitfile the build ships (finn.outputs) its
    default, the partition completed by step_explore's ``kernel_completion``; a bitfile
    built from another design than the partition's now is refused, named, and so is a
    parent graph stating another target than the configuration's. The parent graph
    states the description among its reports."""
    if DataflowOutputType.PYNQ_DRIVER not in cfg.generate_outputs:
        print("PYNQ_DRIVER not in requested outputs, skipping step_make_driver.")
        return model
    _refuse_drift(model, cfg, DataflowOutputType.PYNQ_DRIVER)
    making = MakePynqDriver(Path(cfg.output_dir), completion=_completion(cfg))
    model = model.transform(making)
    description = making.description
    returned = ", ".join(
        f"{each['name']} ({each['element']}, {each['shape']})" for each in description["returns"]
    )
    driver_dir = os.path.join(cfg.output_dir, DRIVER_DIR)
    print(
        f"PYNQ Python driver written into {driver_dir}, at {description['fclk_mhz']} MHz; "
        f"it returns {returned}"
    )
    return model


#: The finn.outputs keys a deployed model does not carry: paths on the build's machine.
_MACHINE_PATHS = (OUTPUT_IP, OUTPUT_PROJECT, OUTPUT_BITFILE, OUTPUT_HWH, OUTPUT_REPORTS)


def _host_model(model: ModelWrapper, directory: Path) -> Path:
    """The parent graph ``model`` as the deployment ships it, into ``directory``: the
    host's nodes and the partition node (parent.onnx), the partition's body beside it
    (partition.onnx), the node's body path relative to the directory, and neither
    stating a path of the build's machine (_MACHINE_PATHS)."""
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
    """Package the bitfile and the driver for deployment, if DEPLOYMENT_PACKAGE is asked,
    with the host's part of the parent graph as a model (deploy/model/, _host_model) and
    the driver's description (deploy/driver.json, report/driver.json). It ships the
    bitfile and the driver the parent graph states (finn.outputs), which this output
    directory's bitfile and driver steps made: a model stating neither, or another
    directory's, or a bitfile built from another design than the partition's now
    (refuse_stale_bitfile), is refused by name, and deploy/ is written afresh."""
    if DataflowOutputType.DEPLOYMENT_PACKAGE not in cfg.generate_outputs:
        print("DEPLOYMENT_PACKAGE not in requested outputs, skipping.")
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
    """Write report/resources.json: the resources of each member of the partition's
    shell (the partition, each end, each IP of the static region) and their total, as
    the shell root states them (the model), as Vivado synthesized each IP out of
    context, and as placed, from the reports the parent graph states
    (finn.builder.build_dataflow_resources); a column the build did not make is stated absent.
    The parent graph states it among its reports (``resources``)."""
    report = Path(cfg.output_dir) / RESOURCES_FILE
    report.parent.mkdir(parents=True, exist_ok=True)
    stated = shell_resources_report(model, _completion(cfg))
    report.write_text(json.dumps(stated, indent=2))
    model.set(OUTPUT_REPORTS, {**(model.get(OUTPUT_REPORTS) or {}), "resources": str(report)})
    absent = ", ".join(f"{column} ({why})" for column, why in stated["absent"].items())
    print(f"Resources per member written to {report}" + (f"; absent: {absent}" if absent else ""))
    return model


#: The kernel path's steps by name, for a DataflowBuildConfig's ``steps``.
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

#: The options type each step with options of its own declares, by the step's name, for
#: a DataflowBuildConfig's ``step_options``.
build_dataflow_step_options: dict[str, type] = {
    "step_prepare_import": PrepareImportOptions,
    "step_prepare_quantization": PrepareQuantizationOptions,
    "step_prepare_io": PrepareIoOptions,
    "step_prepare_streamlining": PrepareStreamliningOptions,
    "step_prepare_topology": PrepareTopologyOptions,
    "step_explore": ExploreOptions,
}
