# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""The sub-phases that take a Brevitas export to the graph the kernel path converts.

Each is a function ``(model, GraphPreparation) -> model`` over the transforms that
own the work, which stay where they are (``finn.transformation.qonnx``,
``finn.transformation.streamline``, qonnx's). They run in this order, the order
finn-dev's end-to-end flow prepared TFC in, and the kernel path's TFC fixture after
it:

- ``imported`` (P0): qonnx's cleanup, the batch or input shape overridden if asked;
- ``quantization_lowered`` (P2): Quant nodes to MultiThresholds and integer weights
  (``ConvertQONNXtoFINN``), then ``tidied``;
- ``io_explicit`` (P1): the preprocessing model merged ahead of the network, the
  input's datatype stated, the label select (TopK) appended, then ``tidied``. It
  follows P2: the preprocessing model is lowered on its own and merged into the
  lowered network, so P2 never sees it;
- ``streamlined`` (P3): the streamlining recipe, scales and biases absorbed into
  thresholds, convolutions lowered to Im2Col and MatMul, bipolar MatMuls to
  XnorPopcountMatMul;
- ``topology_settled`` (P4): the topology recipe, transposes absorbed and
  collapsed, MaxPool made NHWC; then tidied, layouts inferred and unused tensors
  removed;
- ``containers_exact`` (P6): every integer tensor held exactly by its container,
  the integer regions that need it widened to float64
  (``finn.transformation.prepare.containers``). It reads the annotations P4 leaves,
  and has no options.

On TFC P6 changes nothing: its bounds stay within 3456. On CNV (W2A2, W1A1)
P0-P4 give the nodes, initializers and annotations of finn-dev's ``step_streamline``,
which runs the topology between its two ``Streamline`` passes: P3 lowers the
convolutions ahead of the bipolar rewrite, which reads their MatMuls, and P4 follows.

The phase reads no target: what preparation makes of a graph does not depend on
where it is built. ``finn.transformation.prepare.checkpoint`` checks what it
leaves (P7).
"""

from __future__ import annotations

import inspect
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType, ModuleType
from typing import Any

from onnx import helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveStaticGraphInputs,
    RemoveUnusedTensors,
)
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup_model

import finn.transformation.streamline.absorb as absorb
import finn.transformation.streamline.collapse_repeated as collapse_repeated
import finn.transformation.streamline.extract_multithreshold_scale_bias as extract_scale_bias
import finn.transformation.streamline.reorder as reorder
import finn.transformation.streamline.round_thresholds as round_thresholds
import finn.transformation.streamline.sign_to_thres as sign_to_thres
from finn.core.containers import container
from finn.core.containers import name as container_name
from finn.transformation.prepare.containers import exact_containers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator,
)
from finn.transformation.streamline import Streamline


def _defined_transforms(*modules: ModuleType) -> dict[str, type[Transformation]]:
    return {
        name: value
        for module in modules
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, Transformation)
        and value.__module__ == module.__name__
    }


#: The transforms a recipe may name, by class name: streamlining's (absorb, reorder,
#: collapse_repeated, sign_to_thres, round_thresholds, extract_multithreshold_scale_bias,
#: which Thresholding's refusals name as their hint, and ``Streamline``, its fixed
#: sequence), and qonnx's lowerings the default recipes run. Each is constructed with
#: its defaults.
RECIPE_TRANSFORMS: Mapping[str, type[Transformation]] = MappingProxyType(
    {
        **_defined_transforms(
            absorb,
            reorder,
            collapse_repeated,
            sign_to_thres,
            round_thresholds,
            extract_scale_bias,
        ),
        "Streamline": Streamline,
        "LowerConvsToMatMul": LowerConvsToMatMul,
        "ConvertBipolarMatMulToXnorPopcount": ConvertBipolarMatMulToXnorPopcount,
    }
)

#: P3's default recipe: TFC's fixture's streamlining (finn-dev's ``step_streamline``
#: with ``MoveScalarLinearPastInvariants``), with that step's convolution lowering ahead
#: of the bipolar MatMuls' rewrite, which reads the lowered MatMuls.
DEFAULT_STREAMLINING = (
    "AbsorbScalarBiasIntoMultiThreshold",
    "MoveScalarLinearPastInvariants",
    "Streamline",
    "LowerConvsToMatMul",
    "ConvertBipolarMatMulToXnorPopcount",
    "Streamline",
    "AbsorbScalarMulAddIntoTopK",
)

#: P4's default recipe: ``step_streamline``'s topology after lowering convolutions.
DEFAULT_TOPOLOGY = (
    "MakeMaxPoolNHWC",
    "AbsorbTransposeIntoMultiThreshold",
    "MakeMaxPoolNHWC",
    "AbsorbConsecutiveTransposes",
)


@dataclass(frozen=True)
class GraphPreparation:
    """The graph-preparation phase's options (``KernelBuildConfig.preparation``). In
    JSON, for TFC: {"preprocessing": "preproc.onnx", "input_datatype": "UINT8",
    "topk": 1}.

    - ``override_inpsize``: P0, the batch size (an int) or the input's shape (a list)
      qonnx's cleanup sets; None keeps the export's.
    - ``max_multithreshold_bit_width``: P2, the widest Quant activation lowered to a
      MultiThreshold; a wider one stays a Quant.
    - ``preprocessing``: P1, an ONNX model (a QONNX export, as of
      ``finn.util.pytorch.ToTensor``) merged ahead of the network; its input becomes
      the graph's.
    - ``input_datatype``: P1, the graph input's datatype (a qonnx name), the domain the
      network is prepared for; None leaves the input unannotated, read as FLOAT32.
    - ``topk``: P1, a label select of the ``topk`` largest outputs appended (TopK,
      its indices the graph's output).
    - ``streamlining``: P3's recipe, names of ``RECIPE_TRANSFORMS`` run in order.
    - ``topology``: P4's recipe, likewise."""

    override_inpsize: int | list[int] | None = None
    max_multithreshold_bit_width: int = 8
    preprocessing: str | None = None
    input_datatype: str | None = None
    topk: int | None = None
    streamlining: list[str] = field(default_factory=lambda: list(DEFAULT_STREAMLINING))
    topology: list[str] = field(default_factory=lambda: list(DEFAULT_TOPOLOGY))

    def __post_init__(self) -> None:
        for name in ("streamlining", "topology"):
            unknown = [each for each in getattr(self, name) if each not in RECIPE_TRANSFORMS]
            if unknown:
                raise ValueError(
                    f"preparation.{name}: no transform named {', '.join(unknown)} "
                    "(finn.transformation.prepare.RECIPE_TRANSFORMS)"
                )
        if self.input_datatype is not None:
            try:
                DataType[self.input_datatype]
            except KeyError:
                raise ValueError(
                    f"preparation.input_datatype: {self.input_datatype!r} is no qonnx datatype"
                ) from None
        if self.topk is not None and self.topk < 1:
            raise ValueError(f"preparation.topk: {self.topk}; at least 1")


def tidied(model: ModelWrapper) -> ModelWrapper:
    """Shapes inferred, constants folded, nodes and tensors named by their place in the
    graph, every node's outputs annotated by qonnx's rule, static inputs removed: what
    finn-dev's ``step_tidy_up`` runs."""
    for transform in (
        InferShapes(),
        FoldConstants(),  # type: ignore[no-untyped-call]
        GiveUniqueNodeNames(),  # type: ignore[no-untyped-call]
        GiveReadableTensorNames(),
        InferDataTypes(),  # type: ignore[no-untyped-call]
        RemoveStaticGraphInputs(),
    ):
        model = model.transform(transform)
    return model


def _cleaned(model: ModelWrapper, override_inpsize: int | list[int] | None) -> ModelWrapper:
    inpsize = tuple(override_inpsize) if isinstance(override_inpsize, list) else override_inpsize
    cleaned: ModelWrapper = cleanup_model(model, override_inpsize=inpsize)  # type: ignore[no-untyped-call]
    return cleaned


def _lowered(model: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    return model.transform(
        ConvertQONNXtoFINN(  # type: ignore[no-untyped-call]
            filter_function=default_filter_function_generator(  # type: ignore[no-untyped-call]
                max_multithreshold_bit_width=options.max_multithreshold_bit_width
            )
        )
    )


def imported(model: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    """P0: the export through qonnx's cleanup (shapes, constants folded with the Quant
    nodes kept, unique parameters, names), its input size overridden if asked."""
    return _cleaned(model, options.override_inpsize)


def quantization_lowered(model: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    """P2: Quant activations up to ``max_multithreshold_bit_width`` bits to
    MultiThresholds and quantized weights folded to integers (``ConvertQONNXtoFINN``),
    then ``tidied``."""
    return tidied(_lowered(model, options))


def preprocessing_model(path: str, options: GraphPreparation) -> ModelWrapper:
    """The preprocessing model at ``path``, cleaned and lowered as the network is, its
    shapes and constants settled: ready to merge ahead of the network."""
    pre = _lowered(_cleaned(ModelWrapper(path), None), options)
    return pre.transform(InferShapes()).transform(FoldConstants())  # type: ignore[no-untyped-call]


def io_explicit(model: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    """P1: the network's inputs and outputs as the build states them: the
    ``preprocessing`` model merged ahead of it, the graph input's ``input_datatype``
    stated, ``topk``'s label select appended (``InsertTopK``, which leaves a graph that
    already ends in a TopK alone); then ``tidied``. A graph that already holds its
    preprocessing is given none: merging composes it again."""
    if options.preprocessing is not None:
        model = model.transform(
            MergeONNXModels(preprocessing_model(options.preprocessing, options))  # type: ignore[no-untyped-call]
        )
        # MergeONNXModels imports the standard domain only: the custom ops' domains are
        # stated again, so that reading their nodes warns of no fallback version.
        imported_domains = {opset.domain for opset in model.model.opset_import}
        for domain in sorted({node.domain for node in model.graph.node} - imported_domains):
            model.model.opset_import.append(helper.make_opsetid(domain, 1))
    if options.input_datatype is not None:
        model.set_tensor_datatype(model.get_first_global_in(), DataType[options.input_datatype])
    if options.topk is not None:
        model = model.transform(InsertTopK(k=options.topk))  # type: ignore[no-untyped-call]
    return tidied(model)


def _recipe(model: ModelWrapper, names: list[str]) -> ModelWrapper:
    for name in names:
        model = model.transform(RECIPE_TRANSFORMS[name]())
    return model


def streamlined(model: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    """P3: the ``streamlining`` recipe, in order."""
    return _recipe(model, options.streamlining)


def topology_settled(model: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    """P4: the ``topology`` recipe, in order; then ``tidied`` (the recipe's transforms
    name what they add at random), every tensor's layout inferred and the tensors no
    node reads removed."""
    model = tidied(_recipe(model, options.topology))
    return model.transform(InferDataLayouts()).transform(RemoveUnusedTensors())


def containers_exact(model: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    """P6: the integer regions whose bounds pass their container's exact integers
    widened to float64, Casts at their edges (``containers.exact_containers``). The
    rule is fixed: no option reaches it."""
    return exact_containers(model)


SubPhase = Callable[[ModelWrapper, GraphPreparation], ModelWrapper]

#: The sub-phases by their names in the design (P0-P4, P6), in the order they run.
SUB_PHASES: tuple[tuple[str, SubPhase], ...] = (
    ("P0 import", imported),
    ("P2 quantization", quantization_lowered),
    ("P1 io", io_explicit),
    ("P3 streamlining", streamlined),
    ("P4 topology", topology_settled),
    ("P6 containers", containers_exact),
)


def prepared(model: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    """``model``, an export, through every sub-phase in order. Public API on purpose,
    though only tests call it: the builder runs ``SUB_PHASES`` itself, one step each."""
    for _, sub_phase in SUB_PHASES:
        model = sub_phase(model, options)
    return model


def census(before: ModelWrapper, after: ModelWrapper) -> dict[str, Any]:
    """What a sub-phase changed, as the phase's report records it: the nodes removed and
    added, counted by op type; the annotations and the containers changed on tensors
    both graphs name (tensor: [before, after]; absent is null)."""
    counted = Counter(node.op_type for node in before.graph.node)
    counts = Counter(node.op_type for node in after.graph.node)

    def annotations(model: ModelWrapper) -> dict[str, str | None]:
        names = {name for node in model.graph.node for name in (*node.input, *node.output)}
        return {
            name: model.get_tensor_datatype(name).name if model.has_tensor_datatype(name) else None
            for name in names
            if name
        }

    def containers(model: ModelWrapper) -> dict[str, str | None]:
        names = {name for node in model.graph.node for name in (*node.input, *node.output)}
        found = {name: container(model, name) for name in names if name}
        return {
            name: None if each is None else container_name(each) for name, each in found.items()
        }

    stated, restated = annotations(before), annotations(after)
    held, reheld = containers(before), containers(after)
    return {
        "removed": dict(sorted((counted - counts).items())),
        "added": dict(sorted((counts - counted).items())),
        "annotations_changed": {
            name: [stated[name], restated[name]]
            for name in sorted(stated.keys() & restated.keys())
            if stated[name] != restated[name]
        },
        "containers_changed": {
            name: [held[name], reheld[name]]
            for name in sorted(held.keys() & reheld.keys())
            if held[name] != reheld[name]
        },
    }


def reference(export: ModelWrapper, options: GraphPreparation) -> ModelWrapper:
    """What the prepared graph must compute: the ``export`` with P0 and P1 alone
    applied, its Quant nodes and float arithmetic as exported, its inputs and outputs
    as the build states them. The phase's equivalence check executes it."""
    return io_explicit(imported(export, options), options)
