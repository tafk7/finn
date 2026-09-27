"""P4: thresholding's always-present control and set buses have no driver in a netlist.

``netlist`` drives only stream pins (through ``Composition.connect``) and clocks and
resets (through ``_drive``, which skips every bus pin). Validation requires every
child input to be driven. ``s_axis``/``m_axis`` would be stream pins; ``s_axilite``
and ``s_axis_set`` have no mechanism at all, even with ``use_axilite=False`` and one set.
"""

from finn.core.space import design_space
from finn.dataflow.datatypes import resolve_qonnx_datatype_name as dt
from finn.kernels import ThresholdingAxiKernel
from finn.kernels.artifacts.abi import Direction
from finn.kernels.physical.composition import Composition
from finn.kernels.physical.validation import abi_pins, validate_physical_structure
from finn.kernels.streams import _drive, _top_abi

point = design_space(
    ThresholdingAxiKernel(
        input_dtype=dt("INT4"),
        threshold_dtype=dt("INT4"),
        thresholds=(((0, 1, 2),),),
        bias=0,
        pe=1,
        depth_trigger_bram=0,
        depth_trigger_uram=0,
    )
).with_choices(use_axilite=False, deep_pipeline=False)
requirements = point.build_requirements
inputs: dict[str, int] = {}
for info in abi_pins(requirements.abi).values():
    if info.direction is Direction.IN and info.bus_id is not None:
        inputs[info.bus_id] = inputs.get(info.bus_id, 0) + info.width
print("input bits per bus:", inputs)
composition = Composition(_top_abi("top", [], [requirements]))
composition.add("u_thr", requirements)
_drive(composition, "u_thr", requirements, set())  # clocks and resets only; buses skipped
try:
    validate_physical_structure(composition.finish())
    print("valid")
except Exception as error:  # noqa: BLE001 - the probe reports the refusal
    print(type(error).__name__, error)
