# Untimed kernel authoring: start here

This is a reading guide to the **real implementation and runnable examples in this
checkout**, not a proposed interface. Native and compute kernels use the selected
Space/model/Port constructs and the canonical `finn.dataflow` package. Logical
composition and RTL source emission do not require cycle estimates.

## Run the examples

Install the checkout's locked environment with `uv sync`. For offline emission,
set `FINN_RESOURCES_FINNLIB` to the library checkout matching FINN's resource pin;
without an override the normal resource resolver uses the declared pin. No Vivado
or XSim is launched by either example.

From the checkout root, each example is one command:

```sh
uv run python examples/untimed/native.py
```

```sh
uv run python examples/untimed/compute.py
```

Use `--out DIRECTORY` to choose an output directory. The default directories are
`/tmp/finn-untimed-native` and `/tmp/finn-untimed-compute`. Each command prints its
summary and writes:

- `logical-plan.json`: canonical tensor, source-ID, node, port and edge descriptors;
- `summary.json`: logical packing, actual adapter stages and explicit FIFO capacities;
- `rtl/`: the real generated top, referenced RTL sources and any memory image.

These are **source-emission** examples. They do not package a synthesized IP,
execute a kernel numerically, run a simulator, prove an achieved clock or rate,
or certify that the chosen FIFO capacity is temporally sufficient.

## Read the actual definitions

| Start with | What to look for |
|---|---|
| [Native example](../examples/untimed/native.py) | Existing `VpcKernel` and `FifoKernel` placed by `Facts`, real `Port` references, and an explicit depth-four FIFO. |
| [VPC definition](../src/finn/kernels/vpc.py) | Tensor and arriving/leaving `Param`s; native input/output PortModel children; module parameters and actual source/pin assembly. |
| [FIFO definition](../src/finn/kernels/fifo.py) | Explicit depth, memory-style `Decision`, geometry and platform validity, and realized storage capacity. |
| [Compute example](../examples/untimed/compute.py) | A real known-weight `MatMulKernel`, selected packed implementation/folding, owned result/weight facts, source memory and a fixed-capacity channel FIFO. |
| [MatMul definition](../src/finn/kernels/matmul.py) | Model facts, range/type ownership, implementation selectors and authored parent Port surfaces. |
| [Packed compute definition](../src/finn/kernels/dotp.py) | Real scheduled/indexed Ports on the selected compute implementation; no included `ControlModel` on MatMul/Dotp. |
| [Source memory definition](../src/finn/kernels/memstream.py) | `MemStream.config` is the included AXI-Lite `ControlModel`; the weight source owns its physical pins and held/tied-off values for build-time known contents. |
| [Port models](../src/finn/kernels/port_model.py) | Ordinary child Spaces; placed/idle direction and datatype rules; native, scheduled and indexed projections. |
| [Control models](../src/finn/kernels/control.py) | Included control and physical ownership, separate from logical dataflow. |
| [Composer](../src/finn/kernels/composer.py) / [Partition](../src/finn/kernels/partition.py) | Real model/serial module assembly, source and port discovery, stable choice ownership and module wiring. |
| [Channel/adapter definitions](../src/finn/kernels/channels.py) / [adapters](../src/finn/kernels/adapters.py) | Requirements from the canonical beat sequences and their registered existing physical realization. |
| [Example helper](../examples/untimed/common.py) | Actual policy application, complete validity/presence checks, per-edge logical queries and real source emission. |

No kernel is asked to verify its computation here. Known values/ranges that size
or initialize hardware remain kernel design facts; numerical reference execution
and the independent hardware harness are different responsibilities.

## What the two compositions demonstrate

**Native:** twelve INT8 elements enter the VPC as six two-lane beats and leave as
four three-lane beats. The downstream FIFO has requested depth four and the
existing RTL storage contract gives it capacity five words (the primitive minimum).
Requested depth and realized capacity are reported separately, not equated. The channels connect the authored Ports directly. The VPC's
width conversion is an explicit native kernel, not a hidden alternative search.

**Compute:** a `(2,4)` INT3 activation tensor feeds a known-weight `(4,4)` MatMul,
with PE=SIMD=2 on its packed hardware. MatMul owns its result range/encoding and
carried/stored weights. The channel source uses those facts to emit the memory
image. The input's existing `input_gen` recipe supplies replay and reduction
markers; the channel FIFO has requested and realized capacity sixteen words.
The existing FirstValid channel policy chooses registered memory/adapter cases;
no rate-based FIFO strategy or timing-based ranking runs.

The example helper copies **actual channel presentations** into a logical plan
for per-edge packing/adapter queries. Source identities remain distinct even for
equal Tensor values. It does not infer atomic firing groups or profiles from
kernel structure, and is not a new production extractor or executor.

## Composition changes carried by the canonical swap

The native and compute definitions above are existing hardware kernels, not new
recipes. The refined logical package can now derive some compositions the earlier
logical implementation refused: for example, a same-position lane-axis regroup
can connect through width conversion → reorder → width conversion, using the
registered VPC and input-generator stages. Direct connection still refuses when
an adapter is needed, and missing source positions still cannot be repaired.
Cyclic presentation counts whole passes rather than receiver-replayed elements;
lower logical reads/storage bookkeeping is not a faster physical rate.

Logical `Schedule.closing` accepts one contiguous reduced block, not only a
suffix. A non-suffix block whose inner folded extent is one can normalize to a
`LevelEnd` and use an existing TLAST; this is a deliberate authoring behavior
change. A genuine held `LevelDone` remains refused by the physical ScheduledPort
and adapter contracts. In-tree kernels still close their original suffixes;
no held-marker hardware or new kernel capability is added.

These changes do not waive shape/range/source/pass/marker/ownership or physical
capacity checks. Held timed verification, simulation and watchdog clients still
explicitly request their existing cycle readout before timed actions; untimed
configuration/source emission makes no rate or timing-verification claim.

## Logical order is not time

`Schedule.iterations` and `Schedule.fold` describe logical iteration/folding;
`present(reshape=...)` projects positions and `one_pass` removes whole-pass
repetition. A beat ordinal, replay count or marker boundary is not a clock cycle.
The examples still provide a physical 5 ns target where the real hardware requires
one. That is not an achieved frequency, latency estimate or rate proof.

The graph build's `configured_network` preserves stale/open-choice, validity,
presence, budget, pin/count and port-order checks **without asking for cycles**.
Its explicit legacy `.cycles` readout remains a separate request until timing
consumers migrate. Missing/refused characterization fails that request, not
logical configuration. No missing estimate becomes zero.

## Refused, unresolved and unsupported are different

- **Refused:** a known violation, such as depth below the existing FIFO minimum,
  incompatible datatype/range/source/pass/shape, marker loss, forbidden direction
  or broken physical geometry. A recipe being emitted does not waive these rules.
- **Unresolved:** a required fact/Decision/presence/constraint is still open. Useful
  model values can be available on an open owner, but emission demands complete
  validity and settled presence. An open FIFO depth must be explicitly chosen;
  the examples never substitute zero or invent a rate to size it.
- **Unsupported:** the logical model can express something the current query or
  hardware boundary does not implement. `Grouped` and held `LevelDone` remain
  logical descriptors, but existing physical Ports/source/boundary/adapter
  schemas explicitly refuse them where uncovered. Mapped selections, general
  merge queries, new movement/fork recipes and richer adapter alternatives remain
  bounded by their existing contracts. No automatic recipe/DSE expansion occurs.
- **Timing unavailable:** an optional profile lacks an applicable field/clock/rule.
  Logical packing and supported adapter requirements still make sense. Applicable
  modeled/conditional timing results are not measurements or universal liveness
  proofs; timed buffer sufficiency/deadlock/rate remain separate evidence.

See the executable boundary checks in
[`test_untimed_dataflow_bridge.py`](../tests/kernels/test_untimed_dataflow_bridge.py)
and [`test_untimed_build.py`](../tests/kernel_ops/test_untimed_build.py).

## Remaining boundaries

This candidate keeps the supported MemStream/source and native set-Port contract.
Fetched wrapper/source-serving behavior is not adapted or emitted here; no example
or generated source should be read as a fetched-source wrapper demonstration.

Runtime-input/container mechanisms, the provisional set-index revisit, fetched
wrapper/source-serving assembly, a native executor, new kernels, adapter DSE,
timing readouts/search/FIFO strategies/timeouts and physical validation are not
completed by these examples. Fetched-source wrapper behavior is not silently
presented as implemented here. Source emission remains separate from numerical
and hardware acceptance or external tool execution.
