# Declarative dataflow model

`finn.dataflow.model` is an isolated experiment in declaring design spaces as
ordinary Python classes. It does not wrap or preserve the existing
`finn.dataflow.authoring` and `finn.dataflow.kernels` APIs.

The stack is intentionally small:

```text
Space class
    -> flat DesignSpaceSpec
        -> existing _engine validation and evaluation

Kernel(Space)
    -> configured one-Region Kernel
        -> artifact-native source and ABI values
```

`DesignSpaceSpec`, `Engine`, and `DesignPoint` remain the normalized IR and
runtime. The model package is a source-language frontend, not another design
space evaluator.

## A closed Space

A root `Space` may declare problem fields and decisions directly on its class.
Derived values and constraints name their dependencies explicitly.

```python
from finn.dataflow.model import Decision, Problem, Space, compile_space, derived


class Folding(Space):
    extent = Problem(int)
    lanes = Decision(int, values=(1, 2, 4))

    @derived(int, extent=extent, lanes=lanes)
    def cycles(*, extent: int, lanes: int) -> int:
        return extent // lanes


spec = compile_space(
    Folding,
    "folding",
    problem_namespace="problem.folding",
)
```

This produces the same raw declaration records that could be written by hand:
`ProblemField`, `Decision`, and `DerivedProperty`. `Engine.validate(spec)` is
still the authority that validates those records.

A constraint-free `Space` is valid. `ConstraintGroup` and `Readiness` are
available only when a caller needs named aggregate checks.

## Reusable Space fragments

Reusable spaces use `Input`, not `Problem`. `Use` binds each child input to a
typed value in the parent and flattens the child beneath a fresh namespace.

```python
from finn.dataflow.model import Decision, Input, Problem, Space, Use, derived


class Tile(Space):
    extent = Input(int)
    lanes = Decision(int, values=(1, 2, 4))

    @derived(int, extent=extent, lanes=lanes)
    def cycles(*, extent: int, lanes: int) -> int:
        return extent // lanes

    exports = (lanes, cycles)


class Pair(Space):
    extent = Problem(int)
    left = Use(Tile, extent=extent)
    right = Use(Tile, extent=extent)

    @derived(int, left=left.cycles, right=right.cycles)
    def total(*, left: int, right: int) -> int:
        return left + right
```

The result is one flat `DesignSpaceSpec`, one `Engine`, and one `DesignPoint`.
There are no nested engine instances or runtime subspace objects. A child may
not introduce a `Problem`, every `Input` must be bound exactly once, and only
the values named in `exports` are visible to its parent. Reusing the same class
twice creates independently rebased declarations without mutating the class.

`Use(..., when=condition)` applies the existing engine applicability semantics
to the entire child fragment.

## Inheritance

Declarations are collected from the oldest `Space` base to the leaf class.
An override keeps the inherited position. It must retain the declaration
category and compatible value semantics; replacing a declaration with an
ordinary class value is an authoring error. New leaf declarations follow in
class-definition order.

The class body stores immutable templates with relative references only.
Compilation allocates paths and dependency references afresh, so compiling the
same class under several namespaces is deterministic and thread-safe.

## A one-Region Kernel

`Kernel` is the first specialization of `Space`. A Kernel class declares:

- a stable `id` and `version`;
- one `ComputationContract`;
- exactly one `@derived` value named `region` whose value is a
  `DataflowRegion`;
- its own decisions and feasibility constraints;
- scalar physical `Parameter`s;
- an exact `ComponentABI`; and
- an ordered source closure.

```python
from typing_extensions import Self

from finn.dataflow.artifacts.abi import ComponentABI
from finn.dataflow.computation import ComputationContract
from finn.dataflow.design.region import DATAFLOW_REGION_SEMANTICS
from finn.dataflow.model import Decision, Input, Kernel, Parameter, derived
from finn.dataflow.region import DataflowRegion


class ExampleKernel(Kernel):
    id = "example"
    version = "1"
    computation = ComputationContract("example.copy")

    extent = Input(int)
    lanes = Decision(int, values=(1, 2, 4))

    @derived(DATAFLOW_REGION_SEMANTICS, extent=extent, lanes=lanes)
    def region(*, extent: int, lanes: int) -> DataflowRegion: ...

    LANES = Parameter(lanes)

    @classmethod
    def component_abi(cls, configured: Self) -> ComponentABI:
        return ComponentABI(
            "example",
            (),
            (("LANES", str(configured.LANES)),),
        )
```

Kernel compilation automatically adds structural Region validation, a
whole-Kernel feasibility set, and a readiness profile. Configuration succeeds
only after all owned decisions, derived physical parameters, and constraints
resolve. The configured object retains its Region, local assignments, imported
decision provenance, physical parameter table, ABI, and source contributions.
It does not retain an `Engine`, `DesignPoint`, Network, node, edge, filesystem,
tool, or artifact store.

The ABI parameter table must exactly match the Kernel's resolved physical
parameter table. A physical constant uses `Parameter.constant(value, why=...)`
so the reason it is not a design-space value is explicit.

## DotpAxi

`DotpAxiKernel` is the sole production vertical slice in this experiment. It
owns the PE, SIMD, and pumping decisions; constructs its folded dot-product
Region; checks numeric and DSP packing feasibility; derives the FinnLib RTL
parameters; declares the exact ordered FinnLib source closure; and exposes the
physical AXI-Stream ABI of `dotp_axi`.

It imports no DataflowOp, DataflowDesign, MVAU operation implementation, or
legacy Kernel authoring machinery. Direct evidence configures it from a flat
engine point and tests its RTL numerically and through OOC synthesis.

## Artifact boundary

The artifact substrate remains downstream and does not import this package.
The one-way helpers in `kernel_artifacts.py` turn a configured Kernel into
artifact-native values:

```text
configured Kernel + declared source roots
    -> ResolvedContributions
    -> Derivation for the reusable source closure
    -> PortableComponent carrying source ArtifactRef + ComponentABI
```

Artifact keys contain only values read by the corresponding artifact stage.
Occurrence namespaces and filesystem locations do not enter portable
identity. Stores, packaging formats, tool requests, and synthesis stay outside
the Kernel object.

## Deliberate boundary of this experiment

This package does not define or preserve DataflowDesign or DataflowOp. It does
not place Kernels into a Network, connect Region boundaries, attach input
supply, project ONNX graph context, or persist operation selections. Those are
the responsibilities to assess only after this Kernel-level contract is
reviewed. Existing upper-stack collection failures caused by retired artifact
interfaces are therefore non-gating here.
