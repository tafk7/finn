# First kernel refinement pass

This pass implements reusable scalar policy and native stream constructs,
adopts them in the existing kernels, and ports the package to a concrete
FinnLib source baseline. The generic Space engine and parked dataflow package
are unchanged.

The source baseline is recorded in [BASELINE.json](BASELINE.json). FinnLib
changes are on `kernel-contract-refinement-20260925`, commit
`b17eae6a074ea678c633598fa42e7751e6cea194`. The FINN changes are on the existing
`feature/kernel-package-extraction` branch. There are no remote pushes.

**Start the review with the constructs and their consumers.**

| Construct | Purpose | Current consumers |
|---|---|---|
| `Integer` / `SignedInteger` | One integer policy with constraint, decision-domain, and concrete-check adapters | Scalar and AXI declarations, eltwise, thresholding, memstream |
| `Scalar` / `ScalarEncoding` | Accepted scalar storage encoding independent of transport | IntToFp32 and thresholding |
| `ReadyValidStream` / `StreamMarker` | Native pin mapping, width, endpoint, clock/reset, and explicit markers | AXI lowering, FIFO, input generator, eltwise, replay, cyclic delivery, MVAU top buses |
| `FifoStorage` | Effective backing and capacity alongside the native requested preference | FIFO's accepted `storage()` view |

The scalar declaration uses ordinary `ScopeBuilder`/`Subspace` composition.
It preserves canonical QONNX identity and independently attributable policy
constraints. Policies own dependency rebinding; the AXI implementation no
longer special-cases the concrete `Integer` class. The policy protocol remains
small and can be implemented by other datatype families.

A dtype can now be an owned choice with bounded enumeration, and the same
policy can validate its use as a scalar:

```python
from finn.core.space import Decision, Param, Space
from finn.kernels.datatypes.domains import SignedInteger
from finn.kernels.datatypes.scalar import Scalar
from finn.kernels.datatypes.semantics import QONNX_DATATYPE_VALUE_SEMANTICS
from finn.kernels.datatypes.values import resolve_qonnx_datatype_name as dtype

class Precision(Space):
    limit = Param(int)
    dtype = Decision(
        QONNX_DATATYPE_VALUE_SEMANTICS,
        domain=SignedInteger(2, limit).domain(),
    )
    scalar = Scalar(dtype, SignedInteger(2, limit))

base = Precision(limit=5)
candidates = base.field(Precision.dtype).candidates()
chosen = base.with_choices(dtype=dtype("INT3"))
encoding = chosen.scalar.view(Precision.scalar.view())()
assert encoding.bits == 3 and encoding.signed
assert not base.try_with_choices(dtype=dtype("UINT3")).accepted
```

For caller-supplied encodings, use a Param and policy constraints. No policy
silently chooses a dtype. Output AXI declarations also accept `valid_types=`;
port direction no longer prevents validating a caller-chosen result encoding.

The native stream record deliberately does not carry tensor maps or compute
semantics. Numeric packing remains in `AxiStream`/`PackedBeatLayout`, while
`ReadyValidStream` supplies the physical transfer mapping. It lowers to exact
native pins or, when compatible, an AXI bus. It refuses to lower a 13-bit word
or a loop/replay marker as an AXI interface. Marker meanings are explicit:
LAST, LOOP_END, and REPLAY_END.

```python
from finn.kernels import FifoKernel

base = FifoKernel(word_bits=13, depth=2)
input_stream, output_stream = base.interfaces()  # independent of RAM choice
assert input_stream.data_width == 13

fifo = base.with_choices(ram_style="ultra")
storage = fifo.storage()
assert storage.requested_style == "ultra"
assert storage.effective_style == "shift"
assert storage.capacity == 5
```

FIFO capacity includes output storage. The checked projection covers the
native shift, BRAM, and URAM paths, including split memory arrays. It describes
the configured RTL, not post-synthesis primitive inference or target-device
availability. Native storage overrides remain preferences rather than becoming
new strict selections in this pass.

**Admission and diagnostics are stronger.**

- IntToFp32 and thresholding consume accepted scalar encodings before pin
  construction. INT0/UINT0 are normal configuration refusals, including through
  eltwise and memstream, rather than incidental zero-width-pin exceptions.
- Thresholding's type, table, folding, memory, bias, and configuration checks
  settle independently. A known BIPOLAR rejection is visible before the
  AXI-Lite decision is made.
- Dotp's input and output scopes declare integer policies. Unsupported
  encodings cannot escape through raw build output, although other build
  constraints still correctly gate otherwise available output.
- Dotp refuses PE/SIMD values and packed stream widths that overflow its
  native unsigned integer parameters. Replay and cyclic transport also check
  their scalar parameter widths.

**The source port is validated beyond path rewriting.**

All live kernel sources use FinnLib's flat layout. The two corrected modules
previously under Python `resources` now live in FinnLib. The missing replay
module was restored from the retained, previously validated build artifact,
with its numeric/backpressure regression. Eltwise now declares `queue.sv`,
matching this revision's native output queue; simply moving the old `fifo.sv`
path did not elaborate, and the numerical gate caught that mismatch.

Installed-wheel checks still build without checkout imports and compare all
copied sources and manifests with the explicit dependency roots. They no longer
require callers to use Git repositories at an obsolete hardcoded revision.
This pass's tested revisions are recorded separately in BASELINE.json.

FinnLib's new dotp regression sends 128 consecutive one-beat reductions with
long output stalls through DSP48, DSP58 soft-vectorized, and packed DSP58
configurations. It verifies values, counts, and output stability. The corrected
wrapper passes. The original wrapper overflows its output queue at 285 ns in
the packed instance. This establishes that the regression detects the defect
being fixed. These new dotp checks use behavioral arithmetic; they are not
place-and-route or silicon timing measurements.

The FIFO regression measures capacity under a stalled output for eight native
configurations, covering shallow overrides, SRL, BRAM, URAM, and split memory.
The original kernel numerical checks continue to validate the emitted modules
and their native interfaces.

**Scope remaining for a subsequent pass.**

Explicit dotp architecture selection and architecture-specific segmentation
are still to be implemented. Its current automatic RTL selection and admitted
arithmetic profiles remain in place. The conservative buffering correction has
been moved and regressed; timing formulas have not yet been consolidated into
a shared RTL package.

MVAU now shares native-to-AXI lowering, but its assembly remains procedural.
An accepted assembly-plan view and structured stream connection lowering are
still worthwhile next consumers. Cross-port framing relationships remain in
kernel contracts/docstrings; this pass does not introduce a general behavior
language. Thresholding's multi-set AXI-Lite and negative-bias restrictions are
retained pending native repairs. HLS vector/control abstractions and an HLS
dotp comparison are also deferred.

These boundaries leave the new constructs concrete enough to review before
expanding them into architecture and composition selection.

Validation: **301 Space tests and 715 kernel tests passed**, with no skips.
Formatting, lint, and strict typing checks passed, including the new public
view types. The standalone FinnLib regressions also passed. After the full
gate, the loop-marker rejection test was strengthened to use a byte-aligned
payload and rerun successfully. See [VALIDATION.txt](VALIDATION.txt).

Run the complete gate with:

```bash
PYTHON_BIN=/home/tkeller/prj-kernels/.kernel-venv/bin/python \
  bash scripts/check-kernels.sh
```

The local ignored `deps/finnlib` and `deps/qonnx` symlinks supply the recorded
checkouts. Native checks require the Vivado simulator and Vitis headers used
by the existing gate. Standalone FinnLib regressions have ABC entries named
`dotp_axi_backpressure` and `replay_buffer`.
