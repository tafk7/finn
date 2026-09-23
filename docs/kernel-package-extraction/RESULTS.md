# Physical kernel package extraction

Completed 2026-09-23 on `feature/kernel-package-extraction` in `/home/tkeller/prj-kernels/finn-kernels-extraction`.
The original dirty `finn-unified-space-stack` worktree is unchanged.

## Commits and ownership

- Source HEAD: `4f48708170219913fa20a38eb1321d666edf46d1`.
- Inherited baseline checkpoint: `459fc8cc8d16694c8afec14250cff82abb3c6eae`. This preserves
  all 948 tracked/nonignored paths, including untracked implementation files and
  seven tracked deletions. It is separate from extraction work.
- Extraction implementation: `905f9aa275720bc8d99958e37008c73cd2412a68`.
- This evidence commit records validation of that implementation.

The canonical library is [`src/finn/kernels`](../../src/finn/kernels/README.md).
It owns flat dotp, MVAU, streaming builders, the neutral Kernel base, Space/runtime,
artifacts, detached physical structures, QONNX scalar support and resources.
Region/Network models and compiler adapters remain in dataflow and import the
shared implementation. The physical library imports and runs with dataflow and
QONNX graph wrappers unavailable.

[RELOCATION.md](RELOCATION.md) and [relocation-map.json](relocation-map.json)
record every move, split and identity decision. The dotp physical View remains
authoritative; both MVAU entry points retain the child-View rejection regression.
The removed `dotp_axi_requirements` constructor was not restored.

## Validation

| Check | Result |
|---|---|
| Fresh inherited baseline | 2,273 passed |
| Independent kernel suite | 892 passed |
| Remaining dataflow suite | 1,397 passed; parity required |
| MVAU cycle regression | 1 passed |
| Ruff formatting and lint | Passed for both suites and production owners |
| Strict typing | 59 kernel and 82 dataflow/compiler source files; selected test gates passed |
| Relocation equivalence | 26 cases; full values, ABI, wiring, initializers, ordered sources and emitted bytes |
| Installed wheel | Dotp plus external/cyclic MVAU materialized with complete verified manifests |
| Representative XSI | 6 passed; external/cyclic free/stalled MVAU and one-beat DSP58 free/32-cycle-stalled dotp |
| Original worktree | All 948 paths, status and index unchanged |

The final combined gate completed without skips. See [validation.log](validation.log)
and [validation.json](validation.json) for commands, tool/dependency versions and
strict-check counts. Its initial logged HEAD is the inherited checkpoint because
validation ran before the extraction commit; its implementation files are the
ones committed above. The baseline has its own [log](baseline-tests.log).

The before/after construction captures are losslessly compressed JSON. The
comparison permits only the recorded module identities, two resource paths/root,
and recomputed generated module names; it retains every configuration field and
compilation order. Run `compare_equivalence.py` from the repository root with
`PYTHONPATH=src:tests:deps/qonnx/src`. Enriched baseline capture used a clean
detached checkout of the checkpoint. [equivalence-results.json](equivalence-results.json)
records all 26 outcomes.

Two existing dataflow golden fingerprints change solely because the QONNX
codec identifier becomes `finn.kernels.qonnx_datatype@1` on accumulator/output
types. Full payloads in `before-schema.json` and `after-schema.json`, with
[schema-equivalence.json](schema-equivalence.json), establish unchanged remaining
fields and exactly recomputed hashes. `DspBlock` now has its natural
`finn.kernels.target` identity. Generic `dataflow.structural@1` metadata and
retained dataflow comparison payloads remain format identifiers, without imports
back into dataflow from the physical package.

The corrected dotp wrapper remains byte-identical:
`466397881624b4d0a4884762700620d548271541dc97bb265ea4b80129a5cd90`.
Cyclic RTL and the assembly template are also unchanged. FinnLib is pinned to
`dfeafac81cd2a6da27e647ee03915ade5532186e` and QONNX to
`21d4c1a72334002aaf80ae099dbf5d167f001001`. [RTL evidence](rtl/VALIDATION.md) includes
requests, responses, compilation inputs and logs; simulator binaries are excluded.
The MVAU runs each produce eight result beats; both dotp runs produce 64 results
and include a 4,000-cycle drain window. Component framing/wrap/reset simulation
also runs in the kernel suite. No new full MVAU synthesis or timing result is claimed.

## Review and preservation

GPT-5.6 Sol workers handled runtime, support/packaging and concrete kernels with
bounded ownership. Independent review found two validation gaps (relative dynamic
import checks and compilation-order evidence); both were fixed and re-reviewed.
[REVIEW.md](REVIEW.md) records the completed review with no outstanding findings.

[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json), [BASELINE.json](BASELINE.json) and
[PRESERVATION.json](PRESERVATION.json) preserve provenance and final verification.
Dependency/XSI symlinks remain ignored local test inputs. No dependency binaries,
simulator databases or original-worktree edits are committed by this extraction.
