# Physical kernel package relocation

The authoritative machine-readable map is [relocation-map.json](relocation-map.json).
Baseline: `459fc8cc8d16694c8afec14250cff82abb3c6eae`, source HEAD
`4f48708170219913fa20a38eb1321d666edf46d1`. All 948 tracked/nonignored
paths were verified before edits; [SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)
includes modes, symlinks, missing paths, and the original index digest.

## Ownership

- `space/` and `_engine/` move together to `finn.kernels`, with their tests and
  generic Space test host (`tests/kernels/helpers.py`). No domain implementation
  may be imported by these layers.
- `artifacts/` moves intact; requirement and contribution declaration classes
  continue to be the same objects exported by the processing facade.
- Detached AXI declarations, layout, structure, validation and lowering move to
  `physical/`. Binding, capture, interface and conventional physical View adapters
  remain in dataflow.
- QONNX scalar values, domain constraints, value semantics and codec move to
  `datatypes/`. Region-specific semantics and arithmetic analysis remain outside.
- The flat dotp, MVAU, streaming builders, target enum and neutral Kernel base
  move to the package root. Older logical kernels remain dataflow integration.
- Corrected local RTL and the shared assembly template have one owner in
  `finn.kernels.resources`. CopiedSource uses explicit `kernels` source roots
  pointing to `resource_root()`, with resource-relative paths, in both checkout
  and installed-package builds. FinnLib still has its explicit pinned root.
- Engine, Space, artifacts, physical declaration, dotp/MVAU/streaming tests and
  their numerical harnesses move under `tests/kernels`. Mixed Region/compiler
  tests stay in `tests/dataflow`; their shared imports are updated. Pure packing
  helpers and process-isolated RTL transport have one kernel-test owner.

## Deliberate identities

Module-qualified identities follow the explicit module map. `DspBlock` stops
claiming `finn.dataflow.kernels.matmul.base` and uses `finn.kernels.target`.
Artifact declaration compatibility identities move from `finn.dataflow.artifacts`
to the corresponding `finn.kernels.artifacts` facade, without imports into old
modules. The QONNX codec identifier becomes `finn.kernels.qonnx_datatype` (version
1, unchanged canonical payload). Real QONNX datatype instances remain unchanged.
These changes may change construction fingerprints and generated module names.
Comparison preserves all configuration, ABI, wiring, initializer and source
fields, allowing only the recorded names/paths and derived generated names.

The public physical facade is `finn.kernels`. Dataflow's model facades retain
exports useful to model consumers, referencing canonical extracted objects.
Old physical-construction module paths are removed, without forwarding packages.

## Worker ownership

The primary performs the mechanical relocation/import map and owns remaining
consumers, boundary tests, check scripts, equivalence evidence and final review.
GPT-5.6 Sol workers own bounded follow-up work: (1) Space/runtime tests and typing,
(2) artifact/physical/datatype support and installed resources/package tests,
(3) concrete kernels and numerical harness helper extraction. Shared edits are
coordinated before integration; workers do not revert others' work.

## Integration details and review corrections

`tests/dataflow/space/test_value_semantics.py` stays with Region analysis.
The datatype and AXI mixed test files are split without dropping test bodies or
decorators; the map lists both destinations. Domain facade assertions remain
in `tests/dataflow/test_space_model_boundary.py`. The legacy numeric harness
and the independent harness share `tests/kernels/rtlsim/dotp_support.py`.

The generic structural codec retains `dataflow.structural`, version 1: it is
persisted format metadata, not a Python import path. Retained dataflow DSP
serialization likewise keeps its explicit historical enum payload. Dataflow's
comparison-only identity table remains with its analysis consumers; the new
kernel package neither imports it nor claims its compatibility names.
The dataflow local-problem golden fingerprint changes from
`c57ed799923038b5982420500f83dfeabd9adb027601126aa95617c53e0438f9` to
`53a078177783974f90cec6ace924f122199064f25f139bcedd12cb7681fb9a8e` with the
QONNX datatype codec identifier relocation. The graph-context MVAU golden
likewise changes from `fe1dca5910a789071ab5690089b9e84cbf2e128ac6c2ababb076c0d835a242cb`
to `23b4ad5d41a4498308cb52216591f7c3e84ddea6bb1b6cc747d199b8d083eaac`.
`before-schema.json`, `after-schema.json` and `schema-equivalence.json` prove
that only the codec identifiers of accumulator_type/output_type change; every
other payload field is identical, and the new hashes are recomputed.

Review identified two validation gaps, both corrected before completion:
relative dynamic imports now resolve against their package in layer checks,
and construction evidence now records and compares prepared source compilation
order. The enriched pre-move capture was taken from the clean detached baseline
checkpoint; no original dirty files were changed to obtain it.

The production template and RTL hashes are unchanged. The older artifact-test
fixture named `decomposed_wrapper.sv.j2` is a different, test-only template, not
a duplicate production resource.
