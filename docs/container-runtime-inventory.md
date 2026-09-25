# Prototype disposition and implementation inventory

P0 review: 2026-09-20. Starting branch `refactor/container-runtime-implementation` at
`97eb24645`; clean working tree, no untracked inputs. The preserved prototype is
`f20a2a8ea` (92 changed paths relative to `8a05d50a6`). No archive changes, reset,
merge or push. Dispositions describe intent; they do not claim release acceptance.

The private build-engine branch is unavailable. Its exact overlap remains unknown.
Do not edit `src/finn/builder/` or expand the whole-build worker. Shared files
`util/basic.py`, `_legacy_build_env.py`, `core/rtlsim_exec.py`, simulation consumers
and transformation constructors require narrow edits and a later interface review.

| Prototype path | Disposition | Review finding / next gate |
| --- | --- | --- |
| `.devcontainer/devcontainer.json` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `.dockerignore` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `MANIFEST.in` | Revise | Keep wheel/sdist/provenance/entry points; simplify resource mappings in P1. |
| `README.md` | Revise | Current user instructions must describe approved explicit installation behavior. |
| `VERSION` | Retain | Meaningful version shared by sdist and wheel; verify metadata. |
| `ci/README.md` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `ci/scripts/build-images.sh` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `custom_hls/__init__.py` | Delete after P1 | Synthetic resource package anchors replaced by conventional finn._data package. |
| `deps.env` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/Dockerfile.finn` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/README.md` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/build_dataflow` | Retain deletion | Explicit installs and generated entry points replace this mechanism; verify references in P7. |
| `docker/config.py` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/finn-bashenv.sh` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/finn-live.pth` | Retain deletion | Explicit installs and generated entry points replace this mechanism; verify references in P7. |
| `docker/finn_entrypoint.sh` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/finn_paths.py` | Retain deletion | Explicit installs and generated entry points replace this mechanism; verify references in P7. |
| `docker/image-inputs.txt` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/lib.sh` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/run` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docker/sbx/README.md` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `docs/README.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/brainsmith-xsi-comparison.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/container-runtime-implementation-plan.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/container-runtime-refactor-proposal.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/experiments/brainsmith_environment_probe.py` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/experiments/xsi_loader_probe.py` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/finn/getting_started.rst` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/high-value-runtime-plan.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/installation.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/legacy-build-env-ledger.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/runtime-changed-files.txt` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/runtime-information-obligations.json` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/runtime-information-obligations.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/runtime-information-redesign-plan.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/runtime-validation.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `docs/xsi-process-boundary-investigation.md` | Retain | Preserve design/experimental history; update current instructions and validation as implementation lands. |
| `finn-rtllib/__init__.py` | Delete after P1 | Synthetic resource package anchors replaced by conventional finn._data package. |
| `finn_xsi/__init__.py` | Delete after P1 | Synthetic resource package anchors replaced by conventional finn._data package. |
| `scripts/activate.sh` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `setup-local.sh` | Revise | P2 artifact identities/wheelhouse and P3 explicit isolated venvs; P7 removes hooks only after consumer gates. |
| `setup.cfg` | Revise | Keep wheel/sdist/provenance/entry points; simplify resource mappings in P1. |
| `setup.py` | Revise | Keep wheel/sdist/provenance/entry points; simplify resource mappings in P1. |
| `src/finn/builder/build_dataflow.py` | Integrate later | Private branch owns direct build orchestration/global state; do not expand prototype worker or activation. |
| `src/finn/core/rtlsim_exec.py` | Integrate later / revise narrowly | Shared allocation/tool/native seams; P4/P5 direct operations only, P6 resolves actual overlap. |
| `src/finn/custom_op/fpgadataflow/hls/elementwise_binary_hls.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/hls/thresholding_hls.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/hlsbackend.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/hwcustomop.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/matrixvectoractivation.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/convolutioninputgenerator_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/crop_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/elementwise_binary_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/finn_loop.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/fmpadding_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/hwsoftmax_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/hwwhere_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/inner_shuffle_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/layernorm_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/matrixvectoractivation_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/pad1d_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/pwpolyf_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/requant_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/streamingdatawidthconverter_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/streamingfifo_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/thresholding_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/rtl/vectorvectoractivation_rtl.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/templates.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/custom_op/fpgadataflow/vectorvectoractivation.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/transformation/fpgadataflow/create_stitched_ip.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/transformation/fpgadataflow/make_driver.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/transformation/fpgadataflow/make_zynq_proj.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/transformation/fpgadataflow/templates.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/util/_legacy_build_env.py` | Integrate later | Private branch owns direct build orchestration/global state; do not expand prototype worker or activation. |
| `src/finn/util/_toolchain.py` | Retain / complete | Scoped argv/cwd/environment foundation; complete caller and cancellation coverage in P4. |
| `src/finn/util/basic.py` | Integrate later / revise narrowly | Shared allocation/tool/native seams; P4/P5 direct operations only, P6 resolves actual overlap. |
| `src/finn/util/hls.py` | Retain / complete | Scoped argv/cwd/environment foundation; complete caller and cancellation coverage in P4. |
| `src/finn/util/installation.py` | Retain | Inspect actual imports, metadata and immutable image identity separately. |
| `src/finn/util/resources.py` | Revise | Keep resource_path interface; conventional package anchor in P1. |
| `src/finn/util/test.py` | Retain | Resource consumer migration; validate installed and editable outputs in P1. |
| `src/finn/xsi/__init__.py` | Revise | Keep packaged sources/external artifacts; correct setup check, separate discovery/loading and implement P5 session. |
| `src/finn/xsi/paths.py` | Revise | Keep packaged sources/external artifacts; correct setup check, separate discovery/loading and implement P5 session. |
| `src/finn/xsi/setup.py` | Revise | Keep packaged sources/external artifacts; correct setup check, separate discovery/loading and implement P5 session. |
| `tests/container/test_container_conformance.py` | Retain / revise | Re-run observable behavior; update resource paths; worker-specific assertions retire at P6. |
| `tests/util/runtime_resource_smoke.py` | Retain / revise | Re-run observable behavior; update resource paths; worker-specific assertions retire at P6. |
| `tests/util/test_container_cli.py` | Retain / revise | Re-run observable behavior; update resource paths; worker-specific assertions retire at P6. |
| `tests/util/test_container_config.py` | Retain / revise | Re-run observable behavior; update resource paths; worker-specific assertions retire at P6. |
| `tests/util/test_finn_deps_modes.py` | Retain deletion | Explicit installs and generated entry points replace this mechanism; verify references in P7. |
| `tests/util/test_runtime_codegen.py` | Retain / revise | Re-run observable behavior; update resource paths; worker-specific assertions retire at P6. |
| `tests/util/test_runtime_installation.py` | Retain / revise | Re-run observable behavior; update resource paths; worker-specific assertions retire at P6. |
| `tests/util/test_runtime_toolchain.py` | Retain / revise | Re-run observable behavior; update resource paths; worker-specific assertions retire at P6. |
| `tests/util/test_xsi_pkg_ordering.py` | Retain / revise | Re-run observable behavior; update resource paths; worker-specific assertions retire at P6. |

## Delivered files

The prototype inventory above is fixed. Current implementation changes are recorded
in `container-runtime-implementation-files.txt` relative to `97eb24645`; renamed
resource files are mechanical moves, with consumer/build-reference edits listed
separately by Git. The archive remains the recovery point.
