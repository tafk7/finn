# Remaining legacy build environment obligations

Search marker: `FINN_LEGACY_COMPAT`. Interpretation of legacy FINN checkout,
scratch and default tool-selection conventions lives in
`src/finn/util/_legacy_build_env.py`. Permanent resource and toolchain modules
never import that adapter. The root-interpretation allowlist is tested in
`tests/util/test_runtime_installation.py`; forwarding vendor variables is allowed.
No Python import, generic startup hook or command-name detector activates it.

| Assignment/input | Actual consumers and enabling input | Tests | Deletion condition |
| --- | --- | --- | --- |
| `FINN_ROOT` | `get_finn_root()` checkout-only API; default external data under `deps/` when explicit data paths are absent. An explicit `root` or data path wins. No FINN-owned resource consumer reads it. | resource installation journeys; legacy precedence test; interpretation allowlist | External callers stop using the checkout API and all HLS/board users supply installed external data paths. |
| `FINN_BUILD_DIR` | `make_build_dir`, dataflow intermediate-output reporting, XSI artifact default. Explicit directory build boundaries create child scratch; default is `/tmp/finn_build_<uid>`. Direct allocation creates scratch on demand, never on import. | worker inheritance, import-no-repair, installed driver generation | A separately specified build allocation API replaces these consumers; this change does not redesign allocation. |
| `FINN_HLSLIB_PATH`, `FINN_BOARD_FILES_PATH` | HLS C++/Tcl generation, Zynq project generation. Paths are resolved and validated when generating code; versions remain in `deps.env`. | generated HLS references and external input errors | Callers consistently pass external data selection explicitly. |
| `XILINX_*` / `*_PATH`, `FINN_HLS_FRONTEND`, `FINN_TOOL_DIR_OVERRIDE` | Existing `CallHLS` and `CreateStitchedIP` callers translate once to `Selection`; path-version fallback only when no frontend is requested. Site overrides skip local activation. Child builds also supply missing `VIVADO_PATH`/`VITIS_PATH`/`HLS_PATH` aliases for `HLSBackend.compile_singlenode_code` and existing Alveo checks. Other existing vendor callers still use the ambient resolver/shim. | settings isolation, routing, frontend compatibility, fake Vivado/HLS operations | All remaining callers accept permanent tool selections; remove legacy translation only then. |
| `LD_LIBRARY_PATH` before Python startup | Directory build worker's XSI kernel and floating-point library loading. Also preserved by explicit native activation and existing image loader settings. `build_environment` prepares only the child mapping. | child inheritance; actual XSI simulation unavailable in this environment | Fresh-worker XSI experiment passes correctness, loader, cleanup, cancellation and timing tests on supported installations. |
| Image `LD_PRELOAD` and shell tool shim | Existing FLEXlm/libudev workaround and bare vendor convenience. These are retained, not evidence of a general settings model. | existing synthetic shell tests; licensed checkout unavailable | Actual licensed-operation and native-simulation checks prove scoped replacements sufficient. |

`build_dataflow_directory`/`build_dataflow` isolate cwd and compatibility in a fresh
interpreter. `build_dataflow_cfg` intentionally supports in-process custom callbacks
and retains global logging/stdout and legacy configuration limitations. Its XSI
callers must start Python with the required loader environment. Switching native
vendor libraries in an already-running interpreter and concurrent differently
configured XSI sessions remain unsupported. Use independent prepared processes;
no fork of an interpreter with vendor libraries loaded is claimed as isolation.
The optional XSI worker experiment is deferred because real XSI is unavailable.

The XSI extension is loaded only from `FINN_XSI_BUILD_DIR` or the build-directory
default. An old source-tree `finn_xsi/xsi.so` is no longer treated as an installed
resource; select its directory explicitly to reuse it, or rebuild with
`python -m finn.xsi.setup --force`. Cache existence does not establish source/ABI/
toolchain compatibility. Changing Python, FINN/XSI sources or toolchains requires
a clean/rebuild; generated artifacts are never deleted from the installed package.

Checkpoint/model formats and artifact layouts remain unchanged. They retain
absolute intermediate, native-library, HDL memory and installed-resource paths.
Keep the build tree and exact installations for reuse. Moving directories,
upgrading FINN/external data or switching toolchains requires regeneration where
those references or generated code change. An environment adapter cannot repair
those artifacts or turn project copies into self-contained exports.
