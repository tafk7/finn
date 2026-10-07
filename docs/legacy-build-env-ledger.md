# Remaining legacy build environment obligations

## Current implementation obligations (2026-09-25)

FINN's own sources are packages (`finn.rtllib`, `finn.custom_hls`, `finn.xsi` with
`src/`, `finn.deploy` with `data/`), declared by name in `src/finn/resources.toml`. `src/finn_xsi/` is the installed Python driver. There is no synthetic
resource package or checkout-discovery hook. External HLS/board data are external
resources, pinned, fetched on first use and cached (`finn.resources`);
scratch/configuration and whole-build worker reconciliation remain private-branch
integration work.

| Remaining input/mechanism | Current consumers | Gate for further removal |
| --- | --- | --- |
| Legacy root/scratch/config | `_legacy_build_env.py`, existing `basic.py` allocation helpers, builder, XSI artifact default | P6 actual private branch interfaces; no inferred API or allocation change here. |
| Legacy default tool translation | `CallHLS`, `CreateStitchedIP`, Zynq/Vitis/SLASH, FINNLoop IP generation, xelab, C++ and xclbin inspection accept or use scoped toolchains. Existing defaults translate via `_legacy_build_env.toolchain`. | P6 explicit runtime inputs using the real build API. |
| Remaining native compatibility | Legacy live-object simulation and C++ harness still use current interfaces. Imports of simulation consumers no longer load native libraries; compilation/discovery is pure Python. | Integrate and installation-validate the new direct session entry point before retiring the legacy simulation paths. |
| Vendor header/library interpretation | HLS C++ compilation, `outer_shuffle.py`, `basic.py` vendor root/version access and the current C++ harness; private build prerequisite checks remain unchanged | P6 overlap resolution and actual tool execution. |
| Native bridge/design reuse | New `.finn.json` records cover artifact/tool/source/header/ABI/compile inputs. Direct sessions validate these before reuse and after exec. Old legacy artifacts need rebuilds before using sessions. | Real AMD compatibility, concurrent reuse and functional-equivalence evidence. |
| Bare tool shim and `FINN_ENV_APPLIED` | Deliberate vendor shells/site tools including xvlog/xsim/xsc/lmutil/xsct and the RTL library's developer simulation script | Actual licensed and native tests before removing shims or `finn-toolchain.sh`; migrated vendor callers use argv directly. The integration-sensitive C++ performance harness still runs its replay shell through the existing process helper. |
| Global image `LD_PRELOAD` / explicit native activation | Existing libudev workaround and explicitly sourced native tooling | Licence and native simulation gates are unavailable. Do not remove on synthetic evidence. |
| Native sbx Bash integration | `BASH_ENV` exists only in the sbx variant and reads sbx-owned persistent configuration | Keep native integration; no FINN install/activation/repair is added there. |

The generic Bash hook and entrypoint Tcl-copy behavior were removed. Site Tcl uses
explicit mounts; entrypoint work is limited to UID/home/PATH/exec. Native activation
selects scratch without creating it. Normal Python imports do not install packages,
select another checkout or create build storage.

Shared integration-sensitive edits are intentionally narrow: `basic.py` changes
only CppBuilder and its import; `core/rtlsim_exec.py` changes only the import-time
availability assignment. The same lazy assignment changes in `hwcustomop.py` and
`rtlbackend.py`. HLSBackend/FINNLoop and known transformation call sites change
only scoped command/resource/native discovery behavior. `src/finn/builder/` and
`_legacy_build_env.py` remain byte-for-byte unchanged from the handoff.

## P0/prototype inventory and historical behavior

The audit below retains the baseline and deletion conditions, including the
whole-build worker slated for removal in P6. It does not endorse that architecture.

P0 re-audit (2026-09-20): the descriptions below record prototype behavior.
The approved destination is in-process orchestration from the private build-engine
branch and a native simulation-session process; the whole-build worker is temporary.
The file disposition that recorded ownership boundaries is in history at `5dd9df9bc`.

| Remaining mechanism | Actual consumers found in P0 | Replacement gate |
| --- | --- | --- |
| Root/resource interpretation | `_legacy_build_env.py` is the only `FINN_ROOT` reader under `src/`; `util/test.py` still reads checkout-only CIFAR fixtures via `finn.qnn-data`. Shell setup, quicktest and Docker configuration use the root to locate the explicitly selected checkout. | P1 moves shipped data; test fixtures stay checkout-only. P6 owns scratch/config interpretation. |
| Ambient vendor dispatch | `make_zynq_proj.py`, `alveo_build.py` (Vivado packaging, v++ linking, XML generation, slashkit), `rtl/finn_loop.py` (Vivado AXI packaging), and `finn_xsi/finn_xsi/adapter.py` (xelab). | P4 direct operation tests and site-route preservation before deleting tool shims. |
| Native compile/execute | `util/basic.py:CppBuilder`, `hlsbackend.py` C++ simulation, `xsi/setup.py` bridge compilation, `core/rtlsim_exec.py` C++ harness. | Scoped argv/cwd/env and failures in P4/P5. |
| Other subprocesses | `make_driver.py` runs Git init/fetch/checkout/submodule commands for its external C++ driver and invokes `xclbinutil` to inspect XRT bitfiles; builder launches the prototype whole-build child. | Scope `xclbinutil` separately from Git; defer builder to P6. |
| Native imports | `finn_xsi/sim_engine.py` imports `xsi`; `adapter.py` eagerly imports SimEngine. `hwcustomop.py`, `hlsbackend.py`, `rtlbackend.py`, `rtl/finn_loop.py` and `core/rtlsim_exec.py` call `xsi.is_available()` during import. | P5 separates pure compilation/discovery and native loading. Migrate shared simulation consumers with P6 overlap review. |
| Vendor path interpretation | `basic.py` derives Vivado root/version; HLS C++ and `outer_shuffle.py` need vendor headers. `adapter.py` finds xsim/kernel; `build_dataflow_checks.py` and Alveo require Vitis/XRT/platform inputs. | Header/library/tool selections must be explicit at the operation; private build checks stay unchanged until P6. |
| Vendor forwarding | `docker/config.py` and the `docker/sbx/xilinx` kit forward selected installations and licences; forwarding is legitimate runtime configuration. | Do not confuse forwarding with FINN-owned resource discovery. |
| Startup side effects | Entrypoint creates UID home and copies optional site Tcl; `finn-bashenv.sh` sources native sbx persistent env in all image variants; bare-tool shim sources `finn-toolchain.sh` and sets `FINN_ENV_APPLIED`; native activation sources tools. Image preloads libudev globally. | P3 explicit environment preparation; P7 variant-specific sbx hook and explicit Tcl mounts. Licensed/native evidence required before loader/shim retirement. |

The deleted `finn_paths.py`, `finn-live.pth`, and handwritten `docker/build_dataflow`
have no active callers; references in historical documentation are retained.

Search marker: `FINN_LEGACY_COMPAT`. Interpretation of legacy FINN checkout,
scratch and default tool-selection conventions lives in
`src/finn/util/_legacy_build_env.py`. Permanent resource and toolchain modules
never import that adapter. The root-interpretation allowlist is tested in
`tests/util/test_runtime_installation.py`; forwarding vendor variables is allowed.
No Python import, generic startup hook or command-name detector activates it.

| Assignment/input | Actual consumers and enabling input | Tests | Deletion condition |
| --- | --- | --- | --- |
| `FINN_ROOT` | `get_finn_root()` checkout-only API. No FINN-owned resource or external-data consumer reads it; the container entrypoint uses it only to find the checkout to install. | resource installation journeys; legacy precedence test; interpretation allowlist | External callers stop using the checkout API. |
| `FINN_BUILD_DIR` | `make_build_dir`, dataflow intermediate-output reporting, XSI artifact default. Explicit directory build boundaries create child scratch; default is `$FINN_HOME/build` (`~/.finn/build`). Direct allocation creates scratch on demand, never on import. | worker inheritance, import-no-repair, installed driver generation | A separately specified build allocation API replaces these consumers; this change does not redesign allocation. |
| `FINN_RESOURCES_<NAME>` (`FINN_HLSLIB_PATH` as an alias for `hlslib`), `FINN_HOME`, `FINN_RESOURCES_DIR`, `FINN_RESOURCES_SYSTEM_CACHE`, `FINN_RESOURCES_FILES`, `FINN_RESOURCES_OFFLINE`, `FINN_RESOURCES_<NAME>_URL` | `finn.resources`: per-resource directory overrides, cache roots, project declaration files, offline mode and source URL overrides, for HLS C++/Tcl generation (`hlslib`) and Zynq project generation (`vivado-boards`). Defaults are the pins and digests in `finn/resources.toml`, fetched on first use. `FINN_BOARD_FILES_PATH` is removed; FINN warns if it is set. | `tests/util/test_resources.py`; generated HLS and Zynq references | None required: these are supported settings, not legacy inputs. |
| `XILINX_*`, `FINN_HLS_FRONTEND`, `FINN_TOOL_DIR_OVERRIDE` | Existing `CallHLS` and `CreateStitchedIP` callers translate once to `Selection`; the dataflow builder does not read them (its configuration names its `Selection`, and its Alveo check reads that selection's prepared `XILINX_VITIS`); path-version fallback only when no frontend is requested. Site overrides skip local activation. The installations are named only by their `XILINX_*` roots. Other existing vendor callers still use the ambient resolver/shim. | settings isolation, routing, frontend compatibility, fake Vivado/HLS operations | All remaining callers accept permanent tool selections; remove legacy translation only then. |
| `LD_LIBRARY_PATH` before Python startup | Directory build worker's XSI kernel and floating-point library loading. Also preserved by explicit native activation and existing image loader settings. `build_environment` prepares only the child mapping, from the build configuration's `Selection`. | child inheritance; actual XSI simulation unavailable in this environment | Fresh-worker XSI experiment passes correctness, loader, cleanup, cancellation and timing tests on supported installations. |
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
