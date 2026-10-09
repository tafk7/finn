# Remaining build environment obligations

FINN's legacy environment layer is deleted: no legacy environment variable is
read as a compatibility shim. What remains are machine settings, which configure how this
machine runs FINN, and native integration obligations. Portable output (generated
code, scripts, packaged IP, drivers, a build's configuration) does not read the
machine settings.

## Machine settings

| Setting | Read by | Tests |
| --- | --- | --- |
| `FINN_HOME` | `finn.resources.home()`: fetched resources, XSI builds, the default build directory | `tests/util/test_resources.py` |
| `FINN_RESOURCES_<NAME>`, `FINN_RESOURCES_DIR`, `FINN_RESOURCES_SYSTEM_CACHE`, `FINN_RESOURCES_FILES`, `FINN_RESOURCES_OFFLINE`, `FINN_RESOURCES_<NAME>_URL` | `finn.resources`: per-resource directory overrides, cache roots, project declaration files, offline mode and source URL overrides. Defaults are the pins and digests in `finn/resources.toml`, fetched on first use. The retired `FINN_HLSLIB_PATH` and `FINN_BOARD_FILES_PATH` select nothing. | `tests/util/test_resources.py` |
| `FINN_BUILD_DIR` | `finn.resources.scratch()` (default `$FINN_HOME/build`): `make_build_dir`, the dataflow build's report, the build process's environment. Allocation creates it on demand, never on import. | `tests/util/test_resources.py`, `tests/util/test_runtime_installation.py` |
| `FINN_TOOL_DIR_OVERRIDE` | `finn.util.toolchain.machine_selection()`: the site command directory of the machine's toolchain, which every transformation called without a toolchain and every build runs by, a stated selection laid over it. A stated selection's `command_dir` wins. | `tests/util/test_tool_route.py`, `tests/util/test_build_toolchain.py` |
| `FINN_VIVADO_JOBS` | `finn.util.toolchain.machine_selection()`: how many runs Vivado launches at once (`Selection.vivado_jobs`; unset, the machine's cores, at most 16). A stated selection's `vivado_jobs` wins; the HWCustomOp flow reads its own `vivado_jobs` field. | `tests/util/test_build_toolchain.py` |
| The machine file's `FINN_XILINX_*`, `FINN_LICENSE_*`, `PLATFORM_REPO_PATHS` | `finn.util.machine_file`: the installation `activate.sh` and `docker/run` select, the licence a local tool launch gets, and (`FINN_XILINX_VERSION`) the machine toolchain's HLS frontend. | `tests/util/test_tool_route.py`, `tests/util/test_container_config.py` |
| `FINN_XSI_BUILD_DIR`, `NUM_DEFAULT_WORKERS`, `FINN_XELAB_MT`, `LIVENESS_THRESHOLD` | The XSI build's directory, worker counts, xelab threads, the rtlsim watchdog. | `tests/util/test_xsi_build.py`, `tests/util/test_rtlsim_liveness.py` |

`FINN_ROOT` is read by no source under `src/` (`test_no_source_reads_the_checkout_root`);
shell setup and the container entrypoint use it only to find the checkout.

## Native integration obligations (2026-10-07)

| Mechanism | Current consumers | Gate for further removal |
| --- | --- | --- |
| Loader path before Python starts | XSI's simulation kernel and floating-point libraries: `Toolchain.simulation_environment()` for `build_dataflow_directory`'s build process and the XSI C++ driver; `docker/finn-toolchain.sh` for native shells; the image's loader settings. | Fresh-worker XSI experiment passes correctness, loader, cleanup, cancellation and timing tests on supported installations. |
| Remaining native compatibility | Legacy live-object simulation and C++ harness still use current interfaces. Imports of simulation consumers no longer load native libraries; compilation/discovery is pure Python. | Integrate and installation-validate the new direct session entry point before retiring the legacy simulation paths. |
| Vendor header/library interpretation | `XILINX_VIVADO` read ambiently by `basic.get_vivado_root`/`get_vivado_version`, `xsi/setup.py`, `xsi/paths.py` and `outer_shuffle.py`; private build prerequisite checks remain unchanged | These should read the selected `Toolchain`. |
| Native bridge/design reuse | `.finn.json` records cover artifact/tool/source/header/ABI/compile inputs. Direct sessions validate these before reuse and after exec. Old artifacts need rebuilds before using sessions. | Real AMD compatibility, concurrent reuse and functional-equivalence evidence. |
| Bare tool shim and `FINN_ENV_APPLIED` | Deliberate vendor shells/site tools including xvlog/xsim/xsc/lmutil/xsct and the RTL library's developer simulation script | Actual licensed and native tests before removing shims or `finn-toolchain.sh`; migrated vendor callers use argv directly. |
| Global image `LD_PRELOAD` / explicit native activation | Existing libudev workaround and explicitly sourced native tooling | Licence and native simulation gates are unavailable. Do not remove on synthetic evidence. |
| Native sbx Bash integration | `BASH_ENV` exists only in the sbx variant and reads sbx-owned persistent configuration | Keep native integration; no FINN install/activation/repair is added there. |

The generic Bash hook and entrypoint Tcl-copy behavior were removed. Site Tcl uses
explicit mounts; entrypoint work is limited to UID/home/PATH/exec. Native activation
selects scratch without creating it. Normal Python imports do not install packages,
select another checkout or create build storage.

`build_dataflow_directory`/`build_dataflow` isolate cwd and compatibility in a fresh
interpreter. `build_dataflow_cfg` intentionally supports in-process custom callbacks
and retains global logging/stdout and legacy configuration limitations. Its XSI
callers must start Python with the required loader environment. Switching native
vendor libraries in an already-running interpreter and concurrent differently
configured XSI sessions remain unsupported. Use independent prepared processes;
no fork of an interpreter with vendor libraries loaded is claimed as isolation.
The optional XSI worker experiment is deferred because real XSI is unavailable.

The XSI extension is loaded only from `FINN_XSI_BUILD_DIR` or its default,
`$FINN_HOME/xsi` keyed by Vivado installation and Python ABI. An old source-tree
`finn_xsi/xsi.so` is no longer treated as an installed resource; select its directory explicitly to reuse it, or rebuild with
`python -m finn.xsi.setup --force`. Cache existence does not establish source/ABI/
toolchain compatibility. Changing Python, FINN/XSI sources or toolchains requires
a clean/rebuild; generated artifacts are never deleted from the installed package.

Checkpoint/model formats and artifact layouts remain unchanged. They retain
absolute intermediate, native-library, HDL memory and installed-resource paths.
Keep the build tree and exact installations for reuse. Moving directories,
upgrading FINN/external data or switching toolchains requires regeneration where
those references or generated code change. An environment adapter cannot repair
those artifacts or turn project copies into self-contained exports.
