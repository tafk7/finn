# Brainsmith's build and XSI boundaries

Date: 2026-09-18. Inspected sibling checkout `../brainsmith`, commit
`d260fb44fa0f348dd0078dc372e7d1c20d70200a` (2026-08-25). Its worktree was clean
before and after inspection. This investigation changes no Brainsmith files and
no FINN production execution code. Brainsmith is inspiration/reference only;
compatibility with its APIs, callbacks, dependency fork or workflows is not a
requirement for the FINN design.

Brainsmith addresses the loader-startup requirement by preparing the invoking
shell before starting Python. It calls FINN builds and simulation interfaces from
that Python process. It adds neither a whole-build Python worker nor an independent
XSI session worker. It delegates the native implementation to its FINN dependency.
This supports keeping ordinary FINN build orchestration in-process; it supplies
no additional evidence that native simulator sessions are isolated.

**The actual invocation sequence.**

```text
brainsmith project init
    |
    +-- Project configuration -> .brainsmith/env.sh and .envrc

Source env.sh manually, or activate through direnv
    |
    +-- Set tool roots, loader paths and compatibility variables
    +-- Source selected vendor settings64.sh scripts
    |
    v
Start brainsmith CLI / pytest / user Python
    |
    +-- FINNAdapter.build -> build_dataflow_cfg (same process)
    |
    +-- RTL operator / test executor -> FINN simulation API
            |
            +-- Native execution behavior belongs to the installed FINN
```

In Docker, `entrypoint-exec.sh` sources `setup-shell.sh` before `exec "$@"`.
`setup-shell.sh` activates the virtual environment and sources the generated
project environment, with a legacy manual tool-setup fallback. This places loader
settings before Python startup without a second Python build interpreter.
The container startup entrypoint also runs setup unless explicitly skipped;
this differs from FINN's newly explicit installed/editable preparation model.

**Environment generation and what it establishes.**

`brainsmith/settings/env_export.py` exports both `XILINX_*` and `*_PATH` names,
FINN checkout/build paths and worker controls. Its full environment includes
Vivado's `lib/lnx64.o`, the Vitis floating-point library directory and the system
libudev preload when applicable. `SystemConfig.generate_activation_script()`
emits these variables and sources Vitis or Vivado settings, followed by HLS
settings. `generate_direnv_file()` watches project configuration and sources this
activation script before commands launched from the shell.

The test documentation explicitly requires loader/tool variables before Python
starts. CLI validation and test conftest check for `BSMITH_DIR` as a marker; the
FINN adapter only warns if the marker is absent. That marker is evidence of an
activation convention, not proof of the selected native library version,
compatibility, successful licence use or independent simulator state.

This is a coherent model for one selected installation for a Python process's
lifetime. Activating another project in the surrounding shell affects subsequent
processes; it does not replace libraries in an already-running notebook kernel or
Python application.

Sources in the inspected checkout:

- `brainsmith/settings/env_export.py:111`: environment dictionary generation.
- `brainsmith/settings/schema.py:640`: activation-script generation.
- `brainsmith/settings/schema.py:715`: direnv integration.
- `brainsmith/settings/validation.py:72`: activation-marker validation.
- `brainsmith/cli/cli.py:123`: CLI validation before normal commands.
- `tests/pipeline/README.md:44`: documented pre-start setup for simulation.
- `docker/entrypoint-exec.sh:19` and `docker/setup-shell.sh:38`: container execution.

**Build orchestration stays in the existing Python process.**

`brainsmith/_internal/finn/adapter.py:71` imports `build_dataflow_cfg`, converts
configuration, temporarily changes cwd to the output directory, changes
`output_dir` to `"."`, calls FINN directly and restores cwd in a `finally` block.
It explicitly documents that this is not thread-safe and tells callers to use
separate processes for concurrent builds. Its constructor does not create such
processes. The default DSE runner iterates a stack and calls `run_segment`
synchronously; it does not implement a process pool around these builds.

Therefore, the whole-build worker recently added to FINN's
`build_dataflow_directory()` does not apply to this integration. Brainsmith bypasses
that entry point. This illustrates the scope of the whole-build wrapper; it
does not create a requirement to support Brainsmith. FINN should fix its own
configuration/path handling directly on its own merits.

Sources: `brainsmith/_internal/finn/adapter.py:17`, `:92`, `:131`, `:145`;
`brainsmith/dse/runner.py:103`, `:161`, `:258`.

**Simulation uses live FINN interfaces.**

The local Thresholding RTL operator calls `get_rtlsim()`, resets the returned
object, passes it to `rtlsim_multi_io`, and closes it, all directly from
`execute_node()`. The test RTLSim executor applies FINN's preparation transforms
and executes the model via QONNX in the same Python call path. Brainsmith adds no
separate native loader, `dlmopen` namespace handling or simulation-job supervisor
around these calls. The thresholding close sequence is not in a `finally` block.

The XSI setup installer does launch `bash -c 'source ... && python3 -m
finn.xsi.setup'`. That process builds the bridge; it is not a worker for subsequent
simulation sessions. Setup availability also delegates to `finn.xsi.is_available`.
The installer separately contains an old source-tree `finn_xsi/xsi.so` existence
check derived from `finn.__file__`.

Sources: `brainsmith/kernels/thresholding/thresholding_rtl.py:387`;
`tests/support/executors.py:293`, `:364`, `:389`;
`brainsmith/_internal/io/dependency_installers.py:214`;
`brainsmith/cli/commands/setup/helpers.py:20`.

The reference demonstrates how exposing live simulation objects couples callers
to native state. That is an API design tradeoff to learn from, not a compatibility
contract to preserve. FINN can choose a clean session API around its own supported
workloads, with simulation/testbench code inside the session, bulk inputs and
explicit results. Brainsmith-specific call patterns do not constrain that choice.

**A focused environment-generation probe.**

Run:

```bash
python3 docs/experiments/brainsmith_environment_probe.py ../brainsmith
```

The probe directly loads the real exporter and extracts the two actual activation
methods from the inspected source. A minimal configuration fixture and a fake
`settings64.sh` avoid installing the project or executing AMD software. Generated
scripts and results are stored only in a new temporary directory. The fixture
settings script records that it was called and deliberately does not change
library paths.

With an inherited old Vivado library path and a newly selected installation, the
exported and subsequently sourced environment contained:

```text
XILINX_VIVADO = <selected new installation>
LD_LIBRARY_PATH = <old installation>/lib/lnx64.o:
                  /lib/x86_64-linux-gnu/:
                  <selected new installation>/lib/lnx64.o
```

`to_env_dict()` preserves the inherited loader path and appends the selected one.
The generated cleanup shell runs before exporting the already-computed dictionary,
so the export can reintroduce a path the cleanup removed. `brainsmith project init`
uses this direct generation route.

The separate `generate_activation_scripts()` helper temporarily removes PATH and
LD_LIBRARY_PATH before generation, so this result must not be generalized to all
setup entry points. Real vendor settings can also subsequently change ordering.
The experiment demonstrates that the direct generated environment is not an
isolated snapshot of exactly one selected installation; it does not demonstrate
a real AMD library mismatch or failed simulation. The exporter itself preserved
the caller's environment in the probe.

**Dependency and validation scope.**

`pyproject.toml:213` selects editable `deps/finn`. The fetch script defaults to
`tafk7/finn` branch `feature/logging-integration-transformer`; the local path lock
entry does not pin that checkout's Git revision. This sibling checkout currently
has no populated `deps/finn`. The analysis therefore does not assume its installed
FINN fork is identical to the FINN tree investigated previously, nor claim its
native code was executed. It establishes Brainsmith's own preparation, adapter and
simulation call boundaries.

No AMD simulation, hardware build, concurrency or licence operation was run. Source
inspection and the focused exporter/activation probe are the validation evidence.

For FINN's design, the useful distinction is between initial loader setup and
native-runtime isolation. Brainsmith provides an example of shell activation for
the former. It adds no independent boundary for the latter. The recommendation
remains: fix FINN path handling directly, keep build orchestration in-process, and
place version/lifetime/failure isolation around native simulation sessions.
Choose the session API for FINN's requirements, without preserving Brainsmith
compatibility or importing its activation/workaround mechanisms.
