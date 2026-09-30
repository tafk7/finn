# XSI process-boundary investigation

Date: 2026-09-18. This is an investigation and proposed correction to the runtime
implementation, not an implemented simulation worker. Production build execution
has not been changed in this investigation. The accompanying loader experiment is
at `docs/experiments/xsi_loader_probe.py`.

Recommendation: keep FINN build orchestration in the caller and resolve build
configuration paths explicitly. Put native XSI execution in a process that owns
one simulation session. This is a permanent native-runtime boundary with a clear
lifetime, rather than a reason to restart the complete build in Python.

**What actually loads, and when.**

There are three different native objects:

1. FINN's pybind11 extension, `xsi.so`.
2. AMD's simulator kernel, `libxv_simulator_kernel.so` from 2024.2 onward (previously
   `librdi_simulator_kernel.so`).
3. The elaborated design, normally `xsim.dir/<snapshot>/xsimk.so`.

The bridge's build command in `src/finn/xsi/setup.py` links `-ldl -lrt`, with
ordinary compiler runtime dependencies, rather than directly linking the AMD
kernel. Importing the bridge registers Python classes. The AMD kernel is loaded
when `SimEngine` constructs `xsi.Kernel`; `xsi.Design` then opens the design library.
This follows the two-library model described in
[AMD UG900, 2024.2](https://docs.amd.com/r/2024.2-English/ug900-vivado-logic-simulation/Preparing-the-XSI-Functions-for-Dynamic-Linking).
The library rename is documented in
[AMD UG973, 2024.2](https://docs.amd.com/r/2024.2-English/ug973-vivado-release-notes-install-license/Installer).

```text
Python import xsi
    |
    +-- Load FINN bridge and its ordinary runtime dependencies
    +-- Register Kernel, Design and Port classes

Construct SimEngine
    |
    +-- Construct Kernel -> dlopen(AMD simulator kernel)
    +-- Construct Design -> dlopen(elaborated design)
    +-- Resolve xsi_open from design and other XSI functions from kernel
    +-- Run the simulation in this process
```

Consequently, a blanket claim that XSI inherently requires restarting Python is
incorrect. The current basename-based lookup requires a suitable loader search
path; that is one implementation of library discovery. It is not an AMD mandate
to run an entire FINN build in another Python interpreter.

Relevant source locations:

- `finn_xsi/finn_xsi/sim_engine.py`: `SimEngine.__init__` constructs both objects.
- `finn_xsi/finn_xsi/adapter.py`: `get_simkernel_so` chooses a basename;
  `load_sim_obj` temporarily changes cwd to find the design.
- `finn_xsi/xsi_finn.cpp`: `SharedLibrary::load`, `Kernel::open`, `Kernel::close`.
- `finn_xsi/xsi_bind.cpp`: Python bindings and the process-global lifetime map.

**The stronger reasons for a simulation process.**

Both native libraries currently use `RTLD_LAZY | RTLD_GLOBAL`. FINN's own
`Kernel::close()` calls the public XSI close function, unloads the design, and
then looks up `svTypeInfo` inside the kernel and sets it to null. That is an
explicit interaction with kernel-global state beyond simply disposing a Python
object. The kernel library handle remains a member until its owner is destroyed.
The pybind lifetime map can retain that owner; deterministic design close and
actual library unloading are distinct operations.

The Linux loader documentation describes startup-time `LD_LIBRARY_PATH` lookup,
global symbol visibility, dependency loading and reference-counted unloading.
Returning from `dlclose` is not a general promise that an entire dependency graph
has been reset. [Linux dlopen documentation](https://man7.org/linux/man-pages/man3/dlopen.3.html).

AMD documents `xsi_run` as blocking on the calling thread. FINN's binding does not
release the GIL around that call. Its cycle watchdog runs outside the native call,
so it cannot enforce a deadline while a native call fails to return. An external
supervisor can enforce elapsed-time limits and report a crash without destroying
the Python build process. This is a design consequence, not a claim that a real
AMD hang or crash was reproduced here.
[AMD xsi_run](https://docs.amd.com/r/2024.1-English/ug900-vivado-logic-simulation/xsi_run),
[pybind11 GIL behavior](https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil).

No supported multi-version or concurrent-instance guarantee was found in the AMD
pages inspected. Absence of a guarantee is not proof of impossibility. The local
code and loader behavior are sufficient to reject accidental shared-process
version isolation as an untested assumption.

**Reproducible loader experiments.**

Run `python3 docs/experiments/xsi_loader_probe.py`. It requires Linux/glibc and GCC,
uses synthetic libraries written for this experiment, and creates all generated
files in a new temporary directory. It prints the results and artifact location.
No AMD headers, libraries, source, licence or hardware are involved.

Observed on glibc 2.39:

| Experiment | Result |
| --- | --- |
| Set `LD_LIBRARY_PATH` inside running Python, then load an unsearched basename | Fails |
| Supply the path when starting Python, then load that basename | Succeeds |
| Load a simple library by absolute path without a loader-path variable | Succeeds |
| Load a library by absolute path, with an unfindable transitive dependency | Fails |
| Give that library a `$ORIGIN` RUNPATH | Succeeds |
| Load two versions by absolute path, both with correct RUNPATH, using `RTLD_GLOBAL` | Both use version A's dependency: `[101, 101]` |
| Repeat using `RTLD_LOCAL` | Same dependency reuse: `[101, 101]` |
| Run each version in a fresh process with its selected environment | Correct independent values: `[101, 202]` |
| Load each version in a separate `dlmopen` namespace | Correct independent values: `[101, 202]` |

These distinguish resource discovery from isolation. Absolute paths and RUNPATH
are useful, but do not themselves isolate dependency versions. The namespace
experiment shows that in-process linker isolation is possible for the stand-ins;
it does not establish compatibility with AMD's C++ runtime, plugins or internal
loading behavior. It also supplies no process-level crash containment.

**FINN already has two simulation execution models.**

`rtlsim_exec_cppxsi` compiles a standalone `rtlsim_xsi` executable and launches it
with a child loader path. `set_fifo_depths.xsi_fifosim` uses this for sizing and
performance measurements. This already places the native kernel in a separate
process. It does not need a surrounding whole-build Python worker for that reason.
Its current implementation asserts `dummy_data_mode`; its documented real-input
branch is not implemented. It cannot simply replace all functional simulations.

`rtlsim_exec_finnxsi`, single-node operations and characterization use the Python
`SimEngine`, with Python driving streams, AXI transactions and hooks. These are the
paths that need a carefully chosen simulation-session boundary.

```text
Build process: Python transformations, code generation, result handling
    |
    +-- Vivado/HLS/xelab processes
    |
    +-- Simulation session process
            |
            +-- Selected environment established before exec
            +-- One selected AMD kernel and compatible compiled design
            +-- C++ harness OR Python SimEngine/testbench
            +-- Entire clock/port loop stays inside this process
            +-- Close the design; return data, metrics and artifact paths
            +-- Exit
```

A session can include multiple input frames, resets, initialization, readback and
characterization passes for the same design. It should not create a process for
every cycle or send every port access over IPC. Start with one process per session;
only introduce reuse across requests after measuring and validating its lifecycle.
A new executable image with a supplied environment is required: a plain fork of a
process with native state loaded is not that boundary. Likewise, setting
`LD_LIBRARY_PATH` inside a multiprocessing-spawn worker is too late for the
basename lookup demonstrated here; supply it at exec.

**The API boundary requires deliberate design.**

The parent can pass compiled-design paths, a tool selection/identity, input arrays
or buffer files, stream descriptions, reset/clock settings, tracing choices,
AXI initialization data and timeout settings. Results can include output arrays,
cycle counts, transaction traces, readbacks, status and log/waveform paths. Bulk
arrays should use files/memory mapping or another measured bulk transport; keep
live `Kernel`, `Design` and `Port` objects inside the session.

Existing hooks are the substantial migration issue. The URAM test in
`tests/fpgadataflow/test_fpgadataflow_mvau.py` defines closures that write weights
before a run and mutate a parent list with readbacks afterwards. MLO also builds
closures from model data. Those semantics do not transparently survive a process
boundary. Built-in AXI operations should become explicit input/result data;
custom testbench code must execute inside the session through a defined callable
or module entry contract with explicit arguments and results. Do not silently
pickle arbitrary closures or expose a per-port remote proxy to preserve an
accidental API shape.

**Comparison of candidate designs.**

| Candidate | Assessment |
| --- | --- |
| Fix paths, keep XSI in-process | Solves discovery; valid for a deliberately constrained single-runtime embedding, but does not provide version, crash or blocking-call isolation. |
| Use `dlmopen` namespaces | Synthetic dependency isolation works. Real vendor ABI/loading compatibility is unverified; considerably more linker-specific engineering and no crash boundary. |
| Isolate the whole build | Contains native state relative to the original caller, but unrelated build stages inherit the cost/constraints and simulation sessions still share a native process. Too broad as the permanent design. |
| Isolate a simulation session | Recommended. Boundary matches the native runtime's lifetime and failure behavior. Reuse the existing C++ harness and retain Python testbench behavior inside a worker where needed. |

**Required cleanup, independent of isolation.**

- Resolve configuration-relative paths once at the directory API boundary and
  pass paths explicitly. Remove parent/global `os.chdir` dependence.
- Separate XSI compilation/discovery helpers from the Python native bridge. Today
  `_load_modules` can make launching xelab depend on the bridge being importable,
  even for the standalone C++ route.
- Remove import-time `xsi.is_available()` decisions from ordinary build modules.
  They can load the bridge eagerly and leave module-level `finnxsi = None` after
  capabilities change. Probe actual operation prerequisites on demand.
- Guarantee design close on all normal error paths. The stitched Python route
  currently lacks a `finally` around its full reset/hook/run/close sequence. A
  forced process kill can leave traces partial and must report them as such.
- Verify compiled-artifact compatibility with selected tools, source and Python
  ABI as applicable. A fresh process cannot make a stale `xsi.so` or `xsimk.so`
  compatible. Existence/importability alone is insufficient evidence.
- Correct the native-setup check: `setup-local.sh` currently uses `finn.xsi.setup
  --check` as evidence that an extension exists, but that option checks only
  prerequisites. This is a concrete regression in the preceding changes, not
  evidence for a larger process wrapper.

Before shipping the revised boundary, compare real functional outputs, cycle
counts, AXI write/readback, MLO memory behavior and traces with the current engine;
verify exceptions, native failures, timeout cleanup, sequential and concurrent
sessions, and supported installation/version combinations. Measure cold startup,
small-node workloads, large stitched workloads and bulk-transfer overhead before
choosing batching or worker reuse. Existing artifact layouts can remain intact.

No AMD installation is available in this environment. The findings above are
source inspection, vendor/loader documentation and nine passing synthetic loader
cases. Actual XSI correctness, performance and version concurrency have not been
validated. Only this document and its experiment were added for this investigation.
