# Working in this repository

## Never block on a Docker RTL run

Anything that goes through `run-docker.sh` — Vivado, `xelab`/XSI, IP packaging,
synthesis — takes minutes to tens of minutes. Start it in a **background
shell** and keep working; do not sit on a foreground call waiting for it.

```
# right
Bash(run_in_background=true):
    FINN_DOCKER_EXTRA="..." bash run-docker.sh bash tests/dataflow/rtlsim/run_composed_equiv.sh

# wrong
Bash(timeout=3000): bash run-docker.sh ...          # blocks the whole session
```

The runner scripts under `tests/dataflow/rtlsim/` already `tee` to a
gitignored `*.log` and propagate the child's exit status, so a backgrounded
run leaves a readable transcript and a truthful status.

**While it runs**, do the work that does not need Vivado: unit tests, lint,
mypy, the next increment. The completion notification will find you — do **not**
`sleep` waiting for it, or poll it on a timer. A foreground `sleep 300` blocks
the session just as thoroughly as the foreground run you were avoiding.

When the notification arrives, read the log and report what it actually says.
A run that was never checked is not evidence.

**Never report an RTL result you did not see.** If a background run is still
going when the turn ends, say so.

### Delegate a long sweep to an agent

For a matrix of Vivado runs, or when the analysis of the output is itself
substantial, launch an agent instead: it keeps the multi-hundred-line Vivado
transcripts out of the main context and reports the conclusion. One agent per
independent sweep.

### Do not run two `run-docker.sh` invocations at once

They race and one of them dies. `docker/finn_entrypoint.sh` moves
`deps/qonnx/pyproject.toml` aside while it runs `pip install -e`, then moves it
back. The repository is bind-mounted into every container, so that one file is
shared: a second container entering the same window fails with

```
mv: cannot stat '.../deps/qonnx/pyproject.toml': No such file or directory
```

and exits before producing a fixture log at all. The failure names qonnx and
looks nothing like a concurrency problem, which is why it is written down here.

So fixtures 5 and 6 are **sequential**, not parallel — start the second only
after the first has exited. Backgrounding them both at once is the natural
thing to do and the wrong one.

### Running against FinnLib

The decomposed MVAU compiles against FinnLib, which is a separate repository.
`fetch-repos.sh` pins it under `deps/finnlib`; a local working clone is reached
by mounting it and setting `FINNLIB_ROOT`:

```
FINN_DOCKER_EXTRA="-v /path/to/finnlib:/path/to/finnlib -e FINNLIB_ROOT=/path/to/finnlib "
```

Pin only commits that exist on the remote — `fetch-repos.sh` clones and then
checks out, so a local unpushed hash fails the fetch for everyone.

## Vivado licensing on a development machine

Versal parts (`xcvc1902`, ...) need a licensed `Synthesis` feature that a plain
development machine does not have; `xelab`/XSI simulation of the same part
works fine. A fixture that synthesizes must therefore report an unlicensed
device as **skipped**, and a run where everything skipped is not a pass — it
means the question was never asked.

## XSI keeps state across simulations

`close_rtlsim` does not fully reset it, and the failure is per *load*, not per
object:

- loading the same compiled object twice in a process → SIGSEGV;
- the **third** `load_sim_obj` in a process, of any objects → hangs, with no
  diagnostic and without the watchdog firing.

So the rule is **one simulation per process**, not one configuration per
process — a single configuration is already several simulations. The harness
marshals each compile+load+run into a fresh interpreter over a JSON request
(`--simulate <request> --out <response>`) and the parent never loads a
simulation object at all. Keep that shape when adding fixtures.

## Test and lint commands

```
FINN_ROOT=$PWD PYTHONPATH=src:tests:deps/qonnx/src python -m pytest tests/dataflow
ruff check src/finn/dataflow tests/dataflow
ruff format --check src/finn/dataflow tests/dataflow
env -u PYTHONPATH MYPYPATH=src:tests mypy --strict -p finn.dataflow
```

`scripts/check-dataflow-design.sh` runs all of these together, and is the gate
for changes under `src/finn/dataflow/`.

**mypy must not see `deps/qonnx/src` on `PYTHONPATH`.** With qonnx importable,
its missing `py.typed` turns every qonnx import into a different error code and
the `# type: ignore[import-not-found]` comments read as unused — dozens of
false positives. Hence `env -u PYTHONPATH`.
