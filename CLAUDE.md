# Working in this repository

## Never block on an RTL run

Anything that runs Vivado — `xelab`/XSI, IP packaging, synthesis — takes minutes
to tens of minutes, natively or through `./docker/run --fpga`. Start it in a
**background shell** and keep working; do not sit on a foreground call waiting
for it.

```
# right
Bash(run_in_background=true):
    PYTHONPATH=src:tests python -m kernels.rtlsim.mvau_assembly_numeric --case packed

# wrong
Bash(timeout=3000): python -m kernels.rtlsim...     # blocks the whole session
```

Have the runner `tee` to a gitignored `*.log` and propagate the child's exit
status, so a backgrounded run leaves a readable transcript and a truthful status.

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

### Running against FinnLib

The kernels compile against FinnLib, a separate repository that FINN takes as
the `finnlib` resource (`src/finn/_data/resources.toml`). Work against a clone:

```
export FINN_RESOURCES_FINNLIB=/path/to/finnlib       # native
./docker/run --volume /path/to/finnlib:/path/to/finnlib -- ...   # plus the same variable
```

In sbx, use the `finnlib` overlay (`docker/sbx/README.md`). Without the override
FINN uses the pinned commit, fetched over SSH into the resource cache. Move the
pin with `finn-resources update finnlib --ref BRANCH`, and only to commits that
exist on the remote.

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
process — a single configuration is already several simulations. Run each
compile+load+run in a fresh interpreter.

## Packages and gates

```
finn.core.space  <-  finn.dataflow  <-  finn.kernels  <-  finn.parked
```

- `finn.core.space` — the generic Space engine.
- `finn.dataflow` — canonical logical dataflow values (Regions, Networks, maps).
- `finn.kernels` — kernels bound to RTL/HLS sources.
- `finn.parked` — retired code kept as reference only. It is not tested, not
  shipped, and nothing live imports it.

Gates, in the project environment (`uv sync`, then `.venv/bin` on `PATH` or
`source scripts/activate.sh`):

```
bash scripts/check-space.sh
bash scripts/check-dataflow-design.sh
bash scripts/check-kernels.sh      # includes check-space; needs FinnLib
```

Run `ruff format` only on the paths you changed; formatting all of `src`
rewrites unrelated FINN files. The Space, dataflow and kernel packages are
formatted by ruff, the rest of FINN by black and isort (`.pre-commit-config.yaml`).
