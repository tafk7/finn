# `tests/hardware/` — Docker/Vivado validation harnesses

These are **`__main__` scripts, not pytest tests.** They require the FINN Docker
container (Vivado / HLS / the pre-built `finn_xsi/xsi.so`), so they live outside
`tests/unit/` and are excluded from the venv-pure suite by construction. Each is driven by
a sibling `run_*.sh` wrapper via `run-docker.sh` from the FINN root.

## What each harness proves

| Script | Runner | Proves |
|---|---|---|
| `diff_mvau_emit_vs_finn.py` | `run_diff.sh` | Our hermetic MVAU emit is **byte-equivalent** to FINN's own `generate_hdl` / `make_weight_file` codegen on a single-MVAU model. |
| `elaborate_mvau_emit.py` | `run_elaborate.sh` | The emitted `mvau_top.v` + its declared `StaticFile` source set **elaborates** (`xvlog -sv` + `xelab`) — the cheap "sources are complete" check. |
| `elaborate_composed_mvau.py` | `run_elaborate_composed.sh` | The **stitched** composed MVAU (compute wrapper + memstream wrapper + stitch IPI) builds a structurally valid Vivado block design. |
| `rtlsim_composed_mvau.py` | `run_rtlsim_composed.sh` | **The prize** — rtlsim of the composed, stitched MVAU computes a correct matmul end-to-end. |

## How to run

From the FINN root, inside (or launching) the container:

```bash
bash run-docker.sh bash src/finn/kernels/tests/hardware/run_rtlsim_composed.sh
```

Each runner `cd`s to `$FINN_ROOT`, runs its script, and tees output to a gitignored
`_*_out.txt` scratch file at the repo root.

## Vivado-2025.2 toolchain accommodations

Two documented workarounds live inline in the composed harnesses. They are **toolchain
realities, not model defects** — captured here so a reader doesn't mistake them for design
smell. Consolidating them into shared shims is deferred (later Stage 0).

1. **Sibling DSP core.** FINN's un-split `mvu_vvu_axi.sv` references both DSP cores
   (`genINT8 → mvu_vvu_8sx9_dsp58`), so the composed source list must add the sibling
   `finn-rtllib/mvu/mvu_vvu_8sx9_dsp58.sv` explicitly.
   (`elaborate_composed_mvau.py:~93`, `rtlsim_composed_mvau.py:~87`.)
2. **`xelab --relax`.** `axilite.sv` has a legal use-before-declare that Vivado 2025.2's
   project synthesis rejects but `xelab --relax` tolerates (vs FINN's 2022.x baseline).
   The composed region is therefore elaborated **flat** with `--relax` after the block
   design is built. (`elaborate_composed_mvau.py:~137–167`.)
