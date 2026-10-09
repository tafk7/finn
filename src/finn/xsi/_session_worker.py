"""One simulation per exec. Native imports happen only in this worker."""

import importlib.util
import json
import numpy as np
import os
import sys
import sysconfig
import time
from pathlib import Path

from finn.xsi._artifacts import digest, validate_record
from finn.xsi._session import pack_streams, unpack_streams


def stream_testbench(sim, io, request):
    from finn_xsi.adapter import rtlsim_multi_io  # noqa: PLC0415

    sim.do_reset(n_cycles=request["reset_cycles"])
    sim.run()
    counts = request["output_counts"]
    cycles = rtlsim_multi_io(
        sim,
        io,
        next(iter(counts.values())) if len(counts) == 1 else counts,
        sname=request["stream_suffix"],
        liveness_threshold=request["liveness_threshold"],
    )
    return {"cycles": cycles}


def main(directory):
    started = time.monotonic()
    directory = Path(directory).resolve(strict=True)
    request = json.loads((directory / "request.json").read_text())
    if request["python_abi"] != sysconfig.get_config_var("SOABI"):
        raise ValueError("Selected worker Python ABI differs from the bridge selection")
    for key, expected in request["artifacts"].items():
        if digest(request[key]) != expected:
            raise ValueError(f"Selected {key} changed before session execution")
    for key in ("bridge", "design"):
        validate_record(request[key], kind=key, tool=request["tool_identity"])
    # The loader environment was already supplied before exec, never patched here.
    spec = importlib.util.spec_from_file_location("xsi", request["bridge"])
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load selected XSI bridge")
    bridge = importlib.util.module_from_spec(spec)
    sys.modules["xsi"] = bridge
    spec.loader.exec_module(bridge)
    from finn_xsi.sim_engine import SimEngine  # noqa: PLC0415

    trace = str(directory / "trace.wdb") if request["trace"] else None
    # Preserve the compiled working directory for vendor-relative HDL inputs.
    if Path.cwd() != Path(request["design_cwd"]):
        raise ValueError("Site launcher did not preserve the compiled design cwd")
    sim = SimEngine(request["kernel"], request["design"], str(directory / "xsi.log"), trace)
    try:
        if trace:
            sim.top.trace_all()
        testbench = stream_testbench
        if request["testbench"]:
            spec = importlib.util.spec_from_file_location(
                "finn_session_testbench", request["testbench"]
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            testbench = module.run
        io = {
            "inputs": unpack_streams(directory / "inputs.npz"),
            "outputs": {name: [] for name in request["output_counts"]},
        }
        metrics = testbench(sim, io, request)
        if {key: len(value) for key, value in io["outputs"].items()} != request["output_counts"]:
            raise ValueError("Testbench did not produce the requested output buffers")
        from finn_xsi.adapter import close_rtlsim  # noqa: PLC0415

        close_rtlsim(sim)
    finally:
        # No additional simulator cycle is needed to close/flush on exceptions.
        # Explicit close breaks the native/port reference cycle even if callbacks fail.
        sim.top.close()
    np.savez(directory / "outputs.npz", **pack_streams(io["outputs"]))
    result = {
        "status": "success",
        "session": request["session"],
        "artifacts": request["artifacts"],
        "metrics": {
            **metrics,
            "worker_seconds": time.monotonic() - started,
        },
        "trace": trace,
    }
    temporary = directory / "result.pending.json"
    temporary.write_text(json.dumps(result, sort_keys=True) + "\n")
    os.replace(temporary, directory / "result.json")


if __name__ == "__main__":
    main(sys.argv[1])
