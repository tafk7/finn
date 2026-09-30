# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic glibc loader experiments. No AMD code/libraries are used."""
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path


def compile_lib(src, target, *flags):
    subprocess.run(["gcc", "-shared", "-fPIC", str(src), "-o", str(target), *flags], check=True)


def child(code, library_path=None):
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "BASH_ENV")
    }
    if library_path is not None:
        env["LD_LIBRARY_PATH"] = str(library_path)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def main():
    if platform.system() != "Linux" or platform.libc_ver()[0] != "glibc":
        raise SystemExit("This experiment targets Linux/glibc and requires gcc")
    root = Path(tempfile.mkdtemp(prefix="finn-xsi-loader-probe-"))
    for name, number in [("a", 101), ("b", 202)]:
        d = root / name
        d.mkdir(exist_ok=True)
        (d / "simple.c").write_text(f"int identity(void) {{return {number};}}\n")
        compile_lib(d / "simple.c", d / "libfinn_probe_simple.so")
        (d / "dep.c").write_text(f"int dependency_identity(void) {{return {number};}}\n")
        compile_lib(d / "dep.c", d / "libfinn_probe_dep.so", "-Wl,-soname,libfinn_probe_dep.so")
        (d / "kernel.c").write_text(
            "extern int dependency_identity(void); int run(void) {return dependency_identity();}\n"
        )
        compile_lib(
            d / "kernel.c", d / "libfinn_probe_kernel.so", "-L" + str(d), "-lfinn_probe_dep"
        )
        compile_lib(
            d / "kernel.c",
            d / "libfinn_probe_kernel_rpath.so",
            "-L" + str(d),
            "-lfinn_probe_dep",
            "-Wl,-rpath,$ORIGIN",
        )
    prefix = "import ctypes,os,json; "
    results = {}
    attempt = (
        "\ntry:\n x=ctypes.CDLL(target,mode=os.RTLD_NOW|os.RTLD_GLOBAL); "
        'print(json.dumps({"ok": True, "value": x.'
    )
    finish = '()}))\nexcept OSError as e:\n print(json.dumps({"ok": False,"error":str(e)}))'
    results["set_env_after_start_bare_load"] = child(
        prefix
        + f'os.environ["LD_LIBRARY_PATH"]={str(root/"a")!r}; target="libfinn_probe_simple.so";'
        + attempt
        + "identity"
        + finish
    )
    results["env_before_start_bare_load"] = child(
        prefix + 'target="libfinn_probe_simple.so";' + attempt + "identity" + finish, root / "a"
    )
    results["absolute_simple_without_env"] = child(
        prefix
        + f'target={str(root/"a/libfinn_probe_simple.so")!r};'
        + attempt
        + "identity"
        + finish
    )
    results["absolute_kernel_missing_transitive_dep"] = child(
        prefix + f'target={str(root/"a/libfinn_probe_kernel.so")!r};' + attempt + "run" + finish
    )
    results["absolute_kernel_with_runpath"] = child(
        prefix
        + f'target={str(root/"a/libfinn_probe_kernel_rpath.so")!r};'
        + attempt
        + "run"
        + finish
    )
    for mode in ["RTLD_GLOBAL", "RTLD_LOCAL"]:
        results["two_absolute_kernels_" + mode] = child(
            prefix + f'first=ctypes.CDLL({str(root/"a/libfinn_probe_kernel_rpath.so")!r},'
            f"mode=os.RTLD_NOW|os.{mode}); "
            f'second=ctypes.CDLL({str(root/"b/libfinn_probe_kernel_rpath.so")!r},'
            f"mode=os.RTLD_NOW|os.{mode}); "
            "print(json.dumps([first.run(),second.run()]))"
        )
    results["fresh_process_each_version"] = [
        child(
            prefix
            + f'target={str(root/n/"libfinn_probe_kernel.so")!r};'
            + attempt
            + "run"
            + finish,
            root / n,
        )["value"]
        for n in ["a", "b"]
    ]
    assert not results["set_env_after_start_bare_load"]["ok"]
    assert results["env_before_start_bare_load"]["value"] == 101
    assert results["absolute_simple_without_env"]["value"] == 101
    assert not results["absolute_kernel_missing_transitive_dep"]["ok"]
    assert results["absolute_kernel_with_runpath"]["value"] == 101
    assert results["two_absolute_kernels_RTLD_GLOBAL"] == [101, 101]
    assert results["two_absolute_kernels_RTLD_LOCAL"] == [101, 101]
    assert results["fresh_process_each_version"] == [101, 202]
    namespace_probe = """import ctypes, os, json
libc=ctypes.CDLL(None)
libc.dlmopen.argtypes=[ctypes.c_long,ctypes.c_char_p,ctypes.c_int]
libc.dlmopen.restype=ctypes.c_void_p
libc.dlsym.argtypes=[ctypes.c_void_p,ctypes.c_char_p]
libc.dlsym.restype=ctypes.c_void_p
libc.dlerror.restype=ctypes.c_char_p
values=[]
handles=[]
for path in PATHS:
 h=libc.dlmopen(-1,path.encode(),os.RTLD_NOW|os.RTLD_LOCAL)
 if not h: raise RuntimeError(libc.dlerror())
 handles.append(h)
 f=libc.dlsym(h,b"run")
 if not f: raise RuntimeError(libc.dlerror())
 values.append(ctypes.CFUNCTYPE(ctypes.c_int)(f)())
print(json.dumps(values))
"""
    namespace_probe = namespace_probe.replace(
        "PATHS", repr([str(root / name / "libfinn_probe_kernel_rpath.so") for name in ("a", "b")])
    )
    results["separate_dlmopen_namespaces"] = child(namespace_probe)
    assert results["separate_dlmopen_namespaces"] == [101, 202]
    (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    print("Experiment artifacts:", root)


if __name__ == "__main__":
    main()
