"""Real FINN/QONNX uv probe inside the existing dependency image, offline."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(sys.argv[1])
UV = ROOT / "uv"
RESULTS = {"python": sys.version, "phases": []}


def run(args, name, *, cwd, env=None, check=True):
    started = time.monotonic()
    result = subprocess.run(list(map(str, args)), cwd=cwd, env=env, text=True, capture_output=True)
    (ROOT / (name + ".log")).write_text(result.stdout + result.stderr)
    RESULTS["phases"].append(
        {"name": name, "returncode": result.returncode, "seconds": time.monotonic() - started}
    )
    (ROOT / "docker-results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
    if check and result.returncode:
        raise RuntimeError(name + ": " + result.stdout[-2000:] + result.stderr[-2000:])
    return result


manifest = json.loads(Path("/opt/finn/wheelhouse.json").read_text())["wheels"]
dependencies = [f"{name}=={item['version']}" for name, item in manifest.items() if name != "qonnx"]
dependencies += ["finn", "qonnx"]
source_config = """
[tool.uv]
package = false
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
[tool.uv.sources]
finn = {path = "./finn", editable = true}
qonnx = {path = "./qonnx", editable = true}
"""
help_text = run([UV, "sync", "--help"], "sync-help", cwd=ROOT).stdout
RESULTS["uv"] = run([UV, "--version"], "runtime-uv-version", cwd=ROOT).stdout.strip()
RESULTS["no_install_local"] = "--no-install-local" in help_text
if not RESULTS["no_install_local"]:
    raise RuntimeError("This spike requires uv's generic --no-install-local operation")
partial = ["--no-install-local"]
offline = ["--offline", "--no-index", "--find-links", "/opt/finn/wheels", "--no-python-downloads"]
for pair in ("a", "b", "c"):
    project = ROOT / "pairs" / pair
    (project / "pyproject.toml").write_text(
        '[project]\nname = "finn-development"\nversion = "0.0.0"\n'
        'requires-python = ">=3.10,<3.11"\ndependencies = '
        + json.dumps(dependencies, indent=2)
        + "\n"
        + source_config
    )
    environment = dict(os.environ)
    prefix = ROOT / "environments" / pair
    environment.update(
        {
            "UV_CACHE_DIR": str(ROOT / "cache"),
            "UV_PROJECT_ENVIRONMENT": str(prefix),
            "UV_PYTHON": sys.executable,
            "UV_PYTHON_DOWNLOADS": "never",
            "UV_LINK_MODE": "copy",
        }
    )
    run([UV, "lock", *offline], pair + "-lock", cwd=project, env=environment)
    run(
        [UV, "sync", "--locked", *partial, *offline],
        pair + "-partial",
        cwd=project,
        env=environment,
    )
    run(
        [
            prefix / "bin/python",
            "-I",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('finn') is None; "
            "assert importlib.util.find_spec('qonnx') is None",
        ],
        pair + "-excluded",
        cwd=ROOT,
        env=environment,
    )
    run([UV, "sync", "--locked", *offline], pair + "-editable", cwd=project, env=environment)
    probe = (
        "import json; from importlib.metadata import distribution; "
        "import finn.util.uv_spike_marker as f, qonnx.uv_spike_marker as q; "
        f"assert (f.VALUE,q.VALUE)==({pair!r},{pair!r}); "
        "from finn.util.resources import resource_path; "
        "print(json.dumps({'finn':f.__file__,'qonnx':q.__file__,'rtl':resource_path('rtllib'),"
        "'finn_metadata':distribution('finn').read_text('direct_url.json'),"
        "'qonnx_metadata':distribution('qonnx').read_text('direct_url.json')}))"
    )
    data = run(
        [prefix / "bin/python", "-I", "-c", probe], pair + "-imports", cwd=ROOT, env=environment
    )
    RESULTS[pair] = json.loads(data.stdout)
    run(
        [prefix / "bin/python", "-m", "pip", "check"],
        pair + "-pip-check",
        cwd=ROOT,
        env=environment,
    )

source = ROOT / "pairs/a/qonnx/src/qonnx/uv_spike_marker.py"
replacement = source.with_suffix(".new")
replacement.write_text("VALUE = 'edited-a'\n")
replacement.replace(source)
for pair in ("a", "b", "c"):
    expected = "edited-a" if pair == "a" else pair
    run(
        [
            ROOT / "environments" / pair / "bin/python",
            "-I",
            "-c",
            f"from qonnx.uv_spike_marker import VALUE; assert VALUE == {expected!r}",
        ],
        pair + "-atomic-edit",
        cwd=ROOT,
    )
RESULTS["atomic_edit_isolated"] = True
(ROOT / "docker-results.json").write_text(json.dumps(RESULTS, indent=2) + "\n")
