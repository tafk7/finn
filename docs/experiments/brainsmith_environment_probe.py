# SPDX-License-Identifier: BSD-3-Clause
"""Exercise only Brainsmith's exporter/generator, with fake settings and config."""
import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path


def main():
    sys.dont_write_bytecode = True
    repo = (
        Path(sys.argv[1]).resolve()
        if len(sys.argv) > 1
        else Path(__file__).resolve().parents[3] / "brainsmith"
    )
    root = Path(tempfile.mkdtemp(prefix="brainsmith-env-probe-"))
    # Load the real exporter directly, without importing the application/dependencies.
    package = types.ModuleType("bs_env_probe")
    package.__path__ = [str(repo / "brainsmith/settings")]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        "bs_env_probe.env_export", repo / "brainsmith/settings/env_export.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Run the exact activation-generation methods with a minimal config fixture.
    tree = ast.parse((repo / "brainsmith/settings/schema.py").read_text())
    config_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SystemConfig"
    )
    methods = [
        node
        for node in config_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("generate_activation_script", "_generate_cleanup_code")
    ]
    namespace = {"__package__": "bs_env_probe", "Path": Path}
    exec(
        compile(
            ast.Module(body=methods, type_ignores=[]),
            str(repo / "brainsmith/settings/schema.py"),
            "exec",
        ),
        namespace,
    )
    ConfigFixture = type(
        "ConfigFixture",
        (types.SimpleNamespace,),
        {method.name: namespace[method.name] for method in methods},
    )
    vivado = root / "selected/Vivado/2024.2"
    vivado.mkdir(parents=True)
    (vivado / "settings64.sh").write_text("export XSI_PROBE_SETTINGS_CALLED=1\n")
    config = ConfigFixture(
        vivado_path=vivado,
        vivado_ip_cache=None,
        vitis_path=None,
        vitis_hls_path=None,
        vendor_platform_paths="",
        deps_dir=root / "deps",
        netron_port=8080,
        finn_root=root / "finn",
        finn_build_dir=root / "build",
        finn_deps_dir=root / "deps",
        default_workers=2,
        build_dir=root / "build",
        bsmith_dir=repo,
        project_dir=root,
    )
    old = "/tools/Xilinx/Vivado/old/lib/lnx64.o"
    os.environ["LD_LIBRARY_PATH"] = old
    before = dict(os.environ)
    exported = module.EnvironmentExporter(config).to_env_dict()
    assert dict(os.environ) == before
    script = config.generate_activation_script(root / "env.sh")
    env = {k: v for k, v in os.environ.items() if k not in ("BASH_ENV", "ENV", "LD_PRELOAD")}
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            'source "$1"; "$2" -I -c \'import os,json; '
            'keys=["LD_LIBRARY_PATH","XILINX_VIVADO","XSI_PROBE_SETTINGS_CALLED"]; '
            "print(json.dumps({k:os.environ.get(k) for k in keys}))'",
            "probe",
            str(script),
            sys.executable,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(result.stdout)
    assert exported["LD_LIBRARY_PATH"].split(":")[0] == old
    assert observed["LD_LIBRARY_PATH"].split(":")[0] == old
    assert observed["XSI_PROBE_SETTINGS_CALLED"] == "1"
    assert observed["XILINX_VIVADO"] == str(vivado)
    output = {
        "exporter_preserves_caller_environment": dict(os.environ) == before,
        "exported_ld_library_path": exported["LD_LIBRARY_PATH"],
        "activated_child": observed,
        "artifacts": str(root),
    }
    print(json.dumps(output, indent=2))
    (root / "results.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
