"""Compare complete values; normalize only enumerated relocations and generated names."""

import dataclasses
import importlib
import json
import gzip
from pathlib import Path
import tempfile

from finn.kernels.artifacts.build import module_build_fingerprint, prepare_module_build
from finn.kernels.artifacts.store import ArtifactStore
from finn.kernels.resources import resource_root, template_root
from qonnx.core.datatype import DataType

EVIDENCE = Path(__file__).parent


def read_capture(name):
    plain = EVIDENCE / name
    return json.loads(
        plain.read_bytes()
        if plain.exists()
        else gzip.decompress(plain.with_suffix(".json.gz").read_bytes())
    )


before = read_capture("before-constructions.json")
after = read_capture("after-constructions.json")
modules = json.loads((EVIDENCE / "relocation-map.json").read_text())["modules"]
modules["finn.dataflow.kernels.matmul.base.DspBlock"] = "finn.kernels.target.DspBlock"
path_map = {
    f"src/finn/dataflow/kernels/resources/{name}": name
    for name in ("dotp_axi.sv", "cyclic_stream.sv")
}


def identity(name):
    for old, new in sorted(modules.items(), key=lambda x: -len(x[0])):
        if name == old or name.startswith(old + "."):
            return new + name[len(old) :]
    return name


def relocate(v):
    if isinstance(v, list):
        return [relocate(x) for x in v]
    if not isinstance(v, dict):
        return v
    result = {k: relocate(x) for k, x in v.items()}
    for key in ("type", "enum"):
        if key in result:
            result[key] = identity(result[key])
    if result.get("type") == "finn.kernels.artifacts.contributions.CopiedSource":
        fields = result["fields"]
        if fields["root"] == "finn":
            assert fields["path"] in path_map, fields
            fields["root"] = "kernels"
            fields["path"] = path_map[fields["path"]]
    return result


def typename(name):
    module, _, attr = name.rpartition(".")
    return getattr(importlib.import_module(module), attr)


def decode(v):
    if isinstance(v, list):
        return tuple(decode(x) for x in v)
    if not isinstance(v, dict):
        return v
    if "enum" in v:
        return typename(v["enum"])[v["name"]]
    if "type" in v:
        return typename(v["type"])(**{k: decode(x) for k, x in v["fields"].items()})
    if "qonnx_dtype" in v:
        return DataType[v["qonnx_dtype"]]
    return {k: decode(x) for k, x in v.items()}


assert before.keys() == after.keys()
changes = {}
with tempfile.TemporaryDirectory(prefix="relocation-comparison-") as temporary:
    store = ArtifactStore(Path(temporary))
    for key, old in before.items():
        new = after[key]
        requirements = relocate(old["requirements"])
        assert requirements == new["requirements"], (key, "requirements")
        assert relocate(old["assembly"]) == new["assembly"], (key, "assembly")
        req = decode(requirements)
        assert module_build_fingerprint(req) == new["fingerprint"], (
            key,
            "fingerprint recomputation",
        )
        prepared = prepare_module_build(
            req,
            roots={"kernels": resource_root(), "finnlib": Path.cwd() / "deps/finnlib"},
            template_roots=(template_root(),),
            blobs=store,
        )
        assert prepared.abi.entry_point == new["entry_point"], (
            key,
            "generated name recomputation",
        )
        abi = relocate(old["prepared_abi"])
        assert abi["fields"]["entry_point"] == old["entry_point"]
        abi["fields"]["entry_point"] = new["entry_point"]
        assert abi == new["prepared_abi"], (key, "ABI")
        sources = {}
        for path, contents in old["sources"].items():
            new_path = path_map.get(path, path)
            if path == old["entry_point"] + ".sv":
                new_path = new["entry_point"] + ".sv"
                # The renderer places its generated name only in the module declaration.
                old_declaration = "module " + old["entry_point"]
                assert contents.count(old_declaration) == 1
                contents = contents.replace(
                    old_declaration, "module " + new["entry_point"], 1
                )
            sources[new_path] = contents
        assert sources == new["sources"], (key, "emitted source bytes")
        order = [
            new["entry_point"] + ".sv"
            if p == old["entry_point"] + ".sv"
            else path_map.get(p, p)
            for p in old["source_order"]
        ]
        assert order == new["source_order"], (key, "prepared compilation order")
        changes[key] = {
            "old_fingerprint": old["fingerprint"],
            "new_fingerprint": new["fingerprint"],
            "old_entry_point": old["entry_point"],
            "new_entry_point": new["entry_point"],
            "source_count": len(sources),
        }
(EVIDENCE / "equivalence-results.json").write_text(
    json.dumps(
        {
            "cases": len(changes),
            "equal_after_explicit_relocation": True,
            "normalization": {
                "module_map": "relocation-map.json",
                "source_paths": path_map,
                "source_root": {"finn": "kernels"},
                "generated_names": "recomputed from relocated requirements; only top module declaration renamed",
                "compile_order": "ordered rendered source paths compared exactly after path/name relocation",
            },
            "results": changes,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)
print(
    f"{len(changes)} complete configurations equivalent; fingerprints and generated names recomputed"
)
