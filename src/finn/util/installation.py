"""Inspect selected distributions without changing installation or import state."""
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path


def main():
    for name in sys.argv[1:] or ["finn", "qonnx"]:
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            print(json.dumps({"distribution": name, "installed": False}))
            continue
        module = name.replace("-", "_")
        spec = importlib.util.find_spec(module)
        locations = (
            [
                str(Path(path).resolve())
                for path in (spec.submodule_search_locations or [])
                if Path(path).is_dir()
            ]
            if spec
            else []
        )
        installation = dist.read_text("direct_url.json")
        info = {
            "distribution": name,
            "version": dist.version,
            "import_paths": locations or ([spec.origin] if spec else []),
            "installation": json.loads(installation) if installation else None,
        }
        if name == "finn":
            provenance = dist.locate_file("finn/_build_info.json")
            info["build"] = json.loads(provenance.read_text()) if provenance.is_file() else None
        print(json.dumps(info, sort_keys=True))


if __name__ == "__main__":
    main()
