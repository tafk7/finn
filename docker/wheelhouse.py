"""Audit resolved wheels and write the offline development manifest at image build."""

import hashlib
import json
import subprocess
import sys
import zipfile
from email.parser import BytesParser
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pathlib import Path


def manifest(directory, sources):
    records = {}
    for wheel in sorted((directory / "wheels").glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            names = [
                n
                for n in archive.namelist()
                if n.count("/") == 1 and n.endswith(".dist-info/METADATA")
            ]
            if len(names) != 1:
                raise ValueError(f"Invalid wheel metadata: {wheel.name}")
            metadata = BytesParser().parsebytes(archive.read(names[0]))
        name = canonicalize_name(metadata["Name"])
        if name == "finn" or name in records:
            raise ValueError(f"FINN or duplicate distribution in dependency wheelhouse: {name}")
        records[name] = {
            "version": metadata["Version"],
            "wheel": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
            "requires": metadata.get_all("Requires-Dist", []),
        }
    for name, record in records.items():
        for raw in record["requires"]:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            dependency = canonicalize_name(requirement.name)
            if dependency == "finn":
                raise ValueError(f"{name} requires FINN; install it at the application step")
            if (
                dependency not in records
                or records[dependency]["version"] not in requirement.specifier
            ):
                raise ValueError(f"Unsatisfied dependency in wheelhouse: {name}: {raw}")
    for required in ("pip", "setuptools", "wheel", "build", "setuptools-scm"):
        if required not in records:
            raise ValueError(f"Missing editable build requirement: {required}")
    revisions = {}
    for source in sorted(sources.iterdir()):
        if not (source / ".git").exists():
            continue
        revisions[source.name] = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip()
    (directory / "development-requirements.txt").write_text(
        "# Resolved for this image's Python/platform. FINN is installed explicitly.\n"
        + "".join(
            f"{name}=={record['version']} --hash=sha256:{record['sha256']}\n"
            for name, record in sorted(records.items())
        )
    )
    (directory / "wheelhouse.json").write_text(
        json.dumps({"sources": revisions, "wheels": records}, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    manifest(Path(sys.argv[1]), Path(sys.argv[2]))
