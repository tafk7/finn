#!/usr/bin/env python3
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""Move a git resource of src/finn/resources.toml to a new commit, for maintainers.

    scripts/update-resource.py hlslib --ref main      # resolve, fetch, digest, rewrite
    scripts/update-resource.py --digest DIR           # print a directory's tree digest

The new commit's tree is fetched and digested, cached as an entry, and its commit
and digest replace the old ones in the declaration; nothing else in the file changes.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from finn import resources  # noqa: E402
from finn.resources import _declare, _store  # noqa: E402

_HEADER = re.compile(r"\s*\[(?!\[)\s*(?P<key>[^\]]+?)\s*\]\s*(#.*)?")


def resolve(url, ref):
    """The commit a branch, tag or commit id names in a remote repository."""
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    out = subprocess.run(
        ["git", "ls-remote", url],
        env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    refs = {name: sha for sha, name in (line.split("\t") for line in out.splitlines())}
    # A peeled tag (^{}) names the commit, not the tag object.
    for name in (ref, f"refs/heads/{ref}", f"refs/tags/{ref}^{{}}", f"refs/tags/{ref}"):
        if name in refs:
            return refs[name]
    sys.exit(f"{url} has no branch or tag {ref!r}")


def rewrite(path, name, values):
    """Set string fields of [resources.<name>] in a TOML file, changing nothing else:
    the result is re-parsed and must differ in exactly those values."""
    text = Path(path).read_text()
    lines = text.splitlines(keepends=True)
    inside, changed = False, set()
    for number, line in enumerate(lines):
        header = _HEADER.fullmatch(line.rstrip("\n"))
        if header:
            inside = [k.strip().strip("\"'") for k in header["key"].split(".")] == [
                "resources",
                name,
            ]
            continue
        field = re.match(r'(\s*(\w+)\s*=\s*)"[^"\\]*"', line)
        if inside and field and field[2] in values:
            lines[number] = f'{field[1]}"{values[field[2]]}"{line[field.end():]}'
            changed.add(field[2])
    expected = tomllib.loads(text)
    expected["resources"][name].update(values)
    new_text = "".join(lines)
    if changed != set(values) or tomllib.loads(new_text) != expected:
        sys.exit(f"{path}: cannot rewrite {sorted(values)} of [resources.{name}]; edit it by hand")
    Path(path).write_text(new_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?")
    parser.add_argument("--ref", default="HEAD", help="branch, tag or commit (default: HEAD)")
    parser.add_argument("--digest", metavar="DIR", help="print the tree digest of DIR and exit")
    args = parser.parse_args()
    if args.digest:
        print(_store.tree_digest(args.digest))
        return
    resource = resources.declarations().get(args.name)
    if resource is None or not resource.git:
        sys.exit(f"{args.name!r} is not a git resource of {_declare.FINN_FILE}")
    commit = resolve(resource.git, args.ref)
    candidate = replace(resource, commit=commit, digest="sha256:" + "0" * 64)
    root = _store.fetch_root()
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root, prefix=f".update-{args.name}.") as work:
        tree = _store.assemble(candidate, Path(work))
        new = replace(candidate, digest=_store.tree_digest(tree))
        with _store.locked(root, new.entry) as entry:
            if not _store.complete(entry, new.digest):
                _store.publish(tree, entry, new.digest)
    if (resource.commit, resource.digest) == (new.commit, new.digest):
        print(f"{args.name} is already at {commit}")
        return
    rewrite(resource.origin, args.name, {"commit": new.commit, "digest": new.digest})
    print(f"{args.name}: {resource.commit[:12]} -> {commit[:12]}, {new.digest}")


if __name__ == "__main__":
    main()
