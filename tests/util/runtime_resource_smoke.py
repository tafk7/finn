"""Executed by the installation acceptance tests outside any source checkout."""
import json
import os
import sys
from pathlib import Path

from finn.platform.resolve import resolve_target
from finn.shells.pynq.runtime import shipped
from finn.xsi.paths import resource_path

assert not os.environ.get("FINN_ROOT")
work = Path(sys.argv[1])
work.mkdir(exist_ok=True)
# FINN's own sources: the XSI bridge's.
assert Path(resource_path("xsi", "xsi_finn.cpp")).is_file()
# The platform catalog, package data of finn.platform.
target = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
assert target.platform.resources is not None
# The pynq runtime's driver: the shared driver and the PYNQ device, copied as it ships them.
driver_dir = work / "driver"
for path, source in shipped().items():
    (driver_dir / path).parent.mkdir(parents=True, exist_ok=True)
    (driver_dir / path).write_text(source.read_text())
    compile((driver_dir / path).read_text(), str(path), "exec")
print(json.dumps({"driver": str(driver_dir), "part": target.part}))
