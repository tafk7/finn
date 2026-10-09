"""Executed by the installation acceptance tests outside any source checkout."""
import json
import os
import sys
from pathlib import Path

from finn.platform import resolve_target
from finn.shells.pynq import templates
from finn.shells.pynq.driver import write_pynq_driver_support
from finn.shells.pynq.ipgen import data_path as shell_data_path
from finn.util.resources import resource_path

assert not os.environ.get("FINN_ROOT")
work = Path(sys.argv[1])
work.mkdir(exist_ok=True)
# FINN's own sources: the XSI bridge's.
assert Path(resource_path("xsi", "xsi_finn.cpp")).is_file()
# The pynq shell's data: the simulation control, the design description, the driver.
for member in ("sim_ctrl.v", "mdd/finn_design.mdd", "pynq_driver/driver_base.py"):
    assert Path(shell_data_path(*member.split("/"))).is_file()
# The platform catalog, package data of finn.platform.
target = resolve_target(board="Ultra96", period_ns=5.0, shell="pynq")
assert target.platform.resources is not None
# The driver's support files: the base driver, validate.py and the trimmed helpers.
driver_dir = work / "driver"
driver_dir.mkdir(exist_ok=True)
write_pynq_driver_support(str(driver_dir))
for name in ["driver_base.py", "validate.py", "finn/util/data_packing.py"]:
    assert (driver_dir / name).is_file()
    compile((driver_dir / name).read_text(), name, "exec")
assert "FINN_ROOT" not in templates.ipgentcl_template + templates.ipgen_template
print(json.dumps({"driver": str(driver_dir), "part": target.part}))
