"""Executed by the installation acceptance tests outside any source checkout."""
import json
import onnx.helper as oh
import os
import sys
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper

from finn import deploy
from finn.custom_op.fpgadataflow import templates
from finn.custom_op.fpgadataflow.rtl.streamingfifo_rtl import StreamingFIFO_rtl
from finn.shells.pynq.ipgen import data_path as shell_data_path
from finn.transformation.fpgadataflow.make_driver import MakePYNQDriver
from finn.util.basic import fifo_rtl_files
from finn.util.resources import resource_path

assert not os.environ.get("FINN_ROOT")
work = Path(sys.argv[1])
work.mkdir(exist_ok=True)
for family, member in [
    ("rtllib", "fifo/hdl/fifo.sv"),
    ("custom_hls", "checksum.hpp"),
    ("xsi", "xsi_finn.cpp"),
    ("custom_hls", "CNPY_LICENSE"),
]:
    assert Path(resource_path(family, member)).is_file()
assert Path(deploy.data_path("mdd/finn_design.mdd")).is_file()
for member in ("sim_ctrl.v", "mdd/finn_design.mdd", "pynq_driver/driver_base.py"):
    assert Path(shell_data_path(*member.split("/"))).is_file()
node = oh.make_node(
    "StreamingFIFO_rtl",
    ["in"],
    ["out"],
    name="fifo",
    domain="finn.custom_op.fpgadataflow.rtl",
    backend="fpgadataflow",
    depth=4,
    folded_shape=[1, 4],
    dataType="INT8",
    impl_style="rtl",
    code_gen_dir_ipgen=str(work),
)
op = StreamingFIFO_rtl(node)
op.generate_hdl(None, "xc7z020clg400-1", 10)
assert (work / "fifo.v").exists()
assert "$" not in (work / "fifo.v").read_text()
assert all(Path(p).exists() for p in fifo_rtl_files())
# A zero-port partition is sufficient to exercise driver asset/template emission.
model = ModelWrapper(oh.make_model(oh.make_graph([], "empty", [], [])))
model, _ = MakePYNQDriver("zynq-iodma").apply(model)
driver_dir = Path(model.get_metadata_prop("pynq_driver_dir"))
for name in ["driver.py", "driver_base.py", "validate.py", "finn/util/data_packing.py"]:
    assert (driver_dir / name).is_file()
    compile((driver_dir / name).read_text(), name, "exec")
assert "FINN_ROOT" not in templates.ipgentcl_template + templates.ip_gen_loop_op
assert resource_path("custom_hls") in templates.ipgentcl_template
print(json.dumps({"rtl": str(work / "fifo.v"), "driver": str(driver_dir)}))
