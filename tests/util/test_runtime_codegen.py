"""Resource consumers exercise installed paths without vendor software."""
import pytest

import importlib
import onnx.helper as oh
from pathlib import Path

from finn.custom_op.fpgadataflow.hls.streamingdatawidthconverter_hls import (
    StreamingDataWidthConverter_hls,
)
from finn.util.resources import resource_path, tcl_quote


@pytest.mark.parametrize(
    "module,classname",
    [
        ("thresholding_rtl", "Thresholding_rtl"),
        ("pwpolyf_rtl", "PWPolyF_rtl"),
        ("layernorm_rtl", "LayerNorm_rtl"),
        ("hwsoftmax_rtl", "HWSoftmax_rtl"),
        ("streamingdatawidthconverter_rtl", "StreamingDataWidthConverter_rtl"),
        ("requant_rtl", "Requant_rtl"),
        ("elementwise_binary_rtl", "ElementwiseAdd_rtl"),
        ("convolutioninputgenerator_rtl", "ConvolutionInputGenerator_rtl"),
        ("matrixvectoractivation_rtl", "MVAU_rtl"),
        ("vectorvectoractivation_rtl", "VVAU_rtl"),
    ],
)
def test_rtl_source_lists_resolve_installed_files(module, classname, tmp_path, monkeypatch):
    monkeypatch.delenv("FINN_ROOT", raising=False)
    cls = getattr(importlib.import_module("finn.custom_op.fpgadataflow.rtl." + module), classname)
    node = oh.make_node(
        classname,
        ["in"],
        ["out"],
        name="node",
        code_gen_dir_ipgen=str(tmp_path),
        gen_top_module="node",
        inputDataType="INT8",
        outputDataType="INT8",
        dataType="INT8",
        weightDataType="INT8",
        resType="LUT",
        mem_mode="internal_embedded",
    )
    op = cls(node)
    paths = op.get_rtl_file_list(abspath=True)
    package_paths = [p for p in paths if not p.startswith(str(tmp_path))]
    assert package_paths
    missing = [path for path in package_paths if not Path(path).is_file()]
    assert not missing, missing


def test_generated_hls_tcl_resolves_resources_and_external_input(tmp_path, monkeypatch):
    external = tmp_path / "external $ headers [1]"
    external.mkdir()
    monkeypatch.setenv("FINN_RESOURCES_HLSLIB", str(external))
    monkeypatch.delenv("FINN_ROOT", raising=False)
    node = oh.make_node(
        "StreamingDataWidthConverter_hls",
        ["in"],
        ["out"],
        name="dwc",
        inWidth=8,
        outWidth=16,
        shape=[1, 4],
        dataType="INT8",
        code_gen_dir_ipgen=str(tmp_path),
    )
    StreamingDataWidthConverter_hls(node).code_generation_ipgen(None, "xc7z020clg400-1", 10)
    script = (tmp_path / "hls_syn_dwc.tcl").read_text()
    assert "$HLSLIB$" not in script and "$::env(FINN_" not in script
    assert tcl_quote(external) in script
    assert tcl_quote(resource_path("custom_hls")) in script
    # Unquoted: Vitis HLS passes quote characters in -cflags through to the
    # compiler, so -I"dir" names a directory that does not exist.
    assert '-cflags "-std=c++14 -I$config_bnnlibdir -I$config_customhlsdir"' in script


def test_resource_path_rejects_escape():
    with pytest.raises(ValueError):
        resource_path("rtllib", "../outside")
    with pytest.raises(ValueError):
        resource_path("rtllib", "/absolute")
