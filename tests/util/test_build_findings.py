# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

import pytest

import errno
import json
import subprocess
import sys
from onnx import TensorProto, helper
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.findings import Finding, Refusal, collect_findings, report_finding
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import DataflowBuildConfig, DataflowOutputType

pytestmark = pytest.mark.util

FIRST = Finding("first-notice", "ExamplePass", "kept the node", (("node", "Add_3"),))
SECOND = Finding("second-notice", "ExamplePass", "kept the tensor", (("tensor", "sum"),))
REASONS = (
    Finding("unsupported-input", "ExamplePass", "input cannot be lowered", (("node", "Add_3"),)),
    Finding("unsupported-output", "ExamplePass", "output cannot be lowered", (("tensor", "sum"),)),
)


def make_model(op_type="Relu", domain=""):
    graph = helper.make_graph(
        [helper.make_node(op_type, ["inp"], ["out"], name="test_node", domain=domain)],
        "findings_build",
        [helper.make_tensor_value_info("inp", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 4])],
    )
    imports = [helper.make_opsetid("", 13)]
    if domain:
        imports.append(helper.make_opsetid(domain, 1))
    return ModelWrapper(helper.make_model(graph, opset_imports=imports))


def make_cfg(tmp_path, steps):
    return DataflowBuildConfig(
        output_dir=str(tmp_path / "output"),
        synth_clk_period_ns=10.0,
        fpga_part="xc7z020clg400-1",
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        steps=steps,
        enable_build_pdb_debug=False,
    )


def run_build(tmp_path, steps, model=None):
    path = tmp_path / "model.onnx"
    (make_model() if model is None else model).save(str(path))
    return build_dataflow_cfg(str(path), make_cfg(tmp_path, steps))


def read_findings(tmp_path):
    return json.loads((tmp_path / "output/build_findings.json").read_text())


def passing_step(model, cfg):
    return model


def reporting_step(model, cfg):
    report_finding(FIRST)
    with collect_findings():
        report_finding(SECOND)
        report_finding(FIRST)
    return model


def test_a_step_reports_findings_in_order_to_the_log_file_console_and_outer_scope(tmp_path, capfd):
    with collect_findings() as outer:
        assert run_build(tmp_path, [reporting_step, passing_step]) == 0
    assert outer == [FIRST, SECOND]
    assert read_findings(tmp_path) == {
        "reporting_step": {"findings": [FIRST.record(), SECOND.record()]}
    }
    log = (tmp_path / "output/build_dataflow.log").read_text()
    console = capfd.readouterr()
    for finding in (FIRST, SECOND):
        line = finding.subject_line()
        assert log.count("WARNING reporting_step finn.builder.build_dataflow: " + line) == 1
        assert console.err.count("WARNING: " + line) == 1
    assert console.out.splitlines()[-1] == "Completed successfully"
    assert (tmp_path / "output/time_per_step.json").is_file()


@pytest.mark.parametrize("refused", [False, True], ids=["plain-error", "refusal"])
@pytest.mark.parametrize("reported", [False, True], ids=["without-report", "after-report"])
def test_a_failed_step_retains_prior_steps_reports_and_refusal_reasons(
    tmp_path, capfd, refused, reported
):
    ran = []
    error = Refusal(REASONS) if refused else ValueError("step broke")

    def failing_step(model, cfg):
        if reported:
            report_finding(SECOND)
        raise error

    def later_step(model, cfg):
        ran.append(True)
        return model

    with collect_findings() as outer:
        assert run_build(tmp_path, [reporting_step, failing_step, later_step]) == -1
    assert outer == [FIRST, SECOND]
    assert ran == []
    expected = {"reporting_step": {"findings": [FIRST.record(), SECOND.record()]}}
    if reported or refused:
        expected["failing_step"] = {"findings": [SECOND.record()] if reported else []}
    if refused:
        expected["failing_step"]["refused"] = [finding.record() for finding in REASONS]
    assert read_findings(tmp_path) == expected
    assert not (tmp_path / "output/time_per_step.json").exists()
    log = (tmp_path / "output/build_dataflow.log").read_text()
    console = capfd.readouterr()
    assert "Traceback (most recent call last):" in log
    assert type(error).__name__ + ": " in log
    assert console.out.splitlines()[-1] == "Build failed"
    if reported:
        assert "WARNING failing_step finn.builder.build_dataflow: " + SECOND.subject_line() in log
    if refused:
        failure = log.index(
            "ERROR failing_step finn.builder.build_dataflow: Step failing_step failed:"
        )
        for finding in REASONS:
            line = finding.subject_line()
            assert log.index("ERROR failing_step finn.builder.build_dataflow: " + line) < failure
            assert "ERROR: " + line in console.err
    else:
        assert "refused" not in (tmp_path / "output/build_findings.json").read_text()


@pytest.mark.parametrize("steps", [[], [passing_step]])
def test_a_build_with_no_findings_writes_an_empty_object(tmp_path, capfd, steps):
    assert run_build(tmp_path, steps) == 0
    assert read_findings(tmp_path) == {}
    console = capfd.readouterr()
    assert console.err == ""
    assert console.out.splitlines()[-1] == "Completed successfully"


@pytest.mark.parametrize("failure", [False, True], ids=["successful-build", "failed-build"])
def test_report_write_errors_do_not_replace_the_build_outcome(tmp_path, capfd, failure):
    output = tmp_path / "output"
    output.mkdir()
    # A directory at the file's path cannot be opened for writing, even as root.
    (output / "build_findings.json").mkdir()

    def write_error_step(model, cfg):
        report_finding(FIRST)
        if failure:
            raise Refusal(REASONS)
        return model

    assert run_build(tmp_path, [write_error_step]) == (-1 if failure else 0)
    console = capfd.readouterr()
    assert console.err.count("WARNING: Could not write build_findings.json:") == 1
    assert console.out.splitlines()[-1] == ("Build failed" if failure else "Completed successfully")
    assert (output / "time_per_step.json").exists() != failure
    if failure:
        assert "Refusal: " in (output / "build_dataflow.log").read_text()


def test_a_later_build_replaces_the_previous_findings_file(tmp_path):
    assert run_build(tmp_path, [reporting_step]) == 0
    assert read_findings(tmp_path)
    assert run_build(tmp_path, [passing_step]) == 0
    assert read_findings(tmp_path) == {}


def test_tidy_up_reports_a_constant_that_cannot_be_stored_exactly(tmp_path, capfd):
    graph = helper.make_graph(
        [helper.make_node("Add", ["large", "one"], ["sum"], name="Add_3")],
        "constant_add",
        [],
        [helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1])],
        initializer=[
            helper.make_tensor("large", TensorProto.FLOAT, [1], [16777216]),
            helper.make_tensor("one", TensorProto.FLOAT, [1], [1]),
        ],
    )
    model = ModelWrapper(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]))
    for name in ("large", "one", "sum"):
        model.set_tensor_datatype(name, DataType["INT32"])
    assert run_build(tmp_path, ["step_tidy_up"], model) == 0
    findings = read_findings(tmp_path)["step_tidy_up"]["findings"]
    assert len(findings) == 1
    finding = findings[0]
    assert finding["code"] == "fold-unrepresentable"
    assert finding["owner"] == "FoldConstants"
    assert finding["details"]["tensor"] == "sum"
    assert "cannot be stored exactly" in capfd.readouterr().err
    saved = ModelWrapper(str(tmp_path / "output/intermediate_models/step_tidy_up.onnx"))
    assert len(saved.graph.node) == 1
    assert saved.graph.node[0].op_type == "Add"
    assert saved.get_initializer("sum") is None


def test_a_refused_command_keeps_exit_one_and_writes_its_reason(tmp_path):
    make_model("MatrixVectorActivation", "finn.custom_op.fpgadataflow").save(
        str(tmp_path / "model.onnx")
    )
    cfg = make_cfg(tmp_path, ["step_tidy_up"])
    (tmp_path / "dataflow_build_config.json").write_text(cfg.to_json())
    result = subprocess.run(
        [sys.executable, "-m", "finn.builder.build_dataflow", str(tmp_path)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stderr
    assert result.stdout.splitlines()[-1] == "Build failed"
    step = read_findings(tmp_path)["step_tidy_up"]
    assert step["findings"] == []
    assert len(step["refused"]) == 1
    finding = step["refused"][0]
    assert finding["code"] == "domain-retired"
    assert "renamed MVAU" in finding["message"]
    assert "CustomOpRefused:" in result.stderr
    assert "domain-retired:" in result.stderr


@pytest.mark.parametrize("failure", [False, True], ids=["successful-build", "refused-build"])
@pytest.mark.parametrize("existing", [False, True], ids=["no-previous-report", "previous-report"])
def test_a_mid_write_error_leaves_no_partial_findings_file(
    tmp_path, monkeypatch, capfd, failure, existing
):
    output = tmp_path / "output"
    output.mkdir()
    destination = output / "build_findings.json"
    previous = '{"previous_step": {"findings": []}}'
    if existing:
        destination.write_text(previous)
    attempted = []
    dump = json.dump

    def partial_dump(data, stream, *args, **kwargs):
        if "mid_write_step" in data:
            attempted.append(Path(stream.name))
            stream.write('{"partial":')
            stream.flush()
            raise OSError(errno.ENOSPC, "No space left on device")
        return dump(data, stream, *args, **kwargs)

    def mid_write_step(model, cfg):
        report_finding(FIRST)
        if failure:
            raise Refusal(REASONS)
        return model

    monkeypatch.setattr(json, "dump", partial_dump)
    assert run_build(tmp_path, [mid_write_step]) == (-1 if failure else 0)
    assert len(attempted) == 1
    assert attempted[0].parent == output
    if existing:
        assert destination.read_text() == previous
        assert json.loads(destination.read_text()) == {"previous_step": {"findings": []}}
    else:
        assert not destination.exists()
    assert all(not path.exists() for path in attempted if path != destination)
    console = capfd.readouterr()
    assert console.err.count("WARNING: Could not write build_findings.json:") == 1
    assert "No space left on device" in console.err
    assert console.out.splitlines()[-1] == ("Build failed" if failure else "Completed successfully")
    assert (output / "time_per_step.json").exists() != failure
    if failure:
        assert "Refusal: " in (output / "build_dataflow.log").read_text()


@pytest.mark.parametrize("failure", [False, True], ids=["successful-build", "refused-build"])
def test_a_findings_path_symlink_does_not_redirect_the_report(tmp_path, failure):
    output = tmp_path / "output"
    output.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("outside content")
    destination = output / "build_findings.json"
    destination.symlink_to(outside)

    def symlink_step(model, cfg):
        report_finding(FIRST)
        if failure:
            raise Refusal(REASONS)
        return model

    assert run_build(tmp_path, [symlink_step]) == (-1 if failure else 0)
    assert outside.read_text() == "outside content"
    assert not destination.is_symlink()
    expected = {"findings": [FIRST.record()]}
    if failure:
        expected["refused"] = [finding.record() for finding in REASONS]
    assert read_findings(tmp_path) == {"symlink_step": expected}
