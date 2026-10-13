# Copyright (c) 2020 Xilinx, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of Xilinx nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import clize
import contextlib
import logging
import os
import pdb  # NOQA
import sys
import time
from dataclasses import asdict
from datetime import datetime
from qonnx.core.modelwrapper import ModelWrapper

from finn import resources
from finn.builder.build_dataflow_checks import configuration_findings, required_vivado
from finn.builder.build_dataflow_config import (
    ConfigurationRefused,
    DataflowBuildConfig,
    finding_line,
)
from finn.builder.build_dataflow_manifest import (
    MANIFEST_FILE,
    Manifest,
    ManifestHandler,
    finn_revision,
)
from finn.builder.build_dataflow_plan import Plan, resolve_plan
from finn.core.space import Finding, FindingKind, by_code
from finn.platform.refusal import TargetRefused
from finn.platform.target import Target
from finn.toolchain.launcher import RECORDS, ToolFailed, ToolUnavailable, host_selection
from finn.util.run_log import FILE_ONLY, log_run, running_step

#: The configuration file of a build directory (build_dataflow_directory), next to
#: model.onnx.
CONFIG_FILE = "dataflow_build_config.json"

#: The build log, in the output directory.
LOG_FILE = "build_dataflow.log"

#: The runner's logger, log_run's ``runner``: the console shows its INFO lines (the
#: build's paths, each step it runs, the outcome) and, from other loggers, only warnings
#: and above.
_LOG = logging.getLogger(__name__)


def _failure_kind(error: BaseException) -> str:
    """The kind of a failed step's error: ``tool-unavailable`` (``ToolUnavailable``),
    ``tool-failed`` (``ToolFailed``), ``refusal`` (an error with ``findings``),
    ``verdict`` (a subclass of ``AssertionError``: a failed verification), else ``bug``."""
    if isinstance(error, ToolUnavailable):
        return "tool-unavailable"
    if isinstance(error, ToolFailed):
        return "tool-failed"
    if hasattr(error, "findings"):
        return "refusal"
    if isinstance(error, AssertionError) and type(error) is not AssertionError:
        return "verdict"
    return "bug"


def _log_failure(step: str, error: BaseException) -> str:
    """Log a failed step's error according to its kind (_failure_kind), and return the
    kind; the traceback always goes to the log file. A tool unavailable names the tool
    and what selects it, a tool failure its transcript and error lines, a verdict what
    the check found; a refusal the number of findings and one line per code. Anything
    else is a bug, and its traceback is shown on the console too."""
    kind = _failure_kind(error)
    name = type(error).__name__
    if kind in ("tool-unavailable", "tool-failed", "verdict"):
        _LOG.error("%s: %s: %s", step, name, error)
    elif kind == "refusal":
        findings = tuple(error.findings)
        _LOG.error("%s refused: %d findings: %s: %s", step, len(findings), name, error)
        for line in by_code(findings):
            _LOG.error("  %s", line)
    else:
        _LOG.error("%s failed: %s: %s", step, name, error, exc_info=error)
        return kind
    _LOG.error("%s: the traceback", step, exc_info=error, extra=FILE_ONLY)
    return kind


def _run_build_steps(model, cfg: DataflowBuildConfig, plan: Plan, manifest: Manifest):
    """Run the plan's steps in order, each with its settings and each log record stamped
    with its step, and record each step's start and end, and the outcome, in the
    build's ``manifest`` (finn.builder.build_dataflow_manifest).

    Returns 0 on success, or -1 at the first failing step after logging its failure
    (_log_failure), recording it in the manifest, and starting the postmortem debugger
    if ``enable_build_pdb_debug`` is set and stdin is a terminal.
    """
    run = plan.steps[plan.run]
    for number, step in enumerate(run, start=1):
        started = time.time()
        with running_step(step.name):
            try:
                _LOG.info("Running step: %s [%d/%d]", step.name, number, len(run))
                manifest.step_started(step.name)
                model = step.function(model, cfg, **step.settings)
                if cfg.save_intermediate_models:
                    intermediate_model_dir = cfg.output_dir + "/intermediate_models"
                    os.makedirs(intermediate_model_dir, exist_ok=True)
                    model.save("%s/%s.onnx" % (intermediate_model_dir, step.name))
                manifest.step_ended("completed", time.time() - started)
            except BaseException as error:
                kind = _log_failure(step.name, error)
                manifest.failed(step.name, kind, error)
                manifest.step_ended("failed", time.time() - started)
                # The postmortem debugger only where someone can answer it: with stdin not
                # a terminal (pytest, a batch job) pdb's read would fail and report itself
                # instead of the error above.
                if not cfg.enable_build_pdb_debug:
                    _LOG.info("enable_build_pdb_debug not set in build config, exiting...")
                elif sys.stdin is None or not sys.stdin.isatty():
                    _LOG.info("enable_build_pdb_debug set, but stdin is not a terminal, exiting...")
                else:
                    pdb.post_mortem(error.__traceback__)
                _LOG.error("Build failed")
                return -1
    manifest.completed()
    _LOG.info("Completed successfully")
    return 0


def _identity(cfg: DataflowBuildConfig, target: Target | None) -> dict:
    """The build's identity for its manifest: FINN's revision, the toolchain's selection
    (the configuration's over the machine's) and the resolved ``target``, each None where
    it is not known. The model is added as the build reads it, and the Vivado release
    where a probe found it (_indexed)."""
    try:
        selection = asdict(host_selection(stated=cfg.toolchain))
    except ValueError:
        selection = None
    stated = None
    if target is not None:
        stated = {
            "part": target.part,
            "shell": target.shell,
            "board": target.board,
            "period_ns": target.platform.period_ns,
        }
    return {
        "finn": finn_revision(),
        "model": None,
        "output_dir": os.path.abspath(cfg.output_dir),
        "started": datetime.now().isoformat(timespec="seconds"),
        "toolchain": selection,
        "target": stated,
    }


def _vivado_release(
    cfg: DataflowBuildConfig, target: Target | None, manifest: Manifest
) -> str | None:
    """The Vivado release of the build's toolchain where the build ran a tool or checked
    the release its board requires, as the toolchain's probe found it; None where it
    did neither, or nothing probed it. Never asks Vivado."""
    ran = any(step["tools"] for step in manifest.state["steps"])
    if not ran and (target is None or required_vivado(target) is None):
        return None
    toolchain = vars(cfg).get("prepared_toolchain")
    found = None if toolchain is None else toolchain.probed("vivado")
    return None if found is None else "%d.%d" % tuple(found[:2])


@contextlib.contextmanager
def _indexed(cfg: DataflowBuildConfig, target: Target | None):
    """The build's manifest (``MANIFEST_FILE`` in the output directory) for the ``with``
    block, written at once and filled from the records of RECORDS (ManifestHandler), whose
    level is INFO meanwhile; at its end, the Vivado release in its identity where the
    build asked it (_vivado_release)."""
    manifest = Manifest(cfg.output_dir, _identity(cfg, target))
    handler = ManifestHandler(manifest)
    level = RECORDS.level
    RECORDS.setLevel(logging.INFO)
    RECORDS.addHandler(handler)
    try:
        yield manifest
    finally:
        release = _vivado_release(cfg, target, manifest)
        if release is not None:
            manifest.identify(vivado=release)
        RECORDS.removeHandler(handler)
        RECORDS.setLevel(level)
        handler.close()


def _configured(
    cfg: DataflowBuildConfig, target: Target | None, found: list[Finding], manifest: Manifest
) -> Plan:
    """The build's plan, recorded in the manifest, once the configuration is checked:
    ``found`` (the target's refusal) with the plan's rejections, else with the
    configuration checks' findings. The findings are recorded, each limitation logged as
    a warning, and every rejection raised as one ConfigurationRefused."""
    plan = None
    try:
        plan = resolve_plan(cfg)
    except ConfigurationRefused as refused:
        found = [*found, *refused.findings]
    else:
        manifest.planned(plan.record())
        if target is not None:
            found = [*found, *configuration_findings(cfg, plan, target)]
    manifest.configured(found)
    for finding in found:
        if finding.kind is not FindingKind.REJECTION:
            _LOG.warning("%s", finding_line(finding))
    rejections = [finding for finding in found if finding.kind is FindingKind.REJECTION]
    if rejections:
        raise ConfigurationRefused(rejections)
    assert plan is not None
    return plan


def _log_refusal(error: BaseException) -> None:
    """Log a build stopped before its first step: a refused configuration as each of its
    findings; any other error by its type, its traceback in the log file only."""
    if isinstance(error, ConfigurationRefused):
        _LOG.error("Configuration refused: %d findings", len(error.findings))
        for finding in error.findings:
            _LOG.error("  %s", finding_line(finding))
        return
    _LOG.error("Build failed before its first step: %s: %s", type(error).__name__, error)
    _LOG.error("the traceback", exc_info=error, extra=FILE_ONLY)


def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    """Best-effort build a dataflow accelerator using the given configuration, through
    the steps of finn.builder.build_dataflow_steps.

    The build log (``LOG_FILE`` in the output directory, and the console; log_run) and
    the manifest (``MANIFEST_FILE``, finn.builder.build_dataflow_manifest; _indexed) are
    set up first, so a refused configuration is logged and recorded before it raises.

    Before the first step, the target is resolved, then the plan
    (finn.builder.build_dataflow_plan), recorded in the manifest, then the configuration
    checks run (finn.builder.build_dataflow_checks); every rejection of the three is
    raised together as one ConfigurationRefused (_configured).

    :param model_filename: ONNX model filename to build
    :param cfg: Build configuration
    """
    log_file = os.path.join(cfg.output_dir, LOG_FILE)
    with log_run(log_file, runner=_LOG.name, verbose=cfg.verbose):
        try:
            target, found = cfg.resolved_target, []
        except TargetRefused as refused:
            target, found = None, list(refused.findings)
        with _indexed(cfg, target) as manifest:
            try:
                _LOG.info("Intermediate outputs will be generated in %s", resources.scratch())
                _LOG.info("Final outputs will be generated in %s", cfg.output_dir)
                _LOG.info("Build log is at %s", log_file)
                _LOG.info("Build manifest is at %s", os.path.join(cfg.output_dir, MANIFEST_FILE))
                plan = _configured(cfg, target, found, manifest)
                if cfg.start_step is None:
                    source = model_filename
                else:
                    before = plan.steps[plan.run.start - 1].name
                    source = cfg.output_dir + "/intermediate_models/%s.onnx" % before
                _LOG.info("Building dataflow accelerator from %s", source)
                model = ModelWrapper(source)
                manifest.identify(model=os.path.abspath(source))
            except Exception as error:
                _log_refusal(error)
                manifest.refused(error)
                raise
            return _run_build_steps(model, cfg, plan, manifest)


def read_build_config(path_to_cfg_dir: str) -> DataflowBuildConfig:
    """Read a build directory's CONFIG_FILE, its relative paths naming files in the
    directory (``DataflowBuildConfig.base_dir``) wherever the build runs. Refuses a
    directory without one."""
    path = os.path.join(path_to_cfg_dir, CONFIG_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path_to_cfg_dir}: a build directory states {CONFIG_FILE}")
    with open(path) as f:
        return DataflowBuildConfig.from_json(f.read(), os.path.abspath(path_to_cfg_dir))


def build_dataflow_directory(path_to_cfg_dir: str):
    """Best-effort build a dataflow accelerator from the specified directory, in this
    process and without changing the working directory: the configuration's relative
    paths name files in the directory (read_build_config), and a relative
    FINN_BUILD_DIR is relative to the caller's. Returns build_dataflow_cfg's result: 0,
    or -1 for a failed step. A refused configuration raises.

    :param path_to_cfg_dir: Directory containing the model and build config

    The specified directory path_to_cfg_dir must contain the following files:

    * model.onnx : ONNX model to be converted to dataflow accelerator
    * dataflow_build_config.json : its DataflowBuildConfig (read_build_config)

    """
    # get absolute path
    path_to_cfg_dir = os.path.abspath(path_to_cfg_dir)
    assert os.path.isdir(path_to_cfg_dir), "Directory not found: " + path_to_cfg_dir
    onnx_filename = path_to_cfg_dir + "/model.onnx"
    assert os.path.isfile(onnx_filename), "ONNX not found: " + onnx_filename
    return build_dataflow_cfg(onnx_filename, read_build_config(path_to_cfg_dir))


def main():
    """Entry point for dataflow builds. Invokes `build_dataflow_directory` using
    command line arguments, and exits with its result: 0, or 255 for a failed step
    (its -1). A refused configuration raises, exiting 1 with the traceback."""

    def cli(path_to_cfg_dir: str):
        """Build the model and configuration in path_to_cfg_dir."""
        status = build_dataflow_directory(path_to_cfg_dir)
        if status:
            raise SystemExit(status)

    clize.run(cli)


if __name__ == "__main__":
    main()
