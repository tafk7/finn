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

"""The dataflow build's entry points (``build_dataflow_cfg``,
``build_dataflow_directory`` and the ``build_dataflow`` command) and its runner: it
checks the configuration, then runs the plan's steps, logging and recording each."""

import clize
import json
import logging
import os
import pdb  # NOQA
import sys
import time
from dataclasses import asdict
from datetime import datetime
from qonnx.core.modelwrapper import ModelWrapper

from finn import resources
from finn.builder.build_dataflow_checks import configuration_findings
from finn.builder.build_dataflow_config import ConfigurationRefused, DataflowBuildConfig
from finn.builder.build_dataflow_manifest import (
    MANIFEST_FILE,
    BuildManifest,
    finn_revision,
)
from finn.builder.build_dataflow_plan import Plan, resolve_plan
from finn.builder.build_dataflow_steps import (
    build_dataflow_lists,
    build_dataflow_phases,
    build_dataflow_step_lookup,
    default_steps,
)
from finn.core.reporting.findings import Finding, FindingKind, Refusal
from finn.core.reporting.log import FILE_ONLY, running_step
from finn.core.reporting.manifest import note, recorded_run
from finn.platform.refusal import TargetRefused
from finn.platform.target import Target
from finn.toolchain.launcher import ToolFailed, ToolUnavailable, host_selection

#: A build directory's configuration file, beside its model.onnx.
CONFIG_FILE = "dataflow_build_config.json"

#: The build log, in the output directory.
LOG_FILE = "build_dataflow.log"

#: The runner's logger.
_LOG = logging.getLogger(__name__)

#: The build's own loggers, the runner's and the steps': the console shows their INFO
#: lines, and only the warnings and errors of the others.
_BUILD_LOGGERS = "finn.builder"


def _failure_kind(error: BaseException) -> str:
    """The kind of a failed step's error, by its type, as the manifest records it:
    ``tool-unavailable``, ``tool-failed``, ``refusal`` (a ``Refusal``), else ``bug``."""
    if isinstance(error, ToolUnavailable):
        return "tool-unavailable"
    if isinstance(error, ToolFailed):
        return "tool-failed"
    if isinstance(error, Refusal):
        return "refusal"
    return "bug"


def _log_failure(step: str, error: BaseException) -> str:
    """Log a failed step's error by its kind, and return the kind. A refusal is logged
    one line per finding, without its message (those same lines). The traceback goes to
    the log file, and for a bug to the console too."""
    kind = _failure_kind(error)
    name = type(error).__name__
    if isinstance(error, Refusal):
        findings = error.findings
        _LOG.error("%s refused: %d findings (%s)", step, len(findings), name)
        for finding in findings:
            _LOG.error("  %s", finding)
    elif kind != "bug":
        _LOG.error("%s: %s: %s", step, name, error)
    else:
        _LOG.error("%s failed: %s: %s", step, name, error, exc_info=error)
        return kind
    _LOG.error("%s: the traceback", step, exc_info=error, extra=FILE_ONLY)
    return kind


def _run_build_steps(model, cfg: DataflowBuildConfig, plan: Plan, manifest: BuildManifest):
    """Run the plan's steps in order, each with its settings, recording each in the
    ``manifest``. Returns 0, or -1 at the first step that fails, after logging and
    recording the failure and, with ``enable_build_pdb_debug``, opening the postmortem
    debugger."""
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
                # pdb needs a terminal: elsewhere (pytest, a batch job) it would fail and
                # bury the error above.
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


def _identity(output_dir: str, selection: dict | None, target: Target | None) -> dict:
    """The build's identity for its manifest: FINN's revision, the toolchain's
    ``selection`` and the resolved ``target``, each None where it is not known. The model
    is added once the build reads it."""
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
        "output_dir": os.path.abspath(output_dir),
        "started": datetime.now().isoformat(timespec="seconds"),
        "toolchain": selection,
        "target": stated,
    }


def _manifest(cfg: DataflowBuildConfig, target: Target | None) -> BuildManifest:
    """The build's manifest, its identity stated: the toolchain's selection where it
    reads, and the resolved ``target``."""
    try:
        selection = asdict(host_selection(stated=cfg.toolchain))
    except ValueError:
        selection = None
    return BuildManifest(cfg.output_dir, _identity(cfg.output_dir, selection, target))


def build_plan(cfg: DataflowBuildConfig) -> Plan:
    """The plan of ``cfg`` over the build's phases, premade lists and steps, its default
    list without ``steps`` (``resolve_plan``), or its refusal."""
    lists = {**build_dataflow_phases, **build_dataflow_lists}
    return resolve_plan(cfg, lists, build_dataflow_step_lookup, default_steps())


def _configured(
    cfg: DataflowBuildConfig, target: Target | None, found: list[Finding], manifest: BuildManifest
) -> Plan:
    """The build's plan, once the configuration is checked: the target's findings
    (``found``) with the plan's, or with the configuration checks'. Each finding is
    recorded, each limitation noted as a warning, and the rejections raised together
    as one ConfigurationRefused."""
    plan = None
    try:
        plan = build_plan(cfg)
    except ConfigurationRefused as refused:
        found = [*found, *refused.findings]
    else:
        manifest.planned(plan.record())
        if target is not None:
            found = [*found, *configuration_findings(cfg, plan, target)]
    manifest.configured(found)
    for finding in found:
        if finding.kind is not FindingKind.REJECTION:
            note(finding)
    rejections = [finding for finding in found if finding.kind is FindingKind.REJECTION]
    if rejections:
        raise ConfigurationRefused(rejections)
    assert plan is not None
    return plan


def _refused(manifest: BuildManifest, error: BaseException) -> None:
    """Log a build stopped before its first step and record it as ``refused``: a refusal
    (there, only the configuration refuses) finding by finding, any other error with its
    traceback in the log file."""
    if isinstance(error, Refusal):
        _LOG.error("Configuration refused: %d findings", len(error.findings))
        for finding in error.findings:
            _LOG.error("  %s", finding)
    else:
        _LOG.error("Build failed before its first step: %s: %s", type(error).__name__, error)
        _LOG.error("the traceback", exc_info=error, extra=FILE_ONLY)
    manifest.refused(error)


def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    """Build a dataflow accelerator from the ONNX model ``model_filename`` as ``cfg``
    says. Returns 0, or -1 for a failed step.

    The log and the manifest are set up first, so a refused configuration is logged and
    recorded before it raises. Before the first step the target is resolved, the plan
    made and the configuration checked; their rejections are raised together as one
    ConfigurationRefused."""
    log_file = os.path.join(cfg.output_dir, LOG_FILE)
    try:
        target, found = cfg.resolved_target, []
    except TargetRefused as refused:
        target, found = None, list(refused.findings)
    with recorded_run(
        log_file, _manifest(cfg, target), runner=_BUILD_LOGGERS, verbose=cfg.verbose
    ) as manifest:
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
            _refused(manifest, error)
            raise
        return _run_build_steps(model, cfg, plan, manifest)


def _refused_in(output_dir: str, refused: ConfigurationRefused) -> None:
    """Log and record in ``output_dir`` a build refused before its configuration
    decoded, replacing the manifest an earlier build left there."""
    manifest = BuildManifest(output_dir, _identity(output_dir, None, None))
    with recorded_run(os.path.join(output_dir, LOG_FILE), manifest, runner=_BUILD_LOGGERS):
        manifest.configured(list(refused.findings))
        _refused(manifest, refused)


def _stated_output_dir(path_to_cfg_dir: str) -> str | None:
    """The output directory a build directory's configuration states, as JSON states it,
    joined to the directory; None where it states none as a string."""
    try:
        with open(os.path.join(path_to_cfg_dir, CONFIG_FILE)) as f:
            stated = json.load(f)
    except (OSError, ValueError):
        return None
    output_dir = stated.get("output_dir") if isinstance(stated, dict) else None
    return os.path.join(path_to_cfg_dir, output_dir) if isinstance(output_dir, str) else None


def read_build_config(path_to_cfg_dir: str) -> DataflowBuildConfig:
    """A build directory's configuration (``CONFIG_FILE``), its relative paths naming
    files in the directory."""
    path = os.path.join(path_to_cfg_dir, CONFIG_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path_to_cfg_dir}: a build directory states {CONFIG_FILE}")
    with open(path) as f:
        return DataflowBuildConfig.from_json(f.read(), os.path.abspath(path_to_cfg_dir))


def build_dataflow_directory(path_to_cfg_dir: str):
    """Build the directory's ``model.onnx`` as its ``dataflow_build_config.json`` says, in
    this process and without changing the working directory (a relative FINN_BUILD_DIR
    stays the caller's). Returns build_dataflow_cfg's result. A configuration that does
    not decode is refused in the output directory it states as a string, its log and a
    new manifest, as the build's; otherwise with nothing written."""
    path_to_cfg_dir = os.path.abspath(path_to_cfg_dir)
    assert os.path.isdir(path_to_cfg_dir), "Directory not found: " + path_to_cfg_dir
    onnx_filename = path_to_cfg_dir + "/model.onnx"
    assert os.path.isfile(onnx_filename), "ONNX not found: " + onnx_filename
    try:
        cfg = read_build_config(path_to_cfg_dir)
    except ConfigurationRefused as refused:
        output_dir = _stated_output_dir(path_to_cfg_dir)
        if output_dir is not None:
            _refused_in(output_dir, refused)
        raise
    return build_dataflow_cfg(onnx_filename, cfg)


def main():
    """The ``build_dataflow`` command: build_dataflow_directory on its argument. It exits
    0, or 255 for a failed step; a refused configuration exits 1 with the traceback."""

    def cli(path_to_cfg_dir: str):
        """Build the model and configuration in path_to_cfg_dir."""
        status = build_dataflow_directory(path_to_cfg_dir)
        if status:
            raise SystemExit(status)

    clize.run(cli)


if __name__ == "__main__":
    main()
