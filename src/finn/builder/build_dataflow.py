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
import json
import logging
import os
import pdb  # NOQA
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import PurePath
from qonnx.core.modelwrapper import ModelWrapper

from finn import resources
from finn.builder.build_dataflow_checks import (
    log_report,
    run_all_config_checks,
    save_report,
)
from finn.builder.build_dataflow_config import (
    VERIFIED_BY,
    DataflowBuildConfig,
    VerificationStepType,
)
from finn.builder.build_dataflow_manifest import (
    MANIFEST_FILE,
    Manifest,
    ManifestHandler,
    finn_revision,
)
from finn.builder.build_dataflow_plan import BuildPlan, declared_settings, resolve_plan
from finn.core.space import by_code
from finn.toolchain.launcher import ToolFailed, ToolUnavailable, host_selection
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
    and what selects it; a tool failure its transcript and error lines; a refusal the
    number of findings and one line per code; a verdict ``Verification for <step> :
    FAIL``. Anything else is a bug, and its traceback is shown on the console too."""
    kind = _failure_kind(error)
    name = type(error).__name__
    if kind in ("tool-unavailable", "tool-failed"):
        _LOG.error("%s: %s: %s", step, name, error)
    elif kind == "refusal":
        findings = tuple(error.findings)
        _LOG.error("%s refused: %d findings: %s: %s", step, len(findings), name, error)
        for line in by_code(findings):
            _LOG.error("  %s", line)
    elif kind == "verdict":
        _LOG.error("Verification for %s : FAIL", step)
        _LOG.error("%s: %s", name, error)
    else:
        _LOG.error("%s failed: %s: %s", step, name, error, exc_info=error)
        return kind
    _LOG.error("%s: the traceback", step, exc_info=error, extra=FILE_ONLY)
    return kind


def _findings_by_code(error: BaseException) -> dict[str, int]:
    """A refusal's findings by code; none for another error."""
    counts: dict[str, int] = {}
    for finding in getattr(error, "findings", ()):
        code = getattr(finding, "code", None)
        if isinstance(code, str):
            counts[code] = counts.get(code, 0) + 1
    return counts


def resolve_step_filename(step_name: str, cfg: DataflowBuildConfig, step_delta: int = 0):
    """The intermediate model file of the step ``step_delta`` places after ``step_name``
    (for a phase, its first step) in the build's plan (finn.builder.build_dataflow_plan)."""
    plan = resolve_plan(cfg)
    return _intermediate_model(cfg, plan, plan.index(step_name) + step_delta)


def _intermediate_model(cfg: DataflowBuildConfig, plan: BuildPlan, index: int) -> str:
    """The intermediate model file of the plan's step at ``index``."""
    if not 0 <= index < len(plan.edited):
        raise ValueError(f"no step at {index} of this build's {len(plan.edited)} steps")
    return cfg.output_dir + "/intermediate_models/%s.onnx" % plan.edited[index].name


def _run_build_steps(model, cfg, plan: BuildPlan, manifest=None):
    """Run the plan's steps in order, each with its settings and each log record stamped
    with its step, and record each step's start and end, and the outcome, in the
    build's ``manifest`` (finn.builder.build_dataflow_manifest), if given.

    Writes time_per_step.json, the seconds each step took. Returns 0 on success, or -1
    at the first failing step after logging its failure (_log_failure), recording it in
    the manifest, and starting the postmortem debugger if ``enable_build_pdb_debug`` is
    set and stdin is a terminal.
    """
    step_num = 1
    time_per_step = dict()
    for planned in plan.steps:
        step_name = planned.name
        step_start = time.time()
        with running_step(step_name):
            try:
                _LOG.info("Running step: %s [%d/%d]", step_name, step_num, len(plan.steps))
                if manifest is not None:
                    manifest.step_started(step_name, planned.phase)
                step_start = time.time()
                model = planned.run(model, cfg)
                step_end = time.time()
                time_per_step[step_name] = step_end - step_start
                chkpt_name = "%s.onnx" % (step_name)
                if cfg.save_intermediate_models:
                    intermediate_model_dir = cfg.output_dir + "/intermediate_models"
                    if not os.path.exists(intermediate_model_dir):
                        os.makedirs(intermediate_model_dir)
                    model.save("%s/%s" % (intermediate_model_dir, chkpt_name))
                if manifest is not None:
                    manifest.step_ended("completed", time.time() - step_start)
                step_num += 1
            except:  # noqa
                extype, value, tb = sys.exc_info()
                kind = _log_failure(step_name, value)
                if manifest is not None:
                    manifest.failed(step_name, kind, value, _findings_by_code(value))
                    manifest.step_ended("failed", time.time() - step_start)
                # The postmortem debugger only where someone can answer it: with stdin not
                # a terminal (pytest, a batch job) pdb's read would fail and report itself
                # instead of the error above.
                if not cfg.enable_build_pdb_debug:
                    _LOG.info("enable_build_pdb_debug not set in build config, exiting...")
                elif sys.stdin is None or not sys.stdin.isatty():
                    _LOG.info("enable_build_pdb_debug set, but stdin is not a terminal, exiting...")
                else:
                    pdb.post_mortem(tb)
                _LOG.error("Build failed")
                return -1

    with open(cfg.output_dir + "/time_per_step.json", "w") as f:
        json.dump(time_per_step, f, indent=2)
    if manifest is not None:
        manifest.completed()
    _LOG.info("Completed successfully")
    return 0


def _identity(cfg: DataflowBuildConfig) -> dict:
    """The build's identity for its manifest, from the configuration before its checks:
    FINN's revision, the toolchain's selection (the configuration's over the machine's)
    and the target, each None where it cannot be known. The model and the Vivado release
    are added later, as the build reads them."""
    try:
        selection = asdict(host_selection(stated=cfg.toolchain))
    except ValueError:
        selection = None
    try:
        resolved = cfg.resolved_target
        target = {
            "part": resolved.part,
            "shell": resolved.shell,
            "board": resolved.board,
            "period_ns": resolved.platform.period_ns,
        }
    except (ValueError, LookupError, TypeError):
        target = None  # the configuration checks name it
    return {
        "finn": finn_revision(),
        "model": None,
        "output_dir": os.path.abspath(cfg.output_dir),
        "started": datetime.now().isoformat(timespec="seconds"),
        "toolchain": selection,
        "vivado": None,
        "target": target,
    }


def _asked_verifications(cfg: DataflowBuildConfig) -> dict:
    """Each verification ``cfg.verify_steps`` asks for, by name, with the step that
    concludes it (the last in VERIFIED_BY). A name that is not a verification is skipped;
    the configuration checks report it."""
    asked = {}
    for each in cfg.verify_steps:
        try:
            verification = VerificationStepType(each)
        except ValueError:
            continue
        asked[verification.value] = VERIFIED_BY[verification][-1]
    return asked


@contextlib.contextmanager
def _indexed(cfg: DataflowBuildConfig):
    """The build's manifest (``MANIFEST_FILE`` in the output directory) for the ``with``
    block, written at once and filled from FINN's log records (ManifestHandler)."""
    manifest = Manifest(cfg.output_dir, _identity(cfg), _asked_verifications(cfg))
    handler = ManifestHandler(manifest)
    logger = logging.getLogger("finn")
    logger.addHandler(handler)
    try:
        yield manifest
    finally:
        logger.removeHandler(handler)
        handler.close()


def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    """Best-effort build a dataflow accelerator using the given configuration, through
    the steps of finn.builder.build_dataflow_steps.

    The build log (``LOG_FILE`` in the output directory, and the console; log_run) and
    the manifest (``MANIFEST_FILE``, finn.builder.build_dataflow_manifest; _indexed) are
    set up first, so the configuration checks are logged and listed in the
    manifest too, and a refused configuration is logged and recorded before it raises.

    The plan (finn.builder.build_dataflow_plan) is resolved first, and recorded in the
    manifest: its steps, each with its settings, and each setting with its readers.

    :param model_filename: ONNX model filename to build
    :param cfg: Build configuration
    """
    log_file = os.path.join(cfg.output_dir, LOG_FILE)
    with log_run(log_file, runner=_LOG.name, verbose=cfg.verbose), _indexed(cfg) as manifest:
        try:
            plan = resolve_plan(cfg)
            manifest.planned(plan.record())
            # if start_step is specified, override the input model
            if cfg.start_step is None:
                _LOG.info("Building dataflow accelerator from " + model_filename)
                model = ModelWrapper(model_filename)
            else:
                if plan.first == 0:
                    raise ValueError(
                        f"start_step {cfg.start_step}: no step runs before it, so no model "
                        "was saved to start from"
                    )
                intermediate_model_filename = _intermediate_model(cfg, plan, plan.first - 1)
                _LOG.info(
                    "Building dataflow accelerator from intermediate checkpoint"
                    + intermediate_model_filename
                )
                model = ModelWrapper(intermediate_model_filename)
            assert type(model) is ModelWrapper
            manifest.identify(
                model=os.path.abspath(
                    model_filename if cfg.start_step is None else intermediate_model_filename
                )
            )
            finn_build_dir = resources.scratch()

            _LOG.info("Intermediate outputs will be generated in " + str(finn_build_dir))
            _LOG.info("Final outputs will be generated in " + cfg.output_dir)
            _LOG.info("Build log is at " + log_file)
            _LOG.info("Build manifest is at " + os.path.join(cfg.output_dir, MANIFEST_FILE))

            # Run configuration checks
            config_report = run_all_config_checks(cfg, plan)
            log_report(config_report)
            report_path = save_report(config_report, cfg.output_dir)
            _LOG.info(f"Configuration check report saved to: {report_path}")
            release = config_report.vivado_version
            manifest.identify(vivado=None if release is None else "%d.%d" % tuple(release))
            manifest.configuration_checks(config_report, report_path)

            if config_report.has_errors():
                raise AssertionError(
                    "Configuration check failed with errors. Fix the issues above to proceed."
                )
        except Exception as error:
            _LOG.error("Build failed before its first step: %s: %s", type(error).__name__, error)
            _LOG.error("the traceback", exc_info=error, extra=FILE_ONLY)
            manifest.refused(error)
            raise
        return _run_build_steps(model, cfg, plan, manifest)


def _is_path(kind) -> bool:
    """Whether a setting's annotation is a path, or a path or None."""
    if isinstance(kind, type):
        return issubclass(kind, PurePath)
    return any(_is_path(each) for each in getattr(kind, "__args__", ()) if each is not type(None))


def _resolved_against(cfg: DataflowBuildConfig, directory: str) -> DataflowBuildConfig:
    """``cfg`` with each relative path joined to ``directory``: ``output_dir``,
    ``verify_input_npy``, the toolchain's settings scripts and command directory, and
    each setting a step of the build declares as a path (finn.builder.build_dataflow_plan;
    the preprocessing model, the pinned choices). Absolute paths and an empty command
    directory are kept as they are."""

    def resolve(path: str) -> str:
        return os.path.join(directory, path)

    declared = declared_settings(cfg)
    settings = dict(cfg.settings)
    for name, value in settings.items():
        if isinstance(value, str) and _is_path(declared.get(name)):
            settings[name] = resolve(value)
    toolchain = cfg.toolchain
    if toolchain is not None:
        toolchain = replace(
            toolchain,
            settings=tuple(map(resolve, toolchain.settings)),
            command_dir=resolve(toolchain.command_dir)
            if toolchain.command_dir
            else toolchain.command_dir,
        )
    return replace(
        cfg,
        output_dir=resolve(cfg.output_dir),
        verify_input_npy=resolve(cfg.verify_input_npy),
        toolchain=toolchain,
        settings=settings,
    )


def read_build_config(path_to_cfg_dir: str) -> DataflowBuildConfig:
    """Read a build directory's CONFIG_FILE, with its relative paths resolved against
    the directory (``_resolved_against``) so that they name the same files wherever the
    build runs. Refuses a directory without one, and a list of steps that does not
    resolve (its settings' paths are found by the steps that declare them)."""
    path = os.path.join(path_to_cfg_dir, CONFIG_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path_to_cfg_dir}: a build directory states {CONFIG_FILE}")
    with open(path) as f:
        cfg = DataflowBuildConfig.from_json(f.read())
    return _resolved_against(cfg, os.path.abspath(path_to_cfg_dir))


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
