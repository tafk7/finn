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
import warnings
from dataclasses import replace
from qonnx.core.modelwrapper import ModelWrapper

from finn import resources
from finn.builder.build_dataflow_checks import (
    log_report,
    run_all_config_checks,
    save_report,
)
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    default_build_dataflow_steps,
)
from finn.builder.build_dataflow_phases import build_dataflow_phase_lookup
from finn.builder.build_dataflow_steps import (
    ExploreOptions,
    PrepareIoOptions,
    build_dataflow_step_lookup,
)
from finn.core.space import by_code
from finn.toolchain.launcher import ToolFailed, ToolUnavailable

#: The configuration file of a build directory (build_dataflow_directory), next to
#: model.onnx.
CONFIG_FILE = "dataflow_build_config.json"

#: The build log, in the output directory.
LOG_FILE = "build_dataflow.log"

#: The runner's logger. The console shows its INFO lines (the build's paths, each step
#: it runs, the outcome); from other loggers, only warnings and above.
_LOG = logging.getLogger(__name__)

#: Pass as ``extra`` to send a record to the build log only, unless ``verbose`` (used
#: for a refusal's traceback).
FILE_ONLY = {"finn_console": False}

#: The step running now, stamped on every log record. A module global, not a context
#: variable, because a step's thread pool (HLS synthesis) does not inherit context.
_step = "-"


class _StepStamp(logging.Filter):
    """Stamps each record with the running step (``step``), and strips the trailing
    newline warnings.formatwarning adds to a captured warning."""

    def filter(self, record):
        record.step = _step
        if record.name == "py.warnings" and isinstance(record.msg, str):
            record.msg = record.msg.rstrip("\n")
        return True


class _Console(logging.Filter):
    """Selects what the build's console shows: with ``verbose``, everything the log file
    has; otherwise the runner's lines and every logger's warnings and errors. INFO and
    below go to stdout, WARNING and above to stderr (``errors``)."""

    def __init__(self, verbose: bool, errors: bool):
        super().__init__()
        self.verbose = verbose
        self.errors = errors

    def filter(self, record):
        if (record.levelno >= logging.WARNING) != self.errors:
            return False
        if self.verbose:
            return True
        if not getattr(record, "finn_console", True):
            return False
        return record.levelno >= logging.WARNING or record.name == _LOG.name


class _ConsoleFormat(logging.Formatter):
    """The message alone for INFO and below; ``LEVEL: message`` above."""

    def format(self, record):
        text = super().format(record)
        return text if record.levelno < logging.WARNING else f"{record.levelname}: {text}"


@contextlib.contextmanager
def _build_log(cfg):
    """Set up the build's log for the ``with`` block. FINN's loggers (``finn``) and
    Python's warnings (``py.warnings``, captured through ``logging.captureWarnings``) go
    to ``LOG_FILE`` in ``cfg.output_dir``, each line with its level, step and logger,
    and to the console (_Console). ``finn`` logs from INFO unless the caller set a level.
    No stream is redirected, so a third-party print reaches the console unchanged. The
    handlers, levels and warnings hook are restored afterwards, leaving a notebook's own
    logging as it was."""
    global _step
    os.makedirs(cfg.output_dir, exist_ok=True)
    stamp = _StepStamp()
    file_handler = logging.FileHandler(os.path.join(cfg.output_dir, LOG_FILE), mode="a")
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(levelname)s %(step)s %(name)s: %(message)s")
    )
    handlers = [file_handler]
    for stream, errors in ((sys.stdout, False), (sys.stderr, True)):
        console = logging.StreamHandler(stream)
        console.addFilter(_Console(cfg.verbose, errors))
        console.setFormatter(_ConsoleFormat("%(message)s"))
        handlers.append(console)
    loggers = [logging.getLogger("finn"), logging.getLogger("py.warnings")]
    saved = [(logger, logger.level, logger.propagate) for logger in loggers]
    for handler in handlers:
        handler.addFilter(stamp)
        for logger in loggers:
            logger.addHandler(handler)
    for logger in loggers:
        # Not also to the root's handlers (a notebook's), which would print each line twice.
        logger.propagate = False
    if loggers[0].level == logging.NOTSET:
        loggers[0].setLevel(logging.INFO)
    # Leave the warnings hook alone if the caller already captures warnings.
    capture = warnings.showwarning is not getattr(logging, "_showwarning", None)
    if capture:
        logging.captureWarnings(True)
    try:
        yield
    finally:
        _step = "-"
        if capture:
            logging.captureWarnings(False)
        for logger, level, propagate in saved:
            logger.setLevel(level)
            logger.propagate = propagate
            for handler in handlers:
                logger.removeHandler(handler)
        for handler in handlers:
            handler.close()


def _log_failure(step: str, error: BaseException) -> None:
    """Log a failed step's error according to its kind; the traceback always goes to the
    log file. A ``ToolUnavailable`` names the tool and what selects it; a ``ToolFailed``
    its transcript and error lines; a refusal (an error with ``findings``) their number
    and one line per code; a failed verification (a subclass of ``AssertionError``)
    ``Verification for <step> : FAIL``. Anything else is a bug, and its traceback is
    shown on the console too."""
    kind = type(error).__name__
    if isinstance(error, (ToolUnavailable, ToolFailed)):
        _LOG.error("%s: %s: %s", step, kind, error)
    elif hasattr(error, "findings"):
        findings = tuple(error.findings)
        _LOG.error("%s refused: %d findings: %s: %s", step, len(findings), kind, error)
        for line in by_code(findings):
            _LOG.error("  %s", line)
    elif isinstance(error, AssertionError) and type(error) is not AssertionError:
        _LOG.error("Verification for %s : FAIL", step)
        _LOG.error("%s: %s", kind, error)
    else:
        _LOG.error("%s failed: %s: %s", step, kind, error, exc_info=error)
        return
    _LOG.error("%s: the traceback", step, exc_info=error, extra=FILE_ONLY)


def _expanded_steps(cfg: DataflowBuildConfig):
    """The build's steps in run order (``cfg.steps``, else default_build_dataflow_steps),
    with each phase expanded into its steps and the ``inject_steps_before`` and
    ``inject_steps_after`` functions placed around each step or phase. Also returns
    each phase's first and last index in that list. Refuses an unknown name, listing
    the steps and phases."""
    steps = cfg.steps
    if steps is None:
        steps = default_build_dataflow_steps

    steps_as_fxns = []
    phase_spans = {}

    def place(step_name, step_fn):
        steps_as_fxns.extend(cfg.inject_steps_before.get(step_name, []))
        steps_as_fxns.append(step_fn)
        steps_as_fxns.extend(cfg.inject_steps_after.get(step_name, []))

    for transform_step in steps:
        if type(transform_step) is str and transform_step in build_dataflow_phase_lookup:
            first = len(steps_as_fxns)
            steps_as_fxns.extend(cfg.inject_steps_before.get(transform_step, []))
            for step_name in build_dataflow_phase_lookup[transform_step]:
                place(step_name, build_dataflow_step_lookup[step_name])
            steps_as_fxns.extend(cfg.inject_steps_after.get(transform_step, []))
            phase_spans.setdefault(transform_step, (first, len(steps_as_fxns) - 1))
        elif type(transform_step) is str:
            if transform_step not in build_dataflow_step_lookup:
                raise ValueError(
                    f"Unknown step or phase: {transform_step}; the steps: "
                    f"{', '.join(build_dataflow_step_lookup)}; the phases: "
                    f"{', '.join(build_dataflow_phase_lookup)}"
                )
            place(transform_step, build_dataflow_step_lookup[transform_step])
        elif callable(transform_step):
            place(getattr(transform_step, "__name__", None), transform_step)
        else:
            raise ValueError(f"Invalid step type: {type(transform_step)}")
    return steps_as_fxns, phase_spans


def _step_index(name: str, steps_as_fxns, phase_spans, last: bool) -> int:
    """The index of the step ``name`` in the build's expanded steps; for a phase, of its
    first step, or its last if ``last``. Refuses a name the build does not run, listing
    its steps and phases."""
    if name in phase_spans:
        return phase_spans[name][1 if last else 0]
    step_names = [step_fn.__name__ for step_fn in steps_as_fxns]
    if name in step_names:
        return step_names.index(name)
    raise ValueError(
        f"{name!r} is not one of this build's steps or phases; its steps: "
        f"{', '.join(step_names)}; its phases: {', '.join(phase_spans) or 'none'}"
    )


def resolve_build_steps(cfg: DataflowBuildConfig, partial: bool = True):
    """The build's steps as functions in run order, phases expanded and injected steps
    placed. If ``partial``, only those from ``cfg.start_step`` to ``cfg.stop_step``;
    either may name a phase, meaning its first step to start from or its last to stop
    after."""
    steps_as_fxns, phase_spans = _expanded_steps(cfg)
    if partial:
        if cfg.start_step is None:
            start_ind = 0
        else:
            start_ind = _step_index(cfg.start_step, steps_as_fxns, phase_spans, last=False)
        if cfg.stop_step is None:
            stop_ind = len(steps_as_fxns) - 1
        else:
            stop_ind = _step_index(cfg.stop_step, steps_as_fxns, phase_spans, last=True)
        steps_as_fxns = steps_as_fxns[start_ind : (stop_ind + 1)]
    return steps_as_fxns


def resolve_step_filename(step_name: str, cfg: DataflowBuildConfig, step_delta: int = 0):
    """The intermediate model file of the step ``step_delta`` places after ``step_name``
    (for a phase, its first step) in the build's steps."""
    steps_as_fxns, phase_spans = _expanded_steps(cfg)
    step_no = _step_index(step_name, steps_as_fxns, phase_spans, last=False) + step_delta
    assert step_no >= 0, "Invalid step+delta combination"
    assert step_no < len(steps_as_fxns), "Invalid step+delta combination"
    filename = cfg.output_dir + "/intermediate_models/"
    filename += "%s.onnx" % (steps_as_fxns[step_no].__name__)
    return filename


def _run_build_steps(model, cfg, build_dataflow_steps):
    """Run the resolved build steps in order, each log record stamped with its step.

    Writes time_per_step.json, the seconds each step took. Returns 0 on success, or -1
    at the first failing step after logging its failure (_log_failure) and starting the
    postmortem debugger if ``enable_build_pdb_debug`` is set and stdin is a terminal.
    """
    global _step
    step_num = 1
    time_per_step = dict()
    for transform_step in build_dataflow_steps:
        step_name = transform_step.__name__
        try:
            _step = step_name
            _LOG.info("Running step: %s [%d/%d]", step_name, step_num, len(build_dataflow_steps))
            # run the step
            step_start = time.time()
            model = transform_step(model, cfg)
            step_end = time.time()
            time_per_step[step_name] = step_end - step_start
            chkpt_name = "%s.onnx" % (step_name)
            if cfg.save_intermediate_models:
                intermediate_model_dir = cfg.output_dir + "/intermediate_models"
                if not os.path.exists(intermediate_model_dir):
                    os.makedirs(intermediate_model_dir)
                model.save("%s/%s" % (intermediate_model_dir, chkpt_name))
            step_num += 1
        except:  # noqa
            extype, value, tb = sys.exc_info()
            _log_failure(step_name, value)
            # start postmortem debug if configured, and only where someone can answer it:
            # with stdin not a terminal (pytest, a batch job) pdb's read would fail and
            # report itself instead of the error above
            if not cfg.enable_build_pdb_debug:
                _LOG.info("enable_build_pdb_debug not set in build config, exiting...")
            elif sys.stdin is None or not sys.stdin.isatty():
                _LOG.info("enable_build_pdb_debug set, but stdin is not a terminal, exiting...")
            else:
                pdb.post_mortem(tb)
            _LOG.error("Build failed")
            return -1
        finally:
            _step = "-"

    with open(cfg.output_dir + "/time_per_step.json", "w") as f:
        json.dump(time_per_step, f, indent=2)
    _LOG.info("Completed successfully")
    return 0


def build_dataflow_cfg(model_filename, cfg: DataflowBuildConfig):
    """Best-effort build a dataflow accelerator using the given configuration, through
    the steps of finn.builder.build_dataflow_steps.

    The build log (``LOG_FILE`` in the output directory, and the console) is set up
    first, so the configuration checks are logged too, and a refused configuration is
    logged before it raises.

    :param model_filename: ONNX model filename to build
    :param cfg: Build configuration
    """
    with _build_log(cfg):
        try:
            # if start_step is specified, override the input model
            if cfg.start_step is None:
                _LOG.info("Building dataflow accelerator from " + model_filename)
                model = ModelWrapper(model_filename)
            else:
                intermediate_model_filename = resolve_step_filename(cfg.start_step, cfg, -1)
                _LOG.info(
                    "Building dataflow accelerator from intermediate checkpoint"
                    + intermediate_model_filename
                )
                model = ModelWrapper(intermediate_model_filename)
            assert type(model) is ModelWrapper
            finn_build_dir = resources.scratch()

            _LOG.info("Intermediate outputs will be generated in " + str(finn_build_dir))
            _LOG.info("Final outputs will be generated in " + cfg.output_dir)
            _LOG.info("Build log is at " + os.path.join(cfg.output_dir, LOG_FILE))

            # Run configuration checks
            config_report = run_all_config_checks(cfg)
            log_report(config_report)
            report_path = save_report(config_report, cfg.output_dir)
            _LOG.info(f"Configuration check report saved to: {report_path}")

            if config_report.has_errors():
                raise AssertionError(
                    "Configuration check failed with errors. Fix the issues above to proceed."
                )

            build_dataflow_steps = resolve_build_steps(cfg)
        except Exception as error:
            _LOG.error("Build failed before its first step: %s: %s", type(error).__name__, error)
            _LOG.error("the traceback", exc_info=error, extra=FILE_ONLY)
            raise
        return _run_build_steps(model, cfg, build_dataflow_steps)


def _resolved_against(cfg: DataflowBuildConfig, directory: str) -> DataflowBuildConfig:
    """``cfg`` with each relative path joined to ``directory``: ``output_dir``,
    ``verify_input_npy``, step_prepare_io's ``preprocessing`` model, a strategy's
    ``path`` in step_explore's ``kernel_exploration`` (the ``pinned`` choices file), and
    the toolchain's settings scripts and command directory. Absolute paths and an empty
    command directory are kept as they are."""

    def resolve(path: str) -> str:
        return os.path.join(directory, path)

    step_options = dict(cfg.step_options)
    io = cfg.options("step_prepare_io", PrepareIoOptions)
    if io.preprocessing is not None:
        step_options["step_prepare_io"] = replace(io, preprocessing=resolve(io.preprocessing))
    explore = cfg.options("step_explore", ExploreOptions)
    if any(isinstance(spec.get("path"), str) for spec in explore.kernel_exploration):
        step_options["step_explore"] = replace(
            explore,
            kernel_exploration=[
                {**spec, "path": resolve(spec["path"])}
                if isinstance(spec.get("path"), str)
                else spec
                for spec in explore.kernel_exploration
            ],
        )
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
        step_options=step_options,
    )


def read_build_config(path_to_cfg_dir: str) -> DataflowBuildConfig:
    """Read a build directory's CONFIG_FILE, with its relative paths resolved against
    the directory (``_resolved_against``) so that they name the same files wherever the
    build runs. Refuses a directory without one."""
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
