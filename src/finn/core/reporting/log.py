# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: BSD-3-Clause

"""One run's log: FINN's records and Python's warnings, to a log file and the console.

Each record's level, and what reaches the console, follow one rule:

- ERROR: the run stops: a refusal (one line per finding), a tool unavailable or failed,
  a bug (with its traceback). Console: yes; a traceback only for a bug.
- WARNING: the run goes on, giving less than asked or than the user would assume: a
  noted limitation (``finn.core.reporting.manifest.note``). Console: yes.
- INFO: what the run did. Console: the run's own lines only, those of the logger
  ``runner`` names and the loggers under it (a build's ``finn.builder``: the runner's and
  each step's one-line summary); the rest in the file.
- DEBUG: per-item detail; the file only, with ``verbose``.

Python's warnings (``py.warnings``: qonnx's, NumPy's, streamlining's) go to the file at
WARNING; the console shows how many there were at the run's end, and with ``verbose``
each one. With ``verbose`` the console shows everything the file has but DEBUG.

``log_run`` sets it up for a ``with`` block (``finn.core.reporting.manifest.recorded_run``
sets it up with the run's manifest), ``running_step`` names the step stamped on each
record, and ``FILE_ONLY`` keeps a record off the console. The standard library only, so
any entry point can use it.
"""

import contextlib
import logging
import os
import sys
import warnings
from collections.abc import Iterator

#: Pass as ``extra`` to send a record to the log file only, unless ``verbose``.
FILE_ONLY = {"finn_console": False}

#: The step stamped on every record, ``-`` outside a step. A module global, not a context
#: variable: a step's thread pool (HLS synthesis) does not inherit context.
_step = "-"


@contextlib.contextmanager
def running_step(name: str) -> Iterator[None]:
    """Stamp the records of the ``with`` block with the step ``name``."""
    global _step
    outer, _step = _step, name
    try:
        yield
    finally:
        _step = outer


class _StepStamp(logging.Filter):
    """Stamps each record with the running step, and strips the trailing newline of a
    captured warning."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.step = _step
        if record.name == "py.warnings" and isinstance(record.msg, str):
            record.msg = record.msg.rstrip("\n")
        return True


class _Console(logging.Filter):
    """One console stream's share of the records from INFO: below WARNING for stdout, the
    rest for stderr (``errors``). With ``verbose``, everything the log file has;
    otherwise the INFO lines of ``runner`` and the loggers under it and FINN's warnings
    and errors, less Python's warnings and the ``FILE_ONLY`` records."""

    def __init__(self, runner: str, verbose: bool, errors: bool):
        super().__init__()
        self.runner = runner
        self.verbose = verbose
        self.errors = errors

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.INFO or (record.levelno >= logging.WARNING) != self.errors:
            return False
        if self.verbose:
            return True
        if record.name == "py.warnings" or not getattr(record, "finn_console", True):
            return False
        own = record.name == self.runner or record.name.startswith(self.runner + ".")
        return record.levelno >= logging.WARNING or own


class _Counted(logging.Handler):
    """Counts the records it is given."""

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        self.count += 1


class _ConsoleFormat(logging.Formatter):
    """The message alone for INFO and below; ``LEVEL: message`` above."""

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        return text if record.levelno < logging.WARNING else f"{record.levelname}: {text}"


@contextlib.contextmanager
def log_run(path: str | None, *, runner: str, verbose: bool = False) -> Iterator[None]:
    """Log the ``with`` block to the file ``path`` (appended to) and to the console, then
    restore the logging found; with no ``path``, to the console only.

    FINN's loggers and Python's warnings go to the file, each line with its level, step
    and logger, and to the console as the module docstring states: the INFO lines of
    ``runner`` and the loggers under it, FINN's warnings and errors, and at the end how
    many Python warnings the file has; with ``verbose``, all of it. ``finn`` logs from
    INFO, from DEBUG with ``verbose``, unless the caller set its level. No stream is
    redirected, so a third party's print reaches the console unchanged."""
    stamp = _StepStamp()
    handlers: list[logging.Handler] = []
    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        file_handler = logging.FileHandler(path, mode="a")
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(step)s %(name)s: %(message)s")
        )
        handlers.append(file_handler)
    for stream, errors in ((sys.stdout, False), (sys.stderr, True)):
        console = logging.StreamHandler(stream)
        console.addFilter(_Console(runner, verbose, errors))
        console.setFormatter(_ConsoleFormat("%(message)s"))
        handlers.append(console)
    finn, python = logging.getLogger("finn"), logging.getLogger("py.warnings")
    saved = [(logger, logger.level, logger.propagate) for logger in (finn, python)]
    for handler in handlers:
        handler.addFilter(stamp)
        for logger in (finn, python):
            logger.addHandler(handler)
    counted = _Counted()
    python.addHandler(counted)
    for logger in (finn, python):
        # The root's handlers (a notebook's) would print each line twice.
        logger.propagate = False
    shown = finn.level or (logging.DEBUG if verbose else logging.INFO)
    if shown > logging.INFO:
        for handler in handlers:
            handler.addFilter(
                lambda record: record.levelno >= shown or record.name == "py.warnings"
            )
    finn.setLevel(shown)
    # A caller that already captures warnings keeps its hook.
    capture = warnings.showwarning is not getattr(logging, "_showwarning", None)
    if capture:
        logging.captureWarnings(True)
    try:
        yield
    finally:
        if counted.count and not verbose:
            where = "" if path is None else f", in {path}"
            logging.getLogger(runner).info("%d Python warnings%s", counted.count, where)
        if capture:
            logging.captureWarnings(False)
        python.removeHandler(counted)
        for logger, level, propagate in saved:
            logger.setLevel(level)
            logger.propagate = propagate
            for handler in handlers:
                logger.removeHandler(handler)
        for handler in handlers:
            handler.close()
