############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
############################################################################

import pytest

import io
import logging
import os
import subprocess
import sys
import tempfile
import unittest

from finn.util.basic import _detect_log_level, launch_process_helper


@pytest.mark.util
class TestDetectLogLevel(unittest.TestCase):
    """Test log level detection patterns."""

    def test_error_patterns(self):
        """ERROR patterns detected correctly."""
        assert _detect_log_level("ERROR: Something failed", logging.INFO) == logging.ERROR
        assert _detect_log_level("FATAL: Critical error", logging.INFO) == logging.ERROR
        assert _detect_log_level("FAILED to compile", logging.INFO) == logging.ERROR
        assert _detect_log_level("Exception occurred", logging.INFO) == logging.ERROR

    def test_warning_patterns(self):
        """WARNING patterns detected correctly."""
        assert (
            _detect_log_level("WARNING: [XSIM 43-4099] Module has no timescale", logging.INFO)
            == logging.WARNING
        )
        assert _detect_log_level("WARN: Deprecated feature", logging.INFO) == logging.WARNING

    def test_debug_patterns(self):
        """Verbose tool output detected as DEBUG."""
        assert _detect_log_level("Compiling module work.foo", logging.INFO) == logging.DEBUG
        assert (
            _detect_log_level("Compiling architecture xilinx of entity bar", logging.INFO)
            == logging.DEBUG
        )
        assert _detect_log_level("Analyzing entity baz", logging.INFO) == logging.DEBUG
        assert _detect_log_level("Elaborating entity qux", logging.INFO) == logging.DEBUG

    def test_info_patterns(self):
        """INFO patterns detected correctly."""
        assert _detect_log_level("INFO: Build complete", logging.WARNING) == logging.INFO
        assert _detect_log_level("NOTE: Using default settings", logging.WARNING) == logging.INFO

    def test_default_fallback(self):
        """Unknown patterns use default level."""
        assert _detect_log_level("Random output", logging.INFO) == logging.INFO
        assert _detect_log_level("Random output", logging.DEBUG) == logging.DEBUG
        assert _detect_log_level("Random output", logging.WARNING) == logging.WARNING

    def test_case_insensitive(self):
        """Pattern matching is case-insensitive."""
        assert _detect_log_level("error: lower case", logging.INFO) == logging.ERROR
        assert _detect_log_level("ERROR: UPPER CASE", logging.INFO) == logging.ERROR
        assert _detect_log_level("ErRoR: MiXeD CaSe", logging.INFO) == logging.ERROR


@pytest.mark.util
class TestLaunchProcessHelperLegacy(unittest.TestCase):
    """Test legacy mode (backward compatibility)."""

    def test_returns_tuple(self):
        """Legacy mode returns (stdout, stderr) tuple."""
        out, err = launch_process_helper(["echo", "hello"])
        self.assertIsInstance(out, str)
        self.assertIsInstance(err, str)
        self.assertIn("hello", out)

    def test_stdout_content(self):
        """Legacy mode captures stdout correctly."""
        out, err = launch_process_helper(["echo", "test message"])
        self.assertIn("test message", out)

    def test_stderr_content(self):
        """Legacy mode captures stderr correctly."""
        out, err = launch_process_helper(["sh", "-c", "echo error >&2"])
        self.assertIn("error", err)

    def test_writes_to_stdout(self):
        """Legacy mode writes to sys.stdout."""
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            out, err = launch_process_helper(["echo", "stdout_test"])
            output = sys.stdout.getvalue()
            self.assertIn("stdout_test", output)
        finally:
            sys.stdout = old_stdout

    def test_writes_to_stderr(self):
        """Legacy mode writes to sys.stderr."""
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            out, err = launch_process_helper(["sh", "-c", "echo stderr_test >&2"])
            output = sys.stderr.getvalue()
            self.assertIn("stderr_test", output)
        finally:
            sys.stderr = old_stderr

    def test_no_exit_code_check(self):
        """Legacy mode does not raise on error."""
        # Should not raise even though false exits with 1
        out, err = launch_process_helper(["false"])
        self.assertIsInstance(out, str)
        self.assertIsInstance(err, str)

    def test_custom_env(self):
        """Legacy mode respects proc_env parameter."""
        env = os.environ.copy()
        env["TEST_VAR_LEGACY"] = "test_value_12345"

        out, err = launch_process_helper(["sh", "-c", "echo $TEST_VAR_LEGACY"], proc_env=env)
        self.assertIn("test_value_12345", out)

    def test_custom_cwd(self):
        """Legacy mode respects cwd parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out, err = launch_process_helper(["pwd"], cwd=tmpdir)
            self.assertIn(tmpdir, out)

    def test_backward_compatibility(self):
        """Existing call pattern works unchanged."""
        # This is how existing FINN code calls it
        launch_process_helper(["echo", "compat_test"])
        # Should not raise, returns tuple


@pytest.mark.util
class TestLaunchProcessHelperLogging(unittest.TestCase):
    """Test logging mode (new functionality)."""

    def test_returns_exitcode(self):
        """Logging mode returns exit code integer."""
        exitcode = launch_process_helper(["true"], use_logging=True)
        self.assertIsInstance(exitcode, int)
        self.assertEqual(exitcode, 0)

    def test_success_exitcode(self):
        """Logging mode returns 0 on success."""
        exitcode = launch_process_helper(["echo", "success"], use_logging=True)
        self.assertEqual(exitcode, 0)

    def test_error_exitcode(self):
        """Logging mode returns non-zero exit code."""
        exitcode = launch_process_helper(["false"], use_logging=True, raise_on_error=False)
        self.assertEqual(exitcode, 1)

    def test_streams_to_logger(self):
        """Logging mode sends output through logger."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            exitcode = launch_process_helper(["echo", "hello"], use_logging=True)
            self.assertEqual(exitcode, 0)
            # Check that output was logged
            self.assertTrue(any("hello" in msg for msg in cm.output))

    def test_custom_logger(self):
        """Can specify custom logger."""
        custom_logger = logging.getLogger("test.custom")
        with self.assertLogs("test.custom", level="INFO") as cm:
            launch_process_helper(["echo", "custom"], use_logging=True, logger=custom_logger)
            self.assertTrue(any("custom" in msg for msg in cm.output))

    def test_stdout_level(self):
        """Can specify stdout log level."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["echo", "debug_output"], use_logging=True, stdout_level=logging.DEBUG
            )
            # Should be logged at DEBUG level
            log_output = "\n".join(cm.output)
            self.assertIn("DEBUG", log_output)
            self.assertIn("debug_output", log_output)

    def test_stderr_level(self):
        """Stderr uses different log level than stdout."""
        with self.assertLogs("finn.subprocess", level="WARNING") as cm:
            launch_process_helper(
                ["sh", "-c", "echo error >&2"],
                use_logging=True,
                stderr_level=logging.WARNING,
            )
            log_output = "\n".join(cm.output)
            self.assertIn("WARNING", log_output)

    def test_detect_levels_enabled(self):
        """Auto-detection adjusts levels based on content."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", "echo 'ERROR: test error'"],
                use_logging=True,
                stdout_level=logging.INFO,  # Base level
                detect_levels=True,
            )
            log_output = "\n".join(cm.output)
            # Should be promoted to ERROR level
            self.assertIn("ERROR", log_output)
            self.assertIn("test error", log_output)

    def test_detect_levels_disabled(self):
        """Can disable auto-detection."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", "echo 'ERROR: test'"],
                use_logging=True,
                stdout_level=logging.INFO,
                detect_levels=False,  # Disabled
            )
            log_output = "\n".join(cm.output)
            # Should use base level (INFO), not promote to ERROR
            self.assertIn("INFO", log_output)

    def test_raise_on_error_disabled(self):
        """Does not raise when raise_on_error=False."""
        # Should return exit code without raising
        exitcode = launch_process_helper(
            ["sh", "-c", "exit 42"], use_logging=True, raise_on_error=False
        )
        self.assertEqual(exitcode, 42)

    def test_raise_on_error_enabled(self):
        """Raises CalledProcessError when raise_on_error=True."""
        with self.assertRaises(subprocess.CalledProcessError) as cm:
            launch_process_helper(["false"], use_logging=True, raise_on_error=True)
        self.assertEqual(cm.exception.returncode, 1)


@pytest.mark.util
class TestLaunchProcessHelperIntegration(unittest.TestCase):
    """Integration tests with realistic scenarios."""

    def test_multiline_output(self):
        """Handles multi-line output correctly."""
        with self.assertLogs("finn.subprocess", level="INFO") as cm:
            launch_process_helper(
                ["sh", "-c", "echo 'line1'; echo 'line2'; echo 'line3'"],
                use_logging=True,
            )
            log_output = "\n".join(cm.output)
            self.assertIn("line1", log_output)
            self.assertIn("line2", log_output)
            self.assertIn("line3", log_output)

    def test_mixed_stdout_stderr(self):
        """Handles mixed stdout and stderr streams."""
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", "echo 'out'; echo 'err' >&2; echo 'out2'"],
                use_logging=True,
                stdout_level=logging.INFO,
                stderr_level=logging.WARNING,
            )
            log_output = "\n".join(cm.output)
            self.assertIn("out", log_output)
            self.assertIn("err", log_output)
            self.assertIn("out2", log_output)

    def test_working_directory(self):
        """Respects working directory parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
                launch_process_helper(["pwd"], use_logging=True, cwd=tmpdir)
                log_output = "\n".join(cm.output)
                self.assertIn(tmpdir, log_output)

    def test_environment_variables(self):
        """Respects environment variable parameter."""
        env = os.environ.copy()
        env["TEST_VAR_LOGGING"] = "test_value_67890"

        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", "echo $TEST_VAR_LOGGING"],
                use_logging=True,
                proc_env=env,
            )
            log_output = "\n".join(cm.output)
            self.assertIn("test_value_67890", log_output)

    def test_long_output_no_deadlock(self):
        """Handles long output without deadlock."""
        # Generate 1000 lines of output
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            exitcode = launch_process_helper(
                ["sh", "-c", "for i in $(seq 1 1000); do echo line$i; done"],
                use_logging=True,
                stdout_level=logging.DEBUG,
            )
            self.assertEqual(exitcode, 0)
            # Should have logged many lines
            self.assertGreater(len(cm.output), 100)

    def test_concurrent_output(self):
        """Handles concurrent stdout/stderr without issues."""
        # Mix stdout and stderr rapidly
        cmd = "for i in $(seq 1 50); do echo out$i; echo err$i >&2; done"
        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            exitcode = launch_process_helper(
                ["sh", "-c", cmd],
                use_logging=True,
                stdout_level=logging.INFO,
                stderr_level=logging.WARNING,
            )
            self.assertEqual(exitcode, 0)
            log_output = "\n".join(cm.output)
            # Should contain output from both streams
            self.assertIn("out", log_output)
            self.assertIn("err", log_output)

    def test_xilinx_pattern_detection(self):
        """Detects Xilinx tool patterns correctly."""
        xilinx_output = """Compiling module work.foo
Compiling architecture rtl of entity bar
WARNING: [XSIM 43-4099] Module has no timescale
ERROR: Synthesis failed
INFO: Elaboration complete"""

        with self.assertLogs("finn.subprocess", level="DEBUG") as cm:
            launch_process_helper(
                ["sh", "-c", f"echo '{xilinx_output}'"],
                use_logging=True,
                stdout_level=logging.INFO,
                detect_levels=True,
            )
            log_output = "\n".join(cm.output)
            # Should have detected different levels
            self.assertIn("DEBUG", log_output)  # Compiling module
            self.assertIn("WARNING", log_output)  # WARNING:
            self.assertIn("ERROR", log_output)  # ERROR:
            self.assertIn("INFO", log_output)  # INFO:


if __name__ == "__main__":
    unittest.main()
