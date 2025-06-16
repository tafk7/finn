############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN Flexible HLS Backend - Enhanced Diagnostics and Error Handling
############################################################################

import os
import sys
import logging
import traceback
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import json

from .path_resolver import FINNPathResolver, FINNPathType
from .template_engine import FINNTemplateEngine, FINNTemplateType
from .config import FINNConfig


class DiagnosticLevel(Enum):
    """Diagnostic severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DiagnosticMessage:
    """Structured diagnostic message."""
    level: DiagnosticLevel
    component: str
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None
    error_code: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "component": self.component,
            "message": self.message,
            "details": self.details or {},
            "suggestions": self.suggestions or [],
            "error_code": self.error_code
        }
    
    def __str__(self) -> str:
        """String representation for logging."""
        result = f"[{self.level.value.upper()}] {self.component}: {self.message}"
        if self.error_code:
            result += f" (Code: {self.error_code})"
        return result


class FINNDiagnostics:
    """Enhanced diagnostics and error handling for FINN Flexible HLS Backend."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or self._setup_default_logger()
        self.messages: List[DiagnosticMessage] = []
        self.path_resolver: Optional[FINNPathResolver] = None
        self.template_engine: Optional[FINNTemplateEngine] = None
        
    def _setup_default_logger(self) -> logging.Logger:
        """Set up default logger for diagnostics."""
        logger = logging.getLogger("finn.flexible_hls.diagnostics")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def add_message(self, 
                   level: DiagnosticLevel,
                   component: str,
                   message: str,
                   details: Optional[Dict[str, Any]] = None,
                   suggestions: Optional[List[str]] = None,
                   error_code: Optional[str] = None) -> None:
        """Add a diagnostic message."""
        msg = DiagnosticMessage(
            level=level,
            component=component,
            message=message,
            details=details,
            suggestions=suggestions,
            error_code=error_code
        )
        
        self.messages.append(msg)
        
        # Log the message
        log_level = {
            DiagnosticLevel.INFO: logging.INFO,
            DiagnosticLevel.WARNING: logging.WARNING,
            DiagnosticLevel.ERROR: logging.ERROR,
            DiagnosticLevel.CRITICAL: logging.CRITICAL
        }[level]
        
        self.logger.log(log_level, str(msg))
        
        # Print suggestions if available
        if suggestions:
            for suggestion in suggestions:
                self.logger.log(log_level, f"  Suggestion: {suggestion}")
    
    def diagnose_environment(self) -> Dict[str, Any]:
        """Perform comprehensive environment diagnostics."""
        self.logger.info("Starting FINN environment diagnostics...")
        
        # Initialize path resolver for diagnostics
        try:
            self.path_resolver = FINNPathResolver()
        except Exception as e:
            self.add_message(
                DiagnosticLevel.CRITICAL,
                "PathResolver",
                f"Failed to initialize path resolver: {str(e)}",
                error_code="ENV001"
            )
            return {"status": "failed", "error": "Path resolver initialization failed"}
        
        diagnostics = {
            "environment": self._diagnose_environment_variables(),
            "paths": self._diagnose_paths(),
            "files": self._diagnose_required_files(),
            "permissions": self._diagnose_permissions(),
            "templates": self._diagnose_templates(),
            "dependencies": self._diagnose_dependencies()
        }
        
        # Overall status
        has_errors = any(msg.level in [DiagnosticLevel.ERROR, DiagnosticLevel.CRITICAL] 
                        for msg in self.messages)
        has_warnings = any(msg.level == DiagnosticLevel.WARNING for msg in self.messages)
        
        if has_errors:
            status = "error"
        elif has_warnings:
            status = "warning"
        else:
            status = "ok"
        
        diagnostics["status"] = status
        diagnostics["messages"] = [msg.to_dict() for msg in self.messages]
        
        return diagnostics
    
    def _diagnose_environment_variables(self) -> Dict[str, Any]:
        """Diagnose FINN environment variables."""
        env_vars = {
            "FINN_ROOT": os.environ.get("FINN_ROOT"),
            "FINN_DEPS_DIR": os.environ.get("FINN_DEPS_DIR"),
            "FINN_HLSLIB_DIR": os.environ.get("FINN_HLSLIB_DIR"),
            "HLS_PATH": os.environ.get("HLS_PATH"),
            "VITIS_PATH": os.environ.get("VITIS_PATH")
        }
        
        result = {"variables": env_vars, "issues": []}
        
        # Check for missing critical variables
        critical_vars = ["FINN_ROOT", "FINN_DEPS_DIR", "FINN_HLSLIB_DIR"]
        for var in critical_vars:
            if not env_vars[var]:
                self.add_message(
                    DiagnosticLevel.WARNING,
                    "Environment",
                    f"Environment variable {var} not set, using default",
                    details={"variable": var},
                    suggestions=[
                        f"Set {var} environment variable to point to correct directory",
                        "Check FINN installation and setup scripts"
                    ],
                    error_code="ENV002"
                )
                result["issues"].append(f"Missing {var}")
        
        # Check for optional but recommended variables
        optional_vars = ["HLS_PATH", "VITIS_PATH"]
        for var in optional_vars:
            if not env_vars[var]:
                self.add_message(
                    DiagnosticLevel.INFO,
                    "Environment",
                    f"Optional environment variable {var} not set",
                    details={"variable": var},
                    suggestions=[
                        f"Set {var} if you plan to use HLS/Vitis compilation",
                        "This is only needed for actual synthesis"
                    ]
                )
        
        return result
    
    def _diagnose_paths(self) -> Dict[str, Any]:
        """Diagnose path resolution and validation."""
        if not self.path_resolver:
            return {"status": "failed", "error": "Path resolver not available"}
        
        paths = {
            "finn_root": self.path_resolver.get_finn_root(),
            "finn_deps_dir": self.path_resolver.get_finn_deps_dir(),
            "finn_hlslib_dir": self.path_resolver.get_finn_hlslib_dir(),
            "hls_path": self.path_resolver.get_hls_path(),
            "vitis_path": self.path_resolver.get_vitis_path()
        }
        
        result = {"paths": paths, "validation": {}, "issues": []}
        
        # Validate each path
        for name, path in paths.items():
            if path:
                exists = os.path.exists(path)
                is_dir = os.path.isdir(path) if exists else False
                
                result["validation"][name] = {
                    "path": path,
                    "exists": exists,
                    "is_directory": is_dir
                }
                
                if not exists:
                    level = DiagnosticLevel.ERROR if name.startswith("finn_") else DiagnosticLevel.WARNING
                    self.add_message(
                        level,
                        "PathValidation",
                        f"Path does not exist: {name} = {path}",
                        details={"path_name": name, "path": path},
                        suggestions=[
                            f"Create directory: {path}",
                            "Check environment variable configuration",
                            "Verify FINN installation"
                        ],
                        error_code="PATH001"
                    )
                    result["issues"].append(f"Missing directory: {name}")
                elif not is_dir:
                    self.add_message(
                        DiagnosticLevel.ERROR,
                        "PathValidation",
                        f"Path is not a directory: {name} = {path}",
                        details={"path_name": name, "path": path},
                        error_code="PATH002"
                    )
                    result["issues"].append(f"Not a directory: {name}")
            else:
                result["validation"][name] = {
                    "path": None,
                    "exists": False,
                    "is_directory": False
                }
        
        return result
    
    def _diagnose_required_files(self) -> Dict[str, Any]:
        """Diagnose presence of required FINN files."""
        if not self.path_resolver:
            return {"status": "failed", "error": "Path resolver not available"}
        
        # Required files for FINN HLS backend
        required_files = {
            "bnn-library.h": FINNPathType.FINN_HLSLIB,
            "cnpy.h": FINNPathType.FINN_HLSLIB,
            "npy2apintstream.hpp": FINNPathType.FINN_HLSLIB,
            "npy2vectorstream.hpp": FINNPathType.FINN_HLSLIB
        }
        
        result = {"files": {}, "missing": [], "found": []}
        
        for filename, path_type in required_files.items():
            try:
                exists = self.path_resolver.file_exists(filename, path_type)
                full_path = self.path_resolver.resolve_path(filename, path_type)
                
                result["files"][filename] = {
                    "exists": exists,
                    "path": full_path,
                    "path_type": path_type.value
                }
                
                if exists:
                    result["found"].append(filename)
                else:
                    result["missing"].append(filename)
                    self.add_message(
                        DiagnosticLevel.ERROR,
                        "RequiredFiles",
                        f"Required file missing: {filename}",
                        details={
                            "filename": filename,
                            "expected_path": full_path,
                            "path_type": path_type.value
                        },
                        suggestions=[
                            "Check FINN dependencies installation",
                            "Verify FINN_HLSLIB_DIR points to correct directory",
                            "Run FINN dependency setup scripts"
                        ],
                        error_code="FILE001"
                    )
            except Exception as e:
                result["files"][filename] = {
                    "exists": False,
                    "error": str(e),
                    "path_type": path_type.value
                }
                result["missing"].append(filename)
                self.add_message(
                    DiagnosticLevel.ERROR,
                    "RequiredFiles",
                    f"Error checking required file {filename}: {str(e)}",
                    details={"filename": filename, "error": str(e)},
                    error_code="FILE002"
                )
        
        return result
    
    def _diagnose_permissions(self) -> Dict[str, Any]:
        """Diagnose file and directory permissions."""
        if not self.path_resolver:
            return {"status": "failed", "error": "Path resolver not available"}
        
        result = {"permissions": {}, "issues": []}
        
        # Check permissions on key directories
        paths_to_check = {
            "finn_root": self.path_resolver.get_finn_root(),
            "finn_deps_dir": self.path_resolver.get_finn_deps_dir(),
            "finn_hlslib_dir": self.path_resolver.get_finn_hlslib_dir()
        }
        
        for name, path in paths_to_check.items():
            if path and os.path.exists(path):
                try:
                    readable = os.access(path, os.R_OK)
                    writable = os.access(path, os.W_OK)
                    executable = os.access(path, os.X_OK)
                    
                    result["permissions"][name] = {
                        "path": path,
                        "readable": readable,
                        "writable": writable,
                        "executable": executable
                    }
                    
                    if not readable:
                        self.add_message(
                            DiagnosticLevel.ERROR,
                            "Permissions",
                            f"Directory not readable: {path}",
                            details={"path": path, "permission": "read"},
                            suggestions=["Check directory permissions", "Run with appropriate privileges"],
                            error_code="PERM001"
                        )
                        result["issues"].append(f"Not readable: {name}")
                    
                    if not executable:
                        self.add_message(
                            DiagnosticLevel.WARNING,
                            "Permissions",
                            f"Directory not executable: {path}",
                            details={"path": path, "permission": "execute"},
                            suggestions=["Check directory permissions"],
                            error_code="PERM002"
                        )
                        result["issues"].append(f"Not executable: {name}")
                        
                except Exception as e:
                    result["permissions"][name] = {"error": str(e)}
                    self.add_message(
                        DiagnosticLevel.WARNING,
                        "Permissions",
                        f"Could not check permissions for {path}: {str(e)}",
                        error_code="PERM003"
                    )
        
        return result
    
    def _diagnose_templates(self) -> Dict[str, Any]:
        """Diagnose template system functionality."""
        result = {"templates": {}, "issues": []}
        
        try:
            # Try to initialize template engine
            template_paths = []
            if self.path_resolver:
                template_paths = self.path_resolver.get_template_search_paths()
            
            self.template_engine = FINNTemplateEngine(template_paths)
            
            # Test template availability
            template_types = [
                ("ipgen_cpp", FINNTemplateType.IPGEN_CPP),
                ("ipgen_tcl", FINNTemplateType.IPGEN_TCL),
                ("docompute", FINNTemplateType.DOCOMPUTE),
                ("docompute_timeout", FINNTemplateType.DOCOMPUTE_TIMEOUT)
            ]
            
            for template_name, template_type in template_types:
                try:
                    # Test if template can be loaded
                    variables = self.template_engine.get_template_variables(template_name, template_type)
                    available = variables is not None
                    
                    result["templates"][template_name] = {
                        "available": available,
                        "type": template_type.value,
                        "variables_count": len(variables) if variables else 0
                    }
                    
                    if not available:
                        self.add_message(
                            DiagnosticLevel.WARNING,
                            "Templates",
                            f"Template not available: {template_name}",
                            details={"template": template_name, "type": template_type.value},
                            suggestions=[
                                "Check template file exists in template directories",
                                "Verify template search paths are correct"
                            ],
                            error_code="TMPL001"
                        )
                        result["issues"].append(f"Missing template: {template_name}")
                        
                except Exception as e:
                    result["templates"][template_name] = {
                        "available": False,
                        "error": str(e),
                        "type": template_type.value
                    }
                    self.add_message(
                        DiagnosticLevel.ERROR,
                        "Templates",
                        f"Error loading template {template_name}: {str(e)}",
                        details={"template": template_name, "error": str(e)},
                        error_code="TMPL002"
                    )
                    result["issues"].append(f"Error loading: {template_name}")
                    
        except Exception as e:
            result["templates"] = {"initialization_error": str(e)}
            self.add_message(
                DiagnosticLevel.CRITICAL,
                "Templates",
                f"Failed to initialize template engine: {str(e)}",
                details={"error": str(e)},
                suggestions=[
                    "Check template directory structure",
                    "Verify path resolver is working correctly"
                ],
                error_code="TMPL003"
            )
            result["issues"].append("Template engine initialization failed")
        
        return result
    
    def _diagnose_dependencies(self) -> Dict[str, Any]:
        """Diagnose Python dependencies and imports."""
        result = {"dependencies": {}, "issues": []}
        
        # Required Python modules
        required_modules = [
            ("os", "Built-in"),
            ("sys", "Built-in"),
            ("pathlib", "Built-in"),
            ("typing", "Built-in"),
            ("dataclasses", "Built-in"),
            ("enum", "Built-in"),
            ("logging", "Built-in")
        ]
        
        for module_name, module_type in required_modules:
            try:
                __import__(module_name)
                result["dependencies"][module_name] = {
                    "available": True,
                    "type": module_type
                }
            except ImportError as e:
                result["dependencies"][module_name] = {
                    "available": False,
                    "error": str(e),
                    "type": module_type
                }
                self.add_message(
                    DiagnosticLevel.ERROR,
                    "Dependencies",
                    f"Required module not available: {module_name}",
                    details={"module": module_name, "error": str(e)},
                    suggestions=[
                        "Check Python installation",
                        "Verify Python version compatibility"
                    ],
                    error_code="DEP001"
                )
                result["issues"].append(f"Missing module: {module_name}")
        
        return result
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """Generate comprehensive diagnostic report."""
        diagnostics = self.diagnose_environment()
        
        # Generate human-readable report
        report_lines = [
            "FINN Flexible HLS Backend - Diagnostic Report",
            "=" * 50,
            f"Overall Status: {diagnostics['status'].upper()}",
            ""
        ]
        
        # Environment section
        report_lines.append("Environment Variables:")
        for var, value in diagnostics["environment"]["variables"].items():
            status = "✓" if value else "✗"
            report_lines.append(f"  {status} {var}: {value or 'Not set'}")
        report_lines.append("")
        
        # Paths section
        report_lines.append("Path Validation:")
        for name, info in diagnostics["paths"]["validation"].items():
            if info["exists"]:
                status = "✓" if info["is_directory"] else "⚠"
            else:
                status = "✗"
            report_lines.append(f"  {status} {name}: {info['path']}")
        report_lines.append("")
        
        # Files section
        report_lines.append("Required Files:")
        for filename, info in diagnostics["files"]["files"].items():
            status = "✓" if info["exists"] else "✗"
            report_lines.append(f"  {status} {filename}")
        report_lines.append("")
        
        # Templates section
        report_lines.append("Templates:")
        for template, info in diagnostics["templates"]["templates"].items():
            status = "✓" if info["available"] else "✗"
            report_lines.append(f"  {status} {template}")
        report_lines.append("")
        
        # Messages section
        if diagnostics["messages"]:
            report_lines.append("Diagnostic Messages:")
            for msg_dict in diagnostics["messages"]:
                level = msg_dict["level"].upper()
                component = msg_dict["component"]
                message = msg_dict["message"]
                report_lines.append(f"  [{level}] {component}: {message}")
                
                if msg_dict["suggestions"]:
                    for suggestion in msg_dict["suggestions"]:
                        report_lines.append(f"    → {suggestion}")
            report_lines.append("")
        
        # Summary
        error_count = sum(1 for msg in diagnostics["messages"] if msg["level"] in ["error", "critical"])
        warning_count = sum(1 for msg in diagnostics["messages"] if msg["level"] == "warning")
        
        report_lines.append("Summary:")
        report_lines.append(f"  Errors: {error_count}")
        report_lines.append(f"  Warnings: {warning_count}")
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
                f.write("\n\n")
                f.write("Detailed Diagnostics (JSON):\n")
                f.write(json.dumps(diagnostics, indent=2))
        
        return report
    
    def get_error_count(self) -> int:
        """Get count of error-level messages."""
        return sum(1 for msg in self.messages if msg.level in [DiagnosticLevel.ERROR, DiagnosticLevel.CRITICAL])
    
    def get_warning_count(self) -> int:
        """Get count of warning-level messages."""
        return sum(1 for msg in self.messages if msg.level == DiagnosticLevel.WARNING)
    
    def has_errors(self) -> bool:
        """Check if there are any error-level messages."""
        return self.get_error_count() > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warning-level messages."""
        return self.get_warning_count() > 0
    
    def clear_messages(self) -> None:
        """Clear all diagnostic messages."""
        self.messages.clear()


def run_diagnostics(output_file: Optional[str] = None) -> int:
    """Run comprehensive diagnostics and return exit code."""
    diagnostics = FINNDiagnostics()
    
    try:
        report = diagnostics.generate_report(output_file)
        print(report)
        
        # Return appropriate exit code
        if diagnostics.has_errors():
            return 1
        elif diagnostics.has_warnings():
            return 2
        else:
            return 0
            
    except Exception as e:
        print(f"Diagnostic failed with error: {str(e)}")
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FINN Flexible HLS Backend Diagnostics")
    parser.add_argument("--output", "-o", help="Output file for detailed report")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    exit_code = run_diagnostics(args.output)
    sys.exit(exit_code)