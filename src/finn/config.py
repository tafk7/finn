############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Portions of this content consist of AI generated content.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ##########################################################################

"""FINN configuration management using Pydantic.

This module provides a centralized configuration system for FINN that:
- Automatically detects FINN_ROOT if not set
- Provides sensible defaults for common paths
- Validates configuration at startup
- Maintains backward compatibility with environment variables
"""

import os
import warnings
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class XilinxToolsConfig(BaseModel):
    """Configuration for Xilinx tools paths."""

    vivado_path: Optional[Path] = Field(None, description="Path to Vivado installation")
    vitis_path: Optional[Path] = Field(None, description="Path to Vitis installation")
    hls_path: Optional[Path] = Field(None, description="Path to Vitis HLS installation")
    xrt_path: Optional[Path] = Field(None, description="Path to XRT installation")
    platform_repo_paths: Optional[Path] = Field(None, description="Path to Vitis platform files")

    @field_validator("vivado_path", "vitis_path", "hls_path", "xrt_path", "platform_repo_paths")
    @classmethod
    def validate_path_exists(cls, v: Optional[Path]) -> Optional[Path]:
        if v is not None and not v.exists():
            warnings.warn(f"Path does not exist: {v}", UserWarning)
        return v


class RuntimeConfig(BaseModel):
    """Runtime configuration options."""

    rtlsim_trace_depth: int = Field(1, ge=1, description="RTL simulation trace depth")
    liveness_threshold: int = Field(1000000, ge=1, description="Cycles before simulation timeout")
    num_default_workers: int = Field(4, ge=1, description="Default parallel worker count")
    imagenet_val_path: Optional[Path] = Field(None, description="Path to ImageNet validation")
    ohmyxilinx: Optional[Path] = Field(None, description="Path to oh-my-xilinx utilities")
    vivado_ip_cache: Path = Field(Path("/tmp/vivado_ip_cache"), description="Vivado IP cache dir")


class FINNConfig(BaseModel):
    """Main FINN configuration model."""

    finn_root: Path = Field(description="FINN repository root directory")
    finn_deps_dir: Path = Field(description="FINN dependencies directory")
    finn_build_dir: Path = Field(description="FINN build outputs directory")
    xilinx_tools: XilinxToolsConfig = Field(default_factory=XilinxToolsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @field_validator("finn_root", "finn_deps_dir", "finn_build_dir")
    @classmethod
    def validate_directory_paths(cls, v: Path) -> Path:
        if not v.is_absolute():
            raise ValueError(f"Path must be absolute: {v}")
        return v

    @model_validator(mode="after")
    def validate_finn_dirs_exist(self) -> "FINNConfig":
        """Validate that FINN directories exist or can be created."""
        if not self.finn_root.exists():
            raise ValueError(f"FINN_ROOT does not exist: {self.finn_root}")
        
        # Create build dir if it doesn't exist
        if not self.finn_build_dir.exists():
            try:
                self.finn_build_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                warnings.warn(f"Could not create FINN_BUILD_DIR: {e}", UserWarning)
        
        return self

    @classmethod
    def from_environment(cls) -> "FINNConfig":
        """Create config from environment variables with intelligent defaults."""
        # Try to get FINN_ROOT from environment
        finn_root = os.getenv("FINN_ROOT")
        
        if not finn_root:
            # Auto-detect FINN_ROOT from this module's location
            # This file is at src/finn/config.py, so FINN_ROOT is 2 levels up
            module_path = Path(__file__).resolve()
            detected_root = module_path.parent.parent.parent
            
            # Validate it looks like FINN root (has key files/dirs)
            if (detected_root / "src" / "finn").exists() and (detected_root / "setup.py").exists():
                finn_root = str(detected_root)
                # Set the environment variable for legacy code
                os.environ["FINN_ROOT"] = finn_root
            else:
                raise ValueError(
                    "Could not auto-detect FINN_ROOT. Please set FINN_ROOT environment variable."
                )
        
        finn_root_path = Path(finn_root)
        
        # Get or set default for FINN_DEPS_DIR
        finn_deps_dir = os.getenv("FINN_DEPS_DIR", str(finn_root_path / "deps"))
        if "FINN_DEPS_DIR" not in os.environ:
            os.environ["FINN_DEPS_DIR"] = finn_deps_dir
        
        # Get or set default for FINN_BUILD_DIR
        finn_build_dir = os.getenv("FINN_BUILD_DIR", "/tmp/finn_build")
        if "FINN_BUILD_DIR" not in os.environ:
            os.environ["FINN_BUILD_DIR"] = finn_build_dir
        
        # Create Xilinx tools config
        xilinx_config = XilinxToolsConfig(
            vivado_path=Path(p) if (p := os.getenv("XILINX_VIVADO")) else None,
            vitis_path=Path(p) if (p := os.getenv("VITIS_PATH")) else None,
            hls_path=Path(p) if (p := os.getenv("HLS_PATH")) else None,
            xrt_path=Path(p) if (p := os.getenv("XILINX_XRT")) else None,
            platform_repo_paths=Path(p) if (p := os.getenv("PLATFORM_REPO_PATHS")) else None,
        )
        
        # Create runtime config
        runtime_config = RuntimeConfig(
            rtlsim_trace_depth=int(os.getenv("RTLSIM_TRACE_DEPTH", "1")),
            liveness_threshold=int(os.getenv("LIVENESS_THRESHOLD", "1000000")),
            num_default_workers=int(os.getenv("NUM_DEFAULT_WORKERS", "4")),
            imagenet_val_path=Path(p) if (p := os.getenv("IMAGENET_VAL_PATH")) else None,
            ohmyxilinx=Path(p) if (p := os.getenv("OHMYXILINX")) else None,
            vivado_ip_cache=Path(os.getenv("VIVADO_IP_CACHE", "/tmp/vivado_ip_cache")),
        )
        
        return cls(
            finn_root=finn_root_path,
            finn_deps_dir=Path(finn_deps_dir),
            finn_build_dir=Path(finn_build_dir),
            xilinx_tools=xilinx_config,
            runtime=runtime_config,
        )

    def update_environment(self):
        """Update environment variables from config (for backward compatibility)."""
        os.environ["FINN_ROOT"] = str(self.finn_root)
        os.environ["FINN_DEPS_DIR"] = str(self.finn_deps_dir)
        os.environ["FINN_BUILD_DIR"] = str(self.finn_build_dir)
        
        # Only update Xilinx paths if they're set
        if self.xilinx_tools.vivado_path:
            os.environ["XILINX_VIVADO"] = str(self.xilinx_tools.vivado_path)
        if self.xilinx_tools.vitis_path:
            os.environ["VITIS_PATH"] = str(self.xilinx_tools.vitis_path)
        if self.xilinx_tools.hls_path:
            os.environ["HLS_PATH"] = str(self.xilinx_tools.hls_path)
        if self.xilinx_tools.xrt_path:
            os.environ["XILINX_XRT"] = str(self.xilinx_tools.xrt_path)
        if self.xilinx_tools.platform_repo_paths:
            os.environ["PLATFORM_REPO_PATHS"] = str(self.xilinx_tools.platform_repo_paths)
        
        # Update runtime settings
        os.environ["RTLSIM_TRACE_DEPTH"] = str(self.runtime.rtlsim_trace_depth)
        os.environ["LIVENESS_THRESHOLD"] = str(self.runtime.liveness_threshold)
        os.environ["NUM_DEFAULT_WORKERS"] = str(self.runtime.num_default_workers)
        if self.runtime.imagenet_val_path:
            os.environ["IMAGENET_VAL_PATH"] = str(self.runtime.imagenet_val_path)
        if self.runtime.ohmyxilinx:
            os.environ["OHMYXILINX"] = str(self.runtime.ohmyxilinx)
        os.environ["VIVADO_IP_CACHE"] = str(self.runtime.vivado_ip_cache)


# Singleton instance
_config: Optional[FINNConfig] = None


def get_config() -> FINNConfig:
    """Get the global FINN configuration instance."""
    global _config
    if _config is None:
        _config = FINNConfig.from_environment()
        _config.update_environment()
    return _config


def reset_config():
    """Reset the configuration (mainly for testing)."""
    global _config
    _config = None