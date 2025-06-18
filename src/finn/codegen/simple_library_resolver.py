# Copyright (C) 2024, Advanced Micro Devices, Inc.
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
# * Neither the name of FINN nor the names of its
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

import os
from pathlib import Path
from typing import Optional, List


class SimpleLibraryResolver:
    """
    Simplified library resolver for FINN codegen templates.
    
    Provides simple, predictable library resolution with minimal complexity.
    Focuses on the core template directories without complex search logic.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize library resolver.
        
        Args:
            base_path: Base path for template resolution. If None, uses package path.
        """
        if base_path is None:
            # Default to package template directory
            package_dir = Path(__file__).parent
            self.base_path = package_dir / "templates"
        else:
            self.base_path = Path(base_path)
    
    def resolve_template_path(self, template_name: str) -> str:
        """
        Resolve template name to full path.
        
        Args:
            template_name: Template name (e.g., "hls/docompute.cpp.j2")
            
        Returns:
            Full path to template file
            
        Raises:
            FileNotFoundError: If template not found
        """
        template_path = self.base_path / template_name
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        return str(template_path)
    
    def list_available_templates(self, category: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            category: Template category ("hls", "rtl", etc.). If None, lists all.
            
        Returns:
            List of available template names
        """
        templates = []
        
        if category:
            category_path = self.base_path / category
            if category_path.exists():
                for template_file in category_path.glob("*.j2"):
                    templates.append(f"{category}/{template_file.name}")
        else:
            # List all templates in all categories
            for category_dir in self.base_path.iterdir():
                if category_dir.is_dir():
                    for template_file in category_dir.glob("*.j2"):
                        templates.append(f"{category_dir.name}/{template_file.name}")
        
        return sorted(templates)
    
    def template_exists(self, template_name: str) -> bool:
        """
        Check if template exists.
        
        Args:
            template_name: Template name to check
            
        Returns:
            True if template exists, False otherwise
        """
        template_path = self.base_path / template_name
        return template_path.exists()
    
    def get_template_directory(self) -> str:
        """
        Get base template directory path.
        
        Returns:
            Path to template directory
        """
        return str(self.base_path)