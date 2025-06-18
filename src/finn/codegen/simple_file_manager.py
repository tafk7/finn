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
import shutil
from pathlib import Path
from typing import Optional, List


class SimpleFileManager:
    """
    Simplified file manager for FINN codegen operations.
    
    Provides basic file management operations without complex dependency tracking
    or sophisticated caching. Focuses on reliability and simplicity.
    """
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize file manager.
        
        Args:
            base_dir: Base directory for file operations. If None, uses cwd.
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
    
    def write_file(self, filepath: str, content: str, encoding: str = 'utf-8') -> None:
        """
        Write content to file, creating directories as needed.
        
        Args:
            filepath: Path to file (relative to base_dir or absolute)
            content: Content to write
            encoding: File encoding
        """
        full_path = self._resolve_path(filepath)
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        with open(full_path, 'w', encoding=encoding) as f:
            f.write(content)
    
    def read_file(self, filepath: str, encoding: str = 'utf-8') -> str:
        """
        Read content from file.
        
        Args:
            filepath: Path to file
            encoding: File encoding
            
        Returns:
            File content
            
        Raises:
            FileNotFoundError: If file does not exist
        """
        full_path = self._resolve_path(filepath)
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        with open(full_path, 'r', encoding=encoding) as f:
            return f.read()
    
    def file_exists(self, filepath: str) -> bool:
        """
        Check if file exists.
        
        Args:
            filepath: Path to file
            
        Returns:
            True if file exists, False otherwise
        """
        full_path = self._resolve_path(filepath)
        return full_path.exists()
    
    def create_directory(self, dirpath: str) -> None:
        """
        Create directory and any necessary parent directories.
        
        Args:
            dirpath: Path to directory
        """
        full_path = self._resolve_path(dirpath)
        full_path.mkdir(parents=True, exist_ok=True)
    
    def copy_file(self, src: str, dst: str) -> None:
        """
        Copy file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
        """
        src_path = self._resolve_path(src)
        dst_path = self._resolve_path(dst)
        
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copy2(src_path, dst_path)
    
    def delete_file(self, filepath: str) -> None:
        """
        Delete file if it exists.
        
        Args:
            filepath: Path to file
        """
        full_path = self._resolve_path(filepath)
        if full_path.exists():
            full_path.unlink()
    
    def list_files(self, dirpath: str, pattern: str = "*") -> List[str]:
        """
        List files in directory matching pattern.
        
        Args:
            dirpath: Directory path
            pattern: File pattern (glob style)
            
        Returns:
            List of file paths relative to base_dir
        """
        full_path = self._resolve_path(dirpath)
        
        if not full_path.exists() or not full_path.is_dir():
            return []
        
        files = []
        for file_path in full_path.glob(pattern):
            if file_path.is_file():
                try:
                    rel_path = file_path.relative_to(self.base_dir)
                    files.append(str(rel_path))
                except ValueError:
                    # File is outside base_dir, use absolute path
                    files.append(str(file_path))
        
        return sorted(files)
    
    def get_absolute_path(self, filepath: str) -> str:
        """
        Get absolute path for given filepath.
        
        Args:
            filepath: File path
            
        Returns:
            Absolute path
        """
        return str(self._resolve_path(filepath).resolve())
    
    def _resolve_path(self, filepath: str) -> Path:
        """
        Resolve filepath relative to base_dir.
        
        Args:
            filepath: File path
            
        Returns:
            Resolved Path object
        """
        path = Path(filepath)
        if path.is_absolute():
            return path
        else:
            return self.base_dir / path