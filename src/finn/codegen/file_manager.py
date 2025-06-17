"""
File Manager for Unified FINN Code Generation

This module provides centralized file management capabilities for the unified
code generation framework, handling file I/O operations, directory management,
and file organization.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict
import logging


class FileManager:
    """
    Centralized file management for FINN code generation.
    
    Provides utilities for file operations, directory management, and
    file organization that are commonly needed across HLS and RTL generators.
    """
    
    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize the file manager.
        
        Args:
            base_path: Base directory for relative path operations.
                      If None, uses current working directory.
        """
        self.base_path = Path(base_path) if base_path else Path.cwd()
        self.logger = logging.getLogger(__name__)
        
    def ensure_directory(self, directory: Union[str, Path]) -> Path:
        """
        Ensure a directory exists, creating it if necessary.
        
        Args:
            directory: Directory path to ensure exists
            
        Returns:
            Path object for the directory
        """
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
            
        dir_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Ensured directory exists: {dir_path}")
        return dir_path
    
    def write_file(self, file_path: Union[str, Path], content: str, 
                   encoding: str = 'utf-8') -> Path:
        """
        Write content to a file, creating directories as needed.
        
        Args:
            file_path: Path to file to write
            content: Content to write to file
            encoding: File encoding (default: utf-8)
            
        Returns:
            Path object for the written file
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
            
        # Ensure parent directory exists
        self.ensure_directory(file_path.parent)
        
        # Write file content
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
            
        self.logger.info(f"Wrote file: {file_path}")
        return file_path
    
    def read_file(self, file_path: Union[str, Path], 
                  encoding: str = 'utf-8') -> str:
        """
        Read content from a file.
        
        Args:
            file_path: Path to file to read
            encoding: File encoding (default: utf-8)
            
        Returns:
            File content as string
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
            
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.read()
            
        self.logger.debug(f"Read file: {file_path}")
        return content
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> Path:
        """
        Copy a file from source to destination.
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            Path object for the destination file
        """
        src_path = Path(src)
        dst_path = Path(dst)
        
        if not src_path.is_absolute():
            src_path = self.base_path / src_path
        if not dst_path.is_absolute():
            dst_path = self.base_path / dst_path
            
        # Ensure destination directory exists
        self.ensure_directory(dst_path.parent)
        
        # Copy file
        shutil.copy2(src_path, dst_path)
        self.logger.info(f"Copied file: {src_path} -> {dst_path}")
        return dst_path
    
    def copy_files(self, file_mappings: Dict[Union[str, Path], Union[str, Path]]) -> List[Path]:
        """
        Copy multiple files using a mapping dictionary.
        
        Args:
            file_mappings: Dictionary mapping source paths to destination paths
            
        Returns:
            List of destination Path objects
        """
        copied_files = []
        for src, dst in file_mappings.items():
            copied_files.append(self.copy_file(src, dst))
        return copied_files
    
    def list_files(self, directory: Union[str, Path], 
                   pattern: str = "*", recursive: bool = False) -> List[Path]:
        """
        List files in a directory matching a pattern.
        
        Args:
            directory: Directory to search in
            pattern: Glob pattern to match files (default: *)
            recursive: Whether to search recursively (default: False)
            
        Returns:
            List of matching file paths
        """
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
            
        if not dir_path.exists():
            self.logger.warning(f"Directory not found: {dir_path}")
            return []
            
        if recursive:
            files = list(dir_path.rglob(pattern))
        else:
            files = list(dir_path.glob(pattern))
            
        # Filter to only include files (not directories)
        files = [f for f in files if f.is_file()]
        
        self.logger.debug(f"Found {len(files)} files in {dir_path} matching '{pattern}'")
        return files
    
    def get_relative_path(self, file_path: Union[str, Path], 
                         relative_to: Optional[Union[str, Path]] = None) -> Path:
        """
        Get relative path from a reference directory.
        
        Args:
            file_path: File path to make relative
            relative_to: Reference directory (default: base_path)
            
        Returns:
            Relative path
        """
        file_path = Path(file_path)
        if relative_to is None:
            relative_to = self.base_path
        else:
            relative_to = Path(relative_to)
            
        try:
            return file_path.relative_to(relative_to)
        except ValueError:
            # Paths are not relative to each other
            return file_path
    
    def clean_directory(self, directory: Union[str, Path], 
                       keep_patterns: Optional[List[str]] = None) -> None:
        """
        Clean a directory, optionally keeping files matching patterns.
        
        Args:
            directory: Directory to clean
            keep_patterns: List of glob patterns for files to keep
        """
        dir_path = Path(directory)
        if not dir_path.is_absolute():
            dir_path = self.base_path / dir_path
            
        if not dir_path.exists():
            return
            
        keep_patterns = keep_patterns or []
        
        for item in dir_path.iterdir():
            should_keep = False
            
            # Check if item matches any keep pattern
            for pattern in keep_patterns:
                if item.match(pattern):
                    should_keep = True
                    break
                    
            if not should_keep:
                if item.is_dir():
                    shutil.rmtree(item)
                    self.logger.debug(f"Removed directory: {item}")
                else:
                    item.unlink()
                    self.logger.debug(f"Removed file: {item}")
    
    def find_files_by_extension(self, directory: Union[str, Path], 
                               extensions: List[str], recursive: bool = True) -> List[Path]:
        """
        Find files by their extensions.
        
        Args:
            directory: Directory to search in
            extensions: List of file extensions (e.g., ['.cpp', '.hpp'])
            recursive: Whether to search recursively
            
        Returns:
            List of matching file paths
        """
        all_files = []
        for ext in extensions:
            pattern = f"*{ext}" if not ext.startswith('.') else f"*{ext}"
            files = self.list_files(directory, pattern, recursive)
            all_files.extend(files)
            
        return sorted(all_files)
    
    def create_backup(self, file_path: Union[str, Path], 
                     backup_suffix: str = '.bak') -> Optional[Path]:
        """
        Create a backup of a file.
        
        Args:
            file_path: File to backup
            backup_suffix: Suffix for backup file
            
        Returns:
            Path to backup file, or None if source doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.is_absolute():
            file_path = self.base_path / file_path
            
        if not file_path.exists():
            return None
            
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        return self.copy_file(file_path, backup_path)
    
    def get_code_gen_dir(self, operation_name: str, backend: str = 'hls') -> Path:
        """
        Get the code generation directory for an operation.
        
        Args:
            operation_name: Name of the operation
            backend: Backend type ('hls' or 'rtl')
            
        Returns:
            Path to code generation directory
        """
        code_gen_dir = self.base_path / 'code_gen' / backend / operation_name
        return self.ensure_directory(code_gen_dir)
    
    def organize_generated_files(self, files: List[Path], 
                                output_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """
        Organize generated files by type/extension.
        
        Args:
            files: List of generated file paths
            output_dir: Output directory
            
        Returns:
            Dictionary mapping file types to file lists
        """
        output_dir = Path(output_dir)
        organized = {
            'source': [],
            'header': [],
            'template': [],
            'other': []
        }
        
        for file_path in files:
            suffix = file_path.suffix.lower()
            
            if suffix in ['.cpp', '.c', '.cc', '.cxx']:
                organized['source'].append(file_path)
            elif suffix in ['.hpp', '.h', '.hxx']:
                organized['header'].append(file_path)
            elif suffix in ['.v', '.sv', '.vhd', '.vhdl']:
                organized['source'].append(file_path)  # HDL source files
            elif suffix in ['.j2', '.jinja', '.template']:
                organized['template'].append(file_path)
            else:
                organized['other'].append(file_path)
                
        return organized