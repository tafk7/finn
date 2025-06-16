############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN TemplateEngine System - Template inheritance and variable substitution
############################################################################

import os
import re
import logging
from typing import Dict, List, Optional, Any, Set, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import time


logger = logging.getLogger(__name__)


class FINNTemplateType(Enum):
    """FINN-specific template file types."""
    IPGEN_CPP = "ipgen_cpp"
    IPGEN_TCL = "ipgen_tcl"
    DOCOMPUTE = "docompute"
    DOCOMPUTE_TIMEOUT = "docompute_timeout"
    CUSTOM = "custom"


@dataclass
class FINNTemplateMetadata:
    """Metadata for a FINN template file."""
    name: str
    template_type: FINNTemplateType
    variables: Set[str] = field(default_factory=set)
    file_path: Optional[str] = None
    last_modified: float = field(default_factory=time.time)


class FINNTemplateLoader:
    """Loads FINN templates from multiple sources with caching."""
    
    def __init__(self, search_paths: List[str], enable_caching: bool = True):
        self.search_paths = search_paths
        self.enable_caching = enable_caching
        self._cache: Dict[str, str] = {}
        self._metadata_cache: Dict[str, FINNTemplateMetadata] = {}
        self._file_mtimes: Dict[str, float] = {}
    
    def find_template(self, template_name: str, template_type: Optional[FINNTemplateType] = None) -> Optional[str]:
        """Find FINN template file in search paths."""
        # Generate possible filenames for FINN templates
        possible_names = self._generate_finn_template_filenames(template_name, template_type)
        
        for search_path in self.search_paths:
            if not os.path.exists(search_path):
                continue
                
            for filename in possible_names:
                full_path = os.path.join(search_path, filename)
                if os.path.isfile(full_path):
                    return full_path
        
        logger.warning(f"FINN template '{template_name}' not found in search paths")
        return None
    
    def _generate_finn_template_filenames(self, template_name: str, template_type: Optional[FINNTemplateType]) -> List[str]:
        """Generate possible FINN template filenames."""
        names = [template_name]
        
        # Add FINN-specific extensions
        if '.' not in template_name:
            if template_type == FINNTemplateType.IPGEN_CPP:
                names.extend([f"{template_name}.cpp.template", f"ipgen_cpp.template"])
            elif template_type == FINNTemplateType.IPGEN_TCL:
                names.extend([f"{template_name}.tcl.template", f"ipgen_tcl.template"])
            elif template_type == FINNTemplateType.DOCOMPUTE:
                names.extend([f"{template_name}.template", f"docompute.template"])
            elif template_type == FINNTemplateType.DOCOMPUTE_TIMEOUT:
                names.extend([f"{template_name}_timeout.template", f"docompute_timeout.template"])
            else:
                # Try common FINN extensions
                names.extend([
                    f"{template_name}.template",
                    f"{template_name}.cpp.template",
                    f"{template_name}.tcl.template"
                ])
        
        return names
    
    def load_template(self, template_name: str, template_type: Optional[FINNTemplateType] = None, force_reload: bool = False) -> Optional[str]:
        """Load FINN template content with caching."""
        cache_key = f"{template_name}:{template_type}"
        
        # Check cache first
        if not force_reload and self.enable_caching and cache_key in self._cache:
            template_path = self._metadata_cache[cache_key].file_path
            if template_path and not self._is_file_modified(template_path):
                return self._cache[cache_key]
        
        # Find and load template
        template_path = self.find_template(template_name, template_type)
        if not template_path:
            return None
        
        try:
            with open(template_path, 'r') as f:
                content = f.read()
            
            # Cache the content and metadata
            if self.enable_caching:
                self._cache[cache_key] = content
                self._metadata_cache[cache_key] = self._extract_metadata(content, template_name, template_type, template_path)
                self._file_mtimes[template_path] = os.path.getmtime(template_path)
            
            logger.debug(f"Loaded FINN template '{template_name}' from {template_path}")
            return content
            
        except Exception as e:
            logger.error(f"Failed to load FINN template '{template_name}': {e}")
            return None
    
    def _is_file_modified(self, file_path: str) -> bool:
        """Check if file has been modified since last load."""
        if file_path not in self._file_mtimes:
            return True
        
        try:
            current_mtime = os.path.getmtime(file_path)
            return current_mtime > self._file_mtimes[file_path]
        except OSError:
            return True
    
    def _extract_metadata(self, content: str, name: str, template_type: Optional[FINNTemplateType], file_path: str) -> FINNTemplateMetadata:
        """Extract metadata from FINN template content."""
        variables = self._extract_finn_variables(content)
        
        # Determine template type if not provided
        if template_type is None:
            if "ipgen" in file_path and "cpp" in file_path:
                template_type = FINNTemplateType.IPGEN_CPP
            elif "ipgen" in file_path and "tcl" in file_path:
                template_type = FINNTemplateType.IPGEN_TCL
            elif "docompute" in file_path and "timeout" in file_path:
                template_type = FINNTemplateType.DOCOMPUTE_TIMEOUT
            elif "docompute" in file_path:
                template_type = FINNTemplateType.DOCOMPUTE
            else:
                template_type = FINNTemplateType.CUSTOM
        
        return FINNTemplateMetadata(
            name=name,
            template_type=template_type,
            variables=variables,
            file_path=file_path,
            last_modified=time.time()
        )
    
    def _extract_finn_variables(self, content: str) -> Set[str]:
        """Extract FINN variable placeholders from template content using $VARIABLE$ pattern."""
        # FINN uses $VARIABLE$ pattern
        pattern = r'\$([A-Z_][A-Z0-9_]*)\$'
        matches = re.findall(pattern, content)
        return set(matches)
    
    def clear_cache(self):
        """Clear template cache."""
        self._cache.clear()
        self._metadata_cache.clear()
        self._file_mtimes.clear()
        logger.debug("FINN template cache cleared")


class FINNVariableResolver:
    """FINN-compatible variable substitution using $VARIABLE$ pattern."""
    
    def __init__(self):
        self._variables: Dict[str, Any] = {}
    
    def set_variable(self, name: str, value: Any):
        """Set a variable for substitution."""
        self._variables[name] = value
        logger.debug(f"Set FINN variable {name}={value}")
    
    def set_variables(self, variables: Dict[str, Any]):
        """Set multiple variables for substitution."""
        self._variables.update(variables)
        logger.debug(f"Set {len(variables)} FINN variables")
    
    def clear_variables(self):
        """Clear all variables."""
        self._variables.clear()
        logger.debug("Cleared all FINN variables")
    
    def resolve_variables(self, content: str) -> str:
        """Resolve FINN variables in content using $VARIABLE$ pattern."""
        # FINN uses $VARIABLE$ pattern exclusively
        pattern = r'\$([A-Z_][A-Z0-9_]*)\$'
        
        def replace_variable(match):
            var_name = match.group(1)
            if var_name in self._variables:
                value = self._variables[var_name]
                # Convert lists to newline-separated strings (FINN pattern)
                if isinstance(value, list):
                    return '\n'.join(str(v) for v in value)
                return str(value)
            else:
                logger.warning(f"Undefined FINN variable: {var_name}")
                return f"UNDEFINED_{var_name}"
        
        resolved_content = re.sub(pattern, replace_variable, content)
        
        # Log substitution statistics
        substitutions = len(re.findall(pattern, content))
        if substitutions > 0:
            logger.debug(f"Resolved {substitutions} FINN variables")
        
        return resolved_content
    
    def get_undefined_variables(self, content: str) -> Set[str]:
        """Get list of undefined variables in FINN template content."""
        pattern = r'\$([A-Z_][A-Z0-9_]*)\$'
        variables_in_content = set(re.findall(pattern, content))
        defined_vars = set(self._variables.keys())
        return variables_in_content - defined_vars
    
    def validate_variables(self, content: str) -> List[str]:
        """Validate that all FINN variables in content can be resolved."""
        issues = []
        undefined_vars = self.get_undefined_variables(content)
        
        if undefined_vars:
            issues.extend([f"Undefined FINN variable: {var}" for var in undefined_vars])
        
        return issues


class FINNTemplateEngine:
    """Main FINN template engine with simplified variable substitution."""
    
    def __init__(self, search_paths: List[str], enable_caching: bool = True):
        self.loader = FINNTemplateLoader(search_paths, enable_caching)
        self.resolver = FINNVariableResolver()
        
        # Initialize with FINN's default template search paths
        self._setup_finn_search_paths()
    
    def _setup_finn_search_paths(self):
        """Setup search paths to include FINN template directories."""
        # Add template subdirectories to search paths
        additional_paths = []
        for base_path in self.loader.search_paths:
            finn_template_path = os.path.join(base_path, "finn")
            base_template_path = os.path.join(base_path, "base")
            if os.path.exists(finn_template_path):
                additional_paths.append(finn_template_path)
            if os.path.exists(base_template_path):
                additional_paths.append(base_template_path)
        
        self.loader.search_paths.extend(additional_paths)
        logger.debug(f"FINN template engine configured with {len(self.loader.search_paths)} search paths")
    
    def render_template(self, template_name: str, variables: Dict[str, Any], 
                       template_type: Optional[FINNTemplateType] = None) -> Optional[str]:
        """Render FINN template with variable substitution."""
        
        # Load template content
        content = self.loader.load_template(template_name, template_type)
        if content is None:
            logger.error(f"Failed to load FINN template '{template_name}'")
            return None
        
        # Set up variables for resolution
        self.resolver.set_variables(variables)
        
        # Resolve variables using FINN's $VARIABLE$ pattern
        resolved_content = self.resolver.resolve_variables(content)
        
        # Validate that all variables were resolved
        validation_issues = self.resolver.validate_variables(resolved_content)
        if validation_issues:
            for issue in validation_issues:
                logger.warning(issue)
        
        logger.info(f"Rendered FINN template '{template_name}' with {len(variables)} variables")
        return resolved_content
    
    def get_template_variables(self, template_name: str, template_type: Optional[FINNTemplateType] = None) -> Set[str]:
        """Get all variables used in a FINN template."""
        content = self.loader.load_template(template_name, template_type)
        if content:
            metadata = self.loader._extract_metadata(content, template_name, template_type, "")
            return metadata.variables
        return set()
    
    def validate_template(self, template_name: str, variables: Dict[str, Any], 
                         template_type: Optional[FINNTemplateType] = None) -> List[str]:
        """Validate FINN template can be rendered with given variables."""
        issues = []
        
        # Try to load template
        content = self.loader.load_template(template_name, template_type)
        if not content:
            issues.append(f"FINN template '{template_name}' not found")
            return issues
        
        # Validate variables
        self.resolver.set_variables(variables)
        variable_issues = self.resolver.validate_variables(content)
        issues.extend(variable_issues)
        
        return issues
    
    def list_available_templates(self) -> Dict[str, List[str]]:
        """List all available FINN templates by search path."""
        templates_by_path = {}
        
        for search_path in self.loader.search_paths:
            if not os.path.exists(search_path):
                continue
                
            templates = []
            for root, _, files in os.walk(search_path):
                for file in files:
                    if file.endswith('.template'):
                        rel_path = os.path.relpath(os.path.join(root, file), search_path)
                        templates.append(rel_path)
            
            templates_by_path[search_path] = sorted(templates)
        
        return templates_by_path
    
    def get_template_metadata(self, template_name: str, template_type: Optional[FINNTemplateType] = None) -> Optional[FINNTemplateMetadata]:
        """Get metadata for a FINN template."""
        cache_key = f"{template_name}:{template_type}"
        return self.loader._metadata_cache.get(cache_key)
    
    def create_template_from_string(self, template_name: str, template_content: str, 
                                   template_type: FINNTemplateType = FINNTemplateType.CUSTOM) -> bool:
        """Create a template from string content (for migrating FINN static templates)."""
        try:
            # Extract variables from content
            variables = self.loader._extract_finn_variables(template_content)
            
            # Create metadata
            metadata = FINNTemplateMetadata(
                name=template_name,
                template_type=template_type,
                variables=variables,
                last_modified=time.time()
            )
            
            # Cache the template
            cache_key = f"{template_name}:{template_type}"
            self.loader._cache[cache_key] = template_content
            self.loader._metadata_cache[cache_key] = metadata
            
            logger.info(f"Created FINN template '{template_name}' from string with {len(variables)} variables")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create FINN template '{template_name}' from string: {e}")
            return False
    
    def clear_caches(self):
        """Clear all caches."""
        self.loader.clear_cache()
        self.resolver.clear_variables()
        logger.debug("FINN template engine caches cleared")


# FINN Template Migration Helper
class FINNTemplateMigrator:
    """Helper class to migrate FINN's static templates to the template engine."""
    
    def __init__(self, template_engine: FINNTemplateEngine):
        self.engine = template_engine
    
    def migrate_static_templates(self, static_templates: Dict[str, str]) -> bool:
        """Migrate FINN's static string templates to the template engine."""
        try:
            template_mappings = {
                "ipgen_template": FINNTemplateType.IPGEN_CPP,
                "ipgentcl_template": FINNTemplateType.IPGEN_TCL,
                "docompute_template": FINNTemplateType.DOCOMPUTE,
                "docompute_template_timeout": FINNTemplateType.DOCOMPUTE_TIMEOUT
            }
            
            migrated_count = 0
            for template_name, content in static_templates.items():
                template_type = template_mappings.get(template_name, FINNTemplateType.CUSTOM)
                
                if self.engine.create_template_from_string(template_name, content, template_type):
                    migrated_count += 1
                else:
                    logger.error(f"Failed to migrate template: {template_name}")
            
            logger.info(f"Migrated {migrated_count}/{len(static_templates)} FINN static templates")
            return migrated_count == len(static_templates)
            
        except Exception as e:
            logger.error(f"Failed to migrate FINN static templates: {e}")
            return False