"""
FINN Template Validation Framework
Validates template syntax, structure, and output quality for clean codegen architecture.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from jinja2 import Environment, FileSystemLoader, Template, TemplateSyntaxError
from jinja2.exceptions import TemplateError


class TemplateValidator:
    """Validates FINN codegen templates for syntax, structure, and output quality."""
    
    def __init__(self, template_root: str = "templates"):
        """Initialize validator with template directory."""
        self.template_root = Path(template_root)
        self.env = Environment(loader=FileSystemLoader(str(self.template_root)))
        self.validation_results = {}
        
        # Required components for different template types
        self.required_components = {
            'hls': {
                'base': ['operation_name', 'node_name', 'includes', 'stream_declarations'],
                'operation': ['operation_specific_values', 'template_values']
            },
            'rtl': {
                'base': ['module_name', 'io_signals', 'clk_rst_process'],
                'operation': ['operation_specific_values', 'template_values']
            }
        }
        
        # Common validation patterns
        self.validation_patterns = {
            'hls_includes': r'#include\s+[<"][^>"]+[>"]',
            'hls_pragmas': r'#pragma\s+HLS\s+\w+',
            'stream_declarations': r'hls::stream<[^>]+>\s+\w+',
            'rtl_module': r'module\s+\w+\s*\(',
            'rtl_signals': r'(input|output|reg|wire)\s+',
            'rtl_always': r'always\s*@\s*\(',
        }

    def validate_all_templates(self) -> Dict[str, Dict]:
        """Validate all templates in the template directory."""
        results = {
            'base_templates': {},
            'component_templates': {},
            'operation_templates': {},
            'overall_status': 'pending'
        }
        
        # Validate base templates
        base_templates = list(self.template_root.glob("base/*.j2"))
        print(f"Found {len(base_templates)} base templates: {[t.name for t in base_templates]}")
        for template_path in base_templates:
            template_name = template_path.name
            results['base_templates'][template_name] = self._validate_template(template_path)
        
        # Validate component templates
        component_templates = list(self.template_root.glob("components/**/*.j2"))
        print(f"Found {len(component_templates)} component templates: {[str(t.relative_to(self.template_root)) for t in component_templates]}")
        for template_path in component_templates:
            rel_path = str(template_path.relative_to(self.template_root))
            results['component_templates'][rel_path] = self._validate_template(template_path)
        
        # Determine overall status
        all_results = []
        all_results.extend(results['base_templates'].values())
        all_results.extend(results['component_templates'].values())
        
        if all(r['valid'] for r in all_results):
            results['overall_status'] = 'valid'
        elif any(r['errors'] for r in all_results):
            results['overall_status'] = 'errors'
        else:
            results['overall_status'] = 'warnings'
        
        self.validation_results = results
        return results

    def _validate_template(self, template_path: Path) -> Dict:
        """Validate a single template file."""
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'syntax_valid': False,
            'structure_valid': False,
            'components_found': [],
            'missing_components': []
        }
        
        try:
            # Read template content
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()
            
            # Validate syntax
            syntax_result = self._validate_syntax(template_content, template_path.name)
            result.update(syntax_result)
            
            if result['syntax_valid']:
                # Validate structure
                structure_result = self._validate_structure(template_content, template_path)
                result.update(structure_result)
                
                # Validate components
                components_result = self._validate_components(template_content, template_path)
                result.update(components_result)
            
            # Overall validity
            result['valid'] = (result['syntax_valid'] and 
                             result['structure_valid'] and 
                             len(result['errors']) == 0)
            
        except Exception as e:
            result['errors'].append(f"Validation failed: {str(e)}")
        
        return result

    def _validate_syntax(self, content: str, template_name: str) -> Dict:
        """Validate Jinja2 template syntax."""
        result = {'syntax_valid': False, 'syntax_errors': []}
        
        try:
            template = self.env.from_string(content)
            result['syntax_valid'] = True
        except TemplateSyntaxError as e:
            result['syntax_errors'].append(f"Syntax error at line {e.lineno}: {e.message}")
        except TemplateError as e:
            result['syntax_errors'].append(f"Template error: {str(e)}")
        
        return result

    def _validate_structure(self, content: str, template_path: Path) -> Dict:
        """Validate template structure and organization."""
        result = {'structure_valid': True, 'structure_warnings': []}
        
        # Check for proper header comments
        if not re.search(r'{#.*?#}', content[:200]):
            result['structure_warnings'].append("Missing header comment")
        
        # Check for proper macro definitions in component templates
        if 'components' in str(template_path):
            macros = re.findall(r'{% macro\s+(\w+)', content)
            if not macros:
                result['structure_warnings'].append("Component template should define macros")
            else:
                result['macros_defined'] = macros
        
        # Check for proper extends/includes in base templates
        if 'base' in str(template_path):
            if 'from' not in content and 'import' not in content:
                result['structure_warnings'].append("Base template should import components")
        
        # Check for consistent indentation
        lines = content.split('\n')
        inconsistent_indent = False
        for i, line in enumerate(lines, 1):
            if line.strip() and line.startswith(' ') and not line.startswith('    '):
                if len(line) - len(line.lstrip()) % 4 != 0:
                    inconsistent_indent = True
                    break
        
        if inconsistent_indent:
            result['structure_warnings'].append("Inconsistent indentation (use 4 spaces)")
        
        return result

    def _validate_components(self, content: str, template_path: Path) -> Dict:
        """Validate required components are present."""
        result = {
            'components_found': [],
            'missing_components': [],
            'component_warnings': []
        }
        
        # Determine template type
        template_type = None
        if 'hls' in str(template_path):
            template_type = 'hls'
        elif 'rtl' in str(template_path):
            template_type = 'rtl'
        
        if not template_type:
            return result
        
        # Check for required patterns based on template type
        patterns = self.validation_patterns
        
        if template_type == 'hls':
            # Check HLS-specific patterns
            if re.search(patterns['hls_includes'], content):
                result['components_found'].append('hls_includes')
            
            if re.search(patterns['hls_pragmas'], content):
                result['components_found'].append('hls_pragmas')
            
            if re.search(patterns['stream_declarations'], content):
                result['components_found'].append('stream_declarations')
        
        elif template_type == 'rtl':
            # Check RTL-specific patterns
            if re.search(patterns['rtl_module'], content):
                result['components_found'].append('rtl_module')
            
            if re.search(patterns['rtl_signals'], content):
                result['components_found'].append('rtl_signals')
            
            if re.search(patterns['rtl_always'], content):
                result['components_found'].append('rtl_always')
        
        # Check for template variables
        template_vars = re.findall(r'{{\s*(\w+)', content)
        if template_vars:
            result['template_variables'] = list(set(template_vars))
        
        return result

    def generate_validation_report(self) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available. Run validate_all_templates() first."
        
        report = []
        report.append("# FINN Template Validation Report")
        report.append(f"**Overall Status**: {self.validation_results['overall_status'].upper()}")
        report.append("")
        
        # Base templates section
        report.append("## Base Templates")
        for template_name, result in self.validation_results['base_templates'].items():
            status = "✅" if result['valid'] else "❌"
            report.append(f"- {status} `{template_name}`")
            
            if result['errors']:
                for error in result['errors']:
                    report.append(f"  - **Error**: {error}")
            
            if result['warnings']:
                for warning in result['warnings']:
                    report.append(f"  - **Warning**: {warning}")
        
        report.append("")
        
        # Component templates section
        report.append("## Component Templates")
        for template_path, result in self.validation_results['component_templates'].items():
            status = "✅" if result['valid'] else "❌"
            report.append(f"- {status} `{template_path}`")
            
            if 'macros_defined' in result:
                report.append(f"  - **Macros**: {', '.join(result['macros_defined'])}")
            
            if result['components_found']:
                report.append(f"  - **Components**: {', '.join(result['components_found'])}")
            
            if result['errors']:
                for error in result['errors']:
                    report.append(f"  - **Error**: {error}")
        
        report.append("")
        
        # Summary statistics
        total_templates = (len(self.validation_results['base_templates']) + 
                          len(self.validation_results['component_templates']))
        valid_templates = sum(1 for results in [
            *self.validation_results['base_templates'].values(),
            *self.validation_results['component_templates'].values()
        ] if results['valid'])
        
        report.append("## Summary")
        report.append(f"- **Total Templates**: {total_templates}")
        report.append(f"- **Valid Templates**: {valid_templates}")
        if total_templates > 0:
            report.append(f"- **Success Rate**: {valid_templates/total_templates*100:.1f}%")
        else:
            report.append("- **Success Rate**: No templates found")
        
        return "\n".join(report)

    def test_template_rendering(self, template_path: str, test_values: Dict) -> Dict:
        """Test template rendering with sample values."""
        result = {
            'renders_successfully': False,
            'output_length': 0,
            'render_errors': [],
            'output_sample': ""
        }
        
        try:
            template = self.env.get_template(template_path)
            rendered = template.render(**test_values)
            
            result['renders_successfully'] = True
            result['output_length'] = len(rendered)
            result['output_sample'] = rendered[:200] + "..." if len(rendered) > 200 else rendered
            
        except Exception as e:
            result['render_errors'].append(str(e))
        
        return result


def run_template_validation():
    """Run complete template validation and generate report."""
    validator = TemplateValidator()
    results = validator.validate_all_templates()
    report = validator.generate_validation_report()
    
    # Save report
    report_path = "FINN_Template_Validation_Report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Template validation complete. Report saved to {report_path}")
    print(f"Overall status: {results['overall_status']}")
    
    return results


if __name__ == "__main__":
    run_template_validation()