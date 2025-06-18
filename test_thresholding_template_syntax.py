#!/usr/bin/env python3
"""
Template syntax validation test for Thresholding Jinja2 templates.
"""

import os
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError


def test_template_syntax():
    """Test that all Thresholding templates have valid Jinja2 syntax."""
    
    # Get absolute path to templates directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(current_dir, 'src', 'finn', 'codegen', 'templates')
    
    if not os.path.exists(templates_dir):
        raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
    
    # Create Jinja2 environment
    env = Environment(loader=FileSystemLoader(templates_dir))
    
    # Test all Thresholding templates
    templates = [
        'thresholding/hls/docompute.cpp.j2',
        'thresholding/hls/docompute_timeout.cpp.j2', 
        'thresholding/hls/ipgen.cpp.j2',
        'thresholding/hls/ipgen.tcl.j2',
        'thresholding/rtl/wrapper.v.j2'
    ]
    
    print("Testing Thresholding template syntax validation...")
    
    for template_name in templates:
        print(f"  Testing {template_name}...")
        
        try:
            # Try to load the template
            template = env.get_template(template_name)
            
            # Try to compile it (this will catch syntax errors)
            template.render()  # Empty render to test compilation
            
            print(f"    ✓ {template_name} - Valid syntax")
            
        except TemplateSyntaxError as e:
            print(f"    ✗ {template_name} - Syntax error: {e}")
            raise
            
        except Exception as e:
            # Template might fail to render due to missing variables, but syntax should be OK
            if "undefined" in str(e).lower():
                print(f"    ✓ {template_name} - Valid syntax (variables undefined as expected)")
            else:
                print(f"    ✗ {template_name} - Unexpected error: {e}")
                raise
    
    print("✓ All templates have valid Jinja2 syntax!")
    return True


if __name__ == "__main__":
    test_template_syntax()
    print("Template syntax validation completed successfully.")