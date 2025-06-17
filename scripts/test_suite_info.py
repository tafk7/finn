#!/usr/bin/env python3
"""
FINN Unified Codegen - Test Suite Information
Shows comprehensive information about the robustness testing suite.
"""

import sys
import os
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'=' * len(title)}")
    print(title)
    print('=' * len(title))

def print_section(title):
    """Print a formatted section header"""
    print(f"\n{title}")
    print('-' * len(title))

def check_file_exists(filepath):
    """Check if a file exists and return status"""
    if Path(filepath).exists():
        return "✅ Present"
    else:
        return "❌ Missing"

def get_file_info(filepath):
    """Get basic file information"""
    path = Path(filepath)
    if path.exists():
        stat = path.stat()
        lines = 0
        try:
            with open(path, 'r') as f:
                lines = sum(1 for line in f)
        except:
            lines = "Unknown"
        
        executable = "✅" if os.access(path, os.X_OK) else "❌"
        return f"✅ {lines:>4} lines | Executable: {executable}"
    else:
        return "❌ Missing"

def analyze_test_scripts():
    """Analyze all test scripts"""
    
    print_header("🧪 FINN Unified Codegen Robustness Testing Suite")
    
    print(f"📁 Base Directory: {Path.cwd()}")
    print(f"🐍 Python Version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check if we're in the right directory
    if not Path('src/finn/codegen').exists():
        print("❌ ERROR: Not in FINN project root directory")
        print("   Please run from the directory containing src/finn/codegen/")
        return False
    
    print("✅ Running from correct FINN project directory")
    
    print_section("📋 Test Suite Structure")
    
    # Main orchestration script
    print("\n🎯 Main Orchestration:")
    main_script = 'scripts/test_codegen_robustness.sh'
    print(f"  {main_script:<40} {get_file_info(main_script)}")
    
    # Individual test scripts
    print("\n🔬 Individual Test Scripts:")
    test_scripts = [
        ('scripts/test_core_framework.py', 'Core Framework Validation'),
        ('scripts/discover_finn_tests.py', 'FINN Integration Discovery'),
        ('scripts/test_components.py', 'Component Testing'),
        ('scripts/test_performance.py', 'Performance & Stress Testing'),
        ('scripts/test_integration.py', 'Integration Testing')
    ]
    
    for script_path, description in test_scripts:
        info = get_file_info(script_path)
        print(f"  {script_path:<40} {info}")
        print(f"    → {description}")
    
    # Utility scripts
    print("\n🛠️  Utility Scripts:")
    util_scripts = [
        ('scripts/run_tests.py', 'Test Runner Wrapper'),
        ('scripts/test_suite_info.py', 'Test Suite Information')
    ]
    
    for script_path, description in util_scripts:
        info = get_file_info(script_path)
        print(f"  {script_path:<40} {info}")
        print(f"    → {description}")
    
    # Documentation
    print("\n📚 Documentation:")
    doc_files = [
        ('scripts/README.md', 'Complete Test Suite Documentation')
    ]
    
    for doc_path, description in doc_files:
        info = get_file_info(doc_path)
        print(f"  {doc_path:<40} {info}")
        print(f"    → {description}")
    
    print_section("🎯 Test Phase Overview")
    
    phases = [
        {
            'name': 'Phase 1: Core Framework Validation',
            'script': 'test_core_framework.py',
            'purpose': 'Tests fundamental component imports and basic functionality',
            'criteria': 'All core components must be importable and functional',
            'tests': [
                'Core component imports (ModernHLSGenerator, ModernRTLGenerator, etc.)',
                'Basic functionality with complete test operations',
                'Error handling and edge cases',
                'Template engine validation',
                'File manager operations'
            ]
        },
        {
            'name': 'Phase 2: FINN Integration Discovery',
            'script': 'discover_finn_tests.py',
            'purpose': 'Discovers available FINN components and test files',
            'criteria': 'At least 3 out of 5 discovery checks should pass',
            'tests': [
                'Test file discovery in FINN project',
                'Pytest availability and functionality',
                'Real FINN operation imports',
                'FINN utility module discovery',
                'Integration report generation'
            ]
        },
        {
            'name': 'Phase 3: Component Testing',
            'script': 'test_components.py',
            'purpose': 'Validates individual components with comprehensive testing',
            'criteria': 'At least 80% of component tests should pass',
            'tests': [
                'HLS code generation with complete test operations',
                'RTL code generation with complete test operations',
                'File operations and management',
                'Template system functionality',
                'Library resolution system',
                'Real FINN operation integration (when available)'
            ]
        },
        {
            'name': 'Phase 4: Performance & Stress Testing',
            'script': 'test_performance.py',
            'purpose': 'Tests performance characteristics and stress scenarios',
            'criteria': 'At least 80% of performance tests should pass',
            'tests': [
                'Code generation performance with scaling complexity',
                'Memory usage patterns and cleanup',
                'Concurrent operations and thread safety',
                'Template rendering performance',
                'File I/O performance testing'
            ]
        },
        {
            'name': 'Phase 5: Integration Testing',
            'script': 'test_integration.py',
            'purpose': 'Tests end-to-end workflows and integration scenarios',
            'criteria': 'At least 90% of integration tests should pass',
            'tests': [
                'Complete end-to-end HLS workflow',
                'Complete end-to-end RTL workflow',
                'Cross-format compatibility validation',
                'Error recovery scenarios',
                'Example integration testing'
            ]
        }
    ]
    
    for i, phase in enumerate(phases, 1):
        print(f"\n🎯 {phase['name']}")
        print(f"   Script: {phase['script']}")
        print(f"   Purpose: {phase['purpose']}")
        print(f"   Success Criteria: {phase['criteria']}")
        print(f"   Tests:")
        for test in phase['tests']:
            print(f"     • {test}")
    
    print_section("🚀 Quick Start Commands")
    
    print("Complete Test Suite:")
    print("  ./scripts/test_codegen_robustness.sh")
    print("\nIndividual Phases:")
    print("  python scripts/run_tests.py --core          # Phase 1")
    print("  python scripts/run_tests.py --discover      # Phase 2")
    print("  python scripts/run_tests.py --components    # Phase 3")
    print("  python scripts/run_tests.py --performance   # Phase 4")
    print("  python scripts/run_tests.py --integration   # Phase 5")
    print("\nUtility Commands:")
    print("  python scripts/run_tests.py --list          # List all phases")
    print("  python scripts/run_tests.py --check-env     # Check environment")
    print("  python scripts/test_suite_info.py           # This information")
    
    print_section("📊 Expected Results")
    
    print("✅ Fully Successful Run:")
    print("   • All 5 phases pass")
    print("   • Overall success rate: 100%")
    print("   • Framework ready for production use")
    
    print("\n✅ Acceptable Results:")
    print("   • Core Framework: 100% pass (critical)")
    print("   • FINN Integration: 60%+ pass (environment dependent)")
    print("   • Component Testing: 80%+ pass (functional validation)")
    print("   • Performance Testing: 80%+ pass (efficiency validation)")
    print("   • Integration Testing: 90%+ pass (workflow validation)")
    
    print_section("🔧 Troubleshooting")
    
    print("Common Issues:")
    print("  • Import Errors: Ensure running from FINN project root")
    print("  • Permission Errors: Make scripts executable with 'chmod +x'")
    print("  • Missing Dependencies: Install psutil, jinja2")
    print("  • Timeout Errors: May occur in resource-constrained environments")
    
    print("\nFor detailed documentation, see: scripts/README.md")
    
    print_section("📈 Test Suite Statistics")
    
    # Count total lines of code
    total_lines = 0
    total_scripts = 0
    executable_scripts = 0
    
    all_scripts = [
        'scripts/test_codegen_robustness.sh',
        'scripts/test_core_framework.py',
        'scripts/discover_finn_tests.py',
        'scripts/test_components.py',
        'scripts/test_performance.py',
        'scripts/test_integration.py',
        'scripts/run_tests.py',
        'scripts/test_suite_info.py'
    ]
    
    for script in all_scripts:
        path = Path(script)
        if path.exists():
            total_scripts += 1
            if os.access(path, os.X_OK):
                executable_scripts += 1
            
            try:
                with open(path, 'r') as f:
                    lines = sum(1 for line in f)
                    total_lines += lines
            except:
                pass
    
    print(f"📊 Total Scripts: {total_scripts}")
    print(f"📊 Executable Scripts: {executable_scripts}")
    print(f"📊 Total Lines of Test Code: {total_lines:,}")
    print(f"📊 Documentation Files: 1 (README.md)")
    
    # Check for missing critical files
    missing_files = []
    for script in all_scripts:
        if not Path(script).exists():
            missing_files.append(script)
    
    if missing_files:
        print(f"\n❌ Missing Critical Files:")
        for missing in missing_files:
            print(f"   • {missing}")
        return False
    else:
        print(f"\n✅ All test suite components are present and ready")
        return True

def main():
    """Main function"""
    try:
        success = analyze_test_scripts()
        
        print_header("🎉 Test Suite Analysis Complete")
        
        if success:
            print("✅ The FINN Unified Codegen robustness testing suite is fully set up")
            print("✅ All components are present and ready for execution")
            print("\n🚀 Ready to validate framework robustness!")
            return 0
        else:
            print("❌ Test suite setup is incomplete")
            print("❌ Please resolve missing components before running tests")
            return 1
            
    except Exception as e:
        print(f"❌ Error analyzing test suite: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())