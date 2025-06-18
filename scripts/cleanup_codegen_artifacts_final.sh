#!/bin/bash
# FINN Codegen Final Cleanup Script
# Based on complete analysis of all files including ambiguous ones

echo "🧹 FINN Codegen Final Cleanup Script"
echo "===================================="
echo ""

# Check if we're in the right directory
if [ ! -d "src/finn/codegen" ]; then
    echo "❌ Error: src/finn/codegen directory not found!"
    echo "   Please run this script from the FINN root directory."
    exit 1
fi

# Function to create directories if they don't exist
create_dirs() {
    echo "📁 Creating directory structure..."
    mkdir -p tests/finn/codegen
    mkdir -p tools/finn/codegen
    mkdir -p docs/finn/codegen
    mkdir -p scripts/finn/codegen
    echo "✅ Directories created"
}

# Function to move test files
move_tests() {
    echo ""
    echo "🧪 Moving test files to tests/finn/codegen/..."
    local test_files=(
        "test_architecture_validation.py"
        "test_node_factory.py"
        "test_real_finn_nodes.py"
        "test_suite.py"
        "test_thresholding.py"
        "test_fixed_backends.py"
        "test_real_backends_docker.py"
        "test_real_finn_backends.py"
        "test_real_validation.py"
    )
    
    local moved_count=0
    for file in "${test_files[@]}"; do
        if [ -f "src/finn/codegen/$file" ]; then
            echo "   Moving $file"
            mv "src/finn/codegen/$file" "tests/finn/codegen/"
            ((moved_count++))
        fi
    done
    echo "✅ Moved $moved_count test files"
}

# Function to move development tools
move_tools() {
    echo ""
    echo "🔧 Moving development tools to tools/finn/codegen/..."
    local tool_files=(
        "codegen_validator.py"
        "debug_test_failure.py"
        "diagnostic_backend_checker.py"
        "inspect_generated_code.py"
        "inspect_template_values.py"
        "generate_actual_code.py"
        "run_validation.py"
        "baseline_validation.py"
        "backend_instance_manager.py"    # Resolved as tool
        "template_validator.py"           # Resolved as tool
    )
    
    local moved_count=0
    for file in "${tool_files[@]}"; do
        if [ -f "src/finn/codegen/$file" ]; then
            echo "   Moving $file"
            mv "src/finn/codegen/$file" "tools/finn/codegen/"
            ((moved_count++))
        fi
    done
    echo "✅ Moved $moved_count development tools"
}

# Function to move documentation
move_docs() {
    echo ""
    echo "📚 Moving documentation to docs/finn/codegen/..."
    local doc_files=(
        "README.md"
        "IMPLEMENTATION_SUMMARY.md"
        "TESTING_GUIDE.md"
        "VALIDATION_RESULTS.md"
        "FINN_Template_Validation_Report.md"
        "CLEANUP_PLAN.md"
        "CLEANUP_ANALYSIS.md"
        "FINAL_CLEANUP_RECOMMENDATIONS.md"
    )
    
    local moved_count=0
    for file in "${doc_files[@]}"; do
        if [ -f "src/finn/codegen/$file" ]; then
            echo "   Moving $file"
            mv "src/finn/codegen/$file" "docs/finn/codegen/"
            ((moved_count++))
        fi
    done
    
    # Create README symlink for discoverability
    if [ -f "docs/finn/codegen/README.md" ] && [ ! -e "src/finn/codegen/README.md" ]; then
        echo "   Creating README.md symlink in src/finn/codegen/"
        ln -s ../../../docs/finn/codegen/README.md src/finn/codegen/README.md
    fi
    echo "✅ Moved $moved_count documentation files"
}

# Function to move scripts
move_scripts() {
    echo ""
    echo "📜 Moving scripts to scripts/finn/codegen/..."
    if [ -f "src/finn/codegen/run_docker_tests.sh" ]; then
        echo "   Moving run_docker_tests.sh"
        mv "src/finn/codegen/run_docker_tests.sh" "scripts/finn/codegen/"
    fi
    echo "✅ Scripts moved"
}

# Function to clean generated artifacts
clean_artifacts() {
    echo ""
    echo "🗑️  Cleaning generated artifacts..."
    
    # Remove generated C++ files
    if [ -f "src/finn/codegen/generated_clean_backend.cpp" ]; then
        echo "   Removing generated_clean_backend.cpp"
        rm "src/finn/codegen/generated_clean_backend.cpp"
    fi
    
    if [ -f "src/finn/codegen/generated_legacy_backend.cpp" ]; then
        echo "   Removing generated_legacy_backend.cpp"
        rm "src/finn/codegen/generated_legacy_backend.cpp"
    fi
    
    # Remove test output directory
    if [ -d "src/finn/codegen/test_output" ]; then
        echo "   Removing test_output directory"
        rm -rf "src/finn/codegen/test_output"
    fi
    
    echo "✅ Generated artifacts cleaned"
}

# Function to show what remains (core implementation)
show_core_files() {
    echo ""
    echo "📋 Core implementation files remaining in src/finn/codegen/:"
    echo "============================================================"
    local core_files=(
        "__init__.py"
        "backend_registration.py"      # Production registration
        "backend_registry.py"         # Registry system
        "CG_backend_registration.py"  # Clean backend registration
        "codegen.py"                 # Base infrastructure
        "config.py"                  # Configuration
        "file_manager.py"            # File operations
        "library_resolver.py"        # Library resolution
        "simple_file_manager.py"     # Simplified fallback
        "simple_library_resolver.py" # Simplified fallback
        "template_engine.py"         # Jinja2 engine
    )
    
    for file in "${core_files[@]}"; do
        if [ -f "src/finn/codegen/$file" ]; then
            echo "   ✅ $file"
        else
            echo "   ❓ $file (not found)"
        fi
    done
    
    if [ -d "src/finn/codegen/templates" ]; then
        echo "   ✅ templates/ (directory)"
    fi
    
    if [ -L "src/finn/codegen/README.md" ]; then
        echo "   ✅ README.md (symlink)"
    fi
}

# Function to create summary report
create_summary() {
    echo ""
    echo "📊 Cleanup Summary Report"
    echo "========================"
    
    # Count files in each location
    local tests_count=$(ls -1 tests/finn/codegen/*.py 2>/dev/null | wc -l)
    local tools_count=$(ls -1 tools/finn/codegen/*.py 2>/dev/null | wc -l)
    local docs_count=$(ls -1 docs/finn/codegen/*.md 2>/dev/null | wc -l)
    local core_count=$(ls -1 src/finn/codegen/*.py 2>/dev/null | wc -l)
    
    echo "   Test files moved: $tests_count"
    echo "   Tool files moved: $tools_count"
    echo "   Doc files moved: $docs_count"
    echo "   Core files remaining: $core_count"
    echo ""
    echo "✅ Cleanup complete!"
}

# Main execution
echo "This script will organize development artifacts in src/finn/codegen"
echo "based on the complete file analysis."
echo ""
echo "Actions to be performed:"
echo "  - Move 9 test files to tests/finn/codegen/"
echo "  - Move 10 development tools to tools/finn/codegen/"
echo "  - Move documentation to docs/finn/codegen/"
echo "  - Move scripts to scripts/finn/codegen/"
echo "  - Delete generated artifacts"
echo "  - Keep 12 core implementation files in place"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    create_dirs
    move_tests
    move_tools
    move_docs
    move_scripts
    clean_artifacts
    show_core_files
    create_summary
    
    echo ""
    echo "📝 Next steps:"
    echo "   1. Update any import statements in moved files"
    echo "   2. Update CI/CD scripts if they reference old paths"
    echo "   3. Review the final structure"
    echo "   4. Commit changes with: git add -A && git commit -m 'Organize codegen development artifacts'"
else
    echo "❌ Cleanup cancelled"
fi