#!/bin/bash
# FINN Codegen Cleanup Script
# This script organizes development artifacts from src/finn/codegen

echo "🧹 FINN Codegen Cleanup Script"
echo "=============================="
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
    echo "🧪 Moving test files..."
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
    
    for file in "${test_files[@]}"; do
        if [ -f "src/finn/codegen/$file" ]; then
            echo "   Moving $file → tests/finn/codegen/"
            mv "src/finn/codegen/$file" "tests/finn/codegen/"
        fi
    done
    echo "✅ Test files moved"
}

# Function to move development tools
move_tools() {
    echo ""
    echo "🔧 Moving development tools..."
    local tool_files=(
        "codegen_validator.py"
        "debug_test_failure.py"
        "diagnostic_backend_checker.py"
        "inspect_generated_code.py"
        "inspect_template_values.py"
        "generate_actual_code.py"
        "run_validation.py"
        "baseline_validation.py"
    )
    
    for file in "${tool_files[@]}"; do
        if [ -f "src/finn/codegen/$file" ]; then
            echo "   Moving $file → tools/finn/codegen/"
            mv "src/finn/codegen/$file" "tools/finn/codegen/"
        fi
    done
    echo "✅ Development tools moved"
}

# Function to move documentation
move_docs() {
    echo ""
    echo "📚 Moving documentation..."
    local doc_files=(
        "README.md"
        "IMPLEMENTATION_SUMMARY.md"
        "TESTING_GUIDE.md"
        "VALIDATION_RESULTS.md"
        "FINN_Template_Validation_Report.md"
        "CLEANUP_PLAN.md"
    )
    
    for file in "${doc_files[@]}"; do
        if [ -f "src/finn/codegen/$file" ]; then
            echo "   Moving $file → docs/finn/codegen/"
            mv "src/finn/codegen/$file" "docs/finn/codegen/"
        fi
    done
    
    # Create README symlink for discoverability
    if [ -f "docs/finn/codegen/README.md" ] && [ ! -f "src/finn/codegen/README.md" ]; then
        echo "   Creating README.md symlink in src/finn/codegen/"
        ln -s ../../../docs/finn/codegen/README.md src/finn/codegen/README.md
    fi
    echo "✅ Documentation moved"
}

# Function to move scripts
move_scripts() {
    echo ""
    echo "📜 Moving scripts..."
    if [ -f "src/finn/codegen/run_docker_tests.sh" ]; then
        echo "   Moving run_docker_tests.sh → scripts/finn/codegen/"
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

# Function to identify ambiguous files
check_ambiguous() {
    echo ""
    echo "❓ Checking ambiguous files..."
    local ambiguous_files=(
        "backend_instance_manager.py"
        "backend_registration.py"
        "simple_file_manager.py"
        "simple_library_resolver.py"
        "template_validator.py"
    )
    
    local found_ambiguous=false
    for file in "${ambiguous_files[@]}"; do
        if [ -f "src/finn/codegen/$file" ]; then
            echo "   ⚠️  Found: $file (needs manual review)"
            found_ambiguous=true
        fi
    done
    
    if [ "$found_ambiguous" = false ]; then
        echo "   No ambiguous files found"
    fi
}

# Function to show final structure
show_final_structure() {
    echo ""
    echo "📋 Final structure of src/finn/codegen/:"
    echo "----------------------------------------"
    ls -la src/finn/codegen/ | grep -v "^total"
    echo ""
    echo "✅ Core implementation files preserved"
    echo "✅ Development artifacts organized"
}

# Main execution
echo "This script will organize development artifacts in src/finn/codegen"
echo "Production code will remain in place."
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
    check_ambiguous
    show_final_structure
    
    echo ""
    echo "🎉 Cleanup complete!"
    echo ""
    echo "📝 Next steps:"
    echo "   1. Review ambiguous files marked with ⚠️"
    echo "   2. Update any import statements in moved files"
    echo "   3. Update CI/CD scripts if they reference old paths"
    echo "   4. Commit changes with: git add -A && git commit -m 'Organize codegen development artifacts'"
else
    echo "❌ Cleanup cancelled"
fi