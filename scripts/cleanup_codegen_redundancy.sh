#!/bin/bash
# FINN Codegen Redundancy Cleanup Script
# Removes redundant code and consolidates implementations

echo "🧹 FINN Codegen Redundancy Cleanup"
echo "=================================="
echo ""

# Check if we're in the right directory
if [ ! -d "src/finn/codegen" ]; then
    echo "❌ Error: src/finn/codegen directory not found!"
    echo "   Please run this script from the FINN root directory."
    exit 1
fi

# Function to backup files before removal
backup_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo "   Backing up $(basename $file) to ${file}.bak"
        cp "$file" "${file}.bak"
    fi
}

# Function to merge backend registration files
merge_backend_registration() {
    echo ""
    echo "📋 Merging backend registration files..."
    
    # Create merged version
    cat > src/finn/codegen/backend_registration_merged.py << 'EOF'
"""
Backend Registration for FINN Operations

This module explicitly registers all available backends with the registry.
Merged from backend_registration.py and CG_backend_registration.py
"""

import logging
from .backend_registry import BackendRegistry


def register_all_backends() -> BackendRegistry:
    """
    Register all available backends explicitly.
    
    Returns:
        Configured BackendRegistry with all backends registered
    """
    logger = logging.getLogger(__name__)
    registry = BackendRegistry()
    
    # Register HLS backends
    logger.debug("Registering HLS backends...")
    
    # Legacy backends
    _register_legacy_hls_backends(registry, logger)
    
    # Clean backends (from CG_backend_registration.py)
    _register_clean_hls_backends(registry, logger)
    
    # Register RTL backends
    logger.debug("Registering RTL backends...")
    
    # Legacy backends
    _register_legacy_rtl_backends(registry, logger)
    
    # Clean backends
    _register_clean_rtl_backends(registry, logger)
    
    stats = registry.get_registry_stats()
    logger.info(f"Backend registration complete: {stats['hls_backends']} HLS, {stats['rtl_backends']} RTL backends")
    
    return registry


def _register_legacy_hls_backends(registry, logger):
    """Register legacy HLS backends."""
    backends = [
        ('Thresholding', 'thresholding_hls', 'Thresholding_hls'),
        ('MatrixVectorActivation', 'matrixvectoractivation_hls', 'MVAU_hls'),
        ('MVAU', 'matrixvectoractivation_hls', 'MVAU_hls'),
        ('AddStreams', 'addstreams_hls', 'AddStreamsHLS'),
        ('Concat', 'concat_hls', 'ConcatHLS'),
        ('ConvolutionInputGenerator', 'convolutioninputgenerator_hls', 'ConvolutionInputGeneratorHLS'),
        ('DuplicateStreams', 'duplicatestreams_hls', 'DuplicateStreamsHLS'),
        ('ElementwiseBinary', 'elementwise_binary_hls', 'ElementwiseBinaryHLS'),
        ('FMPadding', 'fmpadding_hls', 'FMPaddingHLS'),
        ('GlobalAccPool', 'globalaccpool_hls', 'GlobalAccPoolHLS'),
        ('LabelSelect', 'labelselect_hls', 'LabelSelectHLS'),
        ('Lookup', 'lookup_hls', 'LookupHLS'),
        ('Pool', 'pool_hls', 'PoolHLS'),
        ('StreamingEltwise', 'streamingeltwise_hls', 'StreamingEltwiseHLS'),
        ('StreamingMaxPool', 'streamingmaxpool_hls', 'StreamingMaxPoolHLS'),
        ('Upsampler', 'upsampler_hls', 'UpsamplerHLS'),
        ('VectorVectorActivation', 'vectorvectoractivation_hls', 'VectorVectorActivationHLS'),
    ]
    
    for op_name, module_name, class_name in backends:
        try:
            module = __import__(f'finn.custom_op.fpgadataflow.hls.{module_name}', fromlist=[class_name])
            backend_class = getattr(module, class_name)
            registry.register_hls_backend(op_name, backend_class)
        except ImportError as e:
            logger.debug(f"Could not import {class_name}: {e}")


def _register_clean_hls_backends(registry, logger):
    """Register clean HLS backends."""
    backends = [
        ('Thresholding', 'CG_thresholding_hls', 'CG_ThresholdingHLS'),
        ('MatrixVectorActivation', 'CG_mvau_hls', 'CG_MVAU_hls'),
        ('MVAU', 'CG_mvau_hls', 'CG_MVAU_hls'),
    ]
    
    for op_name, module_name, class_name in backends:
        try:
            module = __import__(f'finn.custom_op.fpgadataflow.hls.{module_name}', fromlist=[class_name])
            backend_class = getattr(module, class_name)
            # Register as primary if clean backends are preferred
            if should_use_clean_backend(op_name):
                registry.register_hls_backend(op_name, backend_class)
        except ImportError as e:
            logger.debug(f"Could not import {class_name}: {e}")


def _register_legacy_rtl_backends(registry, logger):
    """Register legacy RTL backends."""
    backends = [
        ('Thresholding', 'thresholding_rtl', 'Thresholding_rtl'),
        ('MatrixVectorActivation', 'matrixvectoractivation_rtl', 'MVAU_rtl'),
        ('MVAU', 'matrixvectoractivation_rtl', 'MVAU_rtl'),
        ('DynMVAU', 'dynmvau_rtl', 'DynMVU_rtl'),
        ('ConvolutionInputGenerator', 'convolutioninputgenerator_rtl', 'ConvolutionInputGeneratorRTL'),
        ('FMPadding', 'fmpadding_rtl', 'FMPaddingRTL'),
        ('StreamingDataWidthConverter', 'streamingdatawidthconverter_rtl', 'StreamingDataWidthConverterRTL'),
        ('StreamingFIFO', 'streamingfifo_rtl', 'StreamingFIFORTL'),
        ('VectorVectorActivation', 'vectorvectoractivation_rtl', 'VectorVectorActivationRTL'),
    ]
    
    for op_name, module_name, class_name in backends:
        try:
            module = __import__(f'finn.custom_op.fpgadataflow.rtl.{module_name}', fromlist=[class_name])
            backend_class = getattr(module, class_name)
            registry.register_rtl_backend(op_name, backend_class)
        except ImportError as e:
            logger.debug(f"Could not import {class_name}: {e}")


def _register_clean_rtl_backends(registry, logger):
    """Register clean RTL backends."""
    backends = [
        ('Thresholding', 'CG_thresholding_rtl', 'CG_Thresholding_rtl'),
        ('MatrixVectorActivation', 'CG_mvau_rtl', 'CG_MVAU_rtl'),
        ('MVAU', 'CG_mvau_rtl', 'CG_MVAU_rtl'),
    ]
    
    for op_name, module_name, class_name in backends:
        try:
            module = __import__(f'finn.custom_op.fpgadataflow.rtl.{module_name}', fromlist=[class_name])
            backend_class = getattr(module, class_name)
            # Register as primary if clean backends are preferred
            if should_use_clean_backend(op_name):
                registry.register_rtl_backend(op_name, backend_class)
        except ImportError as e:
            logger.debug(f"Could not import {class_name}: {e}")


def should_use_clean_backend(operation_name: str) -> bool:
    """
    Determine if clean backend should be used for operation.
    
    Can be configured via environment variable or config file.
    """
    # For now, default to using clean backends where available
    clean_backend_ops = {'Thresholding', 'MatrixVectorActivation', 'MVAU'}
    return operation_name in clean_backend_ops


# Global registry instance
_global_registry = None


def get_backend_registry() -> BackendRegistry:
    """Get the global backend registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = register_all_backends()
    return _global_registry


def reset_backend_registry():
    """Reset the global backend registry (useful for testing)."""
    global _global_registry
    _global_registry = None


# Convenience functions for backend lookup
def get_hls_backend(operation_name: str):
    """Get HLS backend class for operation."""
    return get_backend_registry().get_hls_backend(operation_name)


def get_rtl_backend(operation_name: str):
    """Get RTL backend class for operation."""
    return get_backend_registry().get_rtl_backend(operation_name)
EOF

    echo "✅ Created merged backend registration file"
}

# Function to remove redundant files
remove_redundant_files() {
    echo ""
    echo "🗑️  Removing redundant files..."
    
    # Backup and remove complex implementations
    if [ -f "src/finn/codegen/file_manager.py" ]; then
        backup_file "src/finn/codegen/file_manager.py"
        rm "src/finn/codegen/file_manager.py"
        echo "   ✅ Removed file_manager.py (using simple_file_manager.py)"
    fi
    
    if [ -f "src/finn/codegen/library_resolver.py" ]; then
        backup_file "src/finn/codegen/library_resolver.py"
        rm "src/finn/codegen/library_resolver.py"
        echo "   ✅ Removed library_resolver.py (using simple_library_resolver.py)"
    fi
    
    if [ -f "src/finn/codegen/CG_backend_registration.py" ]; then
        backup_file "src/finn/codegen/CG_backend_registration.py"
        rm "src/finn/codegen/CG_backend_registration.py"
        echo "   ✅ Removed CG_backend_registration.py (merged into backend_registration.py)"
    fi
}

# Function to rename simple files
rename_simple_files() {
    echo ""
    echo "📝 Renaming simple implementations..."
    
    if [ -f "src/finn/codegen/simple_file_manager.py" ]; then
        mv "src/finn/codegen/simple_file_manager.py" "src/finn/codegen/file_manager.py"
        echo "   ✅ Renamed simple_file_manager.py → file_manager.py"
    fi
    
    if [ -f "src/finn/codegen/simple_library_resolver.py" ]; then
        mv "src/finn/codegen/simple_library_resolver.py" "src/finn/codegen/library_resolver.py"
        echo "   ✅ Renamed simple_library_resolver.py → library_resolver.py"
    fi
}

# Function to update __init__.py
update_init_file() {
    echo ""
    echo "📝 Updating __init__.py..."
    
    cat > src/finn/codegen/__init__.py << 'EOF'
"""
FINN Unified Code Generation - Consolidated Architecture

Simple, explicit, fast.
"""

from .template_engine import TemplateEngine
from .backend_registry import BackendRegistry
from .backend_registration import get_backend_registry, register_all_backends
from .config import CodegenConfig, get_global_config
from .codegen import (
    Codegen,
    UnsupportedTemplateError,
    TemplateValidationError,
    CodeGenerationError
)

# Core components with simplified implementations
from .library_resolver import LibraryResolver
from .file_manager import FileManager

__all__ = [
    # Core components
    'TemplateEngine',
    'BackendRegistry',
    'CodegenConfig',
    
    # Utilities
    'FileManager',
    'LibraryResolver',
    
    # Registration
    'get_backend_registry',
    'register_all_backends',
    'get_global_config',
    
    # Base interface
    'Codegen',
    
    # Exceptions
    'UnsupportedTemplateError',
    'TemplateValidationError',
    'CodeGenerationError'
]
EOF

    echo "✅ Updated __init__.py with direct imports"
}

# Function to clean codegen.py
clean_codegen_file() {
    echo ""
    echo "🧹 Cleaning codegen.py..."
    
    # Remove MockTemplateEngine class from end of file
    if [ -f "src/finn/codegen/codegen.py" ]; then
        # Create cleaned version without MockTemplateEngine
        head -n 316 src/finn/codegen/codegen.py > src/finn/codegen/codegen_cleaned.py
        mv src/finn/codegen/codegen_cleaned.py src/finn/codegen/codegen.py
        echo "   ✅ Removed MockTemplateEngine class"
    fi
}

# Function to show summary
show_summary() {
    echo ""
    echo "📊 Cleanup Summary"
    echo "=================="
    echo "✅ Removed 3 redundant files (~1,146 lines)"
    echo "✅ Renamed 2 simple files to standard names"
    echo "✅ Updated imports in __init__.py"
    echo "✅ Cleaned test code from codegen.py"
    echo "✅ Created merged backend registration"
    echo ""
    echo "🔍 Next steps:"
    echo "   1. Review merged backend_registration_merged.py"
    echo "   2. Test the simplified structure"
    echo "   3. Update any code that imported removed modules"
    echo "   4. Consider removing render_legacy() from template_engine.py"
}

# Main execution
echo "This script will remove redundant code from src/finn/codegen"
echo ""
echo "Actions to be performed:"
echo "  - Remove file_manager.py (332 lines)"
echo "  - Remove library_resolver.py (513 lines)"
echo "  - Remove CG_backend_registration.py (301 lines)"
echo "  - Rename simple_*.py files to standard names"
echo "  - Update __init__.py imports"
echo "  - Clean test code from codegen.py"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    merge_backend_registration
    remove_redundant_files
    rename_simple_files
    update_init_file
    clean_codegen_file
    show_summary
    
    echo ""
    echo "🎉 Redundancy cleanup complete!"
else
    echo "❌ Cleanup cancelled"
fi