############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# FINN Performance Comparison Tests
############################################################################

import os
import sys
import pytest
import tempfile
import shutil
import time
import psutil
import gc
from unittest.mock import Mock, patch, MagicMock

# Add the finn paths for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from finn.util.flexible_hls import (
    FlexibleHLSBackend,
    FINNTemplateEngine,
    FINNTemplateType,
    FINNPathResolver,
    FINNCompilerBuilder,
    FINNConfig
)
from finn.custom_op.fpgadataflow import templates


class PerformanceMeasurement:
    """Helper class for measuring performance metrics."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        
    def start(self):
        """Start performance measurement."""
        gc.collect()  # Clear garbage to get accurate memory reading
        process = psutil.Process()
        self.start_time = time.perf_counter()
        self.start_memory = process.memory_info().rss
        self.peak_memory = self.start_memory
        
    def end(self):
        """End performance measurement."""
        process = psutil.Process()
        self.end_time = time.perf_counter()
        self.end_memory = process.memory_info().rss
        self.peak_memory = max(self.peak_memory, self.end_memory)
        
    def get_duration(self):
        """Get execution duration in seconds."""
        return self.end_time - self.start_time
        
    def get_memory_usage(self):
        """Get memory usage in MB."""
        return (self.end_memory - self.start_memory) / (1024 * 1024)
        
    def get_peak_memory(self):
        """Get peak memory usage in MB."""
        return self.peak_memory / (1024 * 1024)


class TestTemplatePerformance:
    """Test performance of template operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock FINN environment
        self.finn_root = os.path.join(self.temp_dir, "finn")
        self.finn_deps = os.path.join(self.temp_dir, "finn_deps")
        self.finn_hlslib = os.path.join(self.finn_deps, "finn-hlslib")
        
        os.makedirs(self.finn_root)
        os.makedirs(self.finn_hlslib)
        
        # Create template files
        self._create_template_files()
        
        # Standard test variables
        self.test_variables = {
            "AP_INT_MAX_W": "8191",
            "GLOBALS": "#include \"params.h\"\n#include \"weights.h\"\n#include \"thresholds.h\"",
            "DEFINES": "#define PE 16\n#define SIMD 32\n#define MW 784\n#define MH 256\n#define ActVal 0",
            "BLACKBOXFUNCTION": "void MatrixVectorActivation_Stream_0(hls::stream<ap_uint<512>> &in0_V, hls::stream<ap_uint<16>> &out0_V)",
            "PRAGMAS": "#pragma HLS INTERFACE axis port=in0_V register\n#pragma HLS INTERFACE axis port=out0_V register\n#pragma HLS INTERFACE s_axilite port=return bundle=control\n#pragma HLS DATAFLOW",
            "DOCOMPUTE": "MatrixVectorActivation_Stream<784, 256, 16, 32, ap_uint<8>, ap_int<16>, ap_uint<8>>(in0_V, out0_V, weights, thresholds, true);",
            "PROJECTNAME": "MatrixVectorActivation_Stream_0",
            "HWSRCDIR": "/tmp/finn_hw_src_MatrixVectorActivation_Stream_0",
            "FPGAPART": "xc7z020clg400-1",
            "TOPFXN": "MatrixVectorActivation_Stream_0",
            "CLKPERIOD": "10.0",
            "DEFAULT_DIRECTIVES": "config_compile -pipeline_style frp",
            "EXTRA_DIRECTIVES": "config_interface -m_axi_addr64"
        }
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_template_files(self):
        """Create template files for testing."""
        template_dir = os.path.join(self.temp_dir, "templates", "finn")
        os.makedirs(template_dir)
        
        # Create template files with original FINN content
        template_files = {
            "ipgen_cpp.template": templates.ipgen_template,
            "ipgen_tcl.template": templates.ipgentcl_template,
            "docompute.template": templates.docompute_template,
            "docompute_timeout.template": templates.docompute_template_timeout
        }
        
        for filename, content in template_files.items():
            with open(os.path.join(template_dir, filename), 'w') as f:
                f.write(content)
    
    def test_original_string_replacement_performance(self):
        """Measure performance of original string replacement method."""
        measurement = PerformanceMeasurement()
        
        # Test multiple iterations to get stable measurement
        iterations = 1000
        
        measurement.start()
        for _ in range(iterations):
            # Original FINN template processing
            template = templates.ipgen_template
            output = template
            for var, value in self.test_variables.items():
                output = output.replace(f"${var}$", value)
        measurement.end()
        
        original_duration = measurement.get_duration()
        original_memory = measurement.get_memory_usage()
        
        print(f"Original string replacement: {original_duration:.4f}s for {iterations} iterations")
        print(f"Memory usage: {original_memory:.2f}MB")
        
        return {
            'duration': original_duration,
            'memory': original_memory,
            'iterations': iterations
        }
    
    def test_flexible_template_engine_performance(self):
        """Measure performance of flexible template engine."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            measurement = PerformanceMeasurement()
            
            # Initialize template engine (one-time cost)
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            
            # Test multiple iterations to get stable measurement
            iterations = 1000
            
            measurement.start()
            for _ in range(iterations):
                # Flexible template processing
                output = engine.render_template("ipgen_cpp", self.test_variables, FINNTemplateType.IPGEN_CPP)
            measurement.end()
            
            flexible_duration = measurement.get_duration()
            flexible_memory = measurement.get_memory_usage()
            
            print(f"Flexible template engine: {flexible_duration:.4f}s for {iterations} iterations")
            print(f"Memory usage: {flexible_memory:.2f}MB")
            
            return {
                'duration': flexible_duration,
                'memory': flexible_memory,
                'iterations': iterations
            }
    
    def test_template_performance_comparison(self):
        """Compare performance between original and flexible approaches."""
        original_metrics = self.test_original_string_replacement_performance()
        flexible_metrics = self.test_flexible_template_engine_performance()
        
        # Calculate performance ratios
        duration_ratio = flexible_metrics['duration'] / original_metrics['duration']
        memory_ratio = flexible_metrics['memory'] / original_metrics['memory'] if original_metrics['memory'] > 0 else 1.0
        
        print(f"\n=== Performance Comparison ===")
        print(f"Duration ratio (flexible/original): {duration_ratio:.2f}x")
        print(f"Memory ratio (flexible/original): {memory_ratio:.2f}x")
        
        # Flexible should be at most 3x slower (acceptable overhead for added functionality)
        assert duration_ratio < 3.0, f"Flexible template engine too slow: {duration_ratio:.2f}x slower"
        
        # Memory usage should be reasonable (within 2x)
        assert memory_ratio < 2.0, f"Flexible template engine uses too much memory: {memory_ratio:.2f}x more"
        
        return {
            'duration_ratio': duration_ratio,
            'memory_ratio': memory_ratio,
            'original': original_metrics,
            'flexible': flexible_metrics
        }
    
    def test_template_initialization_overhead(self):
        """Measure one-time initialization overhead of template engine."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            measurement = PerformanceMeasurement()
            
            measurement.start()
            # Initialize template engine
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            measurement.end()
            
            init_duration = measurement.get_duration()
            init_memory = measurement.get_memory_usage()
            
            print(f"Template engine initialization: {init_duration:.4f}s")
            print(f"Initialization memory: {init_memory:.2f}MB")
            
            # Initialization should be fast (< 1 second)
            assert init_duration < 1.0, f"Template engine initialization too slow: {init_duration:.4f}s"
            
            return {
                'duration': init_duration,
                'memory': init_memory
            }
    
    def test_large_template_performance(self):
        """Test performance with larger, more complex templates."""
        # Create a large template with many variables
        large_template_content = templates.ipgen_template
        
        # Add more content to simulate larger templates
        for i in range(50):
            large_template_content += f"\n// Additional section {i}\n$EXTRA_VAR_{i}$\n"
        
        # Create large template file
        template_dir = os.path.join(self.temp_dir, "templates", "finn")
        large_template_file = os.path.join(template_dir, "large_template.template")
        with open(large_template_file, 'w') as f:
            f.write(large_template_content)
        
        # Create variables for large template
        large_variables = self.test_variables.copy()
        for i in range(50):
            large_variables[f"EXTRA_VAR_{i}"] = f"Value for extra variable {i}"
        
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Test original approach
            measurement_original = PerformanceMeasurement()
            measurement_original.start()
            
            iterations = 100
            for _ in range(iterations):
                output = large_template_content
                for var, value in large_variables.items():
                    output = output.replace(f"${var}$", value)
            
            measurement_original.end()
            
            # Test flexible approach
            engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
            
            measurement_flexible = PerformanceMeasurement()
            measurement_flexible.start()
            
            for _ in range(iterations):
                output = engine.render_template("large_template", large_variables, FINNTemplateType.CUSTOM)
            
            measurement_flexible.end()
            
            original_duration = measurement_original.get_duration()
            flexible_duration = measurement_flexible.get_duration()
            ratio = flexible_duration / original_duration
            
            print(f"Large template - Original: {original_duration:.4f}s")
            print(f"Large template - Flexible: {flexible_duration:.4f}s")
            print(f"Large template ratio: {ratio:.2f}x")
            
            # Should still be reasonable for large templates
            assert ratio < 5.0, f"Large template processing too slow: {ratio:.2f}x slower"


class TestPathResolutionPerformance:
    """Test performance of path resolution operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock FINN environment
        self.finn_root = os.path.join(self.temp_dir, "finn")
        self.finn_deps = os.path.join(self.temp_dir, "finn_deps")
        self.finn_hlslib = os.path.join(self.finn_deps, "finn-hlslib")
        
        os.makedirs(self.finn_root)
        os.makedirs(self.finn_hlslib)
        
        # Create many files to test performance
        self._create_many_files()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_many_files(self):
        """Create many files to test path resolution performance."""
        # Create many header files
        for i in range(100):
            with open(os.path.join(self.finn_hlslib, f"header_{i}.h"), 'w') as f:
                f.write(f"// Header file {i}\n")
        
        # Create custom HLS directory with files
        custom_hls_dir = os.path.join(self.finn_root, "custom_hls")
        os.makedirs(custom_hls_dir)
        
        for i in range(50):
            with open(os.path.join(custom_hls_dir, f"custom_{i}.h"), 'w') as f:
                f.write(f"// Custom file {i}\n")
    
    def test_path_resolution_performance(self):
        """Test performance of path resolution operations."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            measurement = PerformanceMeasurement()
            measurement.start()
            
            # Test many path resolution operations
            iterations = 1000
            for i in range(iterations):
                # Test different path operations
                resolver.get_finn_root()
                resolver.get_finn_deps_dir()
                resolver.get_finn_hlslib_dir()
                resolver.get_include_paths()
                resolver.get_library_paths()
                resolver.validate_finn_environment()
            
            measurement.end()
            
            duration = measurement.get_duration()
            memory = measurement.get_memory_usage()
            
            print(f"Path resolution: {duration:.4f}s for {iterations} iterations")
            print(f"Memory usage: {memory:.2f}MB")
            
            # Should be very fast (< 1 second for 1000 operations)
            assert duration < 1.0, f"Path resolution too slow: {duration:.4f}s"
            
            return {
                'duration': duration,
                'memory': memory,
                'iterations': iterations
            }
    
    def test_file_existence_checking_performance(self):
        """Test performance of file existence checking."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            resolver = FINNPathResolver()
            
            measurement = PerformanceMeasurement()
            measurement.start()
            
            # Test many file existence checks
            iterations = 100
            for i in range(iterations):
                for j in range(10):  # Check 10 files per iteration
                    resolver.file_exists(f"header_{j}.h", resolver.FINNPathType.FINN_HLSLIB)
                    resolver.file_exists(f"custom_{j}.h", resolver.FINNPathType.FINN_ROOT)
            
            measurement.end()
            
            duration = measurement.get_duration()
            total_checks = iterations * 10 * 2  # 2 checks per iteration
            
            print(f"File existence checking: {duration:.4f}s for {total_checks} checks")
            print(f"Checks per second: {total_checks / duration:.0f}")
            
            # Should be reasonably fast
            assert duration < 5.0, f"File existence checking too slow: {duration:.4f}s"
            
            return {
                'duration': duration,
                'total_checks': total_checks,
                'checks_per_second': total_checks / duration
            }


class TestCompilerBuilderPerformance:
    """Test performance of compiler builder operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock FINN environment
        self.finn_root = os.path.join(self.temp_dir, "finn")
        self.finn_deps = os.path.join(self.temp_dir, "finn_deps")
        self.finn_hlslib = os.path.join(self.finn_deps, "finn-hlslib")
        
        os.makedirs(self.finn_root)
        os.makedirs(self.finn_hlslib)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_compiler_configuration_performance(self):
        """Test performance of compiler configuration building."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            measurement = PerformanceMeasurement()
            measurement.start()
            
            # Test many compiler configurations
            iterations = 1000
            for i in range(iterations):
                config = FINNConfig()
                builder = FINNCompilerBuilder(config)
                
                # Build compiler configuration
                include_paths = [f"/path/include_{j}" for j in range(10)]
                source_files = [f"source_{j}.cpp" for j in range(5)]
                
                builder.add_include_paths(include_paths)
                builder.add_source_files(source_files)
                builder.set_optimization_level(2)
                builder.set_target_device("xc7z020clg400-1")
                
                # Get compile command
                compile_cmd = builder.get_compile_command()
            
            measurement.end()
            
            duration = measurement.get_duration()
            memory = measurement.get_memory_usage()
            
            print(f"Compiler configuration: {duration:.4f}s for {iterations} iterations")
            print(f"Memory usage: {memory:.2f}MB")
            
            # Should be fast
            assert duration < 2.0, f"Compiler configuration too slow: {duration:.4f}s"
            
            return {
                'duration': duration,
                'memory': memory,
                'iterations': iterations
            }


class TestOverallSystemPerformance:
    """Test overall system performance with realistic workloads."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock FINN environment
        self.finn_root = os.path.join(self.temp_dir, "finn")
        self.finn_deps = os.path.join(self.temp_dir, "finn_deps")
        self.finn_hlslib = os.path.join(self.finn_deps, "finn-hlslib")
        
        os.makedirs(self.finn_root)
        os.makedirs(self.finn_hlslib)
        
        # Create template files
        self._create_template_files()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_template_files(self):
        """Create template files for testing."""
        template_dir = os.path.join(self.temp_dir, "templates", "finn")
        os.makedirs(template_dir)
        
        template_files = {
            "ipgen_cpp.template": templates.ipgen_template,
            "ipgen_tcl.template": templates.ipgentcl_template,
            "docompute.template": templates.docompute_template
        }
        
        for filename, content in template_files.items():
            with open(os.path.join(template_dir, filename), 'w') as f:
                f.write(content)
    
    def test_realistic_kernel_compilation_workflow(self):
        """Test performance of realistic kernel compilation workflow."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            measurement = PerformanceMeasurement()
            measurement.start()
            
            # Simulate realistic kernel processing
            iterations = 10  # Fewer iterations for complex workflow
            
            for i in range(iterations):
                # Initialize components (one-time per kernel)
                config = FINNConfig()
                resolver = FINNPathResolver()
                engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
                builder = FINNCompilerBuilder(config)
                
                # Simulate kernel-specific variables
                kernel_variables = {
                    "AP_INT_MAX_W": "8191",
                    "GLOBALS": f"#include \"kernel_{i}_params.h\"",
                    "DEFINES": f"#define PE {16 + i}\n#define SIMD {32 + i}",
                    "BLACKBOXFUNCTION": f"void Kernel_{i}_Stream(hls::stream<ap_uint<64>> &in, hls::stream<ap_uint<64>> &out)",
                    "PRAGMAS": "#pragma HLS INTERFACE axis port=in\n#pragma HLS INTERFACE axis port=out",
                    "DOCOMPUTE": f"Kernel_{i}_Stream<64, 64>(in, out);",
                    "PROJECTNAME": f"Kernel_{i}_Project",
                    "TOPFXN": f"Kernel_{i}_Stream",
                    "FPGAPART": "xc7z020clg400-1",
                    "CLKPERIOD": "10.0"
                }
                
                # Generate multiple templates
                ipgen_cpp = engine.render_template("ipgen_cpp", kernel_variables, FINNTemplateType.IPGEN_CPP)
                ipgen_tcl = engine.render_template("ipgen_tcl", kernel_variables, FINNTemplateType.IPGEN_TCL)
                docompute = engine.render_template("docompute", kernel_variables, FINNTemplateType.DOCOMPUTE)
                
                # Setup compilation
                include_paths = resolver.get_include_paths()
                builder.add_include_paths(include_paths)
                builder.set_target_device("xc7z020clg400-1")
                compile_cmd = builder.get_compile_command()
            
            measurement.end()
            
            duration = measurement.get_duration()
            memory = measurement.get_memory_usage()
            
            print(f"Realistic workflow: {duration:.4f}s for {iterations} kernels")
            print(f"Average per kernel: {duration/iterations:.4f}s")
            print(f"Memory usage: {memory:.2f}MB")
            
            # Should complete reasonable number of kernels quickly
            avg_per_kernel = duration / iterations
            assert avg_per_kernel < 1.0, f"Kernel processing too slow: {avg_per_kernel:.4f}s per kernel"
            
            return {
                'total_duration': duration,
                'avg_per_kernel': avg_per_kernel,
                'memory': memory,
                'iterations': iterations
            }
    
    def test_memory_efficiency(self):
        """Test memory efficiency of the flexible backend system."""
        test_env = {
            'FINN_ROOT': self.finn_root,
            'FINN_DEPS_DIR': self.finn_deps,
            'FINN_HLSLIB_DIR': self.finn_hlslib
        }
        
        with patch.dict(os.environ, test_env):
            # Test with many template engine instances (simulating many kernels)
            engines = []
            
            measurement = PerformanceMeasurement()
            measurement.start()
            
            # Create many instances
            for i in range(100):
                engine = FINNTemplateEngine([os.path.join(self.temp_dir, "templates")])
                engines.append(engine)
            
            measurement.end()
            
            duration = measurement.get_duration()
            memory = measurement.get_memory_usage()
            
            print(f"Memory efficiency test: {duration:.4f}s for 100 instances")
            print(f"Memory usage: {memory:.2f}MB")
            print(f"Memory per instance: {memory/100:.3f}MB")
            
            # Clean up
            del engines
            gc.collect()
            
            # Memory usage should be reasonable
            memory_per_instance = memory / 100
            assert memory_per_instance < 5.0, f"Too much memory per instance: {memory_per_instance:.3f}MB"
            
            return {
                'duration': duration,
                'total_memory': memory,
                'memory_per_instance': memory_per_instance
            }


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])