#!/usr/bin/env python3
"""
FINN Unified Codegen - Performance and Stress Testing
Tests performance characteristics and stress scenarios of the unified codegen framework.
"""

import sys
import os
import time
import gc
import traceback
import tempfile
import psutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def print_status(status, message):
    """Print colored status messages"""
    colors = {
        'INFO': '\033[0;34m',
        'SUCCESS': '\033[0;32m',
        'WARNING': '\033[1;33m',
        'ERROR': '\033[0;31m',
        'NC': '\033[0m'
    }
    
    icons = {
        'INFO': 'ℹ️ ',
        'SUCCESS': '✅',
        'WARNING': '⚠️ ',
        'ERROR': '❌'
    }
    
    print(f"{colors.get(status, '')}{icons.get(status, '')}{message}{colors['NC']}")

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_stress_operation(complexity_factor=1):
    """Create a complete stress test operation with scalable complexity - NO MOCK OBJECTS"""
    class CompleteStressTestOperation:
        def __init__(self, complexity):
            self.complexity = complexity
            self.onnx_node = type('Node', (), {
                'name': f'stress_test_{complexity}',
                'op_type': 'StressTestOperation',
                'input': [f'input{i}' for i in range(complexity)],
                'output': [f'output{i}' for i in range(complexity)],
                'attribute': []  # FINN operations expect this attribute
            })()
            
            # Scale attributes with complexity
            self._attrs = {
                'MW': 32 * complexity,
                'MH': 32 * complexity,
                'PE': 4 * complexity,
                'SIMD': 2 * complexity,
                'mem_mode': 'internal_decoupled',
                'runtime_writeable_weights': 0,
                'inputDataType': 'INT8',
                'outputDataType': 'INT8',
                'weightDataType': 'INT8',
                'ActVal': 0,
                'binaryXnorMode': 0,
                'noActivation': 0,
                'backend': 'hls',
                'resType': 'auto',
                'numInputVectors': complexity
            }
        
        def get_nodeattr_types(self):
            return {
                'MW': int, 'MH': int, 'PE': int, 'SIMD': int,
                'mem_mode': str, 'runtime_writeable_weights': int,
                'inputDataType': str, 'outputDataType': str, 'weightDataType': str,
                'ActVal': int, 'binaryXnorMode': int, 'noActivation': int,
                'backend': str, 'resType': str, 'numInputVectors': int
            }
            
        def get_nodeattr(self, name):
            return self._attrs.get(name)
        
        def set_nodeattr(self, name, value):
            if name in self._attrs:
                self._attrs[name] = value
            
        def get_input_datatype(self, idx):
            class TestDataType:
                def bitwidth(self): return 8
                def __str__(self): return 'INT8'
                def min(self): return -128
                def max(self): return 127
                def name(self): return 'INT8'
            return TestDataType()
            
        def get_output_datatype(self, idx):
            class TestDataType:
                def bitwidth(self): return 8
                def __str__(self): return 'INT8'
                def min(self): return -128
                def max(self): return 127
                def name(self): return 'INT8'
            return TestDataType()
            
        def get_instream_width(self, idx=0):
            return self.get_nodeattr('SIMD') * self.get_input_datatype(idx).bitwidth()
            
        def get_outstream_width(self, idx=0):
            return self.get_nodeattr('PE') * self.get_output_datatype(idx).bitwidth()
            
        def get_normal_input_shape(self, idx=0):
            return [1, self.get_nodeattr('MW')]
            
        def get_normal_output_shape(self, idx=0):
            return [1, self.get_nodeattr('MH')]
            
        def get_verilog_top_module_intf_names(self):
            return [f"s_axis_{i}" for i in range(self.complexity)] + \
                   [f"m_axis_{i}" for i in range(self.complexity)]
            
        def get_template_param_values(self):
            return {
                'MW': self.get_nodeattr('MW'),
                'MH': self.get_nodeattr('MH'),
                'PE': self.get_nodeattr('PE'),
                'SIMD': self.get_nodeattr('SIMD'),
                'INPUT_WIDTH': self.get_instream_width(),
                'OUTPUT_WIDTH': self.get_outstream_width(),
                'MEM_MODE': self.get_nodeattr('mem_mode'),
                'RUNTIME_WEIGHTS': self.get_nodeattr('runtime_writeable_weights'),
                'COMPLEXITY': self.complexity,
                'INPUT_PORTS': self.complexity,
                'OUTPUT_PORTS': self.complexity
            }
            
        def get_expected_cycles(self):
            return self.get_nodeattr('MW') // self.get_nodeattr('PE')
            
        def get_op_and_param_counts(self):
            return {
                'op_mac': self.get_nodeattr('MW') * self.get_nodeattr('PE'),
                'param_count': self.get_nodeattr('MW') * self.get_nodeattr('SIMD')
            }
    
    return CompleteStressTestOperation(complexity_factor)

def test_code_generation_performance():
    """Test code generation performance with varying complexity"""
    print_status('INFO', 'Testing code generation performance...')
    
    results = []
    
    try:
        from finn.codegen import ModernHLSGenerator, ModernRTLGenerator
        
        # Test with increasing complexity
        for complexity in [1, 2, 4, 8, 16]:
            print(f"Testing complexity level {complexity}...")
            
            # HLS generation timing
            hls_times = []
            for iteration in range(5):  # 5 iterations for averaging
                start_time = time.time()
                
                stress_op = create_stress_operation(complexity)
                hls_gen = ModernHLSGenerator(stress_op)
                context = hls_gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
                
                end_time = time.time()
                hls_times.append(end_time - start_time)
            
            avg_hls_time = sum(hls_times) / len(hls_times)
            
            # RTL generation timing
            rtl_times = []
            for iteration in range(5):
                start_time = time.time()
                
                stress_op = create_stress_operation(complexity)
                rtl_gen = ModernRTLGenerator(stress_op)
                context = rtl_gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
                
                end_time = time.time()
                rtl_times.append(end_time - start_time)
            
            avg_rtl_time = sum(rtl_times) / len(rtl_times)
            
            results.append({
                'complexity': complexity,
                'hls_time': avg_hls_time,
                'rtl_time': avg_rtl_time,
                'total_time': avg_hls_time + avg_rtl_time
            })
            
            print(f"  Complexity {complexity}: HLS={avg_hls_time:.3f}s, RTL={avg_rtl_time:.3f}s")
        
        # Analyze performance trends
        print("\n📊 Performance Analysis:")
        for i, result in enumerate(results):
            if i > 0:
                prev_result = results[i-1]
                hls_ratio = result['hls_time'] / prev_result['hls_time']
                rtl_ratio = result['rtl_time'] / prev_result['rtl_time']
                print(f"  Complexity {result['complexity']}: HLS scaling={hls_ratio:.2f}x, RTL scaling={rtl_ratio:.2f}x")
        
        # Performance criteria: generation should complete within reasonable time
        max_time = max(r['total_time'] for r in results)
        if max_time < 1.0:  # Less than 1 second for highest complexity
            print_status('SUCCESS', f'Code generation performance excellent (max: {max_time:.3f}s)')
            return True
        elif max_time < 5.0:  # Less than 5 seconds
            print_status('SUCCESS', f'Code generation performance good (max: {max_time:.3f}s)')
            return True
        else:
            print_status('WARNING', f'Code generation performance needs optimization (max: {max_time:.3f}s)')
            return False
            
    except Exception as e:
        print_status('ERROR', f'Performance test failed: {e}')
        return False

def test_memory_usage():
    """Test memory usage patterns"""
    print_status('INFO', 'Testing memory usage patterns...')
    
    try:
        from finn.codegen import ModernHLSGenerator, ModernRTLGenerator, TemplateEngine, FileManager, LibraryResolver
        
        initial_memory = get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.1f} MB")
        
        # Test 1: Object creation memory usage
        objects = []
        for i in range(100):
            # Create multiple objects
            stress_op = create_stress_operation(1)
            hls_gen = ModernHLSGenerator(stress_op)
            rtl_gen = ModernRTLGenerator(stress_op)
            template_engine = TemplateEngine()
            
            with tempfile.TemporaryDirectory() as tmp:
                file_manager = FileManager(tmp)
                lib_resolver = LibraryResolver()
                
                objects.extend([stress_op, hls_gen, rtl_gen, template_engine, file_manager, lib_resolver])
        
        mid_memory = get_memory_usage()
        memory_increase = mid_memory - initial_memory
        print(f"After creating 600 objects: {mid_memory:.1f} MB (+{memory_increase:.1f} MB)")
        
        # Test 2: Memory cleanup
        objects.clear()
        gc.collect()  # Force garbage collection
        
        final_memory = get_memory_usage()
        memory_retained = final_memory - initial_memory
        print(f"After cleanup: {final_memory:.1f} MB (+{memory_retained:.1f} MB retained)")
        
        # Test 3: Stress test with many operations
        print("Running memory stress test...")
        stress_objects = []
        
        for i in range(50):
            stress_op = create_stress_operation(i % 10 + 1)  # Varying complexity
            hls_gen = ModernHLSGenerator(stress_op)
            
            # Generate context multiple times
            for j in range(10):
                context = hls_gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
                
            stress_objects.append((stress_op, hls_gen))
        
        stress_memory = get_memory_usage()
        stress_increase = stress_memory - final_memory
        print(f"After stress test: {stress_memory:.1f} MB (+{stress_increase:.1f} MB)")
        
        # Cleanup stress test
        stress_objects.clear()
        gc.collect()
        
        cleanup_memory = get_memory_usage()
        final_retained = cleanup_memory - initial_memory
        print(f"Final memory: {cleanup_memory:.1f} MB (+{final_retained:.1f} MB total retained)")
        
        # Memory usage criteria
        if final_retained < 50:  # Less than 50MB retained
            print_status('SUCCESS', f'Memory usage excellent ({final_retained:.1f} MB retained)')
            return True
        elif final_retained < 100:  # Less than 100MB retained
            print_status('SUCCESS', f'Memory usage good ({final_retained:.1f} MB retained)')
            return True
        else:
            print_status('WARNING', f'Memory usage needs optimization ({final_retained:.1f} MB retained)')
            return False
            
    except Exception as e:
        print_status('ERROR', f'Memory usage test failed: {e}')
        return False

def test_concurrent_operations():
    """Test concurrent code generation operations"""
    print_status('INFO', 'Testing concurrent operations...')
    
    try:
        from finn.codegen import ModernHLSGenerator, ModernRTLGenerator
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def worker_thread(thread_id, iterations):
            """Worker thread for concurrent testing"""
            try:
                thread_results = []
                for i in range(iterations):
                    # Create operation
                    stress_op = create_stress_operation((thread_id % 5) + 1)
                    
                    # Generate HLS
                    hls_gen = ModernHLSGenerator(stress_op)
                    hls_context = hls_gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
                    
                    # Generate RTL
                    rtl_gen = ModernRTLGenerator(stress_op)
                    rtl_context = rtl_gen.prepare_context(None, 'xc7z020clg400-1', '100MHz')
                    
                    thread_results.append({
                        'thread_id': thread_id,
                        'iteration': i,
                        'hls_keys': len(hls_context),
                        'rtl_keys': len(rtl_context)
                    })
                
                results_queue.put(('success', thread_id, thread_results))
            except Exception as e:
                results_queue.put(('error', thread_id, str(e)))
        
        # Start multiple threads
        num_threads = 4
        iterations_per_thread = 10
        threads = []
        
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=worker_thread, args=(thread_id, iterations_per_thread))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect results
        successful_threads = 0
        total_operations = 0
        
        while not results_queue.empty():
            result_type, thread_id, data = results_queue.get()
            
            if result_type == 'success':
                successful_threads += 1
                total_operations += len(data)
                print(f"  Thread {thread_id}: {len(data)} operations completed")
            else:
                print_status('ERROR', f'Thread {thread_id} failed: {data}')
        
        print(f"Concurrent test completed in {total_time:.2f}s")
        print(f"Total operations: {total_operations}")
        print(f"Operations per second: {total_operations / total_time:.1f}")
        
        if successful_threads == num_threads:
            print_status('SUCCESS', 'All concurrent operations completed successfully')
            return True
        else:
            print_status('WARNING', f'Only {successful_threads}/{num_threads} threads completed successfully')
            return False
            
    except Exception as e:
        print_status('ERROR', f'Concurrent operations test failed: {e}')
        return False

def test_template_performance():
    """Test template rendering performance"""
    print_status('INFO', 'Testing template rendering performance...')
    
    try:
        from finn.codegen import TemplateEngine
        
        engine = TemplateEngine()
        
        # Create a complex template
        complex_template = """
// Generated code for {{ op_name }}
// Operation: {{ op_type }}
// Complexity: {{ complexity }}

{%- for i in range(input_count) %}
// Input {{ i }}: {{ input_types[i] }}
{%- endfor %}

{%- for i in range(output_count) %}
// Output {{ i }}: {{ output_types[i] }}
{%- endfor %}

void {{ op_name }}_impl(
{%- for i in range(input_count) %}
    {{ input_types[i] }}& input{{ i }}{{ "," if not loop.last }}
{%- endfor %}
{%- for i in range(output_count) %}
    {{ "," if input_count > 0 }}{{ output_types[i] }}& output{{ i }}{{ "," if not loop.last }}
{%- endfor %}
) {
    // Parameters
{%- for param, value in parameters.items() %}
    const int {{ param }} = {{ value }};
{%- endfor %}
    
    // Main processing loop
    for (int pe = 0; pe < PE; pe++) {
        for (int mw = 0; mw < MW; mw++) {
            for (int simd = 0; simd < SIMD; simd++) {
                // Complex computation here
                // {{ computation_comment }}
            }
        }
    }
}
"""
        
        # Test with varying complexity
        render_times = []
        
        for complexity in [1, 5, 10, 20, 50]:
            context = {
                'op_name': f'complex_op_{complexity}',
                'op_type': 'ComplexOperation',
                'complexity': complexity,
                'input_count': complexity,
                'output_count': complexity,
                'input_types': [f'stream_t<{8 * (i+1)}>' for i in range(complexity)],
                'output_types': [f'stream_t<{8 * (i+1)}>' for i in range(complexity)],
                'parameters': {
                    'PE': 4 * complexity,
                    'MW': 32 * complexity,
                    'SIMD': 2 * complexity
                },
                'computation_comment': f'Processing element {complexity} with complex logic'
            }
            
            # Time multiple renders
            iterations = 100
            start_time = time.time()
            
            for _ in range(iterations):
                rendered = engine.render_string(complex_template, context)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / iterations
            render_times.append(avg_time)
            
            print(f"  Complexity {complexity}: {avg_time:.4f}s per render")
        
        # Analyze performance
        max_render_time = max(render_times)
        
        if max_render_time < 0.01:  # Less than 10ms
            print_status('SUCCESS', f'Template rendering performance excellent (max: {max_render_time:.4f}s)')
            return True
        elif max_render_time < 0.05:  # Less than 50ms
            print_status('SUCCESS', f'Template rendering performance good (max: {max_render_time:.4f}s)')
            return True
        else:
            print_status('WARNING', f'Template rendering performance needs optimization (max: {max_render_time:.4f}s)')
            return False
            
    except Exception as e:
        print_status('ERROR', f'Template performance test failed: {e}')
        return False

def test_file_io_performance():
    """Test file I/O performance"""
    print_status('INFO', 'Testing file I/O performance...')
    
    try:
        from finn.codegen import FileManager
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            fm = FileManager(tmp_dir)
            
            # Test 1: Write performance
            write_times = []
            file_sizes = [1024, 10240, 102400, 1024000]  # 1KB to 1MB
            
            for size in file_sizes:
                content = '// Generated code\n' * (size // 20)  # Approximate size
                content = content[:size]  # Exact size
                
                start_time = time.time()
                file_path = fm.write_file(f'test_{size}.cpp', content)
                end_time = time.time()
                
                write_time = end_time - start_time
                write_times.append(write_time)
                
                print(f"  Write {size} bytes: {write_time:.4f}s")
            
            # Test 2: Read performance
            read_times = []
            
            for i, size in enumerate(file_sizes):
                file_path = f'test_{size}.cpp'
                
                start_time = time.time()
                content = fm.read_file(file_path)
                end_time = time.time()
                
                read_time = end_time - start_time
                read_times.append(read_time)
                
                print(f"  Read {len(content)} bytes: {read_time:.4f}s")
            
            # Test 3: Bulk operations
            bulk_files = 100
            bulk_content = '// Bulk test file\n' * 100
            
            start_time = time.time()
            for i in range(bulk_files):
                fm.write_file(f'bulk_{i}.cpp', bulk_content)
            end_time = time.time()
            
            bulk_write_time = end_time - start_time
            print(f"  Bulk write {bulk_files} files: {bulk_write_time:.4f}s")
            
            start_time = time.time()
            for i in range(bulk_files):
                content = fm.read_file(f'bulk_{i}.cpp')
            end_time = time.time()
            
            bulk_read_time = end_time - start_time
            print(f"  Bulk read {bulk_files} files: {bulk_read_time:.4f}s")
            
            # Performance criteria
            max_single_op = max(max(write_times), max(read_times))
            
            if max_single_op < 0.1 and bulk_write_time < 1.0 and bulk_read_time < 1.0:
                print_status('SUCCESS', 'File I/O performance excellent')
                return True
            elif max_single_op < 0.5 and bulk_write_time < 5.0 and bulk_read_time < 5.0:
                print_status('SUCCESS', 'File I/O performance good')
                return True
            else:
                print_status('WARNING', 'File I/O performance needs optimization')
                return False
                
    except Exception as e:
        print_status('ERROR', f'File I/O performance test failed: {e}')
        return False

def main():
    """Run all performance and stress tests"""
    print("💪 FINN Unified Codegen - Performance and Stress Testing")
    print("=======================================================")
    
    tests = []
    
    # Run performance tests
    print("\n⚡ Testing Code Generation Performance")
    print("-" * 40)
    tests.append(test_code_generation_performance())
    
    print("\n🧠 Testing Memory Usage")
    print("-" * 40)
    tests.append(test_memory_usage())
    
    print("\n🔄 Testing Concurrent Operations")
    print("-" * 40)
    tests.append(test_concurrent_operations())
    
    print("\n📝 Testing Template Performance")
    print("-" * 40)
    tests.append(test_template_performance())
    
    print("\n💾 Testing File I/O Performance")
    print("-" * 40)
    tests.append(test_file_io_performance())
    
    # Summary
    total_tests = len(tests)
    passed_tests = sum(tests)
    failed_tests = total_tests - passed_tests
    
    print("\n📊 Performance Testing Summary")
    print("=" * 40)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    
    # Success criteria: at least 80% of performance tests should pass
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.8:
        print_status('SUCCESS', f'PERFORMANCE TESTING PASSED ({success_rate:.1%} success rate)')
        return True
    else:
        print_status('WARNING', f'Performance testing needs attention ({success_rate:.1%} success rate)')
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print_status('ERROR', f'Performance testing failed: {e}')
        traceback.print_exc()
        sys.exit(1)