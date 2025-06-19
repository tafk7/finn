#!/usr/bin/env python3
"""
Performance benchmarking suite for FINN Docker system
Measures and compares performance between old and new approaches
"""

import os
import sys
import time
import subprocess
import json
import statistics
from pathlib import Path
from datetime import datetime
import argparse

# Add FINN root to path
FINN_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FINN_ROOT))

class DockerBenchmark:
    """Benchmark suite for Docker performance"""
    
    def __init__(self, runs=5):
        self.finn_root = FINN_ROOT
        self.finn_docker = self.finn_root / "finn-docker"
        self.run_docker = self.finn_root / "run-docker.sh"
        self.runs = runs
        self.results = {}
        
    def run_command(self, cmd, timeout=300):
        """Run command and measure execution time"""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.finn_root
            )
            elapsed = time.time() - start_time
            return {
                'success': result.returncode == 0,
                'time': elapsed,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'time': timeout,
                'error': 'Timeout'
            }
            
    def cleanup(self):
        """Clean up any existing containers"""
        subprocess.run([str(self.finn_docker), "clean"], capture_output=True)
        # Also try to clean up old style containers
        subprocess.run(["docker", "rm", "-f", "finn_dev_test"], capture_output=True)
        
    def benchmark_cold_start(self):
        """Benchmark cold start (no cache)"""
        print("\n=== Benchmarking Cold Start ===")
        
        # Clear caches
        cache_dir = Path("/tmp/.finn_cache")
        if cache_dir.exists():
            subprocess.run(["rm", "-rf", str(cache_dir)], capture_output=True)
            
        times = []
        for i in range(self.runs):
            print(f"Run {i+1}/{self.runs}...", end='', flush=True)
            
            # Clean up
            self.cleanup()
            
            # Measure initialization time
            result = self.run_command(f"{self.finn_docker} init")
            
            if result['success']:
                times.append(result['time'])
                print(f" {result['time']:.2f}s")
            else:
                print(f" FAILED")
                
            # Clean up after
            self.cleanup()
            
        if times:
            self.results['cold_start'] = {
                'times': times,
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0
            }
            
    def benchmark_warm_start(self):
        """Benchmark warm start (with cache)"""
        print("\n=== Benchmarking Warm Start ===")
        
        # First, do a cold start to populate caches
        print("Initializing cache...", end='', flush=True)
        self.cleanup()
        result = self.run_command(f"{self.finn_docker} init")
        if not result['success']:
            print(" FAILED - skipping warm start benchmark")
            return
        print(" done")
        
        # Stop container but keep cache
        subprocess.run([str(self.finn_docker), "stop"], capture_output=True)
        
        times = []
        for i in range(self.runs):
            print(f"Run {i+1}/{self.runs}...", end='', flush=True)
            
            # Measure restart time
            result = self.run_command(f"{self.finn_docker} init")
            
            if result['success']:
                times.append(result['time'])
                print(f" {result['time']:.2f}s")
            else:
                print(f" FAILED")
                
            # Stop container between runs
            subprocess.run([str(self.finn_docker), "stop"], capture_output=True)
            
        if times:
            self.results['warm_start'] = {
                'times': times,
                'mean': statistics.mean(times),
                'median': statistics.median(times),
                'stdev': statistics.stdev(times) if len(times) > 1 else 0
            }
            
        # Final cleanup
        self.cleanup()
        
    def benchmark_command_execution(self):
        """Benchmark command execution speed"""
        print("\n=== Benchmarking Command Execution ===")
        
        # Initialize container first
        print("Initializing container...", end='', flush=True)
        self.cleanup()
        result = self.run_command(f"{self.finn_docker} init")
        if not result['success']:
            print(" FAILED - skipping execution benchmark")
            return
        print(" done")
        
        # Wait for container to be fully ready
        time.sleep(2)
        
        commands = [
            ("echo 'hello'", "Simple echo"),
            ("python3 -c 'print(1+1)'", "Python calculation"),
            ("ls -la", "Directory listing"),
            ("python3 -c 'import numpy; print(numpy.__version__)'", "Import numpy"),
        ]
        
        for cmd, description in commands:
            print(f"\nBenchmarking: {description}")
            times = []
            
            for i in range(self.runs):
                print(f"  Run {i+1}/{self.runs}...", end='', flush=True)
                
                result = self.run_command(f"{self.finn_docker} exec \"{cmd}\"")
                
                if result['success']:
                    times.append(result['time'])
                    print(f" {result['time']:.3f}s")
                else:
                    print(f" FAILED")
                    
            if times:
                self.results[f'exec_{cmd.split()[0]}'] = {
                    'command': cmd,
                    'description': description,
                    'times': times,
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'stdev': statistics.stdev(times) if len(times) > 1 else 0
                }
                
        # Cleanup
        self.cleanup()
        
    def benchmark_shortcuts(self):
        """Benchmark shortcut commands"""
        print("\n=== Benchmarking Shortcuts ===")
        
        # Initialize container first
        print("Initializing container...", end='', flush=True)
        self.cleanup()
        result = self.run_command(f"{self.finn_docker} init")
        if not result['success']:
            print(" FAILED - skipping shortcuts benchmark")
            return
        print(" done")
        
        # Create a simple Python script for testing
        test_script = self.finn_root / "test_benchmark.py"
        test_script.write_text("print('Hello from benchmark')")
        
        shortcuts = [
            ("python test_benchmark.py", "Python shortcut"),
            ("pytest --version", "Pytest shortcut"),
        ]
        
        for cmd, description in shortcuts:
            print(f"\nBenchmarking: {description}")
            times = []
            
            for i in range(self.runs):
                print(f"  Run {i+1}/{self.runs}...", end='', flush=True)
                
                result = self.run_command(f"{self.finn_docker} {cmd}")
                
                if result['success']:
                    times.append(result['time'])
                    print(f" {result['time']:.3f}s")
                else:
                    print(f" FAILED")
                    
            if times:
                self.results[f'shortcut_{cmd.split()[0]}'] = {
                    'command': cmd,
                    'description': description,
                    'times': times,
                    'mean': statistics.mean(times),
                    'median': statistics.median(times),
                    'stdev': statistics.stdev(times) if len(times) > 1 else 0
                }
        
        # Cleanup
        test_script.unlink(missing_ok=True)
        self.cleanup()
        
    def compare_with_legacy(self):
        """Compare with legacy run-docker.sh if available"""
        if not self.run_docker.exists():
            print("\n=== Legacy Comparison Skipped (run-docker.sh not found) ===")
            return
            
        print("\n=== Comparing with Legacy System ===")
        
        # Simple echo command comparison
        print("Comparing simple command execution...")
        
        # New system
        self.cleanup()
        subprocess.run([str(self.finn_docker), "init"], capture_output=True)
        time.sleep(2)
        
        new_times = []
        for i in range(3):
            result = self.run_command(f"{self.finn_docker} exec \"echo test\"")
            if result['success']:
                new_times.append(result['time'])
                
        # Legacy system (creates new container each time)
        self.cleanup()
        legacy_times = []
        for i in range(3):
            result = self.run_command(f"{self.run_docker} bash -c \"echo test\"")
            if result['success']:
                legacy_times.append(result['time'])
                
        if new_times and legacy_times:
            self.results['comparison'] = {
                'new_system': {
                    'mean': statistics.mean(new_times),
                    'median': statistics.median(new_times)
                },
                'legacy_system': {
                    'mean': statistics.mean(legacy_times),
                    'median': statistics.median(legacy_times)
                },
                'improvement': f"{(1 - statistics.mean(new_times) / statistics.mean(legacy_times)) * 100:.1f}%"
            }
            
        self.cleanup()
        
    def generate_report(self):
        """Generate benchmark report"""
        print("\n" + "="*60)
        print("FINN Docker Performance Benchmark Report")
        print("="*60)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Runs per test: {self.runs}")
        print()
        
        if 'cold_start' in self.results:
            r = self.results['cold_start']
            print("Cold Start Performance:")
            print(f"  Mean: {r['mean']:.2f}s")
            print(f"  Median: {r['median']:.2f}s")
            print(f"  StdDev: {r['stdev']:.2f}s")
            print()
            
        if 'warm_start' in self.results:
            r = self.results['warm_start']
            print("Warm Start Performance:")
            print(f"  Mean: {r['mean']:.2f}s")
            print(f"  Median: {r['median']:.2f}s")
            print(f"  StdDev: {r['stdev']:.2f}s")
            print()
            
        print("Command Execution Performance:")
        for key, value in self.results.items():
            if key.startswith('exec_'):
                print(f"  {value['description']}:")
                print(f"    Mean: {value['mean']*1000:.1f}ms")
                print(f"    Median: {value['median']*1000:.1f}ms")
                
        print()
        
        if 'comparison' in self.results:
            c = self.results['comparison']
            print("Legacy System Comparison:")
            print(f"  New system: {c['new_system']['mean']*1000:.1f}ms")
            print(f"  Legacy system: {c['legacy_system']['mean']*1000:.1f}ms")
            print(f"  Improvement: {c['improvement']}")
            
        # Save detailed results
        report_file = self.finn_root / "docker_benchmark_results.json"
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nDetailed results saved to: {report_file}")
        
    def run_all(self):
        """Run all benchmarks"""
        self.benchmark_cold_start()
        self.benchmark_warm_start()
        self.benchmark_command_execution()
        self.benchmark_shortcuts()
        self.compare_with_legacy()
        self.generate_report()
        

def main():
    parser = argparse.ArgumentParser(description="FINN Docker Performance Benchmark")
    parser.add_argument('-r', '--runs', type=int, default=5,
                        help='Number of runs per benchmark (default: 5)')
    parser.add_argument('-q', '--quick', action='store_true',
                        help='Quick benchmark (fewer runs)')
    parser.add_argument('--cold-only', action='store_true',
                        help='Only run cold start benchmark')
    parser.add_argument('--warm-only', action='store_true',
                        help='Only run warm start benchmark')
    parser.add_argument('--exec-only', action='store_true',
                        help='Only run execution benchmark')
    
    args = parser.parse_args()
    
    if args.quick:
        args.runs = 2
        
    benchmark = DockerBenchmark(runs=args.runs)
    
    try:
        if args.cold_only:
            benchmark.benchmark_cold_start()
        elif args.warm_only:
            benchmark.benchmark_warm_start()
        elif args.exec_only:
            benchmark.benchmark_command_execution()
        else:
            benchmark.run_all()
            
        if any([args.cold_only, args.warm_only, args.exec_only]):
            benchmark.generate_report()
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        benchmark.cleanup()
        sys.exit(1)
    except Exception as e:
        print(f"\nError during benchmark: {e}")
        benchmark.cleanup()
        sys.exit(1)
        

if __name__ == "__main__":
    main()