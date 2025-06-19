#!/usr/bin/env python3
"""
FINN Container Startup Profiler
Measures and optimizes container initialization time
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable
from contextlib import contextmanager

class StartupProfiler:
    def __init__(self):
        self.profile_data = {
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'total_time': 0,
            'environment': {
                'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                'cpu_count': os.cpu_count(),
                'finn_root': os.environ.get('FINN_ROOT', 'unknown'),
                'skip_deps': os.environ.get('FINN_SKIP_DEP_REPOS', '0'),
            }
        }
        self.start_time = time.time()
        self.last_step_time = self.start_time
        
    @contextmanager
    def measure_step(self, name: str, emit_status: bool = True):
        """Context manager to measure execution time of a step"""
        step_start = time.time()
        
        if emit_status:
            print(f"FINN_STATUS:PROFILING:{name}")
            
        step_data = {
            'name': name,
            'start_time': datetime.now().isoformat(),
            'duration': 0,
            'success': True,
            'error': None
        }
        
        try:
            yield step_data
        except Exception as e:
            step_data['success'] = False
            step_data['error'] = str(e)
            raise
        finally:
            duration = time.time() - step_start
            step_data['duration'] = duration
            self.profile_data['steps'].append(step_data)
            
            # Print timing info
            print(f"[PROFILE] {name}: {duration:.2f}s")
            
    def add_measurement(self, name: str, duration: float, metadata: Optional[Dict] = None):
        """Add a manual measurement"""
        step_data = {
            'name': name,
            'duration': duration,
            'success': True,
            'metadata': metadata or {}
        }
        self.profile_data['steps'].append(step_data)
        
    def get_bottlenecks(self, threshold: float = 1.0) -> List[Dict]:
        """Identify steps that take longer than threshold seconds"""
        return [
            step for step in self.profile_data['steps'] 
            if step['duration'] > threshold
        ]
        
    def get_optimization_suggestions(self) -> List[str]:
        """Generate optimization suggestions based on profile data"""
        suggestions = []
        
        # Check total time
        total_time = sum(step['duration'] for step in self.profile_data['steps'])
        if total_time > 30:
            suggestions.append(f"Total startup time ({total_time:.1f}s) exceeds 30s target")
            
        # Check for slow steps
        bottlenecks = self.get_bottlenecks(2.0)
        for step in bottlenecks:
            suggestions.append(f"Step '{step['name']}' took {step['duration']:.1f}s - consider optimization")
            
        # Check specific patterns
        dep_steps = [s for s in self.profile_data['steps'] if 'dependency' in s['name'].lower()]
        if dep_steps:
            dep_time = sum(s['duration'] for s in dep_steps)
            if dep_time > 10:
                suggestions.append(f"Dependency operations took {dep_time:.1f}s - ensure caching is working")
                
        pkg_steps = [s for s in self.profile_data['steps'] if 'package' in s['name'].lower()]
        if pkg_steps:
            pkg_time = sum(s['duration'] for s in pkg_steps)
            if pkg_time > 5:
                suggestions.append(f"Package installations took {pkg_time:.1f}s - check cache validity")
                
        return suggestions
        
    def save_profile(self, output_path: Optional[Path] = None):
        """Save profile data to JSON file"""
        if output_path is None:
            output_path = Path("/tmp/.finn_cache/startup_profile.json")
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Calculate total time
        self.profile_data['total_time'] = time.time() - self.start_time
        self.profile_data['end_time'] = datetime.now().isoformat()
        
        # Add summary
        self.profile_data['summary'] = {
            'total_steps': len(self.profile_data['steps']),
            'successful_steps': sum(1 for s in self.profile_data['steps'] if s.get('success', True)),
            'failed_steps': sum(1 for s in self.profile_data['steps'] if not s.get('success', True)),
            'bottlenecks': [s['name'] for s in self.get_bottlenecks()],
            'suggestions': self.get_optimization_suggestions()
        }
        
        with open(output_path, 'w') as f:
            json.dump(self.profile_data, f, indent=2)
            
        print(f"\n[PROFILE] Profile saved to: {output_path}")
        
    def print_summary(self):
        """Print a summary of the profiling results"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("FINN Container Startup Profile Summary")
        print("="*60)
        print(f"Total startup time: {total_time:.2f}s")
        print(f"Number of steps: {len(self.profile_data['steps'])}")
        
        # Show top 5 slowest steps
        sorted_steps = sorted(
            self.profile_data['steps'], 
            key=lambda x: x['duration'], 
            reverse=True
        )[:5]
        
        if sorted_steps:
            print("\nSlowest steps:")
            for i, step in enumerate(sorted_steps, 1):
                print(f"  {i}. {step['name']}: {step['duration']:.2f}s")
                
        # Show suggestions
        suggestions = self.get_optimization_suggestions()
        if suggestions:
            print("\nOptimization suggestions:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")
        else:
            print("\n✓ Startup time is within acceptable limits")
            
        print("="*60)


# Optimized startup functions
class OptimizedStartup:
    """Optimized startup routines for common operations"""
    
    @staticmethod
    def parallel_import_check() -> Dict[str, bool]:
        """Check multiple Python imports in parallel"""
        import concurrent.futures
        
        modules = [
            'numpy', 'torch', 'onnx', 'qonnx', 
            'finn_experimental', 'brevitas', 'pytest'
        ]
        
        def check_import(module):
            try:
                __import__(module)
                return module, True
            except ImportError:
                return module, False
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            results = dict(executor.map(check_import, modules))
            
        return results
        
    @staticmethod
    def lazy_xilinx_setup():
        """Lazy setup of Xilinx tools - only source when needed"""
        xilinx_setup_done = False
        
        def setup_if_needed():
            nonlocal xilinx_setup_done
            if xilinx_setup_done:
                return
                
            vivado_path = os.environ.get('VIVADO_PATH')
            if vivado_path and os.path.exists(f"{vivado_path}/settings64.sh"):
                # This is expensive, only do it when actually needed
                subprocess.run(
                    f"source {vivado_path}/settings64.sh", 
                    shell=True, 
                    executable='/bin/bash'
                )
                xilinx_setup_done = True
                
        return setup_if_needed
        
    @staticmethod
    def cached_git_status(repo_path: Path) -> Optional[str]:
        """Get git status with caching"""
        cache_file = Path("/tmp/.finn_cache/git_status_cache.json")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load cache
        cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except:
                pass
                
        # Check cache
        repo_key = str(repo_path)
        if repo_key in cache:
            cached_data = cache[repo_key]
            # Check if cache is recent (within 5 minutes)
            cache_time = datetime.fromisoformat(cached_data['timestamp'])
            if (datetime.now() - cache_time).seconds < 300:
                return cached_data['commit']
                
        # Get fresh status
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                commit = result.stdout.strip()
                # Update cache
                cache[repo_key] = {
                    'commit': commit,
                    'timestamp': datetime.now().isoformat()
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache, f)
                return commit
        except:
            pass
            
        return None


def profile_startup():
    """Profile the current startup process"""
    profiler = StartupProfiler()
    
    # Simulate startup steps for testing
    with profiler.measure_step("Environment setup"):
        time.sleep(0.1)  # Simulate work
        
    with profiler.measure_step("Dependency checking"):
        results = OptimizedStartup.parallel_import_check()
        
    with profiler.measure_step("Git repository validation"):
        # Simulate checking multiple repos
        for i in range(5):
            time.sleep(0.05)
            
    # Save and display results
    profiler.save_profile()
    profiler.print_summary()
    

if __name__ == "__main__":
    profile_startup()