#!/usr/bin/env python3
"""
FINN Container Health Check
Comprehensive health monitoring for FINN containers
"""

import os
import sys
import json
import psutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

class HealthChecker:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'checks': {},
            'metrics': {},
            'issues': [],
            'recommendations': []
        }
        
    def check_system_resources(self) -> Dict:
        """Check system resource usage"""
        metrics = {}
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics['cpu_usage'] = cpu_percent
        if cpu_percent > 80:
            self.add_issue("High CPU usage", f"CPU at {cpu_percent}%", "warning")
            
        # Memory usage
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = memory.percent
        metrics['memory_available_gb'] = memory.available / (1024**3)
        if memory.percent > 85:
            self.add_issue("High memory usage", f"Memory at {memory.percent}%", "warning")
        if memory.available < 2 * (1024**3):  # Less than 2GB
            self.add_issue("Low memory", f"Only {memory.available/(1024**3):.1f}GB available", "error")
            
        # Disk usage
        disk = psutil.disk_usage('/')
        metrics['disk_usage'] = disk.percent
        metrics['disk_free_gb'] = disk.free / (1024**3)
        if disk.percent > 90:
            self.add_issue("Low disk space", f"Disk at {disk.percent}%", "error")
            
        self.results['metrics']['system'] = metrics
        return metrics
        
    def check_finn_environment(self) -> bool:
        """Check FINN environment variables and paths"""
        checks = {}
        all_good = True
        
        # Required environment variables
        required_vars = {
            'FINN_ROOT': 'FINN root directory',
            'FINN_BUILD_DIR': 'Build directory',
            'PYTHONPATH': 'Python path'
        }
        
        for var, desc in required_vars.items():
            value = os.environ.get(var)
            if value:
                if var.endswith('_DIR') or var.endswith('_ROOT'):
                    # Check if directory exists
                    if os.path.exists(value):
                        checks[var] = {'status': 'ok', 'value': value}
                    else:
                        checks[var] = {'status': 'error', 'value': value, 'issue': 'Directory not found'}
                        self.add_issue(f"Missing directory", f"{desc} not found: {value}", "error")
                        all_good = False
                else:
                    checks[var] = {'status': 'ok', 'value': value}
            else:
                checks[var] = {'status': 'error', 'issue': 'Not set'}
                self.add_issue(f"Missing environment variable", f"{var} ({desc}) not set", "error")
                all_good = False
                
        # Optional but recommended variables
        optional_vars = {
            'FINN_XILINX_PATH': 'Xilinx tools path',
            'VIVADO_PATH': 'Vivado path',
            'NUM_DEFAULT_WORKERS': 'Worker threads'
        }
        
        for var, desc in optional_vars.items():
            value = os.environ.get(var)
            if value:
                checks[var] = {'status': 'ok', 'value': value}
            else:
                checks[var] = {'status': 'warning', 'issue': 'Not set'}
                
        self.results['checks']['environment'] = checks
        return all_good
        
    def check_python_packages(self) -> bool:
        """Check critical Python packages"""
        packages = {}
        all_good = True
        
        critical_packages = [
            'numpy', 'torch', 'onnx', 'qonnx', 
            'finn_experimental', 'brevitas'
        ]
        
        for package in critical_packages:
            try:
                module = __import__(package.replace('-', '_'))
                version = getattr(module, '__version__', 'unknown')
                packages[package] = {'status': 'ok', 'version': version}
            except ImportError:
                packages[package] = {'status': 'error', 'issue': 'Not installed'}
                self.add_issue(f"Missing package", f"Python package '{package}' not installed", "error")
                all_good = False
                
        self.results['checks']['packages'] = packages
        return all_good
        
    def check_processes(self) -> Dict:
        """Check for problematic processes"""
        process_info = {
            'total_processes': len(psutil.pids()),
            'zombie_processes': 0,
            'high_cpu_processes': [],
            'xilinx_processes': []
        }
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                info = proc.info
                
                # Check for zombie processes
                if info['status'] == psutil.STATUS_ZOMBIE:
                    process_info['zombie_processes'] += 1
                    
                # Check for high CPU processes
                if info['cpu_percent'] > 50:
                    process_info['high_cpu_processes'].append({
                        'name': info['name'],
                        'pid': info['pid'],
                        'cpu': info['cpu_percent']
                    })
                    
                # Check for Xilinx processes
                if any(x in info['name'].lower() for x in ['vivado', 'vitis', 'xsct']):
                    process_info['xilinx_processes'].append({
                        'name': info['name'],
                        'pid': info['pid']
                    })
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        if process_info['zombie_processes'] > 0:
            self.add_issue("Zombie processes", f"{process_info['zombie_processes']} zombie processes found", "warning")
            
        self.results['metrics']['processes'] = process_info
        return process_info
        
    def check_docker_specific(self) -> bool:
        """Check Docker-specific health indicators"""
        checks = {}
        all_good = True
        
        # Check if we're in a container
        if os.path.exists('/.dockerenv'):
            checks['in_container'] = {'status': 'ok'}
        else:
            checks['in_container'] = {'status': 'warning', 'issue': 'Not in container?'}
            
        # Check container limits
        try:
            # Memory limit
            with open('/sys/fs/cgroup/memory/memory.limit_in_bytes', 'r') as f:
                mem_limit = int(f.read().strip())
                if mem_limit < (1 << 62):  # Less than max value
                    mem_limit_gb = mem_limit / (1024**3)
                    checks['memory_limit'] = {'status': 'ok', 'value': f"{mem_limit_gb:.1f}GB"}
                    if mem_limit_gb < 4:
                        self.add_issue("Low memory limit", f"Container limited to {mem_limit_gb:.1f}GB", "warning")
        except:
            pass
            
        self.results['checks']['docker'] = checks
        return all_good
        
    def check_cache_health(self) -> Dict:
        """Check cache directory health"""
        cache_info = {
            'cache_dir': '/tmp/.finn_cache',
            'exists': False,
            'size_mb': 0,
            'file_count': 0,
            'oldest_file_days': 0
        }
        
        cache_dir = Path('/tmp/.finn_cache')
        if cache_dir.exists():
            cache_info['exists'] = True
            
            # Calculate cache size
            total_size = 0
            file_count = 0
            oldest_time = datetime.now()
            
            for path in cache_dir.rglob('*'):
                if path.is_file():
                    file_count += 1
                    total_size += path.stat().st_size
                    mtime = datetime.fromtimestamp(path.stat().st_mtime)
                    if mtime < oldest_time:
                        oldest_time = mtime
                        
            cache_info['size_mb'] = total_size / (1024**2)
            cache_info['file_count'] = file_count
            cache_info['oldest_file_days'] = (datetime.now() - oldest_time).days
            
            # Check for issues
            if cache_info['size_mb'] > 1000:  # More than 1GB
                self.add_issue("Large cache", f"Cache using {cache_info['size_mb']:.0f}MB", "warning")
                self.add_recommendation("Consider cleaning cache with: rm -rf /tmp/.finn_cache/*")
                
            if cache_info['oldest_file_days'] > 30:
                self.add_issue("Old cache files", f"Cache contains files older than {cache_info['oldest_file_days']} days", "info")
                
        self.results['metrics']['cache'] = cache_info
        return cache_info
        
    def add_issue(self, title: str, description: str, severity: str = "warning"):
        """Add an issue to the results"""
        self.results['issues'].append({
            'title': title,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update overall status
        if severity == "error":
            self.results['overall_status'] = "unhealthy"
        elif severity == "warning" and self.results['overall_status'] == "healthy":
            self.results['overall_status'] = "degraded"
            
    def add_recommendation(self, recommendation: str):
        """Add a recommendation"""
        self.results['recommendations'].append(recommendation)
        
    def run_health_check(self) -> bool:
        """Run all health checks"""
        print("Running FINN container health check...")
        
        # System resources
        print("→ Checking system resources...")
        self.check_system_resources()
        
        # FINN environment
        print("→ Checking FINN environment...")
        env_ok = self.check_finn_environment()
        
        # Python packages
        print("→ Checking Python packages...")
        pkg_ok = self.check_python_packages()
        
        # Processes
        print("→ Checking processes...")
        self.check_processes()
        
        # Docker-specific
        print("→ Checking Docker environment...")
        self.check_docker_specific()
        
        # Cache health
        print("→ Checking cache health...")
        self.check_cache_health()
        
        # Generate recommendations based on issues
        if self.results['overall_status'] == "unhealthy":
            self.add_recommendation("Critical issues detected. Check error messages above.")
        elif self.results['overall_status'] == "degraded":
            self.add_recommendation("Some issues detected. Container may not perform optimally.")
            
        return self.results['overall_status'] != "unhealthy"
        
    def get_summary(self) -> str:
        """Get a summary of the health check"""
        status_emoji = {
            'healthy': '✓',
            'degraded': '⚠',
            'unhealthy': '✗'
        }
        
        status = self.results['overall_status']
        emoji = status_emoji.get(status, '?')
        
        summary = [
            f"\n{emoji} Overall Status: {status.upper()}",
            f"Issues: {len(self.results['issues'])}",
            f"CPU: {self.results['metrics']['system']['cpu_usage']:.1f}%",
            f"Memory: {self.results['metrics']['system']['memory_usage']:.1f}%",
            f"Disk: {self.results['metrics']['system']['disk_usage']:.1f}%"
        ]
        
        if self.results['issues']:
            summary.append("\nIssues:")
            for issue in self.results['issues'][:5]:  # Show first 5
                summary.append(f"  • [{issue['severity']}] {issue['title']}: {issue['description']}")
                
        if self.results['recommendations']:
            summary.append("\nRecommendations:")
            for rec in self.results['recommendations']:
                summary.append(f"  • {rec}")
                
        return "\n".join(summary)
        
    def save_results(self, output_path: Optional[Path] = None):
        """Save detailed results to JSON"""
        if output_path is None:
            output_path = Path("/tmp/.finn_cache/health_check.json")
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        return output_path
        

def main():
    checker = HealthChecker()
    
    # Run health check
    is_healthy = checker.run_health_check()
    
    # Save detailed results
    output_path = checker.save_results()
    
    # Print summary
    print(checker.get_summary())
    print(f"\nDetailed results saved to: {output_path}")
    
    # Exit with appropriate code
    if is_healthy:
        return 0
    else:
        return 1
        

if __name__ == "__main__":
    sys.exit(main())