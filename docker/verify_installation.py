#!/usr/bin/env python3
"""
FINN Installation Verification
Comprehensive verification of all dependencies and tools
"""

import os
import sys
import subprocess
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class InstallationVerifier:
    def __init__(self):
        self.finn_root = Path(os.environ.get('FINN_ROOT', '/workspace/finn'))
        self.deps_dir = Path(os.environ.get('FINN_DEPS_DIR', self.finn_root / 'deps'))
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log with formatting"""
        colors = {
            "PASS": "\033[0;32m",   # Green
            "FAIL": "\033[0;31m",   # Red
            "WARN": "\033[0;33m",   # Yellow
            "INFO": "\033[0;34m",   # Blue
        }
        reset = "\033[0m"
        symbol = {
            "PASS": "✓",
            "FAIL": "✗",
            "WARN": "⚠",
            "INFO": "→"
        }.get(level, "•")
        
        color = colors.get(level, "")
        print(f"{color}{symbol} {message}{reset}")
        sys.stdout.flush()
        
    def check_command(self, name: str, command: List[str], 
                     expected_in_output: Optional[str] = None) -> bool:
        """Check if a command runs successfully"""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            success = result.returncode == 0
            if success and expected_in_output:
                success = expected_in_output in result.stdout
                
            self.record_result(name, success, result.stdout if success else result.stderr)
            return success
            
        except subprocess.TimeoutExpired:
            self.record_result(name, False, "Command timed out")
            return False
        except Exception as e:
            self.record_result(name, False, str(e))
            return False
            
    def check_python_import(self, name: str, module_name: str, 
                          version_attr: Optional[str] = None) -> bool:
        """Check if a Python module can be imported"""
        try:
            module = importlib.import_module(module_name)
            
            version = "unknown"
            if version_attr:
                version = getattr(module, version_attr, "unknown")
            elif hasattr(module, '__version__'):
                version = module.__version__
                
            self.record_result(name, True, f"Version: {version}")
            return True
            
        except ImportError as e:
            self.record_result(name, False, str(e))
            return False
        except Exception as e:
            self.record_result(name, False, f"Error: {e}")
            return False
            
    def check_directory(self, name: str, path: Path, 
                       check_git: bool = False) -> bool:
        """Check if a directory exists and optionally if it's a git repo"""
        if not path.exists():
            self.record_result(name, False, f"Directory not found: {path}")
            return False
            
        if check_git:
            git_dir = path / ".git"
            if not git_dir.exists():
                self.record_result(name, False, "Not a git repository")
                return False
                
            # Get current commit
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=path,
                    capture_output=True,
                    text=True
                )
                commit = result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
                self.record_result(name, True, f"Git repo at commit: {commit}")
                return True
            except:
                self.record_result(name, False, "Failed to get git status")
                return False
        else:
            self.record_result(name, True, f"Directory exists: {path}")
            return True
            
    def check_file(self, name: str, path: Path, 
                   check_executable: bool = False) -> bool:
        """Check if a file exists and optionally if it's executable"""
        if not path.exists():
            self.record_result(name, False, f"File not found: {path}")
            return False
            
        if check_executable and not os.access(path, os.X_OK):
            self.record_result(name, False, "File not executable")
            return False
            
        self.record_result(name, True, f"File exists: {path}")
        return True
        
    def record_result(self, name: str, success: bool, details: str = ""):
        """Record verification result"""
        self.results['checks'][name] = {
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['summary']['total'] += 1
        if success:
            self.results['summary']['passed'] += 1
            self.log(f"{name}", "PASS")
        else:
            self.results['summary']['failed'] += 1
            self.log(f"{name} - {details}", "FAIL")
            
        if details:
            self.log(f"  {details}", "INFO")
            
    def verify_core_dependencies(self):
        """Verify core system dependencies"""
        self.log("Verifying core dependencies...", "INFO")
        
        # Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.record_result("Python version", True, f"Python {python_version}")
        
        # Git
        self.check_command("Git", ["git", "--version"])
        
        # GCC
        self.check_command("GCC", ["gcc", "--version"])
        
        # Make
        self.check_command("Make", ["make", "--version"])
        
    def verify_python_packages(self):
        """Verify Python package installations"""
        self.log("\nVerifying Python packages...", "INFO")
        
        # Core packages
        packages = [
            ("numpy", "numpy", "__version__"),
            ("torch", "torch", "__version__"),
            ("onnx", "onnx", "__version__"),
            ("onnxruntime", "onnxruntime", "__version__"),
            ("pytest", "pytest", "__version__"),
            ("jupyter", "jupyter", None),
            ("netron", "netron", "__version__"),
        ]
        
        for name, module, version_attr in packages:
            self.check_python_import(name, module, version_attr)
            
        # FINN dependencies
        finn_packages = [
            ("qonnx", "qonnx"),
            ("finn_experimental", "finn_experimental"),
            ("brevitas", "brevitas"),
        ]
        
        for name, module in finn_packages:
            self.check_python_import(name, module)
            
    def verify_repositories(self):
        """Verify git repository checkouts"""
        self.log("\nVerifying repository checkouts...", "INFO")
        
        repos = [
            "qonnx",
            "finn-experimental", 
            "brevitas",
            "finn-hlslib",
            "cnpy",
            "oh-my-xilinx",
        ]
        
        for repo in repos:
            repo_path = self.deps_dir / repo
            self.check_directory(f"Repository: {repo}", repo_path, check_git=True)
            
    def verify_xilinx_tools(self):
        """Verify Xilinx tool installations"""
        self.log("\nVerifying Xilinx tools...", "INFO")
        
        xilinx_path = os.environ.get('FINN_XILINX_PATH', '/opt/Xilinx')
        xilinx_version = os.environ.get('FINN_XILINX_VERSION', '2022.2')
        
        if not Path(xilinx_path).exists():
            self.record_result("Xilinx tools", False, 
                             f"FINN_XILINX_PATH not found: {xilinx_path}")
            return
            
        # Check Vivado
        vivado_path = Path(xilinx_path) / "Vivado" / xilinx_version
        if vivado_path.exists():
            self.record_result("Vivado", True, f"Found at {vivado_path}")
        else:
            self.record_result("Vivado", False, f"Not found at {vivado_path}")
            
        # Check Vitis HLS
        hls_path = Path(xilinx_path) / "Vitis_HLS" / xilinx_version
        if hls_path.exists():
            self.record_result("Vitis HLS", True, f"Found at {hls_path}")
        else:
            self.record_result("Vitis HLS", False, f"Not found at {hls_path}")
            
        # Check Vitis
        vitis_path = Path(xilinx_path) / "Vitis" / xilinx_version
        if vitis_path.exists():
            self.record_result("Vitis", True, f"Found at {vitis_path}")
        else:
            self.record_result("Vitis", False, f"Not found at {vitis_path}")
            
    def verify_environment(self):
        """Verify environment variables"""
        self.log("\nVerifying environment variables...", "INFO")
        
        required_vars = [
            "FINN_ROOT",
            "FINN_BUILD_DIR", 
            "FINN_DEPS_DIR",
        ]
        
        optional_vars = [
            "FINN_XILINX_PATH",
            "FINN_XILINX_VERSION",
            "VIVADO_PATH",
            "HLS_PATH",
            "VITIS_PATH",
            "PLATFORM_REPO_PATHS",
        ]
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                self.record_result(f"ENV: {var}", True, value)
            else:
                self.record_result(f"ENV: {var}", False, "Not set")
                
        for var in optional_vars:
            value = os.environ.get(var)
            if value:
                self.record_result(f"ENV: {var} (optional)", True, value)
                
    def verify_build_system(self):
        """Verify FINN build system"""
        self.log("\nVerifying FINN build system...", "INFO")
        
        # Check if we can import finn
        try:
            sys.path.insert(0, str(self.finn_root / "src"))
            import finn
            self.record_result("FINN import", True, "FINN module imported successfully")
        except Exception as e:
            self.record_result("FINN import", False, str(e))
            
        # Check build directories
        build_dir = Path(os.environ.get('FINN_BUILD_DIR', '/tmp/finn_build'))
        self.check_directory("Build directory", build_dir)
        
        # Check cache directories
        cache_dir = Path("/tmp/.finn_cache")
        self.check_directory("Cache directory", cache_dir)
        
    def generate_report(self) -> str:
        """Generate verification report"""
        report = [
            "\n" + "="*60,
            "FINN Installation Verification Report",
            "="*60,
            f"Timestamp: {self.results['timestamp']}",
            f"Total checks: {self.results['summary']['total']}",
            f"Passed: {self.results['summary']['passed']}",
            f"Failed: {self.results['summary']['failed']}",
            f"Success rate: {self.results['summary']['passed'] / self.results['summary']['total'] * 100:.1f}%",
            "="*60,
        ]
        
        if self.results['summary']['failed'] > 0:
            report.append("\nFailed checks:")
            for name, result in self.results['checks'].items():
                if not result['success']:
                    report.append(f"  - {name}: {result['details']}")
                    
        # Save detailed report
        report_file = Path("/tmp/.finn_cache/verification_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        report.append(f"\nDetailed report saved to: {report_file}")
        
        return "\n".join(report)
        
    def run_verification(self) -> bool:
        """Run all verification checks"""
        self.log("Starting FINN installation verification...", "INFO")
        
        self.verify_core_dependencies()
        self.verify_python_packages()
        self.verify_repositories()
        self.verify_xilinx_tools()
        self.verify_environment()
        self.verify_build_system()
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        # Return success if all critical checks passed
        critical_failures = [
            "Python version", "Git", "GCC", "numpy", "torch", 
            "onnx", "qonnx", "finn_experimental", "FINN import"
        ]
        
        for check in critical_failures:
            if check in self.results['checks'] and not self.results['checks'][check]['success']:
                return False
                
        return True
        
def main():
    verifier = InstallationVerifier()
    success = verifier.run_verification()
    
    if success:
        print("\n✓ FINN installation verification completed successfully!")
        return 0
    else:
        print("\n✗ FINN installation verification failed!")
        print("Please check the failed items above and run setup again.")
        return 1
        
if __name__ == "__main__":
    sys.exit(main())