#!/usr/bin/env python3
"""
FINN Package Installation Manager
Smart caching and parallel installation of Python packages
"""

import os
import sys
import json
import subprocess
import time
import hashlib
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class PackageInstaller:
    def __init__(self):
        self.cache_dir = Path("/tmp/.finn_cache/packages")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "installed.json"
        self.cache_data = self.load_cache()
        
    def emit_status(self, status: str):
        """Emit status for container monitoring"""
        print(f"FINN_STATUS:{status}")
        sys.stdout.flush()
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp and level"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        color = {
            "INFO": "\033[0;32m",    # Green
            "WARN": "\033[0;33m",    # Yellow  
            "ERROR": "\033[0;31m",   # Red
        }.get(level, "")
        reset = "\033[0m"
        print(f"{color}[{timestamp}] {level}: {message}{reset}")
        sys.stdout.flush()
        
    def load_cache(self) -> Dict:
        """Load cache data from file"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
        
    def save_cache(self):
        """Save cache data to file"""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache_data, f, indent=2)
            
    def get_package_hash(self, package_path: Path) -> str:
        """Get hash of package source for cache validation"""
        if not package_path.exists():
            return ""
            
        # Get git commit if it's a git repo
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=package_path,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
            
        # Fallback to directory content hash
        content = ""
        for py_file in sorted(package_path.rglob("*.py"))[:50]:  # Sample first 50 files
            try:
                content += py_file.read_text()
            except:
                pass
        return hashlib.md5(content.encode()).hexdigest()[:16]
        
    def is_package_cached(self, name: str, package_path: Path) -> bool:
        """Check if package is already installed and up to date"""
        if name not in self.cache_data:
            return False
            
        cached = self.cache_data[name]
        current_hash = self.get_package_hash(package_path)
        
        # Check if hash matches
        if cached.get('hash') != current_hash:
            return False
            
        # Check if not too old (7 days)
        installed_time = datetime.fromisoformat(cached.get('timestamp', '2000-01-01'))
        if datetime.now() - installed_time > timedelta(days=7):
            return False
            
        # Verify package is actually installed
        try:
            result = subprocess.run(
                [sys.executable, "-c", f"import {name}"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except:
            return False
            
    def install_package(self, name: str, package_path: Path, 
                       editable: bool = True, deps: bool = False) -> bool:
        """Install a single package with caching"""
        
        # Normalize name for import checking
        import_name = name.replace('-', '_').replace('finn_experimental', 'finn_experimental')
        
        # Check cache first
        if self.is_package_cached(import_name, package_path):
            self.log(f"✓ {name} already installed (cached)", "INFO")
            return True
            
        self.emit_status(f"INSTALLING_PACKAGES:{name}")
        self.log(f"Installing {name} from {package_path}", "INFO")
        
        # Build pip command
        cmd = [sys.executable, "-m", "pip", "install"]
        if editable:
            cmd.append("-e")
        if not deps:
            cmd.append("--no-deps")
        cmd.append(str(package_path))
        
        # Run installation
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                # Update cache
                self.cache_data[import_name] = {
                    'name': name,
                    'path': str(package_path),
                    'hash': self.get_package_hash(package_path),
                    'timestamp': datetime.now().isoformat(),
                    'editable': editable
                }
                self.save_cache()
                self.log(f"✓ {name} installed successfully", "INFO")
                return True
            else:
                self.log(f"Failed to install {name}: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"Installation of {name} timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error installing {name}: {e}", "ERROR")
            return False
            
    def install_requirements(self, requirements_file: Path) -> bool:
        """Install packages from requirements.txt"""
        if not requirements_file.exists():
            self.log(f"Requirements file not found: {requirements_file}", "WARN")
            return True
            
        self.emit_status("INSTALLING_PACKAGES:requirements")
        self.log(f"Installing requirements from {requirements_file}", "INFO")
        
        # Check if requirements haven't changed
        req_hash = hashlib.md5(requirements_file.read_bytes()).hexdigest()
        if self.cache_data.get('requirements_hash') == req_hash:
            self.log("Requirements already installed (cached)", "INFO")
            return True
            
        # Install requirements
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            if result.returncode == 0:
                self.cache_data['requirements_hash'] = req_hash
                self.cache_data['requirements_timestamp'] = datetime.now().isoformat()
                self.save_cache()
                self.log("✓ Requirements installed successfully", "INFO")
                return True
            else:
                self.log(f"Failed to install requirements: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.log("Requirements installation timed out", "ERROR")
            return False
        except Exception as e:
            self.log(f"Error installing requirements: {e}", "ERROR")
            return False
            
    def install_all_parallel(self, packages: List[Tuple[str, Path]], 
                           max_workers: int = 4) -> Tuple[int, int]:
        """Install multiple packages in parallel"""
        total = len(packages)
        success_count = 0
        failed = []
        
        self.log(f"Installing {total} packages with {max_workers} workers", "INFO")
        
        # First pass: check what's already cached
        to_install = []
        for name, path in packages:
            import_name = name.replace('-', '_').replace('finn_experimental', 'finn_experimental')
            if self.is_package_cached(import_name, path):
                self.log(f"✓ {name} already installed (cached)", "INFO")
                success_count += 1
            else:
                to_install.append((name, path))
                
        if not to_install:
            self.log("All packages already installed!", "INFO")
            return success_count, 0
            
        # Install remaining packages
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_package = {
                executor.submit(self.install_package, name, path): name 
                for name, path in to_install
            }
            
            for future in concurrent.futures.as_completed(future_to_package):
                name = future_to_package[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed.append(name)
                except Exception as e:
                    self.log(f"Exception installing {name}: {e}", "ERROR")
                    failed.append(name)
                    
                # Show progress
                completed = success_count + len(failed)
                progress = completed / total * 100
                self.log(f"Progress: {progress:.0f}% ({completed}/{total})", "INFO")
                
        return success_count, len(failed)
        
    def clean_cache(self, max_age_days: int = 30):
        """Clean old cache entries"""
        self.log(f"Cleaning cache entries older than {max_age_days} days", "INFO")
        
        now = datetime.now()
        cleaned = 0
        
        for key in list(self.cache_data.keys()):
            if key in ['requirements_hash', 'requirements_timestamp']:
                continue
                
            entry = self.cache_data[key]
            if 'timestamp' in entry:
                timestamp = datetime.fromisoformat(entry['timestamp'])
                if now - timestamp > timedelta(days=max_age_days):
                    del self.cache_data[key]
                    cleaned += 1
                    
        if cleaned > 0:
            self.save_cache()
            self.log(f"Cleaned {cleaned} old cache entries", "INFO")
            
def main():
    # Get environment variables
    finn_root = os.environ.get('FINN_ROOT', '/workspace/finn')
    deps_dir = Path(os.environ.get('FINN_DEPS_DIR', os.path.join(finn_root, 'deps')))
    max_workers = int(os.environ.get('FINN_INSTALL_WORKERS', '4'))
    
    installer = PackageInstaller()
    
    # Clean old cache entries
    installer.clean_cache()
    
    # Define packages to install
    packages = [
        ("qonnx", deps_dir / "qonnx"),
        ("finn_experimental", deps_dir / "finn-experimental"),
        ("brevitas", deps_dir / "brevitas"),
    ]
    
    # Filter out non-existent packages
    packages = [(name, path) for name, path in packages if path.exists()]
    
    if not packages:
        installer.log("No packages to install", "WARN")
        return 0
        
    # Install packages
    success, failed = installer.install_all_parallel(packages, max_workers=max_workers)
    
    if failed == 0:
        installer.log(f"All {success} packages installed successfully!", "INFO")
        return 0
    else:
        installer.log(f"Failed to install {failed} packages", "ERROR")
        return 1
        
if __name__ == "__main__":
    sys.exit(main())