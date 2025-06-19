#!/usr/bin/env python3
"""
FINN Dependency Fetcher
Enhanced dependency management with progress tracking and parallel downloads
"""

import os
import sys
import yaml
import json
import subprocess
import time
import hashlib
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class DependencyFetcher:
    def __init__(self, deps_dir: str, config_file: str):
        self.deps_dir = Path(deps_dir)
        self.config_file = config_file
        self.cache_dir = Path("/tmp/.finn_cache/repos")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.deps_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.repositories = self.config.get('repositories', {})
        self.board_files = self.config.get('board_files', [])
        
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
        
    def get_repo_hash(self, repo_info: Dict) -> str:
        """Generate hash for repository configuration"""
        content = f"{repo_info['url']}:{repo_info['commit']}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
        
    def is_repo_cached(self, name: str, repo_info: Dict) -> bool:
        """Check if repository is already cached and up to date"""
        cache_file = self.cache_dir / f"{name}.json"
        if not cache_file.exists():
            return False
            
        with open(cache_file, 'r') as f:
            cached = json.load(f)
            
        return (cached.get('commit') == repo_info['commit'] and 
                cached.get('hash') == self.get_repo_hash(repo_info))
                
    def save_cache_info(self, name: str, repo_info: Dict):
        """Save repository cache information"""
        cache_file = self.cache_dir / f"{name}.json"
        cache_data = {
            'commit': repo_info['commit'],
            'hash': self.get_repo_hash(repo_info),
            'timestamp': datetime.now().isoformat(),
            'url': repo_info['url']
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
            
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, 
                   max_retries: int = 3, retry_delay: int = 2) -> Tuple[bool, str]:
        """Run command with retry logic"""
        for attempt in range(max_retries):
            try:
                result = subprocess.run(
                    cmd, 
                    cwd=cwd, 
                    capture_output=True, 
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                if result.returncode == 0:
                    return True, result.stdout
                    
                error_msg = result.stderr or result.stdout
                if attempt < max_retries - 1:
                    self.log(f"Command failed (attempt {attempt + 1}/{max_retries}): {error_msg}", "WARN")
                    time.sleep(retry_delay)
                else:
                    return False, error_msg
                    
            except subprocess.TimeoutExpired:
                self.log(f"Command timed out (attempt {attempt + 1}/{max_retries})", "WARN")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    return False, "Command timed out"
            except Exception as e:
                return False, str(e)
                
        return False, "Max retries exceeded"
        
    def fetch_repository(self, name: str, repo_info: Dict) -> bool:
        """Fetch a single repository"""
        target_dir = self.deps_dir / name
        
        # Check if already exists at correct commit
        if target_dir.exists():
            success, output = self.run_command(
                ["git", "rev-parse", "HEAD"], 
                cwd=target_dir,
                max_retries=1
            )
            if success and output.strip() == repo_info['commit']:
                if self.is_repo_cached(name, repo_info):
                    self.log(f"✓ {name} already up to date (cached)", "INFO")
                    return True
                    
        self.emit_status(f"FETCHING_DEPENDENCIES:{name}")
        self.log(f"Fetching {name} from {repo_info['url']}", "INFO")
        
        # Remove existing directory if it exists
        if target_dir.exists():
            self.log(f"Removing existing {name} directory", "WARN")
            subprocess.run(["rm", "-rf", str(target_dir)], check=False)
            
        # Clone repository
        success, error = self.run_command([
            "git", "clone", 
            "--quiet",
            repo_info['url'], 
            str(target_dir)
        ])
        
        if not success:
            self.log(f"Failed to clone {name}: {error}", "ERROR")
            return False
            
        # Checkout specific commit
        success, error = self.run_command(
            ["git", "checkout", "--quiet", repo_info['commit']], 
            cwd=target_dir
        )
        
        if not success:
            self.log(f"Failed to checkout commit for {name}: {error}", "ERROR")
            subprocess.run(["rm", "-rf", str(target_dir)], check=False)
            return False
            
        # Save cache info
        self.save_cache_info(name, repo_info)
        self.log(f"✓ {name} fetched successfully", "INFO")
        return True
        
    def fetch_sparse_repository(self, name: str, board_info: Dict) -> bool:
        """Fetch repository with sparse checkout"""
        target_dir = self.deps_dir / "board_files" / name
        
        if target_dir.exists():
            self.log(f"✓ {name} already exists", "INFO")
            return True
            
        self.emit_status(f"FETCHING_DEPENDENCIES:board_{name}")
        self.log(f"Fetching board files for {name}", "INFO")
        
        # Create temporary directory
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize git repo
            cmds = [
                ["git", "init"],
                ["git", "remote", "add", "origin", board_info['url']],
                ["git", "config", "core.sparseCheckout", "true"]
            ]
            
            for cmd in cmds:
                success, error = self.run_command(cmd, cwd=temp_path, max_retries=1)
                if not success:
                    self.log(f"Failed to initialize sparse checkout for {name}: {error}", "ERROR")
                    return False
                    
            # Set sparse checkout path
            sparse_file = temp_path / ".git" / "info" / "sparse-checkout"
            sparse_file.parent.mkdir(parents=True, exist_ok=True)
            sparse_file.write_text(f"{board_info['path']}/*\n")
            
            # Fetch and checkout
            for branch in ["main", "master"]:
                success, _ = self.run_command(
                    ["git", "pull", "--quiet", "origin", branch], 
                    cwd=temp_path
                )
                if success:
                    break
            else:
                self.log(f"Failed to fetch {name}", "ERROR")
                return False
                
            # Checkout specific commit
            success, error = self.run_command(
                ["git", "checkout", "--quiet", board_info['commit']], 
                cwd=temp_path
            )
            if not success:
                self.log(f"Failed to checkout commit for {name}: {error}", "ERROR")
                return False
                
            # Move to target directory
            source_path = temp_path / board_info['path']
            if source_path.exists():
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["mv", str(source_path), str(target_dir)], check=True)
                self.log(f"✓ {name} board files fetched successfully", "INFO")
                return True
            else:
                self.log(f"Path {board_info['path']} not found in repository", "ERROR")
                return False
                
    def fetch_all_parallel(self, max_workers: int = 4) -> Tuple[int, int]:
        """Fetch all repositories in parallel"""
        tasks = []
        
        # Add main repositories
        for name, repo_info in self.repositories.items():
            tasks.append(('repo', name, repo_info))
            
        # Add board files if not skipped
        if os.environ.get('FINN_SKIP_BOARD_FILES') != '1':
            for board_info in self.board_files:
                tasks.append(('board', board_info['name'], board_info))
                
        total = len(tasks)
        success_count = 0
        failed = []
        
        self.log(f"Fetching {total} dependencies with {max_workers} workers", "INFO")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {}
            
            for task_type, name, info in tasks:
                if task_type == 'repo':
                    future = executor.submit(self.fetch_repository, name, info)
                else:  # board
                    if info.get('sparse', False):
                        future = executor.submit(self.fetch_sparse_repository, name, info)
                    else:
                        future = executor.submit(self.fetch_repository, name, info)
                        
                future_to_task[future] = (task_type, name)
                
            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_task):
                task_type, name = future_to_task[future]
                try:
                    if future.result():
                        success_count += 1
                    else:
                        failed.append(name)
                except Exception as e:
                    self.log(f"Exception fetching {name}: {e}", "ERROR")
                    failed.append(name)
                    
                # Show progress
                progress = (success_count + len(failed)) / total * 100
                self.log(f"Progress: {progress:.0f}% ({success_count + len(failed)}/{total})", "INFO")
                
        return success_count, len(failed)
        
    def verify_installations(self) -> bool:
        """Verify all dependencies are properly installed"""
        self.emit_status("VERIFYING_DEPENDENCIES")
        self.log("Verifying dependency installations", "INFO")
        
        all_good = True
        
        # Check repositories
        for name, repo_info in self.repositories.items():
            target_dir = self.deps_dir / name
            if not target_dir.exists():
                self.log(f"✗ {name} not found", "ERROR")
                all_good = False
                continue
                
            # Verify commit
            success, output = self.run_command(
                ["git", "rev-parse", "HEAD"], 
                cwd=target_dir,
                max_retries=1
            )
            if success and output.strip() == repo_info['commit']:
                self.log(f"✓ {name} verified", "INFO")
            else:
                self.log(f"✗ {name} at wrong commit", "ERROR")
                all_good = False
                
        return all_good
        
def main():
    # Get environment variables
    finn_root = os.environ.get('FINN_ROOT', '/workspace/finn')
    deps_dir = os.environ.get('FINN_DEPS_DIR', os.path.join(finn_root, 'deps'))
    config_file = os.path.join(finn_root, 'docker', 'finn_repos.yaml')
    max_workers = int(os.environ.get('FINN_FETCH_WORKERS', '4'))
    
    # Create fetcher instance
    fetcher = DependencyFetcher(deps_dir, config_file)
    
    # Fetch all dependencies
    success, failed = fetcher.fetch_all_parallel(max_workers=max_workers)
    
    # Verify installations
    if failed == 0:
        verified = fetcher.verify_installations()
        if verified:
            fetcher.log(f"All {success} dependencies fetched and verified successfully!", "INFO")
            return 0
        else:
            fetcher.log("Some dependencies failed verification", "ERROR")
            return 1
    else:
        fetcher.log(f"Failed to fetch {failed} dependencies", "ERROR")
        return 1
        
if __name__ == "__main__":
    sys.exit(main())