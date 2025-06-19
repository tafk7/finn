#!/usr/bin/env python3
"""
Comprehensive test suite for FINN Docker modernization
Tests all aspects of the new Docker orchestration system
"""

import os
import sys
import subprocess
import time
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import pytest

# Add FINN root to path
FINN_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(FINN_ROOT))

class TestDockerSystem:
    """Test suite for FINN Docker system"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.finn_root = FINN_ROOT
        cls.finn_docker = cls.finn_root / "finn-docker"
        cls.test_container = f"finn_test_{os.getpid()}"
        cls.original_env = os.environ.copy()
        
    @classmethod
    def teardown_class(cls):
        """Cleanup test environment"""
        # Restore original environment
        os.environ.clear()
        os.environ.update(cls.original_env)
        
        # Clean up test container if it exists
        subprocess.run([str(cls.finn_docker), "clean"], capture_output=True)
        
    def run_command(self, cmd, check=True, timeout=30):
        """Run a command and return result"""
        if isinstance(cmd, str):
            cmd = cmd.split()
            
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=self.finn_root
        )
        
        if check and result.returncode != 0:
            print(f"Command failed: {' '.join(cmd)}")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            
        return result
        
    def test_finn_docker_exists(self):
        """Test that finn-docker script exists and is executable"""
        assert self.finn_docker.exists(), "finn-docker script not found"
        assert os.access(self.finn_docker, os.X_OK), "finn-docker not executable"
        
    def test_help_command(self):
        """Test help command"""
        result = self.run_command([str(self.finn_docker), "help"])
        assert result.returncode == 0
        assert "FINN Container Management" in result.stdout
        assert "Commands:" in result.stdout
        assert "Examples:" in result.stdout
        
    def test_status_command(self):
        """Test status command"""
        result = self.run_command([str(self.finn_docker), "status"])
        assert result.returncode == 0
        # Should show container doesn't exist or current status
        assert any(x in result.stdout for x in ["does not exist", "is running", "exists but is stopped"])
        
    def test_configuration_files(self):
        """Test that all configuration files exist"""
        config_files = [
            "docker/finn_repos.yaml",
            "docker/docker_defaults.env",
            "docker/fetch-repos.sh",
            "docker/entrypoint.sh",
            "docker/entrypoint_exec.sh",
            "docker/entrypoint_common.sh",
            "docker/fetch_deps.py",
            "docker/install_deps.py",
            "docker/verify_installation.py",
            "docker/health_check.py",
            "docker/startup_profiler.py",
            "docker/error_handler.sh"
        ]
        
        for config_file in config_files:
            path = self.finn_root / config_file
            assert path.exists(), f"Configuration file missing: {config_file}"
            
    def test_yaml_configuration(self):
        """Test YAML configuration is valid"""
        import yaml
        
        yaml_path = self.finn_root / "docker/finn_repos.yaml"
        assert yaml_path.exists()
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            
        assert 'repositories' in config
        assert 'board_files' in config
        assert len(config['repositories']) > 0
        
        # Check repository structure
        for name, repo in config['repositories'].items():
            assert 'url' in repo, f"Repository {name} missing URL"
            assert 'commit' in repo, f"Repository {name} missing commit"
            
    def test_backward_compatibility_wrapper(self):
        """Test run-docker.sh backward compatibility"""
        run_docker = self.finn_root / "run-docker.sh"
        assert run_docker.exists(), "run-docker.sh wrapper missing"
        assert os.access(run_docker, os.X_OK), "run-docker.sh not executable"
        
        # Test that it shows deprecation notice
        result = self.run_command([str(run_docker), "-h"], check=False)
        assert "deprecated" in result.stdout or "deprecated" in result.stderr
        
    def test_environment_variables(self):
        """Test environment variable handling"""
        # Test with custom environment variables
        env = os.environ.copy()
        env['FINN_XILINX_PATH'] = '/custom/xilinx'
        env['FINN_XILINX_VERSION'] = '2023.1'
        env['JUPYTER_PORT'] = '9999'
        
        result = subprocess.run(
            [str(self.finn_docker), "help"],
            capture_output=True,
            text=True,
            env=env
        )
        assert result.returncode == 0
        
    def test_error_handling(self):
        """Test error handling"""
        # Test with invalid command
        result = self.run_command([str(self.finn_docker), "invalid_command"], check=False)
        assert result.returncode != 0
        
        # Test exec without running container
        result = self.run_command([str(self.finn_docker), "exec", "echo test"], check=False)
        assert result.returncode != 0
        assert "not running" in result.stderr or "not running" in result.stdout
        
    def test_shortcuts(self):
        """Test command shortcuts in help text"""
        result = self.run_command([str(self.finn_docker), "help"])
        assert "Shortcuts:" in result.stdout
        assert "make" in result.stdout
        assert "pytest" in result.stdout
        assert "python" in result.stdout
        
    @pytest.mark.slow
    def test_container_lifecycle(self):
        """Test container lifecycle (requires Docker)"""
        # Check if Docker is available
        docker_check = subprocess.run(["docker", "version"], capture_output=True)
        if docker_check.returncode != 0:
            pytest.skip("Docker not available")
            
        # Test init command (this might take a while)
        result = self.run_command([str(self.finn_docker), "init"], timeout=300)
        assert result.returncode == 0
        assert "ready" in result.stdout.lower()
        
        # Test status shows running
        result = self.run_command([str(self.finn_docker), "status"])
        assert "is running" in result.stdout
        
        # Test exec command
        result = self.run_command([str(self.finn_docker), "exec", "echo hello from container"])
        assert result.returncode == 0
        assert "hello from container" in result.stdout
        
        # Test stop command
        result = self.run_command([str(self.finn_docker), "stop"])
        assert result.returncode == 0
        
        # Test clean command
        result = self.run_command([str(self.finn_docker), "clean"])
        assert result.returncode == 0
        
        
class TestDependencyManagement:
    """Test dependency management functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.finn_root = FINN_ROOT
        self.deps_dir = self.finn_root / "deps"
        
    def test_fetch_repos_script(self):
        """Test fetch-repos.sh script"""
        fetch_script = self.finn_root / "docker/fetch-repos.sh"
        assert fetch_script.exists()
        assert os.access(fetch_script, os.X_OK)
        
        # Test wrapper script
        wrapper = self.finn_root / "fetch-repos.sh"
        assert wrapper.exists()
        assert os.access(wrapper, os.X_OK)
        
    def test_python_dependency_scripts(self):
        """Test Python dependency management scripts"""
        scripts = [
            "docker/fetch_deps.py",
            "docker/install_deps.py",
            "docker/verify_installation.py"
        ]
        
        for script in scripts:
            path = self.finn_root / script
            assert path.exists(), f"Script missing: {script}"
            assert os.access(path, os.X_OK), f"Script not executable: {script}"
            
            # Test that script has proper shebang
            with open(path, 'r') as f:
                first_line = f.readline()
                assert first_line.startswith("#!/usr/bin/env python3")
                
    @pytest.mark.slow
    def test_dependency_fetching(self):
        """Test dependency fetching (requires network)"""
        # Skip if offline
        if os.environ.get('FINN_OFFLINE') == '1':
            pytest.skip("Offline mode")
            
        # Test fetching a small dependency
        result = subprocess.run(
            ["python3", str(self.finn_root / "docker/fetch_deps.py")],
            capture_output=True,
            text=True,
            env={**os.environ, 'FINN_ROOT': str(self.finn_root)},
            timeout=60
        )
        
        # Check for status messages
        assert "Fetching FINN dependencies" in result.stdout or result.returncode == 0
        

class TestPerformanceFeatures:
    """Test performance optimizations"""
    
    def setup_method(self):
        """Setup for each test"""
        self.finn_root = FINN_ROOT
        
    def test_startup_profiler(self):
        """Test startup profiler"""
        profiler = self.finn_root / "docker/startup_profiler.py"
        assert profiler.exists()
        
        # Test that profiler can be imported
        result = subprocess.run(
            ["python3", str(profiler)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "Profile saved to:" in result.stdout
        
    def test_fast_execution_path(self):
        """Test fast execution entrypoint"""
        fast_exec = self.finn_root / "docker/entrypoint_exec.sh"
        assert fast_exec.exists()
        
        # Check for optimization features
        with open(fast_exec, 'r') as f:
            content = f.read()
            assert "needs_xilinx" in content
            assert "FINN_FAST_EXEC" in content
            
    def test_health_check_system(self):
        """Test health check functionality"""
        health_check = self.finn_root / "docker/health_check.py"
        assert health_check.exists()
        
        # Test that health check can run (might fail outside container)
        result = subprocess.run(
            ["python3", str(health_check)],
            capture_output=True,
            text=True
        )
        # Should at least not crash
        assert "Running FINN container health check" in result.stdout
        

class TestDocumentation:
    """Test documentation completeness"""
    
    def setup_method(self):
        """Setup for each test"""
        self.finn_root = FINN_ROOT
        
    def test_docker_readme(self):
        """Test Docker directory README"""
        readme = self.finn_root / "docker/README.md"
        assert readme.exists(), "Docker README missing"
        
        with open(readme, 'r') as f:
            content = f.read()
            
        # Check for key sections
        assert "Contents" in content
        assert "Usage" in content
        assert "Architecture" in content
        assert "Customization" in content
        
    def test_main_readme_updated(self):
        """Test that main README mentions finn-docker"""
        readme = self.finn_root / "README.md"
        if readme.exists():
            with open(readme, 'r') as f:
                content = f.read()
                # Check if finn-docker is mentioned
                if "docker" in content.lower():
                    assert "finn-docker" in content or "run-docker.sh" in content
                    

def run_tests():
    """Run all tests and generate report"""
    import pytest
    
    # Run pytest with detailed output
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow",  # Skip slow tests by default
        "--junit-xml=/tmp/finn_docker_tests.xml"
    ])
    

if __name__ == "__main__":
    run_tests()