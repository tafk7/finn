# FINN Docker Directory

This directory contains all Docker-related files for the FINN framework.

## Contents

### Core Docker Files
- `Dockerfile.finn` - Main Dockerfile for building FINN images
- `entrypoint.sh` - Primary container entrypoint with initialization
- `entrypoint_exec.sh` - Fast execution entrypoint for running containers
- `entrypoint_common.sh` - Shared environment setup

### Dependency Management
- `finn_repos.yaml` - YAML configuration of all dependencies
- `fetch-repos.sh` - Enhanced repository fetching with retry logic
- `fetch_deps.py` - Python-based parallel dependency fetcher
- `install_deps.py` - Smart package installer with caching
- `verify_installation.py` - Comprehensive installation verifier

### Configuration
- `docker_defaults.env` - Default environment variable settings

### Scripts
- `quicktest.sh` - Quick test runner script

## Usage

### Building the Docker Image
```bash
./finn-docker build
```

### Managing Dependencies
Dependencies are defined in `finn_repos.yaml` and fetched automatically during container initialization. The system includes:
- Parallel downloading for faster fetching
- Automatic retry on network failures
- Smart caching to avoid redundant installations
- Progress tracking and status reporting

### Verification
After initialization, verify the installation:
```bash
./finn-docker verify
```

## Architecture

The Docker system uses a dual-entrypoint architecture:
1. `entrypoint.sh` - Used for container initialization
2. `entrypoint_exec.sh` - Used for fast command execution

This allows persistent containers with minimal overhead for repeated commands.

## Customization

### Adding Dependencies
Edit `finn_repos.yaml` to add new repositories:
```yaml
repositories:
  new_repo:
    url: https://github.com/org/repo.git
    commit: abc123
    type: python
    install: editable
```

### Environment Variables
Default settings are in `docker_defaults.env`. Override by setting environment variables before running finn-docker.