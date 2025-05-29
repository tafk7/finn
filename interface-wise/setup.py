"""Setup configuration for FINN Extensible Parallelism System."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="finn-extensible-parallelism",
    version="0.1.0",
    author="FINN Development Team",
    description="Extensible parallelism system for FINN hardware accelerator generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Xilinx/finn",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.812",
        ],
        "viz": [
            "matplotlib>=3.3.0",
            "networkx>=2.5",
        ],
        "optimization": [
            "scipy>=1.7.0",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Xilinx/finn/issues",
        "Source": "https://github.com/Xilinx/finn",
        "Documentation": "https://finn.readthedocs.io/",
    },
)
