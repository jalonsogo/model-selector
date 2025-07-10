#!/usr/bin/env python3
"""
Setup script for AI Model Selection Assistant
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="model-selector",
    version="1.0.0",
    description="AI Model Selection Assistant with hardware detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Model Selector Team",
    author_email="info@model-selector.com",
    url="https://github.com/your-org/model-selector",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "model_selector": ["../models.json"],
    },
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
        "optional": [
            "psutil>=5.8.0",
            "GPUtil>=1.4.0",
            "PyYAML>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "model-selector=main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="ai models hardware detection gpu cpu memory quantization",
)