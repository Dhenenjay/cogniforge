"""
CogniForge Setup Configuration

This file configures the installation of CogniForge with CLI entrypoints.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements
requirements = [
    "numpy>=1.19.0",
    "torch>=1.9.0",
    "pybullet>=3.0.0",
    "gymnasium>=0.26.0",
    "stable-baselines3>=1.6.0",
    "opencv-python>=4.5.0",
    "pillow>=8.2.0",
    "matplotlib>=3.3.0",
    "scipy>=1.7.0",
    "pyyaml>=5.4.0",
    "fastapi>=0.70.0",
    "uvicorn>=0.15.0",
    "openai>=1.54.0",
    "cma>=3.0.0",
]

setup(
    name="cogniforge",
    version="1.0.0",
    author="CogniForge Team",
    author_email="team@cogniforge.ai",
    description="AI-Powered Robotics Framework for Intelligent Manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cogniforge/cogniforge",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
        "viz": [
            "plotly>=5.0",
            "dash>=2.0",
            "tensorboard>=2.7",
        ],
        "rl": [
            "wandb>=0.12",
            "ray[rllib]>=1.13",
        ],
    },
    entry_points={
        "console_scripts": [
            "cogv=cogniforge.cli:main",
            "cogniforge=cogniforge.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cogniforge": [
            "config/*.yaml",
            "config/*.json",
            "assets/*.json",
            "templates/*.py",
        ],
    },
    zip_safe=False,
    keywords=[
        "robotics",
        "ai",
        "machine-learning",
        "reinforcement-learning",
        "computer-vision",
        "pybullet",
        "pytorch",
        "manipulation",
        "grasping",
        "behavioral-cloning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/cogniforge/cogniforge/issues",
        "Source": "https://github.com/cogniforge/cogniforge",
        "Documentation": "https://cogniforge.readthedocs.io",
    },
)