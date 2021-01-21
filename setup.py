#!/usr/bin/env python

import os
from setuptools import setup

with open(os.path.join("lightning_baselines3", "version.txt"), "r") as file_handler:
    __version__ = file_handler.read().strip()

setup(
    name='lightning_baselines3',
    packages=['lightning_baselines3'],
    package_data={"lightning_baselines3": ["py.typed", "version.txt"]},
    python_requires=">=3.6",
    install_requires=[
        "gym>=0.17",
        "numpy>=1.16.6",
        "pandas",
        "torch>=1.3",
        "pytorch_lightning>=1.1.0",
        # For render
        "opencv-python",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "pytype",
            # Lint code
            "flake8>=3.8",
            # Sort imports
            "isort>=5.0",
        ],
        "docs": [
            "sphinx",
            "sphinx-autobuild",
            "sphinx-rtd-theme",
            # For spelling
            "sphinxcontrib.spelling",
            # Type hints support
            "sphinx-autodoc-typehints",
        ],
    },
    version=__version__,
    description='Adaptation of Stable_Baselines3 for Pytorch Lightning',
    author='Hengjian (Henry) Jia',
    author_email='henryjia18@gmail.com',
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    url='https://github.com/HenryJia/lightning-baselines3',
    )
