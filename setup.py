#!/usr/bin/env python

from setuptools import setup

setup(
    name='lightning_baselines3',
    packages=['lightning_baselines3'],
    install_requires=[
        "gym>=0.17",
        "numpy",
        "torch>=1.4.0",
        "pytorch_lightning>=1.0.0",
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
            # Reformat
            "black",
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
        "extra": [
            # For render
            "opencv-python",
            # For atari games,
            "atari_py~=0.2.0",
            "pillow",
            # Checking memory taken by replay buffer
            "psutil",
        ],
    },
    version='0.1.0',
    description='Adaptation of Stable_Baselines3 for Pytorch Lightning',
    author='Hengjian (Henry) Jia',
    author_email='henryjia18@gmail.com',
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gym openai stable baselines toolbox python data-science",
    license="MIT",
    url='https://github.com/HenryJia/lightning-baselines3',
    )
