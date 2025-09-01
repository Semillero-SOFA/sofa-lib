"""
Setup script for SOFA Library.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sofa-lib",
    version="2.0.0",
    author="SOFA Research Group",
    author_email="",
    description="A modular Python library for optical fiber communication research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "polars>=0.15.0",
        "h5py>=3.0.0",
        "joblib>=1.0.0",
    ],
    extras_require={
        "neural": ["tensorflow>=2.8.0"],
        "dev": ["pytest>=6.0", "black", "flake8"],
    },
    package_data={
        "sofa": ["*.py"],
    },
)
