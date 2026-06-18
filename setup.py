"""OpenPanelFlutter: A Python package for to panel flutter analysis."""

from setuptools import find_packages, setup

setup(
    name="openpanelflutter",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.22,<2.0.0",
        "numba>=0.59.0,<0.61.0",
        "matplotlib>=3.5",
        "scipy>=1.8",
    ],
)
