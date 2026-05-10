"""OpenPanelFlutter: A Python package for to panel flutter analysis."""

from setuptools import find_packages, setup

setup(
    name="openpanelflutter",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
    ],
)
