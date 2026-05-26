"""setup.py — Make alphaguard an installable Python package."""
from setuptools import setup, find_packages

setup(
    name="alphaguard",
    version="1.0.0",
    description="AI-powered Alpha defect detection for Tata Steel Hot Rolling",
    author="Ashik Sharon",
    packages=find_packages(exclude=["tests*", "data*"]),
    python_requires=">=3.10",
    install_requires=open("requirements.txt").read().splitlines(),
)
