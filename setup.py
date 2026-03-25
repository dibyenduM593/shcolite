from setuptools import setup, find_packages

setup(
    name="shcolite",
    version="0.1.0",
    description="System-of-Systems Hydrological-Computational Orchestration (Lite)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
    ],
)
