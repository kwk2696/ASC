from setuptools import find_namespace_packages, setup

setup(
    name="embodied_cd",
    version="1.0.0",
    description="ASC: Amortizing Self-Correction for Continual Embodied Adaptation (AACL-IJCNLP 2026)",
    author="Woo Kyung Kim, Jooyong Kim, Minjong Yoo, Honguk Woo",
    packages=find_namespace_packages(include=["embodied_cd*", "scripts*"]),
    python_requires=">=3.10",
)
