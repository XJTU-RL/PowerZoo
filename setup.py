"""Package metadata using the same dependency lists as the development setup."""

from pathlib import Path

from setuptools import find_packages, setup


def read_requirements(filename):
    """Read plain requirement lines without duplicating version declarations."""
    return [
        line.strip()
        for line in (Path(__file__).parent / filename)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


setup(
    name="PowerZoo",
    version="1.0.0",
    author="PowerZoo contributors",
    description="Reinforcement learning environments for power system control",
    url="https://github.com/XJTU-RL/PowerZoo",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "viz": read_requirements("requirements-viz.txt"),
        "docs": ["sphinx>=5.0.0", "sphinx-rtd-theme>=1.0.0"],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
