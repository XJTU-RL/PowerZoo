from setuptools import find_packages, setup

setup(
    name="PowerZoo",
    version="1.0.0",
    author="PKU-MARL",
    description="PyTorch implementation of PowerZoo Algorithms",
    url="https://github.com/PKU-MARL/PowerZoo",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        # Environment frameworks
        "gym==0.26.2",
        "gymnasium==0.29.1",
        # Power system simulation
        "dss-python==0.15.7",
        # Numerical computing
        "numpy>=1.23.0,<1.25.0",
        "scipy>=1.13.0,<2.0.0",
        "pandas>=2.0.0,<3.0.0",
        # Deep learning
        "torch>=2.0.0,<2.4.0",
        "stable-baselines3>=1.8.0",
        # Visualization
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.11.0,<1.0.0",
        "imageio>=2.30.0",
        "networkx>=3.0,<4.0",
        # Data storage & configuration
        "h5py>=3.0.0",
        "pyyaml>=5.3.1,<7.0.0",
        "absl-py>=1.0.0",
        # Training & monitoring
        "tensorboard>=2.2.1",
        "tensorboardX>=2.0",
        # System utilities
        "setproctitle>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "viz": [
            "plotly>=5.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
