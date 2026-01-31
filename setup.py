"""Setup script for AI Grand Prix package."""

from setuptools import setup, find_packages

setup(
    name="airgrandprix",
    version="0.1.0",
    description="Vision-based drone racing for AI Grand Prix competition",
    author="AI Grand Prix Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gym-pybullet-drones>=2.0.0",
        "gymnasium>=1.0.0",
        "pybullet>=3.2.0",
        "torch>=2.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "rl": ["stable-baselines3>=2.0.0"],
        "dev": ["pytest>=7.0.0", "black>=23.0.0", "isort>=5.12.0"],
    },
    entry_points={
        "console_scripts": [
            "agp-train-gatenet=scripts.train_gate_net:main",
            "agp-train-gcnet=scripts.train_gcnet:main",
            "agp-run=scripts.run_pipeline:main",
        ],
    },
)
