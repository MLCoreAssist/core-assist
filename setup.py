from setuptools import find_packages, setup

setup(
    name="Core Assist",
    version="0.0.10",
    description="Standard Library for computer vision team",
    packages=find_packages(where="."),  # Search all subdirectories for packages
    include_package_data=True,
    author="Aman Gupta , Mukul Kumar , Mohd Saqib , Rohit Chandra Maurya",

    install_requires=[
        "opencv-python",
        "numpy",
        "matplotlib",
        "altair",
        "imagesize",
        "seaborn",
        "scikit-learn",
        "scikit-multilearn"
    ],
    extras_require={

        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.8",
)
