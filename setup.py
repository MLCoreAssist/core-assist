from setuptools import find_packages, setup

def parse_requirements(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="core_assist",
    version="0.0.1",
    description="Library to support data scientist in different core tasks.",
    packages=find_packages(where="."),
    url='https://github.com/MLCoreAssist/core-assist',
    include_package_data=True,
    author="Aman Gupta , Mukul Kumar , Mohd Saqib , Rohit Chandra Maurya",
    install_requires=parse_requirements('requirements.txt'),
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.8",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # Update with your license
        'Operating System :: OS Independent',
    ],
)
