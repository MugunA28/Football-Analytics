"""
Setup configuration for Football Analytics package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="football-analytics",
    version="0.1.0",
    author="Football Analytics Team",
    author_email="info@football-analytics.com",
    description="A sophisticated football betting analytics system for finding value bets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MugunA28/Football-Analytics",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'football-analytics=src.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
