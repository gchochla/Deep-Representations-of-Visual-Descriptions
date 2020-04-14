'''Setup script
Usage: pip install .
To install development dependencies too, run: pip install .[dev]
'''
from setuptools import setup, find_packages

setup(
    name='text2image',
    version='0.1.0',
    packages=find_packages(),
    scripts=[],
    author='Georgios Chochlakis',
    install_requires=[],
    extras_require={
        'dev': [
            'pylint',
            'git-pylint-commit-hook',
            'numpy',
            'torch',
            'torchvision',
            'h5py',
            'torchfile',
            'matplotlib',
        ],
    },
)
