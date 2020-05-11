'''Setup script
Usage: pip install .
To install development dependencies too, run: pip install .[dev]
'''
from setuptools import setup, find_packages

setup(
    name='crnns4captions',
    version='v1.0',
    packages=find_packages(),
    scripts=[],
    url='https://github.com/gchochla/Deep-Representations-of-Visual-Descriptions',
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
