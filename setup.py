from setuptools import find_packages
from setuptools import setup

setup(
    name='talos',
    version='0.0.1.dev0',
    description='Large-scale optimization of CubeSat swarms',
    packages=find_packages(),
    install_requires=[
        'csdl',
        'numpy-stl',
        'trimesh',
        'smt',
    ],
)
