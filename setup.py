from setuptools import setup, find_packages

setup(
    name="dynopt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "autograd",
        "numpy",
        "scipy",
    ],
)
