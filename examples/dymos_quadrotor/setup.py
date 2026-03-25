from setuptools import setup

setup(
    name="quadrotormodels",
    version="0.0.1",
    description="Quadrotor models for control co-design",
    url="",
    keywords="",
    author="",
    author_email="",
    license="",
    packages=[
        "quadrotormodels",
        "quadrotormodels.plotters"],
    # TODO: update the version lower bounds
    install_requires=[
        "numpy",
        "matplotlib",
        "openmdao>=3.16.0",
        "dymos>=1.4.0",
        "pyoptsparse",
        "ccblade-openmdao-examples==0.0.1",
    ],
)
