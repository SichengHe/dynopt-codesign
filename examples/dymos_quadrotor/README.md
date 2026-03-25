# Quadrotor control co-design

# Runscripts
Optimization runscripts can be found in `dymos_quadrotor/runscripts`
- `run_steady_state.py`: optimize rotor design at steady hover condition. No trajectory or control.
- `run_openloop_ccd.py`: open-loop trajectory optimization and co-design.
- `run_closedloop_ccd.py`: closed-loop traejctory optimization and co-design. Linear feedback matrix is directly optimized, not by LQR.
- `run_lqr_ccd.py`: closed-loop traejctory optimization and co-design with LQR control.

# Installation
Install Python first.
The following installation assumes you use a Python virtual environment.
I'm using `pyenv` and `virtualenv`.

### 1. Install python packages and Julia wrapper
1. `pip install numpy matplotlib openmdao dymos`
2. Install `pyoptsparse`.
3. Install the OpenMDAO-Julia interface by `pip install omjlcomps`.

By pip-installing [`omjlcomps`](https://github.com/dingraha/OpenMDAO.jl/tree/master), it automatically installs requires dependency to call Julia package from Python, including Julia itself.

Optinally, you can run the following python tests to make sure your installation:
```
git clone git@github.com:byuflowlab/OpenMDAO.jl.git
cd OpenMDAO.jl/python/omjlcomps/test
python test_julia_explicit_comp.py
python test_julia_implicit_comp.py
```

### 2. Install CCBlade and other Julia packages
Download my fork of [`CCBlade.jl`](https://github.com/kanekosh/CCBlade.jl/tree/reversal).
```
git clone --branch reversal git@github.com:kanekosh/CCBlade.jl.git
```
Here, you'll need to use the `reversal` branch in my fork because I made a minor edits to the original CCBlade code.

Then, in python prompt, install CCBlade by
```
import juliapkg
juliapkg.add("CCBlade", "e1828068-15df-11e9-03e4-ef195ea46fa4", dev=True, version="0.2.3", path=<path to the closed CCBLade.jl repo above>)
(for example:) juliapkg.add("CCBlade", "e1828068-15df-11e9-03e4-ef195ea46fa4", dev=True, version="0.2.3", path="/Users/shugo/packages/CCBlade.jl")
```
Also in python prompt, install some other julia packages by
```
juliapkg.add("ConcreteStructs", "2569d6c7-a4a2-43d3-a901-331e8e4be471", version="0.2.3")
juliapkg.add("ForwardDiff", "f6369f11-7733-5829-9624-2563aa707210", version="0.10.35")
juliapkg.add("ComponentArrays", "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66", version="0.13.14")
juliapkg.add("FLOWMath", "6cb5d3fb-0fe8-4cc2-bd89-9fe0b19a99d3", version="0.3.3")
juliapkg.project()
```
The package versions may not need to be pinned.

### 3. Install OpenMDAO wrapper of CCBlade.
This is a python package, and can be installed by `pip`.
Use `pythoncall` branch in my fork of [`CCBladeOpenMDAOExamples`](https://github.com/kanekosh/CCBladeOpenMDAOExamples/tree/pythoncall).

```
git clone --branch pythoncall git@github.com:kanekosh/CCBladeOpenMDAOExamples.git
cd CCBladeOpenMDAOExamples
pip install -e .
```
If `pip install` failed here, you may need to move the `CCBladeOpenMDAOExamples/data` folder  to somewhere else.

### 4. Install quadrotor models
Finally, install the quadrotor models as a python package `quadrotormodels`.
```
cd article_MDO_w_control/code/dymos_quadrotor
pip install -e .
```

# Test
Run the following test to make sure that OpenMDAO, CCBlade, and its wrapper were successfully installed.
```
cd article_MDO_w_control/code/dymos_quadrotor
python tests/test_hover_trim.py
```

TODO: test including `pyoptsparse` and `dymos`.

# Dependencies
Here is a non-exhaustive list of the packages and their versions. Other versions also probably works.
You don't need to install these one by one.
These will be (automatically) installed if you follow the instruction above.

### Python package
- Python 3.11.1
- numpy 1.25.0
- openmdao 3.26.0
- dymos 1.7.0
- pyoptsparse 2.9.2
- juliacall 0.9.13
- juliapkg 0.1.10
- ccblade-openmdao-examples 0.0.1 (Use `pythoncall` branch in my fork: https://github.com/kanekosh/CCBladeOpenMDAOExamples)

### Julia packages
Tip: Run `sort(collect(Pkg.installed()))` in Julia prompt to list the installed packages.
- Julia 1.9.1
- CCBlade 0.2.3  (Use `reversal` branch in my fork: https://github.com/kanekosh/CCBlade.jl)
- ComponentArrays 0.13.14
- ConcreteStructs 0.2.3
- FLOWMath 0.3.3
- ForwardDiff 0.10.35
- OpenMDAOCore 0.3.1
- PythonCall 0.9.13
