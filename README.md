# dynopt-codesign

Dynamic optimization and co-design framework for closed-loop control systems.

## Overview

This repository provides tools for simultaneous optimization of plant design and closed-loop controller design (co-design) using adjoint-based derivative computation. It supports:

- **Steady-state equilibrium** analysis with Newton solvers
- **LQR controller** synthesis with adjoint sensitivity
- **ODE time integration** with adjoint-based gradients
- **Closed-loop simulation** combining equilibrium, LQR, and ODE components

## Installation

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate mdo_quadrotor

# Install the dynopt package
pip install -e .
```

## Package structure

```
dynopt/               Core library
  residual.py         Residual class for dynamical systems
  residual_cl.py      Closed-loop residual (with LQR feedback)
  residual_reduced.py Reduced residual for steady-state analysis
  equilibrium.py      Equilibrium solver with adjoint
  LQR.py              LQR controller and adjoint
  ODE_solver.py       Forward Euler ODE integrator with adjoint
  CL.py               Closed-loop co-design system
examples/             Example problems
  dynopt/             Cart-pole and quadrotor examples
  dymos_cartpole/     Dymos-based cart-pole example
  dymos_quadrotor/    Dymos-based quadrotor example
tests/                Unit tests
```

## Running tests

```bash
python -m pytest tests/
```

## Examples

- **Cart-pole**: `examples/dynopt/example_opt.py`
- **Quadrotor**: `examples/dynopt/example_opt_quadrotor.py`
