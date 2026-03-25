# Closed loop system
import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
import scipy.linalg
import copy
from functools import partial
import residual
import residual_reduced
import CL
import matplotlib.pyplot as plt
from cart_setting import res_funcs_generator, funcs_generator


# ------------------
# Parameters
# ------------------
m_min = 0.5
m_max = 2.0
M_min = 2.5
M_max = 7.5
L = 1.0

total_m = 3.5

m = m_min
M = M_min

x = np.zeros(3)
x[0] = m
x[1] = M
x[2] = L

# Q and R values
ndof = 4
nctrl = 1
Q = np.eye(ndof) * 0.1
R = np.eye(nctrl) 

# Time setting
dt = 0.001
T = .1
N = int(T / dt)

# Initial state
w_0 = np.zeros(4)
w_0[0] = -1.0
w_0[2] = 2.0

# Provide a close steady-state initial guess
theta_0 = np.array([np.pi - 0.1, 0.13])

# Generate the residual forms
res, res_reduced = res_funcs_generator()

# Generate the closed loop obj using the residuals
cl = CL.CL(x, res, res_reduced, Q, R, T, N, w_0)

# Generate the implicit funciton format
imp_FoI_cl = funcs_generator(x, cl, theta_0)

# Solve
imp_FoI_cl.solve(theta_0 = theta_0)
imp_FoI_cl.compute()

k_a = 0
k_s = 0

def objfunc(xdict):

    global imp_FoI_cl
    global k_a

    k_a += 1
    print("# function eval:", k_a)

    theta_0 = np.array([np.pi - 0.1, 0.13])

    # Extract the design var
    x = xdict["xvars"]

    # Solve the equation
    imp_FoI_cl.set_design(x)
    imp_FoI_cl.solve(theta_0 = theta_0)
    cost = imp_FoI_cl.compute()
    obj = cost

    con = (x[0] + x[1]) - total_m

    # Set the objective function and con
    funcs = {}
    funcs["obj"] = obj
    funcs["con"] = con

    # Set failure flag
    fail = False

    return funcs, fail

def sens(xdict, funcs):

    global imp_FoI_cl
    global k_s

    k_s += 1
    print("# function sens:", k_s)


    # Extract the design var
    x = xdict["xvars"]

    # Solve the equation
    imp_FoI_cl.set_design(x)
    imp_FoI_cl.solve(theta_0 = theta_0)

    # Compute the objective derivative
    imp_FoI_cl.solve_adjoint()
    dfdx = imp_FoI_cl.compute_grad_design()
    print('dfdx:', dfdx)

    # Set the objective function and con derivative
    x = xdict["xvars"]
    funcsSens = {
        "obj": {"xvars": dfdx},
        "con":{"xvars": [1.0, 1.0, 0.0]}
    }

    fail = False
    return funcsSens, fail

# ========================================
#    Optimization problem setup
# ========================================

from pyoptsparse import OPT, Optimization

# Optimization Object
optProb = Optimization("Control co-desgin", objfunc)

x0 = [1, 5, 1]

# Design Variables
# lower = [0.5, 2.5, 1.0]
# upper = [2.0, 7.5, 3.0]
lower = [0.5, 2.5, 1] # HACK: fix the bar length
upper = [2.0, 7.5, 1]
value = x0
optProb.addVarGroup("xvars", len(x0), lower=lower, upper=upper, value=value)


# Objective
optProb.addObj("obj")

# Constraints
lower = [0.0]
upper = [None]
optProb.addConGroup("con", 1, lower=lower, upper=upper)

# Check optimization problem:
print(optProb)

# Optimizer
# optimizer = "snopt"
optimizer = "ipopt"
optOptions = {}
opt = OPT(optimizer, options=optOptions)

# Solution
histFileName = "%s_CCD.hst" % optimizer

sol = opt(optProb, sens=sens, storeHistory=histFileName)

# Check Solution
print(sol) # [m, M, L, d]* = [2, 2.5, 1, 1.06825] -----new: [1.0, 2.5, 1] --> with m + M => 3.5