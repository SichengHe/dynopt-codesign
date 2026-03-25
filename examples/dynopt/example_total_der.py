# Closed loop system
import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
import scipy.linalg
import copy
from functools import partial
from dynopt import residual, residual_reduced
from dynopt import CL
import matplotlib.pyplot as plt
from cart_setting import res_funcs_generator, funcs_generator


# ------------------
# Parameters
# ------------------
m_min = 1.0
m_max = 2.0
M_min = 5.0
M_max = 7.0
L = 2.0

total_m = 7.0

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
T = 0.1
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

    # Set the objective function and con derivative
    x = xdict["xvars"]
    funcsSens = {
        "obj": {"xvars": dfdx},
        "con":{"xvars": [1.0, 1.0, 0.0]}
    }

    fail = False
    return funcsSens, fail



xdict = {}
xdict["xvars"] = np.array([m, M, L])
print('design solution: ', np.array([m, M, L]))

funcs, fail = objfunc(xdict)
print(funcs["obj"])


funcsSens, fail = sens(xdict, funcs)
der_adjoint = funcsSens["obj"]["xvars"]


epsilon = 1e-6
der_FD = np.zeros_like(der_adjoint)
for i in range(xdict["xvars"].shape[0]):
    xdict_per = copy.deepcopy(xdict)
    xdict_per["xvars"][i] += epsilon

    funcs_per, fail = objfunc(xdict_per)
    der_FD[i] = (funcs_per["obj"] - funcs["obj"]) / epsilon

print("=" * 20)
print("Benchmark")
print("=" * 20)
print("Finite differences: ", der_FD)
print("-" * 20)
print("Adjoint           : ", der_adjoint)



