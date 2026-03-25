# Closed loop system
import numpy as np
import CL
import matplotlib.pyplot as plt
# from quadrotor_settings_debug_AD import res_funcs_generator, funcs_generator   # dummy dynamics model; only for debugging
from quadrotor_settings import res_funcs_generator, funcs_generator   # original OpenMDAO model
import copy as copy

import time as time_package

# ------------------
# Parameters
# ------------------
# initial states (x, y, theta, vx, vy, theta_vel)
x_init = np.array([2, 2, 0.1, 0.5, 0.3, 0.05])

# initial design
twist = np.array([28.624217078309165, 20., 15., 7.952156310627215])
chord = np.array([0.046549999999999994, 0.03, 0.02, 0.005414051955711461])
d_init = np.concatenate((np.deg2rad(twist), chord))

# Q and R values
ndof = 6
nctrl = 2
Q = np.eye(ndof)
R = np.eye(nctrl) * 10.0

# Time setting
dt = 0.1
# T = 30.0   # probably T=10.0 is enough
T = 10.0
N = int(T / dt)

# Provide a close steady-state initial guess
theta_0 = np.array([0.00000, 0.00000, 0.00000, 0.00000, 472.25, 472.25])

# Generate the residual forms
num_cp = len(twist)
res, res_reduced = res_funcs_generator(num_cp)

# res_1 = res_reduced.compute(theta_1, d_init, t=0.)
res_0 = res_reduced.compute(theta_0, d_init, t=0.)

# Generate the closed loop obj using the residuals
cl = CL.CL(d_init, res, res_reduced, Q, R, T, N, x_init)

# Generate the implicit funciton format
imp_FoI_cl = funcs_generator(d_init, cl, theta_0)

# set design and reset cache
imp_FoI_cl.set_design(d_init)

# Solve
t_start = time_package.time()
imp_FoI_cl.solve(theta_0=theta_0)
cost = imp_FoI_cl.compute()
print('analysis time:', time_package.time() - t_start)

import matplotlib.pyplot as plt
plt.plot(imp_FoI_cl.w[0, :], imp_FoI_cl.w[1, :], '-o')
plt.show()
print("imp_FoI_cl.w", imp_FoI_cl.w)

# exit()

k_a = 0
k_s = 0

def objfunc(xdict):

    global imp_FoI_cl
    global k_a

    k_a += 1
    print("# function eval:", k_a)

    theta_0 = np.array([0.00000, 0.00000, 0.00000, 0.00000, 472.25, 472.25])

    # Extract the design var
    x = xdict["xvars"]

    # Solve the equation
    imp_FoI_cl.set_design(x)
    imp_FoI_cl.solve(theta_0=theta_0)
    cost = imp_FoI_cl.compute()
    obj = cost

    # Set the objective function and con
    funcs = {}
    funcs["obj"] = obj

    # Set failure flag
    fail = False

    return funcs, fail

def sens(xdict, funcs):

    global imp_FoI_cl
    global k_s

    k_s += 1
    print("# function sens:", k_s)

    theta_0 = np.array([0.00000, 0.00000, 0.00000, 0.00000, 472.25, 472.25])

    # Extract the design var
    x = xdict["xvars"]

    # Solve the equation
    imp_FoI_cl.set_design(x)
    imp_FoI_cl.solve(theta_0=theta_0)

    # Compute the objective derivative
    imp_FoI_cl.solve_adjoint()
    dfdx = imp_FoI_cl.compute_grad_design()

    # Set the objective function and con derivative
    x = xdict["xvars"]
    funcsSens = {
        "obj": {"xvars": dfdx},
    }

    fail = False
    return funcsSens, fail



xdict = {}
xdict["xvars"] = d_init

funcs, fail = objfunc(xdict)
print(funcs["obj"])


funcsSens, fail = sens(xdict, funcs)
print("funcsSens", funcsSens)
der_adjoint = funcsSens["obj"]["xvars"]

epsilon = 1e-6
der_FD = np.zeros_like(der_adjoint)
for i in range(xdict["xvars"].shape[0]):
    xdict_per = copy.deepcopy(xdict)
    delta = xdict_per["xvars"][i] * epsilon
    xdict_per["xvars"][i] += delta

    funcs_per, fail = objfunc(xdict_per)
    der_FD[i] = (funcs_per["obj"] - funcs["obj"]) / delta

print("=" * 20)
print("Benchmark")
print("=" * 20)
print("Finite differences: ", der_FD)
print("-" * 20)
print("Adjoint           : ", der_adjoint)
