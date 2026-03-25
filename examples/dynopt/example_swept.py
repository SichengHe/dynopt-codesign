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

my_blue = "#4C72B0"
my_red = "#C54E52"
my_green = "#56A968"
my_brown = "#b4943e"
my_purple = "#684c6b"
my_orange = "#cc5500"


is_training = False

# Parameters
m_min = 0.1
m_max = 2.0
M_min = 1.5
M_max = 7.5

total_m = 3.5

# Initial design variable setting
m = m_min
M = M_min
L = 1.0

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
T = 30
N = int(T // dt)

# Initial state
w_0 = np.zeros(4)
w_0[0] = -1.0
w_0[2] = 2.0

# Provide a close steady-state initial guess
theta_0 = np.array([np.pi - 0.1, 0.13])

if is_training:

    # Generate the residual forms
    res, res_reduced = res_funcs_generator()

    # Generate the closed loop obj using the residuals
    cl = CL.CL(x, res, res_reduced, Q, R, T, N, w_0)

    # Generate the implicit funciton format
    imp_FoI_cl = funcs_generator(x, cl, theta_0)

    # Sampling
    NN = 10
    m_arr = np.linspace(m_min, m_max, NN)
    M_arr = np.linspace(M_min, M_max, NN)
    obj_arr = np.zeros((NN, NN))

    obj_mat = np.zeros((NN, NN))
    for i in range(NN):
        m_loc = m_arr[i]
        for j in range(NN):

            print("i, j", i, j)

            M_loc = M_arr[j]

            x = np.zeros(3)
            x[0] = m_loc
            x[1] = M_loc
            x[2] = L

            imp_FoI_cl.set_design(x)
            imp_FoI_cl.solve(theta_0 = theta_0)

            FoI_cl_val_p = imp_FoI_cl.compute()

            obj_arr[i, j] = FoI_cl_val_p

    print("obj_arr", obj_arr)
    np.savetxt("example_swept_obj_arr.txt", obj_arr)

else:
    # Optimization path
    filename_hist = "code\dynOpt\opt_hist.dat"
    x_path = np.loadtxt(filename_hist)
    # Optimal solution
    x_opt = [1.0, 2.5]

    fig, ax = plt.subplots(1, figsize=(6, 6))
    NN = 10
    m_arr = np.linspace(m_min, m_max, NN)
    M_arr = np.linspace(M_min, M_max, NN)
    X1_arr, X2_arr = np.meshgrid(m_arr, M_arr)

    # --------------
    # Constraint
    # --------------

    # Countour plot
    obj_arr = np.loadtxt("code\dynOpt\example_swept_obj_arr.txt") / 100
    levels0 = np.arange(np.min(obj_arr), np.max(obj_arr), (np.max(obj_arr) - np.min(obj_arr)) / 25.0)
    levels1 = [25, 50, 75, 100, 125, 150, 175, 200,225, 250,275, 300,330]
    CP0 = ax.contour(X1_arr, X2_arr, obj_arr.T, levels1, extend="both", linewidths=2, cmap="coolwarm", zorder=0)

    ax.clabel(CP0, levels1, inline=True, fmt="%1.2f", fontsize=14)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel(r"$m$", fontsize=20)
    ax.set_ylabel(r"$M$", fontsize=20, rotation=0)

    ax.yaxis.set_label_coords(-0.1, 0.5)

    # Extract the constraint
    num_ind = 40
    x_con = np.linspace(m_min, m_max, num_ind)
    y_con1 = 3.5 - x_con

    y_con2 = 2.5

    for i in range(num_ind):
     if y_con1[i] < y_con2:
        y_con1[i] = y_con2


    # Adding optimization paths
    # Optimal solution
    ax.plot(x_opt[0], x_opt[1], "o")
    # Path
    ax.plot(x_path[:, 0], x_path[:, 1], "o", color="w", markersize=10, zorder=3)
    ax.plot(x_path[:, 0], x_path[:, 1], "o", color=my_brown, markersize=6, zorder=3)
    # Add arrow to the path
    ax.quiver(
    x_path[:-1, 0],
    x_path[:-1, 1],
    x_path[1:, 0] - x_path[:-1, 0],
    x_path[1:, 1] - x_path[:-1, 1],
    color=my_brown,
    scale_units="xy",
    angles="xy",
    scale=1,
    zorder=2,
    )

    # constraint
    ax.plot(x_con, y_con1, "-", color="k", alpha=0.6, zorder=0)
    ax.fill_between(x_con, y_con1, y2=1.5, facecolor=my_purple, alpha=0.3, zorder=0)



    ax.plot(x_path[0, 0], x_path[0, 1], "s", color="w", markersize=10, zorder=4) 
    ax.plot(x_path[0, 0], x_path[0, 1], "s", color=my_brown, markersize=8, zorder=4)
    ax.plot(x_path[-1, 0], x_path[-1, 1], "D", color="w", markersize=10, zorder=4) 
    ax.plot(x_path[-1, 0], x_path[-1, 1], "D", color=my_brown, markersize=8, zorder=4)


    ax.set_yticks([1.5,2.5, 3.4, 4.5,5.5,6.5,7.5])
    ax.set_xticks([0.1,0.5, 1.0, 1.5, 2.0])


    plt.tight_layout()
    plt.savefig("contour_test.pdf")