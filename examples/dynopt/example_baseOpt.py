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

# ==================
# Analysis test
# ==================

# ------------------
# Base
# ------------------
m = 1.0
M = 5.0
L = 2.0

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
T = 30.0
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
imp_FoI_cl_base = funcs_generator(x, cl, theta_0)

# Solve
imp_FoI_cl_base.solve(theta_0 = theta_0)
imp_FoI_cl_base.compute()

t = np.linspace(dt, T, N)
w_base = np.zeros_like(imp_FoI_cl_base.cl.w)
for i in range(N):
    w_base[:, i] = imp_FoI_cl_base.cl.w[:, i] + imp_FoI_cl_base.cl.w_tgt[:]

I_base = imp_FoI_cl_base.FoI_cl.FoI_val_arr



# ------------------
# Opt
# ------------------
m = 1.0
M = 2.5
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
T = 30.0
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
imp_FoI_cl_opt = funcs_generator(x, cl, theta_0)

# Solve
imp_FoI_cl_opt.solve(theta_0 = theta_0)
imp_FoI_cl_opt.compute()

t = np.linspace(dt, T, N)
w_opt = np.zeros_like(imp_FoI_cl_opt.cl.w)
for i in range(N):
    w_opt[:, i] = imp_FoI_cl_opt.cl.w[:, i] + imp_FoI_cl_opt.cl.w_tgt[:]

I_opt = imp_FoI_cl_opt.FoI_cl.FoI_val_arr

# ------------------
# Compare Cost
# ------------------
If_base = I_base[-1]
If_opt = I_opt[-1]

# percent cost savings:
cost_savings = 100*(If_base - If_opt)/If_base
print(f'LQR Cost Saving: {cost_savings:.2f}%')

# ------------------
# Plot
# ------------------
#import niceplots

# Load niceplots
#niceplots.All()

# import colors
custom_colors = ['#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263']
c1 = custom_colors[0]
c2 = custom_colors[1]
c3 = custom_colors[2]
c4 = custom_colors[3]
c5 = custom_colors[4]
c6 = custom_colors[5]

# adjust sizes
fontsize = 16
labelpad = 80

# change defaults
plt.rcParams["font.size"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize

plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["mathtext.it"] = "serif:italic"
plt.rcParams["mathtext.bf"] = "serif:bold"
plt.rcParams["mathtext.fontset"] = "custom"

fig, ax = plt.subplots(3, figsize=(10, 5))
#plt.style.use(niceplots.get_style('james-light'))

# x
ax[0].plot(t, w_base[0, :], color=c1)
ax[0].plot(t, w_opt[0, :], color=c2)
ax[0].plot(t, np.ones_like(t), color='lightgray', lw=0.5, zorder=-2)
ax[0].set_ylabel('x', fontsize=fontsize,labelpad=labelpad, ha="left", rotation=0)
ax[0].yaxis.set_label_coords(-.16, .5)
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)


# theta
ax[1].plot(t, w_base[2, :], color=c1)
ax[1].plot(t, w_opt[2, :], color=c2)
ax[1].plot(t, np.pi*np.ones_like(t), color='lightgray', lw=0.5, zorder=-2)
ax[1].set_ylabel(r'$\theta$', fontsize=fontsize,labelpad=labelpad, ha="left", rotation=0)
ax[1].yaxis.set_label_coords(-.16, .5)
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)


# cost
ax[2].plot(t, I_base/100, color=c1,label='Baseline')
ax[2].plot(t, I_opt/100, color=c2,label='Optimized')
ax[2].plot(t, (I_base[-1]/100)*np.ones_like(t), color='lightgray', lw=0.5, zorder=-2)
ax[2].plot(t, (I_opt[-1]/100)*np.ones_like(t), color='lightgray', lw=0.5, zorder=-2)
ax[2].set_xlabel("Time [s]", rotation=0,fontsize=fontsize)
ax[2].set_ylabel(r"Cost", fontsize=fontsize, labelpad=labelpad, ha="left", rotation=0)
ax[2].yaxis.set_label_coords(-.16, .35)
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)

#fig.align_ylabels(ax[:])

# tick marks
ax[0].set_yticks([0,35])
ax[1].set_yticks([2,3.14])
ax[2].set_yticks([0,100,200])

ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
#ax[2].set_xticklabels([])

# legends
ax[-1].legend(loc='right', fontsize=14, ncol=1, labelspacing=0.1, frameon=False)



fig.tight_layout()
#plt.savefig("../../R0_journal/figure/optimized.pdf", format="pdf", bbox_inches="tight")
plt.savefig("optimized.pdf", format="pdf", bbox_inches="tight")