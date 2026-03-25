"""
Compare adjoint vs finite-difference gradient computation for the LQR quadrotor
co-design optimization (same problem as example_opt_quadrotor.py).

Runs the optimization twice (adjoint, then FD) and plots:
  - time per iteration vs iteration number
  - cumulative wall time vs iteration number
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../dymos_quadrotor'))

import numpy as np
import time as time_package
import copy
import matplotlib.pyplot as plt
from cycler import cycler

from dynopt import CL
# from quadrotor_settings import res_funcs_generator, funcs_generator, QuadrotorSteadyHoverWrapper
from pyoptsparse import OPT, Optimization



# ============================================================
#  Problem parameters (identical to example_opt_quadrotor.py)
# ============================================================
# x_init = np.array([1, 1, 0.1, 0.5, 0.3, 0.05])

# twist = np.array([28.625901569164, 25.25668202258955, 20.762409457456656, 17.05302822457691,
#                   14.166807345926632, 12.050792115126653, 10.687482715596696, 9.623772899617274,
#                   8.73787265213354, 7.830294208487746])
# chord_by_R = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.3102845037371853,
#                        0.24447688356440495, 0.17144033633125266, 0.05])
# d_init = np.concatenate((np.deg2rad(twist), chord_by_R))
# min_hover_power = 99.87704847408797
# hover_power_upper_bound = min_hover_power * 1.03

# num_cp = len(twist)
# n_dv = 2 * num_cp

# Q = np.eye(6)
# R = np.eye(2) * 0.01

# dt = 0.05
# T = 10.0
# N = int(T / dt)

# theta_0 = np.array([0.00000, 0.00000, 0.00000, 0.00000, 472.38, 472.38])

# # FD step sizes (relative, same scale as internal dynamics FD)
# d_ref = np.concatenate([np.full(num_cp, 20 * np.pi / 180), np.full(num_cp, 0.3)])
# h_fd = d_ref * 1e-5

# # ============================================================
# #  Shared resources (setup once, reused by both runs)
# # ============================================================
# print("Setting up residual functions (shared by both runs)...")
# res, res_reduced = res_funcs_generator(num_cp=num_cp)
# steady_hover_wrapper = QuadrotorSteadyHoverWrapper(num_cp=num_cp)


# ============================================================
#  Optimization runner
# ============================================================
def run_optimization(use_adjoint):
    """
    Run the CCD optimization with either adjoint or FD gradients.

    Returns
    -------
    iter_times : list of float
        Wall time for each major iteration (objfunc + sens).
    obj_history : list of float
        Objective value at each major iteration.
    sol : pyoptsparse solution object
    """
    label = "ADJOINT" if use_adjoint else "FD"
    print(f"\n{'='*60}")
    print(f"  Starting {label} optimization")
    print(f"{'='*60}")

    # Fresh closed-loop object for each run
    cl = CL.CL(d_init, res, res_reduced, Q, R, T, N, x_init)
    imp_FoI_cl = funcs_generator(d_init, cl, theta_0)
    imp_FoI_cl.set_design(d_init)

    # Timing state (mutable via list to allow closure mutation)
    t_objfunc_start = [None]
    iter_times = []
    obj_history = []

    # ----------------------------------------------------------
    def objfunc(xdict):
        t_objfunc_start[0] = time_package.time()

        x = xdict["xvars"]
        imp_FoI_cl.set_design(x)
        imp_FoI_cl.solve(theta_0=theta_0)
        cost = imp_FoI_cl.compute()

        power_hover = steady_hover_wrapper.compute_power(x)
        power_constraint = power_hover - hover_power_upper_bound

        funcs = {"obj": cost, "con_power": power_constraint}
        return funcs, False

    # ----------------------------------------------------------
    def sens_adjoint(xdict, funcs):
        x = xdict["xvars"]
        imp_FoI_cl.set_design(x)
        imp_FoI_cl.solve(theta_0=theta_0)
        imp_FoI_cl.solve_adjoint()
        dobj_dx = imp_FoI_cl.compute_grad_design()

        dpower_dx = steady_hover_wrapper.compute_power_grad(x)

        funcsSens = {
            "obj": {"xvars": dobj_dx},
            "con_power": {"xvars": dpower_dx}
        }

        elapsed = time_package.time() - t_objfunc_start[0]
        iter_times.append(elapsed)
        obj_history.append(funcs["obj"])
        print(f"  [{label}] iter {len(iter_times):3d} | obj={funcs['obj']:.6f} | "
              f"time={elapsed:.2f}s")

        return funcsSens, False

    # ----------------------------------------------------------
    def sens_fd(xdict, funcs):
        x = xdict["xvars"].copy()

        # Ensure base-point solve is current
        imp_FoI_cl.set_design(x)
        imp_FoI_cl.solve(theta_0=theta_0)
        obj_0 = imp_FoI_cl.compute()
        power_0 = steady_hover_wrapper.compute_power(x)

        dobj_dx = np.zeros(n_dv)
        dpower_dx = np.zeros(n_dv)

        for i in range(n_dv):
            x_pert = x.copy()
            x_pert[i] += h_fd[i]
            imp_FoI_cl.set_design(x_pert)
            imp_FoI_cl.solve(theta_0=theta_0)
            cost_pert = imp_FoI_cl.compute()
            power_pert = steady_hover_wrapper.compute_power(x_pert)
            dobj_dx[i] = (cost_pert - obj_0) / h_fd[i]
            dpower_dx[i] = (power_pert - power_0) / h_fd[i]

        # Reset to base point
        imp_FoI_cl.set_design(x)
        imp_FoI_cl.solve(theta_0=theta_0)

        funcsSens = {
            "obj": {"xvars": dobj_dx},
            "con_power": {"xvars": dpower_dx}
        }

        elapsed = time_package.time() - t_objfunc_start[0]
        iter_times.append(elapsed)
        obj_history.append(funcs["obj"])
        print(f"  [{label}] iter {len(iter_times):3d} | obj={funcs['obj']:.6f} | "
              f"time={elapsed:.2f}s")

        return funcsSens, False

    # ----------------------------------------------------------
    #  pyoptsparse problem setup
    # ----------------------------------------------------------
    optProb = Optimization("LQR quadrotor co-design", objfunc)

    n_cp = num_cp
    twist_lb = 5 * np.pi / 180
    twist_ub = 40 * np.pi / 180
    chord_lb = 0.05
    chord_ub = 0.35
    lower = [twist_lb] * n_cp + [chord_lb] * n_cp
    upper = [twist_ub] * n_cp + [chord_ub] * n_cp
    ref = np.concatenate([np.full(n_cp, 20 * np.pi / 180), np.full(n_cp, 0.3)])
    optProb.addVarGroup("xvars", n_dv, lower=lower, upper=upper,
                        value=d_init, scale=1 / ref)

    optProb.addObj("obj", scale=0.2)
    optProb.addCon("con_power", lower=None, upper=0.0, scale=0.1)

    # IPOPT settings
    optOptions = {
        "max_iter": 20,
        "tol": 1e-4,
        # "constr_viol_tol": 1e-6,
        "print_level": 5,
        # "mu_strategy": "adaptive",
        "hessian_approximation": "limited-memory",   # L-BFGS; avoids needing exact Hessian
    }
    opt = OPT("ipopt", options=optOptions)

    sens_func = sens_adjoint if use_adjoint else sens_fd
    sol = opt(optProb, sens=sens_func)

    print(f"\n{label} optimization finished: {len(iter_times)} iterations")
    return iter_times, obj_history, sol


# ============================================================
#  Run both optimizations
# ============================================================
# iter_times_adj, obj_adj, sol_adj = run_optimization(use_adjoint=True)
# iter_times_fd,  obj_fd,  sol_fd  = run_optimization(use_adjoint=False)

# # Save results so the optimization does not need to be re-run
# np.savez('compare_FD_adjoint_results.npz',
#          iter_times_adj=iter_times_adj, obj_adj=obj_adj,
#          iter_times_fd=iter_times_fd,  obj_fd=obj_fd)
# print("Saved results to compare_FD_adjoint_results.npz")

# Load saved results
data = np.load('compare_FD_adjoint_results.npz')
iter_times_adj = data['iter_times_adj']
obj_adj        = data['obj_adj']
iter_times_fd  = data['iter_times_fd']
obj_fd         = data['obj_fd']


# ============================================================
#  Plot
# ============================================================

# plot settings
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

fontsize = 10
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["font.size"] = fontsize
plt.rcParams["axes.titlesize"] = fontsize
plt.rcParams["axes.labelsize"] = fontsize
plt.rcParams["xtick.labelsize"] = fontsize
plt.rcParams["ytick.labelsize"] = fontsize

plt.rcParams["mathtext.rm"] = "serif"
plt.rcParams["mathtext.it"] = "serif:italic"
plt.rcParams["mathtext.bf"] = "serif:bold"
plt.rcParams["mathtext.fontset"] = "custom"

# override default colors (C0, C1, ...)
custom_colors = ['#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263']
plt.rcParams['axes.prop_cycle'] = cycler(color=custom_colors)



iters_adj = np.arange(1, len(iter_times_adj) + 1)
iters_fd  = np.arange(1, len(iter_times_fd) + 1)

avg_FD = np.mean(iter_times_fd)
avg_adj = np.mean(iter_times_adj)

fig, axs = plt.subplots(figsize=(3.5, 2))

# --- per-iteration time ---
axs.plot(iters_adj, iter_times_adj, 'C0o-', label='Adjoint', markersize=4)
axs.plot(iters_fd,  iter_times_fd,  'C1o-', label='FD', markersize=4)

axs.axhline(avg_adj, color='C0', alpha=0.3, linewidth=1.5)
axs.axhline(avg_FD,  color='C1', alpha=0.3, linewidth=1.5)

# Dummy line for legend entry
axs.plot([], [], color='black', alpha=0.3, linewidth=1.5, label='Average')

axs.set_xlim(0,20)
axs.set_ylim(0, 300)

axs.set_yticks([round(avg_FD), 150, round(avg_adj)])

axs.set_xlabel('Iteration Number')
axs.set_ylabel('Wall time [s]')
axs.legend(frameon=False)


fig.tight_layout()
plt.savefig('R1_journal/figure/compare_FD_adjoint.pdf', bbox_inches='tight')

# Summary
print("\n=== Summary ===")
print(f"Adjoint: {len(iter_times_adj)} iters, "
      f"avg {np.mean(iter_times_adj):.2f}s/iter, "
      f"total {sum(iter_times_adj):.1f}s")
print(f"FD:      {len(iter_times_fd)} iters, "
      f"avg {np.mean(iter_times_fd):.2f}s/iter, "
      f"total {sum(iter_times_fd):.1f}s")

# TOTAL FD: 5283 s
# TOTAL AJOINT: 1393 s