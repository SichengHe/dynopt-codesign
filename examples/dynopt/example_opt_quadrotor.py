# Closed loop system
import numpy as np
import scipy.interpolate
import CL
import matplotlib.pyplot as plt
from quadrotor_settings import res_funcs_generator, funcs_generator, QuadrotorSteadyHoverWrapper

import time as time_package
import copy

# ------------------
# Parameters
# ------------------
# initial states (x, y, theta, vx, vy, theta_vel)
x_init = np.array([1, 1, 0.1, 0.5, 0.3, 0.05])

# initial design: optimized for minimum hovering power. Run quadrotor_openmdao_setup.py to get these.
twist = np.array([28.625901569164, 25.25668202258955, 20.762409457456656, 17.05302822457691, 14.166807345926632, 12.050792115126653, 10.687482715596696, 9.623772899617274, 8.73787265213354, 7.830294208487746])
chord_by_R = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.3102845037371853, 0.24447688356440495, 0.17144033633125266, 0.05])   # chord normalized by rotor radius
d_init = np.concatenate((np.deg2rad(twist), chord_by_R))
min_hover_power = 99.87704847408797   # W

# We do multiobjective optimization of LQR cost and hovering power by the epsilon-constraint method
# Here, set the upper bound on the hovering power for CCD. This will be used for the constraint.
hover_power_upper_bound = min_hover_power * 1.03

num_cp = len(twist)
if num_cp != len(chord_by_R):
    raise RuntimeError("twist and chord control points must have the same length")

# Q and R values for LQR cost function
ndof = 6
nctrl = 2
Q = np.eye(ndof)
R = np.eye(nctrl) * 0.01   # set small penalty for control (rotor speed) because it doensn't make much sense to penalize [rotor_speed - rotor_speed_equilibrium]^2 ?

# Time setting
# TODO: set appropriate time step and horizon
dt = 0.05
T = 10.0   # probably T=10.0 is enough
N = int(T / dt)

# Provide a close steady-state initial guess
theta_0 = np.array([0.00000, 0.00000, 0.00000, 0.00000, 472.38, 472.38])
# theta_1 = np.array([0.0, 0.0, 0.0, 0.0, 300.0, 300.0]

# Generate the residual forms
res, res_reduced = res_funcs_generator(num_cp=num_cp)

# res_1 = res_reduced.compute(theta_1, d_init, t=0.)
res_0 = res_reduced.compute(theta_0, d_init, t=0.)

# print("res_0", res_0, "res_1", res_1)
print("res_0", res_0)

# setup steady hover power model
steady_hover_wrapper = QuadrotorSteadyHoverWrapper(num_cp=num_cp)

# --- setup problem & some initial solve ---
if 1:
    # Generate the closed loop obj using the residuals
    cl = CL.CL(d_init, res, res_reduced, Q, R, T, N, x_init)

    # Generate the implicit funciton format
    imp_FoI_cl = funcs_generator(d_init, cl, theta_0)

    # set design and reset cache
    imp_FoI_cl.set_design(d_init)

    # Solve
    t_start = time_package.time()
    imp_FoI_cl.solve(theta_0=theta_0)
    cost_init = imp_FoI_cl.compute()
    cost_history_init = copy.deepcopy(imp_FoI_cl.FoI_cl.FoI_val_arr)
    print('analysis time:', time_package.time() - t_start)

    # Compute the objective derivative
    # t_start = time_package.time()
    # imp_FoI_cl.solve_adjoint()
    # dfdx = imp_FoI_cl.compute_grad_design()
    # print('\ndf/dx=', dfdx)
    # print('\ntotal derivatives time:', time_package.time() - t_start)
    # print('-------------\n')

    # --- get state solution ---
    # get state history
    state_his_init = imp_FoI_cl.w
    final_states = state_his_init[:, -1]
    print('Initial design | final states:', final_states)
    print('Initial design | cost:', cost_init)
    # print('\nimp_FoI_cl.theta', imp_FoI_cl.theta)

    # plot state history
    # fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    # times = np.linspace(0, T, N)
    # axs[0, 0].plot(times, state_his_init[0, :])
    # axs[0, 0].set_ylabel('x')
    # axs[1, 0].plot(times, state_his_init[1, :])
    # axs[1, 0].set_ylabel('y')
    # axs[2, 0].plot(times, state_his_init[2, :])
    # axs[2, 0].set_ylabel('theta')
    # axs[0, 1].plot(times, state_his_init[3, :])
    # axs[0, 1].set_ylabel('vx')
    # axs[1, 1].plot(times, state_his_init[4, :])
    # axs[1, 1].set_ylabel('vy')
    # axs[2, 1].plot(times, state_his_init[5, :])
    # axs[2, 1].set_ylabel('theta_vel')

    # # reference line
    # for ax in axs.flatten():
    #     ax.plot(times, np.zeros_like(times), color='gray', lw=0.5)

    # plt.show()


# --- optimization ---
if 1:
    k_a = 0
    k_s = 0

    def objfunc(xdict):

        global imp_FoI_cl
        global k_a

        k_a += 1
        print("# function eval:", k_a)

        theta_0 = np.array([0.00000, 0.00000, 0.00000, 0.00000, 472.38, 472.38])

        # Extract the design var
        x = xdict["xvars"]

        # Solve the equation
        imp_FoI_cl.set_design(x)
        imp_FoI_cl.solve(theta_0=theta_0)
        cost = imp_FoI_cl.compute()
        obj = cost

        # Steady hover power constraint
        power_hover = steady_hover_wrapper.compute_power(x)
        power_constraint = power_hover - hover_power_upper_bound   # <= 0

        # Set the objective function and con
        funcs = {}
        funcs["obj"] = obj
        funcs["con_power"] = power_constraint   # must be <= 0

        # Set failure flag
        fail = False

        return funcs, fail

    def sens(xdict, funcs):

        global imp_FoI_cl
        global k_s

        k_s += 1
        print("# function sens:", k_s)

        theta_0 = np.array([0.00000, 0.00000, 0.00000, 0.00000, 472.38, 472.38])

        # Extract the design var
        x = xdict["xvars"]

        # Solve the equation
        imp_FoI_cl.set_design(x)
        imp_FoI_cl.solve(theta_0=theta_0)

        # Compute the objective derivative
        imp_FoI_cl.solve_adjoint()
        dfdx = imp_FoI_cl.compute_grad_design()

        # Derivatives of steady hover power constraint
        dpower_dx = steady_hover_wrapper.compute_power_grad(x)

        # Set the objective function and con derivative
        x = xdict["xvars"]
        funcsSens = {
            "obj": {"xvars": dfdx},
            "con_power": {"xvars": dpower_dx}
        }

        fail = False
        return funcsSens, fail

    # ========================================
    #    Optimization problem setup
    # ========================================

    from pyoptsparse import OPT, Optimization

    # Optimization Object
    optProb = Optimization("Control co-desgin", objfunc)

    # design variables = [twist_cp, chord_by_R_cp]
    x0 = d_init

    # Design Variables
    n_cp = len(twist)
    twist_ub = 40 * np.pi / 180   # rad
    twist_lb = 5 * np.pi / 180
    chord_by_R_ub = 0.35    # Chord UB = 35% of the radius
    chord_by_R_lb = 0.05    # Chord LB = 5% of the radius
    lower = [twist_lb for i in range(n_cp)] + [chord_by_R_lb for i in range(n_cp)]
    upper = [twist_ub for i in range(n_cp)] + [chord_by_R_ub for i in range(n_cp)]
    value = x0
    ref = np.array([20 * np.pi / 180 for i in range(n_cp)] + [0.3 for i in range(n_cp)])   # reference value for scaling
    optProb.addVarGroup("xvars", len(x0), lower=lower, upper=upper, value=value, scale=1 / ref)

    # Objective
    optProb.addObj("obj", scale=0.2)

    # Constraint on steady hover power
    optProb.addCon("con_power", lower=None, upper=0.0, scale=0.1)

    # Check optimization problem:
    print(optProb)

    # Optimizer
    optimizer = "snopt"
    # optimizer = "ipopt"
    optOptions = {}
    optOptions["Major iterations limit"] = 100
    optOptions["Major optimality tolerance"] = 1e-6   # probably cannot converge very tightly because we do FD for some part
    optOptions["Major feasibility tolerance"] = 1e-6
    optOptions["Verify level"] = -1
    optOptions["Nonderivative linesearch"] = 1
    optOptions['Function precision'] = 1e-7
    optOptions['Hessian full memory'] = 1
    optOptions['Hessian frequency'] = 100

    opt = OPT(optimizer, options=optOptions)

    # Solution
    histFileName = "%s_CCD.hst" % optimizer

    sol = opt(optProb, sens=sens, storeHistory=histFileName)

    # Check Solution
    print(sol)

    # get optimized design
    d_opt = sol.xStar['xvars']
    twist_opt = d_opt[:n_cp]
    chord_by_R_opt = d_opt[n_cp:]
    print('Optimal twist:', np.rad2deg(twist_opt), 'deg')
    print('Optimal chord/R:', chord_by_R_opt)

    # -------------------------------
    # plot optimized CL trajectory
    # -------------------------------
    # compute the optimal trajectory
    cl = CL.CL(d_opt, res, res_reduced, Q, R, T, N, x_init)
    imp_FoI_cl = funcs_generator(d_init, cl, theta_0)
    imp_FoI_cl.set_design(d_opt)
    imp_FoI_cl.solve(theta_0=theta_0)
    cost_final = imp_FoI_cl.compute()
    cost_history_opt = copy.deepcopy(imp_FoI_cl.FoI_cl.FoI_val_arr)
    state_his_opt = imp_FoI_cl.w
    final_states = state_his_opt[:, -1]
    print('Optimal design | final states:', final_states)
    print('Optimal design | cost:', cost_final)

    # plot state history
    color_init = 'C0'
    color_opt = 'C1'
    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    times = np.linspace(0, T, N)
    axs[0, 0].plot(times, state_his_init[0, :], color=color_init)
    axs[0, 0].plot(times, state_his_opt[0, :], color=color_opt)
    axs[0, 0].set_ylabel('x')
    axs[1, 0].plot(times, state_his_init[1, :], color=color_init)
    axs[1, 0].plot(times, state_his_opt[1, :], color=color_opt)
    axs[1, 0].set_ylabel('y')
    axs[2, 0].plot(times, state_his_init[2, :], color=color_init)
    axs[2, 0].plot(times, state_his_opt[2, :], color=color_opt)
    axs[2, 0].set_ylabel(r'$\theta$')
    axs[0, 1].plot(times, state_his_init[3, :], color=color_init)
    axs[0, 1].plot(times, state_his_opt[3, :], color=color_opt)
    axs[0, 1].set_ylabel(r'$v_x$')
    axs[1, 1].plot(times, state_his_init[4, :], color=color_init)
    axs[1, 1].plot(times, state_his_opt[4, :], color=color_opt)
    axs[1, 1].set_ylabel(r'$v_y$')
    axs[2, 1].plot(times, state_his_init[5, :], color=color_init)
    axs[2, 1].plot(times, state_his_opt[5, :], color=color_opt)
    axs[2, 1].set_ylabel(r'$\omega$')
    # reference line
    for ax in axs.flatten():
        ax.plot(times, np.zeros_like(times), color='gray', lw=0.5)
    # hide xticklabels
    axs[0, 0].set_xticklabels([])
    axs[0, 1].set_xticklabels([])
    axs[1, 0].set_xticklabels([])
    axs[1, 1].set_xticklabels([])
    axs[2, 0].set_xlabel('Time [s]')
    axs[2, 1].set_xlabel('Time [s]')
    fig.tight_layout()

    plt.savefig('quadrotor_state_history.pdf', bbox_inches='tight')
    plt.show()

    # plot blade design
    twist_init = d_init[:n_cp]
    chord_by_R_init = d_init[n_cp:]
    spanwise_loc = np.linspace(0.15, 1.0, n_cp)   # hardcoded for 15% hub radius
    # get splines
    spanwise_loc_plot = np.linspace(0.15, 1.0, 100)
    twist_spline_init = scipy.interpolate.Akima1DInterpolator(spanwise_loc, twist_init)(spanwise_loc_plot)
    chord_spline_init = scipy.interpolate.Akima1DInterpolator(spanwise_loc, chord_by_R_init)(spanwise_loc_plot)
    twist_spline_opt = scipy.interpolate.Akima1DInterpolator(spanwise_loc, twist_opt)(spanwise_loc_plot)
    chord_spline_opt = scipy.interpolate.Akima1DInterpolator(spanwise_loc, chord_by_R_opt)(spanwise_loc_plot)

    fig, axs = plt.subplots(2, 1, figsize=(6, 6))
    axs[0].plot(spanwise_loc_plot, np.rad2deg(twist_spline_init), color=color_init)
    axs[0].plot(spanwise_loc_plot, np.rad2deg(twist_spline_opt), color=color_opt)
    axs[0].set_ylabel('Twist [deg]')
    axs[0].set_xticklabels([])

    axs[1].plot(spanwise_loc_plot, chord_spline_init, color=color_init)
    axs[1].plot(spanwise_loc_plot, chord_spline_opt, color=color_opt)
    axs[1].set_ylabel('Chord/R')
    axs[1].set_xlabel('Span/R')

    plt.savefig('quadrotor_blade_design.pdf', bbox_inches='tight')
    plt.show()

    # save solutions to pickle file
    import pickle
    results = {}
    results['time'] = times
    # state history
    results['state_his_init'] = state_his_init
    results['state_his_opt'] = state_his_opt
    # rotor geometry
    results['spanwise_loc'] = spanwise_loc_plot  # normalized by rotor radius
    results['twist_init'] = np.rad2deg(twist_spline_init)
    results['twist_opt'] = np.rad2deg(twist_spline_opt)
    results['chord_init'] = chord_spline_init  # normalized by rotor radius
    results['chord_opt'] = chord_spline_opt
    # hover power required
    results['min_hover_power'] = min_hover_power   # minimum hovering power (from static optimization)
    results['hover_power_upper_bound'] = hover_power_upper_bound    # upper bound on hovering power for CCD. This is basially multiobjective optimizaiton epsilon settings
    # LQR cost
    results['cost_init'] = cost_init   # cost at final time
    results['cost_final'] = cost_final   # cost at final time
    results['cost_history_init'] = cost_history_init   # cost time history
    results['cost_history_opt'] = cost_history_opt   # cost time history

    with open('quadrotor_opt_result.pkl', 'wb') as f:
        pickle.dump(results, f)
