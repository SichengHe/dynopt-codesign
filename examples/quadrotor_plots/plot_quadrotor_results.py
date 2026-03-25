"""
Plot quadrotor CCD optimization results

This scripts loads optimization results from quadrotor_opt_result_***.pkl
Run `example_quadrotor_opt.py` to generate these pkl files.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from cycler import cycler

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
    

def _plot_state_history(axs, times, state_his, cost_his, color, lw):
    """ plot state time history in 7x1 subplots """
    axs[0].plot(times, state_his[0, :], color=color, lw=lw)  # x
    axs[1].plot(times, state_his[1, :], color=color, lw=lw)  # y
    axs[2].plot(times, state_his[2, :], color=color, lw=lw)  # theta
    axs[3].plot(times, state_his[3, :], color=color, lw=lw)  # vx
    axs[4].plot(times, state_his[4, :], color=color, lw=lw)  # vy
    axs[5].plot(times, state_his[5, :], color=color, lw=lw)  # omega
    axs[6].plot(times, cost_his, color=color, lw=lw)  # cost

def _get_airfoil(chord_scaler, twist):
    """ util function called by plot_blade_design_3D """
    # --- prepare airfoil ---
    # unit-chord airfoil shape (t/c=0.12)
    n_af_pts = 500
    t = 0.12
    af_x = np.linspace(0, 1, n_af_pts)
    af_y = t/.2*(.296*np.sqrt(af_x)-.126*af_x-.3516*af_x**2+.2843*af_x**3-.1015*af_x**4)  # upper surface
    ## af_y2 = -af_y1[::-1]
    af_x = np.concatenate((af_x[:-1], af_x[::-1]))  # full airfoil, TE on right
    af_y = np.concatenate((af_y[:-1], -af_y[::-1]))
    af_xy = np.vstack((af_x, af_y)) * chord_scaler   # scaling
    af_xy[0, :] -= chord_scaler * 0.25  # shift center to quarter chord
    # rotate 180 deg so that TE is on the right
    rot_mat = np.array([[-1, 0], [0, -1]])
    af_xy = np.dot(rot_mat, af_xy)

    # rotate by twist
    twist_rad = np.deg2rad(twist)
    rot_mat = np.array([[np.cos(twist_rad), -np.sin(twist_rad)], [np.sin(twist_rad), np.cos(twist_rad)]])
    af_xy_rotated = np.dot(rot_mat, af_xy)

    # get LE and TE
    xy_LE = af_xy_rotated[:, 0]
    xy_TE = af_xy_rotated[:, n_af_pts]

    return af_xy_rotated, xy_LE, xy_TE   # [xy, :]


def _plot_blade_design_3D(ax, radii, chord, twist, lw, color, view_angles=[50, -50, 0]):
    """ plot rotor blade design in 3D """

    # generate slices for airfoil visualization
    num_slices = 10
    r_slice = np.linspace(min(radii), max(radii), num_slices)
    c_slice = np.interp(r_slice, radii, chord)
    t_slice = np.interp(r_slice, radii, twist)

    # plot slices
    for i in range(num_slices):
        af_xy, _, _ = _get_airfoil(c_slice[i], t_slice[i])
        span_slice = r_slice[i] * np.ones_like(af_xy[0, :])
        ax.plot(af_xy[0, :], span_slice, af_xy[1, :], lw=lw, color=color)
    # END FOR

    # get LE and TE lines
    nr = len(radii)
    xy_le = np.zeros((2, nr))
    xy_te = np.zeros((2, nr))

    for i in range(nr):
        _, xy_le[:, i], xy_te[:, i] = _get_airfoil(chord[i], twist[i])
    # END FOR

    # LE / TE lines
    ax.plot(xy_le[0, :], radii, xy_le[1, :], lw=lw, color=color)
    ax.plot(xy_te[0, :], radii, xy_te[1, :], lw=lw, color=color)

    ax.set_aspect('equal', 'box')

    # hide grid and tics
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.view_init(elev=view_angles[0], azim=view_angles[1], roll=view_angles[2])


def plot_state_history(results_all):
    """
    Plot state history and flight path

    Parameters
    ----------
    results_all : list
        List of dicts, each dict containing results from a single optimization
    """

    # --- Flight path plot ---
    # fig1, ax = plt.subplots(figsize=(3.25, 2.5))
    # lw = 0.5   # line width

    # # plot initial solution. This is the same across all optimization cases
    # results = results_all[0]
    # x = results["state_his_init"][0, :]
    # y = results["state_his_init"][1, :]
    # ax.plot(x, y, color="gray", linewidth=lw)

    # # plot optimized flight paths
    # for i, results in enumerate(results_all):
    #     x = results["state_his_opt"][0, :]
    #     y = results["state_his_opt"][1, :]
    #     ax.plot(x, y, color=f'C{i}', linewidth=lw)
    
    # ax.set_xlabel("x [m]")
    # ax.set_ylabel("y [m]")
    # ax.set_aspect("equal")
    # # ax.grid(True)
    # fig1.tight_layout()
    # plt.savefig("quadrotor_flight_paths.pdf", bbox_inches="tight")

    # --- State time history plots ---
    # plot initial solution
    fig2, axs = plt.subplots(7, 1, figsize=(3.25, 7))
    results = results_all[0]
    times = results["time"]
    _plot_state_history(axs, times, results["state_his_init"], results["cost_history_init"], color='gray', lw=1)

    # plot optimized solutions
    for i, results in enumerate(results_all):
        _plot_state_history(axs, times, results["state_his_opt"], results["cost_history_opt"], color=f'C{i}', lw=1)

    # plot reference lines to 0 states
    for ax in axs.flatten()[:-1]:
        ax.plot(times, np.zeros_like(times), color='lightgray', lw=0.5, zorder=-2)

    # legends
    axs[-1].plot([], [], color='gray', label='Sequential')
    axs[-1].plot([], [], color='C0', label=r'$\epsilon=0.005$')
    axs[-1].plot([], [], color='C1', label=r'$\epsilon=0.01$')
    axs[-1].plot([], [], color='C2', label=r'$\epsilon=0.02$')
    axs[-1].plot([], [], color='C3', label=r'$\epsilon=0.03$')
    axs[-1].legend(loc='lower right', fontsize=8.5, ncol=2, labelspacing=0.1)

    labelpad = 10
    axs[0].set_ylabel('x', rotation=0, labelpad=labelpad, ha='left')
    axs[1].set_ylabel('y', rotation=0, labelpad=labelpad, ha='left')
    axs[2].set_ylabel(r'$\theta$', rotation=0, labelpad=labelpad, ha='left')
    axs[3].set_ylabel(r'$v_x$', rotation=0, labelpad=labelpad, ha='left')
    axs[4].set_ylabel(r'$v_y$', rotation=0, labelpad=labelpad, ha='left')
    axs[5].set_ylabel(r'$\omega$', rotation=0, labelpad=labelpad, ha='left')
    axs[6].set_ylabel('Cost', rotation=0, labelpad=labelpad, ha='left')

    # y ticks
    axs[0].set_yticks([0, 0.5, 1.0])
    axs[1].set_yticks([0, 0.5, 1.0])
    axs[2].set_yticks([0, 0.1, 0.2])
    axs[3].set_yticks([-0.5, 0, 0.5])
    axs[4].set_yticks([-0.3, 0, 0.3])
    axs[5].set_yticks([-0.3, 0, 0.3])
    axs[6].set_yticks([0, 3, 6])

    # x axis range
    xlim = [0, 10]
    for ax in axs:
        ax.set_xlim(xlim)

    # hide xticklabels
    axs[0].set_xticklabels([])
    axs[1].set_xticklabels([])
    axs[2].set_xticklabels([])
    axs[3].set_xticklabels([])
    axs[4].set_xticklabels([])
    axs[5].set_xticklabels([])
    axs[6].set_xlabel('Time [s]')
    # fig2.tight_layout()
    fig2.align_ylabels()
    plt.savefig("quadrotor_state_history.pdf", bbox_inches="tight")


def plot_rotor_design(results_all):
    """
    Plot rotor design
    1. chord and twist distributions
    2. 3D plots of geometries
    """

    # --- 2D plot (chord and twist) ---
    fig, axs = plt.subplots(2, 1, figsize=(3.25, 2.5))
    lw = 0.7

    # plot initial design (steady power-minimum design)
    results = results_all[0]
    spanwise_loc = results["spanwise_loc"]
    chord = results["chord_init"]  # normalized by rotor radius
    twist = results["twist_init"]
    axs[0].plot(spanwise_loc, chord, color='gray', lw=lw)
    axs[1].plot(spanwise_loc, twist, color='gray', lw=lw)

    # plot optimized designs
    for i, results in enumerate(results_all):
        chord = results["chord_opt"]
        twist = results["twist_opt"]
        axs[0].plot(spanwise_loc, chord, color=f'C{i}', lw=lw)
        axs[1].plot(spanwise_loc, twist, color=f'C{i}', lw=lw)

    # legends
    # axs[-1].plot([], [], color='gray', label='Sequential')
    # axs[-1].plot([], [], color='C0', label=r'$\epsilon=0.005$')
    # axs[-1].plot([], [], color='C1', label=r'$\epsilon=0.01$')
    # axs[-1].plot([], [], color='C2', label=r'$\epsilon=0.02$')
    # axs[-1].plot([], [], color='C3', label=r'$\epsilon=0.03$')
    # axs[-1].legend(loc='lower right', fontsize=8, ncol=2, labelspacing=0)

    # x axis range
    xlim = [0.15, 1]
    for ax in axs:
        ax.set_xlim(xlim)

    axs[0].set_ylabel('Chord/R')
    axs[1].set_ylabel('Twist [deg]')
    axs[1].set_xlabel('Span/R')
    fig.tight_layout()
    fig.align_ylabels()
    plt.savefig("quadrotor_rotor_design.pdf", bbox_inches="tight")

    # --- 3D rotor geometry ---
    fig = plt.figure(figsize=(3.25, 4))
    axs = []
    n = 5
    spacing = -0.4  # Vertical gap between plots
    height = (1 - spacing * (n + 1)) / n  # Each subplot height

    for i in range(n):
        bottom = 1 - (i + 1) * (height + spacing) + spacing
        ax = fig.add_axes([0.0, bottom, 1.0, height], projection='3d')
        axs.append(ax)

    view_angles = [43, -40, 0]

    # plot baseline rotor
    results = results_all[0]
    spanwise_loc = results["spanwise_loc"]
    chord_base = results["chord_init"]
    twist_base = results["twist_init"]
    _plot_blade_design_3D(axs[0], spanwise_loc, chord_base, twist_base, lw=1, color='gray', view_angles=view_angles)

    # plot optimized rotor
    for i, results in enumerate(results_all):
        chord = results["chord_opt"]
        twist = results["twist_opt"]
        _plot_blade_design_3D(axs[i + 1], spanwise_loc, chord_base, twist_base, lw=0.7, color='gray', view_angles=view_angles)   # overlay baseline design
        _plot_blade_design_3D(axs[i + 1], spanwise_loc, chord, twist, lw=1, color=f'C{i}', view_angles=view_angles)

    for ax in axs:
        ax.set_facecolor('none')

    # y labels
    labels = ['Sequential', r'$\epsilon=0.005$', r'$\epsilon=0.01$', r'$\epsilon=0.02$', r'$\epsilon=0.03$']
    for i, label in enumerate(labels):
        y_pos = 1 - (i + 0.5) * (height + spacing) + spacing + 0.2  # center of each plot
        fig.text(0.15, y_pos, label, va='center', ha='right', fontsize=fontsize)

    plt.savefig("quadrotor_rotor_design_3D.pdf", bbox_inches="tight")   # , pad_inches=0)


def plot_pareto_front(results_all):
    """
    Plot pareto front (hover power vs. LQR cost)
    """

    lqr_cost = []
    hover_power = []

    # --- get solutions ---
    # baseline solution (minimum power, steady optimization)
    results = results_all[0]
    lqr_cost.append(results["cost_init"])
    hover_power.append(results["min_hover_power"])

    # optimized solutions
    for results in results_all:
        lqr_cost.append(results["cost_final"])
        hover_power.append(results["hover_power_upper_bound"])

    # normalize w.r.t. baseline solution
    lqr_cost = np.array(lqr_cost) / lqr_cost[0]
    hover_power = np.array(hover_power) / hover_power[0]

    # --- plot pareto front ---
    fig, ax = plt.subplots(figsize=(3.25, 2.5))
    lw = 1.5
    ax.plot(lqr_cost, hover_power, '-', color='lightgray', lw=lw)
    # plot points with color separately
    for i in range(len(lqr_cost)):
        color = 'gray' if i == 0 else f'C{i-1}'
        ax.plot(lqr_cost[i], hover_power[i], 'o', color=color, markersize=6)

    ax.set_xlabel('Normalized LQR cost')
    ax.set_ylabel('Normalized hovering power')

    ax.set_yticks([1.0, 1.01, 1.02, 1.03])

    # text annotations
    ax.text(1.007, 1.002, 'Sequential', color='gray', ha='center', va='bottom', fontsize=fontsize)
    ax.text(0.955, 1.007, r'$\epsilon=0.005$', color='C0', ha='left', va='center', fontsize=fontsize)
    ax.text(0.94, 1.012, r'$\epsilon=0.01$', color='C1', ha='left', va='center', fontsize=fontsize)
    ax.text(0.92, 1.02, r'$\epsilon=0.02$', color='C2', ha='left', va='center', fontsize=fontsize)
    ax.text(0.907, 1.03, r'$\epsilon=0.03$', color='C3', ha='left', va='center', fontsize=fontsize)

    plt.savefig("quadrotor_pareto_front.pdf", bbox_inches="tight")


def plot_pareto_front_with_geometries(results_all):
    """
    Plot pareto front (hover power vs. LQR cost) + rotor 3D geometries
    """

    lqr_cost = []
    hover_power = []

    # --- get solutions ---
    # baseline solution (minimum power, steady optimization)
    results = results_all[0]
    lqr_cost.append(results["cost_init"])
    hover_power.append(results["min_hover_power"])

    # optimized solutions
    for results in results_all:
        lqr_cost.append(results["cost_final"])
        hover_power.append(results["hover_power_upper_bound"])

    # normalize w.r.t. baseline solution
    lqr_cost = np.array(lqr_cost) / lqr_cost[0]
    hover_power = np.array(hover_power) / hover_power[0]

    # --- plot pareto front ---
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    lw = 1.5
    ax.plot(lqr_cost, hover_power, '-', color='lightgray', lw=lw)
    # plot points with color separately
    for i in range(len(lqr_cost)):
        color = 'gray' if i == 0 else f'C{i-1}'
        ax.plot(lqr_cost[i], hover_power[i], 'o', color=color, markersize=6)

    ax.set_xlabel('Normalized LQR cost')
    ax.set_ylabel('Normalized hovering power')

    ax.set_yticks([1.0, 1.01, 1.02, 1.03])

    # --- overlay 3D rotor blade geometries ---
    ### transform = mtransforms.blended_transform_factory(ax.transData, ax.transData)

    # set embedded figure size and locations for rotor geometries
    axs_inset = []
    xlim = [0.9, 1.0]
    ylim = [1.0, 1.03]
    # adjustment to the inset axes position
    x0_delta = [-0.25, -0.15, -0.15, -0.13, -0.15]
    y0_delta = [0.0, -0.04, -0.04, -0.08, -0.1]
    subfigure_size = 0.5

    for i in range(len(lqr_cost)):
        # lower-left corner of the inset axes
        x0 = (lqr_cost[i] - xlim[0]) / (xlim[1] - xlim[0])
        y0 = (hover_power[i] - ylim[0]) / (ylim[1] - ylim[0])
        x0 += x0_delta[i]
        y0 += y0_delta[i]
        axs_inset.append(ax.inset_axes([x0, y0, subfigure_size, subfigure_size], projection='3d', zorder=-1))
        ### axs_inset[i].axis("off")
    # END FOR

    # plot baseline rotor
    results = results_all[0]
    spanwise_loc = results["spanwise_loc"]
    chord_base = results["chord_init"]
    twist_base = results["twist_init"]
    _plot_blade_design_3D(axs_inset[0], spanwise_loc, chord_base, twist_base, lw=0.8, color='gray')

    # plot optimized rotor
    for i, results in enumerate(results_all):
        chord = results["chord_opt"]
        twist = results["twist_opt"]
        _plot_blade_design_3D(axs_inset[i + 1], spanwise_loc, chord_base, twist_base, lw=0.6, color='gray')   # overlay baseline design
        _plot_blade_design_3D(axs_inset[i + 1], spanwise_loc, chord, twist, lw=0.8, color=f'C{i}')

    # set transparent face colors for each inset figures
    for ax_inset in axs_inset:
        ax_inset.set_facecolor('none')

    # text annotations
    ax.text(1.007, 1.012, 'Sequential', color='gray', ha='center', va='bottom', fontsize=fontsize)
    ax.text(0.977, 1.015, r'$\epsilon=0.05$', color='C0', ha='left', va='center', fontsize=fontsize)
    ax.text(0.959, 1.021, r'$\epsilon=0.01$', color='C1', ha='left', va='center', fontsize=fontsize)
    ax.text(0.94, 1.03, r'$\epsilon=0.02$', color='C2', ha='left', va='center', fontsize=fontsize)
    ax.text(0.922, 1.04, r'$\epsilon=0.03$', color='C3', ha='left', va='center', fontsize=fontsize)
    
    # plt.tight_layout()

    plt.savefig("quadrotor_pareto_front_with_geometry.pdf", bbox_inches="tight")


if __name__ == "__main__":
    # load multiobjective optimization results from pickle files
    filenames = [
        "results/quadrotor_opt_result_power1005.pkl",
        "results/quadrotor_opt_result_power101.pkl",
        "results/quadrotor_opt_result_power102.pkl",
        "results/quadrotor_opt_result_power103.pkl",
    ]
 
    results_all = []
    for filename in filenames:
        with open(filename, "rb") as f:
            results = pickle.load(f)
            results_all.append(results)

    # plot state history
    # plot_state_history(results_all)

    # plot rotor design (2D plot and 3D plot)
    plot_rotor_design(results_all)

    # Pareto front
    # plot_pareto_front(results_all)

    # Pareto front + rotor geometries
    # plot_pareto_front_with_geometries(results_all)

    plt.show()
