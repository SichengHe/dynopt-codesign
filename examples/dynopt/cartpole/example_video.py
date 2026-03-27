# Closed loop system
import os
import subprocess
import copy
from functools import partial
import numpy as np
import scipy.linalg
from scipy import integrate
import matplotlib.pyplot as plt
import niceplots
from cart_setting import r_plt, prpw_plt, prpx_plt, G_func, dtraceG_dx, trace_dprpw_plt_dx_T_prpw_bar, pG_func_px
from dynopt import CL
import plot_cart

def generate_video(x_design_dict, y0, video_name):

    # ==================
    # Run analsis
    # ==================

    nx = len(x_design_dict)

    yr = np.zeros(4)
    yr[0] = 1.0
    yr[2] = np.pi

    t0, t1 = 0, 10.0  # start and end
    dt = 0.1
    t = np.arange(t0, t1, dt)

    n = 4
    N = len(t)

    CL_obj = CL(
        yr,
        y0 - yr,
        r_plt,
        prpw_plt,
        G_func,
        prpx_plt,
        dtraceG_dx,
        trace_dprpw_plt_dx_T_prpw_bar,
        pG_func_px,
        n,
        t1,
        N,
        nx,
        x_design_dict,
    )

    CL_obj.solve()
    cost_0 = CL_obj.compute_FoI()["cost"]
    cost_history = CL_obj.compute_FoI(isHistory=True)
    CL_obj.compute_f_ode()
    u2 = CL_obj.get_u_ode()
    f = CL_obj.get_f_ode()

    y2 = copy.deepcopy(u2)
    for i in range(N + 1):
        y2[:, i] += yr[:]

    if 0:
        
        fig, ax = plt.subplots(3, figsize=(10, 5))

        ax[0].plot(t, y2[0, 1:], "k", alpha=0.5)
        ax[0].set_xlabel("t")
        ax[0].set_ylabel("x")
        ax[1].plot(t, y2[2, 1:], "b", alpha=0.5)
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("theta")
        ax[2].plot(t, cost_history, "r", alpha=0.5)
        ax[2].set_xlabel("t")
        ax[2].set_ylabel("cost")

        plt.show()


    # ==================
    # Run analsis
    # ==================
    # fig, ax = plt.subplots()
    fig, ax = plt.subplots(2, figsize=(10, 5))


    x_ball_arr = np.zeros((y2.shape[1]))
    y_ball_arr = np.zeros((y2.shape[1]))

    h0 = 0.2
    h = x_design_dict["M"] / 5.0
    l_bar = x_design_dict["L"]
    for i in range((y2.shape[1])):
        x_ball_arr[i] = y2[0, i] + l_bar * np.sin(y2[2, i])
        y_ball_arr[i] = h0 + h / 2 - l_bar * np.cos(y2[2, i])


    y2_loc = np.zeros(2)
    for i in range(y2.shape[1]):

        niceplots.setRCParams()
        niceplots.All()

        # Geom
        ax[0].set_aspect(1)

        ax[0].set_xlim([-12, 12])
        ax[0].set_ylim([-3, 3])

        y2_loc[0] = y2[0, i]
        y2_loc[1] = y2[2, i]

        invert_pendulum_instance = plot_cart.plot_invert_pendulum(x_design_dict, ax[0])
        
        invert_pendulum_instance.set_u(y2_loc)
        
        ax[0] = invert_pendulum_instance.plot_one_instance(x_ball_arr[0:i], y_ball_arr[0:i], f[0, i])

        # Cost
        ax[1].set_xlim([t0, t1])
        ax[1].set_ylim([0, max(cost_history) * 1.1])

        ax[1].set_xlabel("t")
        ax[1].set_ylabel("cost")
        ax[1].plot(t[0:i], cost_history[0:i], "r", alpha=0.5)

        plt.savefig("video/file%03d.png" % i)

        ax[0].cla()  
        ax[1].cla() 
        

    framerate = str(int(1.0 / dt))
    subprocess.call([
        'ffmpeg', '-framerate', framerate, '-i', 'video/file%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        video_name + '.mp4'
    ])



m = 1.0
M = 5.0
L = 2.0

x_design_dict = {"m":m, "M":M, "L":L}

y0 = np.zeros(4)
y0[0] = 2.0
y0[2] = 0.0

video_name = "baseline"
generate_video(x_design_dict, y0, video_name)

m = 0.5
M = 2.5
L = 1.0

x_design_dict = {"m":m, "M":M, "L":L}

y0 = np.zeros(4)
y0[0] = 2.0
y0[2] = 0.0

video_name = "optimized"
generate_video(x_design_dict, y0, video_name)
