"""
This is a dummy model for the quadrotor. Only for debugging as the residual function is hacked and does not follow the EoM anymore.
It uses AD for 1st and 2nd order derivatives, like cart_setting.py does.
"""

import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad

from dynopt import residual, residual_reduced
from dynopt import CL

# import quadrotormodels
# from quadrotormodels import VTOLDynamicsGroup_MultiRotor_3DOF

import time as time_package


def res_funcs_generator(num_cp):
    """
    Generate residual and reduced residual functions

    Parameters
    ----------
    num_cp : int
        Number of spanwise spline control points for chord and twist distributions
    """
    ndof = 6
    nctrl = 2
    
    # ===========================
    # Construct the residual 
    # ===========================
    def res_func(w, q, x, t=0.):
        """
        w = [x, y, theta, vx, vy, theta_vel]
        q = [omega1, omega2]
        x = [twist, chord]
        """
            
        # some random model for thrust as a function of design and control
        mass = 1.4   # kg
        mass_inertia = 0.0211   # kg m^2
        arm_length = 0.15   # m

        thrust_ref = 1.4 * 9.81 / 5
        design_ref = np.array([0.49958683, 0.13879131, 0.04655465, 0.00541405])
        thrust_factor = np.sum(x / design_ref) / 5

        thrust1 = thrust_ref * q[0] * thrust_factor
        thrust2 = thrust_ref * q[1] * thrust_factor

        # EoM ignoring drag
        vx_dot = -2 * (thrust1 + thrust2) * np.sin(w[2]) / mass
        vy_dot = 2 * (thrust1 + thrust2) * np.cos(w[2]) / mass - 9.81
        theta_dot = 2 * (thrust1 - thrust2) * arm_length / mass_inertia

        dx = np.array([w[3], w[4], w[5], vx_dot, vy_dot, theta_dot])
        
        return dx

    # Construct partial derivatives using AD
    J = jacobian(res_func, 0)
    G = jacobian(res_func, 1)
    p_res_p_x_func = jacobian(res_func, 2)

    pJ_pw = jacobian(J, 0)
    pJ_pq = jacobian(J, 1)
    pJ_px = jacobian(J, 2)

    pG_pw = jacobian(G, 0)
    pG_pq = jacobian(G, 1)
    pG_px = jacobian(G, 2)

    res = residual.residual(res_func, ndof, nctrl, p_res_p_w_func=J, p_res_p_q_func=G, p_res_p_x_func=p_res_p_x_func, p_J_p_w_func=pJ_pw, p_J_p_q_func=pJ_pq, p_J_p_x_func=pJ_px, p_G_p_w_func=pG_pw, p_G_p_q_func=pG_pq, p_G_p_x_func=pG_px)

    # ===========================
    # Construct the reduced res 
    # ===========================

    # Fixed state variables (target coordinates)
    w_fixed = np.array([0.3, 0.5])
    w_fixed_ind = [0, 1]

    # Fixed control variables (target)
    q_fixed = np.array([])
    q_fixed_ind = []

    # Independent residuals
    res_reduced_ind = range(6)
    
    # Construct reduced residuals
    res_reduced = residual_reduced.residual_reduced(res, w_fixed, q_fixed, res_reduced_ind, w_fixed_ind, q_fixed_ind)

    return res, res_reduced

def funcs_generator(x, cl, theta_0):
    
    cl.solve(theta_0 = theta_0)   # NOTE: is this necessary?

    # ===========================
    # Construct the FoI 
    # ===========================

    def FoI_func(theta, P, dw, x):
    
        
        R = cl.R
        Q = cl.Q
        dt = cl.ode.dt
        N = cl.ode.N

        w_steady, q_steady = cl.res_reduced.map_theta_2_w_q(theta)
        G = cl.res.compute_grad_ctrl(w_steady, q_steady, x, t=-1.)   # set a dummy time because time doesn't matter here
        
        I_cost_val_arr = np.zeros(N)
        I_cost_val = 0.0
        for i in range(N):
            I_cost_val += dt * (dw[:, i].T.dot(Q)).dot(dw[:, i])
            I_cost_val += dt * (dw[:, i].T.dot(P.T.dot(G).dot(np.linalg.inv(R).T).dot(G.T).dot(P)).dot(dw[:, i]))

            I_cost_val_arr[i] = I_cost_val
        
        return I_cost_val, I_cost_val_arr

    def p_FoI_p_theta_func(theta, P, dw, x):
    
        
        R = cl.R
        dt = cl.ode.dt
        N = cl.ode.N

        w_steady, q_steady = cl.res_reduced.map_theta_2_w_q(theta)
        G = cl.res.compute_grad_ctrl(w_steady, q_steady, x, t=-1.)

        p_G_p_w_steady = cl.res.compute_grad_G_state(w_steady, q_steady, x)
        p_G_p_q_steady = cl.res.compute_grad_G_ctrl(w_steady, q_steady, x)

        p_I_cost_p_w_steady = np.zeros_like(w_steady)
        p_I_cost_p_q_steady = np.zeros_like(q_steady)
        n_state = len(w_steady)
        n_ctrl = len(q_steady)
        for i in range(N):
            for j in range(n_state):
                p_I_cost_p_w_steady[j] += dt * (dw[:, i].T.dot(P.T.dot(p_G_p_w_steady[:, :, j]).dot(np.linalg.inv(R).T).dot(G.T).dot(P)).dot(dw[:, i])) +  dt * (dw[:, i].T.dot(P.T.dot(G).dot(np.linalg.inv(R).T).dot(p_G_p_w_steady[:, :, j].T).dot(P)).dot(dw[:, i]))
            for j in range(n_ctrl):
                p_I_cost_p_q_steady[j] += dt * (dw[:, i].T.dot(P.T.dot(p_G_p_q_steady[:, :, j]).dot(np.linalg.inv(R).T).dot(G.T).dot(P)).dot(dw[:, i])) + dt * (dw[:, i].T.dot(P.T.dot(G).dot(np.linalg.inv(R).T).dot(p_G_p_q_steady[:, :, j].T).dot(P)).dot(dw[:, i]))

        p_I_cost_p_theta = cl.res_reduced.map_w_q_2_theta(p_I_cost_p_w_steady, p_I_cost_p_q_steady)

        return p_I_cost_p_theta

    def p_FoI_p_P_func(theta, P, dw, x):
    
        
        R = cl.R
        dt = cl.ode.dt
        N = cl.ode.N

        w_steady, q_steady = cl.res_reduced.map_theta_2_w_q(theta)
        G = cl.res.compute_grad_ctrl(w_steady, q_steady, x, t=-1.)

        H = G.dot(np.linalg.inv(R).T).dot(G.T)

        p_I_cost_p_P = np.zeros_like(P)
        for i in range(N):
            p_I_cost_p_P += (H + H.T).dot(P).dot(np.outer(dw[:, i], dw[:, i])) * dt
        
        return p_I_cost_p_P

    def p_FoI_p_dw_func(theta, P, dw, x):
    
        
        Q = cl.Q
        R = cl.R
        dt = cl.ode.dt
        N = cl.ode.N

        w_steady, q_steady = cl.res_reduced.map_theta_2_w_q(theta)
        G = cl.res.compute_grad_ctrl(w_steady, q_steady, x, t=-1.)

        p_I_p_dw = np.zeros_like(dw)
        for i in range(N):
            p_I_p_dw[:, i] += dt * 2 * Q.dot(dw[:, i])
            p_I_p_dw[:, i] += dt * 2 * (P.T.dot(G).dot(np.linalg.inv(R).T).dot(G.T).dot(P)).dot(dw[:, i])
        
        return p_I_p_dw

    def p_FoI_p_x_func(theta, P, dw, x):
    
        
        R = cl.R
        dt = cl.ode.dt
        N = cl.ode.N

        w_steady, q_steady = cl.res_reduced.map_theta_2_w_q(theta)
        G = cl.res.compute_grad_ctrl(w_steady, q_steady, x, t=-1.)

        p_G_p_x = cl.res.compute_grad_G_design(w_steady, q_steady, x)
        nx = len(x)
        p_FoI_p_x = np.zeros(nx)
        for i in range(N):
            for j in range(nx):
                p_FoI_p_x[j] += dt * (dw[:, i].T.dot(P.T.dot(p_G_p_x[:, :, j]).dot(np.linalg.inv(R).T).dot(G.T).dot(P)).dot(dw[:, i])) + dt * (dw[:, i].T.dot(P.T.dot(G).dot(np.linalg.inv(R).T).dot(p_G_p_x[:, :, j].T).dot(P)).dot(dw[:, i]))

        return p_FoI_p_x

    FoI_cl = CL.FoI_CL(FoI_func, p_FoI_p_theta_func=p_FoI_p_theta_func, p_FoI_p_P_func=p_FoI_p_P_func, \
            p_FoI_p_w_func=p_FoI_p_dw_func, p_FoI_p_x_func=p_FoI_p_x_func)


    imp_FoI_cl = CL.implicit_FoI_CL(FoI_cl, cl, x)
    imp_FoI_cl.solve(theta_0 = theta_0)   # NOTE: is this necessary?

    return imp_FoI_cl
