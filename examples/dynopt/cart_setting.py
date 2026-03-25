import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
import scipy.linalg
import copy
from functools import partial
from dynopt import residual, residual_reduced
from dynopt import CL

def res_funcs_generator():

    ndof = 4
    nctrl = 1

    # ===========================
    # Construct the residual 
    # ===========================
    """
        Residual form of the control problem.
    """
    def res_func(w, q, x, t=0.):
            
        # Extract the parameters
        m = x[0]
        M = x[1]
        L = x[2]

        g = 10

        Sx = np.sin(w[2])
        Cx = np.cos(w[2])

        D1 = m * Cx**2 - (m + M)
        D2 = (m * Cx**2 - (m + M)) * L

        dx = np.array([w[1], 
        (1 / D1) * (-m * g * Sx * Cx - (q[0] + m * L * w[3]**2 * Sx)), 
        w[3], 
        (1 / D2) * ((m + M) * g * Sx + Cx * (q[0] + m * L * w[3]**2 * Sx))])

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

    # Designate the fixed DOF and construct the reduced res from the original res

    w_fixed = np.array([1.0])
    q_fixed = np.array([])

    res_reduced_ind = range(4)
    w_fixed_ind = [0]
    q_fixed_ind = []

    w_fixed = np.zeros(3)
    w_fixed[0] = 1.0
    w_fixed_ind = [0, 1, 3]
    q_fixed = None
    q_fixed_ind = []
    res_reduced_ind = [1, 3]
    res_reduced = residual_reduced.residual_reduced(res, w_fixed, q_fixed, res_reduced_ind, w_fixed_ind, q_fixed_ind)

    return res, res_reduced

def funcs_generator(x, cl, theta_0):

    cl.solve(theta_0 = theta_0)

    # ===========================
    # Construct the FoI 
    # ===========================

    def FoI_func(theta, P, dw, x):
    
        
        R = cl.R
        Q = cl.Q
        dt = cl.ode.dt
        N = cl.ode.N

        w_steady, q_steady = cl.res_reduced.map_theta_2_w_q(theta)
        G = cl.res.compute_grad_ctrl(w_steady, q_steady, x, t=-1.)   # set a dummy time because the time does not matter here
        
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
    imp_FoI_cl.solve(theta_0 = theta_0)


    return imp_FoI_cl








