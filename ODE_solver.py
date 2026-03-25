import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
import residual_cl
import copy
import time


"""
    Setup the functions:
        res_obj: Dynamics function.
        x:       Design variable.  
        T:       Final time.
        N:       Number of time steps.
        w_0:     Initial state var.
"""
class ODE_solver(object):
    def __init__(self, res_obj: residual_cl.residual_cl, T, N, x, w_0):

        # All the functions
        self.res_obj = res_obj

        # The DOF
        self.nctrl = res_obj.get_ctrl_size()
        self.nstate = res_obj.get_state_size()

        # Time, number of steps, and step-size.
        self.T = T
        self.N = N
        self.dt = self.T / self.N
        self.time_array = np.linspace(0, self.T, self.N + 1)

        self.x = x
        self.w_0 = w_0

    def set_design(self, x):

        self.x = x

    def solve(self):

        # Extract the data
        w_0 = self.w_0
        x = self.x

        res_obj = self.res_obj

        nstate = self.nstate

        dt = self.dt
        N = self.N

        # Allocate the arrays
        w = np.zeros((nstate, N + 1))
        w[:, 0] = w_0[:]

        for i in range(N):

            w_i = w[:, i]

            w_ip1 = w_i + res_obj.compute(w_i, x, t=self.time_array[i]) * dt

            w[:, i + 1] = w_ip1[:]

        self.w = w[:, 1:]

        return self.w

    def solve_adjoint(self, p_I_p_w):

        nstate = self.nstate
        N = self.N
        dt = self.dt

        res_obj = self.res_obj
        
        x = self.x
        w = self.w

        psi = np.zeros((nstate, N))
        I = np.eye(nstate)
        for i in range(N):

            ind = N - i - 1

            w_ind = w[:, ind]
            p_I_p_w_ind = p_I_p_w[:, ind]

            if (ind < N - 1):
                psi_ind_p1 = psi[:, ind + 1]
                psi[:, ind] = - p_I_p_w_ind - (-I + (- res_obj.compute_grad_state(w_ind, x, t=self.time_array[ind + 1]).T * dt) ).dot(psi_ind_p1)
            else:
                psi[:, ind] = - p_I_p_w_ind

        self.psi = psi

        return self.psi

    def compute_grad_design(self):

        nstate = self.nstate
        N = self.N
        dt = self.dt
        w_0 = self.w_0

        res_obj = self.res_obj
        
        x = self.x
        w = self.w

        nx = len(x)
        p_res_p_x = np.zeros((nstate, N, nx))
        for i in range(N):
            
            if (not i== 0):
                w_im1 = w[:, i - 1]
            else:
                w_im1 = w_0
            p_res_p_x_i = - dt * res_obj.compute_grad_design(w_im1, x, t=self.time_array[i])

            p_res_p_x[:, i, :] = p_res_p_x_i[:, :]

        
        self.p_res_p_x = p_res_p_x

        return p_res_p_x


    def compute_grad_input_b(self, res_b):

        # HACK: not very general

        nstate = self.nstate
        nctrl = self.nctrl
        N = self.N
        dt = self.dt
        w_0 = self.w_0

        res_obj = self.res_obj
        
        x = self.x
        w = self.w

        W_b = np.zeros((nctrl, nstate))
        wtgt_b = np.zeros(nstate)
        qtgt_b = np.zeros(nctrl)

        for i in range(N):
            if (not i == 0):
                w_i_m1 = w[:, i - 1]
            else:
                w_i_m1 = w_0
            res_b_i = res_b[:, i]
            W_b_i, wtgt_b_i, qtgt_b_i \
                = res_obj.compute_grad_inputs_b(w_i_m1, x, res_b_i, t=self.time_array[i])

            W_b[:, :] += - W_b_i[:, :] * dt
            wtgt_b[:] += - wtgt_b_i[:] * dt
            qtgt_b[:] += - qtgt_b_i[:] * dt

        return W_b, wtgt_b, qtgt_b

    def get_adjoint(self):

        return self.psi

    def get_state(self):

        return self.w

    def get_dt(self):

        return self.dt


if __name__ == "__main__":

    import residual
    import matplotlib.pyplot as plt

    from functools import *
    from scipy import integrate


    class FoI_ODE(object):
        
        def __init__(self, FoI_func, p_FoI_p_w_func=None, p_FoI_p_x_func=None):

            self.FoI_func = FoI_func

            self.p_FoI_p_w_func = p_FoI_p_w_func
            self.p_FoI_p_x_func = p_FoI_p_x_func

        def compute(self, w, x):

            return self.FoI_func(w, x)

        def compute_grad_state(self, w, x):

            return self.p_FoI_p_w_func(w, x)

        def compute_grad_design(self, w, x):

            return self.p_FoI_p_x_func(w, x)

    class implicit_FoI_CL(object):
        
        def __init__(self, FoI_ode: FoI_ODE, ode: ODE_solver, x):

            self.FoI_ode = FoI_ode
            self.ode = ode

            self.x = x

            self.ode.set_design(x)

        def set_design(self, x):

            self.x = x

            self.ode.set_design(x)

        def solve(self):

            ode = self.ode

            self.w = ode.solve()

            return self.w

        def solve_adjoint(self):

            FoI_ode = self.FoI_ode
            ode = self.ode
            x = self.x

            w = self.w

            p_I_p_w = FoI_ode.compute_grad_state(w, x)

            psi = ode.solve_adjoint(p_I_p_w)

            self.psi = psi


        def compute(self):

            FoI_ode = self.FoI_ode
            x = self.x

            w = self.w

            FoI_ode_val = FoI_ode.compute(w, x)

            return FoI_ode_val

        def compute_grad_design(self):

            FoI_ode = self.FoI_ode
            ode = self.ode
            x = self.x

            w = self.w

            psi = self.psi

            p_FoI_p_x_1 = FoI_ode.compute_grad_design(w, x)

            p_FoI_p_x_2 = np.zeros_like(x)
            p_res_p_x = ode.compute_grad_design()
            for i in range(p_res_p_x.shape[1]):
                p_FoI_p_x_2 += p_res_p_x[:, i, :].T.dot(psi[:, i])

            d_FoI_d_x = p_FoI_p_x_1 + p_FoI_p_x_2

            return d_FoI_d_x


    # ==================
    # Analysis test
    # ==================

    def res_func(w, q, x, t):
        
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

    J = jacobian(res_func, 0)
    G = jacobian(res_func, 1)
    p_res_p_x_func = jacobian(res_func, 2)

    pJ_pw = jacobian(J, 0)
    pJ_pq = jacobian(J, 1)
    pJ_px = jacobian(J, 2)

    pG_pw = jacobian(G, 0)
    pG_pq = jacobian(G, 1)
    pG_px = jacobian(G, 2)

    ndof = 4
    nctrl = 1

    # Parameters
    m = 1.0
    M = 5.0
    L = 2.0
    x = np.zeros(3)
    x[0] = m
    x[1] = M
    x[2] = L

    res = residual.residual(res_func, ndof, nctrl, p_res_p_w_func=J, p_res_p_q_func=G, p_res_p_x_func=p_res_p_x_func, p_J_p_w_func=pJ_pw, p_J_p_q_func=pJ_pq, p_J_p_x_func=pJ_px, p_G_p_w_func=pG_pw, p_G_p_q_func=pG_pq, p_G_p_x_func=pG_px)


    w_tgt = np.array([1., 0., 3.14159265, 0.])
    q_tgt = np.zeros(1)
    W = np.array([[   3.16227766,   10.18360368, -243.16185368,  -70.35002653]])

    res_cl = residual_cl.residual_cl(res, W, w_tgt, q_tgt)
    w_0 = np.array([-2.,          0.,         -0.64159265,  0.        ])


    m = 1.0
    M = 5.0
    L = 2.0
    x = np.zeros(3)
    x[0] = m
    x[1] = M
    x[2] = L

    dt = 0.001
    T = 1.0
    N = int(T / dt)

    # ------------------
    # dynOpt integrator
    # ------------------
    ode = ODE_solver(res_cl, T, N, x, w_0)
    w = ode.solve()

    # ------------------
    # Scipy integrator
    # ------------------

    from scipy.integrate import ode as ode_scipy
    def f(t, y, x):

        global res_cl

        return res_cl.compute(y, x, t)

    y = np.zeros_like(w)
    t_0 = 0.0
    r = ode_scipy(f).set_integrator('vode', method='bdf')
    r.set_initial_value(w_0, t_0).set_f_params(x)

    i = 0
    while r.successful() and r.t < T:

        y[:, i] = r.integrate(r.t+dt)[:]

        i += 1

    # plot dynOpt integrator solution
    # plt.plot(w_tgt[0] + w[0, :], ':', lw=3)
    # plt.plot(w_tgt[2] + w[2, :], ':', lw=3)
    #plot scipy integrator solution
    # plt.plot(w_tgt[0] + y[0, :], lw=1)
    # plt.plot(w_tgt[2] + y[2, :], lw=1)

    # plt.show()
   

    # ==================
    # Adjoint test
    # ==================
    
    def FoI_func(w, x):
        
        # return np.sum(theta) + np.sum(P) + np.sum(w) + np.sum(x)
        return np.sum(w) + np.sum(x)

    p_I_p_w_func = jacobian(FoI_func, 0)
    p_I_p_x_func = jacobian(FoI_func, 1)

    FoI_ode = FoI_ODE(FoI_func, p_FoI_p_w_func=p_I_p_w_func, p_FoI_p_x_func=p_I_p_x_func)


    imp_FoI_ode = implicit_FoI_CL(FoI_ode, ode, x)
    t1 = time.time()
    imp_FoI_ode.solve()
    t2 = time.time()
    print("Solution time: ", t2 - t1)

    t1 = time.time()
    imp_FoI_ode.solve_adjoint()
    t2 = time.time()
    print("Adjoint time: ", t2 - t1)
    
    FoI_ode_val = imp_FoI_ode.compute()
    d_FoI_ode_d_x_val = imp_FoI_ode.compute_grad_design()
    

    epsilon = 1e-6
    d_FoI_ode_d_x_val_FD = np.zeros_like(d_FoI_ode_d_x_val)
    for i in range(len(x)):
        x_p = copy.deepcopy(x)
        x_p[i] += epsilon

        imp_FoI_ode = implicit_FoI_CL(FoI_ode, ode, x_p)
        imp_FoI_ode.solve()
        FoI_ode_val_p = imp_FoI_ode.compute()

        d_FoI_ode_d_x_val_FD[i] = (FoI_ode_val_p - FoI_ode_val) / epsilon

    print("d_FoI_ode_d_x_val", d_FoI_ode_d_x_val)
    print("d_FoI_ode_d_x_val_FD", d_FoI_ode_d_x_val_FD)