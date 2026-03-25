# Closed loop system
import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
import scipy.linalg
import copy
from functools import partial
import residual
import residual_reduced
import residual_cl
import equilibrium
import LQR
import ODE_solver
import copy
import cProfile
import time

"""
    Setup the functions:
        eql: Equlibrium. 
        lqr: LQR.
        ode: Ordinary differentitial equation.
"""
class CL(object):
    def __init__(
        self,
        x, 
        res : residual.residual,
        res_reduced: residual_reduced.residual_reduced,
        Q, R,
        T, N, w_0
    ):

        self.x = x

        self.res = res
        self.res_reduced = res_reduced

        self.Q = Q
        self.R = R

        self.T = T
        self.N = N
        self.w_0 = w_0

        self.nx = len(x)

    def set_design(self, x):

        self.x = x

    def solve(self, theta_0 = None):

        # Solve steady state
        res_reduced = self.res_reduced
        x = self.x

        eql = equilibrium.equilibrium(x, res_reduced)
        theta = eql.solve(w_0 = theta_0)
        w_tgt, q_tgt = res_reduced.map_theta_2_w_q(theta)

        self.w_tgt = w_tgt
        self.q_tgt = q_tgt

        # Solve LQR
        Q = self.Q
        R = self.R

        J = eql.res_obj.compute_grad_state_org(theta, x, t=-2.)   # set a dummy time because time does not matter for LQR
        
        G = eql.res_obj.compute_grad_ctrl_org(theta, x, t=-2.)

        self.J = J
        self.G = G

        lqr = LQR.LQR(J, G, Q, R)
        P = lqr.solve()

        W = - (np.linalg.inv(R).dot(G.T)).dot(P)

        # Solve ODE
        res = self.res
        T = self.T
        N = self.N
        w_0 = self.w_0 - w_tgt

        res_cl = residual_cl.residual_cl(res, W, w_tgt, q_tgt)

        ode = ODE_solver.ODE_solver(res_cl, T, N, x, w_0)
        w = ode.solve()

        # Store
        self.eql = eql
        self.lqr = lqr
        self.ode = ode

        self.theta = theta
        self.P = P
        self.w = w

        return [theta, P, w]

    def solve_adjoint(self, p_I_p_theta, p_I_p_P, p_I_p_w):

        x = self.x

        # Solve ODE adjoint
        ode = self.ode

        t1 = time.time()
        psi_ode = ode.solve_adjoint(p_I_p_w)
        t2 = time.time()
        print("ode adjoint dt: ", t2 - t1)

        t1 = time.time()
        W_b, w_tgt_b_1, q_tgt_b_1 = ode.compute_grad_input_b(psi_ode)
        t2 = time.time()
        print("ode rad dt: ", t2 - t1)

        # Solve LQR adjoint
        G = self.G
        J = self.J
        R = self.R
        Q = self.Q
        P = self.P

        lqr = self.lqr

        P_b = (G.dot(-np.linalg.inv(R).T)).dot(W_b)
        G_b_1 = P.dot(W_b.T).dot(-np.linalg.inv(R))

        t1 = time.time()
        Psi_lqr = lqr.solve_adjoint(p_I_p_P + P_b)
        t2 = time.time()
        print("LQR adjoint dt: ", t2 - t1)

        # Solve steady adjoint

        w_tgt = self.w_tgt
        q_tgt = self.q_tgt

        t1 = time.time()
        J_b, G_b_2 = lqr.compute_grad_input_b(Psi_lqr)
        t2 = time.time()
        print("LQR RAD dt: ", t2 - t1)

        G_b = G_b_1 + G_b_2

        eql = self.eql
        res = self.res
        res_reduced = self.res_reduced

        t1 = time.time()
        dJdw = res.compute_grad_J_state(w_tgt, q_tgt, x)
        dJdq = res.compute_grad_J_ctrl(w_tgt, q_tgt, x)
        dGdw = res.compute_grad_G_state(w_tgt, q_tgt, x)
        dGdq = res.compute_grad_G_ctrl(w_tgt, q_tgt, x)
        t2 = time.time()
        print("Partial RAD dt: ", t2 - t1)

        w_tgt_b_2 = np.zeros_like(w_tgt_b_1)
        w_tgt_b_3 = np.zeros_like(w_tgt_b_1)

        q_tgt_b_2 = np.zeros_like(q_tgt_b_1)
        q_tgt_b_3 = np.zeros_like(q_tgt_b_1)

        for i in range(len(w_tgt_b_2)):

            w_tgt_b_2[i] = np.trace(np.dot(dJdw[:, :, i].T, J_b))
            w_tgt_b_3[i] = np.trace(np.dot(dGdw[:, :, i].T, G_b))

        for i in range(len(q_tgt_b_2)):

            q_tgt_b_2[i] = np.trace(np.dot(dJdq[:, :, i].T, J_b))
            q_tgt_b_3[i] = np.trace(np.dot(dGdq[:, :, i].T, G_b))

        w_tgt_b = w_tgt_b_1 + w_tgt_b_2 + w_tgt_b_3
        q_tgt_b = q_tgt_b_1 + q_tgt_b_2 + q_tgt_b_3

        t1 = time.time()
        theta_b = res_reduced.map_w_q_2_theta(w_tgt_b, q_tgt_b)
        t2 = time.time()
        print("Additional RAD dt: ", t2 - t1)

        t1 = time.time()
        psi_eql = eql.solve_adjoint(p_I_p_theta + theta_b)
        t2 = time.time()
        print("eql adjoint dt: ", t2 - t1)

        # Store the adjoint        

        self.psi_ode = psi_ode
        self.Psi_lqr = Psi_lqr
        self.psi_eql = psi_eql

        return [psi_eql, Psi_lqr, psi_ode]

    def compute_grad_design_b(self, r_nl_b,  R_ARE_b, r_cl_b):

        x = self.x
        nx = self.nx
        N = self.N

        res = self.res

        # ODE
        P = self.P
        G = self.G
        R = self.R
        ode = self.ode 

        p_rcl_p_x = ode.compute_grad_design()
        x_b_1 = np.zeros(nx)
        for i in range(N):

            x_b_1 += p_rcl_p_x[:, i, :].T.dot(r_cl_b[:, i])

        W_b, _, _ = ode.compute_grad_input_b(r_cl_b)
        G_b_1 = P.dot(W_b.T).dot(-np.linalg.inv(R))

        # ARE
        w_tgt = self.w_tgt
        q_tgt = self.q_tgt

        J_b = P.dot(R_ARE_b.T) + P.T.dot(R_ARE_b)
        G_b_2 = (((-P.T.dot(R_ARE_b)).dot(P.T)).dot(G)).dot(np.linalg.inv(R).T) + \
            (((- P.dot(R_ARE_b.T)).dot(P)).dot(G)).dot(np.linalg.inv(R))
        G_b = G_b_1 + G_b_2
    

        dJdx = res.compute_grad_J_design(w_tgt, q_tgt, x)
        dGdx = res.compute_grad_G_design(w_tgt, q_tgt, x)

        x_b_2 = np.zeros(nx)
        x_b_3 = np.zeros(nx)

        for i in range(nx):
            x_b_2[i] += np.trace(dJdx[:, :, i].T.dot(J_b))
            x_b_3[i] += np.trace(dGdx[:, :, i].T.dot(G_b))

        # eql
        eql = self.eql
        p_r_nl_p_x = eql.compute_grad_design()
        x_b_4 = p_r_nl_p_x.T.dot(r_nl_b)


        x_b = x_b_1 + x_b_2 + x_b_3 + x_b_4

        return x_b

class FoI_CL(object):
    
    def __init__(self, FoI_func, p_FoI_p_theta_func=None, p_FoI_p_P_func=None, \
        p_FoI_p_w_func=None, p_FoI_p_x_func=None):

        self.FoI_func = FoI_func

        self.p_FoI_p_theta_func = p_FoI_p_theta_func
        self.p_FoI_p_P_func = p_FoI_p_P_func
        self.p_FoI_p_w_func = p_FoI_p_w_func

        self.p_FoI_p_x_func = p_FoI_p_x_func

    def compute(self, theta, P, w, x):

        FoI_val, FoI_val_arr = self.FoI_func(theta, P, w, x)

        self.FoI_val = FoI_val
        self.FoI_val_arr = FoI_val_arr

        return FoI_val

    def compute_grad_state(self, theta, P, w, x):

        p_FoI_p_theta = self.p_FoI_p_theta_func(theta, P, w, x)
        p_FoI_p_P = self.p_FoI_p_P_func(theta, P, w, x)
        p_FoI_p_w = self.p_FoI_p_w_func(theta, P, w, x)

        return [p_FoI_p_theta, p_FoI_p_P, p_FoI_p_w]

    def compute_grad_design(self, theta, P, w, x):

        return self.p_FoI_p_x_func(theta, P, w, x)

class implicit_FoI_CL(object):
    
    def __init__(self, FoI_cl: FoI_CL, cl: CL, x):

        self.FoI_cl = FoI_cl
        self.cl = cl

        self.x = x

        self.cl.set_design(x)

    def set_design(self, x):

        self.x = x

        self.cl.set_design(x)

        # reset the CL cache because we changed the design
        self.cl.res.reset_cache()

    def solve(self, theta_0 = None):

        cl = self.cl

        [theta, P, w] = cl.solve(theta_0 = theta_0)

        self.theta = theta
        self.P = P
        self.w = w

        return [theta, P, w]

    def solve_adjoint(self):

        FoI_cl = self.FoI_cl
        cl = self.cl
        x = self.x

        theta = self.theta
        P = self.P
        w = self.w

        [p_I_p_theta, p_I_p_P, p_I_p_w] = FoI_cl.compute_grad_state(theta, P, w, x)

        [psi_eql, Psi_lqr, psi_ode] = cl.solve_adjoint(p_I_p_theta, p_I_p_P, p_I_p_w)

        self.psi_eql = psi_eql
        self.Psi_lqr = Psi_lqr
        self.psi_ode = psi_ode

    def compute(self):

        FoI_cl = self.FoI_cl
        cl = self.cl
        x = self.x

        theta = self.theta
        P = self.P
        w = self.w

        FoI_cl_val = FoI_cl.compute(theta, P, w, x)

        return FoI_cl_val

    def compute_grad_design(self):

        FoI_cl = self.FoI_cl
        cl = self.cl
        x = self.x

        theta = self.theta
        P = self.P
        w = self.w

        psi_eql = self.psi_eql

        Psi_lqr = self.Psi_lqr 

        psi_ode = self.psi_ode


        p_FoI_p_x_1 = FoI_cl.compute_grad_design(theta, P, w, x)
        p_FoI_p_x_2 = cl.compute_grad_design_b(-psi_eql, Psi_lqr, psi_ode)

        d_FoI_d_x = p_FoI_p_x_1 + p_FoI_p_x_2

        return d_FoI_d_x


if __name__ == "__main__":

    def res_func(w, q, x, t = 0.):
        
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
    m = 2.0
    M = 7.5
    L = 2.0
    x = np.zeros(3)
    x[0] = m
    x[1] = M
    x[2] = L

    # Construct residual and reduced residual object
    res = residual.residual(res_func, ndof, nctrl, p_res_p_w_func=J, p_res_p_q_func=G, p_res_p_x_func=p_res_p_x_func, p_J_p_w_func=pJ_pw, p_J_p_q_func=pJ_pq, p_J_p_x_func=pJ_px, p_G_p_w_func=pG_pw, p_G_p_q_func=pG_pq, p_G_p_x_func=pG_px)

    w_fixed = np.zeros(3)
    w_fixed[0] = 1.0
    w_fixed_ind = [0, 1, 3]
    q_fixed = None
    q_fixed_ind = []
    res_reduced_ind = [1, 3]

    res_reduced = residual_reduced.residual_reduced(res, w_fixed, q_fixed, res_reduced_ind, w_fixed_ind, q_fixed_ind)

    Q = np.eye(ndof) * 0.1
    R = np.eye(nctrl)

    dt = 0.001
    T = 0.1
    N = int(T // dt)
    w_0 = np.zeros(4)
    w_0[0] = -1.0
    w_0[2] = 4

    cl = CL(x, res, res_reduced, Q, R, T, N, w_0)

    theta_0 = np.array([np.pi - 0.1, 0.13])
    cl.solve(theta_0 = theta_0)

    def FoI_func(theta, P, w, x):

        I_cost_val_arr = np.zeros(10) # Hard coded
        
        return np.sum(theta) + np.sum(P) + np.sum(w) + np.sum(x), I_cost_val_arr
    
    def FoI_func_1(theta, P, w, x):

        return FoI_func(theta, P, w, x)[0]

    if 0:  

        custom_colors = ['#52a1fa', '#3eb051', '#faaa48', '#f26f6f', '#ae66de', '#485263']
        c1 = custom_colors[0]
        c2 = custom_colors[1]
        c3 = custom_colors[2]
        import niceplots
        niceplots.All()

        theta = cl.theta
        P = cl.P
        G = cl.G
        w = cl.w
        w_tgt, q_tgt = cl.res_reduced.map_theta_2_w_q(theta)
        K = -np.linalg.inv(R).dot(np.transpose(G)).dot(P)

        u = np.zeros(N)
        I = np.zeros(N)
        I_cumal = np.zeros(N)
        for i in range(N):
            w_int = w[:, i] #+ w_tgt[:]
            u[i] = -K.dot(w_int)[0]
            if i == 0:
                I[i] = w_int.T.dot(Q).dot(w_int) + (u[i]**2 * R[0, 0])
            else:
                I[i] = I[i-1] + w_int.T.dot(Q).dot(w_int) + (u[i]**2 * R[0, 0])


        t = np.linspace(dt, T, N)

        import matplotlib.pyplot as plt

        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False

        fontsize = 16
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
        labelpad = 45

        fig, ax = plt.subplots(3, figsize=(10, 5))
        plt.style.use(niceplots.get_style('james-light'))
        # x
        ax[0].plot(t,w_tgt[0]+ w[0, :], color=c1)
        ax[0].set_xlabel(r"$t$", fontsize=20, rotation=0)
        ax[0].plot(t, np.ones_like(t), color='lightgray', lw=0.5, zorder=-2)
        ax[0].set_ylabel('x', rotation=0, labelpad=labelpad, ha='left')
        ax[0].yaxis.set_label_coords(-.1, .5)

        ax[0].get_xaxis().set_visible(False)

        # theta
        ax[1].plot(t,w_tgt[2] + w[2, :], color=c2)
        ax[1].set_xlabel(r"$t$", fontsize=fontsize, rotation=0)
        ax[1].plot(t, np.pi*np.ones_like(t), color='lightgray', lw=0.5, zorder=-2)
        ax[1].set_ylabel(r'$\theta$', rotation=0, labelpad=labelpad, ha='left')
        ax[1].yaxis.set_label_coords(-.1, .5)

        ax[1].get_xaxis().set_visible(False)

        # cost
        ax[2].plot(t,I/100000, color=c3)
        ax[2].set_xlabel('Time [s]', fontsize=fontsize, rotation=0)
        ax[2].set_ylabel('Cost', rotation=0, labelpad=labelpad, ha='left')
        #ax[2].yaxis.set_label_coords(-.12, .5)

        #fig.align_ylabels()

        # tick marks
        ax[0].set_yticks([0])
        ax[1].set_yticks([3, 4])
        #ax[2].set_yticks([1, 1, 1])

        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])
        #ax[2].set_xticklabels([])

        


        #plt.savefig("../../R0_journal/figure/optimized.pdf", format="pdf", bbox_inches="tight")
        plt.savefig("closed-loop.pdf", format="pdf", bbox_inches="tight")



    p_I_p_theta_func = jacobian(FoI_func_1, 0)
    p_I_p_P_func = jacobian(FoI_func_1, 1)
    p_I_p_w_func = jacobian(FoI_func_1, 2)
    p_I_p_x_func = jacobian(FoI_func_1, 3)

    FoI_cl = FoI_CL(FoI_func, p_FoI_p_theta_func=p_I_p_theta_func, p_FoI_p_P_func=p_I_p_P_func, \
            p_FoI_p_w_func=p_I_p_w_func, p_FoI_p_x_func=p_I_p_x_func)


    imp_FoI_cl = implicit_FoI_CL(FoI_cl, cl, x)
    imp_FoI_cl.solve(theta_0 = theta_0)
    FoI_cl_val = imp_FoI_cl.compute()

    imp_FoI_cl.solve_adjoint()
    d_FoI_cl_d_x_val = imp_FoI_cl.compute_grad_design()


    epsilon = 1e-6
    nx = len(x)
    d_FoI_cl_d_x_val_FD = np.zeros(nx)
    for i in range(nx):

        x_p = copy.deepcopy(x)
        x_p[i] += epsilon

        imp_FoI_cl = implicit_FoI_CL(FoI_cl, cl, x_p)
        imp_FoI_cl.solve(theta_0 = theta_0)

        FoI_cl_val_p = imp_FoI_cl.compute()

        d_FoI_cl_d_x_val_FD[i] = (FoI_cl_val_p - FoI_cl_val) / epsilon


    print("d_FoI_cl_d_x_val", d_FoI_cl_d_x_val)
    print("d_FoI_cl_d_x_val_FD", d_FoI_cl_d_x_val_FD)
