# import numpy as np
import scipy as sp
import residual
import residual_reduced
from scipy.optimize import fsolve
import autograd.numpy as np
from autograd import jacobian, grad, elementwise_grad
import copy



class equilibrium(object):

    """
    Equilibrium class.
    Nomenclature:
        x: design variables
        q: state variables
        r: residual
    Governing equation:
        r(q; x) = 0

    """
    def __init__(self, x, res_obj: residual_reduced.residual_reduced):

        # Design varible
        self.x = x

        # Residual and partials
        self.res_obj = res_obj

        # Number of state variables
        self.nstate = res_obj.get_state_size()

        # State variable
        self.w = np.zeros(self.nstate)

    def set_design(self, x):

        self.x = x

    def solve(self, w_0=None):

        nstate = self.nstate
        x = self.x

        if w_0 is None:
            # ad hoc initial condition
            w_0 = np.zeros(nstate)

        # Construct inline function to mask the design var
        r = lambda w: self.res_obj.compute(w, x, t=-1.)  # set a dummy time because time does not matter for steady-state analysis

        # solver_obj = dynOpt.newton_solver(r, q0, tol=1e-10)
        self.w = fsolve(r, w_0)   # TODO: use gradient here for the solver

        return self.w

    def get_sol(self):

        return self.w

    def solve_adjoint(self, pfpw, is_matrix_based=True):

        res_obj = self.res_obj
        w = self.w
        x = self.x
        nstate = self.nstate

        tol = 1e-10

        x0 = np.zeros(nstate)

        if is_matrix_based:
            Jacobian = res_obj.compute_grad_state(w, x, t=-1.)

            psi, info = sp.sparse.linalg.gmres(Jacobian.T, pfpw, x0)

            self.psi = psi

        return self.psi

    def get_sol_adjoint(self):

        return self.psi

    def compute_grad_design(self):

        res_obj = self.res_obj
        w = self.w
        x = self.x

        prpx = res_obj.compute_grad_design(w, x, t=-1.)

        return prpx

if __name__ == "__main__":

    class FoI_equilibrium(object):
        
        def __init__(self, FoI_func, p_FoI_p_theta_func=None, p_FoI_p_x_func=None):

            self.FoI_func = FoI_func

            self.p_FoI_p_theta_func = p_FoI_p_theta_func
            self.p_FoI_p_x_func = p_FoI_p_x_func

        def compute(self, theta, x):

            return self.FoI_func(theta, x)

        def compute_grad_state(self, theta, x):

            return self.p_FoI_p_theta_func(theta, x)

        def compute_grad_design(self, theta, x):

            return self.p_FoI_p_x_func(theta, x)


    class implicit_FoI_equilibrium(object):

        def __init__(self, FoI_eql: FoI_equilibrium, eql: equilibrium, x):

            self.FoI_eql = FoI_eql
            self.eql = eql

            self.x = x

            self.eql.set_design(x)

        def set_design(self, x):

            self.x = x

            self.eql.set_design(x)

        def solve(self, w_0 = None):

            self.theta = self.eql.solve(w_0 = w_0)

            return self.theta

        def solve_adjoint(self):

            FoI_eql = self.FoI_eql
            eql = self.eql
            x = self.x

            theta = self.theta

            p_FoI_p_theta = FoI_eql.compute_grad_state(theta, x)

            self.psi = self.eql.solve_adjoint(p_FoI_p_theta)

        def compute(self):

            FoI_eql = self.FoI_eql
            eql = self.eql
            x = self.x

            theta = self.theta

            FoI_eql_val = FoI_eql.compute(theta, x)

            return FoI_eql_val

        def compute_grad_design(self):

            FoI_eql = self.FoI_eql
            eql = self.eql
            x = self.x

            theta = self.theta
            psi = self.psi

            p_FoI_p_x_1 = FoI_eql.compute_grad_design(theta, x)
            p_FoI_p_x_2 = - eql.compute_grad_design().T.dot(psi)

            d_FoI_d_x = p_FoI_p_x_1 + p_FoI_p_x_2

            return d_FoI_d_x


    """
        Low level analysis
    """

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

    # Conduct AD
    p_res_p_w_func = jacobian(res_func, 0)
    p_res_p_q_func = jacobian(res_func, 1)
    p_res_p_x_func = jacobian(res_func, 2)

    # Construct the object
    ndof = 4
    nctrl = 1
    res = residual.residual(res_func, ndof, nctrl, p_res_p_w_func=p_res_p_w_func, p_res_p_q_func=p_res_p_q_func, p_res_p_x_func=p_res_p_x_func)
    
    w_fixed = np.zeros(3)
    w_fixed[0] = 1.0
    w_fixed_ind = [0, 1, 3]
    q_fixed = None
    q_fixed_ind = []
    res_reduced_ind = [1, 3]
    res_reduced = residual_reduced.residual_reduced(res, w_fixed, q_fixed, res_reduced_ind, w_fixed_ind, q_fixed_ind)


    # Construct the equilibrium object
    m = 1.0
    M = 5.0
    L = 2.0

    x = np.array([m, M, L])

    eql = equilibrium(x, res_reduced)


    
    theta_0 = np.array([np.pi - 0.1, 0.13])

    eql.solve(w_0 = theta_0)

    theta = eql.get_sol()
    print("=" * 20)
    print("Low level analysis")
    print("=" * 20)
    print("Solved:")
    print("theta:", theta)
    print("-" * 20)
    print("Target:")
    print("theta:", [np.pi, 0.0])
    print("=" * 20)


    """
        High level analysis
    """

    def FoI(theta, x):
        return np.sum(theta) * np.sum(x)

    p_FoI_p_theta_func = jacobian(FoI, 0)
    p_FoI_p_x_func = jacobian(FoI, 1)

    FoI_eql = FoI_equilibrium(FoI, p_FoI_p_theta_func = p_FoI_p_theta_func, p_FoI_p_x_func = p_FoI_p_x_func)



    theta_0 = np.array([np.pi - 0.1, 0.13])
    imp_FoI_eql = implicit_FoI_equilibrium(FoI_eql, eql, x)
    theta = imp_FoI_eql.solve(w_0 = theta_0)
    imp_FoI_val_0 = imp_FoI_eql.compute()

    print("=" * 20)
    print("High level analysis")
    print("=" * 20)
    print("Solved:")
    print("theta:", theta)
    print("-" * 20)
    print("Target:")
    print("theta:", [np.pi, 0.0])
    print("=" * 20)

    """
        Adjoint vs. FD
    """

    imp_FoI_eql.solve_adjoint()
    d_FoI_d_x = imp_FoI_eql.compute_grad_design()

    d_FoI_d_x_FD = np.zeros(len(x))
    epsilon = 1e-6
    for i in range(len(x)):

        x_p = copy.deepcopy(x)
        x_p[i] += epsilon

        theta_0 = np.array([np.pi - 0.1, 0.13])
        imp_FoI_eql = implicit_FoI_equilibrium(FoI_eql, eql, x_p)
        imp_FoI_eql.solve(w_0 = theta_0)

        imp_FoI_val_p = imp_FoI_eql.compute()

        d_FoI_d_x_FD[i] = (imp_FoI_val_p - imp_FoI_val_0) / epsilon

    
    print("=" * 20)
    print("Adjoint vs. FD")
    print("=" * 20)
    print("AD:", d_FoI_d_x)
    print("-" * 20)
    print("FD:", d_FoI_d_x_FD)
    print("=" * 20)