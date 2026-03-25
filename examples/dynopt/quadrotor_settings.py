import numpy as np
import openmdao.api as om

import residual
import residual_reduced
import CL

from quadrotor_openmdao_setup import setup_quadrotor_openmdao

import time as time_package

# TODO: duplicated residuals call in derivatives computation (i.e., in derivatives call, the analysis cache is not being used). Can I fix it?


class QuadrotorSteadyHoverWrapper():
    """
    Wrapper of the quadrotor model for steady hover
    """

    def __init__(self, num_cp):
        """
        Parameters
        ----------
        num_cp : int
            Number of spanwise spline control points for chord and twist distributions
        """
        self.num_cp = num_cp

        # setup OpenMDAO model for steady hover analysis
        self.prob = setup_quadrotor_openmdao(num_cp=num_cp, solve_for_hover=True)

        # initialize cache
        self.cache_design_point = np.zeros(2 * num_cp)

    def compute_power(self, d):
        """
        Compute steady hover power for a given blade design

        Parameters
        ----------
        d : 1d array, length = 2 * num_cp
            Blade design vector [theta_cp, chord_by_R_cp]
            theta_cp = blade twist, rad
            chord_by_R = blade chord normalized by rotor radius

        Returns
        -------
        power : float
            Steady hover power, W
        """

        # set blade design
        self.prob.set_val('rotor|theta_cp', val=d[0:self.num_cp], units='rad')
        self.prob.set_val('rotor|chord_by_R_cp', val=d[self.num_cp:], units=None)

        # run steady-state analysis
        self.prob.run_model()

        # get power
        power = self.prob.get_val('power', units='W')[0]

        # updated latest design point
        self.cache_design_point[:] = d

        return power

    def compute_power_grad(self, d):
        """
        Compute the gradient of the steady hover power w.r.t. the blade design
        """

        if not np.allclose(d, self.cache_design_point, atol=1e-12, rtol=1e-12):
            # the given design d is different from what self.prob stores. Need to re-run analysis
            self.compute_power(d)
        
        # compute power gradient
        total_derivs = self.prob.compute_totals(of=['power'], wrt=['rotor|theta_cp', 'rotor|chord_by_R_cp'])
        derivs = np.concatenate([total_derivs[('power', 'rotor|theta_cp')][0, :], total_derivs[('power', 'rotor|chord_by_R_cp')][0, :]])

        return derivs


class QuadrotorDynamicsWrapper():
    """
    Wrapper of the quadrotor dynamics model (implemented on OpenMDAO) for dynOpt

    States x = [x, y, theta, vx, vy, theta_vel]
    Controls u = [rotor_1_omega, rotor_2_omega]
    Design d = [theta_cp, chord_by_R_cp]

    Units: m, m/s, rad (not deg for angles)

    Solution cache is only activated for t >= 0.0. Set negative time for equilibrium analsysis and FD of J and G to deactivate caching
    The cache needs to be reset at the beginning of every optimization iteration. We trigger the reset in imp_FoI_cl.set_design().

    """

    def __init__(self):
        # finite difference setups
        self.fd_method = 'central'   # 'central' or 'forward'

        # whether to use cache of the analysis/derivatives solutions. set False only for debugging
        self.flag_use_cache = True

    def setup(self, num_cp):
        """
        Parameters
        ----------
        num_cp : int
            Number of spanwise spline control points for chord and twist distributions
        """

        self.num_cp = num_cp

        # setup OpenMDAO model that wrapps the quadrotor dynamics
        self.prob = setup_quadrotor_openmdao(num_cp=num_cp)

        # initialize dicts to store the analysis and derivatives solutions.
        # key: time, value: dx or derivatives
        self.cache_analysis = {}
        self.cache_derivatives = {}

        # --- setup for finite differencing J and G ---
        u_ref = 480.  # control reference value, rad/s. This will be used to determine FD step size
        d_ref = np.array([20 * np.pi / 180 for i in range(self.num_cp)] + [0.3 for i in range(self.num_cp)])   # ref values for [twist (rad); chord_by_R]
        # step size vectors
        self.step_mtx_x = np.eye(6) * 1e-5   # for derivatives w.r.t. state variables
        self.step_mtx_u = np.eye(2) * u_ref * 1e-4  # for derivatives w.r.t. control variables
        self.step_mtx_d = np.eye(2 * self.num_cp) * d_ref * 1e-5  # for derivatives w.r.t. design variables

    def __set_values(self, x, u, d):
        """
        Set values of x, u, d to the OpenMDAO Problem
        """
        # set values. states [x, y] doesn't matter for dynamics model
        self.prob.set_val('theta', val=x[2], units='rad')
        self.prob.set_val('vx', val=x[3], units='m/s')
        self.prob.set_val('vy', val=x[4], units='m/s')
        self.prob.set_val('theta_vel', val=x[5], units='rad/s')
        self.prob.set_val('omega_vert_1', val=u[0], units='rad/s')
        self.prob.set_val('omega_vert_2', val=u[1], units='rad/s')
        self.prob.set_val('rotor|theta_cp', val=d[0:self.num_cp], units='rad')
        self.prob.set_val('rotor|chord_by_R_cp', val=d[self.num_cp:], units=None)

    def __compute_derivatives(self, x, u, d, t):
        """
        Compute derivatives or get them from the cache
        """
        if t >= 0.0 and t in self.cache_derivatives.keys() and self.flag_use_cache:
            # we already called the residual at this point
            derivs = self.cache_derivatives[t]['derivs']
            ### print('using derivatives cache')

            # sanity check if x, u, d are idential
            tol = 1e-12
            if not (np.allclose(x, self.cache_derivatives[t]['x'], rtol=tol, atol=tol) and np.allclose(u, self.cache_derivatives[t]['u'], rtol=tol, atol=tol) and np.allclose(d, self.cache_derivatives[t]['d'], rtol=tol, atol=tol)):
                raise RuntimeError('x, u, d are not identical to the analysis cache, t =' + str(t))

        else:
            # run analysis
            self.__set_values(x, u, d)
            self.prob.run_model()

            # compute derivatives
            ### t_start = time_package.time()
            derivs = self.prob.compute_totals(of=['vx_dot', 'vy_dot', 'theta_dotdot'], wrt=['theta', 'vx', 'vy', 'theta_vel', 'omega_vert_1', 'omega_vert_2', 'rotor|theta_cp', 'rotor|chord_by_R_cp'])
            ### print('compute_totals time:', time_package.time() - t_start)

            # put solution in the cache
            self.cache_derivatives[t] = {'derivs' : derivs, 'x': x, 'u': u, 'd': d}

            if t >= 0.0:
                ### print('*** computed derivatives at t = %.10f sec' % t)
                pass
        # END IF

        return derivs

    def reset_cache(self):
        """
        Reset the analysis and derivatives cache.
        This should be called at the biginning of every optimization iteration
        """
        self.cache_analysis = {}
        self.cache_derivatives = {}
        ### print('resetting closed-loop cache')

    def residuals(self, x, u, d, t):
        """
        Residual function (right-hand side of ODE)

        Units: m, m/s, rad (not deg for angles)
        
        --- Inputs ---
        x : states, [x, y, theta, vx, vy, theta_vel], 1d ndarray with length = 6
        u : controls, [rotor1_omega, rotor2_omega], 1d ndarray with length = 2
        d : blade design, [theta_cp, chord_cp], 1d ndarray with length = 2 * num_cp (number of spanwise control points, defined in setup.py)
        
        --- Outputs ---
        dx : rate of states, [x_dot, y_dot, theta_dot, vx_dot, vy_dot, omega_dot], 1d ndarray with length = 6
        """

        if t > -10.:
            ### print('(resid) t = %.3f sec, x = %.5f, theta = %.5f' % (t, x[0], x[2]))
            pass
        start_time = time_package.time()

        # set states, controls, blade design value
        ### flag_new_point = self.__set_values(x, u, d)

        if t >= 0.0 and t in self.cache_analysis.keys() and self.flag_use_cache:
            # we already called the residual at this point
            dx = self.cache_analysis[t]['dx']
            flag_new_point = False
            ### print('using analysis cache')

            # sanity check if x, u, d are idential
            tol = 1e-12
            if not (np.allclose(x, self.cache_analysis[t]['x'], rtol=tol, atol=tol) and np.allclose(u, self.cache_analysis[t]['u'], rtol=tol, atol=tol) and np.allclose(d, self.cache_analysis[t]['d'], rtol=tol, atol=tol)):
                raise RuntimeError('x, u, d are not identical to the derivatives cache, t =' + str(t))

        else:
            # new point. call run_model
            flag_new_point = True

            # set input values to OpenMDAO problem
            self.__set_values(x, u, d)

            self.prob.run_model()

            # print('\n\n\n')
            # self.prob.check_partials(compact_print=True)
            # print('\n\n\n')

            # get rate of states
            dx = np.zeros(6)
            dx[0] = self.prob.get_val('x_dot', units='m/s')
            dx[1] = self.prob.get_val('y_dot', units='m/s')
            dx[2] = self.prob.get_val('theta_dot', units='rad/s')
            dx[3] = self.prob.get_val('vx_dot', units='m/s**2')
            dx[4] = self.prob.get_val('vy_dot', units='m/s**2')
            dx[5] = self.prob.get_val('theta_dotdot', units='rad/s**2')

            # put solution in the cache
            self.cache_analysis[t] = {'dx' : dx, 'x': x, 'u': u, 'd': d}
        # END IF

        end_time = time_package.time() - start_time
        if flag_new_point:
            ### print('residuals: %.10f sec' % end_time)
            pass

        return dx

    def pr_px(self, x, u, d, t):
        """
        Partials of the residual function w.r.t. states x
        """

        if t > -10.:
            ### print('(pr/px) t = %.3f sec, x = %.5f, theta = %.5f' % (t, x[0], x[2]))
            pass

        derivs = self.__compute_derivatives(x, u, d, t)

        J = np.zeros((6, 6))
        J[0, 3] = 1.   # derivs[('x_dot', 'vx')]  # this is just 1, always
        J[1, 4] = 1.   # derivs[('y_dot', 'vy')]  # this is just 1, always
        J[2, 5] = 1.   # derivs[('theta_dot', 'theta_vel')]  # this is just 1, always
        J[3, 2] = derivs[('vx_dot', 'theta')]
        J[3, 3] = derivs[('vx_dot', 'vx')]
        J[3, 4] = derivs[('vx_dot', 'vy')]
        J[3, 5] = derivs[('vx_dot', 'theta_vel')]
        J[4, 2] = derivs[('vy_dot', 'theta')]
        J[4, 3] = derivs[('vy_dot', 'vx')]
        J[4, 4] = derivs[('vy_dot', 'vy')]
        J[4, 5] = derivs[('vy_dot', 'theta_vel')]
        J[5, 2] = derivs[('theta_dotdot', 'theta')]
        J[5, 3] = derivs[('theta_dotdot', 'vx')]
        J[5, 4] = derivs[('theta_dotdot', 'vy')]
        J[5, 5] = derivs[('theta_dotdot', 'theta_vel')]
        
        return J

    def pr_pu(self, x, u, d, t):
        """
        Partials of the residual function w.r.t. controls u
        """

        if t > -10.:
            ### print('(pr/pu) t = %.3f sec, x = %.5f, theta = %.5f' % (t, x[0], x[2]))
            pass

        derivs = self.__compute_derivatives(x, u, d, t)

        G = np.zeros((6, 2))
        # We know that derivatives of [x_dot, y_dot, theta_dot] w.r.t. control is 0
        G[3, 0] = derivs['vx_dot', 'omega_vert_1']
        G[3, 1] = derivs['vx_dot', 'omega_vert_2']
        G[4, 0] = derivs['vy_dot', 'omega_vert_1']
        G[4, 1] = derivs['vy_dot', 'omega_vert_2']
        G[5, 0] = derivs['theta_dotdot', 'omega_vert_1']
        G[5, 1] = derivs['theta_dotdot', 'omega_vert_2']

        return G

    def pr_pd(self, x, u, d, t):
        """
        Partials of the residual function w.r.t. design d
        """

        if t > -10.:
            ### print('(pr/pd) t = %.3f sec, x = %.5f, theta = %.5f' % (t, x[0], x[2]))
            pass

        derivs = self.__compute_derivatives(x, u, d, t)

        partials = np.zeros((6, 2 * self.num_cp))   # we have 2 * num_cp design variables (twist and chord distributions)
        # We know that derivatives of [x_dot, y_dot, theta_dot] w.r.t. design is 0
        partials[3, 0:self.num_cp] = derivs['vx_dot', 'rotor|theta_cp']
        partials[3, self.num_cp:] = derivs['vx_dot', 'rotor|chord_by_R_cp']
        partials[4, 0:self.num_cp] = derivs['vy_dot', 'rotor|theta_cp']
        partials[4, self.num_cp:] = derivs['vy_dot', 'rotor|chord_by_R_cp']
        partials[5, 0:self.num_cp] = derivs['theta_dotdot', 'rotor|theta_cp']
        partials[5, self.num_cp:] = derivs['theta_dotdot', 'rotor|chord_by_R_cp']

        return partials

    def pJ_px(self, x, u, d):
        """
        Partials of J matrix w.r.t. states x by FD
        """

        t_start = time_package.time()

        pJ_px = np.zeros((6, 6, 6))   # shape (ndof, ndof, ndof)
        if self.fd_method == 'central':
            # central difference
            for i in range(6):   # loop over state variables
                J_plus = self.pr_px(x + self.step_mtx_x[i, :], u, d, t=-999.)   # set dummy time (t = -999)
                J_minus = self.pr_px(x - self.step_mtx_x[i, :], u, d, t=-999.)
                pJ_px[:, :, i] = (J_plus - J_minus) / (2 * self.step_mtx_x[i, i])
        elif self.fd_method == 'forward':
            # forward difference
            J_0 = self.pr_px(x, u, d, t=-999.)
            for i in range(6):   # loop over state variables
                J_plus = self.pr_px(x + self.step_mtx_x[i, :], u, d, t=-999.)
                pJ_px[:, :, i] = (J_plus - J_0) / self.step_mtx_x[i, i]
        else:
            raise RuntimeError('fd_method must be forward or central')

        ### print('FD for pJ/px. time = ', time_package.time() - t_start)
        
        return pJ_px

    def pJ_pu(self, x, u, d):
        """
        Partials of J matrix w.r.t. controls u by FD
        """

        t_start = time_package.time()

        pJ_pu = np.zeros((6, 6, 2))   # shape (ndof, ndof, nctrl)
        if self.fd_method == 'central':
            # central difference
            for i in range(2):   # loop over control variables
                J_plus = self.pr_px(x, u + self.step_mtx_u[i, :], d, t=-999.)
                J_minus = self.pr_px(x, u - self.step_mtx_u[i, :], d, t=-999.)
                pJ_pu[:, :, i] = (J_plus - J_minus) / (2 * self.step_mtx_u[i, i])
        elif self.fd_method == 'forward':
            # forward difference
            J_0 = self.pr_px(x, u, d, t=-999.)
            for i in range(2):   # loop over state variables
                J_plus = self.pr_px(x, u + self.step_mtx_u[i, :], d, t=-999.)
                pJ_pu[:, :, i] = (J_plus - J_0) / self.step_mtx_u[i, i]

        ### print('FD for pJ/pu. time = ', time_package.time() - t_start)

        return pJ_pu

    def pJ_pd(self, x, u, d):
        """
        Partials of J matrix w.r.t. design d by FD
        """

        t_start = time_package.time()

        pJ_pd = np.zeros((6, 6, 2 * self.num_cp))   # shape (ndof, ndof, ndesign)
        if self.fd_method == 'central':
            # central difference
            for i in range(2 * self.num_cp):   # loop over design variables
                J_plus = self.pr_px(x, u, d + self.step_mtx_d[i, :], t=-999.)
                J_minus = self.pr_px(x, u, d - self.step_mtx_d[i, :], t=-999.)
                pJ_pd[:, :, i] = (J_plus - J_minus) / (2 * self.step_mtx_d[i, i])
        elif self.fd_method == 'forward':
            # forward difference
            J_0 = self.pr_px(x, u, d, t=-999.)
            for i in range(2 * self.num_cp):   # loop over state variables
                J_plus = self.pr_px(x, u, d + self.step_mtx_d[i, :], t=-999.)
                pJ_pd[:, :, i] = (J_plus - J_0) / self.step_mtx_d[i, i]

        ### print('FD for pJ/pd. time = ', time_package.time() - t_start)

        return pJ_pd

    def pG_px(self, x, u, d):
        """
        partials of G matrix w.r.t. states x by FD
        """

        t_start = time_package.time()

        pG_px = np.zeros((6, 2, 6))   # shape (ndof, nctrl, ndof)
        if self.fd_method == 'central':
            # central difference
            for i in range(6):   # loop over state variables
                G_plus = self.pr_pu(x + self.step_mtx_x[i, :], u, d, t=-999.)
                G_minus = self.pr_pu(x - self.step_mtx_x[i, :], u, d, t=-999.)
                pG_px[:, :, i] = (G_plus - G_minus) / (2 * self.step_mtx_x[i, i])
        elif self.fd_method == 'forward':
            # forward difference
            G_0 = self.pr_pu(x, u, d, t=-999.)
            for i in range(6):   # loop over state variables
                G_plus = self.pr_pu(x + self.step_mtx_x[i, :], u, d, t=-999.)
                pG_px[:, :, i] = (G_plus - G_0) / self.step_mtx_x[i, i]

        ### print('FD for pG/px. time = ', time_package.time() - t_start)

        return pG_px

    def pG_pu(self, x, u, d):
        """
        partials of G matrix w.r.t. controls u by FD
        """

        t_start = time_package.time()

        pG_pu = np.zeros((6, 2, 2))   # shape (ndof, nctrl, nctrl)
        if self.fd_method == 'central':
            # central difference
            for i in range(2):   # loop over control variables
                G_plus = self.pr_pu(x, u + self.step_mtx_u[i, :], d, t=-999.)
                G_minus = self.pr_pu(x, u - self.step_mtx_u[i, :], d, t=-999.)
                pG_pu[:, :, i] = (G_plus - G_minus) / (2 * self.step_mtx_u[i, i])
        elif self.fd_method == 'forward':
            # forward difference
            G_0 = self.pr_pu(x, u, d, t=-999.)
            for i in range(2):   # loop over state variables
                G_plus = self.pr_pu(x, u + self.step_mtx_u[i, :], d, t=-999.)
                pG_pu[:, :, i] = (G_plus - G_0) / self.step_mtx_u[i, i]

        ### print('FD for pG/pu. time = ', time_package.time() - t_start)

        return pG_pu

    def pG_pd(self, x, u, d):
        """
        partials of G matrix w.r.t. design d by FD
        """

        t_start = time_package.time()

        pG_pd = np.zeros((6, 2, 2 * self.num_cp))   # shape (ndof, nctrl, ndesign)
        if self.fd_method == 'central':
            # central difference
            for i in range(2 * self.num_cp):   # loop over design variables
                G_plus = self.pr_pu(x, u, d + self.step_mtx_d[i, :], t=-999.)
                G_minus = self.pr_pu(x, u, d - self.step_mtx_d[i, :], t=-999.)
                pG_pd[:, :, i] = (G_plus - G_minus) / (2 * self.step_mtx_d[i, i])
        elif self.fd_method == 'forward':
            # forward difference
            G_0 = self.pr_pu(x, u, d, t=-999.)
            for i in range(2 * self.num_cp):   # loop over state variables
                G_plus = self.pr_pu(x, u, d + self.step_mtx_d[i, :], t=-999.)
                pG_pd[:, :, i] = (G_plus - G_0) / self.step_mtx_d[i, i]

        ### print('FD for pG/pd. time = ', time_package.time() - t_start)

        return pG_pd


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
    
    wr = QuadrotorDynamicsWrapper()
    wr.setup(num_cp=num_cp)

    # ===========================
    # Construct the residual
    # ===========================
    res = residual.residual(wr.residuals, ndof, nctrl, p_res_p_w_func=wr.pr_px, p_res_p_q_func=wr.pr_pu, p_res_p_x_func=wr.pr_pd, p_J_p_w_func=wr.pJ_px, p_J_p_q_func=wr.pJ_pu, p_J_p_x_func=wr.pJ_pd, p_G_p_w_func=wr.pG_px, p_G_p_q_func=wr.pG_pu, p_G_p_x_func=wr.pG_pd, reset_cache_func=wr.reset_cache)

    # ===========================
    # Construct the reduced res 
    # ===========================

    # Fixed state variables (target coordinates)
    w_fixed = np.array([0.0, 0.0])
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
    
    ### cl.solve(theta_0 = theta_0)   # NOTE: is this necessary?

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
    ### imp_FoI_cl.solve(theta_0 = theta_0)   # NOTE: is this necessary?

    return imp_FoI_cl


if __name__ == '__main__':
    """
    # rotor design
    twist = np.array([28.624217078309165, 7.952156310627215])
    chord_by_R = np.array([0.35, 0.05])
    num_cp = len(twist)
    num_desvar = 2 * num_cp
    d = np.concatenate((np.deg2rad(twist), chord_by_R))

    wrapper = QuadrotorDynamicsWrapper()
    wrapper.setup(num_cp)

    # reference states and controls
    # x_ref = np.array([0, 0, 0, 0, 0.01, 0])  # [x, y, theta, vx, vy, theta_vel]
    # u_ref = np.array([472.45977086, 472.45977086])
    x_ref = np.array([1, 1, 0.1, 0.5, 0.3, 0.05])
    ### x_ref = np.array([1, 1, 0.0, 1e-10, 1e-10, 0.0])
    ### x_ref = np.array([1, 1, 0., -0.5, -0.3, 0.])
    u_ref = np.array([402.45977086, 432.45977086])

    import time
    start_time = time.time()

    # --- evaluate residuals ---
    x_rate = wrapper.residuals(x_ref, u_ref, d, t=-1.)
    print('dx_dt =', x_rate)

    # --- compute J (dr/dx) and FD check ---
    print('-------------------------')
    print('--- J ---')
    J = wrapper.pr_px(x_ref, u_ref, d, t=-1.)
    ### print('J =', J)

    step = 1e-6
    J_fd = np.zeros((6, 6))
    step_mtx = np.eye(6) * step
    for i in range(6):   # loop over variables
        res_plus = wrapper.residuals(x_ref + step_mtx[i, :], u_ref, d, t=-1.)
        res_minus = wrapper.residuals(x_ref - step_mtx[i, :], u_ref, d, t=-1.)
        J_fd[:, i] = (res_plus - res_minus) / (2 * step)
    # END FOR
    ### print('J_fd =', J_fd)
    print('error (abs) =', np.max(np.abs(J_fd - J)))
    # print('error (rel) =', np.abs(J_fd - J) / np.abs(J))

    # --- compute G (dr/du) and FD check ---
    print('\n\n-------------------------')
    print('--- G ---')
    G = wrapper.pr_pu(x_ref, u_ref, d, t=-1.)
    ### print('G =', G)

    step = 1e-6
    G_fd = np.zeros((6, 2))
    step_mtx = np.eye(2) * step
    for i in range(2):   # loop over variables
        res_plus = wrapper.residuals(x_ref, u_ref + step_mtx[i, :], d, t=-1.)
        res_minus = wrapper.residuals(x_ref, u_ref - step_mtx[i, :], d, t=-1.)
        G_fd[:, i] = (res_plus - res_minus) / (2 * step)
    # END FOR
    ### print('G_fd =', G_fd)
    print('error (abs) =', np.max(np.abs(G_fd - G)))
    # print('error (rel) =', np.abs(G_fd - G) / np.abs(G))

    # --- compute dr/dd and FD check ---
    print('\n\n-------------------------')
    print('---  p_residuals/p_design ---')
    dr_dd = wrapper.pr_pd(x_ref, u_ref, d, t=-1.)
    ### print('p_residuals/p_design =', dr_dd)

    step = 1e-6
    dr_dd_fd = np.zeros((6, num_desvar))
    step_mtx = np.eye(num_desvar) * step
    for i in range(num_desvar):   # loop over variables
        res_plus = wrapper.residuals(x_ref, u_ref, d + step_mtx[i, :], t=-1.)
        res_minus = wrapper.residuals(x_ref, u_ref, d - step_mtx[i, :], t=-1.)
        dr_dd_fd[:, i] = (res_plus - res_minus) / (2 * step)
    # END FOR
    ### print('p_residuals/p_desig_fd =', dr_dd_fd)
    print('error (abs) =', np.max(np.abs(dr_dd_fd - dr_dd)))
    # print('error (rel) =', np.abs(dr_dd_fd - dr_dd) / np.abs(dr_dd))

    # --- second derivatives ---
    # just check the shape of second derivatives
    # print('dJ_dx shape:', wrapper.pJ_px(x_ref, u_ref, d).shape)
    # print('dJ_du shape:', wrapper.pJ_pu(x_ref, u_ref, d).shape)
    # print('dJ_dd shape:', wrapper.pJ_pd(x_ref, u_ref, d).shape)
    # print('dG_dx shape:', wrapper.pG_px(x_ref, u_ref, d).shape)
    # print('dG_du shape:', wrapper.pG_pu(x_ref, u_ref, d).shape)
    # print('dG_dd shape:', wrapper.pG_pd(x_ref, u_ref, d).shape)

    # --- printout non-zero entries for step-size study ---
    # print('dJ_dx:', list(wrapper.pJ_px(x_ref, u_ref, d)[3:, 2:, 2:].flatten()))
    # print('\ndG_dx:', list(wrapper.pG_px(x_ref, u_ref, d)[3:, :, 2:].flatten()))

    # ======================================================
    # check 2nd derivatives by comparing 1st-order FD of J, G vs. 2nd-order FD of residuals

    # "exact" derivatives by the 1st-order FD of J, G
    pJpx = wrapper.pJ_px(x_ref, u_ref, d)
    pJpu = wrapper.pJ_pu(x_ref, u_ref, d)
    pJpd = wrapper.pJ_pd(x_ref, u_ref, d)
    pGpx = wrapper.pG_px(x_ref, u_ref, d)
    pGpu = wrapper.pG_pu(x_ref, u_ref, d)
    pGpd = wrapper.pG_pd(x_ref, u_ref, d)

    # 2nd-order FD of residuals
    pJpx_FD = np.zeros((6, 6, 6))
    pJpu_FD = np.zeros((6, 6, 2))
    pJpd_FD = np.zeros((6, 6, num_desvar))
    pGpx_FD = np.zeros((6, 2, 6))
    pGpu_FD = np.zeros((6, 2, 2))
    pGpd_FD = np.zeros((6, 2, num_desvar))

    # step size
    step = 1e-4
    step_mtx_x = np.eye(6) * step
    step_mtx_u = np.eye(2) * step
    step_mtx_d = np.eye(num_desvar) * step

    res0 = wrapper.residuals(x_ref, u_ref, d, t=-1.)

    # --- p^2r/px^2 ---
    for i in range(6):
        for j in range(6):
            if i == j:
                res_plus = wrapper.residuals(x_ref + step_mtx_x[i, :], u_ref, d, t=-1.)
                res_minus = wrapper.residuals(x_ref - step_mtx_x[i, :], u_ref, d, t=-1.)
                pJpx_FD[:, i, j] = (res_plus - 2 * res0 + res_minus) / (step ** 2)
            else:
                res_pp = wrapper.residuals(x_ref + step_mtx_x[i, :] + step_mtx_x[j, :], u_ref, d, t=-1.)
                res_pm = wrapper.residuals(x_ref + step_mtx_x[i, :] - step_mtx_x[j, :], u_ref, d, t=-1.)
                res_mp = wrapper.residuals(x_ref - step_mtx_x[i, :] + step_mtx_x[j, :], u_ref, d, t=-1.)
                res_mm = wrapper.residuals(x_ref - step_mtx_x[i, :] - step_mtx_x[j, :], u_ref, d, t=-1.)
                pJpx_FD[:, i, j] = (res_pp - res_pm - res_mp + res_mm) / (4 * step ** 2)
    
    # --- p^2r/pxpu ---
    for i in range(6):
        for j in range(2):
            res_pp = wrapper.residuals(x_ref + step_mtx_x[i, :], u_ref + step_mtx_u[j, :], d, t=-1.)
            res_pm = wrapper.residuals(x_ref + step_mtx_x[i, :], u_ref - step_mtx_u[j, :], d, t=-1.)
            res_mp = wrapper.residuals(x_ref - step_mtx_x[i, :], u_ref + step_mtx_u[j, :], d, t=-1.)
            res_mm = wrapper.residuals(x_ref - step_mtx_x[i, :], u_ref - step_mtx_u[j, :], d, t=-1.)
            pJpu_FD[:, i, j] = (res_pp - res_pm - res_mp + res_mm) / (4 * step ** 2)
    
    # --- p^2r/pxpd ---
    for i in range(6):
        for j in range(num_desvar):
            res_pp = wrapper.residuals(x_ref + step_mtx_x[i, :], u_ref, d + step_mtx_d[j, :], t=-1.)
            res_pm = wrapper.residuals(x_ref + step_mtx_x[i, :], u_ref, d - step_mtx_d[j, :], t=-1.)
            res_mp = wrapper.residuals(x_ref - step_mtx_x[i, :], u_ref, d + step_mtx_d[j, :], t=-1.)
            res_mm = wrapper.residuals(x_ref - step_mtx_x[i, :], u_ref, d - step_mtx_d[j, :], t=-1.)
            pJpd_FD[:, i, j] = (res_pp - res_pm - res_mp + res_mm) / (4 * step ** 2)

    # --- p^2r/pupx ---
    for i in range(2):
        for j in range(6):
            res_pp = wrapper.residuals(x_ref + step_mtx_x[j, :], u_ref + step_mtx_u[i, :], d, t=-1.)
            res_pm = wrapper.residuals(x_ref - step_mtx_x[j, :], u_ref + step_mtx_u[i, :], d, t=-1.)
            res_mp = wrapper.residuals(x_ref + step_mtx_x[j, :], u_ref - step_mtx_u[i, :], d, t=-1.)
            res_mm = wrapper.residuals(x_ref - step_mtx_x[j, :], u_ref - step_mtx_u[i, :], d, t=-1.)
            pGpx_FD[:, i, j] = (res_pp - res_pm - res_mp + res_mm) / (4 * step ** 2)

    # --- p^2r/pupu ---
    for i in range(2):
        for j in range(2):
            if i == j:
                res_plus = wrapper.residuals(x_ref, u_ref + step_mtx_u[i, :], d, t=-1.)
                res_minus = wrapper.residuals(x_ref, u_ref - step_mtx_u[i, :], d, t=-1.)
                pGpu_FD[:, i, j] = (res_plus - 2 * res0 + res_minus) / (step ** 2)
            else:
                res_pp = wrapper.residuals(x_ref, u_ref + step_mtx_u[i, :] + step_mtx_u[j, :], d, t=-1.)
                res_pm = wrapper.residuals(x_ref, u_ref + step_mtx_u[i, :] - step_mtx_u[j, :], d, t=-1.)
                res_mp = wrapper.residuals(x_ref, u_ref - step_mtx_u[i, :] + step_mtx_u[j, :], d, t=-1.)
                res_mm = wrapper.residuals(x_ref, u_ref - step_mtx_u[i, :] - step_mtx_u[j, :], d, t=-1.)
                pGpu_FD[:, i, j] = (res_pp - res_pm - res_mp + res_mm) / (4 * step ** 2)

    # --- p^2r/pupd ---
    for i in range(2):
        for j in range(num_desvar):
            res_pp = wrapper.residuals(x_ref, u_ref + step_mtx_u[i, :], d + step_mtx_d[j, :], t=-1.)
            res_pm = wrapper.residuals(x_ref, u_ref + step_mtx_u[i, :], d - step_mtx_d[j, :], t=-1.)
            res_mp = wrapper.residuals(x_ref, u_ref - step_mtx_u[i, :], d + step_mtx_d[j, :], t=-1.)
            res_mm = wrapper.residuals(x_ref, u_ref - step_mtx_u[i, :], d - step_mtx_d[j, :], t=-1.)
            pGpd_FD[:, i, j] = (res_pp - res_pm - res_mp + res_mm) / (4 * step ** 2)
    
    # print errors
    np.set_printoptions(precision=4, linewidth=1e100, sign=' ')

    # TODO: better way of print out...
    pJpx_abs_error = np.abs(pJpx_FD - pJpx)
    pJpx_rel_error = pJpx_abs_error / np.maximum(np.abs(pJpx_FD), 1e-10)   # to avoid division by zero
    print('\n----------------------')
    print('pJ/px     =', pJpx.flatten())
    print('pJ/px_FD  =', pJpx_FD.flatten())
    print('\nabs error =', pJpx_abs_error.flatten())
    print('rel error =', pJpx_rel_error.flatten())
    print('max abs error =', np.max(pJpx_abs_error), 'max rel error =', np.max(pJpx_rel_error))

    pJpu_abs_error = np.abs(pJpu_FD - pJpu)
    pJpu_rel_error = pJpu_abs_error / np.maximum(np.abs(pJpu_FD), 1e-10)
    print('\n----------------------')
    print('pJ/pu     =', pJpu.flatten())
    print('pJ/pu_FD  =', pJpu_FD.flatten())
    print('\nabs error =', pJpu_abs_error.flatten())
    print('rel error =', pJpu_rel_error.flatten())
    print('max abs error =', np.max(pJpu_abs_error), 'max rel error =', np.max(pJpu_rel_error))

    pJpd_abs_error = np.abs(pJpd_FD - pJpd)
    pJpd_rel_error = pJpd_abs_error / np.maximum(np.abs(pJpd_FD), 1e-10)
    print('\n----------------------')
    print('pJ/pd     =', pJpd.flatten())
    print('pJ/pd_FD  =', pJpd_FD.flatten())
    print('\nabs error =', pJpd_abs_error.flatten())
    print('rel error =', pJpd_rel_error.flatten())
    print('max abs error =', np.max(pJpd_abs_error), 'max rel error =', np.max(pJpd_rel_error))

    pGpx_abs_error = np.abs(pGpx_FD - pGpx)
    pGpx_rel_error = pGpx_abs_error / np.maximum(np.abs(pGpx_FD), 1e-10)
    print('\n----------------------')
    print('pG/px     =', pGpx.flatten())
    print('pG/px_FD  =', pGpx_FD.flatten())
    print('\nabs error =', pGpx_abs_error.flatten())
    print('rel error =', pGpx_rel_error.flatten())
    print('max abs error =', np.max(pGpx_abs_error), 'max rel error =', np.max(pGpx_rel_error))

    pGpu_abs_error = np.abs(pGpu_FD - pGpu)
    pGpu_rel_error = pGpu_abs_error / np.maximum(np.abs(pGpu_FD), 1e-10)
    print('\n----------------------')
    print('pG/pu     =', pGpu.flatten())
    print('pG/pu_FD  =', pGpu_FD.flatten())
    print('\nabs error =', pGpu_abs_error.flatten())
    print('rel error =', pGpu_rel_error.flatten())
    print('max abs error =', np.max(pGpu_abs_error), 'max rel error =', np.max(pGpu_rel_error))

    pGpd_abs_error = np.abs(pGpd_FD - pGpd)
    pGpd_rel_error = pGpd_abs_error / np.maximum(np.abs(pGpd_FD), 1e-10)
    print('\n----------------------')
    print('pG/pd     =', pGpd.flatten())
    print('pG/pd_FD  =', pGpd_FD.flatten())
    print('\nabs error =', pGpd_abs_error.flatten())
    print('rel error =', pGpd_rel_error.flatten())
    print('max abs error =', np.max(pGpd_abs_error), 'max rel error =', np.max(pGpd_rel_error))

    print('\nruntime:', time.time() - start_time)
    """

    # -----------------------------------------------------------
    # verify total derivatives of QuadrotorSteadyHoverWrapper
    # -----------------------------------------------------------
    # design point
    twist = np.array([28.625901569164, 25.25668202258955, 20.762409457456656, 17.05302822457691, 14.166807345926632, 12.050792115126653, 10.687482715596696, 9.623772899617274, 8.73787265213354, 7.830294208487746])
    chord_by_R = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.3102845037371853, 0.24447688356440495, 0.17144033633125266, 0.05])   # chord normalized by rotor radius
    d_init = np.concatenate((np.deg2rad(twist), chord_by_R))
    num_cp = len(twist)

    quadrotor_steady = QuadrotorSteadyHoverWrapper(num_cp=num_cp)
    quadrotor_steady.compute_power(d_init)
    derivs = quadrotor_steady.compute_power_grad(d_init)

    # finite difference
    step = 1e-6
    d_ref = np.array([20 * np.pi / 180 for i in range(num_cp)] + [0.3 for i in range(num_cp)])   # ref values for relative step size
    step_mtx = np.eye(2 * num_cp) * step * d_ref

    derivs_fd = np.zeros(2 * num_cp)
    for i in range(2 * num_cp):
        d_plus = d_init + step_mtx[i, :]
        d_minus = d_init - step_mtx[i, :]
        power_plus = quadrotor_steady.compute_power(d_plus)
        power_minus = quadrotor_steady.compute_power(d_minus)
        derivs_fd[i] = (power_plus - power_minus) / (2 * step_mtx[i, i])

    print('derivs    =', list(derivs))
    print('derivs_fd =', list(derivs_fd))
    print('abs error =', list(np.abs(derivs - derivs_fd)))
    print('rel error =', list(np.abs(derivs - derivs_fd) / np.abs(derivs)))
    