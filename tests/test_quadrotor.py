import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'dynopt', 'quadrotor'))

import unittest
import autograd.numpy as np
import numpy.testing as nptest

from dynopt import CL
from quadrotor_settings import res_funcs_generator, funcs_generator


class Test_quadrotor(unittest.TestCase):

    def setUp(self):
        """
        setup quadrotor problem
        """

        # initial states
        x_init = np.array([1, 1, 0.1, 0.5, 0.3, 0.05])

        # initial guess for the steady-state solution
        self.theta_0 = np.array([0.01, 0.01, 0.01, 0.01, 450., 450.])

        # quadrotor initial design
        twist = np.array([28.625901569164, 25.25668202258955, 20.762409457456656, 17.05302822457691, 14.166807345926632, 12.050792115126653, 10.687482715596696, 9.623772899617274, 8.73787265213354, 7.830294208487746])
        chord_by_R = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.3102845037371853, 0.24447688356440495, 0.17144033633125266, 0.05])   # chord normalized by rotor radius
        self.d_init = np.concatenate((np.deg2rad(twist), chord_by_R))
        num_cp = len(twist)

        # Q and R values
        ndof = 6
        nctrl = 2
        Q = np.eye(ndof)
        R = np.eye(nctrl) * 0.01

        # Time setting
        dt = 0.1
        T = 2.0   # setting short time horizon than what is needed to reach the target state
        N = int(T / dt)

        # Generate the residual forms (num_cp matches length of twist/chord arrays)
        num_cp = len(twist)
        res, self.res_reduced = res_funcs_generator(num_cp)

        # Generate the closed loop obj using the residuals
        cl = CL.CL(self.d_init, res, self.res_reduced, Q, R, T, N, x_init)

        # Generate the implicit funciton format
        self.imp_FoI_cl = funcs_generator(self.d_init, cl, self.theta_0)

    def test_residuals(self):
        """
        test residuals function
        """

        # evaluate residual function at the initial design and states
        res_0 = self.res_reduced.compute(self.theta_0, self.d_init, t=0.)

        res_ref = np.array([0.01, 0.01, 0.01, -0.088992, -0.912502, -0.007535])

        nptest.assert_allclose(res_0, res_ref, rtol=1e-4, atol=0)

    def test_solver_and_derivatives(self):
        """
        test the ODE solver and derivatives computation
        """

        # Solve system
        [theta, P, w] = self.imp_FoI_cl.solve(theta_0=self.theta_0)
        cost = self.imp_FoI_cl.compute()

        # Compute the objective derivatives
        self.imp_FoI_cl.solve_adjoint()
        dfdx = self.imp_FoI_cl.compute_grad_design()

        # reference values
        theta_ref = np.array([0., 0., 0., 0., 4.72380334e+02, 4.72380334e+02])
        P_ref = np.array([[ 1.68424780e+00, -1.11501665e-12, -2.61723319e+00,  9.18345331e-01, -1.99808362e-12, -3.22365948e-01],
                          [-1.11501665e-12,  3.07361700e+00, -2.35970687e-12, -1.98805201e-12,  3.40300245e+00, -4.68512098e-13],
                          [-2.61723319e+00, -2.35970687e-12,  1.82522873e+01, -4.08570331e+00, -3.62106798e-12,  2.90418441e+00],
                          [ 9.18345331e-01, -1.98805201e-12, -4.08570331e+00,  1.27992873e+00, -3.86936772e-12, -5.42944140e-01],
                          [-1.99808362e-12,  3.40300245e+00, -3.62106798e-12, -3.86936772e-12,  6.10007343e+00, -7.53454187e-13],
                          [-3.22365948e-01, -4.68512098e-13,  2.90418441e+00, -5.42944140e-01, -7.53454187e-13,  7.78003680e-01]])
        w_final_ref = np.array([ 0.30294308,  0.80016829, -0.04068102, -0.28669725, -0.31196996,  0.09806187])
        cost_ref = 5.77941665659776
        dfdx_ref = np.array([-2.64338459e-03, -5.46668164e-02, -1.33894098e-01, -3.10092491e-01,
                              -5.16584533e-01, -7.29384890e-01, -9.52458232e-01, -9.94070139e-01,
                              -9.95583366e-01, -1.73632968e-01, -9.63221021e-04, -4.37263772e-02,
                              -9.04845154e-02, -1.29419413e-01, -1.66767747e-01, -2.05693725e-01,
                              -2.66839196e-01, -3.39711783e-01, -5.65870669e-01, -1.69134533e-01])

        nptest.assert_allclose(theta, theta_ref, rtol=0, atol=1e-4)
        nptest.assert_allclose(P, P_ref, rtol=0, atol=1e-4)
        nptest.assert_allclose(w[:, -1], w_final_ref, rtol=1e-4, atol=0)
        nptest.assert_allclose(cost, cost_ref, rtol=1e-4, atol=0)
        nptest.assert_allclose(dfdx, dfdx_ref, rtol=1e-4, atol=0)


if __name__ == '__main__':
    unittest.main()