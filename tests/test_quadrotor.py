import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples', 'dynopt'))

import unittest
import autograd.numpy as np
import numpy.testing as nptest

from dynopt import CL
from quadrotor_settings import res_funcs_generator, funcs_generator


class Test_cartpole(unittest.TestCase):

    def setUp(self):
        """
        setup quadrotor problem
        """

        # initial states
        x_init = np.array([1, 1, 0.1, 0.5, 0.3, 0.05])

        # initial guess for the steady-state solution
        self.theta_0 = np.array([0.01, 0.01, 0.01, 0.01, 450., 450.])

        # quadrotor initial design
        twist = np.array([28.624217078309165, 25.24644113988068, 20.757154874023897, 17.05283405965328, 14.168334169507002, 12.054782862617003, 10.690596691349743, 9.64011407359747, 8.72406502497316, 7.952156310627215])
        chord = np.array([0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.04117036620209598, 0.03226153890380773, 0.023085108580096464, 0.005414051955711461])
        self.d_init = np.concatenate((np.deg2rad(twist), chord))

        # Q and R values
        ndof = 6
        nctrl = 2
        Q = np.eye(ndof)
        R = np.eye(nctrl) * 0.01

        # Time setting
        dt = 0.1
        T = 2.0   # setting short time horizon than what is needed to reach the target state
        N = int(T / dt)

        # Generate the residual forms
        res, self.res_reduced = res_funcs_generator()

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

        res_ref = np.array([0.01, 0.01, 0.01, -0.08897896, -0.91379318, -0.00873179])

        nptest.assert_allclose(res_0, res_ref, rtol=1e-6, atol=0)

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

        # print('theta=', theta)
        # print('P=', P)
        # print('w_final=', w[:, -1])
        # print('cost=', cost)
        # print('df/dx=', dfdx)

        # reference values
        theta_ref = np.array([0., 0., 0., 0., 472.412167, 472.412167])
        P_ref = np.array([[1.68422361e+00, -1.04868615e-12, -2.61711805e+00, 9.18304579e-01, -1.90431631e-12, -3.22475819e-01],
                          [-1.04868615e-12, 3.05902096e+00, -2.37859940e-12, -1.88624509e-12, 3.40416228e+00, -5.12875338e-13],
                          [-2.61711805e+00, -2.37859940e-12, 1.82484649e+01, -4.08533619e+00, -3.35001529e-12, 2.90504532e+00],
                          [9.18304579e-01, -1.88624509e-12, -4.08533619e+00, 1.27984961e+00, -3.74461745e-12, -5.43121387e-01],
                          [-1.90431631e-12, 3.40416228e+00, -3.35001529e-12, -3.74461745e-12, 6.17623394e+00, -5.14819794e-13],
                          [-3.22475819e-01, -5.12875338e-13, 2.90504532e+00, -5.43121387e-01, -5.14819794e-13, 7.80097085e-01]])
        w_final_ref = np.array([0.3034933, 0.79564751, -0.04064336, -0.28730599, -0.31527729, 0.0971576])   # final state
        cost_ref = 5.755619742026387
        dfdx_ref = np.array([0.42644075, 7.50153277, 16.76314388, 35.51385974, 57.23850736, 83.28216077, 107.56189285, 112.75674448, 112.59450641, 19.22078551, 0.62301395, 29.44545922, 63.42662385, 92.67788437, 119.77355538, 141.9447104, 184.91705082, 237.88548045, 396.76376478, 127.08370508])

        nptest.assert_allclose(theta, theta_ref, rtol=0, atol=1e-5)
        nptest.assert_allclose(P, P_ref, rtol=0, atol=1e-5)
        nptest.assert_allclose(w[:, -1], w_final_ref, rtol=1e-5, atol=0)
        nptest.assert_allclose(cost, cost_ref, rtol=1e-5, atol=0)
        nptest.assert_allclose(dfdx, dfdx_ref, rtol=1e-5, atol=0)


if __name__ == '__main__':
    unittest.main()