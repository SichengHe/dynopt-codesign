import os
import sys
sys.path.insert(0, os.path.dirname(__file__) + '/../')  # add dynOpt/ to path

import unittest
import autograd.numpy as np
import numpy.testing as nptest

import CL
from cart_setting import res_funcs_generator, funcs_generator


class Test_cartpole(unittest.TestCase):

    def test_solver_and_derivatives(self):
        """
        test the ODE solver and derivatives computation for cart-pole baseline design
        """

        # ------------------
        # Base
        # ------------------
        m = 2.0
        M = 7.0
        L = 1.0

        x = np.zeros(3)
        x[0] = m
        x[1] = M
        x[2] = L

        # Q and R values
        ndof = 4
        nctrl = 1
        Q = np.eye(ndof) * 0.1
        R = np.eye(nctrl) 

        # Time setting
        dt = 0.01
        T = 3.0   # NOTE: setting short time horizon here, the state will not reach the target
        N = int(T / dt)

        # Initial state
        w_0 = np.zeros(4)
        w_0[0] = -1.0
        w_0[2] = 2.0

        # Provide a close steady-state initial guess
        theta_0 = np.array([np.pi - 0.1, 0.13])

        # Generate the residual forms
        res, res_reduced = res_funcs_generator()

        # Generate the closed loop obj using the residuals
        cl = CL.CL(x, res, res_reduced, Q, R, T, N, w_0)

        # Generate the implicit funciton format
        imp_FoI_cl = funcs_generator(x, cl, theta_0)

        # solve system
        [theta, P, w] = imp_FoI_cl.solve(theta_0=theta_0)
        cost = imp_FoI_cl.compute()

        # compute derivatives
        imp_FoI_cl.solve_adjoint()
        dfdx = imp_FoI_cl.compute_grad_design()

        # print(theta)
        # print(P)
        # print(w[:, -1])
        # print(cost)
        # print(dfdx)

        # reference values
        theta_ref = np.array([3.14159265e+00, -4.76758463e-12])
        P_ref = np.array([[8.17139746e-01, 3.28858683e+00, -1.72843565e+01, -5.50218119e+00],
                        [3.28858683e+00, 2.51439144e+01, -1.35735165e+02, -4.32320738e+01],
                        [-1.72843565e+01, -1.35735165e+02, 5.25178996e+03, 1.49369489e+03],
                        [-5.50218119e+00, -4.32320738e+01, 1.49369489e+03, 4.25837614e+02]])
        w_final_ref = np.array([34.28936615, 8.61031583, 0.37672386, -0.075627])
        cost_ref = 28475.762229706517
        dfdx_ref = np.array([6021.87951549, 6238.61504071, 13128.73551609])

        nptest.assert_allclose(theta, theta_ref, rtol=1e-6, atol=0)
        nptest.assert_allclose(P, P_ref, rtol=1e-6, atol=0)
        nptest.assert_allclose(w[:, -1], w_final_ref, rtol=1e-6, atol=0)
        nptest.assert_allclose(cost, cost_ref, rtol=1e-6, atol=0)
        nptest.assert_allclose(dfdx, dfdx_ref, rtol=1e-6, atol=0)


if __name__ == '__main__':
    unittest.main()