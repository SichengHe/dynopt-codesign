import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt

# LQR for quadrotor 3-DOF control

class LQRFeedbackGain(om.Group):
    # given linear system, computes the LQR feedback matrix K, and returns the closed-loop matrix A_cl where dx/dt = A_cl x
    # Inputs: d_vxdot_vx, d_vydot_vy, d_vydot_omega1, d_vydot_omega2, d_thetadotdot_omega1, d_thetadotdot_omega2
    # Outputs: K, A_cl

    def initialize(self):
        self.options.declare('Q', desc='cost matrix for state, (ns * ns')
        self.options.declare('Rinv', desc='inverse of cost matrix for control, (nc * nc)')

    def setup(self):
        # problem size hardcoded for quadrotor 3DOF control
        ns = 6   # number of states
        nc = 2   # number of controls

        # assemble A and B matrix
        self.add_subsystem('assemble_lin_system', ComponentsToMatrices(), promotes=['*'])

        # solve Riccati equation
        self.add_subsystem('CARE', Riccati(ns=ns, nc=nc, Q=self.options['Q'], Rinv=self.options['Rinv']), promotes=['*'])

        # feedback matrix K
        k_comp = om.ExecComp('K = dot(dot(Rinv, B.T), P)',
                             K={'shape' : (nc, ns)},
                             Rinv={'shape' : (nc, nc), 'val' : self.options['Rinv']},
                             B={'shape' : (ns, nc)},
                             P={'shape' : (ns, ns)})
        self.add_subsystem('feedback', k_comp, promotes=['*'])

        # closed-loop system matrix. K = R
        cl_comp = om.ExecComp('A_cl = A - dot(B, K)',
                              A_cl={'shape' : (ns, ns)},
                              A={'shape' : (ns, ns)},
                              B={'shape' : (ns, nc)},
                              K={'shape' : (nc, ns)})
        self.add_subsystem('CL', cl_comp, promotes=['*'])

        # add Newton solver for Riccati eq
        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, atol=1e-12, rtol=1e-12)
        self.linear_solver = om.DirectSolver()


class ComponentsToMatrices(om.ExplicitComponent):
    # compose linear system matrixes A and B for quadrotor 3-DOF control

    def setup(self):
        self.add_input('d_vxdot_vx', val=1)
        self.add_input('d_vydot_vy', val=1)
        self.add_input('d_vydot_omega1', val=1)
        self.add_input('d_vydot_omega2', val=1)
        self.add_input('d_thetadotdot_omega1', val=1)
        self.add_input('d_thetadotdot_omega2', val=1)

        self.add_output('A', shape=(6, 6))
        self.add_output('B', shape=(6, 2))

        self.declare_partials('*', '*', method='fd')   # TODO: implement partials

    def compute(self, inputs, outputs):
        A = np.zeros((6, 6))
        A[0, 3] = 1.0
        A[1, 4] = 1.0
        A[2, 5] = 1.0
        A[3, 2] = -9.81
        A[3, 3] = inputs['d_vxdot_vx'][0]
        A[4, 4] = inputs['d_vydot_vy'][0]
        outputs['A'] = A

        B = np.zeros((6, 2))
        B[4, 0] = inputs['d_vydot_omega1'][0]
        B[4, 1] = inputs['d_vydot_omega2'][0]
        B[5, 0] = inputs['d_thetadotdot_omega1'][0]
        B[5, 1] = inputs['d_thetadotdot_omega2'][0]
        outputs['B'] = B


class Riccati(om.ImplicitComponent):
    # Continuous-time Algebraic Riccati Equation (CARE) for linear system dx/dt = Ax + Bu, obj = int(xQx + uRu)

    def initialize(self):
        self.options.declare('ns', types=int, desc='dimension of state vector x')
        self.options.declare('nc', types=int, desc='dimension of control vector u')
        self.options.declare('Q', desc='cost matrix for state, (ns * ns')
        self.options.declare('Rinv', desc='inverse of cost matrix for control, (nc * nc)')

    def setup(self):
        ns = self.options['ns']
        nc = self.options['nc']

        self.add_input('A', shape=(ns, ns), desc='linear system matrix A')
        self.add_input('B', shape=(ns, nc), desc='linear system matrix B')
        self.add_output('P', val=np.ones((ns, ns)), desc='Implicit variable matrix for Riccati equation')
        self.declare_partials('*', '*', method='cs')   # TODO: implement partials

    def apply_nonlinear(self, inputs, outputs, residuals):
        Q = self.options['Q']
        Rinv = self.options['Rinv']
        A = inputs['A']
        B = inputs['B']
        P = outputs['P']
        residuals['P'] = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
        