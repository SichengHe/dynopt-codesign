import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

class LQRClosedLoopSystem(om.Group):
    # given linear system, computes the LQR feedback matrix K, and returns the (components of) closed-loop matrix A_cl where dx/dt = A_cl x

    def initialize(self):
        self.options.declare('ns', types=int, desc='dimension of state vector x')
        self.options.declare('nc', types=int, desc='dimension of control vector u')
        self.options.declare('Q', desc='cost matrix for state, (ns * ns')
        self.options.declare('R', desc='cost matrix for control, (nc * nc)')

    def setup(self):
        ns = self.options['ns']
        nc = self.options['nc']

        # first, solve Riccati equation
        self.add_subsystem('CARE', Riccati(ns=ns, nc=nc, Q=self.options['Q'], R=self.options['R']), promotes=['*'])

        # feedback matrix K
        R_inv = np.linalg.solve(self.options['R'], np.eye(nc))
        k_comp = om.ExecComp('K = dot(dot(Rinv, B.T), P)',
                             K={'shape' : (nc, ns)},
                             Rinv={'shape' : (nc, nc), 'val' : R_inv},
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

        # convert A_cl matrix to its components. This facilitates the derivative implementation in ODE later
        ### self.add_subsystem('mat_components', MatrixToComponents2D(), promotes_outputs=['*'])
        ### self.connect('A_cl', 'mat_components.A')

        # add Newton solver for Riccati eq
        ### self.nonlinear_solver = om.NewtonSolver(solve_subsystems=False, atol=1e-14, rtol=1e-14)
        ### self.linear_solver = om.DirectSolver()
        # NOTE: for some reason, the solution found by Newton is less accurate than scipy's solver


class MatrixToComponents2D(om.ExplicitComponent):

    def setup(self):
        self.add_input('A', shape=(2, 2))
        self.add_output('A11', val=1.)
        self.add_output('A12', val=1.)
        self.add_output('A21', val=1.)
        self.add_output('A22', val=1.)

        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        A = inputs['A']
        outputs['A11'] = A[0, 0]
        outputs['A12'] = A[0, 1]
        outputs['A21'] = A[1, 0]
        outputs['A22'] = A[1, 1]

    def compute_partials(self, inputs, partials):
        partials['A11', 'A'][0][0] = 1
        partials['A12', 'A'][0][1] = 1
        partials['A21', 'A'][0][2] = 1
        partials['A22', 'A'][0][3] = 1


class Riccati(om.ImplicitComponent):
    # Continuous-time Algebraic Riccati Equation (CARE) for linear system dx/dt = Ax + Bu, obj = int(xQx + uRu)

    def initialize(self):
        self.options.declare('ns', types=int, desc='dimension of state vector x')
        self.options.declare('nc', types=int, desc='dimension of control vector u')
        self.options.declare('Q', desc='cost matrix for state, (ns * ns')
        self.options.declare('R', desc='cost matrix for control, (nc * nc)')

    def setup(self):
        ns = self.options['ns']
        nc = self.options['nc']

        self.add_input('A', shape=(ns, ns), desc='linear system matrix A')
        self.add_input('B', shape=(ns, nc), desc='linear system matrix B')
        self.add_output('P', val=np.ones((ns, ns)), desc='Implicit variable matrix for Riccati equation')
        self.declare_partials('*', '*', method='cs')

        self._R_inv = np.linalg.solve(self.options['R'], np.eye(nc))

    def apply_nonlinear(self, inputs, outputs, residuals):
        Q = self.options['Q']
        A = inputs['A']
        B = inputs['B']
        P = outputs['P']
        residuals['P'] = A.T @ P + P @ A - P @ B @ self._R_inv @ B.T @ P + Q

    def solve_nonlinear(self, inputs, outputs):
        # solve Riccati equation using scipy
        outputs['P'] = solve_continuous_are(inputs['A'], inputs['B'], self.options['Q'], self.options['R'])



if __name__ == '__main__':
    # -------------------
    # define system
    # -------------------
    m = 1  # kg
    A = np.array([[0, 1], [0, 0]])   # (ns, ns)
    B = np.array([[0], [1 / m]])   # (ns, nc)
    ns, nc = B.shape
    print(ns, nc)

    Q = np.eye(ns)
    R = np.eye(nc)
    # -------------------

    p = om.Problem()
    p.model.add_subsystem('LQR', LQRClosedLoopSystem(ns=ns, nc=nc, Q=Q, R=R), promotes=['*'])
    p.model.set_input_defaults('A', A)
    p.model.set_input_defaults('B', B)
    # p.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    # p.model.linear_solver = om.DirectSolver()
    p.setup(check=True)

    p.run_model()
    p.check_partials(compact_print=True)
    # om.n2(p)

    P = p.get_val('P')
    print('P matrix =', P)

    # feedback system
    K = np.linalg.inv(R) @ B.T @ P
    print('feedback matrix K =', K)
    A_cl = p.get_val('A_cl')

    # simultation
    x0 = np.array([[-1, 1]])
    t_final = 10
    steps = 10000
    dt = t_final / steps

    xhis = np.zeros((2, steps))
    xhis[:, 0] = x0
    uhis = np.zeros(steps)

    for i in range(steps - 1):
        xhis[:, i + 1] = xhis[:, i] + A_cl @ xhis[:, i] * dt

    # plot state history
    plt.figure()
    plt.plot(np.linspace(0, t_final, steps), xhis[0, :])
    plt.xlabel('time')
    plt.ylabel('x')

    plt.figure()
    plt.plot(np.linspace(0, t_final, steps), xhis[1, :])
    plt.xlabel('time')
    plt.ylabel('speed')

    # plot control history
    uhis = np.dot(-K, xhis)   # NOTE: u = T - mg
    thrust_his = uhis + m * 9.81
    plt.figure()
    plt.plot(np.linspace(0, t_final, steps), thrust_his.reshape(10000))
    plt.xlabel('time')
    plt.ylabel('thrust')
    plt.show()
