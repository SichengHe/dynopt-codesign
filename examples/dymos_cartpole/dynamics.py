"""
Cart-pole dynamics (ODE)
"""

import numpy as np
import openmdao.api as om


class CartPoleDynamicsWithFriction(om.Group):
    """
    Computes the time derivatives of states given state variables and control inputs.

    Parameters
    ----------
    m_cart : float
        Mass of cart.
    m_pole : float
        Mass of pole.
    l_pole : float
        Length of pole.
    d : float
        Friction coefficient
    x : 1d array
        x location of cart.
    x_dot : 1d array
        x velocity of card.
    theta : 1d array
        Angle of pole, 0 for vertical downward, positive counter clockwise.
    theta_dot : 1d array
        Angluar velocity of pole.
    f : 1d array
        x-wise external force applied to the cart.

    Returns
    -------
    x_dotdot : 1d array
        Acceleration of cart in x direction.
    theta_dotdot : 1d array
        Angular acceleration of pole.
    cost : 1d array
        Integrand of linear-quadratic cost function
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be evaluated (i.e., length of vectors x, theta, etc)')
        self.options.declare('g', default=10., desc='gravity constant')
        self.options.declare('states_ref', types=dict, desc='reference state values with keys = [x, x_dot, theta, theta_dot]')
        self.options.declare('ignore_cost', default=False, desc='if True, will not compute the quadratic cost')

    def setup(self):
        nn = self.options['num_nodes']
        
        # first, compute the equivalent external force for frictionless system. Equivalent force = external force - friction force
        force_equiv = om.ExecComp('force_minus_friction = f - d * x_dot',
                                  has_diag_partials=True,
                                  force_minus_friction={'shape': (nn,), 'units': 'N'},
                                  f={'shape': (nn,), 'units': 'N'},
                                  d={'shape': (1,), 'units': 'N*s/m'},
                                  x_dot={'shape': (nn,), 'units': 'm/s'})
        self.add_subsystem('force_equiv', force_equiv, promotes_inputs=['*'])

        # then, compute the state rate of changes.
        input_names = ['m_cart', 'm_pole', 'l_pole', 'x', 'x_dot', 'theta', 'theta_dot', ]
        self.add_subsystem('dynamics_frictionless', _CartPoleDynamics(num_nodes=nn, g=self.options['g']), promotes_inputs=input_names, promotes_outputs=['*'])
        self.connect('force_equiv.force_minus_friction', 'dynamics_frictionless.f')  # connect equivalent force

        if not(self.options['ignore_cost']):
            # also compute the integrand of the quadratic cost function
            self.add_subsystem('cost', _QuadraticCostRate(num_nodes=nn, states_ref=self.options['states_ref']), promotes_inputs=['*'], promotes_outputs=['*'])


class _QuadraticCostRate(om.ExplicitComponent):
    """
    Computes the integrand of the quadratic cost function
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be evaluated (i.e., length of vectors x, theta, etc)')
        self.options.declare('states_ref', types=dict, desc='reference state values with keys = [x, x_dot, theta, theta_dot]')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('x', shape=(nn,), units='m', desc='cart location')
        self.add_input('x_dot', shape=(nn,), units='m/s', desc='cart speed')
        self.add_input('theta', shape=(nn,), units='rad', desc='pole angle')
        self.add_input('theta_dot', shape=(nn,), units='rad/s', desc='pole angle velocity')
        self.add_input('f', shape=(nn,), units='N', desc='external force applied to cart in x direction')
        self.add_output('cost_rate', shape=(nn,), desc='integrand of quadratic objective (xQx + uRu)')
        self.declare_partials(of=['*'], wrt=['x', 'x_dot', 'theta', 'theta_dot', 'f'], method='exact', rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs):
        states_ref = self.options['states_ref']
        f = inputs['f']

        # error of the states
        dx = inputs['x'] - states_ref['x']
        dvx = inputs['x_dot'] - states_ref['x_dot']
        dtheta = inputs['theta'] - states_ref['theta']
        domega = inputs['theta_dot'] - states_ref['theta_dot']
         
        # cost integrand (assumes cost matrices are identity for both state and cost)
        outputs['cost_rate'] = dx**2 + dvx**2 + dtheta**2 + domega**2 + f**2

    def compute_partials(self, inputs, partials):
        states_ref = self.options['states_ref']
        partials['cost_rate', 'x'] = 2 * (inputs['x'] - states_ref['x'])
        partials['cost_rate', 'x_dot'] = 2 * (inputs['x_dot'] - states_ref['x_dot'])
        partials['cost_rate', 'theta'] = 2 * (inputs['theta'] - states_ref['theta'])
        partials['cost_rate', 'theta_dot'] = 2 * (inputs['theta_dot'] - states_ref['theta_dot'])
        partials['cost_rate', 'f'] = 2. * inputs['f']


class _CartPoleDynamics(om.ExplicitComponent):
    """
    Cart-pole dynamics without friction
    Computes the time derivatives of states given state variables and control inputs.

    Parameters
    ----------
    m_cart : float
        Mass of cart.
    m_pole : float
        Mass of pole.
    l_pole : float
        Length of pole.
    x : 1d array
        x location of cart.
    x_dot : 1d array
        x velocity of card.
    theta : 1d array
        Angle of pole, 0 for vertical downward, positive counter clockwise.
    theta_dot : 1d array
        Angluar velocity of pole.
    f : 1d array
        x-wise force applied to the cart.

    Returns
    -------
    x_dotdot : 1d array
        Acceleration of cart in x direction.
    theta_dotdot : 1d array
        Angular acceleration of pole.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be evaluated (i.e., length of vectors x, theta, etc)')
        self.options.declare('g', default=10., desc='gravity constant')

    def setup(self):
        nn = self.options['num_nodes']

        # --- inputs ---
        # cart-pole parameters
        self.add_input('m_cart', shape=(1,), units='kg', desc='cart mass')
        self.add_input('m_pole', shape=(1,), units='kg', desc='pole mass')
        self.add_input('l_pole', shape=(1,), units='m', desc='pole length')
        ### self.add_input('d', shape=(1,), desc='friction coefficient')
        # state variables
        self.add_input('x', shape=(nn,), units='m', desc='cart location')
        self.add_input('x_dot', shape=(nn,), units='m/s', desc='cart speed')
        self.add_input('theta', shape=(nn,), units='rad', desc='pole angle')
        self.add_input('theta_dot', shape=(nn,), units='rad/s', desc='pole angle velocity')
        # control input
        self.add_input('f', shape=(nn,), units='N', desc='force applied to cart in x direction')

        # --- outputs ---
        # rate of states (accelerations)
        self.add_output('x_dotdot', shape=(nn,), units='m/s**2', desc='x acceleration of cart')
        self.add_output('theta_dotdot', shape=(nn,), units='rad/s**2', desc='angular acceleration of pole')

        # ---  partials ---.
        # Jacobian of outputs w.r.t. state/control inputs is diagonal because each node (corresponds to time discretization) is independent
        self.declare_partials(of=['*'], wrt=['x', 'x_dot', 'theta', 'theta_dot', 'f'], method='exact', rows=np.arange(nn), cols=np.arange(nn))

        # partials of outputs w.r.t. cart-pole parameters. I'll use complex-step for this, but still declare the sparsity structure.
        self.declare_partials(of=['*'], wrt=['m_cart', 'm_pole', 'l_pole'], method='cs', rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_coloring(wrt=['m_cart', 'm_pole', 'l_pole'], method='cs', show_summary=False)
        self.set_check_partial_options(wrt=['m_cart', 'm_pole', 'l_pole'], method='fd', step=1e-6)
        # NOTE: These partials are only required for co-design. For just trajectory optimization (with fixed cart-pole parameters), we don't need these partials.

    def compute(self, inputs, outputs):
        g = self.options['g']
        mc = inputs['m_cart']
        mp = inputs['m_pole']
        lpole = inputs['l_pole']
        theta = inputs['theta']
        omega = inputs['theta_dot']
        f = inputs['f']

        sint = np.sin(theta)
        cost = np.cos(theta)
        det = mp * lpole * cost**2 - lpole * (mc + mp)
        outputs['x_dotdot'] = (-mp * lpole * g * sint * cost - lpole * (f + mp * lpole * omega**2 * sint)) / det
        outputs['theta_dotdot'] = ((mc + mp) * g * sint + cost * (f + mp * lpole * omega**2 * sint)) / det

    def compute_partials(self, inputs, jacobian):
        g = self.options['g']
        mc = inputs['m_cart']
        mp = inputs['m_pole']
        lpole = inputs['l_pole']
        theta = inputs['theta']
        theta_dot = inputs['theta_dot']
        f = inputs['f']

        # --- derivatives of x_dotdot ---
        # Collecting Theta Derivative
        low = mp * lpole * np.cos(theta)**2 - lpole * mc - lpole * mp
        dhigh = mp * g * lpole * np.sin(theta)**2 - mp * g * lpole * np.cos(theta)**2 - lpole**2 * mp * theta_dot**2 * np.cos(theta)
        high = -mp * g * lpole * np.cos(theta) * np.sin(theta) - lpole * f - lpole**2 * mp * theta_dot**2 * np.sin(theta)
        dlow = 2. * mp * lpole * np.cos(theta) * (-np.sin(theta))

        # X_Dot - Partials
        jacobian['x_dotdot', 'x'] = 0.
        jacobian['x_dotdot', 'theta'] = (low * dhigh - high * dlow) / low**2
        jacobian['x_dotdot', 'x_dot'] = 0.
        jacobian['x_dotdot', 'theta_dot'] = -2. * theta_dot * lpole**2 * mp * np.sin(theta) / (mp * lpole * np.cos(theta)**2 - lpole * mc - lpole * mp)
        jacobian['x_dotdot', 'f'] = -lpole / (mp * lpole * np.cos(theta)**2 - lpole * mc - lpole * mp)

        # --- derivatives of theta_dotdot ---
        # Collecting Theta Derivative
        low = mp * lpole * np.cos(theta)**2 - lpole * mc - lpole * mp
        dlow = 2. * mp * lpole * np.cos(theta) * (-np.sin(theta))
        high = (mc + mp) * g * np.sin(theta) + f * np.cos(theta) + mp * lpole * theta_dot**2 * np.sin(theta) * np.cos(theta)
        dhigh = (mc + mp) * g * np.cos(theta) - f * np.sin(theta) + mp * lpole * theta_dot**2 * (np.cos(theta)**2 - np.sin(theta)**2)
        
        # Theta_Dot - Partials
        jacobian['theta_dotdot', 'x'] = 0.
        jacobian['theta_dotdot', 'theta'] = (low * dhigh - high * dlow) / low**2
        jacobian['theta_dotdot', 'x_dot'] = 0.
        jacobian['theta_dotdot', 'theta_dot'] = 2. * theta_dot * mp * lpole * np.sin(theta) * np.cos(theta) / (mp * lpole * np.cos(theta)**2 - lpole * mc - lpole * mp)
        jacobian['theta_dotdot', 'f'] = np.cos(theta) / (mp * lpole * np.cos(theta)**2 - lpole * mc - lpole * mp)


class CartPoleClosedLoopDynamics(om.Group):
    """
    Cart-pole ODE with linear feedback control

    Parameters
    ----------
    m_cart : float
        Mass of cart.
    m_pole : float
        Mass of pole.
    l_pole : float
        Length of pole.
    d : float
        Friction coefficient
    x : 1d array
        x location of cart.
    x_dot : 1d array
        x velocity of card.
    theta : 1d array
        Angle of pole, 0 for vertical downward, positive counter clockwise.
    theta_dot : 1d array
        Angluar velocity of pole.
    K : ndarray (1, 4)
        feedback control matrix

    Returns
    -------
    x_dotdot : 1d array
        Acceleration of cart in x direction.
    theta_dotdot : 1d array
        Angular acceleration of pole.
    cost : 1d array
        Integrand of linear-quadratic cost function
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be evaluated (i.e., length of vectors x, theta, etc)')
        self.options.declare('g', default=10., desc='gravity constant')
        self.options.declare('states_ref', types=dict, desc='reference state values with keys = [x, x_dot, theta, theta_dot]')
        
    def setup(self):
        nn = self.options['num_nodes']
        states_ref = self.options['states_ref']

        # compute control input
        self.add_subsystem('feedback', _FeedbackControl(num_nodes=nn, states_ref=states_ref), promotes=['*'])

        # use open-loop dynamics
        self.add_subsystem('dynamics', CartPoleDynamicsWithFriction(num_nodes=nn, g=self.options['g'], states_ref=states_ref), promotes=['*'])


class _FeedbackControl(om.ExplicitComponent):
    """
    compute feedback control u = Kx for cartpole
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be evaluated (i.e., length of vectors x, theta, etc)')
        self.options.declare('states_ref', types=dict, desc='reference state values with keys = [x, x_dot, theta, theta_dot]')

    def setup(self):
        nn = self.options['num_nodes']
        # state vectors
        self.add_input('x', shape=(nn,), units='m')
        self.add_input('x_dot', shape=(nn,), units='m/s')
        self.add_input('theta', shape=(nn,), units='rad')
        self.add_input('theta_dot', shape=(nn,), units='rad/s')
        # controller
        self.add_input('K', shape=(1, 4))  # feedback matrix for state [x, v, theta, omega]

        # control outputs
        self.add_output('f', shape=(nn,), units='N')

        self.declare_partials(['f'], ['x', 'x_dot', 'theta', 'theta_dot'], method='exact', rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(['f'], 'K', method='cs')

    def compute(self, inputs, outputs):
        states_ref = self.options['states_ref']
        dx = inputs['x'] - states_ref['x']
        dvx = inputs['x_dot'] - states_ref['x_dot']
        dtheta = inputs['theta'] - states_ref['theta']
        domega = inputs['theta_dot'] - states_ref['theta_dot']
        K = inputs['K']

        outputs['f'] = -1. * (K[0, 0] * dx + K[0, 1] * dvx + K[0, 2] * dtheta + K[0, 3] * domega)

    def compute_partials(self, inputs, partials):
        K = inputs['K']

        partials['f', 'x'] = -K[0, 0]
        partials['f', 'x_dot'] = -K[0, 1]
        partials['f', 'theta'] = -K[0, 2]
        partials['f', 'theta_dot'] = -K[0, 3]


class CartPoleLinearizedDynamics(om.ExplicitComponent):
    """
    Computes A and B matrices for linear dynamics: dx/dt = Ax + Bu
    NOTE: hardcoded for the reference state [x, v, theta, omega] = [1, 0, pi, 0]

    Parameters
    ----------
    m_cart : float
        Mass of cart.
    m_pole : float
        Mass of pole.
    l_pole : float
        Length of pole.
    d : float
        Friction coefficient

    Returns
    -------
    A : ndarray (4, 4) for state vector [x, v, theta, omega]
    B : ndarray (4, 1)
    """

    def setup(self):
        self.add_input('m_cart', shape=(1,), units='kg')
        self.add_input('m_pole', shape=(1,), units='kg')
        self.add_input('l_pole', shape=(1,), units='m')
        self.add_input('d', val=0., units='N*s/m')

        # state vector = [x, v, theta, omega]
        self.add_output('A', shape=(4, 4))
        self.add_output('B', shape=(4, 1))

        self.declare_partials('*', '*', method='cs')   # TODO: implement partials
        self.set_check_partial_options(wrt=['m_cart', 'm_pole', 'l_pole', 'd'], method='fd', step=1e-6)

    def compute(self, inputs, outputs):
        M = inputs['m_cart']
        m = inputs['m_pole']
        L = inputs['l_pole']

        g = 10

        Sx = np.sin(np.pi)
        Cx = np.cos(np.pi)
        y3 = 0   # omega_ref

        D1 = m * Cx**2 - (m + M)
        D2 = (m * Cx**2 - (m + M)) * L

        # Fill in the Jacobian
        Jacobian = np.zeros((4, 4), dtype=complex)

        # x residual derivative
        Jacobian[0, 1] = 1.0

        # v residual derivative
        v_n = -m * g * Sx * Cx - m * L * y3**2 * Sx
        v_d = D1

        v_n_d = -m * g * (Cx * Cx + Sx * (- Sx)) - m * L * y3**2 * Cx
        v_d_d = m * 2 * Cx * (- Sx)

        Jacobian[1, 2] = (v_n_d * v_d - v_d_d * v_n) / v_d ** 2

        Jacobian[1, 3] = (1 / D1) * (- m * L * (2 * y3) * Sx)

        # theta residual derivative
        Jacobian[2, 3] = 1.0

        # omega residual derivative
        omega_n = (m + M) * g * Sx + m * L * y3**2 * Cx * Sx
        omega_d = D2
        omega_n_d = (m + M) * g * Cx + m * L * y3**2 * (- Sx * Sx + Cx * Cx)
        omega_d_d = (m * 2 * Cx * (- Sx)) * L
        Jacobian[3, 2] = (omega_n_d * omega_d - omega_d_d * omega_n) / omega_d ** 2

        Jacobian[3, 3] = (1 / D2) * (m * L * (2 * y3) * Cx * Sx)

        B = np.zeros((4, 1), dtype=complex)
        B[1, 0] = -1 / (m * Cx**2 - (m + M))   # d(v_dot) / du
        B[3, 0] = Cx / ((m * Cx**2 - (m + M)) * L)

        outputs['A'] = Jacobian
        outputs['B'] = B


def check_partials():
    # check partial derivative implementations

    nn = 3
    states_ref = {'x': 1., 'x_dot': 0., 'theta': np.pi, 'theta_dot': 0.}

    p = om.Problem()
    p.model.add_subsystem('dynamics', CartPoleDynamicsWithFriction(num_nodes=nn, states_ref=states_ref), promotes=['*'])
    # set inputs
    p.model.set_input_defaults('m_cart', val=5, units='kg')
    p.model.set_input_defaults('m_pole', val=1, units='kg')
    p.model.set_input_defaults('l_pole', val=2, units='m')
    p.model.set_input_defaults('d', val=1, units='N*s/m')
    p.model.set_input_defaults('x', val=np.random.random(nn))
    p.model.set_input_defaults('x_dot', val=np.random.random(nn))
    p.model.set_input_defaults('theta', val=np.random.random(nn))
    p.model.set_input_defaults('theta_dot', val=np.random.random(nn))
    p.model.set_input_defaults('f', val=np.random.random(nn))
    
    # check partials
    p.setup(check=True)
    p.run_model()
    # om.n2(p)
    p.check_partials(compact_print=True)

if __name__ == '__main__':
    check_partials()