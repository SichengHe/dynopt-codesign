import numpy as np
import openmdao.api as om

from quadrotormodels.components_eom import Acceleration_LiftPlusCruise, AngularAcceleration_LiftPlusCruise
from quadrotormodels.components_bemt import MultiRotorGroup


class VTOLDynamicsGroup_MultiRotor_3DOF(om.Group):
    """
    3 DOF dynamics model for quadrotor (no wing) configuration.
    For open-loop trajectory analysis and optimization

    --- inputs ---
    States: theta, vx, vy, theta_vel
    Controls: omega_vert_1, omega_vert_2 (rotational speed of left and right rotor)
    Design: m (quadrotor weight), Ipitch (pitch moment of inertia), loc_rotors (distance from CG to rotor center),
             rotor_vert_1|R (radius), rotor_vert_1|theta_cp (twist distribution), rotor_vert_1|chord_cp (chord distribution), rotor_vert_2|R, rotor_vert_2|theta_cp, rotor_vert_2|chord_cp

    --- outputs ---
    Rate of states: x_dot, y_dot, theta_dot, vx_dot, vy_dot, theta_dotdot
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('params_dict', types=dict)   # dict containing all constant parameter

    def setup(self):
        nn = self.options['num_nodes']
        params = self.options['params_dict']
        n_rotors = params['num_rotors']   # this should be 0.
        config = params['config']

        if config == 'multirotor':
            # params dict must contain the number of vertical rotors.
            n_rotors_vert = params['num_rotors_vert']
            if n_rotors_vert != 4:
                raise RuntimeError('Currently only implements quadrotor. n_rotor_vert must be 4.')
        else:
            raise RuntimeError()('Config must be multirotor.')
        # END IF

        # print('\n---------------------------------')
        # print(' setting up ODE for', config, '| n_rotors =', n_rotors, ',n_rotors_vertical =', n_rotors_vert)
        # print('---------------------------------\n')

        # --- design parameters ---
        rho = params['rho_air']  # air densigy
        Sref_body = params['S_body_ref']  # body reference area

        # --- flight path angle ---
        gamma_comp = om.ExecComp('gamma = arctan2(vy + 1e-100, vx)',   # add 1e-100 to avoid singularity. vx=0, vy=0 then returns gamma=90 (climb)
                                    gamma={'shape': (nn,), 'units': 'rad'},
                                    vx={'shape': (nn,), 'units': 'm/s'},
                                    vy={'shape': (nn,), 'units': 'm/s'},
                                    has_diag_partials=True)
        self.add_subsystem('flightpath', gamma_comp, promotes_inputs=['*'], promotes_outputs=['*'])

        # propeller inflow
        vinf_comp = om.ExecComp('v_inf = (vx**2 + vy**2)**0.5',
                                    v_inf={'shape': (nn,), 'units': 'm/s'},
                                    vx={'shape': (nn,), 'units': 'm/s'},
                                    vy={'shape': (nn,), 'units': 'm/s'},
                                    has_diag_partials=True)
        inflow_comp = om.ExecComp(['v_normal = v_inf * cos(theta - gamma)', 'v_parallel_rev = -1.0 * v_inf * sin(theta - gamma)'],
                                    v_normal={'shape': (nn,), 'units': 'm/s'},
                                    v_parallel_rev={'shape': (nn,), 'units': 'm/s'},   # flipped sign here for Lift+cruise inflow
                                    v_inf={'shape': (nn,), 'units': 'm/s'},
                                    theta={'shape': (nn,), 'units': 'rad'},
                                    gamma={'shape': (nn,), 'units': 'rad'},
                                    has_diag_partials=True)
        self.add_subsystem('freestream', vinf_comp, promotes_inputs=['*'], promotes_outputs=['*'])
        self.add_subsystem('inflow', inflow_comp, promotes_inputs=['*'], promotes_outputs=['*'])

        # --- geometry ---
        geometry_comp = om.ExecComp(['x1 = -1. * loc_rotors', 'x2 = loc_rotors'], shape=(1,), units='m')
        self.add_subsystem('geometry', geometry_comp, promotes=['*'])

        # --- propulsion model ---
        # compute rotor inflows (add (theta_vel * l) to normal inflow.)
        normal_inflow_comp = om.ExecComp(['v_normal_1 = v_parallel_rev + theta_vel * loc_rotors', 'v_normal_2 = v_parallel_rev - theta_vel * loc_rotors'],
                                            v_normal_1={'shape' : (nn), 'units' : 'm/s'},
                                            v_normal_2={'shape' : (nn), 'units' : 'm/s'},
                                            v_parallel_rev={'shape' : (nn), 'units' : 'm/s'},
                                            theta_vel={'shape' : (nn), 'units' : 'rad/s'},
                                            loc_rotors={'shape' : (1,), 'units' : 'm'},
                                            has_diag_partials=True)
        self.add_subsystem('inflow_pitching_correction', normal_inflow_comp, promotes=['v_parallel_rev', 'theta_vel', 'loc_rotors'])

        # 1) rotor_vert_1: rotors for lift, in front of CG. Rotor inflow = -1 * v_parallel = v_parallel_rev
        self.add_subsystem('propulsion_vert_1', MultiRotorGroup(num_nodes=nn, num_rotor=int(n_rotors_vert / 2), rotor_design_dict=params['rotor_vert_design'], rho=params['rho_air']),
                                promotes_inputs=[('omega', 'omega_vert_1'), ('Rtip', 'rotor_vert_1|R'), ('theta_cp', 'rotor_vert_1|theta_cp'), ('chord_cp', 'rotor_vert_1|chord_cp')],
                                promotes_outputs=[('power', 'power_for_lift_1'), ('thrust', 'thrust_vert_1')])
        self.connect('inflow_pitching_correction.v_normal_1', 'propulsion_vert_1.v_normal')
        self.connect('v_normal', 'propulsion_vert_1.v_pal')
        # 2) rotor_vert_2: rotors for lift, rear of CG.
        self.add_subsystem('propulsion_vert_2', MultiRotorGroup(num_nodes=nn, num_rotor=int(n_rotors_vert / 2), rotor_design_dict=params['rotor_vert_design'], rho=params['rho_air']),
                                promotes_inputs=[('omega', 'omega_vert_2'), ('Rtip', 'rotor_vert_2|R'), ('theta_cp', 'rotor_vert_2|theta_cp'), ('chord_cp', 'rotor_vert_2|chord_cp')],
                                promotes_outputs=[('power', 'power_for_lift_2'), ('thrust', 'thrust_vert_2')])
        self.connect('inflow_pitching_correction.v_normal_2', 'propulsion_vert_2.v_normal')
        self.connect('v_normal', 'propulsion_vert_2.v_pal')
        # sum thrusts and powers of the vertical rotors
        # TODO: replace these with ExecComp of with has_diag_partial=True ??
        vert_sum = om.AddSubtractComp()
        vert_sum.add_equation('power', input_names=['power_for_lift_1', 'power_for_lift_2'], vec_size=nn, units='W')
        vert_sum.add_equation('thrust_vert', input_names=['thrust_vert_1', 'thrust_vert_2'], vec_size=nn, units='N')
        self.add_subsystem('vert_rotors_sum', vert_sum, promotes_inputs=['*'], promotes_outputs=['*'])

        # --- aero models ---
        # body drag assuming the body is sphere
        drag_body_comp = om.ExecComp(['drag_body = 0.5 * rho * v_inf**2 * Sref_body * Cd_body', 'lift_body = 0'],
                                        drag_body={'shape': (nn,), 'units': 'N'},
                                        lift_body={'shape': (nn,), 'units': 'N'},
                                        rho={'val': rho, 'units': 'kg/m**3'},
                                        v_inf={'shape': (nn,), 'units': 'm/s'},
                                        Sref_body={'val': Sref_body, 'units': 'm**2'},
                                        Cd_body={'val' : 0.3, 'units': None},
                                        has_diag_partials=True)
        self.add_subsystem('drag_body', drag_body_comp, promotes_inputs=['v_inf'], promotes_outputs=[('drag_body', 'drag'), ('lift_body', 'lift')])

        # --- Equations of motion ---
        self.add_subsystem('accel_linear', Acceleration_LiftPlusCruise(num_nodes=nn), promotes_inputs=['thrust_vert', 'lift', 'drag', 'theta', 'gamma', 'm'], promotes_outputs=['vx_dot', 'vy_dot'])
        self.add_subsystem('accel_angular', AngularAcceleration_LiftPlusCruise(num_nodes=nn), promotes_inputs=['thrust_vert_1', 'thrust_vert_2', 'theta', 'gamma', 'Ipitch', 'x1', 'x2'], promotes_outputs=['theta_dotdot'])
        # set 0 values for not relevant variables; then, we can reuse the L+C EoM for wingless multirotor.
        self.set_input_defaults('accel_linear.thrust', np.zeros(nn))
        self.set_input_defaults('accel_angular.lift_wing', np.zeros(nn))
        self.set_input_defaults('accel_angular.lift_tail', np.zeros(nn))
        self.set_input_defaults('accel_angular.drag_wing', np.zeros(nn))
        self.set_input_defaults('accel_angular.drag_tail', np.zeros(nn))
        self.set_input_defaults('accel_angular.moment_wing', np.zeros(nn))
        self.set_input_defaults('accel_angular.moment_tail', np.zeros(nn))
        self.set_input_defaults('accel_angular.x_ac_wing', 0.)
        self.set_input_defaults('accel_angular.x_ac_tail', 0.)
        
        # other ODEs
        ode_comp = om.ExecComp(['x_dot = vx', 'y_dot = vy', 'theta_dot = theta_vel', 'energy_dot = power_in'],
                                    x_dot={'shape': (nn,), 'units': 'm/s'},
                                    y_dot={'shape': (nn,), 'units': 'm/s'},
                                    vx={'shape': (nn,), 'units': 'm/s'},
                                    vy={'shape': (nn,), 'units': 'm/s'},
                                    theta_dot={'shape': (nn,), 'units': 'rad/s'},
                                    theta_vel={'shape': (nn,), 'units': 'rad/s'},
                                    energy_dot={'shape': (nn,), 'units': 'W'},
                                    power_in={'shape': (nn,), 'units': 'W'},
                                    has_diag_partials=True)
        self.add_subsystem('ode_misc', ode_comp, promotes_inputs=['vx', 'vy', 'theta_vel', ('power_in', 'power')], promotes_outputs=['*'])


class VTOLDynamicsGroup_MultiRotor_3DOF_ClosedLoop(om.Group):
    """
    3 DOF dynamics model for quadrotor (no wing) configuration.
    For closed-loop trajectory analysis and optimization.
    This group has the same E.o.M as VTOLDynamicsGroup_MultiRotor_3DOF, but now the control in determined by linear feedback u = - K x + u_ref.
    It also computes the rate (or integrand) of quadratic cost function. Then cost is computed by integration via the collocation method.

    --- inputs ---
    States: x, y, theta, vx, vy, theta_vel
    Controls: omega_vert_1, omega_vert_2 (rotational speed of left and right rotor)
    Quadrotor design: m (quadrotor weight), Ipitch (pitch moment of inertia), loc_rotors (distance from CG to rotor center),
             rotor_vert_1|R (radius), rotor_vert_1|theta_cp (twist distribution), rotor_vert_1|chord_cp (chord distribution), rotor_vert_2|R, rotor_vert_2|theta_cp, rotor_vert_2|chord_cp
    Control design: K (linear feedback matrix), omega_ref (reference control input at the target steady state)
    Reference steady state: x_ref, y_ref, theta_ref, vx_ref, vy_ref, theta_vel_ref

    --- outputs ---
    Rate of states: x_dot, y_dot, theta_dot, vx_dot, vy_dot, theta_dotdot, cost_rate

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('params_dict', types=dict)   # dict containing all constant parameter
        # NOTE: currently only supports diagonal Q and R
        self.options.declare('Q', desc='cost matrix for state, (ns,)')
        self.options.declare('R', desc='cost matrix for control, (nc,)')

    def setup(self):
        nn = self.options['num_nodes']
        params = self.options['params_dict']

        # control input by feedback
        self.add_subsystem('feedback', _FeedbackControl(num_nodes=nn), promotes=['*'])
        control = om.ExecComp(['omega_vert_1 = delta_omega_1 + omega_ref', 'omega_vert_2 = delta_omega_2 + omega_ref'],
                              omega_vert_1={'shape': (nn), 'units': 'rad/s'},
                              omega_vert_2={'shape': (nn), 'units': 'rad/s'},
                              delta_omega_1={'shape': (nn), 'units': 'rad/s'},
                              delta_omega_2={'shape': (nn), 'units': 'rad/s'},
                              omega_ref={'shape': (1,), 'units': 'rad/s'},
                              has_diag_partials=True)
        self.add_subsystem('control_input', control, promotes=['*'])

        # dynamics (use open-loop model)
        self.add_subsystem('dynamics', VTOLDynamicsGroup_MultiRotor_3DOF(num_nodes=nn, params_dict=params), promotes_inputs=['*'], promotes_outputs=['*'])
        
        # cost
        self.add_subsystem('cost', _QuadraticCostRate(num_nodes=nn, Q=self.options['Q'], R=self.options['R']), promotes=['*'])


class _FeedbackControl(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be evaluated (i.e., length of vectors x, theta, etc)')

    def setup(self):
        nn = self.options['num_nodes']
        # state vectors
        self.add_input('x', shape=(nn,), units='m')
        self.add_input('y', shape=(nn,), units='m')
        self.add_input('theta', shape=(nn,), units='rad')
        self.add_input('vx', shape=(nn,), units='m/s')
        self.add_input('vy', shape=(nn,), units='m/s')
        self.add_input('theta_vel', shape=(nn,), units='rad/s')
        # reference states
        self.add_input('x_ref', val=0., units='m')
        self.add_input('y_ref', val=0., units='m')
        self.add_input('theta_ref', val=0., units='rad')
        self.add_input('vx_ref', val=0., units='m/s')
        self.add_input('vy_ref', val=0.1, units='m/s')   # only non-zero entry
        self.add_input('theta_vel_ref', val=0., units='rad/s')
        # controller
        self.add_input('K', shape=(2, 6))  # control matrix

        # control outputs
        self.add_output('delta_omega_1', shape=(nn,), units='rad/s')  # delta from reference state
        self.add_output('delta_omega_2', shape=(nn,), units='rad/s')

        self.declare_partials(['*'], ['x', 'y', 'theta', 'vx', 'vy', 'theta_vel'], method='exact', rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(['*'], wrt=['x_ref', 'y_ref', 'theta_ref', 'vx_ref', 'vy_ref', 'theta_vel_ref'], method='exact', rows=np.arange(nn), cols=np.zeros(nn))
        self.declare_partials(['*'], 'K', method='cs')

    def compute(self, inputs, outputs):
        dx = inputs['x'] - inputs['x_ref']
        dy = inputs['y'] - inputs['y_ref']
        dtheta = inputs['theta'] - inputs['theta_ref']
        dvx = inputs['vx'] - inputs['vx_ref']
        dvy = inputs['vy'] - inputs['vy_ref']
        dtheta_vel = inputs['theta_vel'] - inputs['theta_vel_ref']
        K = inputs['K']

        outputs['delta_omega_1'] = -1. * (K[0, 0] * dx + K[0, 1] * dy + K[0, 2] * dtheta + K[0, 3] * dvx + K[0, 4] * dvy + K[0, 5] * dtheta_vel)
        outputs['delta_omega_2'] = -1. * (K[1, 0] * dx + K[1, 1] * dy + K[1, 2] * dtheta + K[1, 3] * dvx + K[1, 4] * dvy + K[1, 5] * dtheta_vel)

    def compute_partials(self, inputs, partials):
        K = inputs['K']

        partials['delta_omega_1', 'x'] = -K[0, 0]
        partials['delta_omega_1', 'y'] = -K[0, 1]
        partials['delta_omega_1', 'theta'] = -K[0, 2]
        partials['delta_omega_1', 'vx'] = -K[0, 3]
        partials['delta_omega_1', 'vy'] = -K[0, 4]
        partials['delta_omega_1', 'theta_vel'] = -K[0, 5]

        partials['delta_omega_2', 'x'] = -K[1, 0]
        partials['delta_omega_2', 'y'] = -K[1, 1]
        partials['delta_omega_2', 'theta'] = -K[1, 2]
        partials['delta_omega_2', 'vx'] = -K[1, 3]
        partials['delta_omega_2', 'vy'] = -K[1, 4]
        partials['delta_omega_2', 'theta_vel'] = -K[1, 5]

        partials['delta_omega_1', 'x_ref'] = K[0, 0]
        partials['delta_omega_1', 'y_ref'] = K[0, 1]
        partials['delta_omega_1', 'theta_ref'] = K[0, 2]
        partials['delta_omega_1', 'vx_ref'] = K[0, 3]
        partials['delta_omega_1', 'vy_ref'] = K[0, 4]
        partials['delta_omega_1', 'theta_vel_ref'] = K[0, 5]

        partials['delta_omega_2', 'x_ref'] = K[1, 0]
        partials['delta_omega_2', 'y_ref'] = K[1, 1]
        partials['delta_omega_2', 'theta_ref'] = K[1, 2]
        partials['delta_omega_2', 'vx_ref'] = K[1, 3]
        partials['delta_omega_2', 'vy_ref'] = K[1, 4]
        partials['delta_omega_2', 'theta_vel_ref'] = K[1, 5]


class _QuadraticCostRate(om.ExplicitComponent):
    """
    Computes the integrand of the quadratic cost function
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be evaluated (i.e., length of vectors x, theta, etc)')
        self.options.declare('Q', desc='cost matrix for state, (ns,)')
        self.options.declare('R', desc='cost matrix for control, (nc,)')

    def setup(self):
        nn = self.options['num_nodes']
        # state vectors
        self.add_input('x', shape=(nn,), units='m')
        self.add_input('y', shape=(nn,), units='m')
        self.add_input('theta', shape=(nn,), units='rad')
        self.add_input('vx', shape=(nn,), units='m/s')
        self.add_input('vy', shape=(nn,), units='m/s')
        self.add_input('theta_vel', shape=(nn,), units='rad/s')
        # reference states
        self.add_input('x_ref', val=0., units='m')
        self.add_input('y_ref', val=0., units='m')
        self.add_input('theta_ref', val=0., units='rad')
        self.add_input('vx_ref', val=0., units='m/s')
        self.add_input('vy_ref', val=0.1, units='m/s')   # only non-zero entry
        self.add_input('theta_vel_ref', val=0., units='rad/s')
        # control vectors
        self.add_input('delta_omega_1', shape=(nn,), units='rad/s')   # omega_vert_1 - omega_ref
        self.add_input('delta_omega_2', shape=(nn,), units='rad/s')

        # output
        self.add_output('cost_rate', shape=(nn,), desc='integrand of quadratic objective (xQx + uRu)')

        self.declare_partials(of=['*'], wrt=['x', 'y', 'theta', 'vx', 'vy', 'theta_vel', 'delta_omega_1', 'delta_omega_2'], method='exact', rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials(of=['*'], wrt=['x_ref', 'y_ref', 'theta_ref', 'vx_ref', 'vy_ref', 'theta_vel_ref'], method='exact', rows=np.arange(nn), cols=np.zeros(nn))

    def compute(self, inputs, outputs):
        Q = np.diag(self.options['Q'])
        R = np.diag(self.options['R'])
        
        # error of the states
        dx = inputs['x'] - inputs['x_ref']
        dy = inputs['y'] - inputs['y_ref']
        dtheta = inputs['theta'] - inputs['theta_ref']
        dvx = inputs['vx'] - inputs['vx_ref']
        dvy = inputs['vy'] - inputs['vy_ref']
        dtheta_vel = inputs['theta_vel'] - inputs['theta_vel_ref']
        domega1 = inputs['delta_omega_1']
        domega2 = inputs['delta_omega_2']
         
        # cost integrand (assumes cost matrices are identity for both state and cost)
        outputs['cost_rate'] = Q[0] * dx**2 + Q[1] * dy**2 + Q[2] * dtheta**2 + Q[3] * dvx**2 + Q[4] * dvy**2 + Q[5] * dtheta_vel**2 + R[0] * domega1**2 + R[1] * domega2**2

    def compute_partials(self, inputs, partials):
        Q = np.diag(self.options['Q'])
        R = np.diag(self.options['R'])

        partials['cost_rate', 'x'] = 2 * (inputs['x'] - inputs['x_ref']) * Q[0]
        partials['cost_rate', 'y'] = 2 * (inputs['y'] - inputs['y_ref']) * Q[1]
        partials['cost_rate', 'theta'] = 2 * (inputs['theta'] - inputs['theta_ref']) * Q[2]
        partials['cost_rate', 'vx'] = 2 * (inputs['vx'] - inputs['vx_ref']) * Q[3]
        partials['cost_rate', 'vy'] = 2 * (inputs['vy'] - inputs['vy_ref']) * Q[4]
        partials['cost_rate', 'theta_vel'] = 2 * (inputs['theta_vel'] - inputs['theta_vel_ref']) * Q[5]

        partials['cost_rate', 'x_ref'] = -2 * (inputs['x'] - inputs['x_ref']) * Q[0]
        partials['cost_rate', 'y_ref'] = -2 * (inputs['y'] - inputs['y_ref']) * Q[1]
        partials['cost_rate', 'theta_ref'] = -2 * (inputs['theta'] - inputs['theta_ref']) * Q[2]
        partials['cost_rate', 'vx_ref'] = -2 * (inputs['vx'] - inputs['vx_ref']) * Q[3]
        partials['cost_rate', 'vy_ref'] = -2 * (inputs['vy'] - inputs['vy_ref']) * Q[4]
        partials['cost_rate', 'theta_vel_ref'] = -2 * (inputs['theta_vel'] - inputs['theta_vel_ref']) * Q[5]

        partials['cost_rate', 'delta_omega_1'] = 2 * inputs['delta_omega_1'] * R[0]
        partials['cost_rate', 'delta_omega_2'] = 2 * inputs['delta_omega_2'] * R[1]
