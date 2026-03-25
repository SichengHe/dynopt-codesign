import numpy as np
import openmdao.api as om

"""
Equation of motion components for eVTOL dynamics model
"""

class Acceleration_LiftPlusCruise(om.ExplicitComponent):
    """
    Computes linear acceleration in x and y direction. For Lift+Cruise configuration
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be computed (length of input/output vectors)')

    def setup(self):
        nn = self.options['num_nodes']
        # rotor forces
        self.add_input('thrust', shape=(nn,), units='N')
        self.add_input('thrust_vert', shape=(nn,), units='N')   # "lift" force by rotors
        self.add_input('drag_rotor_pusher', val=np.zeros(nn), units='N')  # positive drag = opposite to thrust direction.
        self.add_input('drag_rotor_vert', val=np.zeros(nn), units='N')   # positive drag = opposite to thrust_vert direction. (downward)
        # aero forces of wing/body
        self.add_input('lift', shape=(nn,), units='N')
        self.add_input('drag', shape=(nn,), units='N')
        # other inputs
        self.add_input('theta', shape=(nn,), units='rad', desc='body tilt angle w.r.t. horizontal plane')
        self.add_input('gamma', shape=(nn,), units='rad', desc='flight path angle')
        self.add_input('m', val=10.0, units='kg', desc='UAV weight')

        self.add_output('vx_dot', shape=(nn,), units='m/s**2', desc='x acceleration')
        self.add_output('vy_dot', shape=(nn,), units='m/s**2', desc='y acceleration')

        self.declare_partials('*', ['thrust', 'thrust_vert', 'drag_rotor_pusher', 'drag_rotor_vert', 'lift', 'drag', 'theta', 'gamma'], rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('*', 'm', rows=np.arange(nn), cols=np.zeros(nn))

        self._gravity = 9.81

    def compute(self, inputs, outputs):
        thrust = inputs['thrust'] - inputs['drag_rotor_vert']
        thrust_v = inputs['thrust_vert'] - inputs['drag_rotor_pusher']
        lift = inputs['lift']
        drag = inputs['drag']
        theta = inputs['theta']
        gamma = inputs['gamma']
        m = inputs['m']

        outputs['vx_dot'] = (thrust * np.cos(theta) - drag * np.cos(gamma) - lift * np.sin(gamma) - thrust_v * np.sin(theta)) / m
        outputs['vy_dot'] = (thrust * np.sin(theta) - drag * np.sin(gamma) + lift * np.cos(gamma) + thrust_v * np.cos(theta)) / m - self._gravity

    def compute_partials(self, inputs, partials):
        thrust = inputs['thrust'] - inputs['drag_rotor_vert']
        thrust_v = inputs['thrust_vert'] - inputs['drag_rotor_pusher']
        lift = inputs['lift']
        drag = inputs['drag']
        theta = inputs['theta']
        gamma = inputs['gamma']
        m = inputs['m']

        partials['vx_dot', 'thrust'] = np.cos(theta) / m
        partials['vy_dot', 'thrust'] = np.sin(theta) / m
        partials['vx_dot', 'thrust_vert'] = -np.sin(theta) / m
        partials['vy_dot', 'thrust_vert'] = np.cos(theta) / m
        partials['vx_dot', 'drag_rotor_vert'] = -np.cos(theta) / m
        partials['vy_dot', 'drag_rotor_vert'] = -np.sin(theta) / m
        partials['vx_dot', 'drag_rotor_pusher'] = np.sin(theta) / m
        partials['vy_dot', 'drag_rotor_pusher'] = -np.cos(theta) / m
        partials['vx_dot', 'lift'] = -np.sin(gamma) / m
        partials['vy_dot', 'lift'] = np.cos(gamma) / m
        partials['vx_dot', 'drag'] = -np.cos(gamma) / m
        partials['vy_dot', 'drag'] = -np.sin(gamma) / m
        partials['vx_dot', 'theta'] = (-thrust * np.sin(theta) - thrust_v * np.cos(theta)) / m
        partials['vy_dot', 'theta'] = (thrust * np.cos(theta) - thrust_v * np.sin(theta)) / m
        partials['vx_dot', 'gamma'] = (drag * np.sin(gamma) - lift * np.cos(gamma)) / m
        partials['vy_dot', 'gamma'] = (-drag * np.cos(gamma) - lift * np.sin(gamma)) / m
        partials['vx_dot', 'm'] = -(thrust * np.cos(theta) - drag * np.cos(gamma) - lift * np.sin(gamma) - thrust_v * np.sin(theta)) / m**2
        partials['vy_dot', 'm'] = -(thrust * np.sin(theta) - drag * np.sin(gamma) + lift * np.cos(gamma) + thrust_v * np.cos(theta)) / m**2


class AngularAcceleration_LiftPlusCruise(om.ExplicitComponent):
    """
    Computes angular acceleration of pitch angle. For Lift+Cruise configuration
    """
    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='number of nodes to be computed (length of input/output vectors)')

    def setup(self):
        nn = self.options['num_nodes']
        
        # forces
        self.add_input('thrust_vert_1', shape=(nn,), units='N')   # lifter thrust 1
        self.add_input('thrust_vert_2', shape=(nn,), units='N')   # lifter thrust 2
        self.add_input('lift_wing', shape=(nn,), units='N')
        self.add_input('lift_tail', shape=(nn,), units='N')
        self.add_input('drag_wing', shape=(nn,), units='N')
        self.add_input('drag_tail', shape=(nn,), units='N')
        self.add_input('moment_wing', shape=(nn,), units='N*m')   # positive for pitch up
        self.add_input('moment_tail', shape=(nn,), units='N*m')
        # angles
        self.add_input('theta', shape=(nn,), units='rad', desc='pitch angle (body tilt angle w.r.t. horizontal plane)')
        self.add_input('gamma', shape=(nn,), units='rad', desc='flight path angle')
        # design variables
        self.add_input('Ipitch', val=1.0, units='kg*m**2', desc='moment of inertia')
        # all locations x are length-wise location w.r.t. CG. Negative when this is in front of CG
        self.add_input('x1', val=0., units='m', desc='location of rotor_vert_1')
        self.add_input('x2', val=0., units='m', desc='location of rotor_vert_2')
        self.add_input('x_ac_wing', val=0., units='m', desc='location of wing aerodynamic center')
        self.add_input('x_ac_tail', val=0., units='m', desc='location of tail aerodynamic center')

        self.add_output('theta_dotdot', shape=(nn,), units='rad/s**2', desc='angular acceleration')

        vars_vector = ['thrust_vert_1', 'thrust_vert_2', 'lift_wing', 'drag_wing', 'moment_wing', 'lift_tail', 'drag_tail', 'moment_tail', 'theta', 'gamma']
        self.declare_partials('*', vars_vector, rows=np.arange(nn), cols=np.arange(nn))
        self.declare_partials('*', ['Ipitch', 'x1', 'x2', 'x_ac_wing', 'x_ac_tail'], rows=np.arange(nn), cols=np.zeros(nn))

        self._gravity = 9.81

    def compute(self, inputs, outputs):
        alpha = inputs['theta'] - inputs['gamma']   # body AoA
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        thrust_torque = -inputs['thrust_vert_1'] * inputs['x1'] - inputs['thrust_vert_2'] * inputs['x2']
        wing_torque = -(inputs['lift_wing'] * ca + inputs['drag_wing'] * sa) * inputs['x_ac_wing'] + inputs['moment_wing']
        tail_torque = -(inputs['lift_tail'] * ca + inputs['drag_tail'] * sa) * inputs['x_ac_tail'] + inputs['moment_tail']

        outputs['theta_dotdot'] = (thrust_torque + wing_torque + tail_torque) / inputs['Ipitch']

    def compute_partials(self, inputs, partials):
        alpha = inputs['theta'] - inputs['gamma']   # body AoA
        Ipitch = inputs['Ipitch']
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        thrust_torque = -inputs['thrust_vert_1'] * inputs['x1'] - inputs['thrust_vert_2'] * inputs['x2']
        wing_torque = -(inputs['lift_wing'] * ca + inputs['drag_wing'] * sa) * inputs['x_ac_wing'] + inputs['moment_wing']
        tail_torque = -(inputs['lift_tail'] * ca + inputs['drag_tail'] * sa) * inputs['x_ac_tail'] + inputs['moment_tail']

        partials['theta_dotdot', 'thrust_vert_1'] = -inputs['x1'] / Ipitch
        partials['theta_dotdot', 'thrust_vert_2'] = -inputs['x2'] / Ipitch
        partials['theta_dotdot', 'lift_wing'] = -ca * inputs['x_ac_wing'] / Ipitch
        partials['theta_dotdot', 'lift_tail'] = -ca * inputs['x_ac_tail'] / Ipitch
        partials['theta_dotdot', 'drag_wing'] = -sa * inputs['x_ac_wing'] / Ipitch
        partials['theta_dotdot', 'drag_tail'] = -sa * inputs['x_ac_tail'] / Ipitch
        partials['theta_dotdot', 'moment_wing'] = 1. / Ipitch
        partials['theta_dotdot', 'moment_tail'] = 1. / Ipitch

        partials['theta_dotdot', 'theta'] = (-(inputs['lift_wing'] * (-sa) + inputs['drag_wing'] * ca) * inputs['x_ac_wing'] - (inputs['lift_tail'] * (-sa) + inputs['drag_tail'] * ca) * inputs['x_ac_tail'])/ Ipitch
        partials['theta_dotdot', 'gamma'] = -1. * partials['theta_dotdot', 'theta']
        partials['theta_dotdot', 'Ipitch'] = -(thrust_torque + wing_torque + tail_torque) / Ipitch**2
        partials['theta_dotdot', 'x1'] = -inputs['thrust_vert_1'] / Ipitch
        partials['theta_dotdot', 'x2'] = -inputs['thrust_vert_2'] / Ipitch
        partials['theta_dotdot', 'x_ac_wing'] = -(inputs['lift_wing'] * np.cos(alpha) + inputs['drag_wing'] * np.sin(alpha))
        partials['theta_dotdot', 'x_ac_tail'] = -(inputs['lift_tail'] * np.cos(alpha) + inputs['drag_tail'] * np.sin(alpha))
      