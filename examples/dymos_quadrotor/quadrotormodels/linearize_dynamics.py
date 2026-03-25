import numpy as np
import openmdao.api as om

import quadrotormodels
from quadrotormodels import VTOLDynamicsGroup_MultiRotor_3DOF as VTOLDynamics

# linearize dynamics given the trim states and control inputs. Nested OpenMDAO model

class LinearizeDynamics(om.ExplicitComponent):
    
    def initialize(self):
        self.options.declare('params_dict', types=dict)
        self.options.declare('fd_step_size', default=1.0e-8)
    
    def setup(self):
        # design variables
        params_dict = self.options['params_dict']
        rotor_design = params_dict['rotor_vert_design']
        self.add_input('m', val=params_dict['mass'], units='kg')
        self.add_input('Ipitch', val=params_dict['Icg'], units='kg*m**2')
        self.add_input('rotor_vert|R', val=rotor_design['Rtip'], units='m',)   # rotor radius
        self.add_input('rotor_vert|theta_cp', val=rotor_design['theta_cp'], units='deg')   # twist
        self.add_input('rotor_vert|chord_cp', val=rotor_design['chord_cp'], units='m')   # chord
        self.add_input('loc_rotors', val=params_dict['loc_rotors'], units='m')
        # inflow condition
        self.add_input('vy_ref', val=0.01, units='m/s')   # target climb velocity. HARDCODED!
        # control input (at trim)
        self.add_input('omega_ref', val=rotor_design['omega_ref'], units='rad/s')

        # output: derivatives of dynamics (elements of A and B matrix)
        self.add_output('d_vxdot_vx', val=1)
        self.add_output('d_vydot_vy', val=1)
        self.add_output('d_vydot_omega1', val=1)
        self.add_output('d_vydot_omega2', val=1)
        self.add_output('d_thetadotdot_omega1', val=1)
        self.add_output('d_thetadotdot_omega2', val=1)

        # uncommend these for for step-size study
        ### self.declare_partials(of=['*'], wrt=['vy_ref', 'omega_ref', 'rotor_vert|theta_cp', 'rotor_vert|chord_cp'], method='fd', step_calc='rel_avg', step=self.options['fd_step_size'], form='central')
        # NOTE: assume m, Ipicth, R, loc_rotors are constant

        # central difference with the best step sizes
        self.declare_partials(of=['*'], wrt=['vy_ref'], method='fd', step_calc='rel_avg', step=1e-5, form='central')
        self.declare_partials(of=['*'], wrt=['omega_ref'], method='fd', step_calc='rel_avg', step=1e-4, form='central')
        self.declare_partials(of=['*'], wrt=['rotor_vert|theta_cp', 'rotor_vert|chord_cp'], method='fd', step_calc='rel_avg', step=1e-5, form='central')
        
        # ------------------------------
        # setup OpenMDAO subproblem
        self.__sub_problem = om.Problem()

        # design variables
        rotor_design = params_dict['rotor_vert_design']
        design_vars = om.IndepVarComp()
        design_vars.add_output('m', val=params_dict['mass'], units='kg')
        design_vars.add_output('Ipitch', val=params_dict['Icg'], units='kg*m**2')
        design_vars.add_output('rotor_vert|R', val=rotor_design['Rtip'], units='m',)   # rotor radius
        design_vars.add_output('rotor_vert|theta_cp', val=rotor_design['theta_cp'], units='deg')   # twist
        design_vars.add_output('rotor_vert|chord_cp', val=rotor_design['chord_cp'], units='m')   # chord
        design_vars.add_output('loc_rotors', val=params_dict['loc_rotors'], units='m')
        self.__sub_problem.model.add_subsystem('design_vars', design_vars, promotes_outputs=['*'])

        # dynamics
        self.__sub_problem.model.add_subsystem('dynamics', VTOLDynamics(num_nodes=1, params_dict=params_dict), promotes=['*'])
        self.__sub_problem.model.connect('rotor_vert|R', ['rotor_vert_1|R', 'rotor_vert_2|R'])
        self.__sub_problem.model.connect('rotor_vert|theta_cp', ['rotor_vert_1|theta_cp', 'rotor_vert_2|theta_cp'])
        self.__sub_problem.model.connect('rotor_vert|chord_cp', ['rotor_vert_1|chord_cp', 'rotor_vert_2|chord_cp'])

        self.__sub_problem.setup(check=False)
        # om.n2(p)

        # set reference states
        self.__sub_problem.set_val('vx', 0., units='m/s')
        self.__sub_problem.set_val('vy', 0.01, units='m/s')   # setting to exact 0 would cause some issue
        self.__sub_problem.set_val('theta', 0., units='deg')
        self.__sub_problem.set_val('theta_vel', 0., units='deg/s')

    def compute(self, inputs, outputs):
        # solve trim system
        derivatives = wrap_om_dynamics(self.__sub_problem, inputs)

        # components of A matrix
        outputs['d_vxdot_vx'] = derivatives[('vx_dot', 'vx')]
        outputs['d_vydot_vy'] = derivatives[('vy_dot', 'vy')]

        # components of B matrix
        outputs['d_vydot_omega1'] = derivatives[('vy_dot', 'omega_vert_1')]
        outputs['d_vydot_omega2'] = derivatives[('vy_dot', 'omega_vert_2')]
        outputs['d_thetadotdot_omega1'] = derivatives[('theta_dotdot', 'omega_vert_1')]
        outputs['d_thetadotdot_omega2'] = derivatives[('theta_dotdot', 'omega_vert_2')]


def wrap_om_dynamics(p, inputs):
    # set design
    p.set_val('m', val=inputs['m'], units='kg')
    p.set_val('Ipitch', val=inputs['Ipitch'], units='kg*m**2')
    p.set_val('rotor_vert|R', val=inputs['rotor_vert|R'], units='m',)   # rotor radius
    p.set_val('rotor_vert|theta_cp', val=inputs['rotor_vert|theta_cp'], units='deg')   # twist
    p.set_val('rotor_vert|chord_cp', val=inputs['rotor_vert|chord_cp'], units='m')   # chord
    p.set_val('loc_rotors', val=inputs['loc_rotors'], units='m')
    # set reference states and controls
    p.set_val('vy', inputs['vy_ref'], units='m/s')
    p.set_val('omega_vert_1', inputs['omega_ref'], units='rad/s')
    p.set_val('omega_vert_2', inputs['omega_ref'], units='rad/s')

    p.run_model()
    ### om.n2(p)

    totals = p.compute_totals(of=['vx_dot', 'vy_dot', 'theta_dotdot'], wrt=['theta', 'vx', 'vy', 'omega_vert_1', 'omega_vert_2'])
    ### print(totals)
    return totals


def _check_model():
    path = quadrotormodels.__path__[0]

    rotor_r = 0.133  # rotor radius, m
    mass = 1.4  # vehicle MTOW, kg
    params_dict = {'S_body_ref' : 0.1365 / (0.5 * 1.225 * 0.3),  # body reference area, m**2, set so that (0.5 rho S CD) = 0.1365 (from Quan's book) with Cd=0.3
                   'rho_air': 1.225,  # air density
                   'mass' : mass,
                   'loc_rotors' : 0.225 / np.sqrt(2),   # m, distance from CG to rotor center in X configuration
                   'num_rotors' : 0,
                   'num_rotors_vert' : 4,  # for lift
                   'config' : 'multirotor',
                   }
    # vehicle moment of inertia
    params_dict['Icg'] = 0.0211   # kg-m^2
    # rotor design (optimized for minimum hover)
    num_cp = 20   # span-wise control points of twist and chord
    rotor_design = {'num_blades' : 2, 'Rtip' : rotor_r, 'num_cp' : num_cp, 'chord_cp' : np.array([0.04306510061311549, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.04396547538022499, 0.039847783680244324, 0.035794016939473317, 0.03163410897693981, 0.02718045311269029, 0.02209681742445619, 0.015477739025270299, 0.005]),
                    'theta_cp' : np.array([27.690181149634036, 27.455172875504946, 25.495688529069533, 23.304234714315378, 21.249495549081875, 19.328331022437958, 17.59816149341496, 16.072594662191236, 14.734347551290448, 13.558417660087255, 12.530640307054233, 11.733759479199708, 11.072322158736556, 10.518817772718993, 10.01215463997566, 9.54456137472585, 9.10900036533994, 8.690094710302423, 8.313328232992301, 7.8662041093808455]), 'cr75' : 0.025 / 0.133, 'omega_ref' : 471.94068266,
                    'xfoil_fnames' : path + '/xfoil-data/xf-n0012-il-100000.dat',
                    'BEMT_num_radius' : 40, 'BEMT_num_azimuth' : 4,
                    }
    params_dict['rotor_vert_design'] = rotor_design
    # ------------------------------

    # -----------------------------------------------------
    # 1) solve for reference control input & linearize
    p = om.Problem()

    # design variables
    rotor_design = params_dict['rotor_vert_design']
    design_vars = om.IndepVarComp()
    design_vars.add_output('m', val=params_dict['mass'], units='kg')
    design_vars.add_output('Ipitch', val=params_dict['Icg'], units='kg*m**2')
    design_vars.add_output('rotor_vert|R', val=rotor_design['Rtip'], units='m',)   # rotor radius
    design_vars.add_output('rotor_vert|theta_cp', val=rotor_design['theta_cp'], units='deg')   # twist
    design_vars.add_output('rotor_vert|chord_cp', val=rotor_design['chord_cp'], units='m')   # chord
    design_vars.add_output('loc_rotors', val=params_dict['loc_rotors'], units='m')
    p.model.add_subsystem('design_vars', design_vars, promotes_outputs=['*'])

    # trim & linearlization
    p.model.add_subsystem('trim', LinearizeDynamics(params_dict=params_dict), promotes=['*'])
    p.model.set_input_defaults('vy_ref', 0.01, units='m/s')
    p.model.set_input_defaults('omega_ref', rotor_design['omega_ref'], units='rad/s')

    p.setup(check=True)
    p.run_model()

    print('d(vx_dot) / d(vx)', p.get_val('d_vxdot_vx'))
    print('d(vy_dot) / d(vy)', p.get_val('d_vydot_vy'))
    print('d(vy_dot) / d(omega1)', p.get_val('d_vydot_omega1'))
    print('d(vy_dot) / d(omega2)', p.get_val('d_vydot_omega2'))
    print('d(theta_dotdot) / d(omega1)', p.get_val('d_thetadotdot_omega1'))
    print('d(theta_dotdot) / d(omega2)', p.get_val('d_thetadotdot_omega2'))


def _run_stepsize_study():

    ### step_sizes = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12]
    step_sizes = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
    totals = {}

    # compute derivatives for each step size
    for step in step_sizes:
        totals[step] = __compute_derivatives(step)
    # END FOR

    # --- print ---
    outputs = ['d_vxdot_vx', 'd_vydot_vy', 'd_vydot_omega1', 'd_vydot_omega2', 'd_thetadotdot_omega1', 'd_thetadotdot_omega2']
    inputs = ['vy_ref', 'omega_ref', 'rotor_vert|theta_cp', 'rotor_vert|chord_cp']
    for in_name in inputs:
        print('\n\n---------------------------------------')
        print('   derivatives w.r.t.', in_name)
        print('---------------------------------------')
        for out_name in outputs:
            print('\n--- output: ', out_name, ' ---')
            for step in step_sizes:
                deriv = totals[step][(out_name, in_name)]
                if in_name in ['vy_ref', 'omega_ref']:
                    # just a scalar input.
                    print('{:.0e}   {:.10e}'.format(step, deriv[0][0]))
                else:
                    # NOTE: hardcoded for 5 inputs!
                    print('{:.0e}   {:.10e} {:.10e} {:.10e} {:.10e} {:.10e}'.format(step, deriv[0, 0], deriv[0, 1], deriv[0, 2], deriv[0, 3], deriv[0, 4]))
            # END FOR
        # END FOR
    # END FOR


def __compute_derivatives(fd_step_size):
    path = quadrotormodels.__path__[0]

    # --- define initial design ---
    rotor_r = 0.133  # rotor radius, m
    mass = 1.4  # vehicle MTOW, kg
    params_dict = {'S_body_ref' : 0.1365 / (0.5 * 1.225 * 0.3),  # body reference area, m**2, set so that (0.5 rho S CD) = 0.1365 (from Quan's book) with Cd=0.3
                   'rho_air': 1.225,  # air density
                   'mass' : mass,
                   'loc_rotors' : 0.225 / np.sqrt(2),   # m, distance from CG to rotor center in X configuration
                   'num_rotors' : 0,
                   'num_rotors_vert' : 4,  # for lift
                   'config' : 'multirotor',
                   }
    # vehicle moment of inertia
    params_dict['Icg'] = 0.0211   # kg-m^2
    # rotor design (optimized for minimum hover)
    num_cp = 20   # span-wise control points of twist and chord
    rotor_design = {'num_blades' : 2, 'Rtip' : rotor_r, 'num_cp' : num_cp, 'chord_cp' : np.array([0.04306510061311549, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.046549999999999994, 0.04396547538022499, 0.039847783680244324, 0.035794016939473317, 0.03163410897693981, 0.02718045311269029, 0.02209681742445619, 0.015477739025270299, 0.005]),
                    'theta_cp' : np.array([27.690181149634036, 27.455172875504946, 25.495688529069533, 23.304234714315378, 21.249495549081875, 19.328331022437958, 17.59816149341496, 16.072594662191236, 14.734347551290448, 13.558417660087255, 12.530640307054233, 11.733759479199708, 11.072322158736556, 10.518817772718993, 10.01215463997566, 9.54456137472585, 9.10900036533994, 8.690094710302423, 8.313328232992301, 7.8662041093808455]), 'cr75' : 0.025 / 0.133, 'omega_ref' : 471.94068266,
                    'xfoil_fnames' : path + '/xfoil-data/xf-n0012-il-100000.dat',
                    'BEMT_num_radius' : 40, 'BEMT_num_azimuth' : 4,
                    }
    params_dict['rotor_vert_design'] = rotor_design

    # -----------------------------------------------------
    # 1) solve for reference control input & linearize
    p = om.Problem()

    # design variables
    rotor_design = params_dict['rotor_vert_design']
    design_vars = om.IndepVarComp()
    design_vars.add_output('m', val=params_dict['mass'], units='kg')
    design_vars.add_output('Ipitch', val=params_dict['Icg'], units='kg*m**2')
    design_vars.add_output('rotor_vert|R', val=rotor_design['Rtip'], units='m',)   # rotor radius
    design_vars.add_output('rotor_vert|theta_cp', val=rotor_design['theta_cp'], units='deg')   # twist
    design_vars.add_output('rotor_vert|chord_cp', val=rotor_design['chord_cp'], units='m')   # chord
    design_vars.add_output('loc_rotors', val=params_dict['loc_rotors'], units='m')
    p.model.add_subsystem('design_vars', design_vars, promotes_outputs=['*'])

    # linearlization
    p.model.add_subsystem('linearize', LinearizeDynamics(params_dict=params_dict, fd_step_size=fd_step_size), promotes=['*'])
    p.model.set_input_defaults('vy_ref', 0.01, units='m/s')
    p.model.set_input_defaults('omega_ref', rotor_design['omega_ref'], units='rad/s')

    p.setup(check=False)
    p.run_model()

    # compute derivatives
    outputs = ['d_vxdot_vx', 'd_vydot_vy', 'd_vydot_omega1', 'd_vydot_omega2', 'd_thetadotdot_omega1', 'd_thetadotdot_omega2']
    inputs = ['vy_ref', 'omega_ref', 'rotor_vert|theta_cp', 'rotor_vert|chord_cp']
    totals = p.compute_totals(of=outputs, wrt=inputs)
    return totals


if __name__ == '__main__':
    _check_model()
    # _run_stepsize_study()