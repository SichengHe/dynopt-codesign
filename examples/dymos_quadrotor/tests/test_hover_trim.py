import unittest
import numpy as np
import openmdao.api as om
import quadrotormodels
from quadrotormodels.solve_trim import QuadrotorTrim

"""
Test case: find control inputs for steady hover.
This tests the installation of quadrotormodels, OpenMDAO, CCBlade.jl, and its wrappers. This does not test any optimization (pyoptsparse) and trajectory (dymos)
"""

class TestCruiseTrim(unittest.TestCase):

    def test_trim_analysis(self):
        
        path = quadrotormodels.__path__[0]

        # --- UAV design parameters ---
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

        # --- set reference steady states ---
        states_ref = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0.01, 'theta': 0, 'theta_vel': 0}   # avoid vy=0

        # -----------------------------------------------------
        # solve for reference control input
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

        # dynamics
        p.model.add_subsystem('trim', QuadrotorTrim(params_dict=params_dict), promotes=['*'])
        p.model.connect('rotor_vert|R', ['rotor_vert_1|R', 'rotor_vert_2|R'])
        p.model.connect('rotor_vert|theta_cp', ['rotor_vert_1|theta_cp', 'rotor_vert_2|theta_cp'])
        p.model.connect('rotor_vert|chord_cp', ['rotor_vert_1|chord_cp', 'rotor_vert_2|chord_cp'])

        p.setup(check=True)
        # om.n2(p)

        # set reference steady states
        p.set_val('vx', states_ref['vx'], units='m/s')  # should be 0
        p.set_val('vy', states_ref['vy'], units='m/s')
        p.set_val('theta', states_ref['theta'], units='deg')  # should be 0
        p.set_val('theta_vel', states_ref['theta_vel'], units='deg/s')  # should be 0

        p.run_model()

        omega1 = p.get_val('omega_vert_1', units='rad/s')
        omega2 = p.get_val('omega_vert_2', units='rad/s')

        np.testing.assert_allclose(omega1, 472.003907, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(omega2, 472.003907, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()