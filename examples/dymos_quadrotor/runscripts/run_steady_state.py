"""
steady-state rotor design optimization
minimize hovering power w.r.t. rotor blade design. No trajectory / controls
"""

import numpy as np
import openmdao.api as om

from quadrotormodels.solve_trim import QuadrotorTrim


def _update_blade_distribution(rotor_design_dict):

    # original arrays in dict
    num_cp_orig = len(rotor_design_dict['chord_cp'])
    num_cp_new = rotor_design_dict['num_cp']

    # build interpolation model
    r = rotor_design_dict['Rtip']
    r_orig = np.linspace(r * 0.15, r, num_cp_orig)
    r_new = np.linspace(r * 0.15, r, num_cp_new)
    rotor_design_dict['chord_cp'] = np.interp(r_new, r_orig, rotor_design_dict['chord_cp'], num_cp_new)
    rotor_design_dict['theta_cp'] = np.interp(r_new, r_orig, rotor_design_dict['theta_cp'], num_cp_new)

    return rotor_design_dict


if __name__ == '__main__':
    path = '/Users/shugo/rsrc/UAV_traj_design/'
    
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
    Icg = 0.0211   # kg-m^2
    params_dict['Icg'] = Icg
    # rotor design. theta_cp in [deg]. Ref_omega = typical omega for cruise or vertical climb
    num_cp = 10   # span-wise control points of twist and chord
    rotor_design = {'num_blades' : 2, 'Rtip' : rotor_r, 'num_cp' : num_cp, 'chord_cp' : np.array([0.06540099, 0.07402644, 0.05074617, 0.03327643, 0.02000162]),
                    'theta_cp' : np.array([30.0713701, 21.28427621, 15.99745591, 14.06969299, 10.15005394]), 'cr75' : 0.025 / 0.133, 'omega_ref' : 500,
                    'xfoil_fnames' : path + '/xfoil-data/xf-n0012-il-100000.dat',
                    'BEMT_num_radius' : 40, 'BEMT_num_azimuth' : 4,
                    }

    # number of spline control-point adjustment
    rotor_design = _update_blade_distribution(rotor_design)
    params_dict['rotor_vert_design'] = rotor_design

    # --- final (target/reference) states ---
    states_final = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0.0, 'theta': 0, 'theta_vel': 0}
    duration = 10  # duration of simulation

    # -------------------------------------
    # --- setup LQR co-design problem ---
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

    p.model.add_design_var('rotor_vert|theta_cp', lower=1, upper=40, ref=20, units='deg')
    p.model.add_design_var('rotor_vert|chord_cp', lower=0.005, upper=rotor_r * 0.35, ref=rotor_r * 0.35, units='m')

    # reference (trim) state = final steady state
    ref_states = om.IndepVarComp()
    ref_states.add_output('x_ref', val=states_final['x'], units='m')
    ref_states.add_output('y_ref', val=states_final['y'], units='m')
    ref_states.add_output('theta_ref', val=states_final['theta'], units='deg')
    ref_states.add_output('vx_ref', val=states_final['vx'], units='m/s')
    ref_states.add_output('vy_ref', val=states_final['vy'], units='m/s')
    ref_states.add_output('theta_vel_ref', val=states_final['theta_vel'], units='deg/s')
    p.model.add_subsystem('ref_states', ref_states, promotes_outputs=['*'])

    # solve trim equation
    p.model.add_subsystem('trim', QuadrotorTrim(params_dict=params_dict), promotes=['m', 'Ipitch', 'loc_rotors'])
    p.model.connect('rotor_vert|R', ['trim.rotor_vert_1|R', 'trim.rotor_vert_2|R'])
    p.model.connect('rotor_vert|theta_cp', ['trim.rotor_vert_1|theta_cp', 'trim.rotor_vert_2|theta_cp'])
    p.model.connect('rotor_vert|chord_cp', ['trim.rotor_vert_1|chord_cp', 'trim.rotor_vert_2|chord_cp'])
    for state_name in ['theta', 'vx', 'vy', 'theta_vel']:
        p.model.connect(state_name + '_ref', 'trim.' + state_name)

    # objective
    p.model.add_objective('trim.power', ref=100)

    # --- set optimizer ---
    p.driver = om.pyOptSparseDriver()
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.options['print_results'] = True
    p.driver.opt_settings['Function precision'] = 1e-8
    p.driver.opt_settings['Major optimality tolerance'] = 1e-6
    p.driver.opt_settings['Major feasibility tolerance'] = 1e-6
    p.driver.opt_settings['Major iterations limit'] = 3000
    p.driver.opt_settings['Minor iterations limit'] = 100_000_000
    p.driver.opt_settings['Iterations limit'] = 100000
    p.driver.opt_settings['Hessian full memory'] = 1
    p.driver.opt_settings['Hessian frequency'] = 100
    p.driver.opt_settings['Nonderivative linesearch'] = 1
    ### p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Print file'] = 'SNOPT_print.out'
    p.driver.opt_settings['Summary file'] = 'SNOPT_summary.out'
    # verify gradient
    p.driver.opt_settings['Verify level'] = -1

    ### p.driver.declare_coloring()
    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True)
    # om.n2(p, show_browser=False)
    # -------------------------------------
    
    p.run_driver()

    om.n2(p, show_browser=False)

    print('--- steady-state control inputs and power output ---')
    print('omega_vert_1 =', p.get_val('trim.omega_vert_1'), 'rad/s')
    print('omega_vert_2 =', p.get_val('trim.omega_vert_2'), 'rad/s')
    print('power:', p.get_val('trim.power'), 'W')
    print('--- rotor blade designs ---')
    print('theta_cp:', list(p.get_val('rotor_vert|theta_cp', units='deg')), 'deg')
    print('chord_cp:', list(p.get_val('rotor_vert|chord_cp', units='m')), 'm')

    # plot rotor design
    chord = p.get_val('trim.propulsion_vert_1.rotor_spline_comp.chord', units='m')
    twist = p.get_val('trim.propulsion_vert_1.rotor_spline_comp.theta', units='deg')

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(np.linspace(0.15, 1, len(chord[0])), chord[0] / rotor_r)
    axs[0].plot([0.15, 1], [0.35, 0.35], lw=1, color='gray')
    axs[0].set_ylabel('chord / R')

    axs[1].plot(np.linspace(0.15, 1, len(chord[0])), twist[0])
    axs[1].set_xlabel('r/R')
    axs[1].set_ylabel('twist, deg')

    plt.savefig('rotor-blade-design.pdf', bbox_inches='tight')
