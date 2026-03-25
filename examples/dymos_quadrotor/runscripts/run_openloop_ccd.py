"""
Quadrotor open-loop trajectory optimization / co-design using Dymos
Minimum-objective recovery from perturbed state to equilibrium steady state.
"""

import os
import numpy as np
import pickle
import time
import openmdao.api as om
import dymos as dm
import quadrotormodels
from quadrotormodels import VTOLDynamicsGroup_MultiRotor_3DOF as VTOLDynamics
from quadrotormodels.plotters import plot_multiphase, plot_traj_details_multirotor

# TODO: reduce rotor blade num_cp to 10?
# TODO: change initial state?
# TODO: decide nsegs (number of 3rd order segments for trajectory discretization)

if __name__ == '__main__':
    start_time = time.time()

    path = quadrotormodels.__path__[0]
    flag_opt_design = True  # set True to optimize rotor blade design (i.e., CCD). False to optimize just trajectory while fixing the design.
    flag_load_init_guess = False  # If True, load initial fuess from the pkl file below. 
    init_traj_filename = 'results_multirotor_recovery_desOpt_False.pkl'   # str or None. File should be stored in <path>/runscripts_quadrotor_CCD/init_guess/
    nsegs = 20  # number of trajectory discritization segments

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

    # --- set initial and final states ---
    states_init = {'x': 5, 'y': -5, 'vx': 0.5, 'vy': 1, 'theta': 0, 'theta_vel': 0}   # theta [deg], theta_vel [deg/s]
    states_final = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0.01, 'theta': 0, 'theta_vel': 0}
    duration = 3  # guess of duration

    # -------------------------------------------------
    #  Define optimization problem
    # -------------------------------------------------
    p = om.Problem()

    # --- vehicle design variables ---
    design_vars = om.IndepVarComp()
    design_vars.add_output('m', val=params_dict['mass'], units='kg')
    design_vars.add_output('Ipitch', val=params_dict['Icg'], units='kg*m**2')
    design_vars.add_output('rotor_vert|R', val=rotor_design['Rtip'], units='m',)   # rotor radius
    design_vars.add_output('rotor_vert|theta_cp', val=rotor_design['theta_cp'], units='deg')   # twist
    design_vars.add_output('rotor_vert|chord_cp', val=rotor_design['chord_cp'], units='m')   # chord
    design_vars.add_output('loc_rotors', val=params_dict['loc_rotors'], units='m')
    p.model.add_subsystem('design_vars', design_vars, promotes_outputs=['*'])

    # --- define a trajectory ---
    
    tx = dm.GaussLobatto(num_segments=nsegs, order=3, solve_segments=False, compressed=False)

    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)
    phase = dm.Phase(transcription=tx, ode_class=VTOLDynamics, ode_init_kwargs={'params_dict': params_dict})
    traj.add_phase('phase', phase)
    phase.set_time_options(fix_initial=True, duration_bounds=(1, duration * 5), duration_ref=duration)
    # states (3 DoF)
    phase.add_state('x', fix_initial=True, rate_source='x_dot', ref=10, defect_ref=10, units='m')
    phase.add_state('y', fix_initial=True, rate_source='y_dot', ref=10, defect_ref=10, units='m')
    phase.add_state('theta', fix_initial=True, rate_source='theta_dot', lower=-45, upper=45, ref=10, defect_ref=10, units='deg')
    phase.add_state('vx', fix_initial=True, lower=-10, upper=10, rate_source='vx_dot', ref0=0, ref=5, defect_ref=100, units='m/s')
    phase.add_state('vy', fix_initial=True, lower=-10, upper=10, rate_source='vy_dot', ref0=0, ref=5, defect_ref=30, units='m/s')
    phase.add_state('theta_vel', fix_initial=True, lower=-np.pi, upper=np.pi, rate_source='theta_dotdot', ref0=0, ref=1, defect_ref=10, units='rad/s')
    phase.add_state('energy', fix_initial=True, rate_source='energy_dot', ref0=0, ref=duration * 100, defect_ref=duration * 100, units='W * s')
    # control variables
    phase.add_control('omega_vert_1', lower=10, upper=1000, ref0=300, ref=700, rate_continuity=False, units='rad/s')
    phase.add_control('omega_vert_2', lower=10, upper=1000, ref0=300, ref=700, rate_continuity=False, units='rad/s')

    # set vehicle design variable s as static parameters of the phase
    phase.add_parameter('m', val=mass, units='kg', static_target=True)
    phase.add_parameter('Ipitch', val=params_dict['Icg'], units='kg*m**2', static_target=True)
    phase.add_parameter('rotor_vert_1|R', val=rotor_design['Rtip'], units='m', static_target=True)
    phase.add_parameter('rotor_vert_1|theta_cp', val=rotor_design['theta_cp'], units='deg', static_target=True)
    phase.add_parameter('rotor_vert_1|chord_cp', val=rotor_design['chord_cp'], units='m', static_target=True)
    phase.add_parameter('rotor_vert_2|R', val=rotor_design['Rtip'], units='m', static_target=True)
    phase.add_parameter('rotor_vert_2|theta_cp', val=rotor_design['theta_cp'], units='deg', static_target=True)
    phase.add_parameter('rotor_vert_2|chord_cp', val=rotor_design['chord_cp'], units='m', static_target=True)
    phase.add_parameter('loc_rotors', val=params_dict['loc_rotors'], units='m', static_target=True)
    p.model.connect('m', 'traj.phase.parameters:m')
    p.model.connect('Ipitch', 'traj.phase.parameters:Ipitch')
    p.model.connect('rotor_vert|R', 'traj.phase.parameters:rotor_vert_1|R')
    p.model.connect('rotor_vert|theta_cp', 'traj.phase.parameters:rotor_vert_1|theta_cp')
    p.model.connect('rotor_vert|chord_cp', 'traj.phase.parameters:rotor_vert_1|chord_cp')
    p.model.connect('rotor_vert|R', 'traj.phase.parameters:rotor_vert_2|R')
    p.model.connect('rotor_vert|theta_cp', 'traj.phase.parameters:rotor_vert_2|theta_cp')
    p.model.connect('rotor_vert|chord_cp', 'traj.phase.parameters:rotor_vert_2|chord_cp')
    p.model.connect('loc_rotors', 'traj.phase.parameters:loc_rotors')

    # Boundary Constraints (steady equilibrium state at the end)
    phase.add_boundary_constraint('x', loc='final', equals=states_final['x'], ref=1, units='m')
    phase.add_boundary_constraint('y', loc='final', equals=states_final['y'], ref=1, units='m')
    phase.add_boundary_constraint('theta', loc='final', equals=states_final['theta'], ref=5, units='deg')
    phase.add_boundary_constraint('vx', loc='final', equals=states_final['vx'], ref=1, units='m/s')
    phase.add_boundary_constraint('vy', loc='final', equals=states_final['vy'], ref=1, units='m/s')  # small vertical climb speed to avoid numerical issue
    phase.add_boundary_constraint('theta_vel', loc='final', equals=states_final['theta_vel'], ref=1, units='rad/s')
    phase.add_boundary_constraint('vx_dot', loc='final', equals=0., ref=1, units='m/s**2')
    phase.add_boundary_constraint('vy_dot', loc='final', equals=0., ref=1, units='m/s**2')
    phase.add_boundary_constraint('theta_dotdot', loc='final', equals=0., ref=1, units='rad/s**2')

    # Path Constraints
    ### phase.add_path_constraint('vx', lower=-10, upper=10, ref=1, units='m/s')
    ### phase.add_path_constraint('vy', lower=-10, upper=10, ref=1, units='m/s')
    ### phase.add_path_constraint('power', lower=0., ref=1000)
    
    # log some variables
    phase.add_timeseries_output(['power', 'thrust_vert_1', 'thrust_vert_2'])

    # objective: minimize total energy
    phase.add_objective('energy', loc='final', ref=duration * 100)

    """
    # --- linear-quadratic control objective --- (not working)
    n_s = tx.grid_data.num_nodes   # state discretization
    n_c = tx.grid_data.num_nodes   # control discretization
    # compute error of vy. For all the other states, the equilibrium state is 0
    p.model.add_subsystem('LQ_vy_error', om.ExecComp('vy_error = vy - vy0', shape=(n_s,), units='m/s'), promotes_inputs=[('vy', 'traj.phase.timeseries.states:vy')])
    p.model.set_input_defaults('LQ_vy_error.vy0', val=states_final['vy'] * np.ones(n_s))
    # dot product of control and states
    dp_comp = om.DotProductComp(vec_size=1, length=n_c, a_name='omega_vert_1', b_name='omega_vert_1', c_name='omega_vert_1_sq', a_units='rad/s', b_units='rad/s', c_units='rad**2/s**2')
    dp_comp.add_product(vec_size=1, length=n_c, a_name='omega_vert_2', b_name='omega_vert_2', c_name='omega_vert_2_sq', a_units='rad/s', b_units='rad/s', c_units='rad**2/s**2')
    dp_comp.add_product(vec_size=1, length=n_s, a_name='x', b_name='x', c_name='x_sq', a_units='m', b_units='m', c_units='m**2')
    dp_comp.add_product(vec_size=1, length=n_s, a_name='y', b_name='y', c_name='y_sq', a_units='m', b_units='m', c_units='m**2')
    dp_comp.add_product(vec_size=1, length=n_s, a_name='theta', b_name='theta', c_name='theta_sq', a_units='deg', b_units='deg', c_units='deg**2')
    dp_comp.add_product(vec_size=1, length=n_s, a_name='vx', b_name='vx', c_name='vx_sq', a_units='m/s', b_units='m/s', c_units='m**2/s**2')
    dp_comp.add_product(vec_size=1, length=n_s, a_name='vy', b_name='vy', c_name='vy_sq', a_units='m/s', b_units='m/s', c_units='m**2/s**2')
    dp_comp.add_product(vec_size=1, length=n_s, a_name='theta_vel', b_name='theta_vel', c_name='theta_vel_sq', a_units='rad/s', b_units='rad/s', c_units='rad**2/s**2')
    p.model.add_subsystem('LQ_dot_products', dp_comp, promotes_outputs=['*'])
    p.model.connect('traj.phase.timeseries.controls:omega_vert_1', 'LQ_dot_products.omega_vert_1')
    p.model.connect('traj.phase.timeseries.controls:omega_vert_2', 'LQ_dot_products.omega_vert_2')
    p.model.connect('traj.phase.timeseries.states:x', 'LQ_dot_products.x')
    p.model.connect('traj.phase.timeseries.states:y', 'LQ_dot_products.y')
    p.model.connect('traj.phase.timeseries.states:theta', 'LQ_dot_products.theta')
    p.model.connect('traj.phase.timeseries.states:vx', 'LQ_dot_products.vx')
    p.model.connect('LQ_vy_error.vy_error', 'LQ_dot_products.vy')
    p.model.connect('traj.phase.timeseries.states:theta_vel', 'LQ_dot_products.theta_vel')

    # weighted sum for objective
    lq_weights = [1., 1., 100, 100, 100, 100, 100, 100]
    lq_sum_comp = om.AddSubtractComp('LQ_obj', input_names=['omega_vert_1_sq', 'omega_vert_2_sq', 'x_sq', 'y_sq', 'theta_sq', 'vx_sq', 'vy_sq', 'theta_vel_sq'], scaling_factors=lq_weights)
    p.model.add_subsystem('LQ_sum', lq_sum_comp, promotes=['*'])

    p.model.add_objective('LQ_obj', ref=1e6)
    # """

    # ---- add rotor blade design variables ----
    if flag_opt_design:
        p.model.add_design_var('rotor_vert|chord_cp', lower=0.005, upper=rotor_r * 0.35, ref=rotor_r * 0.35, units='m')
        p.model.add_design_var('rotor_vert|theta_cp', lower=1, upper=40, ref=20, units='deg')
    # END IF

    # -----------------------------------------
    # # Setup the driver
    p.driver = om.pyOptSparseDriver()
    optimizer = 'SNOPT'

    if optimizer == 'SNOPT':
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.options['print_results'] = True
        p.driver.opt_settings['Function precision'] = 1e-8
        p.driver.opt_settings['Major optimality tolerance'] = 1e-5
        p.driver.opt_settings['Major feasibility tolerance'] = 1e-5
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

    elif optimizer == 'IPOPT':
        p.driver.options['optimizer'] = 'IPOPT'
        p.driver.options['print_results'] = True
        p.driver.opt_settings['max_iter'] = 1000
        p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
        p.driver.opt_settings['recalc_y'] = 'yes'
        p.driver.opt_settings['print_level'] = 5
        p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
        ### p.driver.opt_settings['nlp_scaling_method'] = 'none'

        p.driver.opt_settings['tol'] = 1.0e-3
        p.driver.opt_settings['dual_inf_tol'] = 1e-5   # optimality
        p.driver.opt_settings['constr_viol_tol'] = 1e-5

        p.driver.opt_settings['hessian_approximation'] = 'limited-memory'
        p.driver.opt_settings['limited_memory_max_history'] = 100
        p.driver.opt_settings['limited_memory_max_skipping'] = 5
        ### p.driver.opt_settings['mu_init'] = 1e-4  # starting from small mu_init might help when initial guess is very good.
        # 'monotone' prevents large mu, which usually works better. 'adaptive' might be better for global search
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'

        # restortion phase
        p.driver.opt_settings['required_infeasibility_reduction'] = 0.99
        p.driver.opt_settings['max_resto_iter'] = 100

        p.driver.opt_settings['output_file'] = 'IPOPT.out'
    # END IF

    p.driver.declare_coloring(tol=1e-10)
    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True)

    om.n2(p, outfile="n2-traj-pre.html", show_browser=False)

    # --- set initial guess ---
    p.set_val('traj.phase.t_initial', 0.0)
    p.set_val('traj.phase.t_duration', duration)
    p.set_val('traj.phase.states:x', phase.interpolate(ys=[states_init['x'], states_final['x']], nodes='state_input'), units='m')
    p.set_val('traj.phase.states:y', phase.interpolate(ys=[states_init['y'], states_final['y']], nodes='state_input'), units='m')
    p.set_val('traj.phase.states:theta', phase.interpolate(ys=[states_init['theta'], states_final['theta']], nodes='state_input'), units='deg')
    p.set_val('traj.phase.states:vx', phase.interpolate(ys=[states_init['vx'], states_final['vx']], nodes='state_input'), units='m/s')
    p.set_val('traj.phase.states:vy', phase.interpolate(ys=[states_init['vy'], states_final['vy']], nodes='state_input'), units='m/s')
    p.set_val('traj.phase.states:theta_vel', phase.interpolate(ys=[states_init['theta_vel'], states_final['theta_vel']], nodes='state_input'), units='deg/s')
    p.set_val('traj.phase.states:energy', phase.interpolate(ys=[0, duration * 100], nodes='state_input'))   # assume avg. 1000 W
    omega_ref = rotor_design['omega_ref']
    p.set_val('traj.phase.controls:omega_vert_1', phase.interpolate(ys=np.array([omega_ref, omega_ref]), nodes='control_input'), units='rad/s')
    p.set_val('traj.phase.controls:omega_vert_2', phase.interpolate(ys=np.array([omega_ref, omega_ref]), nodes='control_input'), units='rad/s')

    # When optimizing design, load saved solution for fixed-design trajectory.
    if flag_load_init_guess:
        with open(path + 'runscripts_quadrotor_CCD/init_guess/' + init_traj_filename, mode='rb') as file:
            init_guess = pickle.load(file)
            print('--- loaded initial guess from', init_traj_filename, '---')
        time = init_guess['time']
        p.set_val('traj.phase.t_initial', 0.0)
        p.set_val('traj.phase.t_duration', time[-1])
        p.set_val('traj.phase.states:x', phase.interpolate(xs=time, ys=init_guess['x'], nodes='state_input'), units='m')
        p.set_val('traj.phase.states:y', phase.interpolate(xs=time, ys=init_guess['y'], nodes='state_input'), units='m')
        p.set_val('traj.phase.states:theta', phase.interpolate(xs=time, ys=init_guess['theta'], nodes='state_input'), units='deg')
        p.set_val('traj.phase.states:vx', phase.interpolate(xs=time, ys=init_guess['vx'], nodes='state_input'), units='m/s')
        p.set_val('traj.phase.states:vy', phase.interpolate(xs=time, ys=init_guess['vy'], nodes='state_input'), units='m/s')
        p.set_val('traj.phase.states:theta_vel', phase.interpolate(xs=time, ys=init_guess['theta_vel'], nodes='state_input'), units='rad/s')
        p.set_val('traj.phase.states:energy', phase.interpolate(xs=time, ys=init_guess['energy'], nodes='state_input'), units='W * s')   # assume avg. 1500W
        p.set_val('traj.phase.controls:omega_vert_1', phase.interpolate(xs=time, ys=init_guess['omega_vert_1'], nodes='control_input'), units='rad/s')
        p.set_val('traj.phase.controls:omega_vert_2', phase.interpolate(xs=time, ys=init_guess['omega_vert_2'], nodes='control_input'), units='rad/s')

        # load initial blade design and plot later
        radii_init = init_guess['design']['radii']
        chord_init = init_guess['design']['chord']
        twist_init = init_guess['design']['twist']

    # END IF

    dm.run_problem(p, run_driver=True, simulate=False, make_plots=False)
    om.n2(p, outfile="n2-traj-post.html", show_browser=False)

    # print results
    print('energy [kWs]:', p.get_val('traj.phase.states:energy', units='W*s')[-1] / 1000)
    print('duration [s]:', p.get_val('traj.phase.timeseries.time', units='s')[-1])

    print('design variables:')
    print('    rotor_vert|R =', p.get_val('rotor_vert|R', units='m'), 'm')
    print('    rotor_vert|chord_cp =', list(p.get_val('rotor_vert|chord_cp', units='m')), 'm')
    print('    rotor_vert|theta_cp =', list(p.get_val('rotor_vert|theta_cp', units='deg')), 'deg')
    print('    loc_rotors =', p.get_val('loc_rotors', units='m'), 'm')

    # plot trajectory
    if not os.path.exists('./traj-plots'):
        os.makedirs('traj-plots')

    phases = ['phase']
    plot_multiphase('traj', phases, [('states:x', 'states:y', 'Horizontal loc. [m]', 'Vertical loc. [m]')], 'XY path', p_sol=p)
    plot_multiphase('traj', phases, [('time', 'states:x', 'time [s]', 'x [m]'), ('time', 'states:y', 'time [s]', 'y [m]')], 'XY history', p_sol=p, connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'states:theta', 'time [s]', 'Body tilt angle [deg]')], 'Theta history', p_sol=p, y_units='deg', connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'states:vx', 'time [s]', 'Horizontal speed [m/s]'), ('time', 'states:vy', 'time [s]', 'Vertical speed [m/s]')], 'Velocity history', p_sol=p, connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'states:theta_vel', 'time [s]', 'Pitch angular vel [deg]')], 'Theta_vel history', p_sol=p, y_units='rad/s', connect_phase=False)

    plot_multiphase('traj', phases, [('time', 'controls:omega_vert_1', 'time [s]', 'Omega2 [rad/s]'), ('time', 'controls:omega_vert_2', 'time [s]', 'Omega2 [rad/s]')], 'Omega history', p_sol=p, connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'thrust_vert_1', 'time [s]', 'Thrust 1 [N]'), ('time', 'thrust_vert_2', 'time [s]', 'Thrust 2 [N]')], 'Thrust history', p_sol=p, connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'power', 'time [s]', 'Power [W]')], 'Power history', p_sol=p, connect_phase=False)
    plot_traj_details_multirotor('traj', phases, p, title='traj details', figsize=(8, 6))

    # plot rotor design
    radii = p.get_val('traj.phase.rhs_disc.propulsion_vert_1.rotor_spline_comp.radii', units='m')[0]
    chord = p.get_val('traj.phase.rhs_disc.propulsion_vert_1.rotor_spline_comp.chord', units='m')[0]
    twist = p.get_val('traj.phase.rhs_disc.propulsion_vert_1.rotor_spline_comp.theta', units='deg')[0]
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(radii / rotor_r, chord / rotor_r, color='C0', label='opt')
    axs[0].plot([0.15, 1], [0.35, 0.35], lw=1, color='gray')
    axs[0].set_ylabel('chord / R')
    axs[1].plot(radii / rotor_r, twist, color='C0', label='opt')
    axs[1].set_xlabel('r/R')
    axs[1].set_ylabel('twist, deg')
    if flag_load_init_guess and flag_opt_design:
        # also plot initial design (from hovering power minimization)
        axs[0].plot(radii_init / rotor_r, chord_init / rotor_r, color='C1', label='initial')
        axs[1].plot(radii_init / rotor_r, twist_init, color='C1', label='initial')
        plt.legend()
    # END IF
    plt.savefig('traj-plots/rotor-blade-design.pdf', bbox_inches='tight')

    # save solutions
    results = {}
    results['time'] = p.get_val('traj.phase.timeseries.time', units='s')
    results['x'] = p.get_val('traj.phase.timeseries.states:x', units='m')
    results['y'] = p.get_val('traj.phase.timeseries.states:y', units='m')
    results['theta'] = p.get_val('traj.phase.timeseries.states:theta', units='deg')
    results['vx'] = p.get_val('traj.phase.timeseries.states:vx', units='m/s')
    results['vy'] = p.get_val('traj.phase.timeseries.states:vy', units='m/s')
    results['theta_vel'] = p.get_val('traj.phase.timeseries.states:theta_vel', units='rad/s')
    results['energy'] = p.get_val('traj.phase.timeseries.states:energy', units='W * s')
    results['omega_vert_1'] = p.get_val('traj.phase.timeseries.controls:omega_vert_1', units='rad/s')
    results['omega_vert_2'] = p.get_val('traj.phase.timeseries.controls:omega_vert_2', units='rad/s')

    results_design = {}
    results_design['R'] = p.get_val('rotor_vert|R', units='m')
    results_design['chord_cp'] = p.get_val('rotor_vert|chord_cp', units='m')
    results_design['twist_cp'] = p.get_val('rotor_vert|theta_cp', units='deg')
    results_design['radii'] = radii
    results_design['chord'] = chord
    results_design['twist'] = twist
    results['design'] = results_design

    res_filename = 'results_multirotor_recovery_desOpt_' + str(flag_opt_design) + '.pkl'
    with open(res_filename, mode='wb') as file:
        pickle.dump(results, file)

    runtime = time.time() - start_time
    print('runtime:', runtime, 's')