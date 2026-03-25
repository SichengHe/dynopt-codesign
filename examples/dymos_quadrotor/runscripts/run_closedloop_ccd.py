"""
Closed-loop co-design using Dymos.
Linear feedback control, and the gain matrix is directly optimized (not LQR).
Optimization objective is the quadratic cost function (same as LQR)
"""

import numpy as np
import openmdao.api as om
import dymos as dm
import pickle

import quadrotormodels
from quadrotormodels import VTOLDynamicsGroup_MultiRotor_3DOF_ClosedLoop as QuadrotorODE
from quadrotormodels.solve_trim import QuadrotorTrim
from quadrotormodels.plotters import plot_multiphase

# TODO: reduce rotor blade num_cp to 10?
# TODO: tune Q and R matrix. Normalize state errors and control errors?
# TODO: change initial state?
# TODO: remove bounds on design variables.
# TODO: decide nsegs (number of 3rd order segments for trajectory discretization)

if __name__ == '__main__':
    path = quadrotormodels.__path__[0]
    flag_opt_design = True
    nsegs = 30
    
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

    # initial guess of the feedback matrix (obtained by fixed-design LQR in run_lqr_ccd.py)
    K_init = np.array([[-0.70710678, 0.70710678, 15.63944515, -1.65849962, 0.9066753, 8.47025458], [0.70710678, 0.70710678, -15.63944515, 1.65849962, 0.9066753, -8.47025458]])
    # ------------------------------

    # --- set initial and final states ---
    states_init = {'x': 5, 'y': -5, 'vx': 0.5, 'vy': 1, 'theta': 0, 'theta_vel': 0}   # theta [deg], theta_vel [deg/s]
    states_final = {'x': 0, 'y': 0, 'vx': 0, 'vy': 0.01, 'theta': 0, 'theta_vel': 0}
    duration = 30  # guess of duration

    # --- LQR settings ---
    # these will be used to compute the quadratic cost function, but we don't do LQR control.
    Q = np.eye(6)
    R = np.eye(2)
    Rinv = np.linalg.inv(R)

    # -------------------------------------
    # --- setup co-design problem ---
    p = om.Problem()

    # design variables
    rotor_design = params_dict['rotor_vert_design']
    design_vars = om.IndepVarComp()
    # quadrotor and blade design
    design_vars.add_output('m', val=params_dict['mass'], units='kg')
    design_vars.add_output('Ipitch', val=params_dict['Icg'], units='kg*m**2')
    design_vars.add_output('rotor_vert|R', val=rotor_design['Rtip'], units='m',)   # rotor radius
    design_vars.add_output('rotor_vert|theta_cp', val=rotor_design['theta_cp'], units='deg')   # twist
    design_vars.add_output('rotor_vert|chord_cp', val=rotor_design['chord_cp'], units='m')   # chord
    design_vars.add_output('loc_rotors', val=params_dict['loc_rotors'], units='m')
    # controller
    design_vars.add_output('K', val=K_init)   # feedback gain matrix, shape (2, 6)
    p.model.add_subsystem('design_vars', design_vars, promotes_outputs=['*'])

    if flag_opt_design:
        # NOTE: with CCD, theta tends to increases and chord tends to decreases
        # rotor blade design variables
        p.model.add_design_var('rotor_vert|theta_cp', lower=rotor_design['theta_cp'] - 5, upper=rotor_design['theta_cp'] + 5, ref=20, units='deg')
        chord_lb = np.maximum(rotor_design['chord_cp'] - 0.005, 0.005)
        chord_ub = np.minimum(rotor_design['chord_cp'] + 0.005, rotor_r * 0.35)
        p.model.add_design_var('rotor_vert|chord_cp', lower=chord_lb, upper=chord_ub, ref=0.03, units='m')
        # controller
        p.model.add_design_var('K', lower=K_init - 0.1 * np.abs(K_init), upper=K_init + 0.1 * np.abs(K_init))   # NOTE: element-wise scaling does not work here... needs to flatten K matrix into 1D ndarray

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

    # --- closed-loop trajectory ---
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    tx = dm.Radau(num_segments=nsegs, order=3, solve_segments=False, compressed=False)
    phase = dm.Phase(ode_class=QuadrotorODE, transcription=tx, ode_init_kwargs={'params_dict': params_dict, 'Q': Q, 'R': R})
    phase = traj.add_phase('phase', phase)

    phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=duration)
    # states (3 DoF)
    phase.add_state('x', fix_initial=True, rate_source='x_dot', ref=10, defect_ref=10, units='m')
    phase.add_state('y', fix_initial=True, rate_source='y_dot', ref=10, defect_ref=10, units='m')
    phase.add_state('theta', fix_initial=True, rate_source='theta_dot', ref=10, defect_ref=10, units='deg')
    phase.add_state('vx', fix_initial=True, lower=-10, upper=10, rate_source='vx_dot', ref0=0, ref=5, defect_ref=100, units='m/s')
    phase.add_state('vy', fix_initial=True, lower=-10, upper=10, rate_source='vy_dot', ref0=0, ref=5, defect_ref=30, units='m/s')
    phase.add_state('theta_vel', fix_initial=True, rate_source='theta_dotdot', ref0=0, ref=1, defect_ref=10, units='rad/s')
    phase.add_state('cost', fix_initial=True, rate_source='cost_rate', ref0=0, ref=10, defect_ref=10)
    # NOTE: relaxing (one of) the initial states may help optimization convergence. To do so, set fix_initial=False and add a boundary constraint at the initial state.
    # NOTE: no control variable is defined here because this is closed-loop simulation

    # set vehicle design variables
    phase.add_parameter('m', val=mass, units='kg', static_target=True)
    phase.add_parameter('Ipitch', val=params_dict['Icg'], units='kg*m**2', static_target=True)
    phase.add_parameter('rotor_vert_1|R', val=rotor_design['Rtip'], units='m', static_target=True)
    phase.add_parameter('rotor_vert_1|theta_cp', val=rotor_design['theta_cp'], units='deg', static_target=True, include_timeseries=False)
    phase.add_parameter('rotor_vert_1|chord_cp', val=rotor_design['chord_cp'], units='m', static_target=True, include_timeseries=False)
    phase.add_parameter('rotor_vert_2|R', val=rotor_design['Rtip'], units='m', static_target=True)
    phase.add_parameter('rotor_vert_2|theta_cp', val=rotor_design['theta_cp'], units='deg', static_target=True, include_timeseries=False)
    phase.add_parameter('rotor_vert_2|chord_cp', val=rotor_design['chord_cp'], units='m', static_target=True, include_timeseries=False)
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

    # set controller parameters
    phase.add_parameter('K', val=K_init, static_target=True, include_timeseries=False)   # control feedback matrix, shape (n_control, n_states)
    phase.add_parameter('omega_ref', val=params_dict['rotor_vert_design']['omega_ref'], units='rad/s', static_target=True)
    p.model.connect('K', 'traj.phase.parameters:K')
    p.model.connect('trim.omega_vert_1', 'traj.phase.parameters:omega_ref')   # NOTE: trim.omega_vert_1 = trim.omega_vert_2

    # set reference states for dymos (used for cost calculation)
    phase.add_parameter('x_ref', val=states_final['x'], units='m', static_target=True)
    phase.add_parameter('y_ref', val=states_final['y'], units='m', static_target=True)
    phase.add_parameter('theta_ref', val=states_final['theta'], units='deg', static_target=True)
    phase.add_parameter('vx_ref', val=states_final['vx'], units='m/s', static_target=True)
    phase.add_parameter('vy_ref', val=states_final['vy'], units='m/s', static_target=True)
    phase.add_parameter('theta_vel_ref', val=states_final['theta_vel'], units='deg/s', static_target=True)
    for state_name in ['x', 'y', 'theta', 'vx', 'vy', 'theta_vel']:
        p.model.connect(state_name + '_ref', 'traj.phase.parameters:' + state_name + '_ref')

    phase.add_timeseries_output(['power', 'thrust_vert_1', 'thrust_vert_2', 'omega_vert_1', 'omega_vert_2'])

    # (1) LQR objective
    phase.add_objective('cost', loc='final', ref=500)   # LQR objective
    
    # (2) LQR + static objective (in this case, power at trim condition)
    # obj_comp = om.ExecComp('obj = LQR_cost / 540.0933046890616 + power_trim / 99.84028721', shape=(1,), power_trim={'units': 'W'})   # NOTE: normalized by baseline (fixed-design LQR) LQR cost and hover power
    # p.model.add_subsystem('obj_comp', obj_comp)
    # p.model.connect('traj.phase.states:cost', 'obj_comp.LQR_cost', src_indices=[-1])
    # p.model.connect('trim.power', 'obj_comp.power_trim')
    # p.model.add_objective('obj_comp.obj')

    # --- set optimizer ---
    p.driver = om.pyOptSparseDriver()

    optimizer = 'SNOPT'

    if optimizer == 'SNOPT':
        p.driver.options['optimizer'] = 'SNOPT'
        p.driver.options['print_results'] = True
        p.driver.opt_settings['Function precision'] = 1e-9
        p.driver.opt_settings['Major optimality tolerance'] = 1e-4
        p.driver.opt_settings['Major feasibility tolerance'] = 1e-4
        p.driver.opt_settings['Major iterations limit'] = 100
        p.driver.opt_settings['Minor iterations limit'] = 100_000_000
        p.driver.opt_settings['Iterations limit'] = 100000
        p.driver.opt_settings['Hessian full memory'] = 1
        p.driver.opt_settings['Hessian frequency'] = 100
        p.driver.opt_settings['Nonderivative linesearch'] = 1
        ### p.driver.opt_settings['iSumm'] = 6
        p.driver.opt_settings['Print file'] = 'SNOPT_print_nseg' + str(nsegs) + '_desopt' + str(flag_opt_design) + '.txt'
        p.driver.opt_settings['Summary file'] = 'SNOPT_summary_nseg' + str(nsegs) + '_desopt' + str(flag_opt_design) + '.txt'
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
        p.driver.opt_settings['mu_init'] = 1e-4  # starting from small mu_init might help when initial guess is very good.
        # 'monotone' prevents large mu, which usually works better. 'adaptive' might be better for global search
        p.driver.opt_settings['mu_strategy'] = 'monotone'
        p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'

        # restortion phase
        p.driver.opt_settings['required_infeasibility_reduction'] = 0.99
        p.driver.opt_settings['max_resto_iter'] = 100

        p.driver.opt_settings['output_file'] = 'IPOPT_nseg' + str(nsegs) + '_desopt' + str(flag_opt_design) + '.txt'
    # END IF

    p.model.linear_solver = om.DirectSolver()
    p.driver.declare_coloring()

    # -------------------------------------
    p.setup(check=True)
    om.n2(p, outfile='n2_closedloop_pre.html', show_browser=False)
    
    # set initial conditions of trajectory
    p.set_val('traj.phase.t_initial', val=0.0)
    p.set_val('traj.phase.t_duration', val=duration)
    p.set_val('traj.phase.states:x', phase.interpolate(ys=[states_init['x'], states_final['x']], nodes='state_input'), units='m')
    p.set_val('traj.phase.states:y', phase.interpolate(ys=[states_init['y'], states_final['y']], nodes='state_input'), units='m')
    p.set_val('traj.phase.states:theta', phase.interpolate(ys=[states_init['theta'], states_final['theta']], nodes='state_input'), units='deg')
    p.set_val('traj.phase.states:vx', phase.interpolate(ys=[states_init['vx'], states_final['vx']], nodes='state_input'), units='m/s')
    p.set_val('traj.phase.states:vy', phase.interpolate(ys=[states_init['vy'], states_final['vy'] + 0.01], nodes='state_input'), units='m/s')
    p.set_val('traj.phase.states:theta_vel', phase.interpolate(ys=[states_init['theta_vel'], states_final['theta_vel']], nodes='state_input'), units='deg/s')
    p.set_val('traj.phase.states:cost', phase.interpolate(ys=[0, 100], nodes='state_input'), units='s')

    # run shooting (while fixing design) to get initial guess
    dm.run_problem(p, run_driver=False, simulate=True, make_plots=True, simulation_record_file='dymos_simulation.db')
    p_sim = om.CaseReader('dymos_simulation.db').get_case('final')

    # --- setup recorder ---
    recorder = om.SqliteRecorder('opt_log_closedloop.sql')
    p.driver.add_recorder(recorder)
    control_design = ['K']
    rotor_design = ['rotor_vert|theta_cp', 'rotor_vert|chord_cp', 'trim.dynamics.propulsion_vert_1.BEMT.radii', 'trim.dynamics.propulsion_vert_1.BEMT.chord', 'trim.dynamics.propulsion_vert_1.BEMT.theta']
    traj_timeseries = ['traj.phase.timeseries.*']
    steady_state_sol = ['trim.omega_vert_1', 'trim.omega_vert_2', 'trim.power_for_lift_1', 'trim.power_for_lift_2', 'trim.thrust_vert_1', 'trim.thrust_vert_2']
    p.driver.recording_options['includes'] = control_design + rotor_design + traj_timeseries + steady_state_sol
    p.driver.recording_options['record_objectives'] = True
    p.driver.recording_options['record_constraints'] = False
    p.driver.recording_options['record_desvars'] = False
    p.setup(check=True)

    # set the shooting solution as initial guess
    times = p_sim['traj.phase.timeseries.time']
    p.set_val('traj.phase.t_initial', val=0.0)
    p.set_val('traj.phase.t_duration', val=duration)
    p.set_val('traj.phase.states:x', phase.interpolate(xs=times, ys=p_sim['traj.phase.timeseries.states:x'], nodes='state_input'), units='m')
    p.set_val('traj.phase.states:y', phase.interpolate(xs=times, ys=p_sim['traj.phase.timeseries.states:y'], nodes='state_input'), units='m')
    p.set_val('traj.phase.states:theta', phase.interpolate(xs=times, ys=p_sim['traj.phase.timeseries.states:theta'], nodes='state_input'), units='deg')
    p.set_val('traj.phase.states:vx', phase.interpolate(xs=times, ys=p_sim['traj.phase.timeseries.states:vx'], nodes='state_input'), units='m/s')
    p.set_val('traj.phase.states:vy', phase.interpolate(xs=times, ys=p_sim['traj.phase.timeseries.states:vy'], nodes='state_input'), units='m/s')
    p.set_val('traj.phase.states:theta_vel', phase.interpolate(xs=times, ys=p_sim['traj.phase.timeseries.states:theta_vel'], nodes='state_input'), units='rad/s')
    p.set_val('traj.phase.states:cost', phase.interpolate(xs=times, ys=p_sim['traj.phase.timeseries.states:cost'], nodes='state_input'), units='s')

    # run closed-loop collocation optimization
    # p.check_partials(compact_print=True)
    p.run_driver()
    om.n2(p, outfile='n2_closedloop_aft.html', show_browser=False)

    # --- compute energy consumption ---
    time = p.get_val('traj.phase.timeseries.time', units='s')
    power = p.get_val('traj.phase.timeseries.power', units='W')
    time = time.reshape(len(time))
    power = power.reshape(len(power))
    energy1 = np.trapz(power, time)

    # --- print results ---
    print('--- Results ---')
    print('omega at steady state =', p.get_val('trim.omega_vert_1', units='rad/s')[0], p.get_val('trim.omega_vert_2', units='rad/s')[0])
    
    print('power at steady state:', p.get_val('trim.power', units='W'), 'W')
    print('final LQR cost:', p.get_val('traj.phase.timeseries.input_values:states:cost')[-1, 0])
    print('final energy:', energy1, 'W*s (by trapez integration)')
    print('--- controller design ---')
    print('K =', p.get_val('K'))
    print('--- rotor design ---')
    print('theta_cp:', list(p.get_val('rotor_vert|theta_cp', units='deg')), 'deg')
    print('chord_cp:', list(p.get_val('rotor_vert|chord_cp', units='m')), 'm')
    print()
    print('radii:', list(p.get_val('trim.propulsion_vert_1.rotor_spline_comp.radii', units='m')), 'm')
    print('chord:', list(p.get_val('trim.propulsion_vert_1.rotor_spline_comp.chord', units='m')), 'm')
    print('twist:', list(p.get_val('trim.propulsion_vert_1.rotor_spline_comp.theta', units='deg')), 'deg')
    
    # --- plot results ---
    phases = ['phase']
    plot_multiphase('traj', phases, [('states:x', 'states:y', 'Horizontal loc. [m]', 'Vertical loc. [m]')], 'XY path', p_sol=p)
    plot_multiphase('traj', phases, [('time', 'states:x', 'time [s]', 'x [m]'), ('time', 'states:y', 'time [s]', 'y [m]')], 'XY history', p_sol=p, connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'states:theta', 'time [s]', 'Body tilt angle [deg]')], 'Theta history', p_sol=p, y_units='deg', connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'states:vx', 'time [s]', 'Horizontal speed [m/s]'), ('time', 'states:vy', 'time [s]', 'Vertical speed [m/s]')], 'Velocity history', p_sol=p, connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'states:theta_vel', 'time [s]', 'Pitch angular vel [rad/s]')], 'Theta_vel history', p_sol=p, y_units='rad/s', connect_phase=False)
    # plot_multiphase('traj', phases, [('time', 'omega_vert_1', 'time [s]', 'Omega2 [rad/s]'), ('time', 'omega_vert_2', 'time [s]', 'Omega2 [rad/s]')], 'Omega history', p_sol=p, connect_phase=False)
    # plot_multiphase('traj', phases, [('time', 'thrust_vert_1', 'time [s]', 'Thrust 1 [N]'), ('time', 'thrust_vert_2', 'time [s]', 'Thrust 2 [N]')], 'Thrust history', p_sol=p, connect_phase=False)
    # plot_multiphase('traj', phases, [('time', 'power', 'time [s]', 'Power [W]')], 'Power history', p_sol=p, connect_phase=False)
    plot_multiphase('traj', phases, [('time', 'states:cost', 'time [s]', 'Cost')], 'Cost history', p_sol=p, connect_phase=False)
    # plot_traj_details_multirotor('traj', phases, p, title='traj details', figsize=(8, 6))

    # save results
    phase_name = 'phase'
    res_phase = {}
    res_phase['time'] = p.get_val('traj.' + phase_name + '.timeseries.time', units='s')
    res_phase['x'] = p.get_val('traj.' + phase_name + '.timeseries.states:x', units='m')
    res_phase['y'] = p.get_val('traj.' + phase_name + '.timeseries.states:y', units='m')
    res_phase['vx'] = p.get_val('traj.' + phase_name + '.timeseries.states:vx', units='m/s')
    res_phase['vy'] = p.get_val('traj.' + phase_name + '.timeseries.states:vy', units='m/s')
    res_phase['theta'] = p.get_val('traj.' + phase_name + '.timeseries.states:theta', units='deg')
    res_phase['theta_vel'] = p.get_val('traj.' + phase_name + '.timeseries.states:theta_vel', units='rad/s')
    res_phase['cost'] = p.get_val('traj.' + phase_name + '.timeseries.states:cost')

    res_phase['power'] = p.get_val('traj.' + phase_name + '.timeseries.power', units='W')
    res_phase['thrust_vert_1'] = p.get_val('traj.' + phase_name + '.timeseries.thrust_vert_1', units='N')
    res_phase['thrust_vert_2'] = p.get_val('traj.' + phase_name + '.timeseries.thrust_vert_2', units='N')
    res_phase['omega_vert_1'] = p.get_val('traj.' + phase_name + '.timeseries.omega_vert_1', units='rad/s')
    res_phase['omega_vert_2'] = p.get_val('traj.' + phase_name + '.timeseries.omega_vert_2', units='rad/s')
    
    res_rotor = {}
    res_rotor['chord_cp'] = p.get_val('rotor_vert|chord_cp', units='m')
    res_rotor['theta_cp'] = p.get_val('rotor_vert|theta_cp', units='deg')
    res_rotor['radii'] = p.get_val('trim.propulsion_vert_1.rotor_spline_comp.radii', units='m')
    res_rotor['chord'] = p.get_val('trim.propulsion_vert_1.rotor_spline_comp.chord', units='m')
    res_rotor['theta'] = p.get_val('trim.propulsion_vert_1.rotor_spline_comp.theta', units='deg')

    res_control = {}
    res_control['K'] = p.get_val('K')
    res_control['trim_power'] = p.get_val('trim.power', units='W')

    results = {}
    results['phase'] = res_phase
    results['rotor'] = res_rotor
    results['control'] = res_control

    if flag_opt_design:
        res_filename = 'results_codesign.pkl'
    else:
        res_filename = 'results_fixed_design.pkl'
    
    with open(res_filename, mode='wb') as file:
        pickle.dump(results, file)
