"""
Cart-pole open-loop co-design optimization runscript.

Requirements: openmdao 3.17.0, dymos 1.5.0, pyoptsparse with SNOPT.
              This code likely works with other openmdao/dymos versions, but I haven't checked.
"""

import time
import numpy as np
import openmdao.api as om
import matplotlib.pyplot as plt
import dymos as dm
from dymos.examples.plotting import plot_results

from dynamics import CartPoleDynamicsWithFriction


def main(flag_opt_design=False):
    """
    Run cart-pole optimization.
    set design_flag = True for open-loop co-design
        design_flag = False for trajectory optimization with fixed design
    """

    nsegs = 50  # number of segments

    # -------------------------------------
    # --- define initial design ---
    m_pole = 2.0
    m_cart = 7.5
    l_pole = 1.0

    # --- set initial and final states ---
    states_init = {'x': -1, 'x_dot': 0., 'theta': np.pi + 1, 'theta_dot': 0.}
    states_final = {'x': 1., 'x_dot': 0., 'theta': np.pi, 'theta_dot': 0.}
    duration = 30  # duration of simulation
    # -------------------------------------
    # NOTE: if you change the initial state, you may have to modify the cost scaling, objective scaling, and initial guess to converge optimization.

    # ----------------------------------------------------------
    # define vehicle design variables and constraints
    # ----------------------------------------------------------
    # --- cart-pole design parameters ---
    p = om.Problem()
    design_vars = p.model.add_subsystem('design_var', om.IndepVarComp(), promotes_outputs=['*'])
    design_vars.add_output('m_cart', val=m_cart, units='kg')
    design_vars.add_output('m_pole', val=m_pole, units='kg')
    design_vars.add_output('l_pole', val=l_pole, units='m')
    design_vars.add_output('d', val=0., units='N*s/m')   # set 0 friction

    if flag_opt_design:
        # add design variables and bounds
        p.model.add_design_var('m_cart', units='kg', lower=2.5, upper=7.5, ref=5.)
        p.model.add_design_var('m_pole', units='kg', lower=0.5, upper=2.0, ref=1.)
        ### p.model.add_design_var('l_pole', units='m', lower=1.0, upper=3.0, ref=2.)
        ### p.model.add_design_var('d', units='N*s/m', lower=0.7, upper=1.3, ref=1.)

        # constraint on total mass
        total_mass_comp = om.ExecComp('total_mass = m_pole + m_cart', units='kg')
        p.model.add_subsystem('total_mass', total_mass_comp, promotes=['*'])
        p.model.add_constraint('total_mass', lower=7., ref=7., units='kg')
    
    # --- instantiate trajectory and phase, setup transcription ---
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)
    phase = dm.Phase(transcription=dm.Radau(num_segments=nsegs, order=3, compressed=False, solve_segments=False),  # set solve_segments=True to do solver-based shooting
                     ode_class=CartPoleDynamicsWithFriction,
                     ode_init_kwargs={'g': 10, 'states_ref': states_final})
    traj.add_phase('phase', phase)

    # --- set state and control variables ---
    phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=duration, duration_ref=duration, units='s')   # fixed duration of 30 sec
    # declare state variables. You can also set lower/upper bounds and scalings here.
    phase.add_state('x', fix_initial=True, rate_source='x_dot', ref=1, defect_ref=1, units='m')
    phase.add_state('x_dot', fix_initial=True, rate_source='x_dotdot', ref=1, defect_ref=1, units='m/s')
    phase.add_state('theta', fix_initial=True, rate_source='theta_dot', ref=1, defect_ref=1, units='rad')
    phase.add_state('theta_dot', fix_initial=True, rate_source='theta_dotdot', ref=1, defect_ref=1, units='rad/s')
    phase.add_state('cost', fix_initial=True, rate_source='cost_rate', ref=10000, defect_ref=10000)
    # declare control inputs
    phase.add_control('f', fix_initial=False, ref=10, units='N')   # 60N = 1g for original system
    # add cart-pole parameters (set static_target=True because these params are not time-depencent)
    phase.add_parameter('m_cart', val=m_cart, units='kg', static_target=True)
    phase.add_parameter('m_pole', val=m_pole, units='kg', static_target=True)
    phase.add_parameter('l_pole', val=l_pole, units='m', static_target=True)
    phase.add_parameter('d', val=0., units='N*s/m', static_target=True)

    # --- set terminal constraint ---
    phase.add_boundary_constraint('x', loc='final', equals=states_final['x'], ref=1., units='m')
    phase.add_boundary_constraint('x_dot', loc='final', equals=states_final['x_dot'], ref=1., units='m/s')
    phase.add_boundary_constraint('theta', loc='final', equals=states_final['theta'], ref=1., units='rad')
    phase.add_boundary_constraint('theta_dot', loc='final', equals=states_final['theta_dot'], ref=1., units='rad/s')
    ### phase.add_boundary_constraint('f', loc='final', equals=0, ref=1., units='N')   # 0 force at the end

    # --- set objective function ---
    phase.add_objective('cost', loc='final', ref=10000)

    # linear solver is not necessary for collocation, but sometimes it helps
    ### p.model.linear_solver = om.DirectSolver()   # needed if solve_segments=True

    # --- relate design variables and trajectory ---
    p.model.connect('m_cart', 'traj.phase.parameters:m_cart')
    p.model.connect('m_pole', 'traj.phase.parameters:m_pole')
    p.model.connect('l_pole', 'traj.phase.parameters:l_pole')
    p.model.connect('d', 'traj.phase.parameters:d')

    # --- configure optimizer ---
    p.driver = om.pyOptSparseDriver()
    # SNOPT options
    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.options['print_results'] = False
    p.driver.opt_settings['Function precision'] = 1e-14
    p.driver.opt_settings['Major optimality tolerance'] = 1e-8
    p.driver.opt_settings['Major feasibility tolerance'] = 1e-8
    p.driver.opt_settings['Major iterations limit'] = 3000
    p.driver.opt_settings['Minor iterations limit'] = 100_000_000
    p.driver.opt_settings['Iterations limit'] = 10000
    p.driver.opt_settings['Hessian full memory'] = 1
    p.driver.opt_settings['Hessian frequency'] = 500
    p.driver.opt_settings['Print file'] = 'SNOPT_print_openloop_designopt=' + str(flag_opt_design) + '.out'
    p.driver.opt_settings['Summary file'] = 'SNOPT_summary_openloop_designopt=' + str(flag_opt_design) + '.out'
    p.driver.opt_settings['Verify level'] = -1

    # declare total derivative coloring to accelerate the UDE linear solves
    p.driver.declare_coloring()

    p.setup(check=True)
    # om.n2(p)

    # --- set initial guess ---
    # The initial condition of cart-pole (i.e., state values at time 0) is set here because we set `fix_initial=True` when declaring the states.
    p.set_val('traj.phase.t_initial', 0.0)  # set initial time to 0.
    # linearly interpolate the states between initial and terminal conditions.
    p.set_val('traj.phase.states:x', phase.interp(ys=[states_init['x'], states_final['x']], nodes='state_input'), units='m')
    p.set_val('traj.phase.states:x_dot', phase.interp(ys=[states_init['x_dot'], states_final['x_dot']], nodes='state_input'), units='m/s')
    p.set_val('traj.phase.states:theta', phase.interp(ys=[states_init['theta'], states_final['theta']], nodes='state_input'), units='rad')
    p.set_val('traj.phase.states:theta_dot', phase.interp(ys=[states_init['theta_dot'], states_final['theta_dot']], nodes='state_input'), units='rad/s')
    p.set_val('traj.phase.states:cost', phase.interp(xs=[0, 2, duration], ys=[0, 5000, 10000], nodes='state_input'))
    p.set_val('traj.phase.controls:f', phase.interp(xs=[0, 2, duration], ys=[-200, 10, 0], nodes='control_input'), units='N')
        
    # --- run optimization ---
    start_time = time.time()
    dm.run_problem(p, run_driver=True, simulate=True, simulate_kwargs={'method' : 'Radau', 'times_per_seg' : 10})
    # NOTE: Optimization solution (from collocation) and simulation (implicit time integration) typically doesn't match at the end
    #       because the system is unsteady near the terminal conditions: small numerical error (perturbation in state) amplifies.
    print('Optimization & simulation runtime:', time.time() - start_time, 's')

    # --- get results and plot ---
    # objective value
    obj = p.get_val('traj.phase.states:cost')[-1]
    print('objective value:', obj)
    print('cart mass:', p.get_val('m_cart', units='kg'))
    print('pole mass:', p.get_val('m_pole', units='kg'))
    print('pole length:', p.get_val('l_pole', units='m'))
    print('fric coeff:', p.get_val('d', units='N*s/m'))

    sim_sol = om.CaseReader('dymos_simulation.db').get_case('final')

    # plot time histories of x, x_dot, theta, theta_dot
    plot_results([('traj.phase.timeseries.time', 'traj.phase.timeseries.states:x', 'time (s)', 'x (m)'),
                  ('traj.phase.timeseries.time', 'traj.phase.timeseries.states:x_dot', 'time (s)', 'vx (m/s)'),
                  ('traj.phase.timeseries.time', 'traj.phase.timeseries.states:theta', 'time (s)', 'theta (rad)'),
                  ('traj.phase.timeseries.time', 'traj.phase.timeseries.states:theta_dot', 'time (s)', 'theta_dot (rad/s)'),
                  ('traj.phase.timeseries.time', 'traj.phase.timeseries.controls:f', 'time (s)', 'control (N)'),
                  ('traj.phase.timeseries.time', 'traj.phase.timeseries.states:cost', 'time (s)', 'cost')],
                 title='Cart-Pole Problem', p_sol=p, p_sim=sim_sol)
    import matplotlib.pyplot as plt
    plt.savefig('cartpole-open-loop-designopt=' + str(flag_opt_design) + '.pdf', bbox_inches='tight')

    # plot the animation of cart motion
    # x = p.get_val('traj.phase.timeseries.states:x', units='m')
    # theta = p.get_val('traj.phase.timeseries.states:theta', units='rad')
    # force = p.get_val('traj.phase.timeseries.controls:f', units='N')
    # npts = len(x)

    # # from animate_cartpole import animate_cartpole
    # # pole_len = p.get_val('l_pole', units='m')
    # # animate_cartpole(x.reshape(npts), theta.reshape(npts), force.reshape(npts), l=pole_len, interval=20, force_scaler=0.02, save_gif=True)
    # # plt.show()

    return p, sim_sol
  

if __name__ == '__main__':
    # 1) solve fixed-design problem. We don't optimize anything (except built-in LQR), but we still run optimizer to satisfy all defect constraints
    p1, _ = main(flag_opt_design=False)

    # 2) optimize cart-pole design variables
    p2, _ = main(flag_opt_design=True)

    # plot
    time1 = p1.get_val('traj.phase.timeseries.time', units='s')
    x1 = p1.get_val('traj.phase.timeseries.states:x', units='m')
    theta1 = p1.get_val('traj.phase.timeseries.states:theta', units='rad')
    cost1 = p1.get_val('traj.phase.timeseries.states:cost')

    time2 = p2.get_val('traj.phase.timeseries.time', units='s')
    x2 = p2.get_val('traj.phase.timeseries.states:x', units='m')
    theta2 = p2.get_val('traj.phase.timeseries.states:theta', units='rad')
    cost2 = p2.get_val('traj.phase.timeseries.states:cost') 

    fig, axs = plt.subplots(3, 1, figsize=(10, 6))
    axs[0].plot(time1, x1, color='k', alpha=0.5)   # baseline
    axs[0].plot(time2, x2, color='k', alpha=1.0)   # opt
    axs[0].set_ylabel('x, m')

    axs[1].plot(time1, theta1, color='b', alpha=0.5)
    axs[1].plot(time2, theta2, color='b', alpha=1.0)
    axs[1].set_ylabel('theta, rad')
    
    axs[2].plot(time1, cost1, color='r', alpha=0.5)
    axs[2].plot(time2, cost2, color='r', alpha=1.0)
    axs[2].set_ylabel('cost')
    axs[2].set_xlabel('time, s')
    axs[0].set_title('Open loop - baseline vs optimal design')

    plt.savefig('dymos-cartpole-open-loop.pdf', bbox_inches='tight')
