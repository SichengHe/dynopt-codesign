"""
Solves closed-loop co-design of cartpole using dymos (Radau collocation)
"""

import numpy as np
import matplotlib.pyplot as plt
import openmdao.api as om
import dymos as dm
from dymos.examples.plotting import plot_results

from dynamics import CartPoleLinearizedDynamics, CartPoleClosedLoopDynamics
from lqr import LQRClosedLoopSystem


def main(flag_opt_design=False):
    # flag_opt_desig: True to optimize plant design variables, False to fix the design
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

    # --- setup openmdao problem ---
    p = om.Problem()

    # set plant design variables
    design_vars = om.IndepVarComp()
    design_vars.add_output('m_cart', val=m_cart, units='kg')
    design_vars.add_output('m_pole', val=m_pole, units='kg')
    design_vars.add_output('l_pole', val=l_pole, units='m')
    design_vars.add_output('d', val=0., units='N*s/m')   # set 0 friction
    p.model.add_subsystem('design_vars', design_vars, promotes_outputs=['*'])

    if flag_opt_design:
        # set design variables and bounds
        p.model.add_design_var('m_pole', units='kg', lower=0.5, upper=2.0, ref=2.0)
        p.model.add_design_var('m_cart', units='kg', lower=2.5, upper=7.5, ref=7.5)
        ### p.model.add_design_var('l_pole', units='m', lower=1.0, upper=3.0, ref=2.)

        # constraint on total mass
        total_mass_comp = om.ExecComp('total_mass = m_pole + m_cart', units='kg')
        p.model.add_subsystem('total_mass', total_mass_comp, promotes=['*'])
        p.model.add_constraint('total_mass', lower=7., ref=7., units='kg')
    # END IF

    # set target state & rate of states
    target_states = om.IndepVarComp()
    target_states.add_output('x_tar', val=states_final['x'], units='m')
    target_states.add_output('x_dot_tar', val=states_final['x_dot'], units='m/s')
    target_states.add_output('v_dot_tar', val=0., units='m/s**2')
    target_states.add_output('theta_dot_tar', val=states_final['theta_dot'], units='rad/s')
    target_states.add_output('omega_dot_tar', val=0., units='rad/s**2')
    p.model.add_subsystem('target_states', target_states, promotes_outputs=['*'])

    # --- reference-state equation ---
    # solve for control and theta (although we know f=0 and theta=pi)
    ### p.model.add_subsystem('solve_ref_state', CartPoleRefState(), promotes_inputs=['m_cart', 'm_pole', 'l_pole', 'd', '*_tar'])
    # NOTE: this component is actually not necesarry for this case because we already know the reference states (=[1, 0, pi, 0]) and control(=0), which are independent of design variables
    
    # --- linearize system at the reference state ---
    p.model.add_subsystem('linearize', CartPoleLinearizedDynamics(), promotes_inputs=['m_cart', 'm_pole', 'l_pole', 'd'], promotes_outputs=['A', 'B'])
    # NOTE: reference state of [x, v, theta, omega] = [1, 0, pi, 0] is hardcoded for linearization. More generally, the reference state depends on design variables (but not in this example)

    # --- solve LQR for the optimal feedback gain K ---
    p.model.add_subsystem('LQR', LQRClosedLoopSystem(ns=4, nc=1, Q=np.eye(4), R=np.eye(1)), promotes_inputs=['*'], promotes_outputs=['K', 'A_cl'])

    # --- closed-loop simulation using direct transcription ---
    traj = p.model.add_subsystem('traj', dm.Trajectory())
    tx = dm.Radau(num_segments=nsegs, order=3, solve_segments=False, compressed=False)

    # NOTE: reference state hardcoded. Also, the identity for Q and R matrices are hardcoded to compute the time history of cost
    phase = dm.Phase(ode_class=CartPoleClosedLoopDynamics, transcription=tx, ode_init_kwargs={'g': 10, 'states_ref': states_final})
    phase = traj.add_phase('phase', phase)

    # --- set state and control variables ---
    phase.set_time_options(fix_initial=True, fix_duration=True, duration_val=duration, duration_ref=duration, units='s')   # fixed duration of 30 sec
    # declare state variables. You can also set lower/upper bounds and scalings here.
    phase.add_state('x', fix_initial=True, rate_source='x_dot', ref=1, defect_ref=1, units='m')
    phase.add_state('x_dot', fix_initial=True, rate_source='x_dotdot', ref=1, defect_ref=1, units='m/s')
    phase.add_state('theta', fix_initial=True, rate_source='theta_dot', ref=1, defect_ref=1, units='rad')
    phase.add_state('theta_dot', fix_initial=True, rate_source='theta_dotdot', ref=1, defect_ref=1, units='rad/s')
    phase.add_state('cost', fix_initial=True, rate_source='cost_rate', ref=1000, defect_ref=1000)
    
    # add cart-pole parameters (set static_target=True because these params are not time-depencent)
    phase.add_parameter('m_cart', val=m_cart, units='kg', static_target=True)
    phase.add_parameter('m_pole', val=m_pole, units='kg', static_target=True)
    phase.add_parameter('l_pole', val=l_pole, units='m', static_target=True)
    phase.add_parameter('d', val=0., units='N*s/m', static_target=True)
    p.model.connect('m_cart', 'traj.phase.parameters:m_cart')
    p.model.connect('m_pole', 'traj.phase.parameters:m_pole')
    p.model.connect('l_pole', 'traj.phase.parameters:l_pole')
    p.model.connect('d', 'traj.phase.parameters:d')

    # set controller parameters
    phase.add_parameter('K', val=np.zeros((1, 4)), static_target=True, include_timeseries=False)   # control feedback matrix, shape (n_control, n_states)
    p.model.connect('K', 'traj.phase.parameters:K')

    # set objective (quadratic cost function)
    phase.add_objective('cost', loc='final', ref=10)
    
    # --- set optimizer ---
    p.driver = om.pyOptSparseDriver()

    p.driver.options['optimizer'] = 'SNOPT'
    p.driver.options['print_results'] = False
    p.driver.opt_settings['Function precision'] = 1e-12
    p.driver.opt_settings['Major optimality tolerance'] = 1e-8
    p.driver.opt_settings['Major feasibility tolerance'] = 1e-8
    p.driver.opt_settings['Major iterations limit'] = 3000
    p.driver.opt_settings['Minor iterations limit'] = 100_000_000
    p.driver.opt_settings['Iterations limit'] = 100000
    p.driver.opt_settings['Hessian full memory'] = 1
    p.driver.opt_settings['Hessian frequency'] = 100
    p.driver.opt_settings['Nonderivative linesearch'] = 1
    p.driver.opt_settings['Print file'] = 'SNOPT_print_closedloop_designopt=' + str(flag_opt_design) + '.out'
    p.driver.opt_settings['Summary file'] = 'SNOPT_summary_closedloop_designopt=' + str(flag_opt_design) + '.out'
    # verify gradient
    p.driver.opt_settings['Verify level'] = -1

    p.driver.declare_coloring()
    p.model.linear_solver = om.DirectSolver()

    p.setup(check=True)
    om.n2(p, show_browser=False)

    # --- set initial guess & initial conditions ----
    p.set_val('traj.phase.t_initial', val=0.0)
    p.set_val('traj.phase.t_duration', val=duration)
    p.set_val('traj.phase.states:x', phase.interpolate(ys=[states_init['x'], states_final['x']], nodes='state_input'), units='m')
    p.set_val('traj.phase.states:theta', phase.interpolate(ys=[states_init['theta'], states_final['theta']], nodes='state_input'), units='rad')
    p.set_val('traj.phase.states:x_dot', phase.interpolate(ys=[states_init['x_dot'], states_final['x_dot']], nodes='state_input'), units='m/s')
    p.set_val('traj.phase.states:theta_dot', phase.interpolate(ys=[states_init['theta_dot'], states_final['theta_dot']], nodes='state_input'), units='rad/s')
    p.set_val('traj.phase.states:cost', phase.interpolate(ys=[0, 100], nodes='state_input'))

    dm.run_problem(p, run_driver=True, simulate=True, make_plots=False, simulate_kwargs={'method' : 'RK45', 'times_per_seg' : 1000})
    
    # om.n2(p, show_browser=False)

    # print('--- reference-state solutions ---')
    # print('theta_ref =', p.get_val('solve_ref_state.theta'))
    # print('f_ref=', p.get_val('solve_ref_state.f'))
    print('--- problem setup ---')
    print('optimize plant design =', flag_opt_design)
    print('--- controls ---')
    print('A =', p.get_val('A'))
    print('B =', p.get_val('B'))
    print('K =', p.get_val('K'))
    print('P =', p.get_val('P'))
    print('--- designs ---')
    print('m_cart:', list(p.get_val('m_cart', units='kg')), 'kg')
    print('m_pole:', list(p.get_val('m_pole', units='kg')), 'kg')
    print('l_pole:', list(p.get_val('l_pole', units='m')), 'm')
    print('--- objective ---')
    print('final LQR cost:', p.get_val('traj.phase.timeseries.states:cost')[-1, 0])

    # get simulation (explicit shooting) result
    sim = om.CaseReader('dymos_simulation.db').get_case('final')   # explicit simulation result

    plot_results([('traj.phase.timeseries.time', 'traj.phase.timeseries.states:x', 'time (s)', 'x (m)'),
                  ('traj.phase.timeseries.time', 'traj.phase.timeseries.states:theta', 'time (s)', 'theta (rad)'),
                  ('traj.phase.timeseries.time', 'traj.phase.timeseries.states:cost', 'time (s)', 'cost')],
                 title='Cart-pole', p_sol=p, p_sim=sim)
    plt.savefig('cartpole-closed-loop-designopt=' + str(flag_opt_design) + '.pdf', bbox_inches='tight')
    
    return p, sim


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
    axs[0].set_title('Closed loop - baseline vs optimal design')

    plt.savefig('dymos-cartpole-closed-loop.pdf', bbox_inches='tight')


    
    
