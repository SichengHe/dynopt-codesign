"""
Setup OpenMDAO model for the quadrotor dynamics
"""

import sys
import os

# Add the folder to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '../dymos_quadrotor'))

import numpy as np
import openmdao.api as om
import quadrotormodels
from quadrotormodels import VTOLDynamicsGroup_MultiRotor_3DOF
 
def setup_quadrotor_openmdao(num_cp, solve_for_hover=False, optimize_blade_design=False):
    """

    Parameters
    ----------
    num_cp : int
        Number of control points for blade twist and chord design variables
    solve_for_hover : bool
        If True, setup a Newton solver to solve for steady hover w.r.t. RPM
        If False, this returns the rate of change of the state variables. Set False for dynamic simulation.
    optimize_blade_design : bool
        If True, optimize blade twist and chord for minimum hovering power.
        If False, just run analysis for a given design.

    Returns
    -------
    prob : OpenMDAO Problem
        OpenMDAO problem instance
    """

    # Following quadrotor parameters are based on "Quan Quan et al, Multicopter Design and Control Practice, 2018"
    # --- quadrotor's fixed design parameters ---
    params_dict = {
        'S_body_ref' : 0.1365 / (0.5 * 1.225 * 0.3),  # body reference area, m**2, set so that (0.5 rho S CD) = 0.1365 (from Quan's book) with Cd=0.3
        'rho_air': 1.225,  # air density, kg/m**3
        'mass' : 1.4,   # drone total weight, kg
        'Icg' : 0.0211,   # kg-m^2
        'loc_rotors' : 0.225 / np.sqrt(2),   # m, distance between drone CG and rotor center (thrust force acting point)
        'num_rotors' : 0,
        'num_rotors_vert' : 4,  # quadrotor
        'config' : 'multirotor',
    }
    
    # --- set initial rotor blade design ---
    chord_by_R_cp = np.linspace(0.35, 0.05, num_cp)    # blade normalized chord (chord-to-radius ratio)
    twist_cp = np.linspace(28.6242, 7.9522, num_cp)   # blade twist, deg
    rotor_design = {
        'num_blades' : 2,
        'Rtip' : 0.133,  # rotor radius, m
        'num_cp' : num_cp,
        'chord_by_R_cp' : chord_by_R_cp,
        'twist_cp' : twist_cp,
        'cr75' : 0.2,   # approximate chord-to-radius ratio at 75% radius
        'omega_ref' : 470,   # reference rotor speed, rad/s
        'xfoil_fnames' : quadrotormodels.__path__[0] + '/xfoil-data/xf-n0012-il-100000.dat',
        'BEMT_num_radius' : 40,   # number of blade elements
        'BEMT_num_azimuth' : 4,   # number of azimuthal discretization
    }
    params_dict['rotor_vert_design'] = rotor_design

    # --- setup OpenMDAO problem ---
    prob = om.Problem(reports=False)

    # design parameters and variables
    design_vars = om.IndepVarComp()
    design_vars.add_output('m', val=params_dict['mass'], units='kg')
    design_vars.add_output('Ipitch', val=params_dict['Icg'], units='kg*m**2')
    design_vars.add_output('rotor|R', val=rotor_design['Rtip'], units='m',)   # rotor radius
    design_vars.add_output('rotor|theta_cp', val=np.deg2rad(rotor_design['twist_cp']), units='rad')   # twist
    design_vars.add_output('rotor|chord_by_R_cp', val=rotor_design['chord_by_R_cp'], units=None)   # chord
    design_vars.add_output('loc_rotors', val=params_dict['loc_rotors'], units='m')
    prob.model.add_subsystem('design_vars', design_vars, promotes_outputs=['*'])

    # compute dimansional chord_cp
    chord_dim_comp = om.ExecComp(
        'chord_cp = chord_by_R_cp * radius',
        radius={'shape': (1,), 'units': 'm'},
        chord_by_R_cp={'shape': (num_cp,), 'units': None},
        chord_cp={'shape': (num_cp,), 'units': 'm'},
        has_diag_partials=True
    )
    prob.model.add_subsystem('chord_dimensional', chord_dim_comp, promotes_inputs=[('radius', 'rotor|R'), ('chord_by_R_cp', 'rotor|chord_by_R_cp')], promotes_outputs=[('chord_cp', 'rotor|chord_cp')])

    # quadrotor dynamics model (right hand side of ODE)
    prob.model.add_subsystem('dynamics', VTOLDynamicsGroup_MultiRotor_3DOF(num_nodes=1, params_dict=params_dict), promotes=['*'])
    prob.model.connect('rotor|R', ['rotor_vert_1|R', 'rotor_vert_2|R'])
    prob.model.connect('rotor|theta_cp', ['rotor_vert_1|theta_cp', 'rotor_vert_2|theta_cp'])
    prob.model.connect('rotor|chord_cp', ['rotor_vert_1|chord_cp', 'rotor_vert_2|chord_cp'])

    # solve for steady hover
    if solve_for_hover:
        # solve vy_dot = 0 w.r.t. rotor omega
        balance = om.BalanceComp()
        balance.add_balance('omega_hover', val=470., units='rad/s', eq_units='m/s**2', lhs_name='vy_dot', rhs_name='vy_dot_target', rhs_val=0.)
        prob.model.add_subsystem('steady_hover', balance, promotes_inputs=['vy_dot'])
        prob.model.connect('steady_hover.omega_hover', ['omega_vert_1', 'omega_vert_2'])   # same omega for all rotors

        # set steady hovering states
        prob.model.set_input_defaults('theta', val=0., units='deg')   # body pitch angle
        prob.model.set_input_defaults('vx', val=0., units='m/s')   # horizontal speed
        prob.model.set_input_defaults('vy', val=0., units='m/s')   # vertical speed
        prob.model.set_input_defaults('theta_vel', val=0., units='rad/s')   # body pitch rate
        
        # Newton solver
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, iprint=0, maxiter=20, atol=1e-10, rtol=1e-10, err_on_non_converge=True)
        prob.model.nonlinear_solver.linear_solver = om.DirectSolver()
    # END IF

    prob.model.linear_solver = om.DirectSolver()   # this may accelerate the total derivatives computation, maybe not

    # steady-state optimization
    if optimize_blade_design:
        if not solve_for_hover:
            raise ValueError("optimize_blade_design=True requires solve_for_hover=True")

        # objective: minimize power
        prob.model.add_objective('power', ref=100, units='W')

        # blade design variables
        prob.model.add_design_var('rotor|theta_cp', lower=0., upper=45., ref=10, units='deg')
        prob.model.add_design_var('rotor|chord_by_R_cp', lower=0.05, upper=0.35, ref=0.1)

        # optimizer
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'IPOPT'
        prob.driver.options['print_results'] = True
        prob.driver.opt_settings['tol'] = 1e-6
        prob.driver.opt_settings['constr_viol_tol'] = 1e-6
        prob.driver.opt_settings['max_iter'] = 100
        prob.driver.opt_settings['print_level'] = 5
    # END IF

    prob.setup(check=False)

    return prob


if __name__ == "__main__":

    prob = setup_quadrotor_openmdao(num_cp=10, solve_for_hover=True, optimize_blade_design=False)
    
    # set blade design
    twist_cp = np.array([28.624217078309165, 25.24644113988068, 20.757154874023897, 17.05283405965328, 14.168334169507002, 12.054782862617003, 10.690596691349743, 9.64011407359747, 8.72406502497316, 7.952156310627215])
    chord_by_R_cp = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.30955163, 0.24256796, 0.17357224, 0.04070716])
    prob.set_val('rotor|theta_cp', np.deg2rad(twist_cp), units='rad')
    prob.set_val('rotor|chord_by_R_cp', chord_by_R_cp, units=None)

    # run steady hover optimization
    prob.run_driver()
    # om.n2(prob)
    print('--- blade optimization for steady hover ---')
    print('hover power =', prob.get_val('power', units='W')[0], 'W')
    print('hover omega =', prob.get_val('steady_hover.omega_hover', units='rad/s')[0], 'rad/s')
    print('hover vy_dot =', prob.get_val('vy_dot', units='m/s**2')[0], '= 0 ?  m/s**2')
    print('twist cp =', list(prob.get_val('rotor|theta_cp', units='deg')), 'deg')
    print('chord_by_R cp =', list(prob.get_val('rotor|chord_by_R_cp', units=None)))