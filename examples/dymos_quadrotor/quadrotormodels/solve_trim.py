import openmdao.api as om

from quadrotormodels import VTOLDynamicsGroup_MultiRotor_3DOF as VTOLDynamics


class QuadrotorTrim(om.Group):
    # solve control inputs for trim given target vy (climb speed) using Newton solver

    def initialize(self):
        self.options.declare('params_dict', types=dict)

    def setup(self):
        params_dict = self.options['params_dict']
        omega_ref = params_dict['rotor_vert_design']['omega_ref']

        # dynamics
        self.add_subsystem('dynamics', VTOLDynamics(num_nodes=1, params_dict=params_dict), promotes=['*'])

        # solve trim w.r.t. rotor omegas
        bal1 = om.BalanceComp('omega_vert_1', eq_units='m/s**2', lhs_name='vy_dot', rhs_name='zeros', units='rad/s', lower=omega_ref - 100, upper=omega_ref + 100, val=omega_ref, ref=omega_ref)
        bal2 = om.BalanceComp('omega_vert_2', eq_units='rad/s**2', lhs_name='theta_dotdot', rhs_name='zeros', units='rad/s', lower=omega_ref - 100, upper=omega_ref + 100, val=omega_ref, ref=omega_ref)
        self.add_subsystem('trim1', bal1, promotes=['omega_vert_1', 'vy_dot'])
        self.add_subsystem('trim2', bal2, promotes=['omega_vert_2', 'theta_dotdot'])
        self.set_input_defaults('trim1.zeros', val=0., units='m/s**2')
        self.set_input_defaults('trim2.zeros', val=0., units='rad/s**2')

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, debug_print=False, atol=1e-12, rtol=1e-12, maxiter=30)
        # self.nonlinear_solver = om.NonlinearBlockGS(debug_print=True)
        self.linear_solver = om.DirectSolver()
