import numpy as np
import openmdao.api as om

# for CCblade
from openmdao.utils.spline_distributions import cell_centered
from ccblade_openmdao_examples.ccblade_openmdao_component_nonper import juliamodule
from omjlcomps import JuliaImplicitComp

"""
BEMT-related components / groups for eVTOL dynamics model
"""

class MultiRotorGroup(om.Group):
    """
    Computes the power required by multiple rotors, given omega (RPM) as control input
    Inputs: omega, v_pal, v_normal, n_rotor, Rtip, chord_cp, theta_cp
    Outputs: thrust, power, drag (total of multiple rotors). Drag is an in-plane aero force in skewed inflow direction (drag=0 if inflow is normal)
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('num_rotor', types=int)  # number of rotors
        self.options.declare('rotor_design_dict', types=dict)
        self.options.declare('rho', default=1.2)   # air density
        self.options.declare('thrust_loss_factor', default=1.0)   # thrust_installed / thrust_isolated. set 0.97... to account for boom-lifter interaction loss

    def setup(self):
        nn = self.options['num_nodes']
        n_rotors = self.options['num_rotor']
        rotor_design_dict = self.options['rotor_design_dict']
        rho_air = self.options['rho']

        # --- avoid 0 or negative inflow ---
        # if normal inflow is <0.001, set it to 0.001. Negative inflow is not physically valid for BEMT (momentum theory), and can be approximated by hover or near-0 positive inflow (because the magnitude of negative inflow is small eigher way)
        self.add_subsystem('clip_inflow', InflowCorrection(num_nodes=nn, vel_eps=0.001), promotes_inputs=[('v_in', 'v_normal')], promotes_outputs=[('v_out', 'v_normal_clip')])

        # --- spline component for chord and twist distribution ---
        num_cp = rotor_design_dict['num_cp']
        self.add_subsystem('Rhub_comp', om.ExecComp('Rhub = 0.15 * Rtip', units='m'), promotes=['*'])
        self.add_subsystem('radius_discritize', RotorRadiusDiscretization(num_cp=num_cp), promotes_inputs=['Rhub', 'Rtip'], promotes_outputs=['radii_cp'])

        num_radial = rotor_design_dict['BEMT_num_radius']   # number of radius-wise elements for BEMT analysis
        x_cp = np.linspace(0.0, 1.0, num_cp)
        x_interp = cell_centered(num_radial, 0.0, 1.0)
        if num_cp < 5:
            spline_comp = om.SplineComp(method="slinear", x_cp_val=x_cp, x_interp_val=x_interp)
        else:
            spline_comp = om.SplineComp(method="akima", interp_options={'delta_x': 0.05, 'eps': 1e-10}, x_cp_val=x_cp, x_interp_val=x_interp)   # akima requires num_cp > 5
        # END IF
        spline_comp.add_spline(y_cp_name="radii_cp", y_interp_name="radii", y_units="m")
        spline_comp.add_spline(y_cp_name="chord_cp", y_interp_name="chord", y_units="m")
        spline_comp.add_spline(y_cp_name="theta_cp", y_interp_name="theta", y_units="rad")
        self.add_subsystem("rotor_spline_comp", spline_comp, promotes_inputs=["radii_cp", "chord_cp", "theta_cp"])

        # --- BEMT component ---
        af_fname = rotor_design_dict['xfoil_fnames']
        # NOTE: Akima is more accurate and can achieve tight convergence because it's smooth, but linear interp is not smooth.
        bemt_comp = JuliaImplicitComp(jlcomp=juliamodule.BEMTRotorCACompSideFlow(
            af_fname=af_fname, cr75=rotor_design_dict['cr75'], Re_exp=0.6,
            num_operating_points=nn, num_blades=rotor_design_dict['num_blades'],
            num_radial=num_radial, num_azimuth=rotor_design_dict['BEMT_num_azimuth'], rho=rho_air, mu=rho_air * 1.461e-5, speedofsound=340.297))
        # NOTE: set num_azimuth=1 to disable the sideflow effect.    # number of azimuth discretization for BEMT analysis

        self.add_subsystem('BEMT', bemt_comp, promotes_inputs=[('v', 'v_normal_clip'), 'v_pal', 'omega', 'Rtip', 'Rhub'], promotes_outputs=[('thrust', 'thrust_each'), ('drag', 'drag_each')])
        self.connect('rotor_spline_comp.radii', 'BEMT.radii')
        self.connect('rotor_spline_comp.chord', 'BEMT.chord')
        self.connect('rotor_spline_comp.theta', 'BEMT.theta')
        # set initial values so that BEMT works with first Newton iteration
        self.set_input_defaults('Rtip', val=rotor_design_dict['Rtip'], units='m')
        self.set_input_defaults('Rhub', val=rotor_design_dict['Rtip'], units='m')
        self.set_input_defaults('BEMT.radii', val=np.linspace(0.15, 1.0) * rotor_design_dict['Rtip'], units='m')
        self.set_input_defaults('BEMT.chord', val=0.05, units='m')
        self.set_input_defaults('BEMT.theta', val=10., units='deg')

        # set 0 pitch
        self.set_input_defaults('BEMT.pitch', np.zeros(nn))

        # compute power = torque * omega
        power_comp = om.ExecComp('power_each = torque * omega',
                                 power_each={'shape': (nn,), 'units': 'W'},
                                 torque={'shape': (nn,), 'units': 'N*m'},
                                 omega={'shape': (nn,), 'units': 'rad/s'},
                                 has_diag_partials=True)
        self.add_subsystem('torque_to_power', power_comp, promotes_inputs=['omega'], promotes_outputs=['power_each'])
        self.connect('BEMT.torque', 'torque_to_power.torque')

        # sum multiple rotors. also apply motor power efficiency (0.95) and thrust loss due to body-prop interaction
        multi_rotor_comp = om.ExecComp(['power_total = power_each * num_rotors / 0.95', 'thrust_total = thrust_each * num_rotors / thrust_loss_factor', 'drag_total = drag_each * num_rotors'],
                                       power_total={'shape': (nn,), 'units': 'W'},
                                       power_each={'shape': (nn,), 'units': 'W'},
                                       thrust_total={'shape': (nn,), 'units': 'N'},
                                       thrust_each={'shape': (nn,), 'units': 'N'},
                                       drag_total={'val': np.zeros(nn,), 'units': 'N'},
                                       drag_each={'val': np.zeros(nn,), 'units': 'N'},
                                       num_rotors={'val': n_rotors},
                                       thrust_loss_factor={'val': self.options['thrust_loss_factor']},
                                       has_diag_partials=True)
        self.add_subsystem('sum_rotors', multi_rotor_comp, promotes_inputs=['power_each', 'thrust_each', 'drag_each'], promotes_outputs=[('power_total', 'power'), ('thrust_total', 'thrust'), ('drag_total', 'drag')])

        # add linear solver. For nonlinear solver, we use CCBlade's internal solver.
        self.linear_solver = om.DirectSolver()


class RotorRadiusDiscretization(om.ExplicitComponent):
    # util class for BEMT. Given Rtip and Rhub, return radii_cp

    def initialize(self):
        self.options.declare('num_cp', default=2, desc='number of radius-wise control points. Must be consistent with len(theta_cp) etc')

    def setup(self):
        nr = self.options['num_cp']
        self.add_input('Rtip', val=2.0, units='m', desc='tip radius')
        self.add_input('Rhub', val=1.0, units='m', desc='hub radius')
        self.add_output('radii_cp', shape=(nr,), units='m', desc='radius discretization')
        self.declare_partials('radii_cp', ['*'], rows=np.arange(nr), cols=np.zeros(nr))

    def compute(self, inputs, outputs):
        nr = self.options['num_cp']
        outputs['radii_cp'] = np.linspace(inputs['Rhub'], inputs['Rtip'], nr)

    def compute_partials(self, inputs, partials):
        nr = self.options['num_cp']
        weight = np.linspace(0., 1., nr)
        partials['radii_cp', 'Rtip'] = weight
        partials['radii_cp', 'Rhub'] = 1 - weight


class ClipVelocity(om.ExplicitComponent):
    # clipping near-zero velocity to avoid zero inflow velocity in BEMT

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('vel_eps', default=1e-4, desc='velocity clipping tolerance')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('v_in', shape=(nn,), units='m/s', desc='input velocity')
        self.add_output('v_out', shape=(nn,), units='m/s', desc='output clippted velocity')
        self.declare_partials('v_out', 'v_in', rows=np.arange(nn), cols=np.arange(nn), val=np.ones(nn))
        # derivative is not 1 near v_in = 0, but we ignore that

    def compute(self, inputs, outputs):
        v_in = inputs['v_in']
        v_out = v_in.copy()
        eps = self.options['vel_eps']
        mask1 = np.logical_and(v_in >= 0., v_in <= eps)
        mask2 = np.logical_and(v_in < 0., v_in >= -eps)
        v_out[mask1] = eps
        v_out[mask2] = -eps
        outputs['v_out'] = v_out


class InflowCorrection(om.Group):
    """
    Override negative inflow velocity with near-zero positive velocity.
    Inputs: v_in
    Outputs: v_out
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('vel_eps', default=1e-4, desc='velocity correction threshold')

    def setup(self):
        nn = self.options['num_nodes']
        vel_eps = self.options['vel_eps']

        indep = self.add_subsystem('zero_vel', om.IndepVarComp(), promotes_outputs=['*'])
        indep.add_output('v0', val=vel_eps * np.ones(nn), units='m/s', desc='zero inflow velocity')

        # softmax between v_normal and v0
        self.add_subsystem('softmax', SoftMaximum(num_nodes=nn, alpha=50., units='m/s'), promotes_inputs=[('x2', 'v_in')], promotes_outputs=[('softmax', 'v_out')])
        self.connect('v0', 'softmax.x1')
        

class SoftMaximum(om.ExplicitComponent):
    # soft maximum between vector a and b
    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('alpha', default=100.)   # nonlinearness factor, If this is too high, exp overflows and optimization fails.
        self.options.declare('units', default=None)

    def setup(self):
        nn = self.options['num_nodes']
        units = self.options['units']
        self.add_input('x1', shape=(nn,), units=units)   # original BEMT output
        self.add_input('x2', shape=(nn,), units=units)   # linear model
        self.add_output('softmax', shape=(nn,), units=units)
        self.declare_partials('softmax', ['*'], rows=np.arange(nn), cols=np.arange(nn))

    def compute(self, inputs, outputs):
        alp = self.options['alpha']
        x1 = inputs['x1']
        x2 = inputs['x2']

        # soft minimum
        outputs['softmax'] = np.log(np.exp(alp * x1) + np.exp(alp * x2)) / alp

    def compute_partials(self, inputs, partials):
        alp = self.options['alpha']
        x1 = inputs['x1']
        x2 = inputs['x2']

        # softmin
        exp1 = np.exp(alp * x1)
        exp2 = np.exp(alp * x2)
        partials['softmax', 'x1'] = exp1 / (exp1 + exp2)
        partials['softmax', 'x2'] = exp2 / (exp1 + exp2)