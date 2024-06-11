import numpy as np
import openmdao.api as om

from uavsizing.aerodynamics import WingedCruiseDrag, BodyDrag
from uavsizing.rotors import ThrustOfEachRotor, ThrustCoefficient
from uavsizing.rotors import RotorInflow, InducedVelocity
from uavsizing.rotors import PropellerRevolutionFromAdvanceRatio, RotorAdvanceRatio
from uavsizing.rotors import RotorRevolutionFromAdvanceRatio, MultiRotorTrim
from uavsizing.utils import SoftMax

class PowerHover(om.ExplicitComponent):
	"""
	Computes the power required for hover
	Parameters:
		n_rotor		: number or rotors
		hover_FM	: hover figure of merit
		rho_air		: air density 
	Inputs:
		UAV|W_total
		UAV|r_rotor
	Outputs:
		power_hover
		thrust_each
	"""
	def initialize(self):
		self.options.declare('n_rotor', types=int, desc='Number of rotors')
		self.options.declare('hover_FM', types=float, desc='Hover figure of merit')
		self.options.declare('rho_air', default=1.225, desc='air density')

	def setup(self):
		self.add_input('UAV|W_total', units='kg', desc='Total weight (MTOW)')
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_output('power_hover', units='W', desc='Power required for hover')
		self.add_output('thrust_each', units='N', desc='Thrust of each rotor during hover')
		self.declare_partials('*', '*')

		self.g = 9.81 #m/s2; gravitational acceleration

	def compute(self, inputs, outputs):
		n_rotor = self.options['n_rotor']
		hover_FM = self.options['hover_FM']
		rho_air = self.options['rho_air']
		W_total = inputs['UAV|W_total']
		r = inputs['UAV|r_rotor']

		S_disk = np.pi * r**2
		outputs['power_hover'] = 1/hover_FM * np.sqrt(((W_total*self.g)**3)/(2*rho_air*S_disk*n_rotor))
		outputs['thrust_each'] = (W_total*self.g)/n_rotor

	def compute_partials(self, inputs, partials):
		n_rotor = self.options['n_rotor']
		hover_FM = self.options['hover_FM']
		rho_air = self.options['rho_air']
		W_total = inputs['UAV|W_total']
		r = inputs['UAV|r_rotor']

		S_disk = np.pi * r**2
		dS_disk_dr = 2 * np.pi * r

		partials['power_hover', 'UAV|W_total'] = 1.5/hover_FM * np.sqrt((W_total * self.g**3)/(2*rho_air*S_disk*n_rotor))
		partials['power_hover', 'UAV|r_rotor'] = -0.5/hover_FM * np.sqrt(((W_total*self.g)**3)/(2*rho_air*n_rotor*S_disk**3)) * dS_disk_dr
		partials['thrust_each', 'UAV|W_total'] = self.g / n_rotor

class PowerForwardEdgewise(om.Group):
	"""
	Computes the power required in edgewise forward flight
	(cruise of wingless multirotor)
	Inputs:
		UAV|W_total
		UAV|Speed
		UAV|r_rotor
		Rotor|mu
		Rotor|alpha
	Outputs:
		power_forward
		Rotor|Thrust
	"""
	def initialize(self):
		self.options.declare('n_rotor', types=int, desc='Number of lifting rotors')
		self.options.declare('hover_FM', types=float, desc='Hover figure of merit')
		self.options.declare('rotor_sigma', types=float, desc='Rotor solidity')
		self.options.declare('rho_air', default=1.225, desc='Air density')

	def setup(self):
		n_rotor = self.options['n_rotor']
		hover_FM = self.options['hover_FM']
		rotor_sigma = self.options['rotor_sigma']
		rho_air = self.options['rho_air']

		# Step 1: Calculate BodyDrag() for the multirotor in cruise
		self.add_subsystem('body_drag',
							BodyDrag(),
							promotes_inputs=['UAV|*', 'Body|sin_beta'],
							promotes_outputs=['Aero|Drag'])
		
		# Step 2: Calculate thrust required for trim and the body tilt angle
		self.add_subsystem('trim',
							MultiRotorTrim(),
							promotes_inputs=['UAV|W_total', 'Aero|Drag'],
							promotes_outputs=[('Thrust', 'Thrust_all'), 'Body|sin_beta'])

		# Step 3: Convert Body|sin_beta into Rotor|alpha
		self.add_subsystem('beta2alpha',
							om.ExecComp('alpha = arccos(sin_beta)', alpha={'units':'rad'}),
							promotes_inputs=[('sin_beta', 'Body|sin_beta')],
							promotes_outputs=[('alpha', 'Rotor|alpha')])

		# Step 4: Calculate the thrust required by each rotor
		self.add_subsystem('thrust_each',
							ThrustOfEachRotor(n_rotor=n_rotor),
							promotes_inputs=['Thrust_all'],
							promotes_outputs=['Rotor|Thrust'])

		# Step 5: Calculate rotor omega given the advance ratio mu
		self.add_subsystem('rotor_revolution',
							RotorRevolutionFromAdvanceRatio(),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*', ('v_inf', 'UAV|Speed')],
							promotes_outputs=['Rotor|omega'])
		self.set_input_defaults('Rotor|mu', 0.15)

		# Step 6: Calculate the thrust coefficient Ct
		self.add_subsystem('Ct',
							ThrustCoefficient(rho_air=rho_air),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*'],
							promotes_outputs=['Rotor|Ct'])

		# Step 7: Calculate profile power
		self.add_subsystem('profile_power',
							ProfilePower(rho_air=rho_air, sigma=rotor_sigma),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*'],
							promotes_outputs=['Rotor|Profile_power'])

		# Step 8: Calculate induced power
		self.add_subsystem('rotor_inflow',
							RotorInflow(),
							promotes_inputs=['Rotor|*'],
							promotes_outputs=['Rotor|lambda'])
		self.add_subsystem('v_induced',
							InducedVelocity(),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*', ('v_inf', 'UAV|Speed')],
							promotes_outputs=['v_induced'])
		self.add_subsystem('kappa',
							InducedPowerFactor(hover_FM=hover_FM, rho_air=rho_air),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*'],
							promotes_outputs=['Rotor|kappa'])

		# Step 9: Calculate total power required
		self.add_subsystem('power_req',
							PowerForwardComp(n_rotor=n_rotor),
							promotes_inputs=['Rotor|*', 'v_induced', ('v_inf', 'UAV|Speed')],
							promotes_outputs=['power_forward'])

class PowerForwardWithWing(om.Group):
	"""
	Power in winged cruise (of Lift+Cruise UAV)
	Inputs:
		UAV|W_total
		UAV|S_wing
		UAV|Speed
		UAV|r_rotor
		Rotor|J
		Rotor|alpha (tilt angle w.r.t vertical direction)
	Outputs:
		power_forward
		Rotor|Thrust
	"""
	def initialize(self):
		self.options.declare('n_rotor', types=int, desc='Number of cruising rotors')
		self.options.declare('hover_FM', types=float, desc='Hover figure of merit')
		self.options.declare('rho_air', default=1.225, desc='Air density')
		self.options.declare('Cd0', types=float, desc='Minimum CD of the drag polar')
		self.options.declare('wing_AR', types=float, desc='Wing aspect ratio')
		self.options.declare('wing_e', types=float, desc='Oswald efficiency')
		self.options.declare('rotor_sigma', types=float, desc='Rotor solidity')

	def setup(self):
		n_rotor = self.options['n_rotor']
		hover_FM = self.options['hover_FM']
		rho_air = self.options['rho_air']
		Cd0 = self.options['Cd0']
		wing_AR = self.options['wing_AR']
		wing_e = self.options['wing_e']
		rotor_sigma = self.options['rotor_sigma']

		# Step 1: Lift should be equal to total weight
		lift_comp = om.ExecComp('lift = 9.81 * weight', lift={'units':'N'}, weight={'units':'kg'})
		self.add_subsystem('lift',
							lift_comp,
							promotes_inputs=[('weight', 'UAV|W_total')],
							promotes_outputs=['lift'])

		# Step 2: Calculate drag in cruise using simple polar equations
		self.add_subsystem('drag',
							WingedCruiseDrag(rho_air=rho_air,
											 Cd0=Cd0,
											 wing_AR=wing_AR,
											 wing_e=wing_e),
							promotes_inputs=[('Aero|Lift', 'lift'), 'UAV|S_wing', 'UAV|Speed'],
							promotes_outputs=['Aero|Drag', 'Aero|CL_cruise'])

		# Step 3: Calculate thrust required by each rotor (thrust = drag)
		self.add_subsystem('thrust_each',
							ThrustOfEachRotor(n_rotor=n_rotor),
							promotes_inputs=[('Thrust_all', 'Aero|Drag')],
							promotes_outputs=['Rotor|Thrust'])

		# Step 4: Calculate rotor omega given propeller advance ratio J; freestream speed = UAV speed
		self.add_subsystem('prop_revolution',
							PropellerRevolutionFromAdvanceRatio(),
							promotes_inputs=['UAV|r_rotor', 'Rotor|J', ('v_inf', 'UAV|Speed')],
							promotes_outputs=['Rotor|omega'])
		self.set_input_defaults('Rotor|J', 1.0) # default J

		# Step 5: Calculate rotor advance ratio mu and thrust coefficient Ct
		self.add_subsystem('mu',
							RotorAdvanceRatio(),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*', ('v_inf', 'UAV|Speed')],
							promotes_outputs=['Rotor|mu'])
		self.add_subsystem('Ct',
							ThrustCoefficient(rho_air=rho_air),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*'],
							promotes_outputs=['Rotor|Ct'])

		# Step 6: Calculate profile power of a rotor
		self.add_subsystem('profile_power',
							ProfilePower(rho_air=rho_air, sigma=rotor_sigma),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*'],
							promotes_outputs=['Rotor|Profile_power'])

		# Step 7: Calculate induced power
		self.add_subsystem('rotor_inflow',
							RotorInflow(),
							promotes_inputs=['Rotor|*'],
							promotes_outputs=['Rotor|lambda'])
		self.add_subsystem('v_induced',
							InducedVelocity(),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*', ('v_inf', 'UAV|Speed')],
							promotes_outputs=['v_induced'])
		self.add_subsystem('kappa',
							InducedPowerFactor(hover_FM=hover_FM, rho_air=rho_air),
							promotes_inputs=['UAV|r_rotor', 'Rotor|*'],
							promotes_outputs=['Rotor|kappa'])

		# Step 8: Calculate total power required
		self.add_subsystem('power_req',
							PowerForwardComp(n_rotor=n_rotor),
							promotes_inputs=['Rotor|*', 'v_induced', ('v_inf', 'UAV|Speed')],
							promotes_outputs=['power_forward'])

		# Assume the rotor tilt angle is 85, or AoA = 5
		self.set_input_defaults('Rotor|alpha', val=85, units='deg')

class PowerForwardComp(om.ExplicitComponent):
	"""
	Computes power required for forward flight
	Parameters:
		n_rotor: number or rotors
	Inputs:
		Rotor|Thrust
		Rotor|Profile_power
		Rotor|alpha
		Rotor|kappa
		v_inf
		v_induced
	Outputs:
		power_forward
	"""
	def initialize(self):
		self.options.declare('n_rotor', types=int, desc='Number of rotors')

	def setup(self):
		self.add_input('Rotor|Thrust', units='N', desc='Thrust of each rotor')
		self.add_input('Rotor|Profile_power', units='W', desc='Profile power of each rotor, P0')
		self.add_input('Rotor|alpha', units='rad', desc='Rotor tilt angle: 90 for being a propeller, 0 for hover')
		self.add_input('Rotor|kappa', desc='Induced power factor')
		self.add_input('v_inf', units='m/s', desc='Freestream velocity')
		self.add_input('v_induced', units='m/s', desc='Induced velocity')
		self.add_output('power_forward', units='W', desc='Power required for forward flight (sum of all rotors)')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		n_rotor = self.options['n_rotor']
		P0 = inputs['Rotor|Profile_power']
		T_rotor = inputs['Rotor|Thrust']
		a = inputs['Rotor|alpha']
		k = inputs['Rotor|kappa']
		v_inf = inputs['v_inf']
		v_ind = inputs['v_induced']

		power_fwd_each = P0 + T_rotor * (k*v_ind + v_inf*np.sin(a))
		outputs['power_forward'] = n_rotor * power_fwd_each

	def compute_partials(self, inputs, partials):
		n_rotor = self.options['n_rotor']
		P0 = inputs['Rotor|Profile_power']
		T_rotor = inputs['Rotor|Thrust']
		a = inputs['Rotor|alpha']
		k = inputs['Rotor|kappa']
		v_inf = inputs['v_inf']
		v_ind = inputs['v_induced']

		partials['power_forward', 'Rotor|Thrust'] = n_rotor * (k*v_ind + v_inf*np.sin(a))
		partials['power_forward', 'Rotor|Profile_power'] = n_rotor
		partials['power_forward', 'Rotor|alpha'] = n_rotor * T_rotor * v_inf*np.cos(a)
		partials['power_forward', 'Rotor|kappa'] = n_rotor * T_rotor * v_ind
		partials['power_forward', 'v_inf'] = n_rotor * T_rotor * np.sin(a)
		partials['power_forward', 'v_induced'] = n_rotor * T_rotor * k

class ProfilePower(om.ExplicitComponent):
	"""
	Computes the profile power of a rotor
	Inputs:
		UAV|r_rotor
		Rotor|mu
		Rotor|omega
	Outputs:
		Rotor|Profile_power
	"""
	def initialize(self):
		self.options.declare('rho_air', default=1.225, desc='Air density')
		self.options.declare('sigma', types=float, desc='Rotor solidity, e.g., 0.13')
		self.options.declare('Cd0', default=0.012, desc='Zero lift drag of a rotor')

	def setup(self):
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_input('Rotor|mu', desc='Rotor advance ratio')
		self.add_input('Rotor|omega', units='rad/s', desc='Rotor angular velocity')
		self.add_output('Rotor|Profile_power', units='W', desc='Profile power of a rotor, P0')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		rho_air = self.options['rho_air']
		sigma = self.options['sigma']
		Cd0 = self.options['Cd0']
		mu = inputs['Rotor|mu']
		omega = inputs['Rotor|omega']
		r = inputs['UAV|r_rotor']

		P0_each = (sigma*Cd0/8) * (1 + 4.65*mu**2) * (np.pi * rho_air * omega**3 * r**5)
		outputs['Rotor|Profile_power'] = P0_each

	def compute_partials(self, inputs, partials):
		rho_air = self.options['rho_air']
		sigma = self.options['sigma']
		Cd0 = self.options['Cd0']
		mu = inputs['Rotor|mu']
		omega = inputs['Rotor|omega']
		r = inputs['UAV|r_rotor']

		k1 = sigma * Cd0 / 8
		k2 = 1 + 4.65*mu**2
		k3 = np.pi * rho_air * omega**3 * r**5

		partials['Rotor|Profile_power', 'Rotor|mu'] = k1 * k3 * (2 * 4.65 * mu)
		partials['Rotor|Profile_power', 'Rotor|omega'] = k1 * k2 * (np.pi * rho_air * 3 * omega**2 * r**5)
		partials['Rotor|Profile_power', 'UAV|r_rotor'] = k1 * k2 * (np.pi * rho_air * omega**3 * 5 * r**4)

class InducedPowerFactor(om.Group):
	"""
	Computes the induced power factor kappa in forward flight
	Inputs:
		UAV|r_rotor
		Rotor|Thrust
		Rotor|Profile_power
	Outputs:
		Rotor|kappa
	"""
	def initialize(self):
		self.options.declare('hover_FM', types=float, desc='Hover figure of merit')
		self.options.declare('rho_air', default=1.225, desc='Air density')

	def setup(self):
		hover_FM = self.options['hover_FM']
		rho_air = self.options['rho_air']

		# Compute kappa value
		self.add_subsystem('kappa_raw',
							InducedPowerFactorComp(hover_FM=hover_FM, rho_air=rho_air),
							promotes_inputs=['*'])
		# minimum value of kappa
		indep = self.add_subsystem('kappa_min', om.IndepVarComp())
		indep.add_output('kappa_min', val=1.15)

		# kappa = SoftMax(kappa_raw, kappa_min)
		self.add_subsystem('softmax',
							SoftMax(rho=30),
							promotes_outputs=[('fmax', 'Rotor|kappa')])

		self.connect('kappa_raw.kappa_raw', 'softmax.f1')
		self.connect('kappa_min.kappa_min', 'softmax.f2')

class InducedPowerFactorComp(om.ExplicitComponent):
	"""
	Computes the induced power factor kappa in forward flight
	Inputs:
		Rotor|Thrust
		Rotor|Profile_power
		UAV|r_rotor
	Outputs:
		kappa_raw
	"""
	def initialize(self):
		self.options.declare('hover_FM', types=float, desc='Hover figure of merit')	
		self.options.declare('rho_air', default=1.225, desc='Air density')

	def setup(self):
		self.add_input('Rotor|Thrust', units='N', desc='Thrust of a rotor')
		self.add_input('Rotor|Profile_power', units='W', desc='Profile power of a rotor, P0')
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_output('kappa_raw', desc='Induced power factor')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		hover_FM = self.options['hover_FM']
		rho_air = self.options['rho_air']
		thrust = inputs['Rotor|Thrust']
		P0_each = inputs['Rotor|Profile_power']
		r = inputs['UAV|r_rotor']

		S_disk = np.pi * r**2
		kappa = 1/hover_FM -  P0_each * np.sqrt((2*rho_air*S_disk)/thrust**3)
		outputs['kappa_raw'] = kappa

	def compute_partials(self, inputs, partials):
		rho_air = self.options['rho_air']
		thrust = inputs['Rotor|Thrust']
		P0_each = inputs['Rotor|Profile_power']
		r = inputs['UAV|r_rotor']

		S_disk = np.pi * r**2
		dk_dt = 1.5 * P0_each * np.sqrt((2*rho_air*S_disk)/thrust**5)
		dk_dp = - np.sqrt((2*rho_air*S_disk)/thrust**3)
		dk_dr = - P0_each * np.sqrt(rho_air/(2*S_disk*thrust**3)) * 2*np.pi*r

		partials['kappa_raw', 'Rotor|Thrust'] = dk_dt
		partials['kappa_raw', 'Rotor|Profile_power'] = dk_dp 
		partials['kappa_raw', 'UAV|r_rotor'] = dk_dr

