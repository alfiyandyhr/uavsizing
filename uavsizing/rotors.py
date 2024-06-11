import numpy as np
import openmdao.api as om

class RotorInflow(om.ImplicitComponent):
	"""
	Computes the inflow of a rotor (lambda)
	Inputs:
		Rotor|mu
		Rotor|alpha
		Rotor|Ct
	Outputs:
		Rotor|lambda
	"""
	def setup(self):
		self.add_input('Rotor|mu', desc='Advance ratio')
		self.add_input('Rotor|alpha', units='rad', desc='Rotor tilt angle')
		self.add_input('Rotor|Ct', desc='Thrust coefficient')
		self.add_output('Rotor|lambda', val=0.1, lower=0.0, upper=10.0, desc='Rotor inflow')
		self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		mu = inputs['Rotor|mu']
		a = inputs['Rotor|alpha']
		Ct = inputs['Rotor|Ct']
		lmbd = outputs['Rotor|lambda']
		# Compute residuals
		residuals['Rotor|lambda'] = mu*np.tan(a) + Ct / (2*np.sqrt((mu**2 + lmbd**2))) - lmbd

	def linearize(self, inputs, outputs, partials):
		mu = inputs['Rotor|mu']
		a = inputs['Rotor|alpha']
		Ct = inputs['Rotor|Ct']
		lmbd = outputs['Rotor|lambda']

		partials['Rotor|lambda', 'Rotor|mu'] = np.tan(a) - (Ct*mu)/(2*np.sqrt((mu**2 + lmbd**2)**3))
		partials['Rotor|lambda', 'Rotor|alpha'] = mu / (np.cos(a) * np.cos(a))
		partials['Rotor|lambda', 'Rotor|Ct'] = 1 / ( 2 * np.sqrt(mu**2 + lmbd**2) )
		partials['Rotor|lambda', 'Rotor|lambda'] = - (Ct*lmbd) / (2 * np.sqrt((mu**2 + lmbd**2)**3)) - 1

class InducedVelocity(om.ExplicitComponent):
	"""
	Computes the induced velocity
	Inputs:
		UAV|r_rotor
		Rotor|omega
		Rotor|alpha
		Rotor|lambda
		v_inf
	Outputs:
		v_induced
	"""
	def setup(self):
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_input('Rotor|alpha', units='rad', desc='Rotor tilt angle')
		self.add_input('Rotor|omega', units='rad/s', desc='Rotor angular velocity')
		self.add_input('Rotor|lambda', desc='Rotor inflow')
		self.add_input('v_inf', units='m/s', desc='Freestream velocity')
		self.add_output('v_induced', units='m/s', desc='Induced velocity')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		r = inputs['UAV|r_rotor']
		a = inputs['Rotor|alpha']
		omega = inputs['Rotor|omega']
		lmbd = inputs['Rotor|lambda']
		v_inf = inputs['v_inf']

		outputs['v_induced'] = omega * r * lmbd - v_inf * np.sin(a)

	def compute_partials(self, inputs, partials):
		r = inputs['UAV|r_rotor']
		a = inputs['Rotor|alpha']
		omega = inputs['Rotor|omega']
		lmbd = inputs['Rotor|lambda']
		v_inf = inputs['v_inf']

		partials['v_induced', 'UAV|r_rotor'] = omega * lmbd
		partials['v_induced', 'Rotor|alpha'] = - v_inf * np.cos(a)
		partials['v_induced', 'Rotor|omega'] = r * lmbd
		partials['v_induced', 'Rotor|lambda'] = omega * r
		partials['v_induced', 'v_inf'] = - np.sin(a)

class ThrustCoefficient(om.ExplicitComponent):
	"""
	Computes the thrust coefficient
	Inputs:
		UAV|r_rotor
		Rotor|Thrust
		Rotor|omega
	Outputs:
		Rotor|Ct
	"""
	def initialize(self):
		self.options.declare('rho_air', default=1.225, desc='Air density')

	def setup(self):
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_input('Rotor|Thrust', units='N', desc='Thrust of a rotor')
		self.add_input('Rotor|omega', units='rad/s', desc='Rotor angular velocity')
		self.add_output('Rotor|Ct', desc='Thrust coefficient')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		rho_air = self.options['rho_air']
		thrust = inputs['Rotor|Thrust']
		omega = inputs['Rotor|omega']
		r = inputs['UAV|r_rotor']

		outputs['Rotor|Ct'] = thrust/(np.pi * rho_air * omega**2 * r**4)

	def compute_partials(self, inputs, partials):
		rho_air = self.options['rho_air']
		thrust = inputs['Rotor|Thrust']
		omega = inputs['Rotor|omega']
		r = inputs['UAV|r_rotor']

		partials['Rotor|Ct', 'Rotor|Thrust'] = 1 / (np.pi * rho_air * omega**2 * r**4)
		partials['Rotor|Ct', 'Rotor|omega'] = thrust/(np.pi*rho_air*r**4) * (-2/omega**3)
		partials['Rotor|Ct', 'UAV|r_rotor'] = thrust/(np.pi*rho_air*omega**2) * (-4/r**5)

class RotorAdvanceRatio(om.ExplicitComponent):
	"""
	Computes the rotor advance ratio
		mu = V cos(alpha) / (omega * r)
	Inputs:
		UAV|r_rotor
		v_inf
		Rotor|alpha
		Rotor|omega
	Outputs:
		Rotor|mu
	"""
	def setup(self):
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_input('v_inf', units='m/s', desc='Freestream velocity')
		self.add_input('Rotor|alpha', units='rad', desc='Rotor tilt angle')
		self.add_input('Rotor|omega', units='rad/s', desc='Rotor angular velocity')
		self.add_output('Rotor|mu', desc='Advance ratio of rotor')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_inf = inputs['v_inf']
		a = inputs['Rotor|alpha']
		r = inputs['UAV|r_rotor']
		omega = inputs['Rotor|omega']

		outputs['Rotor|mu'] = v_inf * np.cos(a) / (omega * r)

	def compute_partials(self, inputs, partials):
		v_inf = inputs['v_inf']
		a = inputs['Rotor|alpha']
		r = inputs['UAV|r_rotor']
		omega = inputs['Rotor|omega']

		partials['Rotor|mu', 'v_inf'] = np.cos(a) / (omega * r)
		partials['Rotor|mu', 'Rotor|alpha'] = - v_inf * np.sin(a) / (omega * r) 
		partials['Rotor|mu', 'Rotor|omega'] = - v_inf * np.cos(a) / (omega**2 * r)
		partials['Rotor|mu', 'UAV|r_rotor'] = - v_inf * np.cos(a) / (omega * r**2)

class RotorRevolutionFromCT(om.ExplicitComponent):
	"""
	Computes the rotor revolution (omega) given the thrust coefficient CT
	Inputs:
		UAV|r_rotor
		Rotor|Thrust
		Rotor|Ct
	Outputs:
		Rotor|omega
	"""
	def initialize(self):
		self.options.declare('rho_air', default=1.225, desc='Air density')

	def setup(self):
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_input('Rotor|Thrust', units='N', desc='Thrust of a rotor')
		self.add_input('Rotor|Ct', desc='Thrust coefficient')
		self.add_output('Rotor|omega', units='rad/s', desc='Rotor angular velocity')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		rho_air = self.options['rho_air']
		thrust = inputs['Rotor|Thrust']
		Ct = inputs['Rotor|Ct']
		r = inputs['Rotor|r_rotor']

		outputs['Rotor|omega'] = np.sqrt(thrust / (np.pi * rho_air * Ct * r**4))

	def compute_partials(self, inputs, partials):
		rho_air = self.options['rho_air']
		thrust = inputs['Rotor|Thrust']
		Ct = inputs['Rotor|Ct']
		r = inputs['Rotor|r_rotor']

		partials['Rotor|omega', 'Rotor|Thrust'] = 1 / (np.sqrt(4 * np.pi * thrust * rho_air * Ct * r**4))
		partials['Rotor|omega', 'Rotor|Ct'] = - np.sqrt(thrust / (4 * np.pi * rho_air * Ct**3 * r**4))
		partials['Rotor|omega', 'UAV|r_rotor'] = - np.sqrt((4 * thrust) / (np.pi * rho_air * Ct * r**6))

class RotorRevolutionFromAdvanceRatio(om.ExplicitComponent):
	"""
	Computes the rotor revolution (omega) given the advance ratio
		omega = V cos(alpha) / (mu * r)
	Inputs:
		UAV|r_rotor
		v_inf
		Rotor|alpha
		Rotor|mu
	Outputs:
		Rotor|omega
	"""
	def setup(self):
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_input('v_inf', units='m/s', desc='Freestream velocity')
		self.add_input('Rotor|alpha', units='rad', desc='Rotor tilt angle')
		self.add_input('Rotor|mu', desc='Advance ratio of rotor')
		self.add_output('Rotor|omega', units='rad/s', desc='Rotor angular velocity')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_inf = inputs['v_inf']
		a = inputs['Rotor|alpha']
		r = inputs['UAV|r_rotor']
		mu = inputs['Rotor|mu']
		
		outputs['Rotor|omega'] = v_inf * np.cos(a) / (mu * r)

	def compute_partials(self, inputs, partials):
		v_inf = inputs['v_inf']
		a = inputs['Rotor|alpha']
		r = inputs['UAV|r_rotor']
		mu = inputs['Rotor|mu']

		partials['Rotor|omega', 'v_inf'] = np.cos(a) / (mu * r)
		partials['Rotor|omega', 'Rotor|alpha'] = - v_inf * np.sin(a) / (mu * r) 
		partials['Rotor|omega', 'Rotor|mu'] = - v_inf * np.cos(a) / (mu**2 * r)
		partials['Rotor|omega', 'UAV|r_rotor'] = - v_inf * np.cos(a) / (mu * r**2)

class PropellerRevolutionFromAdvanceRatio(om.ExplicitComponent):
	"""
	Computes the propeller revolution (omega) given the advance ratio
		J = V / (n D)
	Inputs:
		UAV|r_rotor
		v_inf
		Rotor|J
	Outputs:
		Rotor|omega
	"""
	def setup(self):
		self.add_input('UAV|r_rotor', units='m', desc='Rotor radius')
		self.add_input('v_inf', units='m/s', desc='Freestream velocity')
		self.add_input('Rotor|J', desc='Advance ratio of propeller')
		self.add_output('Rotor|omega', units='rad/s', desc='Rotor angular velocity')
		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_inf = inputs['v_inf']
		r = inputs['UAV|r_rotor']
		J = inputs['Rotor|J']
		n = v_inf / (2 * r * J) # revolutions/second

		outputs['Rotor|omega'] = 2 * np.pi * n

	def compute_partials(self, inputs, partials):
		v_inf = inputs['v_inf']
		r = inputs['UAV|r_rotor']
		J = inputs['Rotor|J']

		partials['Rotor|omega', 'v_inf'] = np.pi / (r * J)
		partials['Rotor|omega', 'UAV|r_rotor'] = - (np.pi * v_inf) / (r**2 * J)
		partials['Rotor|omega', 'Rotor|J'] = - (np.pi * v_inf) / (r * J**2)

class MultiRotorTrim(om.ExplicitComponent):
	"""
	Computes the body tilt angle for wingless multirotor in cruise
	Inputs:
		UAV|W_total
		Aero|Drag
	Outputs:
		Thrust
		Body|sin_beta
	"""
	def setup(self):
		self.add_input('UAV|W_total', units='kg', desc='Vehicle total weight (MTOW)')
		self.add_input('Aero|Drag', units='N', desc='Drag')
		self.add_output('Thrust', units='N', desc='Thrust required as a vehicle')
		self.add_output('Body|sin_beta', desc='sin(beta), beta: Body incidence angle')
		self.declare_partials('*', '*')

		self.g = 9.81 # gravitational acceleration

	def compute(self, inputs, outputs):
		W_total = inputs['UAV|W_total']
		D = inputs['Aero|Drag']
		thrust = np.sqrt((W_total*self.g)**2 + D**2)

		outputs['Thrust'] = thrust
		outputs['Body|sin_beta'] = (W_total*self.g) / thrust

	def compute_partials(self, inputs, partials):
		W_total = inputs['UAV|W_total']
		D = inputs['Aero|Drag']
		thrust = np.sqrt((W_total*self.g)**2 + D**2)

		partials['Thrust', 'UAV|W_total'] = (W_total * self.g**2) / thrust
		partials['Thrust', 'Aero|Drag'] = D / thrust
		partials['Body|sin_beta', 'UAV|W_total'] = self.g/thrust - (W_total**2 * self.g**3)/(thrust**3)
		partials['Body|sin_beta', 'Aero|Drag'] = - (W_total*self.g*D) / (thrust**3)

class ThrustOfEachRotor(om.ExplicitComponent):
	"""
	Computes the thrust required by each rotor given the weight or drag requirement
	Parameters:
		n_rotor: number of rotors
	Inputs:
		Thrust_all: thrust required (sum of all rotors)
	Outputs:
		Rotor|Thrust
	"""
	def initialize(self):
		self.options.declare('n_rotor', types=int, desc='Number of rotors')

	def setup(self):
		n_rotor = self.options['n_rotor']

		self.add_input('Thrust_all', units='N', desc='Thrust required (sum of all rotors)')
		self.add_output('Rotor|Thrust', units='N', desc='Thrust required by each rotor')

		# partial is constant
		self.declare_partials('Rotor|Thrust', 'Thrust_all', val=1/n_rotor)

	def compute(self, inputs, outputs):
		outputs['Rotor|Thrust'] = inputs['Thrust_all'] / self.options['n_rotor']
