import numpy as np
import openmdao.api as om

from uavsizing.energy import EnergyConsumption

# --- Total weight (MTOW) estimation --- #

class MTOWEstimation(om.Group):
	"""
	Computes UAV total weight estimation given the design variables and mission requirement.
	Must be used with an optimizer (or a nonlinear solver) to converge the weight residual.

	Inputs:
	(mission requirements)
		flight_distance
		hover_time
		payload_weight
	(UAV design variables)
		UAV|W_total			: MTOW, including the payload
		UAV|Speed			: Cruise speed
		UAV|r_rotor_lift	: Lifting rotor radius
		UAV|r_rotor_cruise	: Cruising rotor radius 	(for lift+cruise only)
		UAV|S_wing			: Wing area 				(for lift+cruise only)
		Rotor|mu 			: Rotor advance ratio		(for multirotor only)
		Rotor|J 			: Propeller advance ratio	(for lift+cruise only)

	Outputs:
	(weight of each component)
		UAV|W_battery
		UAV|W_rotor_all
		UAV|W_motor_all
		UAV|W_ESC_all
	(major performances)
		power_hover
		power_cruise
	(for some constraints)
		disk_loading_hover
		disk_loading_cruise
		Rotor|Ct (in cruise)
		CL_cruise
	"""

	def initialize(self):
		self.options.declare('UAV_options', types=dict, desc='Dict containing all options parameters')
		self.options.declare('use_solver', types=bool, desc='Whether to use non linear solver')

	def setup(self):
		params = self.options['UAV_options']
		use_solver = self.options['use_solver']

		# Unpacking options
		uav_config = params['uav_config']
		battery_rho = params['battery_rho']
		battery_eff = params['battery_eff']
		battery_max_discharge = params['battery_max_discharge']
		n_rotors_lift = params['n_rotors_lift']		# number of lifting rotors

		if uav_config == 'multirotor':
			pass
		elif uav_config == 'lift+cruise':
			n_rotors_cruise = params['n_rotors_cruise']
		else:
			raise RuntimeError('UAV configuration is not available')

		# --- Calculate total energy consumptions to fly the mission --- #
		self.add_subsystem('energy',
							EnergyConsumption(UAV_options=params),
							promotes_inputs=['*'],
							promotes_outputs=['*'])

		# --- Calculate weight estimation of each component --- #

		# 1. Battery weight
		# battery weight is computed taking into account loss in efficiency,
		# avionics power, and its maximum discharge rate
		battery_weight_comp = om.ExecComp('W_battery = energy_req / (battery_rho * battery_eff * battery_max_discharge)',
										   W_battery={'units': 'kg'},
										   energy_req={'units': 'W * h'},
										   battery_rho={'units': 'W * h / kg', 'val': battery_rho},
										   battery_eff={'val': battery_eff},
										   battery_max_discharge={'val': battery_max_discharge})
		self.add_subsystem('battery_weight',
							battery_weight_comp,
							promotes_inputs=[('energy_req', 'energy_cnsmp')],
							promotes_outputs=[('W_battery', 'UAV|W_battery')])

		# 2. Propulsion weight
		# includes the weight of rotors, motors, and ESCs

		# For multirotor
		if uav_config == 'multirotor':
			self.add_subsystem('rotor_weight',
								RotorWeight(n_rotor=n_rotors_lift),
								promotes_inputs=[('UAV|r_rotor', 'UAV|r_rotor_lift')],
								promotes_outputs=['UAV|W_rotor_all'])
			self.add_subsystem('motor_weight',
								MotorWeight(n_motor=n_rotors_lift), # one rotor typically has one motor + ESC
								promotes_outputs=['UAV|W_motor_all'])
			self.connect('power_hover', 'motor_weight.max_power')   # assume max power output = power in hover
			self.add_subsystem('ESC_weight',
								ESCWeight(n_motor=n_rotors_lift), # one rotor typically has one motor + ESC
								promotes_outputs=['UAV|W_ESC_all'])
			self.connect('power_hover', 'ESC_weight.max_power')   # assume max power output = power in hover

		elif uav_config == 'lift+cruise':
			# Lifting rotors
			self.add_subsystem('rotor_weight_lift',
								RotorWeight(n_rotor=n_rotors_lift),
								promotes_inputs=[('UAV|r_rotor', 'UAV|r_rotor_lift')],
								promotes_outputs=[('UAV|W_rotor_all', 'W_rotors_lift')])
			self.add_subsystem('motor_weight_lift',
								MotorWeight(n_motor=n_rotors_lift), # one rotor typically has one motor + ESC
								promotes_outputs=[('UAV|W_motor_all', 'W_motors_lift')])
			self.connect('power_hover', 'motor_weight_lift.max_power')   # assume max power output = power in hover
			self.add_subsystem('ESC_weight_lift',
								ESCWeight(n_motor=n_rotors_lift), # one rotor typically has one motor + ESC
								promotes_outputs=[('UAV|W_ESC_all', 'W_ESCs_lift')])
			self.connect('power_hover', 'ESC_weight_lift.max_power')   # assume max power output = power in hover

			# Cruising rotors
			self.add_subsystem('rotor_weight_cruise',
								RotorWeight(n_rotor=n_rotors_cruise),
								promotes_inputs=[('UAV|r_rotor', 'UAV|r_rotor_cruise')],
								promotes_outputs=[('UAV|W_rotor_all', 'W_rotors_cruise')])
			self.add_subsystem('motor_weight_cruise',
								MotorWeight(n_motor=n_rotors_cruise), # one rotor typically has one motor + ESC
								promotes_outputs=[('UAV|W_motor_all', 'W_motors_cruise')])
			self.connect('power_forward', 'motor_weight_cruise.max_power')   # assume max power output = power in cruise
			self.add_subsystem('ESC_weight_cruise',
								ESCWeight(n_motor=n_rotors_cruise), # one rotor typically has one motor + ESC
								promotes_outputs=[('UAV|W_ESC_all', 'W_ESCs_cruise')])
			self.connect('power_forward', 'ESC_weight_cruise.max_power')   # assume max power output = power in cruise

			# Sum both systems weight
			adder = om.AddSubtractComp()
			adder.add_equation('W_rotors',
								input_names=['W_rotors_lift', 'W_rotors_cruise'],
								units='kg',
								scaling_factors=[1., 1.])
			adder.add_equation('W_motors',
								input_names=['W_motors_lift', 'W_motors_cruise'],
								units='kg',
								scaling_factors=[1., 1.])
			adder.add_equation('W_ESCs',
								input_names=['W_ESCs_lift', 'W_ESCs_cruise'],
								units='kg',
								scaling_factors=[1., 1.])
			self.add_subsystem('propulsion_weight',
								adder,
								promotes_inputs=['*'],
								promotes_outputs=[('W_rotors', 'UAV|W_rotor_all'), ('W_motors', 'UAV|W_motor_all'), ('W_ESCs', 'UAV|W_ESC_all')])

		# 3. Wing weight
		# wing is possessed by a lift+cruise configuration
		if uav_config == 'lift+cruise':
			self.add_subsystem('wing_weight',
								WingWeight(),
								promotes_inputs=['UAV|S_wing'],
								promotes_outputs=['UAV|W_wing'])

		# 4. Weight residuals
		# W_residual = W_total - W_battery - W_payload - W_wing - W_propulsion - W_frame
		# where:
		# W_propulsion = W_rotor_all + W_motor_all + W_ESC_all
		# and
		# W_frame = 0.2 * W_total + 0.5 (kg)
		# W_residual should then be driven to 0 by a nonlinear solver or treated as an optimization constraint
		input_list = [('W_total', 'UAV|W_total'),
					  ('W_battery', 'UAV|W_battery'),
					  ('W_payload', 'payload_weight'),
					  ('W_wing', 'UAV|W_wing'),
					  ('W_rotor_all', 'UAV|W_rotor_all'),
					  ('W_motor_all', 'UAV|W_motor_all'),
					  ('W_ESC_all', 'UAV|W_ESC_all')] 

		W_residual_eqn = 'W_residual = W_total - W_battery - W_payload - W_wing - W_rotor_all - W_motor_all - W_ESC_all - 0.2*W_total - 0.5'
		self.add_subsystem('w_residual_comp',
							om.ExecComp(W_residual_eqn, units='kg'),
							promotes_inputs=input_list,
							promotes_outputs=['W_residual'])
		# for wingless multirotor, set 0 weight for wing
		if uav_config == 'multirotor':
			self.set_input_defaults('UAV|W_wing', 0.0)

		# If nonlinear solver to be used
		if use_solver:
			# This drives W_residual = 0 by varying W_total. LB and UB of W_total should be given.
			residual_balance = om.BalanceComp('W_total',
											   units='kg',
											   eq_units='kg',
											   lower=1.0,
											   upper=10.0,
											   val=5.0,
											   rhs_val=0.0,
											   use_mult=False)
			self.add_subsystem('weight_balance',
								residual_balance,
								promotes_inputs=[('lhs:W_total', 'W_residual')],
								promotes_outputs=[('W_total', 'UAV|W_total')])

			# self.connect('weight_balance.W_total', 'UAV|W_total')
			# self.connect('w_residual_comp.W_residual', 'weight_balance.lhs:W_total')

			self.set_input_defaults('weight_balance.rhs:W_total', 0.0)

			# Add solvers for implicit relations
			self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=0, rtol=1e-10)
			self.nonlinear_solver.options['err_on_non_converge'] = True
			self.nonlinear_solver.options['reraise_child_analysiserror'] = True
			self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
			self.nonlinear_solver.linesearch.options['maxiter'] = 10
			self.nonlinear_solver.linesearch.options['iprint'] = 0
			self.linear_solver = om.DirectSolver(assemble_jac=True)



# --- Component Weights --- #

class RotorWeight(om.ExplicitComponent):
	"""
	Computes rotor weight
	Parameters:
		n_rotor: number or rotors
	Inputs:
		UAV|r_rotor: rotor radius [m]
	Outputs:
		UAV|W_rotor_all: weight of all rotors
	"""
	def initialize(self):
		self.options.declare('n_rotor', types=int, desc='Number of rotors')

	def setup(self):
		self.add_input('UAV|r_rotor', val=0.3*0.0254, units='m', desc='Rotor radius')
		self.add_output('UAV|W_rotor_all', units='kg', desc='Weight of all rotors')
		self.declare_partials('UAV|W_rotor_all', 'UAV|r_rotor')

	def compute(self, inputs, outputs):
		n_rotor = self.options['n_rotor']
		r_rotor = inputs['UAV|r_rotor']

		W_rotor = 0.1870854*(2*r_rotor)**2 - 0.0201654*(2*r_rotor)
		outputs['UAV|W_rotor_all'] = n_rotor * W_rotor

	def compute_partials(self, inputs, partials):
		n_rotor = self.options['n_rotor']
		r_rotor = inputs['UAV|r_rotor']

		dW_rotor_dr = 0.1870854*4*2*r_rotor - 0.0201654*2
		partials['UAV|W_rotor_all', 'UAV|r_rotor'] = n_rotor * dW_rotor_dr

class MotorWeight(om.ExplicitComponent):
	"""
	Computes motor weight
	Parameters:
		n_motor: number of motors
	Inputs:
		max_power: maximum power
	Outputs:
		UAV|W_motor_all: weight of all motors
	"""
	def initialize(self):
		self.options.declare('n_motor', types=int, desc='number of motors')

	def setup(self):
		self.add_input('max_power', units='hp', desc='max power')
		self.add_output('UAV|W_motor_all', units='kg', desc='all motors weight')
		self.declare_partials('UAV|W_motor_all', 'max_power')

		self.POF = 1.5   # installed power = overhead factor * max power output

	def compute(self, inputs, outputs):
		n_motor = self.options['n_motor']
		W_motor = 0.412 * self.POF * inputs['max_power'] / n_motor    # [lb], Make sure power in [hp]
		outputs['UAV|W_motor_all'] = W_motor * n_motor * 0.453592       # [kg]

	def compute_partials(self, inputs, partials):
		n_motor = self.options['n_motor']
		dW_motor_dp = 0.412 * self.POF / n_motor
		partials['UAV|W_motor_all', 'max_power'] = n_motor * dW_motor_dp * 0.453592 

class ESCWeight(om.ExplicitComponent):
	"""
	Computes ESC weight
	Parameters:
		n_motor: number of motors
	Inputs:
		max_power: maximum power
	Outputs:
		UAV|W_ESC_all: weight of all ESCs
	"""
	def initialize(self):
		self.options.declare('n_motor', types=int, desc='number of motors')

	def setup(self):
		self.add_input('max_power', units='hp', desc='max power')
		self.add_output('UAV|W_ESC_all', units='kg', desc='all ESCs weight')
		self.declare_partials('UAV|W_ESC_all', 'max_power')

		self.POF = 1.5   # installed power = overhead factor * max power output

	def compute(self, inputs, outputs):
		n_motor = self.options['n_motor']
		W_ESC = 0.591 * self.POF * inputs['max_power'] / n_motor    # [lb], Make sure power in [hp]
		outputs['UAV|W_ESC_all'] = W_ESC * n_motor * 0.453592        # [kg]

	def compute_partials(self, inputs, partials):
		n_motor = self.options['n_motor']
		dW_ESC_dp = 0.591 * self.POF / n_motor
		partials['UAV|W_ESC_all', 'max_power'] = n_motor * dW_ESC_dp * 0.453592 

# class MotorWeight(om.ExplicitComponent):
# 	"""
# 	Computes motor weight
# 	Parameters:
# 		n_motor: number of motors
# 	Inputs:
# 		max_power: maximum power
# 	Outputs:
# 		UAV|W_motor_all: weight of all motors
# 	"""
# 	def initialize(self):
# 		self.options.declare('n_motor', types=int, desc='Number of motors')

# 	def setup(self):
# 		self.add_input('max_power', units='W', desc='Maximum power')
# 		self.add_output('UAV|W_motor_all', units='kg', desc='Weight of all motors')
# 		self.declare_partials('UAV|W_motor_all', 'max_power')

# 		# POF: Power Overhead Factor
# 		# Installed_Power = POF * Max_Power_Output
# 		self.POF = 1.5

# 	def compute(self, inputs, outputs):
# 		n_motor = self.options['n_motor']

# 		W_motor = 0.0005525 * self.POF * inputs['max_power'] / n_motor
# 		outputs['UAV|W_motor_all'] = W_motor * n_motor

# 	def compute_partials(self, inputs, partials):
# 		n_motor = self.options['n_motor']

# 		dW_motor_dp = 0.0005525 * self.POF / n_motor
# 		partials['UAV|W_motor_all', 'max_power'] = n_motor * dW_motor_dp

# class ESCWeight(om.ExplicitComponent):
# 	"""
# 	Computes ESC weight
# 	Parameters:
# 		n_motor: number of motors
# 	Inputs:
# 		max_power: maximum power
# 	Outputs:
# 		UAV|W_ESC_all: weight of all ESCs
# 	"""
# 	def initialize(self):
# 		self.options.declare('n_motor', types=int, desc='Number of motors')

# 	def setup(self):
# 		self.add_input('max_power', units='W', desc='Maximum power')
# 		self.add_output('UAV|W_ESC_all', units='kg', desc='Weight of all ESCs')
# 		self.declare_partials('UAV|W_ESC_all', 'max_power')

# 		# POF: Power Overhead Factor
# 		# Installed_Power = POF * Max_Power_Output
# 		self.POF = 1.5

# 	def compute(self, inputs, outputs):
# 		n_motor = self.options['n_motor']

# 		W_ESC = 0.0007925 * self.POF * inputs['max_power'] / n_motor
# 		outputs['UAV|W_ESC_all'] = W_ESC * n_motor

# 	def compute_partials(self, inputs, partials):
# 		n_motor = self.options['n_motor']

# 		dW_ESC_dp = 0.0007925 * self.POF / n_motor
# 		partials['UAV|W_ESC_all', 'max_power'] = n_motor * dW_ESC_dp

class WingWeight(om.ExplicitComponent):
	"""
	Computes wing weight given the wing area
	Inputs:
		UAV|S_wing: wing area (m^2)
	Outputs:
		UAV|W_wing: wing weight (kg)
	"""
	def setup(self):
		self.add_input('UAV|S_wing', units='m**2', desc='Wing area')
		self.add_output('UAV|W_wing', units='kg', desc='Wing weight')
		self.declare_partials('UAV|W_wing', 'UAV|S_wing')

	def compute(self, inputs, outputs):
		outputs['UAV|W_wing'] = -0.08017 + 2.2854 * inputs['UAV|S_wing']

	def compute_partials(self, inputs, partials):
		partials['UAV|W_wing', 'UAV|S_wing'] = 2.2854

