import numpy as np
import openmdao.api as om

from uavsizing.powers import PowerHover, PowerForwardEdgewise, PowerForwardWithWing

class EnergyConsumption(om.Group):
	"""
	Computes the energy consumption of a UAV given the vehicle specifications and mission requirements.
	It also computes disk loading in hover and cruise.

	Inputs:
	(mission requirements)
		mission_range
		hover_time
	(UAV design variables)
		UAV|W_total			: MTOW, including the payload
		UAV|Speed			: Cruise speed
		UAV|r_rotor_lift	: Lifting rotor radius
		UAV|r_rotor_cruise	: Cruising rotor radius 	(for lift+cruise only)
		UAV|S_wing			: Wing area 				(for lift+cruise only)
		Rotor|mu 			: Rotor advance ratio		(for multirotor only)
		Rotor|J 			: Propeller advance ratio	(for lift+cruise only)

	Outputs:
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
		self.options.declare('UAV_options', types=dict, desc='Dict containing all option parameters')

	def setup(self):
		params = self.options['UAV_options']

		# Unpacking options
		uav_config = params['uav_config']
		n_rotors_lift = params['n_rotors_lift']		# number of lifting rotors
		rotor_sigma = params['rotor_lift_solidity']	# solidity of lifting rotors 
		hover_FM = params['hover_FM']				# hover figure of merit

		if uav_config == 'multirotor':
			pass
		elif uav_config == 'lift+cruise':
			n_rotors_cruise = params['n_rotors_cruise']		# number of cruising rotors
			Cd0 = params['Cd0'] 							# minimum drag of the drag polar
			wing_AR = params['wing_AR']						# wing aspect ratio
			wing_e = params['wing_e']						# Oswald efficiency
			prop_sigma = params['rotor_cruise_solidity']	# solidty of cruising rotors
		else:
			raise RuntimeError('UAV configuration is not available.')


		# --- Calculate power consumptions for each flight segment --- #
		# power in hover
		self.add_subsystem('power_hover',
							PowerHover(n_rotor=n_rotors_lift, hover_FM=hover_FM),
							promotes_inputs=['UAV|W_total', ('UAV|r_rotor', 'UAV|r_rotor_lift')],
							promotes_outputs=['power_hover'])

		# power in cruise
		if uav_config == 'multirotor':
			input_list = ['UAV|W_total', 'UAV|Speed', 'Rotor|mu', ('UAV|r_rotor', 'UAV|r_rotor_lift')]
			self.add_subsystem('power_forward_edgewise',
								PowerForwardEdgewise(n_rotor=n_rotors_lift, hover_FM=hover_FM, rotor_sigma=rotor_sigma),
								promotes_inputs=input_list,
								# promotes_outputs=['power_forward', 'Rotor|Thrust', 'Rotor|Ct'])
								promotes_outputs=['*'])

		elif uav_config == 'lift+cruise':
			input_list = ['UAV|W_total', 'UAV|Speed', 'UAV|S_wing', 'Rotor|J', ('UAV|r_rotor', 'UAV|r_rotor_cruise')]
			# input_list = ['UAV|W_total', 'UAV|Speed', 'UAV|S_wing']
			self.add_subsystem('power_forward_wing',
								PowerForwardWithWing(n_rotor=n_rotors_cruise, hover_FM=hover_FM, Cd0=Cd0, wing_AR=wing_AR, wing_e=wing_e, rotor_sigma=prop_sigma),
								promotes_inputs=input_list,
								# promotes_outputs=['power_forward', 'Rotor|Thrust', 'Rotor|Ct', 'Aero|CL_cruise'])
								promotes_outputs=['*'])

		# --- Calculate energy consumption --- #
		# energy = power_hover * hover_time + power_cruise * cruise_time
		energy_comp = om.ExecComp('energy_cnsmp = (power_hover * hover_time) + (power_forward * flight_distance / speed)',
								   energy_cnsmp={'units': 'W * s'},
								   power_hover={'units': 'W'},
								   power_forward={'units': 'W'},
								   hover_time={'units': 's'},
								   flight_distance={'units': 'm'},
								   speed={'units': 'm/s'})
		self.add_subsystem('energy', energy_comp,
							promotes_inputs=['power_hover', 'power_forward', 'hover_time', 'flight_distance', ('speed', 'UAV|Speed')],
							promotes_outputs=['energy_cnsmp'])


		# --- Calculate disk loadings --- #
		# in hover
		disk_loading_comp_1 = om.ExecComp('disk_loading = thrust / (pi * r**2)',
										   disk_loading={'units': 'N/m**2'},
										   thrust={'units': 'N'},
										   r={'units': 'm'})
		self.add_subsystem('disk_loading_hover', disk_loading_comp_1,
							promotes_inputs=[('r', 'UAV|r_rotor_lift')],
							promotes_outputs=[('disk_loading', 'disk_loading_hover')])
		self.connect('power_hover.thrust_each', 'disk_loading_hover.thrust')

		# in cruise
		disk_loading_comp_2 = om.ExecComp('disk_loading = thrust / (pi * r**2)',
										   disk_loading={'units': 'N/m**2'},
										   thrust={'units': 'N'},
										   r={'units': 'm'})
		self.add_subsystem('disk_loading_cruise', disk_loading_comp_2,
							promotes_outputs=[('disk_loading', 'disk_loading_cruise')])
		if uav_config == 'multirotor':
			self.promotes('disk_loading_cruise', inputs=[('r', 'UAV|r_rotor_lift')])
		elif uav_config == 'lift+cruise':
			self.promotes('disk_loading_cruise', inputs=[('r', 'UAV|r_rotor_cruise')])
		self.connect('Rotor|Thrust', 'disk_loading_cruise.thrust')


		# --- Add nonlinear solvers for implicit equations --- #
		self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True, maxiter=30, iprint=0, rtol=1e-10)
		self.nonlinear_solver.options['err_on_non_converge'] = True
		self.nonlinear_solver.options['reraise_child_analysiserror'] = True
		self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
		self.nonlinear_solver.linesearch.options['maxiter'] = 10
		self.nonlinear_solver.linesearch.options['iprint'] = 0
		self.linear_solver = om.DirectSolver(assemble_jac=True)













