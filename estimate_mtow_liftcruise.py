import numpy as np
import openmdao.api as om

from uavsizing.weights import MTOWEstimation

if __name__ == '__main__':
	
	# =========================================== #
	# UAV parameters
	uav_params = {}
	uav_params['uav_config'] = 'lift+cruise'

	# Rotor parameters
	uav_params['n_rotors_lift'] = 4
	uav_params['n_rotors_cruise'] = 2
	uav_params['rotor_lift_solidity'] = 0.13
	uav_params['rotor_cruise_solidity'] = 0.13
	uav_params['hover_FM'] = 0.75

	# Wing parameters
	uav_params['Cd0'] = 0.0397 # Bacchini et al, 2021
	uav_params['wing_AR'] = 6.0
	uav_params['wing_e'] = 0.8

	# Battery parameters
	uav_params['battery_rho'] = 158 # Wh/kg
	uav_params['battery_eff'] = 0.85
	uav_params['battery_max_discharge'] = 0.7

	# Mission requirements
	payload_weight = 2.0 # kg
	flight_range = 5000.0 # m
	hover_time = 240.0 # s
	# =========================================== #

	# --- Setting up as an OpenMDAO problem --- #

	prob = om.Problem()

	# Define model inputs
	indeps = prob.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])

	# Mission requirements
	indeps.add_output('payload_weight', payload_weight, units='kg')
	indeps.add_output('flight_distance', flight_range, units='m')
	indeps.add_output('hover_time', hover_time, units='s')

	# Design variables (and their initial gueses)
	indeps.add_output('UAV|W_total', 10.0, units='kg')
	indeps.add_output('UAV|Speed', 25.0, units='m/s')
	indeps.add_output('UAV|r_rotor_lift', 0.2, units='m')
	indeps.add_output('UAV|r_rotor_cruise', 0.2, units='m')
	indeps.add_output('Rotor|J', 1.0, units=None)
	indeps.add_output('UAV|S_wing', 0.2, units='m**2')

	# UAV MTOW Estimation model
	prob.model.add_subsystem('UAV_weight_model',
							  MTOWEstimation(UAV_options=uav_params, use_solver=False),
							  promotes_inputs=['*'],
							  promotes_outputs=['*'])

	# --- Optimization problem --- #
	# Design variables and their lower/upper bounds
	prob.model.add_design_var('UAV|W_total', lower=0.1, upper=20.0, ref=10.0, units='kg')
	prob.model.add_design_var('UAV|Speed', lower=2.0, upper=30.0, ref=10.0, units='m/s')
	prob.model.add_design_var('UAV|r_rotor_lift', lower=0.05, upper=0.25, ref=0.1, units='m')
	prob.model.add_design_var('UAV|r_rotor_cruise', lower=0.05, upper=0.25, ref=0.1, units='m')
	prob.model.add_design_var('Rotor|J', lower=0.01, upper=1.3)
	prob.model.add_design_var('UAV|S_wing', lower=0.01, upper=1.0, units='m**2')

	# Constraints
	prob.model.add_constraint('W_residual', lower=0.0, upper=0.0, ref=10.0)
	prob.model.add_constraint('disk_loading_hover', upper=250.0, ref=100.0, units='N/m**2')
	prob.model.add_constraint('disk_loading_cruise', upper=250.0, ref=100.0, units='N/m**2')
	# in cruise. CT / solidity <= 0.14 to avoid too high blade loading
	prob.model.add_constraint('Rotor|Ct', lower=0.0, upper=0.14 * uav_params['rotor_cruise_solidity'], ref=0.01)
	# CL_max at cruise = 0.6
	prob.model.add_constraint('Aero|CL_cruise', lower=0.0, upper=0.6, ref=0.5)

	# Objective
	prob.model.add_objective('UAV|W_total', ref=10.0)

	# Optimizer settings
	prob.driver = om.ScipyOptimizeDriver()
	prob.driver.options['optimizer'] = 'SLSQP'
	prob.driver.options['tol'] = 1e-8
	prob.driver.options['disp'] = True

	# Run optimization
	prob.setup(check=True)
	prob.run_driver()
	# prob.run_model()

	# get weights
	W_total = prob.get_val('UAV|W_total', 'kg')
	W_payload = prob.get_val('payload_weight', 'kg')
	W_battery = prob.get_val('UAV|W_battery', 'kg')
	W_rotors = prob.get_val('UAV|W_rotor_all', 'kg')
	W_motors = prob.get_val('UAV|W_motor_all', 'kg')
	W_ESCs = prob.get_val('UAV|W_ESC_all', 'kg')
	W_wing = prob.get_val('UAV|W_wing', 'kg')
	W_frame = W_total - (W_payload + W_battery + W_rotors + W_motors + W_ESCs + W_wing)

	# --- print results ---
	print('--------------------------------------------')
	print('--- problem settings ---')
	print('  UAV parameters echo:', uav_params)
	print('  payload weight [kg]:', list(W_payload))
	print('  flight range [m]   :', list(prob.get_val('flight_distance', 'm')))
	print('  hovering time [s]  :', list(prob.get_val('hover_time', 's')))
	print('\n--- design optimization results ---')
	print('Design variables')
	print('  lifting rotor radius [m] :', list(prob.get_val('UAV|r_rotor_lift', 'm')))
	print('  cruising rotor radius [m] :', list(prob.get_val('UAV|r_rotor_cruise', 'm')))
	print('  cruise speed [m/s]       :', list(prob.get_val('UAV|Speed', 'm/s')))
	print('  wing area [m**2] :', list(prob.get_val('UAV|S_wing', 'm**2')))
	print('  prop advance ratio J:', list(prob.get_val('Rotor|J')))
	print('Component weights [kg]')
	print('  total weight :', list(W_total))
	print('  payload      :', list(W_payload))
	print('  battery      :', list(W_battery))
	print('  rotors       :', list(W_rotors))
	print('  motors 	  :', list(W_motors))
	print('  ESCs 		  :', list(W_ESCs))
	print('  wing         :', W_wing)
	print('  frame        :', W_frame)
	print('Performances')
	print('  power in hover: [W] :', list(prob.get_val('power_hover', 'W')))
	print('  power in cruise: [W]:', list(prob.get_val('power_forward', 'W')))
	print('  CL in cruise:', list(prob.get_val('Aero|CL_cruise')))
	print('Sanity check: W_residual [kg]:', list(prob.get_val('W_residual', 'kg')), ' = 0?')
	print('--------------------------------------------')


