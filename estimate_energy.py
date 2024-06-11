import numpy as np
import matplotlib.pyplot as plt

import openmdao.api as om
from uavsizing.energy import EnergyConsumption

if __name__ == '__main__':

	# --- UAV 1: Wingless Multirotor --- #
	# General parameters
	uav1_params = {}
	uav1_params['uav_config'] = 'multirotor'
	uav1_params['n_rotors_lift'] = 6
	uav1_params['rotor_lift_solidity'] = 0.13
	uav1_params['hover_FM'] = 0.75
	# Battery parameters
	uav1_params['battery_rho'] = 158.0	# Wh/kg
	uav1_params['battery_eff'] = 0.85
	uav1_params['battery_max_discharge'] = 0.7
	# Design parameters
	uav1_weight = 5.0 # kg
	uav1_r_rotor_lift = 0.2 # m
	uav1_speed = 10.0 # m/s
	uav1_rotor_mu = 0.3

	# --- UAV 2: Lift+Cruise --- #
	# General parameters
	uav2_params = {}
	uav2_params['uav_config'] = 'lift+cruise'
	uav2_params['n_rotors_lift'] = 4
	uav2_params['n_rotors_cruise'] = 2
	uav2_params['rotor_lift_solidity'] = 0.13
	uav2_params['rotor_cruise_solidity'] = 0.13
	uav2_params['hover_FM'] = 0.75
	# Wing parameters
	uav2_params['Cd0'] = 0.0397
	uav2_params['wing_AR'] = 6.0
	uav2_params['wing_e'] = 0.8
	# Battery parameters
	uav2_params['battery_rho'] = 158.0	# Wh/kg
	uav2_params['battery_eff'] = 0.85
	uav2_params['battery_max_discharge'] = 0.7
	# Design parameters
	uav2_weight = 5.0 # kg
	uav2_r_rotor_lift = 0.2 # m
	uav2_r_rotor_cruise = 0.2 # m
	uav2_speed = 20.0 # m/s
	uav2_wing_area = 0.2 # m**2
	uav2_rotor_J = 1.0

	# --- Mission requirements --- #
	n_missions = 15
	flight_ranges = np.linspace(100.0, 5000.0, n_missions) # m
	hover_times = n_missions * [240.0] # s

	# --- 1) Energy consumption for Wingless Multirotor --- #
	energy_list1 = np.zeros(n_missions)

	for i in range(n_missions):
		prob1 = om.Problem()
		indeps = prob1.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
		indeps.add_output('flight_distance', flight_ranges[i], units='m')
		indeps.add_output('hover_time', hover_times[i], units='s')
		indeps.add_output('UAV|W_total', uav1_weight, units='kg')
		indeps.add_output('UAV|Speed', uav1_speed, units='m/s')
		indeps.add_output('UAV|r_rotor_lift', uav1_r_rotor_lift, units='m')
		indeps.add_output('Rotor|mu', uav1_rotor_mu, units=None)

		prob1.model.add_subsystem('energy_model',
								   EnergyConsumption(UAV_options=uav1_params),
								   promotes_inputs=['*'],
								   promotes_outputs=['*'])
		prob1.setup(check=False)
		prob1.run_model()
		energy_list1[i] = prob1.get_val('energy_cnsmp', 'W*h')[0]

	# --- 2) Energy consumption for Lift+Cruise --- #
	energy_list2 = np.zeros(n_missions)

	for i in range(n_missions):
		prob1 = om.Problem()
		indeps = prob1.model.add_subsystem('indeps', om.IndepVarComp(), promotes=['*'])
		indeps.add_output('flight_distance', flight_ranges[i], units='m')
		indeps.add_output('hover_time', hover_times[i], units='s')
		indeps.add_output('UAV|W_total', uav2_weight, units='kg')
		indeps.add_output('UAV|Speed', uav2_speed, units='m/s')
		indeps.add_output('UAV|r_rotor_lift', uav2_r_rotor_lift, units='m')
		indeps.add_output('UAV|r_rotor_cruise', uav2_r_rotor_cruise, units='m')
		indeps.add_output('UAV|S_wing', uav2_wing_area, units='m**2')
		indeps.add_output('Rotor|J', uav2_rotor_J, units=None)

		prob1.model.add_subsystem('energy_model',
								   EnergyConsumption(UAV_options=uav2_params),
								   promotes_inputs=['*'],
								   promotes_outputs=['*'])
		prob1.setup(check=False)
		prob1.run_model()
		energy_list2[i] = prob1.get_val('energy_cnsmp', 'W*h')[0]


	# --- Plotting the results --- #
	plt.plot(flight_ranges/1000.0, energy_list1, label='Multirotor')
	plt.plot(flight_ranges/1000.0, energy_list2, label='Lift+Cruise')
	plt.xlabel('Flight range [km]')
	plt.ylabel('Energy Consumption [Wh]')
	plt.legend()
	plt.show()


