# uavsizing
This repository contains my learning experience in sizing a small UAV.
The total weight of the UAV is composed of the following weights:
1) Payload
2) Battery
3) Rotor
4) Motor + ESC
5) Wing
6) Frame

Two configurations are considered: 1) Wingless multirotor, 2) Lift+Cruise.
The sizing process depends on the configuration to be used.

The battery weight is estimated by calculating the energy required to fly the mission.
The mission requirement includes the payload weight, range, and hover time.
The power models are derived using simple aerodynamics (momentum theory and polar drag) during hover and cruise.
The rotor, motor, ESC, wing, and frame weights are estimated through empirical equations.
After all component weights are defined, a simple bisection method is utilized for the total weight iteratively.
Moreover, optimization for each configuration is also carried out using gradient-based methods implemented in OpenMDAO.
The above methods are implemented in Python as well as an Excel file for easy demonstration.
Finally, all the theories and equations are summarized in the PPT file.

Acknowledgment: this work is greatly inspired by https://github.com/kanekosh/eVTOL_sizing
