import numpy as np

from talos.utils.options_dictionary import OptionsDictionary
from talos.specifications.attitude_actuator_spec import AttitudeActuatorSpec


class AttitudeSpec(OptionsDictionary):

    def initialize(self):
        # TODO: type of orientation depends on
        self.declare('initial_orientation_euler', types=np.ndarray)
        self.declare('initial_orientation_mtx', types=np.ndarray)
        self.declare('initial_orientation_q', types=np.ndarray)
        self.declare('model_reaction_wheels', types=bool)
        # make Euler angles over time design variables
        self.declare('design_attitude_profile', types=bool)
        # make torque inputs over time design variables; do not model
        # reaction wheels
        self.declare('design_torque_profile', types=bool)
        self.declare('gravity_gradient', types=bool)
        self.declare('actuator', types=AttitudeActuatorSpec)
