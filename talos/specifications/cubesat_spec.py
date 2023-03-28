import numpy as np

from talos.utils.options_dictionary import OptionsDictionary

from talos.specifications.attitude_spec import AttitudeSpec


class CubesatSpec(OptionsDictionary):

    def initialize(self):
        self.declare('name', types=str)
        self.declare('use_cp', types=bool, default=True)
        self.declare('attitude',
                     types=AttitudeSpec,
                     default=None,
                     allow_none=True)
        self.declare('dry_mass', types=float)
        self.declare('initial_orbit_state', types=np.ndarray)
        self.declare('specific_impulse', default=47., types=float)
        self.declare('perigee_altitude', default=500.1, types=float)
        self.declare('apogee_altitude', default=499.9, types=float)
        self.declare('RAAN', default=66.279, types=float)
        self.declare('inclination', default=82.072, types=float)
        # self.declare('inclination', default=97.4, types=float)
        self.declare('argument_of_periapsis', default=0., types=float)
        self.declare('true_anomaly', default=337.987, types=float)
        self.declare(
            'pthrust_scaler',
            default=1.,
            types=(float, np.ndarray),
        )
        self.declare(
            'pthrust_cp',
            types=np.ndarray,
        )
        self.declare(
            'nthrust_scaler',
            default=1.,
            types=(float, np.ndarray),
        )
        self.declare(
            'nthrust_cp',
            types=np.ndarray,
        )
        self.declare(
            'initial_propellant_mass',
            types=float,
        )
        self.declare(
            'initial_propellant_mass_scaler',
            types=(float, np.ndarray),
        )
