import numpy as np

from talos.utils.options_dictionary import OptionsDictionary
"""
Independent variables in communication group:
Ground Station coordinates default to set in UCSD


"""


class Groundcomm(OptionsDictionary):

    def initialize(self):
        self.declare('lon', default=32.8563, types=float)
        self.declare('lat', default=-117.2500, types=float)
        self.declare('alt', default=0.4849368, types=float)
        self.declare('P_comm', default=15.0, types=float)
        self.declare('Gain',
                     default=np.random.randint(1.76, 40, size=num_times),
                     types=float)
