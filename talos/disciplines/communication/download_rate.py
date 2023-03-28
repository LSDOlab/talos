"""
RK4 component for Data Download
"""

from ozone.api import ODEProblem
from csdl import Model
import numpy as np

class DownloadRate(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        num_nodes = self.parameters['num_nodes']
        download_rate = self.declare_variable(
            'download_rate',
            shape=(num_nodes, ),
        )
        data = self.create_input(
            'data',
            shape=(num_nodes, ),
        )
        # KLUDGE: ozone doesn't support xdot = x = u
        dr = (2*download_rate)/2
        self.register_output('dr', dr)

class ODEProblemTest(ODEProblem):

    def setup(self):
        self.add_parameter('download_rate',
                           dynamic=True,
                           shape=(self.num_times, ))
        self.add_state(
            'data',
            'dr',
            shape=(1, ),
            initial_condition_name='initial_data',
            output='data',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(DownloadRate)


