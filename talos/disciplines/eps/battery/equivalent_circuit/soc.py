from csdl import Model
from ozone.api import ODEProblem


class SOC_Dynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('capacity', types=float)

    def define(self):
        n = self.parameters['num_nodes']
        capacity = self.parameters['capacity']

        # m, m/s
        state_of_charge = self.create_input('state_of_charge', shape=(n, 1))

        current = self.declare_variable('current', shape=(n, 1))

        ds_dt = -current / capacity
        self.register_output('ds_dt', ds_dt)


class SOC_Integrator(ODEProblem):

    def setup(self):
        self.add_parameter('current', dynamic=True, shape=(self.num_times, 1))
        self.add_state(
            'state_of_charge',
            'ds_dt',
            shape=(1, ),
            initial_condition_name='s_0',
            output='state_of_charge',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(SOC_Dynamics)
