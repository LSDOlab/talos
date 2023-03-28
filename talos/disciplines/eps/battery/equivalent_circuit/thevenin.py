from csdl import Model
from ozone.api import ODEProblem


class TheveninVoltageDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('capacity', types=float)

    def define(self):
        n = self.parameters['num_nodes']
        capacity = self.parameters['capacity']

        # m, m/s
        thevenin_voltage = self.declare_variable('thevenin_voltage',
                                                 shape=(n, 1))
        polarization_resistance = self.declare_variable(
            'polarization_resistance', shape=(n, 1))
        equivalent_capacitance = self.declare_variable(
            'equivalent_capacitance', shape=(n, 1))
        current = self.declare_variable('current', shape=(n, 1))

        dv_dt = -thevenin_voltage / (
            polarization_resistance *
            equivalent_capacitance) + current / equivalent_capacitance
        self.register_output('dv_dt', dv_dt)


class TheveninVoltageIntegrator(ODEProblem):

    def setup(self):
        self.add_parameter('polarization_resistance',
                           dynamic=True,
                           shape=(self.num_times, 1))
        self.add_parameter('equivalent_capacitance',
                           dynamic=True,
                           shape=(self.num_times, 1))
        self.add_parameter('current', dynamic=True, shape=(self.num_times, 1))
        self.add_state(
            'thevenin_voltage',
            'dv_dt',
            shape=(1, ),
            initial_condition_name='th_0',
            output='thevenin_voltage',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(TheveninVoltageDynamics)
