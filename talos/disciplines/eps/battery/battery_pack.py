from csdl import Model, NewtonSolver, ScipyKrylov, NonlinearBlockGS
import csdl

from talos.disciplines.eps.battery.battery_cell import BatteryCell


class BatteryPackSolver(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']

        num_series = self.declare_variable('num_series')
        num_parallel = self.declare_variable('num_parallel')

        current = self.declare_variable('current', shape=(num_times, 1))
        cell_current = current / csdl.expand(num_parallel, (num_times, 1))
        self.register_output('cell_current', cell_current)

        self.connect('cell_current', 'cell.current')
        self.add(
            BatteryCell(num_times=num_times),
            name='cell',
            promotes=[
                'state_of_charge',
                'temperature',
                # 'open_circuit_voltage',
                # 'internal_resistance',
                # 'polarization_resistance',
                # 'equivalent_capacitance',
                # 'thevenin_voltage',
            ],
        )

        cell_voltage = self.declare_variable('cell_voltage',
                                             shape=(num_times, 1))
        self.connect('cell.voltage', 'cell_voltage')

        voltage = cell_voltage * csdl.expand(num_series, (num_times, 1))
        battery_power = current * voltage
        self.register_output('voltage', voltage)
        self.register_output('battery_power', battery_power)

        # residual
        power_supply = self.declare_variable('power_supply',
                                             shape=(num_times, 1))
        power_draw = self.declare_variable('power_draw', shape=(num_times, 1))
        self.register_output('net_power',
                             battery_power - power_supply + power_draw)

        expose(self, 'state_of_charge', (num_times, 1))
        expose(self, 'open_circuit_voltage', (num_times, 1))
        expose(self, 'internal_resistance', (num_times, 1))
        expose(self, 'polarization_resistance', (num_times, 1))
        expose(self, 'equivalent_capacitance', (num_times, 1))


def expose(model, name, shape):
    # KLUDGE: exposed outputs must not only be promoted to this
    # level, they must also be registered at this level; here, we
    # create new variables to expose
    state_of_charge = model.declare_variable(name, shape=shape)
    model.register_output(f'double_{name}', 2 * state_of_charge)


# TODO: solve for current using iterative solver with power as input,
# cell current as input, and pack current as exposed variable
class BatteryPack(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']

        num_series = self.create_input('num_series', val=1)
        num_parallel = self.create_input('num_parallel', val=6)
        self.add_design_variable('num_series', lower=1)
        self.add_design_variable('num_parallel', lower=1)

        power_supply = self.declare_variable('power_supply',
                                             shape=(num_times, 1))
        power_draw = self.declare_variable('power_draw', shape=(num_times, 1))

        # TODO: OpenMDAO: No matches were found for wrt="num_parallel"
        # TODO: This implicit operation contains 2 ODEs; when using a
        # Newton solver, this is going to be very slow.
        s = self.create_implicit_operation(
            BatteryPackSolver(num_times=num_times))
        # s.nonlinear_solver = NewtonSolver()
        s.nonlinear_solver = NonlinearBlockGS()
        s.linear_solver = ScipyKrylov()
        s.declare_state('current', residual='net_power')
        (
            current,
            cell_current,
            voltage,
            battery_power,
            # KLUDGE: need to register, not only promote variables
            # to expose
            double_state_of_charge,
            double_open_circuit_voltage,
            double_internal_resistance,
            double_polarization_resistance,
            double_equivalent_capacitance,
        ) = s.apply(
            num_series,
            num_parallel,
            power_supply,
            power_draw,
            expose=[
                'cell_current',
                'voltage',
                'battery_power',
                # KLUDGE: need to register, not only promote variables
                # to expose
                'double_state_of_charge',
                'double_open_circuit_voltage',
                'double_internal_resistance',
                'double_polarization_resistance',
                'double_equivalent_capacitance',
            ],
        )
        # KLUDGE: undo multiplication by 2 performed in implicit
        # operation so that these variables are registered in implicit
        # operation
        self.register_output('state_of_charge', double_state_of_charge / 2)
        self.register_output('open_circuit_voltage',
                             double_open_circuit_voltage / 2)
        self.register_output('internal_resistance',
                             double_internal_resistance / 2)
        self.register_output('polarization_resistance',
                             double_polarization_resistance / 2)
        self.register_output('equivalent_capacitance',
                             double_equivalent_capacitance / 2)


if __name__ == '__main__':
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    import numpy as np

    n = 100
    h = 0.19
    A = 1.

    rep = GraphRepresentation(BatteryPack(num_times=n))

    sim = Simulator(rep)

    # TODO: define time step size
    # sim['h'] = np.arange(n)*h
    sim['power_supply'] = A * np.sin(np.arange(n) * h)
    sim['power_draw'] = A * np.cos(np.arange(n) * h)

    sim.run()
    print(sim['state_of_charge'])
    print(sim['current'])
    print(sim['cell_current'])
    print(sim['voltage'])
