from csdl import Model
import csdl

from talos.disciplines.eps.battery.equivalent_circuit.li_ion import LiIon
from talos.disciplines.eps.battery.equivalent_circuit.soc import SOC_Integrator
from talos.disciplines.eps.battery.equivalent_circuit.thevenin import TheveninVoltageIntegrator


class BatteryCell(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']
        current = self.declare_variable('current', shape=(num_times, 1))

        self.add(
            # ODEProblemTest('GaussLegendre4', 'collocation',
            SOC_Integrator('RK4', 'time-marching',
                           num_times).create_solver_model(
                               ODE_parameters={'capacity': LiIon.capacity}, ),
            name='soc_integrator',
        )
        state_of_charge = self.declare_variable('state_of_charge',
                                                shape=(num_times, 1))

        temperature = self.declare_variable('temperature',
                                            shape=(num_times, 1))
        open_circuit_voltage, internal_resistance, polarization_resistance, equivalent_capacitance = csdl.custom(
            temperature,
            state_of_charge,
            op=LiIon(shape=(num_times, 1), ).create_op(),
        )
        self.register_output('open_circuit_voltage', open_circuit_voltage)
        self.register_output('internal_resistance', internal_resistance)
        self.register_output('polarization_resistance',
                             polarization_resistance)
        self.register_output('equivalent_capacitance', equivalent_capacitance)

        self.create_input('s_0', val=0.95)
        self.create_input('th_0', val=0.)

        self.add(
            # ODEProblemTest('GaussLegendre4', 'collocation',
            TheveninVoltageIntegrator(
                'RK4', 'time-marching', num_times).create_solver_model(
                    ODE_parameters={'capacity': LiIon.capacity}, ),
            name='thevenin_integrator',
        )

        thevenin_voltage = self.declare_variable('thevenin_voltage',
                                                 shape=(num_times, 1))
        voltage = open_circuit_voltage - thevenin_voltage - current * internal_resistance
        self.register_output('voltage', voltage)

        # TODO: temperature ODE (X-57 paper); only if efficiency < 1
