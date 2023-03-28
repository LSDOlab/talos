from talos.csdl_future.surrogate_models.rmtb import RMTB
import numpy as np

# Data taken from:
# A comprehensive equivalent circuit model for lithium-ion batteries,
# incorporating the effects of state of health, state of charge, and
# temperature on model parameters


class LiIon(RMTB):
    # Capacity = 18.4 Ah
    # Volume = 7*160*227/1000**3 m**3
    # Mass = 0.496 kg
    # Voltage = 3.3 V
    capacity = 18.4
    volume = 7 * 160 * 227 / 1000**3
    mass = 0.496
    voltage = 3.3
    # TODO: find efficiency from paper
    efficiency = 1

    def override_tuning_parameters(self):
        self.order = 10
        self.num_ctrl_pts = 10
        self.energy_weight = 1e-15
        self.regularization_weight = 1e-6

    def define_training_inputs(self):
        self.training_inputs['temperature'] = np.array([5, 15, 25, 35, 45])
        self.training_inputs['state_of_charge'] = np.array(
            [10, 20, 30, 40, 50, 60, 70, 80, 90]) / 100.

    def define_training_outputs(self):
        self.training_outputs['open_circuit_voltage'] = np.array([
            3.24, 3.25, 3.25, 3.3, 3.3, 3.3, 3.3, 3.35, 3.35, 3.24, 3.25, 3.25,
            3.3, 3.3, 3.3, 3.3, 3.35, 3.35, 3.24, 3.25, 3.25, 3.3, 3.3, 3.3,
            3.3, 3.35, 3.35, 3.24, 3.25, 3.25, 3.3, 3.3, 3.3, 3.3, 3.35, 3.35,
            3.24, 3.25, 3.25, 3.3, 3.3, 3.3, 3.3, 3.35, 3.35
        ])
        self.training_outputs['internal_resistance'] = np.array([
            5910, 5810, 5710, 5610, 5520, 5440, 5360, 5280, 5170, 3490, 3440,
            3380, 3310, 3280, 3230, 3220, 3160, 3110, 2880, 2840, 2820, 2760,
            2760, 2720, 2700, 2660, 2610, 2466, 2445, 2412, 2347, 2337, 2321,
            2318, 2301, 2270, 2340, 2330, 2290, 2290, 2250, 2270, 2230, 2240,
            2230
        ]) / 1000.
        self.training_outputs['polarization_resistance'] = np.array([
            7850, 6360, 5390, 4480, 3830, 3680, 3770, 4370, 4070, 6220, 4740,
            3800, 3190, 3170, 2780, 3520, 3670, 2980, 5360, 3830, 3190, 2400,
            2430, 2080, 2810, 2400, 2150, 4613, 3136, 2711, 1886, 1913, 2011,
            2368, 2399, 1654, 4010, 3220, 1850, 2140, 1520, 1720, 1600, 1880,
            1900
        ]) / 1000.
        self.training_outputs['equivalent_capacitance'] = np.array([
            5075, 5981, 6665, 7207, 7781, 8293, 8476, 8025, 8942, 7823, 9129,
            10123, 10653, 11140, 11686, 11726, 10933, 12001, 10126, 11778,
            12992, 13689, 14049, 14361, 14406, 13780, 15140, 12573, 14746,
            16276, 17081, 17221, 17762, 17727, 16837, 18445, 14846, 17382,
            19457, 20348, 20685, 21595, 21373, 20064, 21977
        ])


if __name__ == '__main__':
    from csdl import Model, GraphRepresentation
    import csdl
    from python_csdl_backend import Simulator
    import matplotlib.pyplot as plt

    class M(Model):

        def initialize(self):
            self.parameters.declare('num_times', types=int)

        def define(self):
            num_times = self.parameters['num_times']

            temperature = self.declare_variable('temperature',
                                                shape=(num_times, ))
            state_of_charge = self.declare_variable('state_of_charge',
                                                    shape=(num_times, ))
            open_circuit_voltage, internal_resistance, polarization_resistance, equivalent_capacitance = csdl.custom(
                temperature,
                state_of_charge,
                op=LiIon(shape=(num_times, ), ).create_op(),
            )
            self.register_output('open_circuit_voltage', open_circuit_voltage)
            self.register_output('internal_resistance', internal_resistance)
            self.register_output('polarization_resistance',
                                 polarization_resistance)
            self.register_output('equivalent_capacitance',
                                 equivalent_capacitance)

    num_times = 400
    dummy = LiIon(shape=(num_times, ))
    ti = dummy.training_inputs
    to = dummy.training_outputs

    rep = GraphRepresentation(M(num_times=num_times))
    sim = Simulator(rep)
    np.random.seed(0)
    sim['temperature'] = np.random.rand(num_times) * 40 + 5
    sim['state_of_charge'] = np.random.rand(num_times)
    sim.run()
    T, z = np.meshgrid(ti['temperature'], ti['state_of_charge'], indexing='ij')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T.flatten(), z.flatten(), to['open_circuit_voltage'].flatten(),
               'x')
    ax.scatter(sim['temperature'], sim['state_of_charge'],
               sim['open_circuit_voltage'], 'o')
    ax.set_title('open_circuit_voltage')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T.flatten(), z.flatten(), to['internal_resistance'].flatten(),
               'x')
    ax.scatter(sim['temperature'], sim['state_of_charge'],
               sim['internal_resistance'], 'o')
    ax.set_title('internal_resistance')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T.flatten(), z.flatten(),
               to['polarization_resistance'].flatten(), 'x')
    ax.scatter(sim['temperature'], sim['state_of_charge'],
               sim['polarization_resistance'], 'o')
    ax.set_title('polarization_resistance')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(T.flatten(), z.flatten(),
               to['equivalent_capacitance'].flatten(), 'x')
    ax.scatter(sim['temperature'], sim['state_of_charge'],
               sim['equivalent_capacitance'], 'o')
    ax.set_title('equivalent_capacitance')
    plt.show()
