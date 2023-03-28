from csdl import Model
import csdl
from talos.disciplines.eps.solar.ivt import IVT
from talos.disciplines.eps.battery.battery_pack import BatteryPack


class ElectricalPowerSystem(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']
        # TODO: define IVT
        self.add(IVT())

        # NOTE: BatteryPack requires power_draw from other subsystems
        self.add(BatteryPack(num_times=num_times), name='battery_pack')
        self.connect('solar_power', 'power_supply')
