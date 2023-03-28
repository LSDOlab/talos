import numpy as np
from talos.csdl_future.surrogate_models.rmtb import RMTB


class SolarPanelVoltage(RMTB):

    def override_tuning_parameters(self):
        self.order = (3, 3, 3)
        self.num_ctrl_pts = (6, 6, 15)
        self.energy_weight = 1e-15
        self.regularization_weight = 0.0

    def define_training_inputs(self):
        # build surrogate model
        home = '/Users/victor/'
        self.dat = np.genfromtxt(
            home + 'packages/talos/talos/eps/solar/cadre_iv_curve.dat',
            delimiter='\n')
        nT, nA, nI = self.dat[:3]
        nT = int(nT)
        nA = int(nA)
        nI = int(nI)
        T = self.dat[3:3 + nT]
        A = self.dat[3 + nT:3 + nT + nA]
        I = self.dat[3 + nT + nA:3 + nT + nA + nI]

        self.training_inputs['temperature'] = T
        self.training_inputs['sunlit_area'] = A
        self.training_inputs['current'] = I

    def define_training_outputs(self):
        nT, nA, nI = self.dat[:3]
        nT = int(nT)
        nA = int(nA)
        nI = int(nI)
        V = self.dat[3 + nT + nA + nI:].reshape((nT, nA, nI), order='F')

        self.training_outputs['voltage'] = V
