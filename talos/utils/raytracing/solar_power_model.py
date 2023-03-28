import numpy as np
from openmdao.api import ExplicitComponent

from lsdo_utils.miscellaneous_functions import structure_data
from smt_exposure import smt_exposure


class SolarPowerModel(ExplicitComponent):
    """
    Generic model for computing solar power as a function of azimuth and
    elevation (roll and pitch)
    """
    def initialize(self):
        self.options.declare('times', types=int)

    def setup(self):
        n = self.options['times']
        self.add_input('roll', shape=(n))
        self.add_input('pitch', shape=(n))
        self.add_output('sunlit_area', shape=(n))
        self.declare_partials(
            'sunlit_area',
            'roll',
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            'sunlit_area',
            'pitch',
            rows=np.arange(n),
            cols=np.arange(n),
        )

        # load training data
        az = np.genfromtxt('arrow_xData.csv', delimiter=',')
        el = np.genfromtxt('arrow_yData.csv', delimiter=',')
        yt = np.genfromtxt('arrow_zData.csv', delimiter=',')

        # generate surrogate model with 20 training points
        # must be the same as the number of points used to create model
        self.sm = smt_exposure(20, az, el, yt)

    def compute(self, inputs, outputs):
        n = self.options['times']
        r = inputs['roll']
        p = inputs['pitch']
        for i in range(n):
            rp = np.array([r[i], p[i]]).reshape((1, 2))
            outputs['sunlit_area'][i] = self.sm.predict_values(rp)

    def compute_partials(self, inputs, partials):
        n = self.options['times']
        r = inputs['roll']
        p = inputs['pitch']
        for i in range(n):
            rp = np.array([r[i], p[i]]).reshape((1, 2))
            partials['sunlit_area',
                     'roll'][i] = self.sm.predict_derivatives(
                         rp,
                         0,
                     )
            partials['sunlit_area',
                     'pitch'][i] = self.sm.predict_derivatives(
                         rp,
                         1,
                     )


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    np.random.seed(0)

    times = 200

    # check partials
    ivc = IndepVarComp()
    ivc.add_output('roll', val=np.random.rand(times))
    ivc.add_output('pitch', val=np.random.rand(times))
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem(
        'ivc',
        ivc,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'spm',
        SolarPowerModel(times=times),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
