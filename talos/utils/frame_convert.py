import numpy as numpy
from openmdao.api import ExplicitComponent


class Earth_Quaternion(ExplicitComponent):
    """
    Return the Earth Quaternion as a function of time.
    """
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']
        # Inputs
        self.add_input('t', np.zeros(num_times), units='s', desc='Time')

        # Outputs
        self.add_output(
            'q_E',
            np.zeros((4, num_times)),
            units=None,
            desc='Quarternion matrix in Earth-fixed frame over time')

        self.J = np.zeros((num_times, 4))

#        self.declare_partials('*','*')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        num_times = self.options['num_times']

        q_E = outputs['q_E']

        fact = np.pi / 3600.0 / 24.0
        theta = fact * inputs['t']
        print('theta', theta)

        outputs['q_E'][0, :] = np.cos(theta)
        outputs['q_E'][3, :] = -np.sin(theta)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        num_times = self.options['num_times']

        self.dq_dt = np.zeros((num_times, 4))

        fact = np.pi / 3600.0 / 24.0
        theta = fact * inputs['t']

        self.dq_dt[:, 0] = -np.sin(theta) * fact
        self.dq_dt[:, 3] = -np.cos(theta) * fact


#        self.J = dq_dt

if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3
    comp.add_output('t', val=np.array([3600 * 2, 3600 * 4, 3600 * 6]))

    group.add('Inputcomp', comp, promotes=['*'])
    group.add('frameconvert',
              Earth_Quaternion(num_times=num_times),
              promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()
    print(prob['t'])
    print(prob['q_E'])

    prob.check_partials(compact_print=True)
