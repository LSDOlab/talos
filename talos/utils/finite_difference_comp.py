import numpy as np
import scipy.sparse

from csdl import CustomExplicitOperation


class FiniteDifferenceComp(CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('in_name', types=str)
        self.parameters.declare('out_name', types=str)

    def define(self):
        num_times = self.parameters['num_times']
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        self.add_input(in_name, shape=num_times)
        self.add_output(out_name, shape=num_times)

        indices = np.arange(num_times)

        data = np.zeros((num_times, 2))
        rows = np.zeros((num_times, 2), int)
        cols = np.zeros((num_times, 2), int)

        data[:, 0] = -1.
        rows[:, 0] = indices
        cols[1:, 0] = indices[:-1]
        cols[0, 0] = indices[0]

        data[:, 1] = 1.
        rows[:, 1] = indices
        cols[:-1, 1] = indices[1:]
        cols[-1, 1] = indices[-1]

        data = data.flatten()
        rows = rows.flatten()
        cols = cols.flatten()

        self.mtx = scipy.sparse.csr_matrix((data, (rows, cols)),
                                           shape=(num_times, num_times))

        self.declare_derivatives(out_name,
                                 in_name,
                                 val=data,
                                 rows=rows,
                                 cols=cols)

    def compute(self, inputs, outputs):
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        outputs[out_name] = self.mtx.dot(inputs[in_name])


if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp

    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('in', val=np.random.rand(num_times))
    prob.model.add('inputs_comp', comp, promotes=['*'])

    comp = FiniteDifferenceComp(
        num_times=num_times,
        in_name='in',
        out_name='out',
    )
    prob.model.add('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    print(prob['in'])
    print(prob['out'])
