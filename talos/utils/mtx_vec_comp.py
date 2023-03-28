from __future__ import print_function
import numpy as np

from openmdao.api import ExplicitComponent

from talos.utils.utils import get_array_indices


class MtxVecComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('mtx_name', types=str)
        self.options.declare('vec_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        num_times = self.options['num_times']
        mtx_name = self.options['mtx_name']
        vec_name = self.options['vec_name']
        out_name = self.options['out_name']

        self.add_input(mtx_name, shape=(3, 3, num_times))
        self.add_input(vec_name, shape=(3, num_times))
        self.add_output(out_name, shape=(3, num_times))

        ones_3 = np.ones(3, int)
        mtx_indices = get_array_indices(*(3, 3, num_times))
        vec_indices = get_array_indices(*(3, num_times))

        rows = np.einsum('in,j->ijn', vec_indices, ones_3).flatten()
        cols = mtx_indices.flatten()
        self.declare_partials(out_name, mtx_name, rows=rows, cols=cols)

        rows = np.einsum('in,j->ijn', vec_indices, ones_3).flatten()
        cols = np.einsum('jn,i->ijn', vec_indices, ones_3).flatten()
        self.declare_partials(out_name, vec_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        mtx_name = self.options['mtx_name']
        vec_name = self.options['vec_name']
        out_name = self.options['out_name']

        outputs[out_name] = np.einsum('ijn,jn->in', inputs[mtx_name],
                                      inputs[vec_name])

    def compute_partials(self, inputs, partials):
        mtx_name = self.options['mtx_name']
        vec_name = self.options['vec_name']
        out_name = self.options['out_name']

        partials[out_name, mtx_name] = np.einsum('jn,i->ijn', inputs[vec_name],
                                                 np.ones(3)).flatten()
        partials[out_name, vec_name] = inputs[mtx_name].flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 4

    prob = Problem()
    comp = IndepVarComp()
    comp.add_output('mtx', np.random.rand(3, 3, num_times))

    comp.add_output('vec', np.random.rand(3, num_times))
    prob.model.add('inputs_comp', comp, promotes=['*'])

    comp = MtxVecComp(
        num_times=num_times,
        mtx_name='mtx',
        vec_name='vec',
        out_name='out',
    )
    prob.model.add('array_expansion', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.check_partials(compact_print=True)
    print(prob['mtx'])
    print(prob['vec'])
    print(prob['out'])
