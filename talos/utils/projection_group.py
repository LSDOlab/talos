from openmdao.api import Group

from lsdo_utils.api import ArrayExpansionComp, ArrayContractionComp, PowerCombinationComp


class ProjectionGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('in1_name', types=str)
        self.options.declare('in2_name', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        num_times = self.options['num_times']
        in1_name = self.options['in1_name']
        in2_name = self.options['in2_name']
        out_name = self.options['out_name']

        shape = (3, num_times)

        c = a * b**2

        comp = PowerCombinationComp(
            shape=shape,
            out_name='tmp_{}_multiplied'.format(out_name),
            powers_dict={
                in1_name: 1.,
                in2_name: 2.,
            },
        )
        self.add('tmp_{}_multiplied_comp'.format(out_name),
                 comp,
                 promotes=['*'])

        comp = ArrayContractionComp(
            shape=shape,
            contract_indices=[0],
            out_name='tmp_{}_summed'.format(out_name),
            in_name='tmp_{}_multiplied'.format(out_name),
        )
        self.add('{}_summed_comp'.format(out_name), comp, promotes=['*'])

        comp = ArrayExpansionComp(
            shape=shape,
            expand_indices=[0],
            in_name='tmp_{}_summed'.format(out_name),
            out_name=out_name,
        )
        self.add('{}_comp'.format(out_name), comp, promotes=['*'])
