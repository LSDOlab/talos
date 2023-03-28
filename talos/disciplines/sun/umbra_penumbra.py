from csdl import Model
import csdl
from talos.csdl_future.mask import MaskLT, MaskLE, MaskGT, MaskGE


class UmbraPenumbra(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('R', types=float)
        self.parameters.declare('alfa', types=float, default=0.9)
        self.parameters.declare('position_name', types=str)
        self.parameters.declare('light_source_name', types=str)
        self.parameters.declare('line_of_sight_name', types=str)

    def define(self):
        num_times = self.parameters['num_times']
        R = self.parameters['R']
        alfa = self.parameters['alfa']
        position_name = self.parameters['position_name']
        light_source_name = self.parameters['light_source_name']
        line_of_sight_name = self.parameters['line_of_sight_name']

        position = self.declare_variable(
            position_name,
            shape=(num_times, 3),
        )
        light_source_direction = self.declare_variable(
            light_source_name,
            shape=(num_times, 3),
        )

        dot = csdl.dot(position, light_source_direction, axis=1)
        self.register_output('dot', dot)

        # distance from earth center projected onto plane normal to sun
        # direction

        # TODO: cannot be zero!
        ds = csdl.pnorm(
            csdl.cross(position, light_source_direction, axis=1),
            axis=1,
        )
        self.register_output('ds', ds)

        dot_lt_0 = csdl.custom(
            dot,
            op=MaskLT(
                num_times=num_times,
                in_name='dot',
                out_name='dot_lt_0',
                threshold=0,
            ),
        )
        dot_ge_0 = csdl.custom(
            dot,
            op=MaskGE(
                num_times=num_times,
                in_name='dot',
                out_name='dot_ge_0',
                threshold=0,
            ),
        )
        ds_gt_R = csdl.custom(
            ds,
            op=MaskGT(
                num_times=num_times,
                in_name='ds',
                out_name='ds_gt_R',
                threshold=R,
            ),
        )
        ds_le_R = csdl.custom(
            ds,
            op=MaskLE(
                num_times=num_times,
                in_name='ds',
                out_name='ds_le_R',
                threshold=R,
            ),
        )
        ds_ge_aR = csdl.custom(
            ds,
            op=MaskGE(
                num_times=num_times,
                in_name='ds',
                out_name='ds_ge_aR',
                threshold=alfa * R,
            ),
        )
        self.register_output('dot_lt_0', dot_lt_0)
        self.register_output('dot_ge_0', dot_ge_0)
        self.register_output('ds_gt_R', ds_gt_R)
        self.register_output('ds_le_R', ds_le_R)
        self.register_output('ds_ge_aR', ds_ge_aR)
        ds_trans = ds_le_R * ds_ge_aR * ds

        # capture umbra and penumbra effect between light and shadow
        eta = (ds_trans - alfa * R) / (R - alfa * R)
        umbra_penumbra_transition = 3 * eta**2 - 2 * eta**3

        los = dot_ge_0 + (ds_gt_R + umbra_penumbra_transition) * dot_lt_0

        self.register_output(line_of_sight_name, los)


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    n = 200
    R = 5
    ds = 1.1 * R * np.arange(n) / n
    print(ds)
    alfa = 0.9

    # capture umbra and penumbra effect between light and shadow
    umbra_penumbra_indices = np.where(ds < R)[0]
    eta = np.zeros(n)
    eta[umbra_penumbra_indices] = (ds[umbra_penumbra_indices] -
                                   alfa * R) / (R - alfa * R)
    umbra_penumbra_indices = np.where(ds < alfa * R)[0]
    eta[umbra_penumbra_indices] = 0
    print(eta)
    umbra_penumbra_transition = 3 * eta**2 - 2 * eta**3
    print(umbra_penumbra_transition)

    # ds < alfa * R => umbra
    # alfa * R < ds < R => penumbra
    umbra_penumbra_indicator = np.zeros(n)
    umbra_penumbra_indices = np.where(ds > R)
    umbra_penumbra_indicator[umbra_penumbra_indices[0]] = 1
    umbra_penumbra_indicator += umbra_penumbra_transition
    # umbra_penumbra_indices = np.where(ds > R)
    # umbra_penumbra_indicator[umbra_penumbra_indices[0]] = 0
    # umbra_penumbra_indices = np.where(ds > alfa * R)[0]
    # umbra_penumbra_indicator[umbra_penumbra_indices] *= umbra_penumbra_transition[umbra_penumbra_indices]
    # umbra_penumbra_indices = np.where(ds > R)
    # umbra_penumbra_indicator[umbra_penumbra_indices[0]] = 1
    print(umbra_penumbra_indicator)

    los = umbra_penumbra_indicator
    print(los)

    plt.plot(los, '.')
    plt.show()
# if __name__ == "__main__":
#     from csdl import GraphRepresentation
#     from python_csdl_backend import Simulator
#     from talos.specifications.swarm_spec import SwarmSpec
#     import numpy as np
#     from talos.constants import RADII

#     Re = RADII['Earth']

#     num_times = 30
#     rep = GraphRepresentation(
#         UmbraPenumbra(
#             num_times=num_times,
#             R=Re,
#             position_name='position_km',
#             light_source_name='sun_direction',
#             line_of_sight_name='sun_LOS',
#         ))
#     sim = Simulator(rep)
#     sim['position_km'] = 7000 * np.random.rand(np.prod(
#         (num_times, 3))).reshape((num_times, 3))
#     sun_direction = np.zeros((num_times, 3))
#     sun_direction[:, 0] = np.ones((num_times, ))
#     sim['sun_direction'] = sun_direction
#     sim.run()
#     sim.check_partials(compact_print=True)
