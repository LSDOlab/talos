import numpy as np

from talos.csdl_future.mask import *

from talos.constants import deg2arcsec, s
from csdl import Model
import csdl

from sys import float_info

# TODO: Use the "NOTE" comments in this file to design error messages
# where domain of finite derivatives is not the same as domain of valid
# inputs for individual functions; raise error if these functions are
# used and no runtime assertions provided; warn if runtime assertions
# are either too restrictive or guaranteed to pass; runtime assertions
# are required (1) for the compiler to verify that the user has
# communicated how the standard library functions are to be used, and
# (2) so that other users can be reminded of valid domains for inputs


class TelescopePointing(Model):

    def initialize(self):
        self.parameters.declare('virtual_telescope')
        self.parameters.declare('telescope_view_halfangle_tol_arcsec',
                                types=float)
        self.parameters.declare('telescope_view_angle_scaler',
                                default=1.,
                                types=(float, np.ndarray))

    def define(self):
        virtual_telescope = self.parameters['virtual_telescope']
        num_times = virtual_telescope['num_times']
        telescope_view_halfangle_tol_arcsec = self.parameters[
            'telescope_view_halfangle_tol_arcsec']
        telescope_view_angle_scaler = self.parameters[
            'telescope_view_angle_scaler']

        telescope_vector_component_in_sun_direction = self.declare_variable(
            'telescope_vector_component_in_sun_direction',
            shape=(num_times, 1),
        )
        separation_m = self.declare_variable(
            'separation_m',
            shape=(num_times, 1),
        )

        # Orientation of telescope during observation
        cos_view_angle = csdl.reshape(
            telescope_vector_component_in_sun_direction / separation_m,
            (num_times, ))
        self.register_output('cos_view_angle', cos_view_angle)

        # need to be able to take arccos safely, so only use values
        # where -1 < cos_view_angle < 1; derivatives are finite where
        # -1 < cos_view_angle < 1
        cos_view_angle_lt1 = csdl.custom(
            cos_view_angle,
            op=MaskLT(
                num_times=num_times,
                threshold=1.,
                in_name='cos_view_angle',
                out_name='cos_view_angle_lt1',
            ),
        )
        self.register_output('cos_view_angle_lt1', cos_view_angle_lt1)
        cos_view_angle_gt_neg1 = csdl.custom(
            cos_view_angle,
            op=MaskGT(
                num_times=num_times,
                threshold=-1.,
                in_name='cos_view_angle',
                out_name='cos_view_angle_gt_neg1',
            ))

        self.register_output('cos_view_angle_gt_neg1', cos_view_angle_gt_neg1)

        # These are used to (1) ensure derivatives of arccos can be
        # computed near -1 and 1, and (2) ensure that any values of
        # cos_view_angle that are outside of (-1, 1) interval have a
        # view angle (smallest representable nonzero value) that is
        # below the max view angle constraint value. The second case is
        # unlikely.
        cos_view_angle_ge1 = csdl.custom(
            cos_view_angle,
            op=MaskGE(
                num_times=num_times,
                threshold=1.,
                in_name='cos_view_angle',
                out_name='cos_view_angle_ge1',
            ),
        )
        self.register_output('cos_view_angle_ge1', cos_view_angle_ge1)

        cos_view_angle_le_neg1 = csdl.custom(
            cos_view_angle,
            op=MaskLE(
                num_times=num_times,
                threshold=-1.,
                in_name='cos_view_angle',
                out_name='cos_view_angle_le_neg1',
            ),
        )
        self.register_output(
            'cos_view_angle_le_neg1',
            cos_view_angle_le_neg1,
        )

        telescope_view_angle_unmasked = csdl.arccos(
            cos_view_angle * cos_view_angle_gt_neg1 * cos_view_angle_lt1)  # +
        # (cos_view_angle_ge1 - float_info.min) -
        # (cos_view_angle_le_neg1 - float_info.min))
        observation_phase_indicator = self.declare_variable(
            'observation_phase_indicator', shape=(num_times, ))
        telescope_view_angle = observation_phase_indicator * telescope_view_angle_unmasked

        self.register_output(
            'telescope_view_angle_unmasked',
            telescope_view_angle_unmasked,
        )
        self.register_output(
            'telescope_view_angle',
            telescope_view_angle,
        )
        max_telescope_view_angle = csdl.max(telescope_view_angle, rho=5000.)
        self.register_output(
            'max_telescope_view_angle',
            max_telescope_view_angle,
        )
        self.add_constraint(
            'max_telescope_view_angle',
            upper=telescope_view_halfangle_tol_arcsec / 3600 * np.pi / 180.,
            scaler=telescope_view_angle_scaler,
        )


if __name__ == "__main__":
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    from talos.specifications.virtual_telescope import VirtualTelescopeSpec
    import matplotlib.pyplot as plt
    np.random.seed(1)

    num_times = 30
    rep = GraphRepresentation(
        TelescopePointing(
            virtual_telescope=VirtualTelescopeSpec(
                num_times=num_times,
                num_cp=10,
                # TODO: move to telescope spec
                cross_threshold=0.,
                step_size=0.1,
                duration=5.,
            ),
            telescope_view_halfangle_tol_arcsec=90.,
        ))
    sim = Simulator(rep)

    # NOTE: these inputs are computed outside this discipline, and this
    # test only validates derivatives for valid inputs; e.g.
    # telescope_vector cannot have any zero vectors, separation
    # cannot be zero, telescope_vector_component_in_sun_direction must
    # be defined to be a component of telescope_vector

    telescope_vector = np.zeros((num_times, 3))
    telescope_vector[:, 0] = 100 * np.random.rand(num_times * 1).reshape(
        (num_times, ))
    telescope_vector[:, 1] = 0.001 * np.random.rand(num_times * 1).reshape(
        (num_times, ))
    telescope_vector[:, 2] = 0.001 * np.random.rand(num_times * 1).reshape(
        (num_times, ))
    # telescope_vector = 100 * np.random.rand(num_times * 3).reshape( (num_times, 3))
    sim['telescope_vector_component_in_sun_direction'] = telescope_vector[:, 0]
    sim['separation_m'] = np.linalg.norm(telescope_vector, axis=1)
    sim['observation_phase_indicator'] = np.random.randint(2, size=num_times)
    if np.any(sim['separation_m'] == 0):
        raise ValueError("Zero values for separation will prevent proper test")
    print(sim['separation_m'])
    sim.run()
    print(sim['cos_view_angle'])
    sim.check_partials(
        compact_print=True,
        form='reverse',
        method='cs',
        step=1e-6,
        # show_only_incorrect=True,
    )
    plt.plot(sim['cos_view_angle'])
    plt.show()

# (of,wrt)                                                                             calc norm               relative error             absolute error
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# ('cos_view_angle', 'telescope_vector_component_in_sun_direction')                    86.96924438453017       5.851447510931035e-12      5.088959685813109e-10
# ('cos_view_angle', 'separation_m')                                                   86.50767113465707       8.696631921926587e-05      0.007522599541009436
# ('cos_view_angle', 'observation_phase_indicator')                                    0.0                     0.0                        0.0
# ('cos_view_angle_lt1', 'telescope_vector_component_in_sun_direction')                0.0                     1.0                        5385164.807134504
# ('cos_view_angle_lt1', 'separation_m')                                               0.0                     0.0                        0.0
# ('cos_view_angle_lt1', 'observation_phase_indicator')                                0.0                     0.0                        0.0
# ('cos_view_angle_gt_neg1', 'telescope_vector_component_in_sun_direction')            0.0                     0.0                        0.0
# ('cos_view_angle_gt_neg1', 'separation_m')                                           0.0                     0.0                        0.0
# ('cos_view_angle_gt_neg1', 'observation_phase_indicator')                            0.0                     0.0                        0.0
# ('cos_view_angle_ge1', 'telescope_vector_component_in_sun_direction')                0.0                     1.0                        5385164.807134504
# ('cos_view_angle_ge1', 'separation_m')                                               0.0                     0.0                        0.0
# ('cos_view_angle_ge1', 'observation_phase_indicator')                                0.0                     0.0                        0.0
# ('cos_view_angle_le_neg1', 'telescope_vector_component_in_sun_direction')            0.0                     0.0                        0.0
# ('cos_view_angle_le_neg1', 'separation_m')                                           0.0                     0.0                        0.0
# ('cos_view_angle_le_neg1', 'observation_phase_indicator')                            0.0                     0.0                        0.0
# ('telescope_view_angle_unmasked', 'telescope_vector_component_in_sun_direction')     10768.51511230071       11.59367791133177          10595.177392229225
# ('telescope_view_angle_unmasked', 'separation_m')                                    10768.163913796074      5.631840236903503          9671.276029002616
# ('telescope_view_angle_unmasked', 'observation_phase_indicator')                     0.0                     0.0                        0.0
# ('telescope_view_angle', 'telescope_vector_component_in_sun_direction')              5337.357166299116       6.0530126326231635         5192.592981507902
# ('telescope_view_angle', 'separation_m')                                             5336.648582883626       3.955033547945778          4582.788143313666
# ('telescope_view_angle', 'observation_phase_indicator')                              0.10307524319108206     3.2400426680096616e-11     3.339681859437576e-12
# ('max_telescope_view_angle', 'telescope_vector_component_in_sun_direction')          230.03100062466632      0.784048323008966          140.92016630309953
# ('max_telescope_view_angle', 'separation_m')                                         229.28284433829734      0.6869581939598873         124.31790372414929
# ('max_telescope_view_angle', 'observation_phase_indicator')                          0.02197073301041763     8.101969373429952e-07      1.7800635018128872e-08
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
