from csdl import Model
import csdl
import numpy as np

# TODO: Use the "NOTE" comments in this file to design error messages
# where domain of finite derivatives is not the same as domain of valid
# inputs for individual functions; raise error if these functions are
# used and no runtime assertions provided; warn if runtime assertions
# are either too restrictive or guaranteed to pass; runtime assertions
# are required (1) for the compiler to verify that the user has
# communicated how the standard library functions are to be used, and
# (2) so that other users can be reminded of valid domains for inputs


class Alignment(Model):

    def initialize(self):
        self.parameters.declare('virtual_telescope')
        self.parameters.declare('telescope_view_plane_tol_mm',
                                default=18.,
                                types=float)
        self.parameters.declare('view_plane_error_scaler',
                                default=1.,
                                types=(float, np.ndarray))

    def define(self):
        virtual_telescope = self.parameters['virtual_telescope']
        num_times = virtual_telescope['num_times']
        telescope_view_plane_tol_mm = self.parameters[
            'telescope_view_plane_tol_mm']
        view_plane_error_scaler = self.parameters['view_plane_error_scaler']

        telescope_vector = self.declare_variable('telescope_vector',
                                                 shape=(num_times, 3))
        telescope_vector_component_in_sun_direction = self.declare_variable(
            'telescope_vector_component_in_sun_direction',
            shape=(num_times, ),
        )

        separation_m = self.declare_variable('separation_m',
                                             shape=(num_times, ))
        telescope_direction_in_view_plane = telescope_vector * (
            1. - csdl.expand(
                telescope_vector_component_in_sun_direction,
                shape=(num_times, 3),
                indices='i->ij',
            ) /
            csdl.expand(separation_m, shape=(num_times, 3), indices='i->ij'))
        self.register_output('telescope_direction_in_view_plane',
                             telescope_direction_in_view_plane)
        view_plane_error = csdl.pnorm(
            telescope_direction_in_view_plane,
            axis=1,
        )
        self.register_output('view_plane_error', view_plane_error)

        observation_phase_indicator = self.declare_variable(
            'observation_phase_indicator',
            shape=(num_times, ),
        )

        view_plane_error_during_observation = observation_phase_indicator * view_plane_error
        self.register_output('view_plane_error_during_observation',
                             view_plane_error_during_observation)

        max_view_plane_error = csdl.max(
            view_plane_error_during_observation,
            rho=50.,
        )
        self.register_output('max_view_plane_error', max_view_plane_error)
        # self.add_constraint(
        #     'max_view_plane_error',
        #     upper=(telescope_view_plane_tol_mm / 1000.)**2,
        #     scaler=view_plane_error_scaler,
        # )
