import numpy as np

from talos.csdl_future.mask import MaskGE, MaskLT, MaskGT

from talos.constants import deg2arcsec, s
from csdl import Model, GraphRepresentation
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


class SpacecraftPointing(Model):

    def initialize(self):
        self.parameters.declare('swarm')
        self.parameters.declare('telescope_length_m', default=40., types=float)
        self.parameters.declare('telescope_length_tol_mm',
                                default=15.,
                                types=float)
        self.parameters.declare('telescope_view_plane_tol_mm',
                                default=18.,
                                types=float)
        # constrain telescope and each s/c to satisfy pointing accuracy
        self.parameters.declare('telescope_view_halfangle_tol_arcsec',
                                default=90.,
                                types=float)
        # TODO: for a later paper, find appropriate speed constraint
        self.parameters.declare('relative_speed_tol_um_s',
                                default=100.,
                                types=float)

    def define(self):
        swarm = self.parameters['swarm']
        telescope_length_m = self.parameters['telescope_length_m']
        telescope_length_tol_mm = self.parameters['telescope_length_tol_mm']
        telescope_view_plane_tol_mm = self.parameters[
            'telescope_view_plane_tol_mm']
        num_times = swarm['num_times']
        telescope_view_halfangle_tol_arcsec = self.parameters[
            'telescope_view_halfangle_tol_arcsec']
        relative_speed_tol_um_s = self.parameters['relative_speed_tol_um_s']

        if telescope_length_m <= 0:
            raise ValueError('Telescope length must be a positive value')
        if telescope_length_tol_mm <= 0:
            raise ValueError(
                'Telescope length tolerance must be a positive value')

        # optics_relative_orbit_state_m = self.declare_variable(
        # 'optics_relative_orbit_state_m', shape=(num_times, 6))
        # detector_relative_orbit_state_m = self.declare_variable(
        # 'detector_relative_orbit_state_m', shape=(num_times, 6))
        optics_orbit_state = self.declare_variable('optics_orbit_state',
                                                   shape=(num_times, 6))
        detector_orbit_state = self.declare_variable('detector_orbit_state',
                                                     shape=(num_times, 6))

        sun_direction = self.declare_variable('sun_direction',
                                              shape=(num_times, 3))

        attitude = False
        if attitude is True:
            # optics cubesat orients so that -y-axis is aligned with sun
            # detector cubesat orients so that -x-axis is aligned with sun
            # -optics_B_from_ECI[:,0,:][1, :] -> -optics_B_from_ECI[1,0,:]
            # -(sr * sp * cy - cr * sy) == 1
            # -detector_B_from_ECI[:,0,:][0, :] -> -detector_B_from_ECI[0,0,:]
            # -(cp * cy) == 1

            # Orientation of each s/c during observation
            optics_B_from_ECI = self.declare_variable('optics_B_from_ECI',
                                                      shape=(3, 3, num_times))
            detector_B_from_ECI = self.declare_variable('detector_B_from_ECI',
                                                        shape=(3, 3,
                                                               num_times))

            optics_sun_direction_body = csdl.einsum(optics_B_from_ECI,
                                                    sun_direction,
                                                    subscripts='ijk,kj->ik')
            detector_sun_direction_body = csdl.einsum(detector_B_from_ECI,
                                                      sun_direction,
                                                      subscripts='ijk,kj->ik')

            # # optics and detector oriented differently on each s/c
            # optics_cos_view_angle = csdl.reshape(
            #     optics_sun_direction_body[1, :], (num_times, ))
            # detector_cos_view_angle = csdl.reshape(
            #     detector_sun_direction_body[0, :], (num_times, ))
            # self.register_output('optics_cos_view_angle',
            #                      optics_cos_view_angle)
            # self.register_output('detector_cos_view_angle',
            #                      detector_cos_view_angle)
            # optics_cos_view_angle_during_observation = observation_phase_indicator * (
            #     optics_cos_view_angle - 1) + 1
            # detector_cos_view_angle_during_observation = observation_phase_indicator * (
            #     detector_cos_view_angle - 1) + 1

            # self.register_output('optics_cos_view_angle_during_observation',
            #                      optics_cos_view_angle_during_observation)
            # self.register_output('detector_cos_view_angle_during_observation',
            #                      detector_cos_view_angle_during_observation)

            # min_optics_cos_view_angle = csdl.min(
            #     optics_cos_view_angle_during_observation)
            # min_detector_cos_view_angle = csdl.min(
            #     detector_cos_view_angle_during_observation)
            # print([op.name
            #        for op in min_optics_cos_view_angle.dependencies])
            # for op in min_optics_cos_view_angle.dependencies:
            #     print([(var.name, var.shape) for var in op.dependencies])
            # print([op.name
            #        for op in min_detector_cos_view_angle.dependencies])
            # for op in min_detector_cos_view_angle.dependencies:
            #     print([(var.name, var.shape) for var in op.dependencies])
            # print('observation_phase_indicator',observation_phase_indicator.shape)
            # print('optics_cos_view_angle',optics_cos_view_angle.shape)
            # print('detector_cos_view_angle',detector_cos_view_angle.shape)
            # print('optics_cos_view_angle_during_observation',optics_cos_view_angle_during_observation.shape)
            # print('detector_cos_view_angle_during_observation',detector_cos_view_angle_during_observation.shape)
            # print('min_optics_cos_view_angle',min_optics_cos_view_angle.shape)
            # print('min_detector_cos_view_angle',min_detector_cos_view_angle.shape)
            # exit()

            # self.register_output('min_optics_cos_view_angle',
            #                      min_optics_cos_view_angle)
            # self.register_output('min_detector_cos_view_angle',
            #                      min_detector_cos_view_angle)
            # self.add_constraint(
            #     'min_optics_cos_view_angle',
            #     lower=np.cos(telescope_view_halfangle_tol_arcsec / deg2arcsec *
            #                  np.pi / 180),
            # )
            # self.add_constraint(
            #     'min_detector_cos_view_angle',
            #     lower=np.cos(telescope_view_halfangle_tol_arcsec / deg2arcsec *
            #                  np.pi / 180),
            # )
