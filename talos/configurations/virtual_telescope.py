from talos.configurations.cubesat import Cubesat
from talos.specifications.swarm_spec import SwarmSpec
from talos.specifications.virtual_telescope import VirtualTelescopeSpec
from talos.disciplines.virtual_telescope.observation_phase import ObservationPhase
from talos.disciplines.virtual_telescope.alignment import Alignment
from talos.disciplines.virtual_telescope.separation import Separation
from talos.disciplines.virtual_telescope.spacecraft_pointing import SpacecraftPointing
from talos.disciplines.virtual_telescope.telescope_pointing import TelescopePointing

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


class VirtualTelescope(Model):

    def initialize(self):
        self.parameters.declare('swarm')
        self.parameters.declare('virtual_telescope',
                                types=VirtualTelescopeSpec)

    def define(self):
        virtual_telescope = self.parameters['virtual_telescope']
        num_times: int = virtual_telescope['num_times']
        num_cp: int = virtual_telescope['num_cp']
        step_size: float = virtual_telescope['step_size']
        duration: float = virtual_telescope['duration']

        optics_cubesat_spec = virtual_telescope['optics_cubesat']
        detector_cubesat_spec = virtual_telescope['detector_cubesat']
        telescope_length_m = virtual_telescope['telescope_length_m']
        telescope_length_tol_mm = virtual_telescope['telescope_length_tol_mm']
        telescope_view_plane_tol_mm = virtual_telescope[
            'telescope_view_plane_tol_mm']
        telescope_view_halfangle_tol_arcsec = virtual_telescope[
            'telescope_view_halfangle_tol_arcsec']
        telescope_view_angle_scaler = virtual_telescope[
            'telescope_view_angle_scaler']
        min_separation_scaler = virtual_telescope['min_separation_scaler']
        max_separation_scaler = virtual_telescope['max_separation_scaler']
        max_separation_all_phases_scaler = virtual_telescope[
            'max_separation_all_phases_scaler']
        view_plane_error_scaler = virtual_telescope['view_plane_error_scaler']

        if telescope_length_m <= 0:
            raise ValueError('Telescope length must be a positive value')
        if telescope_length_tol_mm <= 0:
            raise ValueError(
                'Telescope length tolerance must be a positive value')

        # TODO: What are we modeling here?
        earth_orbit_angular_speed_rad_min = 2 * np.pi / 365 * 1 / 24 * 1 / 60
        step_size_min = duration / num_times
        earth_orbit_angular_position = earth_orbit_angular_speed_rad_min * step_size_min * np.arange(
            num_times)

        h = np.ones(num_times - 1) * step_size
        self.create_input('h', val=h)
        # sun_direction (in ECI frame) used in TelescopeConfiguration;
        # here it is an input because it is precomputed

        v = np.zeros((num_times, 3))
        v[:, 0] = 1
        # v[:, 0] = np.cos(earth_orbit_angular_position)
        # v[:, 1] = np.sin(earth_orbit_angular_position)
        sun_direction = self.create_input(
            'sun_direction',
            shape=(num_times, 3),
            val=v,
        )

        # add model for each cubesat in telescope
        for cubesat in [optics_cubesat_spec, detector_cubesat_spec]:
            cubesat_name = cubesat['name']
            submodel_name = '{}_cubesat'.format(cubesat_name)
            self.add(
                Cubesat(
                    num_times=num_times,
                    num_cp=num_cp,
                    step_size=step_size,
                    cubesat=cubesat,
                ),
                name='{}_cubesat'.format(cubesat_name),
                promotes=['reference_orbit_state', 'h'],
            )
            self.connect('sun_direction',
                         '{}.sun_direction'.format(submodel_name))

        self.connect_vars('relative_orbit_state')

        # add constraints for defining telescope configuration
        # self.connect_vars('B_from_ECI')

        optics_relative_orbit_state = self.declare_variable(
            'optics_relative_orbit_state',
            shape=(num_times, 6),
        )
        detector_relative_orbit_state = self.declare_variable(
            'detector_relative_orbit_state',
            shape=(num_times, 6),
        )
        telescope_vector = (optics_relative_orbit_state[:, :3] -
                            detector_relative_orbit_state[:, :3]) * 1000
        self.register_output('telescope_vector', telescope_vector)

        # Constrain view plane error to meet requirements
        telescope_vector_component_in_sun_direction = csdl.dot(
            sun_direction,
            telescope_vector,
            axis=1,
        )
        self.register_output('telescope_vector_component_in_sun_direction',
                             telescope_vector_component_in_sun_direction)

        self.add(
            ObservationPhase(virtual_telescope=virtual_telescope),
            name='observation_phase',
        )
        self.connect_vars('sun_LOS')
        self.connect_vars('orbit_state')

        self.add(
            TelescopePointing(
                virtual_telescope=virtual_telescope,
                telescope_view_halfangle_tol_arcsec=
                telescope_view_halfangle_tol_arcsec,
                telescope_view_angle_scaler=telescope_view_angle_scaler),
            name='telescope_pointing',
        )
        self.add(
            Separation(
                virtual_telescope=virtual_telescope,
                telescope_length_m=telescope_length_m,
                telescope_length_tol_mm=telescope_length_tol_mm,
                min_separation_scaler=min_separation_scaler,
                max_separation_scaler=max_separation_scaler,
                max_separation_all_phases_scaler=
                max_separation_all_phases_scaler,
            ),
            name='separation',
        )
        self.add(
            Alignment(
                virtual_telescope=virtual_telescope,
                telescope_view_plane_tol_mm=telescope_view_plane_tol_mm,
                view_plane_error_scaler=view_plane_error_scaler,
            ),
            name='alignment',
        )

        # Define Objective

        ## acceleration_due_to_thrust
        # optics_acceleration_due_to_thrust, detector_acceleration_due_to_thrust = self.import_vars(
        # 'acceleration_due_to_thrust', shape=(num_times, 3))
        # a = optics_acceleration_due_to_thrust + detector_acceleration_due_to_thrust
        # total_acceleration_due_to_thrust = csdl.sum(a*csdl.tanh(5*a))
        # self.register_output('total_acceleration_due_to_thrust', total_acceleration_due_to_thrust)
        # obj = 10*total_acceleration_due_to_thrust

        # ## total_thrust
        optics_total_thrust, detector_total_thrust = self.import_vars(
            'total_thrust', shape=(num_times, ))
        total_thrust = csdl.sum(optics_total_thrust + detector_total_thrust)
        self.register_output('total_thrust', total_thrust)

        ## initial_propellant
        # optics_initial_propellant_mass, detector_initial_propellant_mass = self.import_vars(
        #     'initial_propellant_mass', shape=(1, ))
        # total_propellant = optics_initial_propellant_mass + detector_initial_propellant_mass
        # self.register_output('total_propellant', total_propellant)

        ## total_propellant_used
        # optics_total_propellant_used, detector_total_propellant_used = self.import_vars(
        #     'total_propellant_used')
        # total_propellant_used = optics_total_propellant_used + detector_total_propellant_used
        # self.register_output('total_propellant_used', total_propellant_used)

        ## total_propellant_mass using initial propellant mass
        # optics_initial_propellant_mass, detector_initial_propellant_mass = self.import_vars(
        #     'initial_propellant_mass')
        # total_propellant_mass = optics_initial_propellant_mass + detector_initial_propellant_mass
        # self.register_output('total_propellant_mass',
        # total_propellant_mass)
        max_telescope_view_angle = self.declare_variable(
            'max_telescope_view_angle')

        # obj = 1.001 * max_telescope_view_angle
        obj = 1.00001 * total_thrust
        # obj = max_telescope_view_angle + total_thrust
        # obj = 1000*max_telescope_view_angle + total_propellant_used
        # obj = 1000*max_telescope_view_angle + total_propellant
        # obj = total_propellant

        # obj = csdl.sum(
        # csdl.pnorm(optics_acceleration_due_to_thrust +
        #    detector_acceleration_due_to_thrust,
        #    axis=1))
        self.register_output('obj', obj)
        self.add_objective('obj', scaler=virtual_telescope['obj_scaler'])

        # for testing derivatives
        # self.add_constraint('optics_cubesat.relative_orbit_state', equals=0)
        # self.add_constraint('detector_cubesat.relative_orbit_state', equals=0)
        # self.add_constraint('telescope_vector', equals=0)

    def import_vars(self, name, shape=(1, )):
        optics = self.declare_variable(f'optics_{name}', shape=shape)
        detector = self.declare_variable(f'detector_{name}', shape=shape)

        self.connect_vars(name)

        return optics, detector

    def connect_vars(self, name):
        self.connect(
            f'optics_cubesat.{name}',
            f'optics_{name}',
        )
        self.connect(
            f'detector_cubesat.{name}',
            f'detector_{name}',
        )
