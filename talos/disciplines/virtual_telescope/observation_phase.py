from talos.csdl_future.mask import MaskGE

from csdl import Model
import csdl

# TODO: Use the "NOTE" comments in this file to design error messages
# where domain of finite derivatives is not the same as domain of valid
# inputs for individual functions; raise error if these functions are
# used and no runtime assertions provided; warn if runtime assertions
# are either too restrictive or guaranteed to pass; runtime assertions
# are required (1) for the compiler to verify that the user has
# communicated how the standard library functions are to be used, and
# (2) so that other users can be reminded of valid domains for inputs


class ObservationPhase(Model):

    def initialize(self):
        self.parameters.declare('virtual_telescope')

    def define(self):
        virtual_telescope = self.parameters['virtual_telescope']
        num_times = virtual_telescope['num_times']

        # TODO: ???
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

        # define observation phase as time when both s/c are flying
        # towards the sun; magnitude is not important except for setting
        # minimum threshold for indicating observation phase
        optics_observation_dot = csdl.reshape(optics_orbit_state[:, 3],
                                              (num_times, ))
        detector_observation_dot = csdl.reshape(detector_orbit_state[:, 3],
                                                (num_times, ))
        self.register_output(
            'optics_observation_dot',
            optics_observation_dot,
        )
        self.register_output(
            'detector_observation_dot',
            detector_observation_dot,
        )

        # define observation phase as time when both s/c are flying
        # towards the sun and have line of sight; use mask operation to
        # limit observation to time when telescope is sufficiently
        # aligned
        optics_sun_LOS = self.declare_variable('optics_sun_LOS',
                                               shape=(num_times, 1))
        detector_sun_LOS = self.declare_variable('detector_sun_LOS',
                                                 shape=(num_times, 1))

        # This hasn't shown to shift the values between 0 and 1 down by
        # a noticeable amount
        los = csdl.reshape(optics_sun_LOS * detector_sun_LOS, (num_times, ))
        self.register_output('los', los)
        telescope_sun_LOS_indicator = csdl.custom(
            los,
            op=MaskGE(
                num_times=num_times,
                threshold=1,
                in_name='los',
                out_name='telescope_sun_LOS_indicator',
            ),
        )
        optics_observation_phase_indicator = csdl.custom(
            optics_observation_dot,
            op=MaskGE(
                num_times=num_times,
                threshold=virtual_telescope['cross_threshold'],
                in_name='optics_observation_dot',
                out_name='optics_observation_phase_indicator',
            ),
        )
        detector_observation_phase_indicator = csdl.custom(
            detector_observation_dot,
            op=MaskGE(
                num_times=num_times,
                threshold=virtual_telescope['cross_threshold'],
                in_name='detector_observation_dot',
                out_name='detector_observation_phase_indicator',
            ),
        )
        self.register_output(
            'optics_observation_phase_indicator',
            optics_observation_phase_indicator,
        )
        self.register_output(
            'detector_observation_phase_indicator',
            detector_observation_phase_indicator,
        )

        # All three conditions must be true to have an observation: sun
        # must be in line of sight, optics s/c must meet observatoin
        # phase requirement, and detector s/c mus tmeet observation
        # requirement
        # observation_phase_indicator = optics_observation_phase_indicator * detector_observation_phase_indicator * telescope_sun_LOS_indicator
        # self.register_output('observation_phase_indicator',
        #                      observation_phase_indicator)

        observation_phase_indicator = self.create_input(
            'observation_phase_indicator', shape=(num_times, ))
        # for testing derivatives
        # self.add_constraint('optics_observation_dot',equals=0)
        # self.add_constraint('detector_observation_dot',equals=0)
        # self.add_constraint('optics_observation_phase_indicator',equals=0)
        # self.add_constraint('detector_observation_phase_indicator',equals=0)
        # self.add_constraint('observation_phase_indicator',equals=0)


if __name__ == "__main__":
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    from talos.specifications.virtual_telescope import VirtualTelescopeSpec
    import numpy as np

    num_times = 30
    rep = GraphRepresentation(
        ObservationPhase(virtual_telescope=VirtualTelescopeSpec(
            num_times=num_times,
            num_cp=10,
            cross_threshold=0.,
            step_size=0.1,
            duration=5.,
        ), ))
    sim = Simulator(rep)
    sim['optics_orbit_state'] = 7000 * np.random.rand(np.prod(
        (num_times, 6))).reshape((num_times, 6))
    sim['detector_orbit_state'] = 7000 * np.random.rand(np.prod(
        (num_times, 6))).reshape((num_times, 6))

    def generate_los():
        los = np.random.rand(np.prod((num_times, 1))).reshape((num_times, 1))
        offset = 0 - np.min(los)
        range = np.max(los) - np.min(los)
        return (los + offset) / range

    sim['optics_sun_LOS'] = generate_los()
    sim['detector_sun_LOS'] = generate_los()
    print(sim['optics_sun_LOS'])
    print(sim['detector_sun_LOS'])
    sim.run()
    sim.check_partials(compact_print=True)
