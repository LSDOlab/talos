"""
Determine if the Satellite has line of sight with the ground stations
"""

# https://github.com/vgucsd/talos/tree/b30e749b106638441777636d43d3c25b84248376/talos/communication

import numpy as np
from csdl import Model
import csdl
from talos.disciplines.communication.download_rate import ODEProblemTest
from talos.specifications.ground_station import GroundStationSpec
from talos.specifications.cubesat_spec import CubesatSpec
from talos.csdl_future.mask import MaskGT

Re = 6378.137
d2r = np.pi / 180.
speed_of_light_in_vacuum = 299792458 / 1e3  # km/s
receiver_gain = 10**(12.9 / 10)  # dB
line_loss_factor = 10**(-2.0 / 10)  # dB
transmission_frequency = 437e6  # Hz
boltzman = 1.3806503e-23  # J/K
signal_to_noise_ratio = 10**(5.0 / 10)  # dB
system_noise_temperature = 500.  # K
alpha = speed_of_light_in_vacuum**2 * receiver_gain * line_loss_factor / (
    16.0 * np.pi**2 * transmission_frequency**2 * boltzman *
    signal_to_noise_ratio * system_noise_temperature)

# m**2 * dB**2 * W * dB * K
# s**2 * MHz**2 * J * K * dB * km**2

# dB**2
# s

# download_rate has units of dB**2/s


def sigmoid(x):
    return 1 / (1 + csdl.exp(-x))


def quaternion_to_rotation_mtx(model: Model, name: str, q, num_times):
    DCM = model.create_output(name, val=np.zeros((num_times, 3, 3)))
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    DCM[:, 0, 0] = csdl.reshape(q0**2 + q1**2 - q2**2 - q3**2,
                                (num_times, 1, 1))
    DCM[:, 0, 1] = csdl.reshape(2 * (q1 * q2 + q0 * q3), (num_times, 1, 1))
    DCM[:, 0, 2] = csdl.reshape(2 * (q1 * q3 - q0 * q2), (num_times, 1, 1))
    DCM[:, 1, 0] = csdl.reshape(2 * (q1 * q2 - q0 * q3), (num_times, 1, 1))
    DCM[:, 1, 1] = csdl.reshape(q0**2 - q1**2 + q2**2 - q3**2,
                                (num_times, 1, 1))
    DCM[:, 1, 2] = csdl.reshape(2 * (q2 * q3 + q0 * q1), (num_times, 1, 1))
    DCM[:, 2, 0] = csdl.reshape(2 * (q1 * q3 + q0 * q2), (num_times, 1, 1))
    DCM[:, 2, 1] = csdl.reshape(2 * (q2 * q3 - q0 * q1), (num_times, 1, 1))
    DCM[:, 2, 2] = csdl.reshape(q0**2 - q1**2 - q2**2 + q3**2,
                                (num_times, 1, 1))
    return DCM


class Groundstation(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('groundstation', types=GroundStationSpec)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        groundstation = self.parameters['groundstation']

        # locate groundstation
        lon = groundstation['lon']
        lat = groundstation['lat']
        alt = groundstation['alt']

        cos_lat = np.cos(d2r * lat)
        distance_center_of_earth_to_groundstation = (Re + alt)

        # compute grounstation position so that line of sight can be
        # computed
        gp_ECEF = np.zeros((num_times, 3))
        gp_ECEF[:,
                0] = distance_center_of_earth_to_groundstation * cos_lat * np.cos(
                    d2r * lon)
        gp_ECEF[:,
                1] = distance_center_of_earth_to_groundstation * cos_lat * np.sin(
                    d2r * lon)
        gp_ECEF[:, 2] = distance_center_of_earth_to_groundstation * np.sin(
            d2r * lat)
        groundstation_position_ECEF = self.create_input(
            'groundstation_position_ECEF', val=gp_ECEF)

        # Model the rotation of the Earth to establish location of
        # groundstation in ECI frame to establish line of sight with
        # spacecraft
        t = self.create_input(
            'times',
            shape=num_times,
            val=np.linspace(0., step_size * (num_times - 1), num_times),
        )

        theta = np.pi / 3600.0 / 24.0 * t

        q_E = self.create_output('q_E', shape=(num_times, 4), val=0)
        q_E[:, 0] = csdl.reshape(csdl.cos(theta), (num_times, 1))
        q_E[:, 3] = -csdl.reshape(csdl.sin(theta), (num_times, 1))

        ECEF_from_ECI = quaternion_to_rotation_mtx(self, 'ECEF_from_ECI', q_E,
                                                   num_times)
        # use ECI frame, same as spacecraft position
        groundstation_position_ECI = csdl.einsum(
            ECEF_from_ECI,
            groundstation_position_ECEF,
            subscripts='nij,nj->ni',
        )
        self.register_output('groundstation_position_ECI',
                             groundstation_position_ECI)


class GroundstationSpacecraft(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']

        orbit_state = self.declare_variable('orbit_state',
                                            shape=(num_times, 6))

        groundstation_position_ECI = self.declare_variable(
            'groundstation_position_ECI', shape=(num_times, 3))
        groundstation_to_spacecraft_ECI = orbit_state[:, :
                                                      3] - groundstation_position_ECI
        self.register_output(
            'groundstation_to_spacecraft_ECI',
            groundstation_to_spacecraft_ECI,
        )

        # compute line of sight to groundstation
        gs_dot = csdl.dot(
            groundstation_position_ECI,
            groundstation_to_spacecraft_ECI,
            axis=1,
        )
        self.register_output('gs_dot', gs_dot)
        los_comm = csdl.custom(
            gs_dot,
            op=MaskGT(
                num_times=num_times,
                threshold=0,
                in_name='gs_dot',
                out_name='los_comm',
            ),
        )
        self.register_output('los_comm', los_comm)

        # TODO: set value for arXiv paper
        comm_power = self.create_input('comm_power', shape=num_times, val=1.)
        self.add_design_variable('comm_power')
        transmitter_gain = self.declare_variable('transmitter_gain',
                                                 shape=num_times)

        distance_to_groundstation = csdl.pnorm(groundstation_to_spacecraft_ECI,
                                               axis=1)
        self.register_output('distance_to_groundstation',
                             distance_to_groundstation)

        download_rate = alpha * comm_power * transmitter_gain * los_comm / distance_to_groundstation**2
        self.register_output('download_rate', download_rate)


class Communication(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('groundstations', types=dict, allow_none=True)
        self.parameters.declare(
            'omnidirectional_antenna',
            types=bool,
            default=True,
        )

    def define(self):
        num_times = self.parameters['num_times']
        groundstations = self.parameters['groundstations']

        # # TODO: add design variable?
        # # TODO: angle relative to what part of s/c?
        # antenna_angle = self.create_input('antenna_angle', val=0.0)

        # q_A is unused
        # q_A = self.create_output('q_A', np.zeros((num_times, 4)))
        # rt2 = np.sqrt(2)
        # q_A[:, 0] = csdl.expand(csdl.cos(antenna_angle / 2.), (num_times, 1))
        # q_A[:, 1] = csdl.expand(
        #     csdl.sin(antenna_angle / 2.) / rt2, (num_times, 1))
        # q_A[:, 2] = -csdl.expand(
        #     csdl.sin(antenna_angle / 2.) / rt2, (num_times, 1))

        # groundstation_position_relative_to_spacecraft_A = csdl.einsum(
        #     B_from_A,
        #     groundstation_position_relative_to_spacecraft_B,
        #     'ijn,jn->in',
        # )
        self.add(
            MaxDownloadRate(
                num_times=num_times,
                groundstations=groundstations,
            ), )

        initial_data = self.create_input('initial_data', val=0.0)
        # self.add(DownloadRate(num_nodes=num_times, ))
        self.add(
            # ODEProblemTest('GaussLegendre4', 'collocation',
            ODEProblemTest('RK4', 'time-marching',
                           num_times).create_solver_model(),
            name='reaction_wheel_speed_integrator',
        )

        data = self.declare_variable('data', shape=(num_times, ))

        total_data_downloaded = data[-1] - data[0]
        self.register_output(
            'total_data_downloaded',
            total_data_downloaded,
        )


class MaxDownloadRate(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('groundstations', types=dict)

    def define(self):
        num_times = self.parameters['num_times']
        groundstations = self.parameters['groundstations']

        download_rates = []
        for gs_name in groundstations.keys():
            dr = self.declare_variable(f'{gs_name}_download_rate',
                                       shape=(num_times, ))
            download_rates.append(dr)
        if len(download_rates) > 0:
            # FIXME: not zero when all download rates are zero,
            # regardless of value of rho
            download_rate = csdl.max(*download_rates)
            self.register_output('download_rate', download_rate)


if __name__ == "__main__":
    from talos.disciplines.orbit.reference_orbit import ReferenceOrbitTrajectory
    from csdl import GraphRepresentation
    # from csdl_om import Simulator
    from python_csdl_backend import Simulator

    same_hemisphere = -1
    shift_east = 5
    offset = 10.

    groundstations = dict()

    # ax.scatter(-117.2340, 32.8801, marker="p", label="UCSD")
    # 32.86346° N, 117.23091° W
    groundstations['UCSD'] = GroundStationSpec(
        name='UCSD',
        lon=-117.2340 + shift_east * offset,
        lat=same_hemisphere * 32.8801,
        alt=0.4849,
    )
    # 40.10080° N, 88.22605° W
    # ax.scatter(-88.2272, 40.1020, marker="p", label="UIUC")
    groundstations['UIUC'] = GroundStationSpec(
        name='UIUC',
        lon=-88.2272 + shift_east * offset,
        lat=same_hemisphere * 40.1020,
        alt=0.2329,
    )
    # 33.77230° N, 84.39469° W
    # ax.scatter(-84.3963, 33.7756, marker="p", label="Georgia")
    groundstations['GT'] = GroundStationSpec(
        name='GT',
        lon=-84.3963 + shift_east * offset,
        lat=same_hemisphere * 33.7756,
        alt=0.2969,
    )
    # 45.66733° N, 111.04935° W
    # ax.scatter(-109.533691, 46.9653, marker="p", label="Montana")
    groundstations['Montana'] = GroundStationSpec(
        name='Montana',
        lon=-109.5337 + shift_east * offset,
        lat=46.9653,
        alt=1.04,
    )
    cubesats = dict()
    cubesats['optics'] = CubesatSpec(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=np.array([
            40,
            0,
            0,
            0.,
            0.,
            0.,
            # 1.76002146e+03,
            # 6.19179823e+03,
            # 6.31576531e+03,
            # 4.73422022e-05,
            # 1.26425269e-04,
            # 5.39731211e-05,
        ]),
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    )

    # cubesats['detector'] = CubesatSpec(
    #     name='detector',
    #     dry_mass=1.3,
    #     initial_orbit_state=np.array([
    #         0,
    #         0,
    #         0,
    #         0,
    #         0.,
    #         0.,
    #         # 1.76002146e+03,
    #         # 6.19179823e+03,
    #         # 6.31576531e+03,
    #         # 4.73422022e-05,
    #         # 1.26425269e-04,
    #         # 5.39731211e-05,
    #     ]),
    #     specific_impulse=47.,
    #     perigee_altitude=500.002,
    #     apogee_altitude=499.98,
    # )


    class CubeSat(Model):

        def initialize(self):
            self.parameters.declare('num_times', types=int)
            self.parameters.declare('cubesat', types=CubesatSpec)
            self.parameters.declare('groundstations',
                                    types=dict,
                                    allow_none=True)

        def define(self):
            num_times = self.parameters['num_times']
            groundstations = self.parameters['groundstations']

            if groundstations is not None:
                self.add(
                    Communication(
                        num_times=num_times,
                        groundstations=groundstations,
                    ), )

    class Mission(Model):

        def initialize(self):
            self.parameters.declare('num_times', types=int)
            self.parameters.declare('step_size', types=float)
            self.parameters.declare('groundstations',
                                    types=dict,
                                    allow_none=True)
            self.parameters.declare('cubesats', types=dict)

        def define(self):
            num_times = self.parameters['num_times']
            step_size = self.parameters['step_size']
            groundstations = self.parameters['groundstations']
            cubesats = self.parameters['cubesats']
            cubesat_names = cubesats.keys()

            num_orbits = 30
            duration = 90
            step_size = 19.
            # step_size = 0.1
            # num_times = int(duration * 60 * num_orbits / step_size)
            # num_times = 3

            self.add(ReferenceOrbitTrajectory(num_times=num_times, ), )

            if groundstations is not None:
                for gs_name, groundstation in groundstations.items():
                    self.add(
                        Groundstation(
                            num_times=num_times,
                            step_size=step_size,
                            groundstation=groundstation,
                        ),
                        name=gs_name,
                        promotes=[],
                    )
                    for sc_name in cubesat_names:
                        self.add(
                            GroundstationSpacecraft(num_times=num_times),
                            name=f'{gs_name}_{sc_name}',
                            promotes=['orbit_state'],
                        )

                for gs_name, groundstation in groundstations.items():
                    for sc_name in cubesat_names:
                        self.connect(
                            f'{gs_name}.groundstation_position_ECI',
                            f'{gs_name}_{sc_name}.groundstation_position_ECI')

                for sc_name, cubesat in cubesats.items():
                    self.add(
                        CubeSat(
                            num_times=num_times,
                            groundstations=groundstations,
                            cubesat=cubesat,
                        ),
                        name=sc_name,
                        promotes=['h'],
                    )
                    for gs_name, groundstation in groundstations.items():
                        self.connect(f'{gs_name}_{sc_name}.download_rate',
                                     f'{sc_name}.{gs_name}_download_rate')

    rep = GraphRepresentation(
        Mission(
            num_times=301,
            step_size=19.,
            groundstations=groundstations,
            # groundstations=None,
            cubesats=cubesats,
        ), )
    sim = Simulator(rep)
    sim['initial_position_km'] = np.array([-6257.51, 354.136, 2726.69])
    sim['initial_velocity_km_s'] = np.array([3.08007, 0.901071, 6.93116])
    # sim.visualize_implementation()
    sim.run()

    import matplotlib.pyplot as plt
    n = len(sim['h'])
    A = np.tril(np.ones((n, n)), k=-1)
    t = np.concatenate((np.array([0]), np.matmul(A, sim['h'])))
    plt.plot(t, sim['UCSD_optics.gs_dot'])
    plt.plot(t, sim['UIUC_optics.gs_dot'])
    plt.plot(t, sim['GT_optics.gs_dot'])
    plt.plot(t, sim['Montana_optics.gs_dot'])
    plt.show()
    # plt.plot(t, sim['UCSD_optics.groundstation_to_spacecraft_ECI'][:, 0])
    # plt.plot(t, sim['UIUC_optics.groundstation_to_spacecraft_ECI'][:, 0])
    # plt.plot(t, sim['GT_optics.groundstation_to_spacecraft_ECI'][:, 0])
    # plt.plot(t, sim['Montana_optics.groundstation_to_spacecraft_ECI'][:, 0])
    # plt.show()
    plt.plot(t, sim['UCSD.groundstation_position_ECI'][:, 0])
    plt.plot(t, sim['UIUC.groundstation_position_ECI'][:, 0])
    plt.plot(t, sim['GT.groundstation_position_ECI'][:, 0])
    plt.plot(t, sim['Montana.groundstation_position_ECI'][:, 0])
    plt.plot(t, sim['orbit_state'][:, 0])
    plt.show()
    plt.plot(t, np.linalg.norm(sim['UCSD.groundstation_position_ECEF'],
                               axis=1))
    plt.plot(t, np.linalg.norm(sim['UIUC.groundstation_position_ECEF'],
                               axis=1))
    plt.plot(t, np.linalg.norm(sim['GT.groundstation_position_ECEF'], axis=1))
    plt.plot(
        t, np.linalg.norm(sim['Montana.groundstation_position_ECEF'], axis=1))
    plt.plot(t, np.linalg.norm(sim['orbit_state'], axis=1))
    plt.show()
    # plt.plot(t, sim['UCSD.groundstation_position_ECI'][:, 1])
    # plt.plot(t, sim['UIUC.groundstation_position_ECI'][:, 1])
    # plt.plot(t, sim['GT.groundstation_position_ECI'][:, 1])
    # plt.plot(t, sim['Montana.groundstation_position_ECI'][:, 1])
    # plt.show()
    # plt.plot(t, sim['UCSD.groundstation_position_ECI'][:, 2])
    # plt.plot(t, sim['UIUC.groundstation_position_ECI'][:, 2])
    # plt.plot(t, sim['GT.groundstation_position_ECI'][:, 2])
    # plt.plot(t, sim['Montana.groundstation_position_ECI'][:, 2])
    # plt.show()
    # # plt.plot(t, sim['optics.data'])
    # plt.plot(t, sim['UCSD_optics.download_rate'])
    # plt.plot(t, sim['UCSD_optics.los_comm'])
    # plt.plot(t, sim['optics.UCSD_download_rate'])
    # plt.show()
    # plt.plot(t, sim['GT_optics.download_rate'])
    # plt.plot(t, sim['GT_optics.los_comm'])
    # plt.plot(t, sim['optics.GT_download_rate'])
    # plt.show()
    # plt.plot(t, sim['UIUC_optics.download_rate'])
    # plt.plot(t, sim['UIUC_optics.los_comm'])
    # plt.plot(t, sim['optics.UIUC_download_rate'])
    # plt.show()
    # plt.plot(t, sim['Montana_optics.download_rate'])
    # plt.plot(t, sim['Montana_optics.los_comm'])
    # plt.plot(t, sim['optics.Montana_download_rate'])
    # plt.show()
    plt.title("LOS")
    plt.xlabel('time [s]')
    plt.ylabel('LOS')
    plt.plot(t, sim['UCSD_optics.los_comm'], label='UCSD/optics')
    plt.plot(t, sim['GT_optics.los_comm'], label='GT/optics')
    plt.plot(t, sim['UIUC_optics.los_comm'], label='UIUC/optics')
    plt.plot(t, sim['Montana_optics.los_comm'], label='Montana/optics')
    plt.legend()
    plt.show()
    plt.title("Download Rate")
    plt.xlabel('time [s]')
    plt.ylabel('Download Rate')
    plt.plot(t, sim['UCSD_optics.download_rate'], label='UCSD/optics')
    plt.plot(t, sim['GT_optics.download_rate'], label='GT/optics')
    plt.plot(t, sim['UIUC_optics.download_rate'], label='UIUC/optics')
    plt.plot(t, sim['Montana_optics.download_rate'], label='Montana/optics')
    plt.plot(t, sim['optics.download_rate'], label='max')
    plt.legend()
    plt.show()
    # plt.plot(sim['orbit_state'][:, 0], sim['orbit_state'][:, 1])
    # plt.plot(sim['orbit_state'][:, 0], sim['orbit_state'][:, 2])
    # plt.plot(sim['orbit_state'][:, 1], sim['orbit_state'][:, 2])
    # plt.show()
    plt.title("Download Rate for Optics Spacecraft")
    plt.xlabel('time [s]')
    plt.ylabel('Download Rate')
    plt.plot(t, sim['optics.download_rate'])
    plt.show()
    plt.title("Data Downloaded from Optics Spacecraft")
    plt.xlabel('time [s]')
    plt.ylabel('Download Rate')
    plt.plot(sim['optics.data'])
    plt.show()
