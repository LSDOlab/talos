from csdl import Model
import csdl
from ozone.api import ODEProblem
import numpy as np

Re = 6378.137
mu = 3.986004418 * 1e14 / (1e3)**3
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6
B = -(3 / 2 * mu * J2 * Re**2)
C = -(5 / 2 * mu * J3 * Re**3)
D = (15 / 8 * mu * J4 * Re**4)


# p = rmag
# s = r + u
# ry, uy = r, u
def aa(ry, uy, rz, uz, p, s):
    return (p**11 * (ry + uy) *
            (B * s**4 * (s**2 - 5 * (rz + uz)**2) + C * s**2 * (rz + uz) *
             (3 * s**2 - 7 *
              (rz + uz)**2) + D * (s**4 - 14 * s**2 * (rz + uz)**2 + 21 *
                                   (rz + uz)**4) - mu * s**8) - ry * s**11 *
            (B * p**4 * (p**2 - 5 * rz**2) + C * p**2 * rz *
             (3 * p**2 - 7 * rz**2) + D *
             (p**4 - 14 * p**2 * rz**2 + 21 * rz**4) - mu * p**8)) / (p**11 *
                                                                      s**11)


# p = rmag
# s = r + u
def bb(rx, ry, rz, ux, uy, uz, p, s):
    return -3 * C / (5 * s**5) + 3 * C * (rz + uz)**2 / s**7 + 3 * C / (
        5 * p**5) - 3 * C * rz**2 / p**7 - rz * (
            2 * B / p**5 - D *
            (28 * rz**2 / (3 * rx**2 + 3 * ry**2 + 3 * rz**2) - 4) / p**7) + (
                rz + uz) * (2 * B / s**5 - D *
                            (28 * (rz + uz)**2 /
                             (3 * (rx + ux)**2 + 3 * (ry + uy)**2 + 3 *
                              (rz + uz)**2) - 4) / s**7)


# approximate; do not use
# def aa(ry, uy, rz, uz, p, s):
#     return (-ry * (B * p**4 * (p**2 - 5 * rz**2) + C * p**2 * rz *
#                    (3 * p**2 - 7 * rz**2) + D *
#                    (p**4 - 14 * p**2 * rz**2 + 21 * rz**4) - mu * p**8) +
#             (ry + uy) * (B * p**4 * (p**2 - 5 * (rz + uz)**2) + C * p**2 *
#                          (3 * p**2 - 7 * (rz + uz)**2) * (rz + uz) + D *
#                          (p**4 - 14 * p**2 * (rz + uz)**2 + 21 *
#                           (rz + uz)**4) - mu * p**8)) / p**11

# def bb(rx, ry, rz, ux, uy, uz, p, s):
#     return (9 * C * (-rz**2 +
#                      (rz + uz)**2) * (rx**2 + ry**2 + rz**2) - 2 * rz *
#             (3 * B * p**2 * (rx**2 + ry**2 + rz**2) - 2 * D *
#              (-3 * rx**2 - 3 * ry**2 + 4 * rz**2)) + 2 * (rz + uz) *
#             (3 * B * p**2 * (rx**2 + ry**2 + rz**2) - 2 * D *
#              (-3 * rx**2 - 3 * ry**2 - 3 * rz**2 + 7 *
#               (rz + uz)**2))) / (3 * p**7 * (rx**2 + ry**2 + rz**2))


class RelativeOrbitDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        n = self.parameters['num_nodes']

        acceleration_due_to_thrust = self.declare_variable(
            'acceleration_due_to_thrust',
            shape=(n, 3),
            val=0,
        )

        r = self.declare_variable('reference_orbit_state', shape=(n, 6))
        rx = r[:, 0]
        ry = r[:, 1]
        rz = r[:, 2]
        rzn3 = csdl.expand(csdl.reshape(rz, (n, )), (n, 3), 'i->ij')

        p_flat = csdl.pnorm(r[:, :3], axis=1)
        p = csdl.reshape(p_flat, (n, 1))
        pn3 = csdl.expand(p_flat, (n, 3), 'i->ij')

        u = self.declare_variable('relative_orbit_state', shape=(n, 6))
        ux = u[:, 0]
        uy = u[:, 1]
        uz = u[:, 2]
        uzn3 = csdl.expand(csdl.reshape(uz, (n, )), (n, 3), 'i->ij')

        r_pos = r[:, :3]
        u_pos = u[:, :3]

        s_flat = csdl.pnorm(r_pos + u_pos, axis=1)
        s = csdl.reshape(s_flat, (n, 1))
        sn3 = csdl.expand(s_flat, (n, 3), 'i->ij')

        # approximate EOM using position relative to reference orbit
        a = aa(r_pos, u_pos, rzn3, uzn3, pn3, sn3)
        b = bb(rx, ry, rz, ux, uy, uz, p, s)
        udot = self.create_output('udot', shape=(n, 6))
        udot[:, :3] = u[:, 3:]
        udot[:, 3] = a[:, 0] + acceleration_due_to_thrust[:, 0]
        udot[:, 4] = a[:, 1] + acceleration_due_to_thrust[:, 1]
        udot[:, 5] = a[:, 2] + b + acceleration_due_to_thrust[:, 2]


class ODEProblemTest(ODEProblem):

    def setup(self):
        self.add_parameter('reference_orbit_state',
                           dynamic=True,
                           shape=(self.num_times, 6))
        self.add_parameter(
            'acceleration_due_to_thrust',
            dynamic=True,
            shape=(self.num_times, 3),
        )
        self.add_state(
            'relative_orbit_state',
            'udot',
            shape=(6, ),
            initial_condition_name='relative_initial_state',
            output='relative_orbit_state',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(RelativeOrbitDynamics)


class RelativeOrbitTrajectory(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('relative_initial_state',
                                types=np.ndarray,
                                default=np.zeros(6))
        self.parameters.declare(
            'approach',
            values=['time-marching', 'solver-based', 'collocation'])

    def define(self):
        num_times = self.parameters['num_times']
        relative_initial_state = self.parameters['relative_initial_state']
        approach = self.parameters['approach']

        reference_orbit_state = self.declare_variable(
            'reference_orbit_state',
            shape=(num_times, 6),
        )
        self.create_input('relative_initial_state',
                          val=relative_initial_state,
                          shape=(6, ))
        if approach == 'time-marching':
            self.add(
                ODEProblemTest('RK4', 'time-marching',
                               num_times).create_solver_model(),
                name='relative_orbit_integrator',
            )
        elif approach == 'collocation':
            fc = np.zeros(6, )
            fc[:3] = np.ones(3, )

            class ODEProblemTestC(ODEProblem):

                def setup(self):
                    self.add_parameter('reference_orbit_state',
                                       dynamic=True,
                                       shape=(self.num_times, 6))
                    self.add_parameter(
                        'acceleration_due_to_thrust',
                        dynamic=True,
                        shape=(self.num_times, 3),
                    )
                    self.add_state(
                        'relative_orbit_state',
                        'udot',
                        shape=(6, ),
                        initial_condition_name='relative_initial_state',
                        output='relative_orbit_state',
                        interp_guess=[relative_initial_state, fc])
                    self.add_times(step_vector='h')
                    self.set_ode_system(RelativeOrbitDynamics)

            self.add(
                ODEProblemTestC('GaussLegendre4', 'collocation',
                                num_times).create_solver_model(),
                name='relative_orbit_integrator',
            )
        elif approach == 'solver-based':
            self.add(
                ODEProblemTest('GaussLegendre4', 'solver-based',
                               num_times).create_solver_model(),
                name='relative_orbit_integrator',
            )
        relative_orbit_state = self.declare_variable(
            'relative_orbit_state',
            shape=(num_times, 6),
        )
        orbit_state = reference_orbit_state + relative_orbit_state
        self.register_output('orbit_state', orbit_state)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    from talos.disciplines.orbit.reference_orbit import ReferenceOrbitTrajectory

    r_0 = np.array([
        -6.25751454e+03, 3.54135435e+02, 2.72669181e+03, 3.07907496e+00,
        9.00070771e-01, 6.93016106e+00
    ])

    class VehicleDynamics(Model):

        def initialize(self):
            self.parameters.declare('num_times', types=int)
            self.parameters.declare('step_size', types=float)

        def define(self):
            num_times = self.parameters['num_times']
            step_size = self.parameters['step_size']

            # time steps for all integrators
            self.create_input('h', shape=(num_times - 1, ), val=step_size)

            self.add(
                RelativeOrbitTrajectory(
                    num_times=num_times,
                    relative_initial_state=np.array([0.02, 0., 0., 0, 0, 0]),
                    approach='time-marching',
                ),
                name='optics_cubesat',
                promotes=['h', 'reference_orbit_state'],
            )
            self.add(
                RelativeOrbitTrajectory(
                    num_times=num_times,
                    relative_initial_state=np.array([-0.02, 0., 0., 0, 0, 0]),
                    approach='time-marching',
                ),
                name='detector_cubesat',
                promotes=['h', 'reference_orbit_state'],
            )
            self.create_input('acceleration_due_to_thrust',
                              shape=(num_times, 3),
                              val=0)
            self.connect('acceleration_due_to_thrust',
                         'optics_cubesat.acceleration_due_to_thrust')
            self.connect('acceleration_due_to_thrust',
                         'detector_cubesat.acceleration_due_to_thrust')

    class ReferenceOrbit(Model):

        def initialize(self):
            self.parameters.declare('num_times', types=int)
            self.parameters.declare('step_size', types=float)

        def define(self):
            num_times = self.parameters['num_times']
            step_size = self.parameters['step_size']

            # time steps for all integrators
            self.create_input('h', shape=(num_times - 1, ), val=step_size)

            self.add(ReferenceOrbitTrajectory(
                num_times=num_times,
                r_0=r_0,
            ), )

    num_orbits = 3
    duration = 90
    step_size = 19.
    num_times = int(duration * 60 * num_orbits / step_size)

    rep = GraphRepresentation(
        ReferenceOrbit(
            num_times=num_times,
            step_size=step_size,
        ), )
    sim1 = Simulator(rep)
    sim1.run()

    rep = GraphRepresentation(
        VehicleDynamics(
            num_times=num_times,
            step_size=step_size,
        ), )
    # from csdl_om import Simulator as S
    # sim2 = S(rep)
    # sim2['reference_orbit_state'] = sim1['orbit_state']
    # sim2.visualize_implementation()
    # exit()
    sim2 = Simulator(rep)
    sim2['reference_orbit_state'] = sim1['orbit_state']
    sim2.run()
    # sim2.check_partials(
    #     compact_print=True,
    #     method='cs',
    # )
    # exit()

    import matplotlib.pyplot as plt
    from matplotlib import rc

    rc('text', usetex=True)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plot trajectory
    x21 = sim2['optics_cubesat.orbit_state'][:, 0]
    y21 = sim2['optics_cubesat.orbit_state'][:, 1]
    z21 = sim2['optics_cubesat.orbit_state'][:, 2]
    # ax.plot(x21[0], y21[0], z21[0], 'o')
    ax.plot(x21, y21, z21)
    x22 = sim2['detector_cubesat.orbit_state'][:, 0]
    y22 = sim2['detector_cubesat.orbit_state'][:, 1]
    z22 = sim2['detector_cubesat.orbit_state'][:, 2]
    # ax.plot(x22[0], y22[0], z22[0], 'x')
    ax.plot(x22, y22, z22)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    plt.title('absolute orbit, both spacecraft')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    u1 = sim2['optics_cubesat.relative_orbit_state']
    u2 = sim2['detector_cubesat.relative_orbit_state']
    ax.plot(u1[:, 0], u1[:, 1], u1[:, 2])
    ax.plot(u2[:, 0], u2[:, 1], u2[:, 2])
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    plt.title('Position of two spacecraft relative to reference orbit')
    plt.show()
    print(np.array2string(u1, separator=',', threshold=10 * num_times))
    print(np.array2string(u2, separator=',', threshold=10 * num_times))
    exit()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # u1 = sim2['optics_cubesat.orbit_state'] - sim2['optics_cubesat.relative_orbit_state']
    # u2 = sim2['detector_cubesat.orbit_state'] - sim2['detector_cubesat.relative_orbit_state']
    u1 = sim2['optics_cubesat.orbit_state'] - sim1['orbit_state']
    u2 = sim2['detector_cubesat.orbit_state'] - sim1['orbit_state']
    ax.plot(u1[:, 0], u1[:, 1], u1[:, 2])
    ax.plot(u2[:, 0], u2[:, 1], u2[:, 2])
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    plt.title('relative orbits, both spacecraft')
    plt.show()

# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     from csdl import GraphRepresentation
#     from python_csdl_backend import Simulator
#     from talos.disciplines.orbit.reference_orbit import ReferenceOrbitTrajectory

#     num_times = 1500 + 1
#     min = 360
#     s = min * 60
#     step_size = s / num_times
#     print('step_size', step_size)

#     rep = GraphRepresentation(
#         ReferenceOrbitTrajectory(
#             num_times=num_times,
#         ), )
#     sim = Simulator(rep, mode='rev')
#     sim.run()

#     reference_orbit_state_km = sim['reference_orbit_state_km']
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     r = sim['reference_orbit_state_km']
#     x = r[:, 0]
#     y = r[:, 1]
#     z = r[:, 2]

#     rep = GraphRepresentation(
#         RelativeOrbitTrajectory(
#             num_times=num_times,
#             step_size=step_size,
#             initial_orbit_state=np.array([
#                 0,
#                 0,
#                 0,
#                 0.,
#                 0.,
#                 0,
#             ]),
#         ), )
#     sim = Simulator(rep, mode='rev')
#     print(r.shape)
#     print(sim['reference_orbit_state_km'].shape)
#     sim['reference_orbit_state_km'] = r
#     sim.run()
#     sim.compute_total_derivatives()
#     exit()
#     relative_orbit_state_m = sim['relative_orbit_state_m']
#     orbit_state = sim['orbit_state']

#     rx = relative_orbit_state_m[:, 0]
#     ry = relative_orbit_state_m[:, 1]
#     rz = relative_orbit_state_m[:, 2]

#     axx = orbit_state[:, 0]
#     ay = orbit_state[:, 1]
#     az = orbit_state[:, 2]

#     ax.plot(x, y, z)
#     ax.plot(x + rx / 1000, y + ry / 1000, z + rz / 1000)
#     ax.set_title('Reference and Absolute Orbit')
#     ax.plot(axx, ay, az)
#     ax.set_xlabel('X [km]')
#     ax.set_ylabel('Y [km]')
#     ax.set_zlabel('Z [km]')
#     plt.show()

#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     ax.plot(rx, ry, rz)
#     ax.set_title('Relative Orbit')
#     ax.set_xlabel('X [m]')
#     ax.set_ylabel('Y [m]')
#     ax.set_zlabel('Z [m]')
#     plt.show()

#     np.savetxt('relative_orbit', relative_orbit_state_m)
