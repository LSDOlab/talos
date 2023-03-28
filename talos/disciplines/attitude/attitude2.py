from ozone.api import ODEProblem
from csdl import Model
import csdl
import numpy as np

# TODO: make sure you're plotting nutation
# TODO: plot angular velocity


class AttitudeDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('sc_mmoi', types=np.ndarray, allow_none=True)
        self.parameters.declare('mmoi_ratios',
                                types=np.ndarray,
                                allow_none=True)
        # self.parameters.declare('gravity_gradient', types=bool)

    def define(self):
        num_nodes = self.parameters['num_nodes']
        mmoi_ratios = self.parameters['mmoi_ratios']
        sc_mmoi = self.parameters['sc_mmoi']
        # gravity_gradient = self.parameters['gravity_gradient']
        gravity_gradient = True
        if gravity_gradient:
            if mmoi_ratios is None:
                raise ValueError(
                    "`mmoi_ratios` cannot be `None` if `gravity_gradient` is `True`"
                )
            if sc_mmoi is not None:
                raise ValueError(
                    "`sc_mmoi` must be `None` if `mmoi_ratios` is not `None`")
            if mmoi_ratios.shape != (3, ):
                raise ValueError(
                    'mmoi_ratios must have shape (3,); has shape {}'.format(
                        mmoi_ratios.shape))
        else:
            if sc_mmoi is None:
                raise ValueError(
                    "`sc_mmoi` cannot be `None` if `gravity_gradient` is `False`"
                )
            if mmoi_ratios is not None:
                raise ValueError(
                    "`mmoi_ratios` must be `None` if `sc_mmoi` is not `None`")
            if sc_mmoi.shape != (3, ):
                raise ValueError(
                    'sc_mmoi must have shape (3,); has shape {}'.format(
                        sc_mmoi.shape))

        # moment of inertia ratios
        K = np.array([
            (sc_mmoi[1] - sc_mmoi[2]) / sc_mmoi[0],
            (sc_mmoi[2] - sc_mmoi[0]) / sc_mmoi[1],
            (sc_mmoi[0] - sc_mmoi[1]) / sc_mmoi[2],
        ]) if sc_mmoi is not None else mmoi_ratios

        body_angular_velocity = self.declare_variable('body_angular_velocity',
                                                      shape=(num_nodes, 3))

        # torque free motion
        torque_free_motion = self.create_output('', shape=(num_nodes, 3))
        torque_free_motion[:, 0] = K[
            0] * body_angular_velocity[:, 1] * body_angular_velocity[:, 2]
        torque_free_motion[:, 1] = K[
            1] * body_angular_velocity[:, 2] * body_angular_velocity[:, 0]
        torque_free_motion[:, 2] = K[
            2] * body_angular_velocity[:, 0] * body_angular_velocity[:, 1]

        osculating_orbit_angular_speed = self.declare_variable(
            'osculating_orbit_angular_speed',
            shape=(num_nodes, ),
        )

        # use transpose of B_from_RTN
        B_from_RTN = self.declare_variable('B_from_RTN',
                                           shape=(3, 3, num_nodes))

        # Rate of change of Reference frame transformation
        B_from_RTN_dot = self.create_output(
            'B_from_RTN_dot',
            val=0,
            shape=(3, 3, num_nodes),
        )
        B_from_RTN_dot[0, 2, :] = B_from_RTN[1, 2, :] * csdl.reshape(
            body_angular_velocity[:, 2],
            (1, 1, num_nodes)) - B_from_RTN[2, 2, :] * csdl.reshape(
                body_angular_velocity[:, 1], (1, 1, num_nodes))
        B_from_RTN_dot[1, 2, :] = B_from_RTN[2, 2, :] * csdl.reshape(
            body_angular_velocity[:, 0],
            (1, 1, num_nodes)) - B_from_RTN[0, 2, :] * csdl.reshape(
                body_angular_velocity[:, 2], (1, 1, num_nodes))
        B_from_RTN_dot[2, 2, :] = B_from_RTN[0, 2, :] * csdl.reshape(
            body_angular_velocity[:, 1],
            (1, 1, num_nodes)) - B_from_RTN[1, 2, :] * csdl.reshape(
                body_angular_velocity[:, 0], (1, 1, num_nodes))

        B_from_RTN_dot[0, 0, :] = B_from_RTN[1, 0, :] * csdl.reshape(
            body_angular_velocity[:, 2],
            (1, 1, num_nodes)) - B_from_RTN[2, 0, :] * csdl.reshape(
                body_angular_velocity[:, 1], (1, 1, num_nodes)) + csdl.reshape(
                    osculating_orbit_angular_speed,
                    (1, 1,
                     num_nodes)) * (B_from_RTN[2, 0, :] * B_from_RTN[1, 2, :] -
                                    B_from_RTN[1, 0, :] * B_from_RTN[2, 2, :])
        B_from_RTN_dot[1, 0, :] = B_from_RTN[2, 0, :] * csdl.reshape(
            body_angular_velocity[:, 0],
            (1, 1, num_nodes)) - B_from_RTN[0, 0, :] * csdl.reshape(
                body_angular_velocity[:, 2], (1, 1, num_nodes)) + csdl.reshape(
                    osculating_orbit_angular_speed,
                    (1, 1,
                     num_nodes)) * (B_from_RTN[0, 0, :] * B_from_RTN[2, 2, :] -
                                    B_from_RTN[2, 0, :] * B_from_RTN[0, 2, :])
        B_from_RTN_dot[2, 0, :] = B_from_RTN[0, 0, :] * csdl.reshape(
            body_angular_velocity[:, 1],
            (1, 1, num_nodes)) - B_from_RTN[1, 0, :] * csdl.reshape(
                body_angular_velocity[:, 0], (1, 1, num_nodes)) + csdl.reshape(
                    osculating_orbit_angular_speed,
                    (1, 1,
                     num_nodes)) * (B_from_RTN[1, 0, :] * B_from_RTN[0, 2, :] -
                                    B_from_RTN[0, 0, :] * B_from_RTN[1, 2, :])

        # Terms associated with orientation of spacecraft used to
        # compute effect of gravity field on spacecraft angular momentum
        # over time
        angular_acceleration_due_to_gravity_gradient = self.create_output(
            'gravity_term', shape=(num_nodes, 3))

        angular_acceleration_due_to_gravity_gradient[:, 0] = csdl.reshape(
            -3 * K[0] *
            (csdl.reshape(B_from_RTN[1, 0, :] * B_from_RTN[2, 0, :],
                          (num_nodes, )) * osculating_orbit_angular_speed**2),
            (num_nodes, 1))
        angular_acceleration_due_to_gravity_gradient[:, 1] = csdl.reshape(
            -3 * K[1] *
            (csdl.reshape(B_from_RTN[2, 0, :] * B_from_RTN[0, 0, :],
                          (num_nodes, )) * osculating_orbit_angular_speed**2),
            (num_nodes, 1))
        angular_acceleration_due_to_gravity_gradient[:, 2] = csdl.reshape(
            -3 * K[2] *
            (csdl.reshape(B_from_RTN[0, 0, :] * B_from_RTN[1, 0, :],
                          (num_nodes, )) * osculating_orbit_angular_speed**2),
            (num_nodes, 1))

        body_angular_acceleration = torque_free_motion + angular_acceleration_due_to_gravity_gradient
        self.register_output('body_angular_acceleration',
                             body_angular_acceleration)


class ODEProblemTest(ODEProblem):

    def setup(self):
        # self.add_parameter('external_torque',
        #                    dynamic=True,
        #                    shape=(self.num_times, 3))
        self.add_parameter('osculating_orbit_angular_speed',
                           dynamic=True,
                           shape=(self.num_times, ))
        self.add_state(
            'body_angular_velocity',
            'body_angular_acceleration',
            shape=(3, ),
            initial_condition_name='initial_body_angular_velocity',
            output='body_angular_velocity',
        )
        self.add_state(
            'B_from_RTN',
            'B_from_RTN_dot',
            shape=(3, 3),
            initial_condition_name='initial_B_from_RTN',
            output='B_from_RTN',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(AttitudeDynamics)


class Attitude(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']

        h = np.ones(num_times - 1) * step_size
        self.create_input('h', val=h)

        Omega = 1

        C0 = np.array([
            [0.9924, 0, -0.0789],
            [-0.0868, 0, 0.0944],
            [0.0872, 0, 0.9924],
        ])
        c1 = 1 - C0[:, 0]**2 - C0[:, 2]**2
        C0[:, 1] = c1

        self.create_input(
            'initial_body_angular_velocity',
            shape=(3, ),
            val=np.array([0.1 * Omega, 0.1 * Omega, 1.1 * Omega]),
        )
        self.create_input(
            'initial_B_from_RTN',
            shape=(3, 3),
            val=C0,
        )
        self.create_input(
            'osculating_orbit_angular_speed',
            shape=(num_times, ),
            val=Omega,
        )

        K1 = -0.5
        K2 = 0.9
        K3 = -(K1 + K2) / (1 + K1 * K2)
        K = np.array([K1, K2, K3])
        self.add(
            # ODEProblemTest('GaussLegendre4', 'collocation',
            ODEProblemTest('RK4', 'time-marching',
                           num_times).create_solver_model(ODE_parameters={
                               'mmoi_ratios': K,
                               'sc_mmoi': None,
                           }, ),
            name='reaction_wheel_speed_integrator',
        )


if __name__ == "__main__":
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    np.random.seed(0)

    h = 0.1
    num_orbits = 10
    num_times = int(2 * np.pi * num_orbits / h)
    m = Attitude(num_times=num_times, step_size=h)

    rep = GraphRepresentation(m)
    sim = Simulator(rep)
    sim.run()

    import matplotlib.pyplot as plt
    from matplotlib import rc

    rc('text', usetex=True)

    n = len(sim['h'])
    A = np.tril(np.ones((n, n)), k=-1)
    x = np.concatenate((np.array([0]), np.matmul(A, sim['h']))) / (2 * np.pi)

    plt.plot(x, np.arccos(sim['B_from_RTN'][:, 2, 2]) * 180 / np.pi)
    plt.title('Nutation angle of spacecraft relative to orbit frame')
    plt.xlabel('Number of orbits')
    plt.ylabel('Nutation angle (degrees)')
    plt.show()

    plt.plot(
        x, sim['body_angular_velocity'][:, 0] * 180 / np.pi *
        sim['osculating_orbit_angular_speed'], label='$\\omega_x$')
    plt.plot(
        x, sim['body_angular_velocity'][:, 1] * 180 / np.pi *
        sim['osculating_orbit_angular_speed'], label='$\\omega_y$')
    plt.plot(
        x, sim['body_angular_velocity'][:, 2] * 180 / np.pi *
        sim['osculating_orbit_angular_speed'], label='$\\omega_z$')
    plt.title('Angular velocity of spacecraft relative to spacecraft body')
    plt.xlabel('Number of orbits')
    plt.ylabel('Angular velocity (degrees/sec)')
    plt.legend()
    plt.show()
