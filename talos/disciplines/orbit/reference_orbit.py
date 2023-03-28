from ozone.api import ODEProblem
from csdl import Model
import csdl
import numpy as np

mu = 398600.44
Re = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

B = -(3 / 2 * mu * J2 * Re**2)
C = -(5 / 2 * mu * J3 * Re**3)
D = (15 / 8 * mu * J4 * Re**4)


def f(rmag, rz, r):
    return (-mu / rmag**3 + B / rmag**5 *
            (1 - 5 * rz**2 / rmag**2) + C / rmag**7 *
            (3 * rz - 7 * rz**3 / rmag**2) + D / rmag**7 *
            (1 - 14 * rz**2 / rmag**2 + 21 * rz**4 / rmag**4)) * r


def g(rmag, rz):
    return (2 * B / rmag**5 + C / rmag**7 * (3 * rz) + D / rmag**7 *
            (4 - 28 / 3 * rz**2 / rmag**2)) * rz - 3 / 5 * C / rmag**5


class ReferenceOrbitDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        n = self.parameters['num_nodes']

        r = self.create_input('r', shape=(n, 6))
        rz = r[:, 2]
        rz_n3 = csdl.expand(csdl.reshape(rz, (n, )), (n, 3), 'i->ij')

        pos = r[:, :3]

        rmag_flat = csdl.pnorm(r[:, :3], axis=1)
        rmag = csdl.reshape(rmag_flat, (n, 1))
        rmag_n3 = csdl.expand(rmag_flat, (n, 3), 'i->ij')

        a = f(rmag_n3, rz_n3, pos)
        b = g(rmag, rz)

        dr_dt = self.create_output('dr_dt', shape=(n, 6))
        dr_dt[:, :3] = r[:, 3:]
        dr_dt[:, 3] = a[:, 0]
        dr_dt[:, 4] = a[:, 1]
        dr_dt[:, 5] = a[:, 2] + b


class ReferenceOrbitTrajectory(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('r_0', types=np.ndarray)

    def define(self):
        num_times = self.parameters['num_times']
        r_0 = self.parameters['r_0']

        eccentricity = 0.001
        # 450-550 km altitude
        periapsis = Re + 450.
        semimajor_axis = periapsis / (1 - eccentricity)

        self.create_input('r_0', val=r_0, shape=(6, ))

        reference_orbit = ODEProblem('RK4', 'time-marching', num_times)
        reference_orbit.add_state('r',
                                  'dr_dt',
                                  shape=(6, ),
                                  initial_condition_name='r_0',
                                  output='orbit_state')
        reference_orbit.add_times(step_vector='h')
        reference_orbit.set_ode_system(ReferenceOrbitDynamics)

        self.add(reference_orbit.create_solver_model())


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    from matplotlib import rc

    rc('text', usetex=True)

    num_orbits = 30
    duration = 90
    step_size = 19.
    num_times = int(duration * 60 * num_orbits / step_size)
    r_0 = np.array([-6257.51, 354.136, 2726.69, 3.08007, 0.901071, 6.93116])

    class OrbitTest(Model):

        def initialize(self):
            self.parameters.declare('num_times', types=int)
            self.parameters.declare('step_size', types=float)

        def define(self):
            num_times = self.parameters['num_times']
            step_size = self.parameters['step_size']
            self.create_input('h', val=step_size, shape=(num_times - 1, ))
            self.add(ReferenceOrbitTrajectory(
                num_times=num_times,
                r_0=r_0,
            ))

    rep = GraphRepresentation(
        OrbitTest(
            num_times=num_times,
            step_size=step_size,
        ), )
    sim = Simulator(rep, mode='rev')
    sim.run()
    # sim.compute_total_derivatives()
    # exit()
    # print(sim['reference_orbit_state_km'])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r = sim['orbit_state']
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    ax.plot(x, y, z)
    ax.plot(x[0], y[0], z[0], 'o')
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    plt.title('Position of Spacecraft in Sun-synchronous Low Earth Orbit')
    plt.show()
