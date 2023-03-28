from ozone.api import ODEProblem
from csdl.utils.get_bspline_mtx import get_bspline_mtx
from talos.disciplines.reference_frames.body123 import Body123ReferenceFrameChange
from csdl import Model
import csdl
import numpy as np


class OrbitBodyReferenceFrameChange(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']

        # Earth Centered Inertial to Body
        RTN_from_ECI = self.declare_variable(
            'RTN_from_ECI',
            shape=(3, 3, num_times),
        )
        B_from_ECI = self.declare_variable(
            'B_from_ECI',
            shape=(3, 3, num_times),
        )
        ECI_from_RTN = csdl.reorder_axes(RTN_from_ECI, 'ijk->jik')
        self.register_output('ECI_from_RTN', ECI_from_RTN)
        B_from_RTN = csdl.einsum(B_from_ECI,
                                 ECI_from_RTN,
                                 subscripts='ijl,jkl->ikl')
        self.register_output('B_from_RTN', B_from_RTN)

        # Rate of change of Reference frame transformation
        B_from_ECI_dot = self.create_output(
            'B_from_ECI_dot',
            val=0,
            shape=(3, 3, num_times),
        )
        B_from_ECI_dot[:, :, 1:] = (B_from_ECI[:, :, 1:] -
                                    B_from_ECI[:, :, :-1]) / step_size


class BodyRates(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('sc_mmoi', types=np.ndarray)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('gravity_gradient', types=bool)

    def define(self):
        num_times = self.parameters['num_times']
        sc_mmoi = self.parameters['sc_mmoi']
        step_size = self.parameters['step_size']
        gravity_gradient = self.parameters['gravity_gradient']

        osculating_orbit_angular_speed = self.declare_variable(
            'osculating_orbit_angular_speed',
            shape=(1, num_times),
        )

        B_from_ECI = self.declare_variable(
            'B_from_ECI',
            shape=(3, 3, num_times),
        )
        B_from_ECI_dot = self.declare_variable(
            'B_from_ECI_dot',
            shape=(3, 3, num_times),
        )
        # Angular velocity of spacecraft in inertial frame
        # (skew symmetric cross operator)
        wcross = csdl.einsum(
            B_from_ECI_dot,
            # transpose
            csdl.einsum(B_from_ECI, subscripts='ijk->jik'),
            subscripts='ijl,jkl->ikl')
        body_rates = self.create_output('body_rates', shape=(num_times, 3))
        body_rates[:, 0] = csdl.reshape(wcross[2, 1, :], (num_times, 1))
        body_rates[:, 1] = csdl.reshape(wcross[0, 2, :], (num_times, 1))
        body_rates[:, 2] = csdl.reshape(wcross[1, 0, :], (num_times, 1))

        # Angular acceleration of spacecraft in inertial frame, compute
        # via finite differences
        body_accels = self.create_output(
            'body_accels',
            val=0,
            shape=(num_times, 3),
        )
        body_accels[1:, :] = (body_rates[1:, :] -
                              body_rates[:-1, :]) / step_size

        Jw = body_rates * np.einsum('i,j->ij', np.ones(num_times), sc_mmoi)
        bt1 = self.create_output('bt1', shape=(num_times, 3))
        bt1[:, 0] = sc_mmoi[0] * body_accels[:, 0]
        bt1[:, 1] = sc_mmoi[1] * body_accels[:, 1]
        bt1[:, 2] = sc_mmoi[2] * body_accels[:, 2]

        bt2 = csdl.cross(body_rates, Jw, axis=1)
        if gravity_gradient is True:
            # Terms associated with orientation of spacecraft used to
            # compute effect of gravity field on spacecraft angular momentum
            # over time
            bt3 = self.create_output('gravity_term', shape=(3, num_times))
            # bt3[0, :] = -3 * (sc_mmoi[1] - sc_mmoi[2]) * (csdl.reshape(
            #     B_from_RTN[0, 1, :] * B_from_RTN[0, 2, :],
            #     (1, num_times)) * osculating_orbit_angular_speed**2)
            # bt3[1, :] = -3 * (sc_mmoi[2] - sc_mmoi[0]) * (csdl.reshape(
            #     B_from_RTN[0, 2, :] * B_from_RTN[0, 0, :],
            #     (1, num_times)) * osculating_orbit_angular_speed**2)
            # bt3[2, :] = -3 * (sc_mmoi[0] - sc_mmoi[1]) * (csdl.reshape(
            #     B_from_RTN[0, 0, :] * B_from_RTN[0, 1, :],
            #     (1, num_times)) * osculating_orbit_angular_speed**2)

            # use transpose of B_from_RTN
            B_from_RTN = self.declare_variable('B_from_RTN',
                                               shape=(3, 3, num_times))
            bt3[0, :] = -3 * (sc_mmoi[1] - sc_mmoi[2]) * (csdl.reshape(
                B_from_RTN[1, 0, :] * B_from_RTN[2, 0, :],
                (1, num_times)) * osculating_orbit_angular_speed**2)
            bt3[1, :] = -3 * (sc_mmoi[2] - sc_mmoi[0]) * (csdl.reshape(
                B_from_RTN[2, 0, :] * B_from_RTN[0, 0, :],
                (1, num_times)) * osculating_orbit_angular_speed**2)
            bt3[2, :] = -3 * (sc_mmoi[0] - sc_mmoi[1]) * (csdl.reshape(
                B_from_RTN[0, 0, :] * B_from_RTN[1, 0, :],
                (1, num_times)) * osculating_orbit_angular_speed**2)

            body_torque = bt1 + bt2 + csdl.reorder_axes(bt3, 'ij->ji')
        else:
            body_torque = bt1 + bt2
        self.register_output('body_torque', body_torque)


class ReactionWheelDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)
        self.parameters.declare('rw_mmoi', types=np.ndarray)

    def define(self):
        n = self.parameters['num_nodes']
        rw_mmoi = self.parameters['rw_mmoi']

        body_torque = self.declare_variable('body_torque', shape=(n, 3))
        body_rates = self.declare_variable('body_rates', shape=(n, 3))
        reaction_wheel_velocity = self.create_input('reaction_wheel_velocity',
                                                    shape=(n, 3))
        x = self.create_output('x', shape=(n, 3))
        x[:, 0] = rw_mmoi[0] * body_rates[:, 0]
        x[:, 1] = rw_mmoi[1] * body_rates[:, 1]
        x[:, 2] = rw_mmoi[2] * body_rates[:, 2]
        dw_dt = csdl.cross(body_rates, x, axis=1) - body_torque
        self.register_output('dw_dt', dw_dt)


class ODEProblemTest(ODEProblem):

    def setup(self):
        self.add_parameter('body_torque',
                           dynamic=True,
                           shape=(self.num_times, 3))
        self.add_parameter(
            'body_rates',
            dynamic=True,
            shape=(self.num_times, 3),
        )
        self.add_state(
            'reaction_wheel_velocity',
            'dw_dt',
            shape=(3, ),
            initial_condition_name='initial_reaction_wheel_velocity',
            output='reaction_wheel_velocity',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(ReactionWheelDynamics)


class Attitude(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('max_rw_torque', default=0.004, types=float)
        self.parameters.declare('max_rw_power', default=1., types=float)
        # 6U mmoi based on CADRE 3U cubesat
        self.parameters.declare('sc_mmoi',
                                default=6 * np.array([2, 1, 3]) * 1e-3,
                                types=np.ndarray)
        self.parameters.declare('rw_mmoi',
                                default=6 * np.ones(3) * 1e-5,
                                types=np.ndarray)
        self.parameters.declare('gravity_gradient', types=bool)

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        max_rw_torque = self.parameters['max_rw_torque']
        max_rw_power = self.parameters['max_rw_torque']
        max_rw_speed = max_rw_power / max_rw_torque
        gravity_gradient = self.parameters['gravity_gradient']
        sc_mmoi = self.parameters['sc_mmoi']
        rw_mmoi = self.parameters['rw_mmoi']
        if sc_mmoi.shape != (3, ):
            raise ValueError(
                'sc_mmoi must have shape (3,); has shape {}'.format(
                    sc_mmoi.shape))

        if rw_mmoi.shape != (3, ):
            raise ValueError(
                'rw_mmoi must have shape (3,); has shape {}'.format(
                    rw_mmoi.shape))

        yaw_cp = self.create_input(
            'yaw_cp',
            shape=(num_cp, ),
            val=0,
        )
        pitch_cp = self.create_input(
            'pitch_cp',
            shape=(num_cp, ),
            val=0,
        )
        roll_cp = self.create_input(
            'roll_cp',
            shape=(num_cp, ),
            val=0,
        )
        self.add_design_variable('yaw_cp')
        self.add_design_variable('pitch_cp')
        self.add_design_variable('roll_cp')

        bspline_mtx = get_bspline_mtx(num_cp, num_times)
        yaw = csdl.matvec(bspline_mtx, yaw_cp)
        pitch = csdl.matvec(bspline_mtx, pitch_cp)
        roll = csdl.matvec(bspline_mtx, roll_cp)
        yaw = self.register_output('yaw', yaw)
        pitch = self.register_output('pitch', pitch)
        roll = self.register_output('roll', roll)

        self.add(Body123ReferenceFrameChange(num_times=num_times, ), )
        self.add(
            OrbitBodyReferenceFrameChange(
                num_times=num_times,
                step_size=step_size,
            ), )
        self.connect('C', 'B_from_ECI')
        self.add(
            BodyRates(
                num_times=num_times,
                step_size=step_size,
                sc_mmoi=sc_mmoi,
                gravity_gradient=gravity_gradient,
            ))

        initial_reaction_wheel_velocity = self.create_input(
            'initial_reaction_wheel_velocity',
            shape=(3, ),
            val=0,
        )
        self.add_design_variable(
            'initial_reaction_wheel_velocity',
            lower=-max_rw_speed,
            upper=max_rw_speed,
        )

        self.add(
            # ODEProblemTest('GaussLegendre4', 'collocation',
            ODEProblemTest('RK4', 'time-marching',
                           num_times).create_solver_model(
                               ODE_parameters={'rw_mmoi': rw_mmoi}, ),
            name='reaction_wheel_speed_integrator',
        )
        reaction_wheel_velocity = self.declare_variable(
            'reaction_wheel_velocity',
            shape=(num_times, 3),
        )

        # TODO: can we get this from ozone?
        rw_accel = self.create_output(
            'rw_accel',
            shape=(num_times, 3),
            val=0,
        )
        rw_accel[1:, :] = (reaction_wheel_velocity[1:, :] -
                           reaction_wheel_velocity[:-1, :]) / step_size

        # use reaction wheel torque to compute current draw required for
        # operation
        reaction_wheel_torque = self.create_output(
            'reaction_wheel_torque',
            shape=(num_times, 3),
        )
        reaction_wheel_torque[:, 0] = rw_mmoi[0] * rw_accel[:, 0]
        reaction_wheel_torque[:, 1] = rw_mmoi[1] * rw_accel[:, 1]
        reaction_wheel_torque[:, 2] = rw_mmoi[2] * rw_accel[:, 2]

        # Each reaction wheel has a maximum torque
        min_reaction_wheel_torque = csdl.min(
            # csdl.min(
            reaction_wheel_torque,
            # axis=0,
            # rho=10. / 1e12,
            # ),
            # axis=1,
        )
        max_reaction_wheel_torque = csdl.max(
            # csdl.max(
            reaction_wheel_torque,
            # axis=0,
            # rho=10. / 1e13,
            # ),
            # axis=1,
        )
        self.register_output(
            'min_reaction_wheel_torque',
            min_reaction_wheel_torque,
        )
        self.register_output(
            'max_reaction_wheel_torque',
            max_reaction_wheel_torque,
        )
        self.add_constraint(
            'min_reaction_wheel_torque',
            lower=-max_rw_torque,
        )
        self.add_constraint(
            'max_reaction_wheel_torque',
            upper=max_rw_torque,
        )

        # RW rate saturation
        # rw_speed_min = csdl.min(rw_speed, axis=1, rho=50. / 1e0)
        # rw_speed_max = csdl.max(rw_speed, axis=1, rho=10. / 1e0)
        # self.register_output('rw_speed_min', rw_speed_min)
        # self.register_output('rw_speed_max', rw_speed_max)
        # self.add_constraint('rw_speed_min', lower=-max_rw_speed)
        # self.add_constraint('rw_speed_max', upper=max_rw_speed)

        # NOTE: DEBUGGING DERIVARIVES ONLY -- NEED A SCALAR OBJECTIVE
        # reaction_wheel_speed = csdl.pnorm(reaction_wheel_velocity )
        # self.register_output('reaction_wheel_speed',reaction_wheel_speed  )


if __name__ == "__main__":
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    np.random.seed(0)

    num_times = 301
    num_cp = int((num_times - 1) / 5)
    duration = 95.
    step_size = duration * 60 / (num_times - 1)

    m = Attitude(
        num_times=num_times,
        step_size=step_size,
        num_cp=num_cp,
        gravity_gradient=True,
    )
    # m.add_objective('reaction_wheel_speed')

    rep = GraphRepresentation(m)
    sim = Simulator(rep)
    sim.run()
    sim.compute_total_derivatives(check_failure=True)
