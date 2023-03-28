from csdl import Model
import csdl


class Body123ReferenceFrameChange(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']

        yaw = self.declare_variable(
            'yaw',
            shape=(num_times, ),
        )
        pitch = self.declare_variable(
            'pitch',
            shape=(num_times, ),
        )
        roll = self.declare_variable(
            'roll',
            shape=(num_times, ),
        )

        # Earth Centered Inertial to Radial Tangential Normal
        # RTN frame is fixed in osculating orbit plane, so RTN_from_ECI
        # varies with time in all three axes
        sr = csdl.sin(roll)
        cr = csdl.cos(roll)
        sp = csdl.sin(pitch)
        cp = csdl.cos(pitch)
        sy = csdl.sin(yaw)
        cy = csdl.cos(yaw)
        # Use 123 rotation sequence to express coordinates in the body
        # frame that were originally defined in the ECI frame
        C = self.create_output(
            'C',
            shape=(3, 3, num_times),
        )
        C[0, 0, :] = csdl.expand(
            cp * cy,
            (1, 1, num_times),
            indices='i->jki',
        )
        C[0, 1, :] = csdl.expand(
            cp * sy,
            (1, 1, num_times),
            indices='i->jki',
        )
        C[0, 2, :] = csdl.expand(
            -sp,
            (1, 1, num_times),
            indices='i->jki',
        )
        C[1, 0, :] = csdl.expand(
            sr * sp * cy - cr * sy,
            (1, 1, num_times),
            indices='i->jki',
        )
        C[1, 1, :] = csdl.expand(
            sr * sp * sy + cr * cy,
            (1, 1, num_times),
            indices='i->jki',
        )
        C[1, 2, :] = csdl.expand(
            cp * sr,
            (1, 1, num_times),
            indices='i->jki',
        )
        C[2, 0, :] = csdl.expand(
            cr * sp * cy + sr * sy,
            (1, 1, num_times),
            indices='i->jki',
        )
        C[2, 1, :] = csdl.expand(
            cr * sp * sy - sr * cy,
            (1, 1, num_times),
            indices='i->jki',
        )
        C[2, 2, :] = csdl.expand(
            cp * cr,
            (1, 1, num_times),
            indices='i->jki',
        )
