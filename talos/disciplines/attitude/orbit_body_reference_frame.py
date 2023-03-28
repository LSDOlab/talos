from csdl import Model
import csdl


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
