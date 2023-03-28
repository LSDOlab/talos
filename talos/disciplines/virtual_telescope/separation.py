import numpy as np
from csdl import Model
import csdl
from talos.csdl_future.mask import MaskLT

# TODO: Use the "NOTE" comments in this file to design error messages
# where domain of finite derivatives is not the same as domain of valid
# inputs for individual functions; raise error if these functions are
# used and no runtime assertions provided; warn if runtime assertions
# are either too restrictive or guaranteed to pass; runtime assertions
# are required (1) for the compiler to verify that the user has
# communicated how the standard library functions are to be used, and
# (2) so that other users can be reminded of valid domains for inputs


class Separation(Model):

    def initialize(self):
        self.parameters.declare('virtual_telescope')
        self.parameters.declare('telescope_length_m', types=float)
        self.parameters.declare('telescope_length_tol_mm', types=float)
        self.parameters.declare('min_separation_scaler',
                                default=1.,
                                types=(float, np.ndarray))
        self.parameters.declare('max_separation_scaler',
                                default=1.,
                                types=(float, np.ndarray))
        self.parameters.declare('max_separation_all_phases_scaler',
                                default=1.,
                                types=(float, np.ndarray))

    def define(self):
        virtual_telescope = self.parameters['virtual_telescope']
        telescope_length_tol_mm = self.parameters['telescope_length_tol_mm']
        num_times = virtual_telescope['num_times']
        min_separation_scaler = self.parameters['min_separation_scaler']
        max_separation_scaler = self.parameters['max_separation_scaler']
        max_separation_all_phases_scaler = self.parameters[
            'max_separation_all_phases_scaler']

        # NOTE: compute separation in terms of positions relative to
        # reference orbit to satisfy constraints on order of mm when
        # radius is on order of thousands of km
        # telescope_vector = (optics_relative_orbit_state_m[:, :3] -
        #                     detector_relative_orbit_state_m[:, :3])
        telescope_vector = self.declare_variable(
            'telescope_vector',
            shape=(num_times, 3),
        )

        # NOTE: if spacecraft start from different initial positions,
        # this is highly unlikely to be zero, i.e. nondifferentiable
        separation_m = csdl.pnorm(telescope_vector, axis=1)
        self.register_output('separation_m', separation_m)

        # Separation between spacecraft cannot be more than 500 m over
        # entire mission to ensure communication between spacecraft;
        # this also helps ensure that the problem is not poorly scaled
        max_separation_m = csdl.max(separation_m)
        self.register_output('max_separation_m', max_separation_m)
        self.add_constraint(
            'max_separation_m',
            upper=5000.,
            scaler=max_separation_all_phases_scaler,
        )

        # Enforce telescope length constraints
        observation_phase_indicator = self.declare_variable(
            'observation_phase_indicator',
            shape=(num_times, ),
        )

        separation_during_observation = observation_phase_indicator * separation_m
        self.register_output('separation_during_observation',
                             separation_during_observation)

        standby_phase_indicator = csdl.custom(
            observation_phase_indicator,
            op=MaskLT(
                num_times=(num_times, ),
                threshold=0.5,
                in_name='observation_phase_indicator',
                out_name='standby_phase_indicator',
            ),
        )

        self.register_output('standby_phase_indicator',
                             standby_phase_indicator)

        # shift constraint values so that they are relative to zero;
        # this helps with scaling while `adder` is not implemented in
        # back end
        min_separation_during_observation = csdl.min(
            40. * standby_phase_indicator + separation_during_observation -
            40.,
            rho=500,
        )

        max_separation_during_observation = csdl.max(
            40. * standby_phase_indicator + separation_during_observation -
            40.,
            rho=500,
        )

        self.register_output('min_separation_during_observation',
                             min_separation_during_observation)
        self.register_output('max_separation_during_observation',
                             max_separation_during_observation)

        self.add_constraint(
            'min_separation_during_observation',
            lower=-telescope_length_tol_mm / 1000.,
            scaler=min_separation_scaler,
        )
        self.add_constraint(
            'max_separation_during_observation',
            upper=telescope_length_tol_mm / 1000.,
            scaler=max_separation_scaler,
        )


if __name__ == "__main__":
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    from talos.specifications.virtual_telescope import VirtualTelescopeSpec

    np.random.seed(0)

    num_times = 30
    rep = GraphRepresentation(
        Separation(
            virtual_telescope=VirtualTelescopeSpec(
                num_times=num_times,
                num_cp=10,
                # TODO: move to telescope spec
                cross_threshold=0.,
                step_size=0.1,
                duration=5.,
            ),
            telescope_length_m=40.,
            telescope_length_tol_mm=0.15,
        ))
    sim = Simulator(rep)
    sim['telescope_vector'] = np.random.rand(np.prod((num_times, 3))).reshape(
        (num_times, 3))
    sim['observation_phase_indicator'] = np.random.randint(2, size=num_times)
    sim.run()
    sim.check_partials(compact_print=True)
    import matplotlib.pyplot as plt
    plt.plot(sim['observation_phase_indicator'])
    plt.plot(sim['standby_phase_indicator'])
    plt.show()
    print(sim['min_separation_during_observation'])
    print(sim['max_separation_during_observation'])
    plt.plot(sim['separation_during_observation'])
    plt.plot(sim['separation_m'])
    plt.show()

# (of,wrt)                                                                 calc norm              relative error             absolute error
# -------------------------------------------------------------------------------------------------------------------------------------------------
# ('separation_m', 'telescope_vector')                                     5.477225575051661      8.43301478670684e-07       4.618954684388444e-06
# ('separation_m', 'observation_phase_indicator')                          0.0                    0.0                        0.0
# ('standby_phase_indicator', 'telescope_vector')                          0.0                    0.0                        0.0
# ('standby_phase_indicator', 'observation_phase_indicator')               0.0                    0.0                        0.0
# ('max_separation_m', 'telescope_vector')                                 0.897332198685371      1.0991552406997568e-06     9.863083948191886e-07
# ('max_separation_m', 'observation_phase_indicator')                      0.0                    0.0                        0.0
# ('separation_during_observation', 'telescope_vector')                    3.1622776601683795     9.78307176469261e-07       3.0936805546738678e-06
# ('separation_during_observation', 'observation_phase_indicator')         5.332812069021036      6.326019816127714e-11      3.3735474823255434e-10
# ('max_separation_during_observation', 'telescope_vector')                0.9767530388040463     5.664031891720261e-07      5.532363235321486e-07
# ('max_separation_during_observation', 'observation_phase_indicator')     1.3517491342534103     3.61837645830351e-07       4.891138838151422e-07
# ('min_separation_during_observation', 'telescope_vector')                0.9445873445906682     2.068252564964537e-06      1.953646153895412e-06
# ('min_separation_during_observation', 'observation_phase_indicator')     0.2931971442572809     4.093773758153408e-07      1.2002825258743975e-07
# -------------------------------------------------------------------------------------------------------------------------------------------------
