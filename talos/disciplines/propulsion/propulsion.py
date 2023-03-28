from ozone.api import ODEProblem
from csdl import Model
from csdl.utils.get_bspline_mtx import get_bspline_mtx
import csdl
import numpy as np
from talos.csdl_future.mask import *
from talos.specifications.cubesat_spec import CubesatSpec


class PropellantMassDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        n = self.parameters['num_nodes']

        mass_flow_rate = self.declare_variable('mass_flow_rate', shape=(n, 1))
        # unused, so we need to create an input
        # This DOES change over time when ozone integrates this
        propellant_mass = self.create_input('propellant_mass', shape=(n, 1))

        dm_dt = -mass_flow_rate
        self.register_output('dm_dt', dm_dt)


class ODEProblemTest(ODEProblem):

    def setup(self):
        self.add_parameter('mass_flow_rate',
                           dynamic=True,
                           shape=(self.num_times, 1))
        self.add_state(
            'propellant_mass',
            'dm_dt',
            shape=(1, ),
            initial_condition_name='initial_propellant_mass',
            output='propellant_mass',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(PropellantMassDynamics)


class Propulsion(Model):
    """
    This Goup computes the mass and volume of the total propellant
    consumed based on thrust profile.

    Options
    ----------
    num_times : int
        Number of time steps over which to integrate dynamics
    num_cp : int
        Dimension of design variables/number of control points for
        BSpline components.
    step_size : float
        Constant time step size to use for integration
    cubesat : Cubesat
        Cubesat OptionsDictionary with initial orbital elements,
        specific impulse
    mtx : array
        Matrix that translates control points (num_cp) to actual points
        (num_times)
    """

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('cubesat', types=CubesatSpec)
        self.parameters.declare('max_thrust', types=float, default=20000.)

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        cubesat = self.parameters['cubesat']
        max_thrust = self.parameters['max_thrust']
        pthrust_cp_val = cubesat['pthrust_cp']
        pthrust_scaler = cubesat['pthrust_scaler']
        nthrust_cp_val = cubesat['nthrust_cp']
        nthrust_scaler = cubesat['nthrust_scaler']
        use_cp = cubesat['use_cp']

        # kg
        initial_propellant_mass = self.create_input(
            'initial_propellant_mass',
            val=cubesat['initial_propellant_mass'],
        )

        # self.add_design_variable(
        #     'initial_propellant_mass',
        #     lower=0.,
        #     scaler=cubesat['initial_propellant_mass_scaler'],
        # )

        # Split thrust between positive and negative parts to compute
        # absolute value of thrust to compute total thrust to compute
        # mass flow rate
        if use_cp == True:
            pthrust_cp = self.create_input(
                'pthrust_cp',
                val=pthrust_cp_val,
                shape=(num_cp, 3),
            )
            self.add_design_variable(
                'pthrust_cp',
                lower=0,
                # upper=max_thrust,
                scaler=pthrust_scaler,
            )
            nthrust_cp = self.create_input(
                'nthrust_cp',
                val=nthrust_cp_val,
                shape=(num_cp, 3),
            )
            self.add_design_variable(
                'nthrust_cp',
                # lower=-max_thrust,
                upper=0,
                scaler=nthrust_scaler,
            )
            v = get_bspline_mtx(num_cp, num_times).toarray()
            bspline_mtx = self.declare_variable(
                'bspline_mtx',
                val=v,
                shape=v.shape,
            )
            pthrust = csdl.einsum(bspline_mtx,
                                  pthrust_cp,
                                  subscripts='kj,ji->ki')
            nthrust = csdl.einsum(bspline_mtx,
                                  nthrust_cp,
                                  subscripts='kj,ji->ki')
            self.register_output('pthrust', pthrust)
            self.register_output('nthrust', nthrust)
        else:
            pthrust = self.create_input(
                'pthrust',
                val=pthrust_cp_val,
                shape=(num_times, 3),
            )
            self.add_design_variable(
                'pthrust',
                lower=0,
                # upper=max_thrust,
                scaler=pthrust_scaler,
            )
            nthrust = self.create_input(
                'nthrust',
                val=nthrust_cp_val,
                shape=(num_times, 3),
            )
            self.add_design_variable(
                'nthrust',
                # lower=-max_thrust,
                upper=0,
                scaler=nthrust_scaler,
            )
        thrust = pthrust + nthrust
        total_thrust = csdl.sum(pthrust - nthrust, axes=(1, ))

        self.register_output('thrust', thrust)
        self.register_output('total_thrust', total_thrust)

        mass_flow_rate = total_thrust / (9.81 * cubesat['specific_impulse'])
        self.register_output('mass_flow_rate', mass_flow_rate)

        self.add(
            ODEProblemTest('RK4', 'time-marching',
                           num_times).create_solver_model(),
            name='propellant_mass_integrator',
        )
        propellant_mass = self.declare_variable('propellant_mass',
                                                shape=(num_times, 1))

        final_propellant_mass = propellant_mass[-1, 0]
        self.register_output('final_propellant_mass', final_propellant_mass)
        self.add_constraint('final_propellant_mass', lower=0)
        total_propellant_used = csdl.reshape(propellant_mass[0, 0] -
                                             final_propellant_mass,
                                             new_shape=(1, ))
        self.register_output('total_propellant_used', total_propellant_used)

        # NOTE: Use Ideal Gas Law
        # boltzmann = 1.380649e-23
        # avogadro = 6.02214076e23
        boltzmann_avogadro = 1.380649 * 6.02214076
        # https://advancedspecialtygases.com/pdf/R-236FA_MSDS.pdf
        r236fa_molecular_mass_kg = 152.05 / 1000
        # 100 psi to Pa
        pressure = 100 * 6.894757e5
        temperature = 273.15 + 56
        # (273.15+25)*1.380649*6.02214076/(152.05/1000)/(100*6.894757e5)
        # total_propellant_volume = temperature * boltzmann_avogadro / r236fa_molecular_mass_kg / pressure * initial_propellant_mass

        # self.register_output(
        #     'total_propellant_volume',
        #     total_propellant_volume,
        # )

        # for testing derivatives
        # self.add_constraint('pthrust', equals=0)
        # self.add_constraint('nthrust', equals=0)
        # self.add_constraint('total_thrust', equals=0)
        # self.add_constraint('mass_flow_rate', equals=0)
        # self.add_constraint('propellant_mass', equals=0)


# if __name__ == "__main__":
#     from csdl_om import Simulator
#     from csdl import GraphRepresentation
#     from talos.specifications.cubesat_spec import CubesatSpec

#     rep = GraphRepresentation(
#         Propulsion(
#             num_times=100,
#             num_cp=20,
#             cubesat=CubesatSpec(
#                 name='optics',
#                 dry_mass=1.3,
#                 initial_orbit_state=np.array([
#                     40,
#                     0,
#                     0,
#                     0.,
#                     0.,
#                     0.,
#                     # 1.76002146e+03,
#                     # 6.19179823e+03,
#                     # 6.31576531e+03,
#                     # 4.73422022e-05,
#                     # 1.26425269e-04,
#                     # 5.39731211e-05,
#                 ]),
#                 specific_impulse=47.,
#                 perigee_altitude=500.,
#                 apogee_altitude=500.,
#             )))
#     sim = Simulator(rep)
#     sim.visualize_implementation()

if __name__ == "__main__":
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    from talos.specifications.swarm_spec import SwarmSpec
    np.random.seed(1)

    num_times = 30
    num_cp = 10

    rep = GraphRepresentation(
        Propulsion(
            num_times=num_times,
            num_cp=num_cp,
            cubesat=CubesatSpec(
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
                pthrust_scaler=np.random.randn(num_cp * 3).reshape(
                    (num_cp, 3)),
                nthrust_scaler=np.random.randn(num_cp * 3).reshape(
                    (num_cp, 3)),
                pthrust_cp=np.random.randn(num_cp * 3).reshape((num_cp, 3)),
                nthrust_cp=np.random.randn(num_cp * 3).reshape((num_cp, 3)),
            )))

    sim = Simulator(rep)

    sim['mass_flow_rate'] = 100 * np.random.rand(num_times * 1).reshape(
        (num_times, 1))
    sim['bspline_mtx'] = get_bspline_mtx(num_cp, num_times).toarray()
    sim.run()
    sim.check_partials(
        compact_print=True,
        show_only_incorrect=True,
    )

# (of,wrt)                                                 calc norm                relative error             absolute error
# -----------------------------------------------------------------------------------------------------------------------------------
# ('pthrust', 'initial_propellant_mass')                   0.0                      0.0                        0.0
# ('pthrust', 'pthrust_cp')                                6.685821962713442        1.2130900485488166e-10     8.11050408937455e-10
# ('pthrust', 'nthrust_cp')                                0.0                      0.0                        0.0
# ('pthrust', 'bspline_mtx')                               26.84211112791099        4.399024254425199e-11      1.1807909789187137e-09
# ('pthrust', 'h')                                         0.0                      0.0                        0.0
# ('nthrust', 'initial_propellant_mass')                   0.0                      0.0                        0.0
# ('nthrust', 'pthrust_cp')                                0.0                      0.0                        0.0
# ('nthrust', 'nthrust_cp')                                6.685821962713442        1.2046617774846658e-10     8.054154169484542e-10
# ('nthrust', 'bspline_mtx')                               22.907002518500555       6.107787148456777e-11      1.3991109559167675e-09
# ('nthrust', 'h')                                         0.0                      0.0                        0.0
# ('thrust', 'initial_propellant_mass')                    0.0                      0.0                        0.0
# ('thrust', 'pthrust_cp')                                 6.685821962713442        1.69671178968808e-10       1.134391294785014e-09
# ('thrust', 'nthrust_cp')                                 6.685821962713442        1.6028877704319256e-10     1.071662225922856e-09
# ('thrust', 'bspline_mtx')                                31.439852786986833       7.679478237205237e-11      2.4144166525828746e-09
# ('thrust', 'h')                                          0.0                      0.0                        0.0
# ('total_thrust', 'initial_propellant_mass')              0.0                      0.0                        0.0
# ('total_thrust', 'pthrust_cp')                           6.685821962713442        3.138366685084253e-10      2.0982560910417917e-09
# ('total_thrust', 'nthrust_cp')                           6.685821962713442        3.1484293753820807e-10     2.104983826630594e-09
# ('total_thrust', 'bspline_mtx')                          45.722507309673865       7.009499838995002e-11      3.204919076267679e-09
# ('total_thrust', 'h')                                    0.0                      0.0                        0.0
# ('mass_flow_rate', 'initial_propellant_mass')            0.0                      0.0                        0.0
# ('mass_flow_rate', 'pthrust_cp')                         0.014500665761627174     4.125939821917533e-10      5.982887431158946e-12
# ('mass_flow_rate', 'nthrust_cp')                         0.014500665761627174     3.7587600317024954e-10     5.450452289884818e-12
# ('mass_flow_rate', 'bspline_mtx')                        0.09916608608166626      7.860186106217604e-11      7.794638920305844e-12
# ('mass_flow_rate', 'h')                                  0.0                      0.0                        0.0
# ('propellant_mass', 'initial_propellant_mass')           5.477225575051661        2.524757298496048e-09      1.3828665211206873e-08
# ('propellant_mass', 'pthrust_cp')                        0.13818875375557602      2.5728691722633875e-05     3.5554861283159727e-06
# ('propellant_mass', 'nthrust_cp')                        0.13818875375557602      2.9023392211465213e-05     4.010640647580477e-06
# ('propellant_mass', 'bspline_mtx')                       0.3712663132394355       1.103054898416196e-05      4.095277878068523e-06
# ('propellant_mass', 'h')                                 0.06059960250555483      1.5398148616224918e-05     9.331218379737166e-07
# ('final_propellant_mass', 'initial_propellant_mass')     1.0                      2.524757298496048e-09      2.5247572921216488e-09
# ('final_propellant_mass', 'pthrust_cp')                  0.037274270723277        2.2290042357328638e-05     8.308583572119063e-07
# ('final_propellant_mass', 'nthrust_cp')                  0.037274270723277        2.301684280641118e-05      8.57929877498088e-07
# ('final_propellant_mass', 'bspline_mtx')                 0.09665514470176235      1.0674145131224482e-05     1.0317124460616854e-06
# ('final_propellant_mass', 'h')                           0.013916153918349999     1.674488822793528e-05      2.3302423373894218e-07
# ('total_propellant_used', 'initial_propellant_mass')     0.0                      0.0                        0.0
# ('total_propellant_used', 'pthrust_cp')                  0.037274270723277        2.2290042357328638e-05     8.308583572119063e-07
# ('total_propellant_used', 'nthrust_cp')                  0.037274270723277        2.301684280641118e-05      8.57929877498088e-07
# ('total_propellant_used', 'bspline_mtx')                 0.09665514470176235      1.0674145131224482e-05     1.0317124460616854e-06
# ('total_propellant_used', 'h')                           0.013916153918349999     1.674488822793528e-05      2.3302423373894218e-07
# ('total_mass', 'initial_propellant_mass')                9.486832980505138        2.5247572984960486e-09     2.3951950746670504e-08
# ('total_mass', 'pthrust_cp')                             0.23934994253928216      2.5728691722633875e-05     6.158282619849621e-06
# ('total_mass', 'nthrust_cp')                             0.23934994253928216      2.9023392211465203e-05     6.94663337251033e-06
# ('total_mass', 'bspline_mtx')                            0.643052117669484        1.1030548984161286e-05     7.093229355927108e-06
# ('total_mass', 'h')                                      0.10496159045809922      1.5398148616224915e-05     1.6162144330225305e-06
# -----------------------------------------------------------------------------------------------------------------------------------
