from csdl import Model
import csdl
import numpy as np

from csdl.utils.get_bspline_mtx import get_bspline_mtx
from talos.disciplines.attitude.attitude import Attitude
from talos.disciplines.propulsion.propulsion import Propulsion
from talos.disciplines.orbit.relative_orbit import RelativeOrbitTrajectory
from talos.disciplines.orbit.reference_orbit import ReferenceOrbitTrajectory
from talos.constants import RADII
from talos.specifications.cubesat_spec import CubesatSpec
from talos.specifications.attitude_spec import AttitudeSpec

Re = RADII['Earth']


class VehicleDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('initial_orbit_state',
                                types=np.ndarray,
                                default=np.ones(6))
        self.parameters.declare('cubesat', types=CubesatSpec)

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        initial_orbit_state = self.parameters['initial_orbit_state']
        cubesat = self.parameters['cubesat']
        attitude: AttitudeSpec = cubesat['attitude']

        # time steps for all integrators
        self.declare_variable('h', shape=(num_times - 1, ))

        self.add(
            Propulsion(
                num_times=num_times,
                num_cp=num_cp,
                cubesat=cubesat,
            ),
            name='propulsion',
        )
        thrust = self.declare_variable('thrust', shape=(num_times, 3))
        initial_propellant_mass = self.declare_variable(
            'initial_propellant_mass')

        propellant_mass = self.declare_variable('propellant_mass',
                                                shape=(num_times, 1))
        dry_mass = self.create_input('dry_mass', val=cubesat['dry_mass'])
        # self.add_design_variable('dry_mass', lower=0)
        total_mass = csdl.expand(dry_mass, shape=(num_times, )) + csdl.reshape(
            propellant_mass, (num_times, ))
        total_mass = self.register_output('total_mass', total_mass)
        # self.add_constraint('total_mass', equals=12.)

        acceleration_due_to_thrust = thrust / csdl.expand(
            total_mass, shape=(num_times, 3), indices='i->ik')
        self.register_output('acceleration_due_to_thrust',
                             acceleration_due_to_thrust)

        self.add(
            RelativeOrbitTrajectory(
                num_times=num_times,
                relative_initial_state=initial_orbit_state,
                # approach='collocation',
                approach='time-marching',
                # approach='solver-based',
            ),
            name='relative_orbit',
        )

        orbit_state = self.declare_variable(
            'orbit_state',
            shape=(num_times, 6),
        )
        radius = orbit_state[:, :3]
        velocity = orbit_state[:, 3:]

        # # altitude constraints
        # alt = csdl.pnorm(radius, axis=1) - Re
        # min_alt = csdl.min(alt)
        # max_alt = csdl.max(alt)
        # self.register_output('min_alt', min_alt)
        # self.register_output('max_alt', max_alt)
        # self.add_constraint('min_alt', lower=50.)
        # self.add_constraint('max_alt', upper=50.)
        self.register_output(
            'radius',
            radius,
        )
        self.register_output(
            'velocity',
            velocity,
        )
        self.add(OrbitReferenceFrame(num_times=num_times), )

        if attitude is not None:
            self.add(
                Attitude(
                    # TODO: make OptionsDictionary class unpackable
                    # **attitude may already be possible
                    # https://stackoverflow.com/a/71145561
                    **{k: v['value']
                       for (k, v) in attitude._dict.items()}, ),
                name='attitude',
            )


# TODO: rename, move to its own file
class OrbitReferenceFrame(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']

        radius = self.declare_variable(
            'radius',
            shape=(num_times, 3),
        )
        velocity = self.declare_variable(
            'velocity',
            shape=(num_times, 3),
        )

        # unit vector in radial direction
        a0 = radius / csdl.expand(
            csdl.pnorm(radius, axis=1),
            (num_times, 3),
            indices='i->ij',
        )
        osculating_orbit_angular_velocity = csdl.cross(radius,
                                                       velocity,
                                                       axis=1)
        self.register_output(
            'osculating_orbit_angular_velocity',
            osculating_orbit_angular_velocity,
        )
        # unit vector normal to orbit plane
        a2 = osculating_orbit_angular_velocity / csdl.expand(
            csdl.pnorm(osculating_orbit_angular_velocity, axis=1),
            (num_times, 3),
            indices='i->ij',
        )
        # unit vector normal to radial vector in orbit plane
        a1 = csdl.cross(a2, a0, axis=1)

        RTN_from_ECI = self.create_output(
            'RTN_from_ECI',
            shape=(3, 3, num_times),
        )
        RTN_from_ECI[0, :, :] = csdl.expand(
            csdl.transpose(a0),
            (1, 3, num_times),
            indices='jk->ijk',
        )
        RTN_from_ECI[1, :, :] = csdl.expand(
            csdl.transpose(a1),
            (1, 3, num_times),
            indices='jk->ijk',
        )
        RTN_from_ECI[2, :, :] = csdl.expand(
            csdl.transpose(a2),
            (1, 3, num_times),
            indices='jk->ijk',
        )

        osculating_orbit_angular_speed = csdl.reshape(
            csdl.pnorm(osculating_orbit_angular_velocity, axis=1),
            (1, num_times))
        self.register_output(
            'osculating_orbit_angular_speed',
            osculating_orbit_angular_speed,
        )


if __name__ == "__main__":
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    # from csdl_om import Simulator
    from talos.specifications.cubesat_spec import CubesatSpec
    num_times = 2
    step_size = 95 * 3600 / (num_times - 1)

    r_0 = np.array([
        -6.25751454e+03, 3.54135435e+02, 2.72669181e+03, 3.07907496e+00,
        9.00070771e-01, 6.93016106e+00
    ])

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

    num_orbits = 0.1
    duration = 90
    step_size = 19.
    num_times = int(duration * 60 * num_orbits / step_size)
    num_cp = int(num_times / 5)

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
            num_cp=num_cp,
            step_size=step_size,
            cubesat=CubesatSpec(
                name='cubesat',
                dry_mass=1.3,
                initial_orbit_state=np.array([-0.02, 0, 0, 0, 0, 0]),
                pthrust_scaler=1.,
                nthrust_scaler=1.,
                pthrust_cp=np.random.randn(num_cp * 3).reshape((num_cp, 3)),
                nthrust_cp=np.random.randn(num_cp * 3).reshape((num_cp, 3)),
            ),
        ), )
    sim = Simulator(rep)
    sim['reference_orbit_state'] = sim1['orbit_state']
    sim['bspline_mtx'] = get_bspline_mtx(num_cp, num_times).toarray()
    sim.run()
    sim.check_partials(
        compact_print=True,
        method='cs',
    )
    exit()

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # plot trajectory
    x1 = sim['orbit_state'][:, 0]
    y1 = sim['orbit_state'][:, 1]
    z1 = sim['orbit_state'][:, 2]
    ax.plot(x1[0], y1[0], z1[0], 'o')
    ax.plot(x1, y1, z1)
    plt.show()

# (of,wrt)                                                      calc norm                 relative error             absolute error
# -----------------------------------------------------------------------------------------------------------------------------------------
# ('pthrust', 'pthrust_cp')                                     6.34427612241091          1.1190365506581596e-10     7.099476868451215e-10
# ('pthrust', 'nthrust_cp')                                     0.0                       0.0                        0.0
# ('pthrust', 'bspline_mtx')                                    12.883052101769401        5.923295424082718e-11      7.631012356222853e-10
# ('pthrust', 'initial_propellant_mass')                        0.0                       0.0                        0.0
# ('pthrust', 'h')                                              0.0                       0.0                        0.0
# ('pthrust', 'reference_orbit_state')                          0.0                       0.0                        0.0
# ('pthrust', 'relative_initial_state')                         0.0                       0.0                        0.0
# ('nthrust', 'pthrust_cp')                                     0.0                       0.0                        0.0
# ('nthrust', 'nthrust_cp')                                     6.34427612241091          1.371054417785025e-10      8.698347805171113e-10
# ('nthrust', 'bspline_mtx')                                    20.520827158713303        4.1204665124618684e-11     8.455538111493559e-10
# ('nthrust', 'initial_propellant_mass')                        0.0                       0.0                        0.0
# ('nthrust', 'h')                                              0.0                       0.0                        0.0
# ('nthrust', 'reference_orbit_state')                          0.0                       0.0                        0.0
# ('nthrust', 'relative_initial_state')                         0.0                       0.0                        0.0
# ('thrust', 'pthrust_cp')                                      6.34427612241091          1.6554094766177167e-10     1.0502374815394932e-09
# ('thrust', 'nthrust_cp')                                      6.34427612241091          1.6356800951139936e-10     1.0377206171309684e-09
# ('thrust', 'bspline_mtx')                                     25.25871824474411         6.27380028700739e-11       1.584681537727853e-09
# ('thrust', 'initial_propellant_mass')                         0.0                       0.0                        0.0
# ('thrust', 'h')                                               0.0                       0.0                        0.0
# ('thrust', 'reference_orbit_state')                           0.0                       0.0                        0.0
# ('thrust', 'relative_initial_state')                          0.0                       0.0                        0.0
# ('total_thrust', 'pthrust_cp')                                6.34427612241091          3.218478120525046e-10      2.0418913890176036e-09
# ('total_thrust', 'nthrust_cp')                                6.34427612241091          3.144774422441314e-10      1.9951317278192628e-09
# ('total_thrust', 'bspline_mtx')                               21.708720890543574        8.56518600933099e-11       1.8593923245165512e-09
# ('total_thrust', 'initial_propellant_mass')                   0.0                       0.0                        0.0
# ('total_thrust', 'h')                                         0.0                       0.0                        0.0
# ('total_thrust', 'reference_orbit_state')                     0.0                       0.0                        0.0
# ('total_thrust', 'relative_initial_state')                    0.0                       0.0                        0.0
# ('mass_flow_rate', 'pthrust_cp')                              0.013759897894920312      3.8619359343220227e-10     5.31398441296175e-12
# ('mass_flow_rate', 'nthrust_cp')                              0.013759897894920312      4.043391204239041e-10      5.563665011831059e-12
# ('mass_flow_rate', 'bspline_mtx')                             0.047083351531315355      9.275743511322836e-11      4.36733092456207e-12
# ('mass_flow_rate', 'initial_propellant_mass')                 0.0                       0.0                        0.0
# ('mass_flow_rate', 'h')                                       0.0                       0.0                        0.0
# ('mass_flow_rate', 'reference_orbit_state')                   0.0                       0.0                        0.0
# ('mass_flow_rate', 'relative_initial_state')                  0.0                       0.0                        0.0
# ('propellant_mass', 'pthrust_cp')                             0.16149352032818995       2.0719415929654053e-09     3.346051418868923e-10
# ('propellant_mass', 'nthrust_cp')                             0.16149352032818995       2.9296509806383294e-09     4.731196507201764e-10
# ('propellant_mass', 'bspline_mtx')                            0.169877992549391         2.6062667315117094e-09     4.4274736042838756e-10
# ('propellant_mass', 'initial_propellant_mass')                5.291502622129181         1.0000889005813408e-12     5.2919730397937475e-12
# ('propellant_mass', 'h')                                      0.06329770521410819       2.311437655010571e-09      1.463086992000845e-10
# ('propellant_mass', 'reference_orbit_state')                  0.0                       0.0                        0.0
# ('propellant_mass', 'relative_initial_state')                 0.0                       0.0                        0.0
# ('final_propellant_mass', 'pthrust_cp')                       0.047401583815285114      1.629910055600418e-09      7.726031802933187e-11
# ('final_propellant_mass', 'nthrust_cp')                       0.047401583815285114      3.0730261011887794e-09     1.4566630454756925e-10
# ('final_propellant_mass', 'bspline_mtx')                      0.045804831612220126      2.286426300854232e-09      1.047293717041711e-10
# ('final_propellant_mass', 'initial_propellant_mass')          1.0                       1.0000889005813408e-12     1.000088900582341e-12
# ('final_propellant_mass', 'h')                                0.01492131454686279       2.3174370892149223e-09     3.4579207735465185e-11
# ('final_propellant_mass', 'reference_orbit_state')            0.0                       0.0                        0.0
# ('final_propellant_mass', 'relative_initial_state')           0.0                       0.0                        0.0
# ('total_propellant_used', 'pthrust_cp')                       0.047401583815285114      1.629910055600418e-09      7.726031802933187e-11
# ('total_propellant_used', 'nthrust_cp')                       0.047401583815285114      3.0730261011887794e-09     1.4566630454756925e-10
# ('total_propellant_used', 'bspline_mtx')                      0.045804831612220126      2.286426300854232e-09      1.047293717041711e-10
# ('total_propellant_used', 'initial_propellant_mass')          0.0                       0.0                        0.0
# ('total_propellant_used', 'h')                                0.01492131454686279       2.3174370892149223e-09     3.4579207735465185e-11
# ('total_propellant_used', 'reference_orbit_state')            0.0                       0.0                        0.0
# ('total_propellant_used', 'relative_initial_state')           0.0                       0.0                        0.0
# ('total_mass', 'pthrust_cp')                                  0.27971498230158237       1.1258068326246157e-08     3.1490503834471698e-09
# ('total_mass', 'nthrust_cp')                                  0.27971498230158237       1.116579515927388e-08      3.1232401985109944e-09
# ('total_mass', 'bspline_mtx')                                 0.2942373141833524        2.4793639110982728e-08     7.295213757893036e-09
# ('total_mass', 'initial_propellant_mass')                     9.16515138991168          1.0206073731089113e-10     9.354021084027155e-10
# ('total_mass', 'h')                                           0.10963484143335282       2.644910987885023e-08      2.8997439694777184e-09
# ('total_mass', 'reference_orbit_state')                       0.0                       0.0                        0.0
# ('total_mass', 'relative_initial_state')                      0.0                       0.0                        0.0
# ('acceleration_due_to_thrust', 'pthrust_cp')                  4.431827583661442         5.042449612822592e-09      2.2347267378926086e-08
# ('acceleration_due_to_thrust', 'nthrust_cp')                  4.443359014781251         5.016406335016925e-09      2.2289694215558786e-08
# ('acceleration_due_to_thrust', 'bspline_mtx')                 17.683579004994417        1.114815013506274e-09      1.971391937470098e-08
# ('acceleration_due_to_thrust', 'initial_propellant_mass')     3.169152010097715         7.029917707806538e-07      2.22788621735517e-06
# ('acceleration_due_to_thrust', 'h')                           0.04071514700351277       4.5638506640455545e-08     1.8581785118119955e-09
# ('acceleration_due_to_thrust', 'reference_orbit_state')       0.0                       0.0                        0.0
# ('acceleration_due_to_thrust', 'relative_initial_state')      0.0                       0.0                        0.0
# ('relative_orbit_state', 'pthrust_cp')                        571.0745346441918         4.087762191705507e-09      2.3344169003301768e-06
# ('relative_orbit_state', 'nthrust_cp')                        569.9116158775461         3.493464647786526e-09      1.9909660761558548e-06
# ('relative_orbit_state', 'bspline_mtx')                       709.0714648984055         2.487472285166285e-09      1.7637956182842943e-06
# ('relative_orbit_state', 'initial_propellant_mass')           327.17770977846754        6.961728514355125e-07      0.00022777208057780693
# ('relative_orbit_state', 'h')                                 250.5791035181921         3.189255985175307e-08      7.99160926464068e-06
# ('relative_orbit_state', 'reference_orbit_state')             6.039150641080692e-06     0.1836105098937447         1.045791102327839e-06
# ('relative_orbit_state', 'relative_initial_state')            144.76878358046991        2.5661319236336295e-09     3.7149579717093274e-07
# ('orbit_state', 'pthrust_cp')                                 571.0745346441918         1.549806731495864e-08      8.850551602926072e-06
# ('orbit_state', 'nthrust_cp')                                 569.9116158775461         1.671173089510697e-08      9.524209515056899e-06
# ('orbit_state', 'bspline_mtx')                                709.0714648984055         3.020364727624153e-08      2.141654442637558e-05
# ('orbit_state', 'initial_propellant_mass')                    327.17770977846754        6.963911097590853e-07      0.0002278434897455835
# ('orbit_state', 'h')                                          250.5791035181921         4.754913646804092e-08      1.1914820278397754e-05
# ('orbit_state', 'reference_orbit_state')                      12.961481396817124        7.41726037657225e-07       9.61386890303649e-06
# ('orbit_state', 'relative_initial_state')                     144.76878358046991        3.933625406860928e-08      5.6946616285465014e-06
# ('radius', 'pthrust_cp')                                      568.7208409783797         1.5556948313521323e-08     8.847560750948012e-06
# ('radius', 'nthrust_cp')                                      567.5433400746814         1.677668826826723e-08      9.521497651661344e-06
# ('radius', 'bspline_mtx')                                     706.2264543378346         3.0324836717525894e-08     2.141620192032011e-05
# ('radius', 'initial_propellant_mass')                         325.64830652098885        6.963693621646692e-07      0.00022677134559810077
# ('radius', 'h')                                               250.09998175027349        4.763989936937324e-08      1.1914738253136816e-05
# ('radius', 'reference_orbit_state')                           9.165151389913612         1.0489444588115492e-06     9.613736093532786e-06
# ('radius', 'relative_initial_state')                          144.47837447118238        3.941489783629798e-08      5.694600345674928e-06
# ('velocity', 'pthrust_cp')                                    51.79506883791754         4.441938254359162e-09      2.3007049863293984e-07
# ('velocity', 'nthrust_cp')                                    51.901898318173615        4.378740571236611e-09      2.27264946940634e-07
# ('velocity', 'bspline_mtx')                                   63.45500395225548         1.9087814444507262e-09     1.2112173414654555e-07
# ('velocity', 'initial_propellant_mass')                       31.59800999897363         6.98697135569155e-07       2.2077423651154836e-05
# ('velocity', 'h')                                             15.488261635628843        2.8544939851752494e-09     4.421114968668087e-08
# ('velocity', 'reference_orbit_state')                         9.165151389911737         5.513628868388894e-09      5.053324328595459e-08
# ('velocity', 'relative_initial_state')                        9.16515191001911          2.8825554972861257e-09     2.6419059007997578e-08
# -----------------------------------------------------------------------------------------------------------------------------------------
