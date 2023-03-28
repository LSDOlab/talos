from csdl.std.pnorm import pnorm
from talos.constants import GRAVITATIONAL_PARAMTERS

from csdl import Model
import csdl

from csdl.std import sin as s
from csdl.std import cos as c
from csdl.lang.concatenation import Concatenation
from csdl import Variable

import numpy as np


def body_313(
        R: Concatenation,
        p: Variable,
        q: Variable,
        r: Variable,
        axes=(0, 1),
):
    """
    Construct transformation matrix that transforms 3-vector, or array
    of 3-vectors from reference frame A to reference frame B, where
    reference frame B is constructed by rotating reference frame A by a
    body 3-1-3 sequence
    """
    R[0, 0] = csdl.expand(c(p) * c(r) - s(p) * c(q) * s(r), (1, 1))
    R[0, 1] = csdl.expand(c(p) * s(r) + s(p) * c(q) * c(r), (1, 1))
    R[0, 2] = csdl.expand(s(p) * s(q), (1, 1))
    R[1, 0] = csdl.expand(-s(p) * c(r) - c(p) * c(q) * s(r), (1, 1))
    R[1, 1] = csdl.expand(-s(p) * s(r) + c(p) * c(q) * c(r), (1, 1))
    R[1, 2] = csdl.expand(c(p) * s(q), (1, 1))
    R[2, 0] = csdl.expand(s(q) * s(r), (1, 1))
    R[2, 1] = csdl.expand(-s(q) * c(r), (1, 1))
    R[2, 2] = csdl.expand(c(q), (1, 1))


def keplerian_to_perifocal(
    r: Concatenation,
    v: Concatenation,
    mu: float,
    apoapsis: Variable,
    periapsis: Variable,
    true_anomaly: Variable,
):
    eccentricity = (apoapsis - periapsis) / (apoapsis + periapsis)
    semimajor_axis = (periapsis + apoapsis) / 2
    semilatus_rectum = semimajor_axis * (1 - eccentricity**2)

    rmag = semilatus_rectum / (1 + eccentricity * csdl.cos(true_anomaly))

    r[0] = rmag * csdl.cos(true_anomaly)
    r[1] = rmag * csdl.sin(true_anomaly)

    # h = csdl.pnorm(mu * semilatus_rectum)
    # v[0] = -mu / h * csdl.sin(true_anomaly)
    # v[1] = mu / h * (eccentricity + csdl.cos(true_anomaly))

    speed = (mu * (2. / rmag - 1. / semimajor_axis))**0.5

    v[0] = -speed * csdl.sin(true_anomaly)
    v[1] = speed * (eccentricity + csdl.cos(true_anomaly))
    return eccentricity, semimajor_axis, speed


class KeplerianToCartesian(Model):

    def initialize(self):
        self.parameters.declare(
            'central_body',
            types=str,
            default='Earth',
        )
        self.parameters.declare(
            'r_name',
            types=str,
            default='position',
        )
        self.parameters.declare(
            'v_name',
            types=str,
            default='velocity',
        )
        self.parameters.declare(
            'periapsis_name',
            types=str,
            default='periapsis',
        )
        self.parameters.declare(
            'apoapsis_name',
            types=str,
            default='apoapsis',
        )
        self.parameters.declare(
            'longitude_name',
            types=str,
            default='longitude_of_ascending_node',
        )
        self.parameters.declare(
            'inclination_name',
            types=str,
            default='inclination',
        )
        self.parameters.declare(
            'arg_peri_name',
            types=str,
            default='argument_of_periapsis',
        )
        self.parameters.declare(
            'true_anomaly_name',
            types=str,
            default='true_anomaly',
        )

        self.parameters.declare(
            'periapsis',
            types=float,
        )
        self.parameters.declare(
            'apoapsis',
            types=float,
        )
        self.parameters.declare(
            'true_anomaly',
            types=float,
        )
        self.parameters.declare(
            'longitude',
            types=float,
        )
        self.parameters.declare(
            'argument_of_periapsis',
            types=float,
        )
        self.parameters.declare(
            'inclination',
            types=float,
        )

    def define(self):
        central_body = self.parameters['central_body']
        mu = GRAVITATIONAL_PARAMTERS[central_body] * 1e-9

        periapsis_name = self.parameters['periapsis_name']
        apoapsis_name = self.parameters['apoapsis_name']
        longitude_name = self.parameters['longitude_name']
        inclination_name = self.parameters['inclination_name']
        arg_peri_name = self.parameters['arg_peri_name']
        true_anomaly_name = self.parameters['true_anomaly_name']
        r_name = self.parameters['r_name']
        v_name = self.parameters['v_name']

        periapsis_val = self.parameters['periapsis']
        apoapsis_val = self.parameters['apoapsis']
        longitude_val = self.parameters['longitude']
        inclination_val = self.parameters['inclination']
        arg_peri_val = self.parameters['argument_of_periapsis']
        true_anomaly_val = self.parameters['true_anomaly']

        periapsis = self.declare_variable(
            periapsis_name,
            val=periapsis_val,
        )
        apoapsis = self.declare_variable(
            apoapsis_name,
            val=apoapsis_val,
        )
        longitude_of_ascending_node = self.declare_variable(
            longitude_name,
            val=longitude_val,
        )
        inclination = self.declare_variable(
            inclination_name,
            val=inclination_val,
        )
        argument_of_periapsis = self.declare_variable(
            arg_peri_name,
            val=arg_peri_val,
        )
        true_anomaly = self.declare_variable(
            true_anomaly_name,
            val=true_anomaly_val,
        )

        r = self.create_output(r_name + '_perifocal', shape=(3, ), val=0)
        v = self.create_output(v_name + '_perifocal', shape=(3, ), val=0)
        eccentricity, semimajor_axis, speed = keplerian_to_perifocal(
            r, v, mu, apoapsis, periapsis, true_anomaly)
        self.register_output('eccentricity', eccentricity)
        self.register_output('semimajor_axis', semimajor_axis)
        self.register_output('speed', speed)

        R = self.create_output('R', shape=(3, 3))
        body_313(R, argument_of_periapsis, inclination,
                 longitude_of_ascending_node)

        # r = csdl.matvec(csdl.transpose(R), r)
        r = csdl.matvec(R, r)
        self.register_output(r_name, r)
        # v = csdl.matvec(csdl.transpose(R), v)
        v = csdl.matvec(R, v)
        self.register_output(v_name, v)
