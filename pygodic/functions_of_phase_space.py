# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np

import pygodic.eddington_inversion as eddington_inversion
import pygodic.numalg.integrate as integrate
import pygodic.plot


def relative_energy(radius, speed, model):
    """
    For the given model, calculate the relative energy as a function of the
    radial coordinate and the speed.

    Parameters
    ----------

    `radius`, `speed` : array_like, array_like
    The phase space coordinates.

    `model` : object
    The model from which to extract the relative potential. Must be a
    `SphericallySymmetric` object.

    Returns
    -------

    out : ndarray
    The relative energy.

    """
    return model.relative_potential(radius) - 0.5 * speed * speed
