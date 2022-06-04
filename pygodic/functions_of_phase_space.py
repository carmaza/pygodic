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


def speed_moment(radius,
                 nth,
                 energy_min,
                 drho_dpsi_spline,
                 model,
                 norm=None,
                 divmax=20,
                 make_plots=False):
    """
    Get the $n$--th moment of the speed as a function of the radial coordinate.
    The moment is weighted with the spherical DF.

    Parameters
    ----------

    `radius` : ndarray
    The radial coordinate.

    `nth` : double
    The power defining the nth moment.

    `energy_min` : float
    The minimum value for which the DF is valid. Should equal the minimum energy
    for which the given interpolant is valid.

    `drho_dpsi_spline` : object
    The B-spline representation of the interpolant of the derivative of the

    `model` : object
    The model in consideration.

    `norm` : ndarray (optional, default: None)
    If given, use as a normalization for the DF. Must be the same shape as
    `radius`.

    `divmax` : int (optional, default: 20)
    The maximum number of iterations in the integration. Increase this
    parameter if more accuracy is needed.

    `make_plots` : bool (optional, default: False)
    Whether to make a plot of the moment vs the radial coordinate.

    Returns
    -------

    out : ndarray
    The $n$--th speed moment.

    """

    def integrand(v, r):
        return 4. * np.pi * v**(2. + nth) * eddington_inversion.df(
            relative_energy(r, v, model), energy_min, drho_dpsi_spline)

    # The upper limit of the integral is a function of the radius.
    v_max = np.sqrt(2. * (model.relative_potential(radius) - energy_min))

    moment = np.zeros_like(radius)
    for k, r in enumerate(radius):
        moment[k] = integrate.romberg(integrand,
                                      0.,
                                      v_max[k],
                                      args=(r, ),
                                      divmax=divmax)
    if norm is not None:
        moment = moment / norm

    if make_plots:
        label = "{:1.0f}".format(nth)
        pygodic.plot.speed_moment_profile(radius, moment, label, model)

    return moment
