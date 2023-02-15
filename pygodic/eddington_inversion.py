# Distributed under the MIT License.
# See LICENSE for details.
"""
Contains algorithms to calculate the DF using Eddington's inversion formula.

"""

import numpy as np

from pygodic.numalg import interpolate, special


def antideriv_df(energy, energy_min, drho_dpsi_spline, n_quad=10):
    """
    Calculate the antiderivative of the DF at the given energy.

    Details
    -------

    The antiderivative of the DF is computed via Gauss-Jacobi quadrature with
    the weight exponents $\alpha = -1/2$, $\beta = 0$. The required values of
    the integrand at the quadrature nodes are obtained by evaluating the given
    interpolant at such nodes.

    Parameters
    ----------

    `energy` : array_like
    The energy at which to evaluate the function. It must be within the range
    for which the given interpolant is valid, but not smaller than
    `energy_min`.

    `energy_min` : float
    The lowest bound of the integral. Should equal the minimum energy for
    which the given interpolant is valid.

    `drho_dpsi_spline` : object
    The B-spline representation of the interpolant of the derivative of the
    mass density.

    `n_quad` : int (optional, default: 10)
    The number of sample points (nodes) used in the Gaussian quadrature.

    Returns
    -------

    out : ndarray
    The function evaluated at the given energy.

    """

    def g(x):
        potential = 0.5 * (energy * (x + 1.) - energy_min * (x - 1.))
        return interpolate.spline_evaluation(potential, drho_dpsi_spline)

    integral = np.zeros_like(energy)
    try:
        roots, weights = special.roots_jacobi(n_quad, -0.5, 0.)
    except:
        roots, weights, _ = special.roots_jacobi(n_quad, -0.5, 0.)

    # Quadrature sum.
    for j in range(n_quad):
        integral += weights[j] * g(roots[j])

    return np.sqrt(energy - energy_min) * integral / 4. / np.pi**2.


def df(energy, energy_min, antideriv_df_spline):
    """
    Evaluate the DF at the given relative energy, using the given interpolator
    for the antiderivative of the DF.

    Parameters
    ----------

    `energy` : array_like
    The relative energy.

    `energy_min` : float
    The relative energy below which the DF vanishes. Should equal the minimum
    energy for which the interpolant of the antiderivative of the DF is valid.

    `antideriv_df_spline` : object
    The B-spline representation of the interpolant of the antiderivative of the
    DF.

    Returns
    -------

    out : ndarray
    The DF evaluated at the given energy.

    """
    return np.where(
        energy > energy_min,
        interpolate.spline_evaluation(energy, antideriv_df_spline, der=1), 0.)
