# Distributed under the MIT License.
# See LICENSE for details.

import numpy as np

from .spherically_symmetric import SphericallySymmetric


class Exponential(SphericallySymmetric):
    """
    Model derived from a wavefunction that decays exponentially with the radial
    distance to the coordinate origin.

    """

    def name(cls):
        return "Exponential"

    def has_analytic_df(cls):
        return False

    def mass_density(self, r):
        """
        The radial profile of the mass density.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        Returns
        -------

        out : ndarray
        Mass density in units of $M/R^3$.

        """
        return np.exp(-2. * r) / np.pi

    def deriv_mass_density(self, r):
        """
        The radial derivative of the profile of the mass density.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        Returns
        -------

        out : ndarray
        Derivative of the mass density in units of $M/R^4$.

        """
        return -2. * np.exp(-2. * r) / np.pi

    def relative_potential(self, r, offset=1.e-6):
        """
        The radial profile of the relative potential.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        `offset` : float (optional, default: 1.e-6)
        The radial coordinate below which to regularize the potential by
        evaluating its Taylor series around $r = 0$.

        Returns
        -------

        out : ndarray
        Relative potential in units of $GM/R$.

        """

        @np.vectorize
        def impl(r):
            if np.abs(r) < offset:
                return 1. - 2. * r * r / 3.
            return (1. - np.exp(-2. * r) * (1. + r)) / r

        return impl(r)

    def deriv_relative_potential(self, r, offset=1.e-6):
        """
        The radial derivative of the profile of the relative potential.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        `offset` : float (optional, default: 1.e-6)
        The radial coordinate below which to regularize the derivative of
        the potential by evaluating its Taylor series around $r = 0$.

        Returns
        -------

        out : ndarray
        Radial derivative of the relative potential in units of $GM/R^2$.

        """

        @np.vectorize
        def impl(r):
            if np.abs(r) < offset:
                return -4. * r / 3. + 2. * r * r
            return np.exp(-2. * r) * (1. - np.exp(2. * r) + 2. * r *
                                      (1. + r)) / (r * r)

        return impl(r)

    def drho_dpsi(self, r, offset=1.e-6):
        """
        The ratio `deriv_mass_density(r)` / `deriv_relative_potential(r)`,
        including regularization near $r = 0$.

        Parameters
        ----------

        `r` : array_like
        Radial coordinate in units of $R$.

        `offset` : float (optional, default: 1.e-6)
        The radial coordinate below which to regularize the ratio by evaluating
        its Taylor series around $r = 0$.

        Returns
        -------

        out : ndarray
        Ratio of radial derivatives in units of $1/GR^2$.

        """

        @np.vectorize
        def impl(r):
            if np.abs(r) < offset:
                return (0.75 * (2. / r - 1.) + 0.0125 * r * (6. + r)) / np.pi
            return -2. * r * r / (1. - np.exp(2. * r) + 2. * r *
                                  (1. + r)) / np.pi

        return impl(r)
