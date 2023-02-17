# Distributed under the MIT License.
# See LICENSE for details.
"""
Defines class `models.ExponentialLinear`.

"""

import numpy as np

from .spherically_symmetric import SphericallySymmetric


class ExponentialLinear(SphericallySymmetric):
    """
    Model derived from a wavefunction that, as a function of the distance to the
    coordinate origin, combines a linear growth and an exponential decay.

    """

    @classmethod
    def name(cls):
        """
        The class name.

        """
        return "ExponentialLinear"

    @classmethod
    def has_analytic_df(cls):
        """
        Whether the model has an analytic DF.

        """
        return False

    @property
    def r90():
        """
        The radius containing 90% of the matter.

        """
        return 3.61

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
        return (1. + r) * (1. + r) * np.exp(-2. * r) / (7. * np.pi)

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
        return -2. * r * (1. + r) * np.exp(-2. * r) / (7. * np.pi)

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
                return 9. / 14. - 2. * r * r / 21.
            return (1. - np.exp(-2. * r) * (2. + r) *
                    (7. + 6. * r + 2 * r * r) / 14.) / r

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
                return -4. * r / 21.
            return (7. * (1. - np.exp(2. * r)) + 2. * r *
                    (7. + r * (7. + r *
                               (4. + r)))) * np.exp(-2. * r) / (7. * r * r)

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
                return 0.3 * (5. + r * (3. * r - 5.)) / np.pi
            return -2. * r * r * r * (1. + r) / (7. * (1. - np.exp(2. * r)) +
                                                 2. * r * (7. + r *
                                                           (7. + r *
                                                            (4. + r)))) / np.pi

        return impl(r)
